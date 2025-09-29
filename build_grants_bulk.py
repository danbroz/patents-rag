#!/usr/bin/env python3
import datetime
import os
import subprocess
import logging
import re
import sys
import zipfile
import time
import numpy as np
import torch
import multiprocessing as mp
from sentence_transformers import SentenceTransformer
from lxml import etree as ET
from html import unescape
import glob

# =============================================================================
# Hardware / Parallelism Tweaks
# =============================================================================

# Use all CPU threads for CPU-bound tasks (e.g. tokenization).
torch.set_num_threads(16)
torch.set_num_interop_threads(16)

# If input shapes vary, enabling benchmark mode speeds up cuDNN heuristics.
torch.backends.cudnn.benchmark = True

# =============================================================================
# Configuration Constants and Directories
# =============================================================================

# Directory Configuration
DOWNLOAD_DIR = "zips"
BACKUP_DIR = "bulk"  # Patent grant files location
XML_FILES_DIR = "xml-files"
PARTIAL_DIR = "partial-patents"
FINAL_INDEX_DIR = "patents-index"
FINAL_EMB_DIR = "patents-embeddings"
PROGRESS_FILE = "checkpoints/build-progress.txt"

DOWNLOAD_TIMEOUT = 600  # 10 minutes

# =============================================================================
# GPU Configuration for 3 RTX 4090s on single node
# =============================================================================

BATCH_SIZE = 64         # Increased for RTX 4090s
MODEL_BATCH_SIZE = 64   # Increased for RTX 4090s

# Create directories if they don't exist
for d in [DOWNLOAD_DIR, XML_FILES_DIR, PARTIAL_DIR, FINAL_INDEX_DIR, FINAL_EMB_DIR]:
    os.makedirs(d, exist_ok=True)
os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

# =============================================================================
# Progress File Functions
# =============================================================================

def load_progress():
    processed = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            for line in f:
                processed.add(line.strip())
    return processed

def update_progress(filename, lock):
    with lock:
        with open(PROGRESS_FILE, "a") as f:
            f.write(filename + "\n")
    logging.info(f"Updated progress file with {filename}")

# =============================================================================
# Chunk Index Resume Logic
# =============================================================================

def find_resume_chunk_index():
    """
    Inspect partial-patents/ to find the highest existing chunk index,
    then return that plus 1. If none exist, return 0.
    This prevents overwriting partial .npy files on resume.
    """
    max_idx = -1
    for fname in os.listdir(PARTIAL_DIR):
        match = re.match(r'embeddings_chunk_(\d+)\.npy', fname)
        if match:
            idx = int(match.group(1))
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1 if max_idx >= 0 else 0

# =============================================================================
# Extraction and XML Parsing for Patent Grants
# =============================================================================

def extract_zip_file(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        extracted_files = []
        for member in zip_ref.namelist():
            member = member.lstrip("./")
            full_path = os.path.join(extract_dir, member)
            if os.path.isfile(full_path):
                extracted_files.append(full_path)
    logging.info(f"Extracted {zip_path} to {extract_dir}")
    return extracted_files

def parse_patent_grants(xml_file):
    """
    Parse patent grant XML files (different structure from applications)
    """
    try:
        with open(xml_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logging.error(f"Error reading {xml_file}: {e}")
        return

    # Remove XML declarations / DOCTYPE
    content = re.sub(r'<\?xml[^>]+\?>', '', content)
    content = re.sub(r'<!DOCTYPE.*?\]>', '', content, flags=re.DOTALL)
    # Fix weird & references
    content = re.sub(r'&;', '&amp;;', content)
    content = re.sub(r'&(?![#a-zA-Z0-9]+;)', '&amp;', content)
    content = unescape(content)

    wrapped = "<root>" + content + "</root>"
    try:
        parser = ET.XMLParser(resolve_entities=False, recover=True)
        root = ET.fromstring(wrapped.encode('utf-8'), parser=parser)
    except Exception as e:
        logging.error(f"Failed to parse {xml_file}: {e}")
        return

    # Parse us-patent-grant elements (patent grants)
    for elem in root.xpath('//us-patent-grant'):
        # Extract title
        title_elem = elem.find("us-bibliographic-data-grant/invention-title")
        title = title_elem.text.strip() if (title_elem is not None and title_elem.text) else "No Title"
        
        # Extract description/abstract
        description = ""
        
        # Try abstract first
        abstract_elem = elem.find("abstract")
        if abstract_elem is not None:
            paragraphs = [p.text.strip() for p in abstract_elem.findall(".//p") if p.text]
            description = " ".join(paragraphs)
        
        # If no abstract, try description
        if not description:
            desc_elem = elem.find("description")
            if desc_elem is not None:
                paragraphs = [p.text.strip() for p in desc_elem.findall(".//p") if p.text]
                description = " ".join(paragraphs)
        
        # Extract patent number
        docnum_elem = elem.find("us-bibliographic-data-grant/publication-reference/document-id/doc-number")
        patent_id = docnum_elem.text.strip() if (docnum_elem and docnum_elem.text) else "NoID"
        
        yield {"id": patent_id, "title": title, "description": description}

# =============================================================================
# Embedding Buffer & Merge
# =============================================================================

def flush_buffer(embeddings_list, titles_list, chunk_index, lock):
    emb_array = np.array(embeddings_list, dtype=np.float32)
    titles_array = np.array(titles_list, dtype=object)

    emb_file = os.path.join(PARTIAL_DIR, f"embeddings_chunk_{chunk_index}.npy")
    titles_file = os.path.join(PARTIAL_DIR, f"titles_chunk_{chunk_index}.npy")

    with lock:
        np.save(emb_file, emb_array)
        np.save(titles_file, titles_array)
    logging.info(f"Flushed chunk {chunk_index}: saved {len(titles_list)} embeddings.")

def merge_partial_files():
    emb_files = sorted([
        os.path.join(PARTIAL_DIR, f)
        for f in os.listdir(PARTIAL_DIR)
        if f.startswith("embeddings_chunk_") and f.endswith(".npy")
    ])
    titles_files = sorted([
        os.path.join(PARTIAL_DIR, f)
        for f in os.listdir(PARTIAL_DIR)
        if f.startswith("titles_chunk_") and f.endswith(".npy")
    ])

    # Merge titles
    merged_titles = []
    for tfile in titles_files:
        arr = np.load(tfile, allow_pickle=True)
        merged_titles.extend(arr.tolist())
    final_titles_path = os.path.join(FINAL_INDEX_DIR, "patent_titles.npy")
    np.save(final_titles_path, np.array(merged_titles, dtype=object))
    logging.info(f"Merged {len(merged_titles)} patent titles into {final_titles_path}")

    # Merge embeddings into a single memmap
    total_rows = 0
    for emb_file in emb_files:
        arr = np.load(emb_file)
        total_rows += arr.shape[0]
    if total_rows == 0:
        return

    sample = np.load(emb_files[0])
    emb_dim = sample.shape[1]
    final_emb_path = os.path.join(FINAL_EMB_DIR, "embeddings.memmap")
    mm = np.memmap(final_emb_path, dtype=np.float32, mode='w+', shape=(total_rows, emb_dim))
    current_row = 0

    for emb_file in emb_files:
        arr = np.load(emb_file)
        n = arr.shape[0]
        mm[current_row:current_row + n] = arr
        current_row += n
    mm.flush()
    del mm
    logging.info(f"Merged embeddings into {final_emb_path}")

    # Clean partial files
    for f in emb_files + titles_files:
        os.remove(f)
    logging.info("Cleaned up partial files.")

# =============================================================================
# Worker Process Function
# =============================================================================

def worker_process(worker_id, files_to_process, progress_lock, chunk_lock):
    """Worker process that processes files on a specific GPU"""
    
    # Set up device
    device_id = worker_id
    device_str = f"cuda:{device_id}"
    
    # Check if device is available
    if not torch.cuda.is_available() or device_id >= torch.cuda.device_count():
        logging.error(f"Worker {worker_id}: CUDA device {device_id} not available. Available devices: {torch.cuda.device_count()}")
        device_str = "cpu"
    
    logging.info(f"Worker {worker_id} using device={device_str}")
    
    # Load model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device_str)
    logging.info(f"Worker {worker_id} loaded SentenceTransformer on {device_str}.")
    
    # Get starting chunk index
    chunk_index = find_resume_chunk_index() + worker_id * 10000  # Large offset for each worker
    
    # Process files assigned to this worker
    batch_embeddings = []
    batch_titles = []
    
    for i, filename in enumerate(files_to_process):
        if i % 3 != worker_id:  # Distribute files across workers
            continue
            
        # Check if already processed
        processed_files = load_progress()
        if filename in processed_files:
            logging.info(f"Worker {worker_id} skipping {filename}, already processed.")
            continue

        # Process file
        file_path = os.path.join(BACKUP_DIR, filename)
        if not os.path.exists(file_path):
            logging.warning(f"Worker {worker_id}: File {file_path} not found, skipping.")
            continue

        # Extract and process
        try:
            extracted_files = extract_zip_file(file_path, XML_FILES_DIR)
            xml_files = [f for f in extracted_files if f.lower().endswith(".xml")]
            
            if not xml_files:
                logging.error(f"Worker {worker_id}: no XML files found in {filename}.")
                update_progress(filename, progress_lock)
                continue

            for xml_file in xml_files:
                logging.info(f"Worker {worker_id} processing XML: {xml_file}")
                for patent in parse_patent_grants(xml_file):
                    desc = patent["description"]
                    if not desc:
                        continue
                    batch_titles.append(patent["title"])
                    batch_embeddings.append(desc)

                    if len(batch_embeddings) >= BATCH_SIZE:
                        logging.info(f"Worker {worker_id} embedding {len(batch_embeddings)} from {filename}")
                        embeddings = model.encode(
                            batch_embeddings,
                            batch_size=MODEL_BATCH_SIZE,
                            show_progress_bar=False
                        )
                        flush_buffer(embeddings, batch_titles, chunk_index, chunk_lock)
                        chunk_index += 1
                        batch_titles.clear()
                        batch_embeddings.clear()

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                # Clean up XML file
                try:
                    os.remove(xml_file)
                    logging.info(f"Worker {worker_id} deleted {xml_file}")
                except Exception as e:
                    logging.error(f"Worker {worker_id} failed to delete {xml_file}: {e}")

            # Process remaining batch
            if batch_embeddings:
                logging.info(f"Worker {worker_id} embedding final {len(batch_embeddings)} from {filename}")
                embeddings = model.encode(
                    batch_embeddings,
                    batch_size=MODEL_BATCH_SIZE,
                    show_progress_bar=False
                )
                flush_buffer(embeddings, batch_titles, chunk_index, chunk_lock)
                chunk_index += 1
                batch_titles.clear()
                batch_embeddings.clear()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            update_progress(filename, progress_lock)
            
        except Exception as e:
            logging.error(f"Worker {worker_id} error processing {filename}: {e}")
            continue

    logging.info(f"Worker {worker_id} completed processing")

# =============================================================================
# Main Function
# =============================================================================

def main():
    logging.info("Starting patent grant processing with multiprocessing for all files in bulk folder")
    
    # Get all zip files from bulk folder
    bulk_files = glob.glob(os.path.join(BACKUP_DIR, "*.zip"))
    files_to_process = [os.path.basename(f) for f in bulk_files]
    
    logging.info(f"Found {len(files_to_process)} files to process in bulk folder")
    
    if not files_to_process:
        logging.error("No files to process")
        return
    
    # Create locks for shared resources
    progress_lock = mp.Lock()
    chunk_lock = mp.Lock()
    
    # Start worker processes
    num_workers = min(3, torch.cuda.device_count())  # Use up to 3 GPUs
    logging.info(f"Starting {num_workers} worker processes")
    
    processes = []
    for worker_id in range(num_workers):
        p = mp.Process(target=worker_process, args=(worker_id, files_to_process, progress_lock, chunk_lock))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    logging.info("All worker processes completed")
    
    # Merge partial files
    merge_partial_files()
    logging.info("Processing complete.")

if __name__ == '__main__':
    main()

