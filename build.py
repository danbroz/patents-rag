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
import torch.distributed as dist  # <-- for distributed
from sentence_transformers import SentenceTransformer
from lxml import etree as ET
from html import unescape

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

START_DATE = datetime.date(2001, 3, 15)  # Only Thursdays processed
END_DATE   = datetime.date(2025, 2, 20)

DOWNLOAD_DIR    = "zips"
BACKUP_DIR      = "/home/dan/patents-rag/bulk"  # only accessible on rank=0 node
XML_FILES_DIR   = "xml-files"
PARTIAL_DIR     = "partial-patents"
FINAL_INDEX_DIR = "patents-index"
FINAL_EMB_DIR   = "patents-embeddings"
PROGRESS_FILE   = "checkpoints/build-progress.txt"

DOWNLOAD_TIMEOUT = 600  # 10 minutes

# =============================================================================
# Larger Batch Sizes
# =============================================================================

BATCH_SIZE = 32         # how many documents to collect before embedding
MODEL_BATCH_SIZE = 32   # internal batch size for the SentenceTransformer

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

def update_progress(filename):
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
# Wget-based Download Function
# =============================================================================

def download_with_wget(url, dest_dir, user_agent, filename, timeout=DOWNLOAD_TIMEOUT):
    dest_path = os.path.join(dest_dir, filename)
    logging.info(f"Downloading {url} ...")
    try:
        subprocess.run([
            "wget",
            "-c",  # resume partial downloads
            "-nc", # skip existing
            f"--user-agent={user_agent}",
            "-P", dest_dir,
            url
        ], timeout=timeout, check=True)
    except subprocess.TimeoutExpired:
        logging.error(f"Download of {filename} timed out after {timeout}s.")
        raise
    except subprocess.CalledProcessError as e:
        logging.error(f"Download of {filename} failed: {e}")
        raise

    if not os.path.exists(dest_path):
        logging.error(f"File {filename} was not downloaded properly.")
        raise IOError(f"File {filename} not found after wget.")

# =============================================================================
# Extraction and XML Parsing
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

def parse_patents(xml_file):
    try:
        with open(xml_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logging.error(f"Error reading {xml_file}: {e}")
        sys.exit(1)

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
        sys.exit(1)

    # us-patent-application
    for elem in root.xpath('//us-patent-application'):
        title_elem = elem.find("us-bibliographic-data-application/invention-title")
        title = title_elem.text.strip() if (title_elem is not None and title_elem.text) else "No Title"
        desc_elem = elem.find("description")
        description = ""
        if desc_elem is not None:
            paragraphs = [p.text.strip() for p in desc_elem.findall(".//p") if p.text]
            description = " ".join(paragraphs)
        docnum_elem = elem.find("us-bibliographic-data-application/publication-reference/document-id/doc-number")
        patent_id = docnum_elem.text.strip() if (docnum_elem and docnum_elem.text) else "NoID"
        yield {"id": patent_id, "title": title, "description": description}

    # patent-application-publication
    for elem in root.xpath('//patent-application-publication'):
        title_elem = elem.find("subdoc-bibliographic-information/technical-information/title-of-invention")
        title = title_elem.text.strip() if (title_elem and title_elem.text) else "No Title"
        desc_elem = elem.find("subdoc-abstract/paragraph")
        description = desc_elem.text.strip() if (desc_elem and desc_elem.text) else ""
        docnum_elem = elem.find("subdoc-bibliographic-information/document-id/doc-number")
        patent_id = docnum_elem.text.strip() if (docnum_elem and docnum_elem.text) else "NoID"
        yield {"id": patent_id, "title": title, "description": description}

# =============================================================================
# Embedding Buffer & Merge
# =============================================================================

def flush_buffer(embeddings_list, titles_list, chunk_index):
    emb_array = np.array(embeddings_list, dtype=np.float32)
    titles_array = np.array(titles_list, dtype=object)

    emb_file = os.path.join(PARTIAL_DIR, f"embeddings_chunk_{chunk_index}.npy")
    titles_file = os.path.join(PARTIAL_DIR, f"titles_chunk_{chunk_index}.npy")

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
# Retry Logic (Download, Extract) with Exponential Backoff
# =============================================================================

def download_and_extract_with_retry(url, filename, user_agent):
    wait_time = 601
    while True:
        zip_path = os.path.join(DOWNLOAD_DIR, filename)
        try:
            download_with_wget(url, DOWNLOAD_DIR, user_agent, filename, timeout=DOWNLOAD_TIMEOUT)
            extracted_files = extract_zip_file(zip_path, XML_FILES_DIR)
            return extracted_files
        except zipfile.BadZipFile:
            logging.error(f"File {filename} is not a valid ZIP (BadZipFile). Will retry.")
        except Exception as e:
            err_str = str(e)
            if "not a zip file" in err_str.lower() or "BadZipFile" in err_str:
                logging.error(f"File {filename} is not a valid ZIP. Will retry.")
            else:
                logging.error(f"Error downloading/extracting {filename}: {err_str}")

        # Remove partial or bad file
        bad_path = os.path.join(DOWNLOAD_DIR, filename)
        if os.path.exists(bad_path):
            try:
                os.remove(bad_path)
                logging.info(f"Deleted bad file: {bad_path}")
            except:
                pass

        logging.info(f"Waiting {wait_time} seconds before next retry...")
        time.sleep(wait_time)
        wait_time *= 2

# =============================================================================
# Main (Distributed)
# =============================================================================

def main():
    # Initialize torch distributed
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 1) Load progress
    processed_files = load_progress()

    # 2) Find highest chunk index so we resume without overwriting
    #    We do this for each process, so each has a unique chunk index region?
    chunk_index = find_resume_chunk_index()
    logging.info(f"Rank={rank}: starting chunk index at {chunk_index}")

    # 3) Set up model with GPU if available
    device_id = rank  # if each node has 1 GPU, or each node uses the same device=0
    device_str = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    logging.info(f"Rank={rank} using device={device_str}")
    # Use a smaller model as requested:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device_str)
    logging.info(f"Rank={rank} loaded SentenceTransformer on {device_str}.")

    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/109.0.0.0 Safari/537.36"
    )

    # We'll iterate over all Thursdays, but only process the subset
    # for which i_thursday % world_size == rank
    current = START_DATE
    i_thursday = 0
    while current <= END_DATE:
        if current.weekday() == 3:  # Thursday
            if i_thursday % world_size == rank:
                # we do the processing for this node
                year_4digit = current.year
                prefix = "pa" if year_4digit < 2005 else "ipa"
                yy = str(year_4digit % 100).zfill(2)
                mm_val = str(current.month).zfill(2)
                dd = str(current.day).zfill(2)
                filename = f"{prefix}{yy}{mm_val}{dd}.zip"

                if filename in processed_files:
                    logging.info(f"Rank={rank} skipping {filename}, already processed.")
                else:
                    url = f"https://data.uspto.gov/api/v1/datasets/products/files/APPXML/{year_4digit}/{filename}"

                    backup_path = os.path.join(BACKUP_DIR, filename)
                    if os.path.exists(backup_path):
                        logging.info(f"Rank={rank} found backup {backup_path}, extracting.")
                        extracted_files = extract_zip_file(backup_path, XML_FILES_DIR)
                    else:
                        extracted_files = download_and_extract_with_retry(url, filename, user_agent)

                    xml_files = [f for f in extracted_files if f.lower().endswith(".xml")]
                    if not xml_files:
                        logging.error(f"Rank={rank}: no XML files found in {filename}.")
                        update_progress(filename)
                    else:
                        batch_embeddings = []
                        batch_titles = []

                        for xml_file in xml_files:
                            logging.info(f"Rank={rank} processing XML: {xml_file}")
                            for patent in parse_patents(xml_file):
                                desc = patent["description"]
                                if not desc:
                                    continue
                                batch_titles.append(patent["title"])
                                batch_embeddings.append(desc)

                                if len(batch_embeddings) >= BATCH_SIZE:
                                    logging.info(f"Rank={rank} embedding {len(batch_embeddings)} from {filename}")
                                    embeddings = model.encode(
                                        batch_embeddings,
                                        batch_size=MODEL_BATCH_SIZE,
                                        show_progress_bar=False
                                    )
                                    flush_buffer(embeddings, batch_titles, chunk_index)
                                    chunk_index += 1
                                    batch_titles.clear()
                                    batch_embeddings.clear()

                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()

                            # done with that XML file
                            try:
                                os.remove(xml_file)
                                logging.info(f"Rank={rank} deleted {xml_file}")
                            except Exception as e:
                                logging.error(f"Rank={rank} failed to delete {xml_file}: {e}")

                        # leftover
                        if batch_embeddings:
                            logging.info(f"Rank={rank} embedding final {len(batch_embeddings)} from {filename}")
                            embeddings = model.encode(
                                batch_embeddings,
                                batch_size=MODEL_BATCH_SIZE,
                                show_progress_bar=False
                            )
                            flush_buffer(embeddings, batch_titles, chunk_index)
                            chunk_index += 1
                            batch_titles.clear()
                            batch_embeddings.clear()

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        zip_path = os.path.join(DOWNLOAD_DIR, filename)
                        if os.path.exists(zip_path):
                            try:
                                os.remove(zip_path)
                                logging.info(f"Rank={rank} removed processed zip {zip_path}")
                            except Exception as e:
                                logging.error(f"Rank={rank} failed to delete zip {zip_path}: {e}")

                        update_progress(filename)
            i_thursday += 1
        current += datetime.timedelta(days=1)

    # Wait for all ranks to finish
    dist.barrier()

    # Only rank=0 merges partial files
    if dist.get_rank() == 0:
        merge_partial_files()
        logging.info("Rank=0 merged partial files. Processing complete.")

    dist.destroy_process_group()

if __name__ == '__main__':
    main()

