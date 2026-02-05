#!/usr/bin/env python3
"""
Process patent grant PDFs from bulk-older/ (1790-2002): extract text (with OCR fallback),
embed with the same model as the XML pipeline, write chunks to partial-patents-older/.
"""

import os
import re
import logging
import tarfile
import tempfile
import shutil
import glob
import numpy as np
import torch
import multiprocessing as mp
from sentence_transformers import SentenceTransformer
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

# =============================================================================
# Hardware / Parallelism
# =============================================================================
torch.set_num_threads(16)
torch.set_num_interop_threads(16)
torch.backends.cudnn.benchmark = True

# =============================================================================
# Configuration
# =============================================================================
BULK_OLDER_DIR = "bulk-older"
PARTIAL_DIR = "partial-patents-older"
FINAL_INDEX_DIR = "patents-index"
FINAL_EMB_DIR = "patents-embeddings"
PROGRESS_FILE = "checkpoints/build-progress-older.txt"

BATCH_SIZE = 64
MODEL_BATCH_SIZE = 64

for d in [PARTIAL_DIR, FINAL_INDEX_DIR, FINAL_EMB_DIR]:
    os.makedirs(d, exist_ok=True)
os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


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


def find_resume_chunk_index():
    max_idx = -1
    if not os.path.isdir(PARTIAL_DIR):
        return 0
    for fname in os.listdir(PARTIAL_DIR):
        match = re.match(r"embeddings_chunk_(\d+)\.npy", fname)
        if match:
            idx = int(match.group(1))
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1 if max_idx >= 0 else 0


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF; use OCR when text layer is missing or minimal."""
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required; pip install pymupdf")
    text_parts = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            block_list = page.get_text("text") or ""
            if block_list.strip() and len(block_list.strip()) >= 20:
                text_parts.append(block_list)
            elif pytesseract:
                # Minimal or no text: try OCR on rendered page
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                try:
                    import io
                    from PIL import Image
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    ocr_text = pytesseract.image_to_string(img)
                    if ocr_text and ocr_text.strip():
                        text_parts.append(ocr_text)
                except Exception as e:
                    logging.debug(f"OCR failed for {pdf_path} page {page_num}: {e}")
        doc.close()
    except Exception as e:
        logging.warning(f"Error reading PDF {pdf_path}: {e}")
        return ""
    text = " ".join(text_parts)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def patent_from_pdf_path(pdf_path):
    """Derive patent id and title hint from PDF path. Returns (patent_id, title_hint)."""
    stem = Path(pdf_path).stem
    # Remove path components; use last part if nested
    stem = os.path.basename(stem)
    # USPTO-style: sometimes US + number, or number only
    patent_id = stem if stem else "Unknown"
    return patent_id, "No Title"


def process_pdf(pdf_path):
    """Extract (title, description) for one PDF. Returns None if unreadable or empty."""
    text = extract_text_from_pdf(pdf_path)
    if not text or len(text) < 10:
        return None
    patent_id, title_hint = patent_from_pdf_path(pdf_path)
    # Use first line or first 100 chars as title if we have no better hint
    first_line = text.split("\n")[0].strip() if "\n" in text else text[:100].strip()
    if first_line and title_hint == "No Title":
        title = first_line[:200]
    else:
        title = title_hint
    # Description for embedding: full text, normalized
    description = re.sub(r"\s+", " ", text).strip()
    if len(description) > 50000:
        description = description[:50000]
    return {"id": patent_id, "title": title, "description": description}


def flush_buffer(embeddings_list, titles_list, chunk_index, lock):
    emb_array = np.array(embeddings_list, dtype=np.float32)
    titles_array = np.array(titles_list, dtype=object)
    emb_file = os.path.join(PARTIAL_DIR, f"embeddings_chunk_{chunk_index}.npy")
    titles_file = os.path.join(PARTIAL_DIR, f"titles_chunk_{chunk_index}.npy")
    with lock:
        np.save(emb_file, emb_array)
        np.save(titles_file, titles_array)
    logging.info(f"Flushed chunk {chunk_index}: saved {len(titles_list)} embeddings.")


def worker_process(worker_id, tar_filenames, progress_lock, chunk_lock):
    device_id = worker_id
    device_str = f"cuda:{device_id}"
    if not torch.cuda.is_available() or device_id >= torch.cuda.device_count():
        device_str = "cpu"
    logging.info(f"Worker {worker_id} using device={device_str}")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device_str)
    logging.info(f"Worker {worker_id} loaded SentenceTransformer on {device_str}.")

    chunk_index = find_resume_chunk_index() + worker_id * 10000
    batch_titles = []
    batch_descriptions = []

    for i, tar_basename in enumerate(tar_filenames):
        if i % 3 != worker_id:
            continue
        processed = load_progress()
        if tar_basename in processed:
            logging.info(f"Worker {worker_id} skipping {tar_basename}, already processed.")
            continue

        tar_path = os.path.join(BULK_OLDER_DIR, tar_basename)
        if not os.path.isfile(tar_path):
            logging.warning(f"Worker {worker_id}: {tar_path} not found, skipping.")
            continue

        extract_dir = tempfile.mkdtemp(prefix=f"older_worker{worker_id}_")
        try:
            with tarfile.open(tar_path, "r:*") as tf:
                tf.extractall(extract_dir)
            pdf_files = []
            for root, _dirs, files in os.walk(extract_dir):
                for f in files:
                    if f.lower().endswith(".pdf"):
                        pdf_files.append(os.path.join(root, f))
            logging.info(f"Worker {worker_id} processing {tar_basename}: {len(pdf_files)} PDFs")

            for pdf_path in pdf_files:
                try:
                    rec = process_pdf(pdf_path)
                    if rec is None or not (rec.get("description") or "").strip():
                        continue
                    batch_titles.append(rec["title"])
                    batch_descriptions.append(rec["description"])

                    if len(batch_descriptions) >= BATCH_SIZE:
                        logging.info(f"Worker {worker_id} embedding {len(batch_descriptions)} from {tar_basename}")
                        embeddings = model.encode(
                            batch_descriptions,
                            batch_size=MODEL_BATCH_SIZE,
                            show_progress_bar=False,
                        )
                        flush_buffer(embeddings, batch_titles, chunk_index, chunk_lock)
                        chunk_index += 1
                        batch_titles.clear()
                        batch_descriptions.clear()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                except Exception as e:
                    logging.debug(f"Worker {worker_id} skip PDF {pdf_path}: {e}")
                    continue

            if batch_descriptions:
                logging.info(f"Worker {worker_id} embedding final {len(batch_descriptions)} from {tar_basename}")
                embeddings = model.encode(
                    batch_descriptions,
                    batch_size=MODEL_BATCH_SIZE,
                    show_progress_bar=False,
                )
                flush_buffer(embeddings, batch_titles, chunk_index, chunk_lock)
                chunk_index += 1
                batch_titles.clear()
                batch_descriptions.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            update_progress(tar_basename, progress_lock)
        except Exception as e:
            logging.error(f"Worker {worker_id} error processing {tar_basename}: {e}")
        finally:
            if os.path.isdir(extract_dir):
                try:
                    shutil.rmtree(extract_dir)
                except OSError as e:
                    logging.warning(f"Worker {worker_id} could not remove {extract_dir}: {e}")

    logging.info(f"Worker {worker_id} completed processing.")


def main():
    logging.info("Starting patent grant PDF processing (1790-2002) for bulk-older/")
    if fitz is None:
        logging.error("PyMuPDF is required. Install with: pip install pymupdf")
        return

    tar_files = glob.glob(os.path.join(BULK_OLDER_DIR, "*.tar"))
    files_to_process = [os.path.basename(f) for f in tar_files]
    logging.info(f"Found {len(files_to_process)} tar files in {BULK_OLDER_DIR}")

    if not files_to_process:
        logging.error("No .tar files to process in bulk-older/")
        return

    progress_lock = mp.Lock()
    chunk_lock = mp.Lock()
    num_workers = min(3, max(1, torch.cuda.device_count()))
    logging.info(f"Starting {num_workers} worker processes")

    processes = []
    for worker_id in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(worker_id, files_to_process, progress_lock, chunk_lock),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    logging.info("All worker processes completed. Run merge_rag_index.py to merge with XML index.")


if __name__ == "__main__":
    main()
