#!/usr/bin/env python3
import os
import logging
import re
import sys
import zipfile
import tarfile
import tempfile
import shutil
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from lxml import etree as ET
from html import unescape

try:
    import faiss  # faiss-gpu (or faiss-cpu)
except Exception:
    faiss = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import pytesseract
except Exception:
    pytesseract = None

# =============================================================================
# Hardware / Parallelism Auto-Tuning
# =============================================================================

def autotune_torch_threads(max_threads: int = 16):
    cores = os.cpu_count() or 1
    n = min(max_threads, cores)
    torch.set_num_threads(n)
    torch.set_num_interop_threads(n)


def detect_workers(max_gpus: int = 3) -> int:
    gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return min(max_gpus, max(1, gpus))


torch.backends.cudnn.benchmark = True

# =============================================================================
# Configuration
# =============================================================================

ZIP_DIR = "bulk"
TAR_DIR = "bulk-older"
XML_FILES_DIR = "xml-files"
PDF_TEMP_PREFIX = "pdf_extract_"
FINAL_INDEX_DIR = "patents-index"
FINAL_EMB_DIR   = "patents-embeddings"
FAISS_PATH = os.path.join(FINAL_EMB_DIR, "index.faiss")
PROGRESS_ZIP_FILE = "checkpoints/build-progress.txt"
PROGRESS_TAR_FILE = "checkpoints/build-progress-older.txt"

# =============================================================================
# Batch sizes (defaults; adjusted by GPU presence)
# =============================================================================

DEFAULT_BATCH_SIZE_GPU = 64
DEFAULT_MODEL_BATCH_GPU = 64
DEFAULT_BATCH_SIZE_CPU = 16
DEFAULT_MODEL_BATCH_CPU = 16

# Create directories if they don't exist
for d in [XML_FILES_DIR, FINAL_INDEX_DIR, FINAL_EMB_DIR]:
    os.makedirs(d, exist_ok=True)
os.makedirs(os.path.dirname(PROGRESS_ZIP_FILE), exist_ok=True)
os.makedirs(os.path.dirname(PROGRESS_TAR_FILE), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

# =============================================================================
# Progress File Functions
# =============================================================================

def load_progress(path: str) -> set[str]:
    processed = set()
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                processed.add(line.strip())
    return processed

def update_progress(path: str, filename: str):
    with open(path, "a") as f:
        f.write(filename + "\n")
    logging.info(f"Updated progress file with {filename}")

# =============================================================================
# XML Parsing for Patent Grants (2002+)
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

    for elem in root.xpath('//us-patent-grant'):
        title_elem = elem.find("us-bibliographic-data-grant/invention-title")
        title = title_elem.text.strip() if (title_elem is not None and title_elem.text) else "No Title"

        description = ""
        abstract_elem = elem.find("abstract")
        if abstract_elem is not None:
            paragraphs = [p.text.strip() for p in abstract_elem.findall(".//p") if p.text]
            description = " ".join(paragraphs)

        if not description:
            desc_elem = elem.find("description")
            if desc_elem is not None:
                paragraphs = [p.text.strip() for p in desc_elem.findall(".//p") if p.text]
                description = " ".join(paragraphs)

        docnum_elem = elem.find("us-bibliographic-data-grant/publication-reference/document-id/doc-number")
        patent_id = docnum_elem.text.strip() if (docnum_elem is not None and docnum_elem.text) else "NoID"

        yield {"id": patent_id, "title": title, "description": description}

# =============================================================================
# PDF Extraction (1790-2002)
# =============================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required for PDF processing. Install pymupdf.")
    text_parts: list[str] = []
    doc = fitz.open(pdf_path)
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            t = page.get_text("text") or ""
            if t.strip() and len(t.strip()) >= 20:
                text_parts.append(t)
            elif pytesseract is not None:
                # OCR fallback for image-only pages
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
    finally:
        doc.close()
    text = " ".join(text_parts)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def process_pdf(pdf_path: str) -> Optional[dict]:
    text = extract_text_from_pdf(pdf_path)
    if not text or len(text) < 10:
        return None
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    title = (text.split("\n")[0].strip() if "\n" in text else text[:100].strip())[:200] or "No Title"
    desc = re.sub(r"\s+", " ", text).strip()
    if len(desc) > 50000:
        desc = desc[:50000]
    return {"id": stem or "Unknown", "title": title, "description": desc}


@dataclass
class BuildConfig:
    batch_size: int
    model_batch_size: int
    num_workers: int


def atomic_save_titles(titles: list[str], final_path: str):
    tmp_path = final_path + ".tmp"
    np.save(tmp_path, np.array(titles, dtype=object))
    # np.save appends .npy if not present; ensure we replace correctly
    if not tmp_path.endswith(".npy"):
        tmp_path = tmp_path + ".npy"
    os.replace(tmp_path, final_path)


def atomic_save_faiss(index, final_path: str):
    tmp_path = final_path + ".tmp"
    faiss.write_index(index, tmp_path)
    os.replace(tmp_path, final_path)


def ensure_faiss():
    if faiss is None:
        raise SystemExit("FAISS not available. Install faiss-gpu (or faiss-cpu).")


def build_config() -> BuildConfig:
    autotune_torch_threads()
    workers = detect_workers()
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return BuildConfig(
            batch_size=DEFAULT_BATCH_SIZE_GPU,
            model_batch_size=DEFAULT_MODEL_BATCH_GPU,
            num_workers=workers,
        )
    return BuildConfig(
        batch_size=DEFAULT_BATCH_SIZE_CPU,
        model_batch_size=DEFAULT_MODEL_BATCH_CPU,
        num_workers=1,
    )


def discover_sources() -> tuple[list[str], list[str]]:
    zips = sorted([os.path.join(ZIP_DIR, f) for f in os.listdir(ZIP_DIR)] ) if os.path.isdir(ZIP_DIR) else []
    zips = [p for p in zips if p.lower().endswith(".zip")]
    tars = sorted([os.path.join(TAR_DIR, f) for f in os.listdir(TAR_DIR)] ) if os.path.isdir(TAR_DIR) else []
    tars = [p for p in tars if p.lower().endswith(".tar")]
    return zips, tars


def main():
    ensure_faiss()

    cfg = build_config()
    logging.info(f"Detected resources: cpu_cores={os.cpu_count()}, gpus={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    logging.info(f"Using batch_size={cfg.batch_size} model_batch_size={cfg.model_batch_size} workers={cfg.num_workers}")

    zips, tars = discover_sources()
    logging.info(f"Discovered sources: {len(zips)} zip(s) in {ZIP_DIR}/, {len(tars)} tar(s) in {TAR_DIR}/")
    if not zips and not tars:
        logging.error("No source archives found. Expected bulk/*.zip and/or bulk-older/*.tar")
        return

    processed_zips = load_progress(PROGRESS_ZIP_FILE)
    processed_tars = load_progress(PROGRESS_TAR_FILE)

    # Load or create FAISS index + titles
    titles_path = os.path.join(FINAL_INDEX_DIR, "patent_titles.npy")
    titles: list[str] = []
    if os.path.isfile(titles_path):
        titles = np.load(titles_path, allow_pickle=True).tolist()

    index = None
    if os.path.isfile(FAISS_PATH):
        index = faiss.read_index(FAISS_PATH)
        logging.info(f"Loaded existing FAISS index from {FAISS_PATH} (ntotal={index.ntotal})")

    # Model per-process (single-process by default; multi-GPU parallelization can be added later)
    device = "cuda:0" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    logging.info(f"Loaded SentenceTransformer on {device}")

    def ensure_index(dim: int):
        nonlocal index
        if index is None:
            index = faiss.IndexFlatIP(dim)
            logging.info(f"Created FAISS IndexFlatIP(dim={dim})")

    buffered_texts: list[str] = []
    buffered_titles: list[str] = []
    flush_every = max(5_000, cfg.batch_size * 50)
    added_total = 0

    def flush():
        nonlocal added_total
        if not buffered_texts:
            return
        vecs = model.encode(
            buffered_texts,
            batch_size=cfg.model_batch_size,
            show_progress_bar=False,
        ).astype(np.float32)
        ensure_index(vecs.shape[1])
        index.add(vecs)
        titles.extend(buffered_titles)
        added_total += len(buffered_titles)
        buffered_texts.clear()
        buffered_titles.clear()
        if added_total % flush_every == 0:
            atomic_save_titles(titles, titles_path)
            atomic_save_faiss(index, FAISS_PATH)
            logging.info(f"Checkpointed FAISS+titles (total_titles={len(titles)} ntotal={index.ntotal})")

    # Process zip (XML) sources
    for zip_path in zips:
        base = os.path.basename(zip_path)
        if base in processed_zips:
            continue
        if not os.path.isfile(zip_path):
            continue
        try:
            extracted = extract_zip_file(zip_path, XML_FILES_DIR)
            xml_files = [p for p in extracted if p.lower().endswith(".xml")]
            if not xml_files:
                logging.warning(f"No XML files found in {zip_path}")
            for xml_file in xml_files:
                for patent in parse_patent_grants(xml_file) or []:
                    desc = (patent.get("description") or "").strip()
                    if not desc:
                        continue
                    buffered_titles.append(patent.get("title") or "No Title")
                    buffered_texts.append(desc)
                    if len(buffered_texts) >= cfg.batch_size:
                        flush()
                try:
                    os.remove(xml_file)
                except OSError:
                    pass
            flush()
            update_progress(PROGRESS_ZIP_FILE, base)
            # delete source zip after successful processing
            try:
                os.remove(zip_path)
                logging.info(f"Deleted processed zip: {zip_path}")
            except OSError as e:
                logging.warning(f"Failed to delete zip {zip_path}: {e}")
        except Exception as e:
            logging.error(f"Failed processing zip {zip_path}: {e}")
            continue

    # Process tar (PDF) sources
    for tar_path in tars:
        base = os.path.basename(tar_path)
        if base in processed_tars:
            continue
        if not os.path.isfile(tar_path):
            continue
        extract_dir = tempfile.mkdtemp(prefix=PDF_TEMP_PREFIX)
        try:
            with tarfile.open(tar_path, "r:*") as tf:
                tf.extractall(extract_dir)
            pdf_files: list[str] = []
            for root, _dirs, files in os.walk(extract_dir):
                for f in files:
                    if f.lower().endswith(".pdf"):
                        pdf_files.append(os.path.join(root, f))
            for pdf_path in pdf_files:
                rec = None
                try:
                    rec = process_pdf(pdf_path)
                except Exception as e:
                    logging.debug(f"Skip PDF {pdf_path}: {e}")
                if not rec:
                    continue
                buffered_titles.append(rec["title"])
                buffered_texts.append(rec["description"])
                if len(buffered_texts) >= cfg.batch_size:
                    flush()
            flush()
            update_progress(PROGRESS_TAR_FILE, base)
            try:
                os.remove(tar_path)
                logging.info(f"Deleted processed tar: {tar_path}")
            except OSError as e:
                logging.warning(f"Failed to delete tar {tar_path}: {e}")
        except Exception as e:
            logging.error(f"Failed processing tar {tar_path}: {e}")
            continue
        finally:
            try:
                shutil.rmtree(extract_dir)
            except OSError:
                pass

    # Final write
    flush()
    if index is None:
        logging.error("No vectors were added; nothing to write.")
        return

    atomic_save_titles(titles, titles_path)
    atomic_save_faiss(index, FAISS_PATH)
    logging.info(f"Build complete. titles={len(titles)} faiss_ntotal={index.ntotal}")


if __name__ == "__main__":
    main()

