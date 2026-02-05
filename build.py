#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import logging
import re
import sys
import zipfile
import tarfile
import tempfile
import shutil
from dataclasses import dataclass
from typing import Iterable, Optional, Any

import numpy as np
import torch
import requests
from dotenv import load_dotenv
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
PROGRESS_TAR_FAILED_FILE = "checkpoints/build-progress-older-failed.txt"

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
os.makedirs(os.path.dirname(PROGRESS_TAR_FAILED_FILE), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

# =============================================================================
# USPTO API helpers (integrated former download.py logic)
# =============================================================================

LOG = logging.getLogger("build")


def _today_iso() -> str:
    return __import__("datetime").date.today().isoformat()


def _get_api_key() -> str:
    load_dotenv()
    key = os.environ.get("USPTO_API_KEY", "").strip()
    if not key:
        raise SystemExit("USPTO_API_KEY is not set. Copy .env-example to .env and add your API key.")
    return key


def _get_file_list(api_base_url: str, api_key: str, start_date: str, end_date: str) -> dict[str, Any]:
    params = {"fileDataFromDate": start_date, "fileDataToDate": end_date, "includeFiles": "true"}
    headers = {"Accept": "application/json", "Content-Type": "application/json", "x-api-key": api_key}
    LOG.info("Fetching file list from: %s", api_base_url)
    LOG.info("Date range: %s to %s", start_date, end_date)
    r = requests.get(api_base_url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def _extract_files_to_download(api_data: Any) -> list[Any]:
    files_to_download: list[Any] = []
    if isinstance(api_data, dict) and "bulkDataProductBag" in api_data:
        product_bag = api_data["bulkDataProductBag"]
        if isinstance(product_bag, list) and product_bag:
            product = product_bag[0]
            if isinstance(product, dict):
                if "productFileBag" in product and isinstance(product["productFileBag"], dict):
                    pfb = product["productFileBag"]
                    if "fileDataBag" in pfb:
                        files_to_download = pfb["fileDataBag"] or []
                elif "productFileArray" in product:
                    files_to_download = product["productFileArray"] or []
                elif "fileArray" in product:
                    files_to_download = product["fileArray"] or []
    elif isinstance(api_data, list):
        files_to_download = api_data
    return files_to_download


def _file_name_and_url(file_info: Any) -> tuple[str | None, str | None, int | None]:
    filename = None
    file_url = None
    file_size = None
    if isinstance(file_info, dict):
        filename = file_info.get("fileName") or file_info.get("filename") or file_info.get("name") or file_info.get("file")
        file_url = file_info.get("fileDownloadURI") or file_info.get("url") or file_info.get("downloadUrl") or file_info.get("link")
        try:
            file_size = int(file_info.get("fileSize")) if file_info.get("fileSize") is not None else None
        except Exception:
            file_size = None
    elif isinstance(file_info, str):
        file_url = file_info
        filename = file_url.split("/")[-1]
    return filename, file_url, file_size


def _download_archive(api_key: str, dest_dir: str, file_url: str, filename: str) -> bool:
    dest_path = os.path.join(dest_dir, filename)
    if os.path.exists(dest_path):
        LOG.info("File already exists, skipping download: %s", filename)
        return True
    headers = {"x-api-key": api_key, "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
    LOG.info("Downloading: %s", filename)
    LOG.info("URL: %s", file_url)
    try:
        r = requests.get(file_url, headers=headers, stream=True, timeout=300)
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0 and downloaded % (10 * 1024 * 1024) == 0:
                    pct = (downloaded / total_size) * 100
                    LOG.info(
                        "Downloaded %.1fMB / %.1fMB (%.1f%%)",
                        downloaded / (1024 * 1024),
                        total_size / (1024 * 1024),
                        pct,
                    )
        LOG.info("Successfully downloaded: %s", filename)
        return True
    except requests.exceptions.RequestException as e:
        LOG.error("Error downloading %s: %s", filename, e)
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except OSError:
                pass
        return False
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


def deduplicate_titles_and_index(titles: list[str], index):
    """
    Deduplicate entries by title at the end of the run.
    This prevents duplicate vectors/titles when the same patent appears
    multiple times (e.g. across reissues or overlapping sources).
    """
    if index is None or not titles:
        return titles, index

    n = len(titles)
    if index.ntotal != n:
        logging.warning(
            "Cannot deduplicate: index.ntotal (%d) != len(titles) (%d)",
            index.ntotal,
            n,
        )
        return titles, index

    seen: dict[str, int] = {}
    unique_positions: list[int] = []
    for i, t in enumerate(titles):
        if t in seen:
            continue
        seen[t] = i
        unique_positions.append(i)

    if len(unique_positions) == n:
        logging.info("No duplicate titles found; skipping deduplication.")
        return titles, index

    dim = index.d
    try:
        xb = faiss.vector_to_array(index.xb).reshape(-1, dim)
    except Exception as e:
        logging.warning("Failed to access index vectors for deduplication: %s", e)
        return titles, index

    xb_unique = xb[unique_positions]
    new_index = faiss.IndexFlatIP(dim)
    new_index.add(xb_unique)
    new_titles = [titles[i] for i in unique_positions]

    logging.info(
        "Deduplicated index by title: %d -> %d entries",
        n,
        len(unique_positions),
    )
    return new_titles, new_index


def discover_sources() -> tuple[list[str], list[str]]:
    zips = sorted([os.path.join(ZIP_DIR, f) for f in os.listdir(ZIP_DIR)]) if os.path.isdir(ZIP_DIR) else []
    zips = [p for p in zips if p.lower().endswith(".zip")]
    tars = sorted([os.path.join(TAR_DIR, f) for f in os.listdir(TAR_DIR)]) if os.path.isdir(TAR_DIR) else []
    tars = [p for p in tars if p.lower().endswith(".tar")]
    return zips, tars


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from USPTO patent data (download + process).")
    parser.add_argument("--xml-only", action="store_true", help="Process/download only XML grant zips (bulk/).")
    parser.add_argument("--pdf-only", action="store_true", help="Process/download only older PDF tars (bulk-older/).")
    parser.add_argument("--xml-start", default="2002-01-01", help="Start date for XML API (YYYY-MM-DD).")
    parser.add_argument("--xml-end", default=_today_iso(), help="End date for XML API (YYYY-MM-DD).")
    args = parser.parse_args()

    ensure_faiss()

    cfg = build_config()
    logging.info(f"Detected resources: cpu_cores={os.cpu_count()}, gpus={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    logging.info(f"Using batch_size={cfg.batch_size} model_batch_size={cfg.model_batch_size} workers={cfg.num_workers}")

    want_xml = args.xml_only or not args.pdf_only
    want_pdf = args.pdf_only or not args.xml_only

    processed_zips = load_progress(PROGRESS_ZIP_FILE) if want_xml else set()
    processed_tars = load_progress(PROGRESS_TAR_FILE) if want_pdf else set()
    failed_tars = load_progress(PROGRESS_TAR_FAILED_FILE) if want_pdf else set()

    # Clean up any already-processed archives left on disk from older runs.
    # If a filename is in the checkpoint, we assume its contents are already
    # reflected in FAISS/titles and we can safely delete the source archive.
    if want_xml and os.path.isdir(ZIP_DIR):
        for base in os.listdir(ZIP_DIR):
            if not base.lower().endswith(".zip"):
                continue
            if base in processed_zips:
                zip_path = os.path.join(ZIP_DIR, base)
                try:
                    os.remove(zip_path)
                    logging.info("Deleted previously processed zip lingering in %s: %s", ZIP_DIR, base)
                except OSError as e:
                    logging.warning("Failed to delete previously processed zip %s: %s", zip_path, e)

    if want_pdf and os.path.isdir(TAR_DIR):
        for base in os.listdir(TAR_DIR):
            if not base.lower().endswith(".tar"):
                continue
            if base in processed_tars or base in failed_tars:
                tar_path = os.path.join(TAR_DIR, base)
                try:
                    os.remove(tar_path)
                    logging.info("Deleted previously processed/failed tar lingering in %s: %s", TAR_DIR, base)
                except OSError as e:
                    logging.warning("Failed to delete previously processed/failed tar %s: %s", tar_path, e)

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

    api_key: str | None = None

    # Process XML (download + build) via API
    if want_xml:
        api_key = _get_api_key()
        try:
            api_data_xml = _get_file_list(
                "https://api.uspto.gov/api/v1/datasets/products/ptgrxml",
                api_key,
                args.xml_start,
                args.xml_end,
            )
            xml_files_info = _extract_files_to_download(api_data_xml)
        except Exception as e:
            logging.error("Failed to fetch XML file list from USPTO API: %s", e)
            xml_files_info = []

        logging.info("XML: %d file(s) from API", len(xml_files_info))
        for i, file_info in enumerate(xml_files_info, 1):
            filename, file_url, file_size = _file_name_and_url(file_info)
            if not filename or not file_url:
                logging.warning("XML: could not extract filename/URL for item %d", i)
                continue
            # Skip non-zip artefacts like DTDs / sequence listings that are not grant archives
            if not filename.lower().endswith(".zip"):
                logging.info("Skipping non-zip XML API file: %s", filename)
                continue
            if filename in processed_zips:
                continue
            zip_path = os.path.join(ZIP_DIR, filename)

            def process_single_zip(path: str) -> bool:
                try:
                    extracted = extract_zip_file(path, XML_FILES_DIR)
                    xml_files = [p for p in extracted if p.lower().endswith(".xml")]
                    if not xml_files:
                        logging.warning("No XML files found in %s", path)
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
                    update_progress(PROGRESS_ZIP_FILE, filename)
                    try:
                        os.remove(path)
                        logging.info("Deleted processed zip: %s", path)
                    except OSError as e:
                        logging.warning("Failed to delete zip %s: %s", path, e)
                    return True
                except Exception as e:
                    logging.error("Failed processing zip %s: %s", path, e)
                    return False

            # Ensure we have the archive locally
            if not os.path.isfile(zip_path):
                os.makedirs(ZIP_DIR, exist_ok=True)
                if not _download_archive(api_key, ZIP_DIR, file_url, filename):
                    continue

            ok = process_single_zip(zip_path)
            if not ok:
                # If it failed because the local file is corrupt / not a zip, delete and redownload once
                # (common case: "File is not a zip file"). Re-run with a fresh copy from USPTO.
                try:
                    # Best-effort remove corrupt local archive
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
                        logging.info("Deleted corrupt zip before re-download: %s", zip_path)
                except OSError as rm_err:
                    logging.warning("Failed to delete corrupt zip %s: %s", zip_path, rm_err)

                # Attempt one re-download + re-process
                logging.info("Re-downloading zip after error: %s", filename)
                if _download_archive(api_key, ZIP_DIR, file_url, filename):
                    if not process_single_zip(zip_path):
                        logging.error("Re-download did not fix zip %s; skipping.", zip_path)
                        continue
                else:
                    logging.error("Re-download failed for %s; skipping.", filename)
                    continue

        # Fallback: process any remaining local zip archives in ZIP_DIR that
        # are not yet in the checkpoint file (e.g. manually downloaded or
        # left over from older runs but not covered by the current API query).
        if os.path.isdir(ZIP_DIR):
            for base in sorted(os.listdir(ZIP_DIR)):
                if not base.lower().endswith(".zip"):
                    continue
                if base in processed_zips:
                    continue
                zip_path = os.path.join(ZIP_DIR, base)
                logging.info("Processing leftover local zip not from API list: %s", zip_path)
                ok = process_single_zip(zip_path)
                if not ok:
                    try:
                        if os.path.exists(zip_path):
                            os.remove(zip_path)
                            logging.info("Deleted leftover corrupt zip: %s", zip_path)
                    except OSError as rm_err:
                        logging.warning("Failed to delete leftover corrupt zip %s: %s", zip_path, rm_err)

    # Process PDFs (download + build) via API
    if want_pdf:
        if api_key is None:
            api_key = _get_api_key()

        def process_single_tar(path: str, logical_name: str) -> bool:
            extract_dir = tempfile.mkdtemp(prefix=PDF_TEMP_PREFIX)
            try:
                with tarfile.open(path, "r:*") as tf:
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
                        logging.debug("Skip PDF %s: %s", pdf_path, e)
                    if not rec:
                        continue
                    buffered_titles.append(rec["title"])
                    buffered_texts.append(rec["description"])
                    if len(buffered_texts) >= cfg.batch_size:
                        flush()
                flush()
                update_progress(PROGRESS_TAR_FILE, logical_name)
                try:
                    os.remove(path)
                    logging.info("Deleted processed tar: %s", path)
                except OSError as e:
                    logging.warning("Failed to delete tar %s: %s", path, e)
                return True
            except Exception as e:
                logging.error("Failed processing tar %s: %s", path, e)
                return False
            finally:
                try:
                    shutil.rmtree(extract_dir)
                except OSError:
                    pass

        try:
            api_data_pdf = _get_file_list(
                "https://api.uspto.gov/api/v1/datasets/products/ptgrmp2",
                api_key,
                "1790-07-31",
                "2002-02-01",
            )
            pdf_files_info = _extract_files_to_download(api_data_pdf)
        except Exception as e:
            logging.error("Failed to fetch PDF file list from USPTO API: %s", e)
            pdf_files_info = []

        logging.info("PDF: %d file(s) from API", len(pdf_files_info))
        for i, file_info in enumerate(pdf_files_info, 1):
            filename, file_url, file_size = _file_name_and_url(file_info)
            if not filename or not file_url:
                logging.warning("PDF: could not extract filename/URL for item %d", i)
                continue
            if filename in processed_tars or filename in failed_tars:
                continue
            tar_path = os.path.join(TAR_DIR, filename)
            if not os.path.isfile(tar_path):
                os.makedirs(TAR_DIR, exist_ok=True)
                if not _download_archive(api_key, TAR_DIR, file_url, filename):
                    continue

            ok = process_single_tar(tar_path, filename)
            if not ok:
                # As with zips, assume corruption or non-tar content (e.g. HTML error page),
                # delete the local file, re-download once, and retry.
                try:
                    if os.path.exists(tar_path):
                        os.remove(tar_path)
                        logging.info("Deleted corrupt tar before re-download: %s", tar_path)
                except OSError as rm_err:
                    logging.warning("Failed to delete corrupt tar %s: %s", tar_path, rm_err)

                logging.info("Re-downloading tar after error: %s", filename)
                if _download_archive(api_key, TAR_DIR, file_url, filename):
                    if not process_single_tar(tar_path, filename):
                        logging.error("Re-download did not fix tar %s; marking as failed and skipping.", tar_path)
                        update_progress(PROGRESS_TAR_FAILED_FILE, filename)
                        continue
                else:
                    logging.error("Re-download failed for %s; marking as failed and skipping.", filename)
                    update_progress(PROGRESS_TAR_FAILED_FILE, filename)
                    continue

    # Final write
    flush()
    if index is None:
        logging.error("No vectors were added; nothing to write.")
        return

    # Deduplicate any duplicate titles/vectors before saving final artifacts.
    final_titles, final_index = deduplicate_titles_and_index(titles, index)

    atomic_save_titles(final_titles, titles_path)
    atomic_save_faiss(final_index, FAISS_PATH)
    logging.info(
        "Build complete. titles=%d faiss_ntotal=%d",
        len(final_titles),
        final_index.ntotal,
    )


if __name__ == "__main__":
    main()

