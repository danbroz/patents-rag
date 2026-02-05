#!/usr/bin/env python3
"""
Unified downloader for patents-rag.

Downloads:
- Patent Grant XML (2002-present) into ./bulk/
- Patent Grant multi-page PDF images (1790-2002) into ./bulk-older/

Uses USPTO Open Data API and requires USPTO_API_KEY in .env.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


LOG = logging.getLogger("download")


def _today_iso() -> str:
    return _dt.date.today().isoformat()


def _get_api_key() -> str:
    load_dotenv()
    key = os.environ.get("USPTO_API_KEY", "").strip()
    if not key:
        raise SystemExit("USPTO_API_KEY is not set. Copy .env-example to .env and add your API key.")
    return key


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


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


def _download_file(api_key: str, bulk_dir: str, file_url: str, filename: str) -> bool:
    file_path = os.path.join(bulk_dir, filename)
    if os.path.exists(file_path):
        LOG.info("File already exists, skipping: %s", filename)
        return True

    headers = {"x-api-key": api_key, "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
    LOG.info("Downloading: %s", filename)
    LOG.info("URL: %s", file_url)

    try:
        r = requests.get(file_url, headers=headers, stream=True, timeout=300)
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0 and downloaded % (10 * 1024 * 1024) == 0:
                    pct = (downloaded / total_size) * 100
                    LOG.info("Downloaded %.1fMB / %.1fMB (%.1f%%)", downloaded / (1024 * 1024), total_size / (1024 * 1024), pct)
        LOG.info("Successfully downloaded: %s", filename)
        return True
    except requests.exceptions.RequestException as e:
        LOG.error("Error downloading %s: %s", filename, e)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        return False


def download_dataset(*, name: str, api_base_url: str, bulk_dir: str, start_date: str, end_date: str) -> None:
    api_key = _get_api_key()
    _ensure_dir(bulk_dir)

    api_data = _get_file_list(api_base_url, api_key, start_date, end_date)
    files_to_download = _extract_files_to_download(api_data)
    if not files_to_download:
        LOG.error("[%s] No files found in API response.", name)
        LOG.info("[%s] Response keys: %s", name, list(api_data.keys()) if isinstance(api_data, dict) else type(api_data))
        return

    LOG.info("[%s] Found %d files to download", name, len(files_to_download))
    ok = 0
    fail = 0
    for i, file_info in enumerate(files_to_download, 1):
        filename, file_url, file_size = _file_name_and_url(file_info)
        if not filename or not file_url:
            LOG.warning("[%s] Could not extract filename/URL for item %d/%d", name, i, len(files_to_download))
            fail += 1
            continue
        if file_size:
            LOG.info("[%s] %d/%d: %s (%.1f MB)", name, i, len(files_to_download), filename, file_size / (1024 * 1024))
        else:
            LOG.info("[%s] %d/%d: %s", name, i, len(files_to_download), filename)

        if _download_file(api_key, bulk_dir, file_url, filename):
            ok += 1
        else:
            fail += 1

    LOG.info("[%s] Download complete: %d successful, %d failed", name, ok, fail)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified patents-rag downloader (XML + older PDFs).")
    parser.add_argument("--xml", action="store_true", help="Download patent grant XML zips (2002-present)")
    parser.add_argument("--pdf", action="store_true", help="Download older patent grant PDF tars (1790-2002)")
    parser.add_argument("--xml-start", default="2002-01-01")
    parser.add_argument("--xml-end", default=_today_iso())
    parser.add_argument("--pdf-start", default="1790-07-31")
    parser.add_argument("--pdf-end", default="2002-02-01")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")

    want_xml = args.xml or (not args.xml and not args.pdf)
    want_pdf = args.pdf or (not args.xml and not args.pdf)

    if want_xml:
        download_dataset(
            name="XML",
            api_base_url="https://api.uspto.gov/api/v1/datasets/products/ptgrxml",
            bulk_dir="/home/dan/patents-rag/bulk",
            start_date=args.xml_start,
            end_date=args.xml_end,
        )
    if want_pdf:
        download_dataset(
            name="PDF",
            api_base_url="https://api.uspto.gov/api/v1/datasets/products/ptgrmp2",
            bulk_dir="/home/dan/patents-rag/bulk-older",
            start_date=args.pdf_start,
            end_date=args.pdf_end,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import datetime
import os
import subprocess

def download_uspto_thursday_files():
    start_date = datetime.date(2001, 3, 15)
    end_date   = datetime.date(2025, 9, 25)

    # Ensure the download directory exists
    download_dir = "/home/dan/patents-rag/bulk"
    os.makedirs(download_dir, exist_ok=True)

    # Use a Chrome or Firefox user agent to look more like a normal browser
    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/109.0.0.0 Safari/537.36"
    )
    # If you prefer Firefox, you could use something like:
    # user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:110.0) Gecko/20100101 Firefox/110.0"

    current = start_date
    while current <= end_date:
        # weekday() == 3 => Thursday (Monday=0, Tuesday=1, etc.)
        if current.weekday() == 3:
            year_4digit = current.year

            # For years < 2005, prefix is "pa"; for >= 2005, prefix is "ipa"
            prefix = "pa" if year_4digit < 2005 else "ipa"

            # Build the 2-digit year, month, and day
            yy = str(year_4digit % 100).zfill(2)
            mm = str(current.month).zfill(2)
            dd = str(current.day).zfill(2)

            # Construct the filename and full URL
            filename = f"{prefix}{yy}{mm}{dd}.zip"
            url = (
                "https://data.uspto.gov/api/v1/datasets/products/files/APPXML/"
                f"{year_4digit}/{filename}"
            )

            print(f"Downloading {url} ...")

            # Use wget with:
            #  -c  : resume partial downloads if interrupted
            #  -nc : skip if the file already exists completely
            #  --user-agent=... : set our custom user agent
            #  -P /home/dan/patents/bulk : download into this directory
            subprocess.run([
                "wget",
                "-c",             # resume partial downloads
                "-nc",            # skip existing files
                f"--user-agent={user_agent}",
                "-P", download_dir,
                url
            ])

        # Move to the next calendar day
        current += datetime.timedelta(days=1)

if __name__ == "__main__":
    download_uspto_thursday_files()

