#!/usr/bin/env python3
"""
Moved to old_scripts/. Kept for reference.

Original: wget-based Thursday APPXML downloads (legacy).
"""

import datetime
import os
import subprocess


def download_uspto_thursday_files():
    start_date = datetime.date(2001, 3, 15)
    end_date = datetime.date(2025, 9, 25)

    # Ensure the download directory exists
    download_dir = "/home/dan/patents-rag/bulk"
    os.makedirs(download_dir, exist_ok=True)

    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/109.0.0.0 Safari/537.36"
    )

    current = start_date
    while current <= end_date:
        if current.weekday() == 3:
            year_4digit = current.year
            prefix = "pa" if year_4digit < 2005 else "ipa"
            yy = str(year_4digit % 100).zfill(2)
            mm = str(current.month).zfill(2)
            dd = str(current.day).zfill(2)

            filename = f"{prefix}{yy}{mm}{dd}.zip"
            url = (
                "https://data.uspto.gov/api/v1/datasets/products/files/APPXML/"
                f"{year_4digit}/{filename}"
            )

            print(f"Downloading {url} ...")
            subprocess.run(
                [
                    "wget",
                    "-c",
                    "-nc",
                    f"--user-agent={user_agent}",
                    "-P",
                    download_dir,
                    url,
                ]
            )

        current += datetime.timedelta(days=1)


if __name__ == "__main__":
    download_uspto_thursday_files()

