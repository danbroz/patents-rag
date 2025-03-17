#!/usr/bin/env python3
import datetime
import os
import subprocess

def download_uspto_thursday_files():
    start_date = datetime.date(2001, 3, 15)
    end_date   = datetime.date(2025, 2, 20)

    # Ensure the download directory exists
    download_dir = "/home/dan/patents/bulk"
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

