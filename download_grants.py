#!/usr/bin/env python3
"""
Download Patent Grant Full-Text Data (No Images) - XML from USPTO API
"""

import requests
import os
import json
import logging
from pathlib import Path

# Configuration
API_KEY = "yyfujzllgeprxhjuwurymehlwhdefh"
API_BASE_URL = "https://api.uspto.gov/api/v1/datasets/products/ptgrxml"
BULK_DIR = "/home/dan/patents-rag/bulk"

# Date range for the data
START_DATE = "2002-01-01"
END_DATE = "2025-09-26"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

def ensure_bulk_dir():
    """Ensure the bulk directory exists"""
    Path(BULK_DIR).mkdir(parents=True, exist_ok=True)
    logging.info(f"Bulk directory ready: {BULK_DIR}")

def get_file_list():
    """Get the list of files available for download from the API"""
    url = f"{API_BASE_URL}"
    params = {
        'fileDataFromDate': START_DATE,
        'fileDataToDate': END_DATE,
        'includeFiles': 'true'
    }
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'x-api-key': API_KEY
    }
    
    logging.info(f"Fetching file list from: {url}")
    logging.info(f"Date range: {START_DATE} to {END_DATE}")
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        logging.info(f"API response received successfully")
        
        # Log the structure of the response for debugging
        logging.info(f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        
        return data
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching file list: {e}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON response: {e}")
        logging.error(f"Response content: {response.text[:500]}...")
        raise

def download_file(file_url, filename):
    """Download a single file from the given URL"""
    file_path = os.path.join(BULK_DIR, filename)
    
    # Skip if file already exists
    if os.path.exists(file_path):
        logging.info(f"File already exists, skipping: {filename}")
        return True
    
    logging.info(f"Downloading: {filename}")
    logging.info(f"URL: {file_url}")
    
    try:
        headers = {
            'x-api-key': API_KEY,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(file_url, headers=headers, stream=True, timeout=300)
        response.raise_for_status()
        
        # Get file size for progress tracking
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # Log progress every 10MB
                    if total_size > 0 and downloaded_size % (10 * 1024 * 1024) == 0:
                        progress = (downloaded_size / total_size) * 100
                        logging.info(f"Downloaded {downloaded_size / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB ({progress:.1f}%)")
        
        logging.info(f"Successfully downloaded: {filename}")
        return True
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading {filename}: {e}")
        # Remove partial file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)
        return False

def main():
    """Main function to orchestrate the download process"""
    logging.info("Starting Patent Grant XML download process")
    
    # Ensure bulk directory exists
    ensure_bulk_dir()
    
    # Get file list from API
    try:
        api_data = get_file_list()
        
        # Parse the response to extract file information
        files_to_download = []
        
        # The API response structure contains 'bulkDataProductBag'
        if isinstance(api_data, dict) and 'bulkDataProductBag' in api_data:
            product_bag = api_data['bulkDataProductBag']
            if isinstance(product_bag, list) and len(product_bag) > 0:
                product = product_bag[0]  # Get the first (and likely only) product
                
                # Look for file information in the product
                if 'productFileBag' in product and 'fileDataBag' in product['productFileBag']:
                    files_to_download = product['productFileBag']['fileDataBag']
                    logging.info(f"Found {len(files_to_download)} files in productFileBag.fileDataBag")
                elif 'productFileArray' in product:
                    files_to_download = product['productFileArray']
                elif 'fileArray' in product:
                    files_to_download = product['fileArray']
                else:
                    # Log the product structure to understand what's available
                    logging.info(f"Product structure: {list(product.keys())}")
                    logging.info(f"Full product data: {json.dumps(product, indent=2)[:2000]}...")
                    
                    # Try to find any nested file information
                    for key, value in product.items():
                        if isinstance(value, list) and len(value) > 0:
                            if isinstance(value[0], dict) and any('url' in str(item).lower() or 'file' in str(item).lower() for item in value):
                                files_to_download = value
                                break
                    
        elif isinstance(api_data, list):
            files_to_download = api_data
        
        if not files_to_download:
            logging.error("No files found to download. Please check the API response structure.")
            return
        
        logging.info(f"Found {len(files_to_download)} files to download")
        
        # Download each file
        successful_downloads = 0
        failed_downloads = 0
        
        for i, file_info in enumerate(files_to_download, 1):
            logging.info(f"Processing file {i}/{len(files_to_download)}")
            
            # Extract filename and URL from file info
            # The structure may vary, so we'll try different approaches
            filename = None
            file_url = None
            
            if isinstance(file_info, dict):
                # Try common field names for filename and URL
                filename = (file_info.get('fileName') or 
                          file_info.get('filename') or 
                          file_info.get('name') or 
                          file_info.get('file'))
                file_url = (file_info.get('fileDownloadURI') or 
                          file_info.get('url') or 
                          file_info.get('downloadUrl') or 
                          file_info.get('link'))
            elif isinstance(file_info, str):
                # If it's just a string, assume it's a URL and extract filename
                file_url = file_info
                filename = file_url.split('/')[-1]
            
            if not filename or not file_url:
                logging.warning(f"Could not extract filename or URL from: {file_info}")
                failed_downloads += 1
                continue
            
            # Log file information
            file_size = file_info.get('fileSize', 'unknown') if isinstance(file_info, dict) else 'unknown'
            if file_size != 'unknown':
                file_size_mb = file_size / (1024 * 1024)
                logging.info(f"File: {filename} ({file_size_mb:.1f} MB)")
            else:
                logging.info(f"File: {filename}")
            
            # Download the file
            if download_file(file_url, filename):
                successful_downloads += 1
            else:
                failed_downloads += 1
        
        logging.info(f"Download complete: {successful_downloads} successful, {failed_downloads} failed")
        
    except Exception as e:
        logging.error(f"Fatal error in main process: {e}")
        raise

if __name__ == "__main__":
    main()
