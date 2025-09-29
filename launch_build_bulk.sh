#!/bin/bash

# Launch script for build_grants_bulk.py with 3 RTX 4090 GPUs using multiprocessing
# Processes all files in the bulk folder

echo "Launching build_grants_bulk.py with 3 RTX 4090 GPUs using multiprocessing..."
echo "Processing all files in bulk folder..."

# Activate virtual environment and run
source venv/bin/activate && python3 build_grants_bulk.py

echo "Bulk build process completed!"

