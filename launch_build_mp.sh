#!/bin/bash

# Launch script for build_grants_mp.py with 3 RTX 4090 GPUs using multiprocessing

echo "Launching build_grants_mp.py with 3 RTX 4090 GPUs using multiprocessing..."

# Activate virtual environment and run
source venv/bin/activate && python3 build_grants_mp.py

echo "Build process completed!"

