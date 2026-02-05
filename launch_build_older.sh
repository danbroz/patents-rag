#!/bin/bash

# Launch script for build_older_grants.py (1790-2002 PDFs)
# Processes all .tar files in bulk-older/ with multi-GPU support

echo "Launching build_older_grants.py for 1790-2002 patent PDFs..."
echo "Processing all .tar files in bulk-older/..."

# Activate virtual environment if present, then run
if [ -f venv/bin/activate ]; then
  source venv/bin/activate
fi
python3 build_older_grants.py

echo "Older grants build process completed!"
echo "Run: python3 merge_rag_index.py to merge with XML index."
