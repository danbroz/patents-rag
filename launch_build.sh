#!/bin/bash

# Launch script for build_grants.py with 3 RTX 4090 GPUs on single node

# Set environment variables for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=12356
export WORLD_SIZE=3

# Launch 3 processes, one for each GPU
echo "Launching build_grants.py with 3 RTX 4090 GPUs..."

# Process 0 on GPU 0
RANK=0 CUDA_VISIBLE_DEVICES=0 bash -c "source venv/bin/activate && python3 build_grants.py" &
PID0=$!

# Process 1 on GPU 1  
RANK=1 CUDA_VISIBLE_DEVICES=1 bash -c "source venv/bin/activate && python3 build_grants.py" &
PID1=$!

# Process 2 on GPU 2
RANK=2 CUDA_VISIBLE_DEVICES=2 bash -c "source venv/bin/activate && python3 build_grants.py" &
PID2=$!

echo "Launched processes: $PID0, $PID1, $PID2"

# Wait for all processes to complete
wait $PID0
wait $PID1
wait $PID2

echo "All processes completed!"
