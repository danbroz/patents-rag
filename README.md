# Patent RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system for US Patent data, built with multi-GPU processing capabilities and support for both modern XML patents and historical PDF images.

## 🎯 Overview

This system downloads, processes, and creates embeddings for US Patent data from the USPTO, enabling semantic search across millions of patent documents. It supports:

- **Patent Grant XML files** (2002-2025) - Full-text patent data
- **Patent Grant PDF Images** (1790-2002) - Historical patent documents
- **Multi-GPU processing** with 3x RTX 4090 support
- **Distributed processing** with checkpointing and resume capabilities

## 📊 Current Status

- ✅ **2.27+ million patent embeddings** processed and ready
- ✅ **Patent Grant XML files** (2002-2025): 1,259 files, 113 GB
- ✅ **Patent Grant PDF Images** (1790-2002): 108 files, 1 TB
- ✅ **RAG system ready** for semantic search applications

## 🏗️ Architecture

### Data Pipeline
```
USPTO API → Download Scripts → XML Processing → Embedding Generation → RAG Index
```

### Key Components
- **Download Scripts**: Automated USPTO data retrieval
- **Processing Pipeline**: XML parsing and text extraction
- **Embedding Generation**: Sentence Transformers with GPU acceleration
- **Index Storage**: NumPy arrays and memory-mapped files

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPUs (tested with RTX 4090)
- 4+ TB storage space
- USPTO API key

### Installation
```bash
# Clone repository
git clone git@github.com:danbroz/patents-rag.git
cd patents-rag

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install torch torchvision torchaudio
pip install sentence-transformers lxml requests numpy

# Optional: for 1790-2002 PDFs with no text layer (OCR)
# pip install pytesseract Pillow
# System: sudo apt install tesseract-ocr   # Linux

# Set up your USPTO API key (get one at https://developer.uspto.gov/)
cp .env-example .env
# Edit .env and set USPTO_API_KEY=your-actual-key
```

### Basic Usage

#### 1. Download Patent Data
```bash
# Download modern XML patents (2002-2025)
python3 download_grants.py

# Download historical PDF patents (1790-2002)
python3 download_older_grants.py
```

#### 2. Process and Build Embeddings

To include **all** patents (1790–present), run in order:

```bash
# Step 1: Process XML patents (2002–present) from bulk/
./launch_build_bulk.sh
# or: python3 build_grants_bulk.py

# Step 2: Process PDF patents (1790–2002) from bulk-older/ (text extraction + optional OCR)
python3 build_older_grants.py
# or: ./launch_build_older.sh

# Step 3: Merge both into the final RAG index (older first, then 2002+)
python3 merge_rag_index.py
```

To use only XML data (2002–present), run step 1 only; the index is written by `build_grants_bulk.py` (no merge needed). The 1790–2002 pipeline uses PDF text extraction (PyMuPDF) and optional OCR (Tesseract) for image-only scans, so it is slower and noisier than the XML pipeline.

#### 3. Use the RAG System
```python
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load patent data
titles = np.load('patents-index/patent_titles.npy')
embeddings = np.memmap('patents-embeddings/embeddings.memmap', 
                       dtype='float32', mode='r')

# Semantic search
query = "artificial intelligence machine learning"
query_embedding = model.encode([query])

# Find similar patents
similarities = np.dot(embeddings, query_embedding.T).flatten()
top_indices = np.argsort(similarities)[-10:][::-1]

for idx in top_indices:
    print(f"Patent: {titles[idx]}")
    print(f"Similarity: {similarities[idx]:.4f}")
```

## 📁 Project Structure

```
patents-rag/
├── download_grants.py          # Download XML patents (2002-present)
├── download_older_grants.py    # Download PDF patents (1790-2002)
├── build_grants_bulk.py        # Process bulk/ XML → partial-patents/
├── build_older_grants.py       # Process bulk-older/ PDFs → partial-patents-older/
├── merge_rag_index.py          # Merge older + XML chunks → final index
├── launch_build_bulk.sh        # Run XML pipeline
├── launch_build_older.sh       # Run PDF (1790-2002) pipeline
├── checkpoints/                # Progress tracking (build-progress.txt, build-progress-older.txt)
├── bulk/                       # XML patent files (113+ GB)
├── bulk-older/                 # PDF patent tarballs (1 TB)
├── partial-patents/            # Chunks from XML pipeline
├── partial-patents-older/      # Chunks from PDF pipeline
├── patents-index/              # Processed patent titles
├── patents-embeddings/         # Generated embeddings
└── README.md                   # This file
```

## ⚙️ Configuration

### API key (`.env`)

The project uses a **USPTO API key** from environment variables. Keys are loaded from a `.env` file so you never commit secrets.

1. **Copy the example file and add your key:**
   ```bash
   cp .env-example .env
   ```
2. **Edit `.env`** and set your key:
   ```
   USPTO_API_KEY=your-uspto-api-key-here
   ```
3. Get a free key at [developer.uspto.gov](https://developer.uspto.gov/).

The `.env` file is gitignored. Scripts that call the USPTO API (`download_grants.py`, `download_older_grants.py`, `build_grants.py`, `build_grants_mp.py`) will exit with a clear error if `USPTO_API_KEY` is missing.

### Hardware Configuration
The system is optimized for:
- **3x RTX 4090 GPUs** with 24GB VRAM each
- **16 CPU threads** for parallel processing
- **Multi-GPU processing** with automatic load balancing

### Processing Options
- **Distributed Training**: Uses `torch.distributed` (may have NCCL issues)
- **Multiprocessing**: Uses `torch.multiprocessing` (recommended)
- **Single GPU**: Fallback to single GPU processing

## 🔧 Advanced Usage

### Custom Date Ranges
Modify date ranges in download scripts:
```python
START_DATE = datetime.date(2020, 1, 1)
END_DATE = datetime.date(2024, 12, 31)
```

### Different Embedding Models
Change the model in build scripts:
```python
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
```

### Resume Processing
The system automatically resumes from checkpoints:
```bash
# XML pipeline progress
cat checkpoints/build-progress.txt
./launch_build_bulk.sh

# PDF (1790-2002) pipeline progress
cat checkpoints/build-progress-older.txt
python3 build_older_grants.py
```

## 📈 Performance

### Processing Speed
- **XML Processing**: ~1,000 patents/minute per GPU
- **Embedding Generation**: ~2,000 embeddings/minute per GPU
- **Total Processing Time**: ~2-3 hours for 2.27M patents

### Storage Requirements
- **Raw Data**: ~1.2 TB (XML + PDF files)
- **Processed Data**: ~3.5 GB (embeddings + titles)
- **Temporary Files**: ~50 GB during processing

## 🛠️ Troubleshooting

### Common Issues

#### NCCL Peer Access Errors
```bash
# Use multiprocessing instead of distributed training
./launch_build_mp.sh
```

#### CUDA Out of Memory
```python
# Reduce batch size in build scripts
BATCH_SIZE = 32  # Reduce from 64
```

#### Download Failures
```bash
# Resume downloads (automatic retry built-in)
python3 download_grants.py
```

### GPU Monitoring
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check process status
ps aux | grep python
```

## 📝 Data Sources

- **USPTO Patent Grant XML**: https://api.uspto.gov/api/v1/datasets/products/ptgrxml
- **USPTO Patent Grant PDF**: https://api.uspto.gov/api/v1/datasets/products/ptgrmp2
- **Date Range**: 1790-2025 (comprehensive historical coverage)

## 🤝 Contributing

This is a private repository. For contributions or questions, please contact the repository owner.

## 📄 License

Private repository - All rights reserved.

## 🙏 Acknowledgments

- **USPTO** for providing comprehensive patent data
- **Sentence Transformers** for embedding models
- **PyTorch** for GPU acceleration
- **NVIDIA** for RTX 4090 GPU support

---

**Last Updated**: September 2024  
**Total Patents Processed**: 2,274,031  
**System Status**: ✅ Ready for Production Use
