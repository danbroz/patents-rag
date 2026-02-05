#!/usr/bin/env python3
"""
Merge partial chunk directories into the final RAG index.
Order: partial-patents-older/ (1790-2002) first, then partial-patents/ (2002+).
Produces patents-index/patent_titles.npy and patents-embeddings/embeddings.memmap.
"""

import os
import re
import logging
import numpy as np

PARTIAL_OLDER_DIR = "partial-patents-older"
PARTIAL_XML_DIR = "partial-patents"
FINAL_INDEX_DIR = "patents-index"
FINAL_EMB_DIR = "patents-embeddings"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


def chunk_number(path):
    """Extract numeric chunk index from path for sorting."""
    basename = os.path.basename(path)
    m = re.search(r"embeddings_chunk_(\d+)\.npy", basename)
    return int(m.group(1)) if m else -1


def list_chunk_pairs(partial_dir):
    """Return sorted list of (emb_file, titles_file) for a partial dir. Empty if dir missing or no chunks."""
    if not os.path.isdir(partial_dir):
        return []
    emb_files = [
        os.path.join(partial_dir, f)
        for f in os.listdir(partial_dir)
        if f.startswith("embeddings_chunk_") and f.endswith(".npy")
    ]
    emb_files.sort(key=chunk_number)
    pairs = []
    for emb_file in emb_files:
        base = emb_file.replace("embeddings_chunk_", "titles_chunk_").replace(".npy", ".npy")
        # titles_chunk_N.npy
        titles_file = os.path.join(partial_dir, os.path.basename(base))
        if os.path.isfile(titles_file):
            pairs.append((emb_file, titles_file))
        else:
            logging.warning(f"Missing titles file for {emb_file}, skipping")
    return pairs


def main():
    os.makedirs(FINAL_INDEX_DIR, exist_ok=True)
    os.makedirs(FINAL_EMB_DIR, exist_ok=True)

    older_pairs = list_chunk_pairs(PARTIAL_OLDER_DIR)
    xml_pairs = list_chunk_pairs(PARTIAL_XML_DIR)

    if not older_pairs and not xml_pairs:
        logging.error("No chunk files found in %s or %s", PARTIAL_OLDER_DIR, PARTIAL_XML_DIR)
        return

    merged_titles = []
    all_emb_files = []
    emb_dim = None

    for partial_dir, pairs in [(PARTIAL_OLDER_DIR, older_pairs), (PARTIAL_XML_DIR, xml_pairs)]:
        if not pairs:
            continue
        logging.info(f"Merging {len(pairs)} chunks from {partial_dir}")
        for emb_file, titles_file in pairs:
            titles_arr = np.load(titles_file, allow_pickle=True)
            merged_titles.extend(titles_arr.tolist())
            emb_arr = np.load(emb_file)
            if emb_dim is None:
                emb_dim = emb_arr.shape[1]
            elif emb_arr.shape[1] != emb_dim:
                logging.error(f"Embedding dimension mismatch: {emb_file} has {emb_arr.shape[1]}, expected {emb_dim}")
                return
            all_emb_files.append(emb_file)

    total_rows = sum(np.load(f).shape[0] for f in all_emb_files)
    if total_rows == 0:
        logging.error("No embeddings to merge")
        return

    final_titles_path = os.path.join(FINAL_INDEX_DIR, "patent_titles.npy")
    np.save(final_titles_path, np.array(merged_titles, dtype=object))
    logging.info(f"Merged {len(merged_titles)} patent titles into {final_titles_path}")

    final_emb_path = os.path.join(FINAL_EMB_DIR, "embeddings.memmap")
    mm = np.memmap(final_emb_path, dtype=np.float32, mode="w+", shape=(total_rows, emb_dim))
    current_row = 0
    for emb_file in all_emb_files:
        arr = np.load(emb_file)
        n = arr.shape[0]
        mm[current_row : current_row + n] = arr
        current_row += n
    mm.flush()
    del mm
    logging.info(f"Merged embeddings into {final_emb_path} (shape {total_rows} x {emb_dim})")
    logging.info("Merge complete.")


if __name__ == "__main__":
    main()
