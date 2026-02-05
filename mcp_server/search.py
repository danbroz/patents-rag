import os
import threading
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None


@dataclass(frozen=True)
class SearchConfig:
    faiss_index_path: str
    titles_path: str
    model_name: str


_lock = threading.Lock()
_loaded = False
_index = None
_titles: np.ndarray | None = None
_model: SentenceTransformer | None = None


def get_config() -> SearchConfig:
    return SearchConfig(
        faiss_index_path=os.environ.get("FAISS_INDEX_PATH", "patents-embeddings/index.faiss"),
        titles_path=os.environ.get("TITLES_PATH", "patents-index/patent_titles.npy"),
        model_name=os.environ.get("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
    )


def _ensure_loaded() -> None:
    global _loaded, _index, _titles, _model
    if _loaded:
        return
    if faiss is None:
        raise RuntimeError("faiss import failed; install faiss-cpu or faiss-gpu")

    cfg = get_config()
    if not os.path.exists(cfg.faiss_index_path):
        raise FileNotFoundError(
            f"FAISS index not found at {cfg.faiss_index_path}. Build it first with: python3 build.py"
        )
    if not os.path.exists(cfg.titles_path):
        raise FileNotFoundError(f"Titles file not found at {cfg.titles_path}")

    _index = faiss.read_index(cfg.faiss_index_path)
    _titles = np.load(cfg.titles_path, allow_pickle=True)
    _model = SentenceTransformer(cfg.model_name)
    _loaded = True


def search(query: str, k: int = 10) -> list[dict]:
    if not query or not query.strip():
        return []
    if k <= 0:
        return []
    with _lock:
        _ensure_loaded()
        assert _index is not None and _titles is not None and _model is not None

        q = _model.encode([query]).astype("float32")
        scores, ids = _index.search(q, k)

        out: list[dict] = []
        for rank, (idx, score) in enumerate(zip(ids[0].tolist(), scores[0].tolist()), start=1):
            if idx < 0 or idx >= len(_titles):
                continue
            out.append(
                {
                    "rank": rank,
                    "id": int(idx),
                    "title": str(_titles[idx]),
                    "score": float(score),
                }
            )
        return out

