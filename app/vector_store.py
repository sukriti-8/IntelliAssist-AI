from pathlib import Path
from typing import List, Dict
import json
import faiss
import numpy as np

STORAGE_DIR = Path("storage/faiss")
INDEX_PATH = STORAGE_DIR / "index.faiss"
METADATA_PATH = STORAGE_DIR / "metadata.json"

def save_metadata(chunks: List[Dict]) -> None:
    """Save chunk metadata using FAISS vector position as the list index."""

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)


def load_metadata() -> List[Dict]:
    """Load chunk metadata from disk."""

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)
def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS index using inner-product similarity."""

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype("float32"))

    return index


def save_index(index: faiss.Index) -> None:
    """Save a FAISS index to disk."""

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))


def load_index() -> faiss.Index:
    """Load a previously saved FAISS index."""

    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")

    return faiss.read_index(str(INDEX_PATH))