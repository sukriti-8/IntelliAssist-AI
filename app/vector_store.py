import faiss
import numpy as np


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS index using inner-product similarity."""

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype("float32"))

    return index