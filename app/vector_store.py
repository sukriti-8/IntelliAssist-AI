from pathlib import Path
from typing import List, Dict
import json

import faiss
import numpy as np


STORAGE_DIR = Path("storage/faiss")


def get_document_storage_dir(document_id: str) -> Path:
    if not document_id.strip():
        raise ValueError("document_id cannot be empty.")

    return STORAGE_DIR / document_id


def get_index_path(document_id: str) -> Path:
    return get_document_storage_dir(document_id) / "index.faiss"


def get_metadata_path(document_id: str) -> Path:
    return get_document_storage_dir(document_id) / "metadata.json"


def save_metadata(
    chunks: List[Dict],
    document_id: str,
) -> None:

    storage_dir = get_document_storage_dir(document_id)
    storage_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = get_metadata_path(document_id)

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )


def load_metadata(
    document_id: str,
) -> List[Dict]:

    metadata_path = get_metadata_path(document_id)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found for document: {document_id}"
        )

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def build_index(
    embeddings: np.ndarray,
) -> faiss.Index:

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2D array."
        )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(
        embeddings.astype("float32")
    )

    return index


def save_index(
    index: faiss.Index,
    document_id: str,
) -> None:

    storage_dir = get_document_storage_dir(document_id)
    storage_dir.mkdir(parents=True, exist_ok=True)

    index_path = get_index_path(document_id)

    faiss.write_index(
        index,
        str(index_path),
    )


def load_index(
    document_id: str,
) -> faiss.Index:

    index_path = get_index_path(document_id)

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found for document: {document_id}"
        )

    return faiss.read_index(
        str(index_path)
    )


def document_index_exists(
    document_id: str,
) -> bool:

    return (
        get_index_path(document_id).exists()
        and
        get_metadata_path(document_id).exists()
    )