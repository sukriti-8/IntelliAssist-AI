from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def get_semantic_scores(
    query: str,
    chunks: List[Dict],
    model: SentenceTransformer,
) -> List[Dict]:
    """
    Calculate semantic similarity between the query and every chunk.

    Returns all chunks with their semantic scores.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if not chunks:
        return []

    chunk_texts = [chunk["text"] for chunk in chunks]

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )

    chunk_embeddings = model.encode(
        chunk_texts,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    chunk_embeddings = np.asarray(
        chunk_embeddings,
        dtype="float32",
    )

    scores = cosine_similarity(
        query_embedding,
        chunk_embeddings,
    )[0]

    results = []

    for chunk, score in zip(chunks, scores):
        result = chunk.copy()
        result["semantic_score"] = float(score)
        results.append(result)

    return results


def retrieve(
    query: str,
    chunks: List[Dict],
    model: SentenceTransformer,
    top_k: int = 3,
) -> List[Dict]:
    """
    Retrieve the top-k chunks using semantic similarity.
    """

    results = get_semantic_scores(
        query=query,
        chunks=chunks,
        model=model,
    )

    results.sort(
        key=lambda item: item["semantic_score"],
        reverse=True,
    )

    for result in results:
        result["score"] = result["semantic_score"]

    return results[:top_k]