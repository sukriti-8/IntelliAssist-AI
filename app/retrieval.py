from typing import List, Dict

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def retrieve(
    query: str,
    chunks: List[Dict],
    model: SentenceTransformer,
    top_k: int = 3,
) -> List[Dict]:
    """
    Retrieve the top-k chunks most semantically similar to the query.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if not chunks:
        return []

    chunk_texts = [chunk["text"] for chunk in chunks]

    query_embedding = model.encode([query])
    chunk_embeddings = model.encode(chunk_texts)

    scores = cosine_similarity(query_embedding, chunk_embeddings)[0]

    results = []

    for chunk, score in zip(chunks, scores):
        result = chunk.copy()
        result["score"] = float(score)
        results.append(result)

    results.sort(key=lambda item: item["score"], reverse=True)

    return results[:top_k]