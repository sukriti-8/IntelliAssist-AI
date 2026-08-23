from typing import List, Dict

from app.keyword_search import keyword_score


def hybrid_search(
    query: str,
    semantic_results: List[Dict],
    top_k: int = 3,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
) -> List[Dict]:
    """
    Combine semantic and keyword scores for every chunk.
    """

    if not semantic_results:
        return []

    if abs((semantic_weight + keyword_weight) - 1.0) > 1e-9:
        raise ValueError(
            "semantic_weight + keyword_weight must equal 1.0."
        )

    results = []

    for chunk in semantic_results:
        keyword = keyword_score(
            query,
            chunk["text"],
        )

        semantic = chunk["semantic_score"]

        combined_score = (
            semantic_weight * semantic
            + keyword_weight * keyword
        )

        result = chunk.copy()

        result["keyword_score"] = keyword
        result["hybrid_score"] = combined_score

        results.append(result)

    results.sort(
        key=lambda item: item["hybrid_score"],
        reverse=True,
    )

    return results[:top_k]