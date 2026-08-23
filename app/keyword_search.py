import re
from typing import List, Dict


def tokenize(text: str) -> set[str]:
    """Convert text into a simple set of lowercase word tokens."""

    return set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower(),
        )
    )


def keyword_score(query: str, text: str) -> float:
    """
    Calculate a simple keyword-overlap score.

    Score = proportion of query terms found in the text.
    """

    query_terms = tokenize(query)
    text_terms = tokenize(text)

    if not query_terms:
        return 0.0

    overlap = query_terms.intersection(text_terms)

    return len(overlap) / len(query_terms)


def keyword_search(
    query: str,
    chunks: List[Dict],
    top_k: int = 3,
) -> List[Dict]:
    """Return chunks ranked by keyword overlap."""

    results = []

    for chunk in chunks:
        result = chunk.copy()
        result["keyword_score"] = keyword_score(
            query,
            chunk["text"],
        )
        results.append(result)

    results.sort(
        key=lambda item: item["keyword_score"],
        reverse=True,
    )

    return results[:top_k]