import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_pages
from app.retrieval import get_semantic_scores
from app.hybrid_search import hybrid_search


QUESTIONS_PATH = Path("data/evaluation_questions.json")
RESULTS_PATH = Path("data/hybrid_retrieval.json")


# ---------------------------------------------------------
# 1. Load evaluation questions
# ---------------------------------------------------------

with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
    questions = json.load(file)


# ---------------------------------------------------------
# 2. Load and chunk document
# ---------------------------------------------------------

pages = load_pdf("data/raw/test.pdf")
chunks = chunk_pages(pages)

print(f"Pages loaded: {len(pages)}")
print(f"Chunks created: {len(chunks)}")


# ---------------------------------------------------------
# 3. Load embedding model
# ---------------------------------------------------------

model = SentenceTransformer("BAAI/bge-m3")


# ---------------------------------------------------------
# 4. Run hybrid retrieval for every question
# ---------------------------------------------------------

results = []

for item in questions:
    query = item["question"]

    # Get semantic scores for ALL chunks
    semantic_results = get_semantic_scores(
        query=query,
        chunks=chunks,
        model=model,
    )

    # Combine semantic + keyword signals
    retrieved = hybrid_search(
        query=query,
        semantic_results=semantic_results,
        top_k=3,
        semantic_weight=0.7,
        keyword_weight=0.3,
    )

    retrieved_pages = [
        chunk["page_number"]
        for chunk in retrieved
    ]

    expected_pages = item["expected_pages"]

    # -----------------------------------------------------
    # Recall@1
    # -----------------------------------------------------

    recall_at_1 = (
        bool(expected_pages)
        and retrieved_pages[0] in expected_pages
    )

    # -----------------------------------------------------
    # Recall@3
    # -----------------------------------------------------

    recall_at_3 = (
        bool(expected_pages)
        and any(
            page in expected_pages
            for page in retrieved_pages
        )
    )

    result = {
        "id": item["id"],
        "question": query,
        "answerable": item["answerable"],
        "expected_pages": expected_pages,
        "retrieved": [
            {
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "page": chunk["page_number"],
                "semantic_score": chunk["semantic_score"],
                "keyword_score": chunk["keyword_score"],
                "hybrid_score": chunk["hybrid_score"],
            }
            for chunk in retrieved
        ],
        "recall_at_1": recall_at_1,
        "recall_at_3": recall_at_3,
    }

    results.append(result)

    # -----------------------------------------------------
    # Print question result
    # -----------------------------------------------------

    print("\n" + "=" * 80)
    print(f"Question {item['id']}: {query}")
    print(f"Expected pages: {expected_pages}")
    print(f"Recall@1: {recall_at_1}")
    print(f"Recall@3: {recall_at_3}")

    print("\nRetrieved:")

    for rank, chunk in enumerate(retrieved, start=1):
        print(
            f"{rank}. "
            f"hybrid={chunk['hybrid_score']:.4f} | "
            f"semantic={chunk['semantic_score']:.4f} | "
            f"keyword={chunk['keyword_score']:.4f} | "
            f"page={chunk['page_number']} | "
            f"chunk={chunk['chunk_id']}"
        )


# ---------------------------------------------------------
# 5. Calculate overall metrics
# ---------------------------------------------------------

answerable_results = [
    result
    for result in results
    if result["answerable"]
]

recall_at_1 = (
    sum(
        result["recall_at_1"]
        for result in answerable_results
    )
    / len(answerable_results)
)

recall_at_3 = (
    sum(
        result["recall_at_3"]
        for result in answerable_results
    )
    / len(answerable_results)
)


# ---------------------------------------------------------
# 6. Save results
# ---------------------------------------------------------

output = {
    "total_questions": len(results),
    "answerable_questions": len(answerable_results),
    "semantic_weight": 0.7,
    "keyword_weight": 0.3,
    "recall_at_1": recall_at_1,
    "recall_at_3": recall_at_3,
    "questions": results,
}

with RESULTS_PATH.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        output,
        file,
        indent=2,
        ensure_ascii=False,
    )


# ---------------------------------------------------------
# 7. Print final metrics
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("HYBRID RETRIEVAL METRICS")
print(f"Answerable questions: {len(answerable_results)}")
print(f"Recall@1: {recall_at_1:.2%}")
print(f"Recall@3: {recall_at_3:.2%}")

print("\nHybrid retrieval evaluation complete.")
print(f"Saved to: {RESULTS_PATH}")