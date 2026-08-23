import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_pages
from app.vector_store import build_index


QUESTIONS_PATH = Path("data/evaluation_questions.json")
RETRIEVAL_RESULTS_PATH = Path("data/baseline_retrieval.json")


# ---------------------------------------------------------
# 1. Load evaluation questions
# ---------------------------------------------------------

with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
    questions = json.load(file)


# ---------------------------------------------------------
# 2. Load and chunk the document
# ---------------------------------------------------------

pages = load_pdf("data/raw/test.pdf")
chunks = chunk_pages(pages)

print(f"Pages loaded: {len(pages)}")
print(f"Chunks created: {len(chunks)}")


# ---------------------------------------------------------
# 3. Load the embedding model
# ---------------------------------------------------------

model = SentenceTransformer("BAAI/bge-m3")


# ---------------------------------------------------------
# 4. Create embeddings for all chunks ONCE
# ---------------------------------------------------------

texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True,
)

embeddings = np.asarray(
    embeddings,
    dtype="float32",
)


# ---------------------------------------------------------
# 5. Build FAISS index
# ---------------------------------------------------------

index = build_index(embeddings)

print(f"Vectors indexed: {index.ntotal}")


# ---------------------------------------------------------
# 6. Evaluate retrieval for every question
# ---------------------------------------------------------

results = []

for item in questions:
    query = item["question"]

    # Create query embedding
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    # Search top 3 chunks
    scores, indices = index.search(
        query_embedding,
        3,
    )

    retrieved = []

    for score, position in zip(scores[0], indices[0]):
        position = int(position)
        chunk = chunks[position]

        retrieved.append(
            {
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "page": chunk["page_number"],
                "score": float(score),
                "text": chunk["text"],
            }
        )

    # -----------------------------------------------------
    # Calculate Recall@1 and Recall@3
    # -----------------------------------------------------

    retrieved_pages = [
        chunk["page"]
        for chunk in retrieved
    ]

    expected_pages = item["expected_pages"]

    # Recall@1:
    # Was at least one expected page returned as the top result?
    recall_at_1 = (
        bool(expected_pages)
        and retrieved_pages[0] in expected_pages
    )

    # Recall@3:
    # Did any of the top 3 results come from an expected page?
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
        "retrieved": retrieved,
        "recall_at_1": recall_at_1,
        "recall_at_3": recall_at_3,
    }

    results.append(result)

    # -----------------------------------------------------
    # Print results for this question
    # -----------------------------------------------------

    print("\n" + "=" * 80)
    print(f"Question {item['id']}: {query}")
    print(f"Expected answerable: {item['answerable']}")
    print(f"Expected pages: {expected_pages}")
    print(f"Recall@1: {recall_at_1}")
    print(f"Recall@3: {recall_at_3}")

    print("\nRetrieved:")

    for rank, chunk in enumerate(retrieved, start=1):
        print(
            f"{rank}. "
            f"score={chunk['score']:.4f} "
            f"page={chunk['page']} "
            f"chunk={chunk['chunk_id']}"
        )


# ---------------------------------------------------------
# 7. Calculate overall baseline metrics
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
# 8. Save results
# ---------------------------------------------------------

output = {
    "total_questions": len(results),
    "answerable_questions": len(answerable_results),
    "recall_at_1": recall_at_1,
    "recall_at_3": recall_at_3,
    "questions": results,
}

with RETRIEVAL_RESULTS_PATH.open(
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
# 9. Print final baseline metrics
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("BASELINE RETRIEVAL METRICS")
print(f"Answerable questions: {len(answerable_results)}")
print(f"Recall@1: {recall_at_1:.2%}")
print(f"Recall@3: {recall_at_3:.2%}")

print("\nRetrieval baseline complete.")
print(f"Saved to: {RETRIEVAL_RESULTS_PATH}")