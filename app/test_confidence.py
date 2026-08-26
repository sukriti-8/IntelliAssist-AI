from sentence_transformers import SentenceTransformer, CrossEncoder

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_pages
from app.retrieval import retrieve
from app.confidence import assess_evidence

import json


# ---------------------------------------------------------
# 1. Load document
# ---------------------------------------------------------

pages = load_pdf("data/raw/test.pdf")
chunks = chunk_pages(pages)

print(f"Pages loaded: {len(pages)}")
print(f"Chunks created: {len(chunks)}")


# ---------------------------------------------------------
# 2. Load models
# ---------------------------------------------------------

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "BAAI/bge-m3"
)

print("Embedding model loaded.")

print("\nLoading reranker...")

reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3"
)

print("Reranker loaded.")


# ---------------------------------------------------------
# 3. Load evaluation questions
# ---------------------------------------------------------

with open(
    "data/evaluation_questions.json",
    "r",
    encoding="utf-8"
) as f:

    evaluation_questions = json.load(f)


# ---------------------------------------------------------
# 4. Evaluate confidence
# ---------------------------------------------------------

for index, item in enumerate(
    evaluation_questions,
    start=1
):

    question = item["question"]
    expected_answerable = item["answerable"]

    print("\n" + "=" * 80)
    print(f"Question {index}: {question}")
    print(
        f"Expected answerable: "
        f"{expected_answerable}"
    )

    # -----------------------------------------------------
    # Semantic candidate retrieval
    # -----------------------------------------------------

    candidates = retrieve(
        query=question,
        chunks=chunks,
        model=embedding_model,
        top_k=min(10, len(chunks)),
    )

    # -----------------------------------------------------
    # Reranking
    # -----------------------------------------------------

    pairs = [
        [question, candidate["text"]]
        for candidate in candidates
    ]

    scores = reranker.predict(pairs)

    reranked_results = []

    for candidate, score in zip(
        candidates,
        scores
    ):

        result = candidate.copy()

        result["reranker_score"] = float(score)

        reranked_results.append(result)

    reranked_results.sort(
        key=lambda item: item["reranker_score"],
        reverse=True
    )

    # -----------------------------------------------------
    # Confidence assessment
    # -----------------------------------------------------

    assessment = assess_evidence(
        reranked_results
    )

    print(
        f"Confidence: "
        f"{assessment['confidence']:.4f}"
    )

    print(
        f"Level: "
        f"{assessment['level']}"
    )

    print(
        f"Decision: "
        f"{assessment['decision']}"
    )

    # -----------------------------------------------------
    # Top evidence
    # -----------------------------------------------------

    if reranked_results:

        top = reranked_results[0]

        print(
            f"Top evidence: "
            f"page={top['page_number']} "
            f"chunk={top['chunk_id']} "
            f"reranker="
            f"{top['reranker_score']:.4f}"
        )