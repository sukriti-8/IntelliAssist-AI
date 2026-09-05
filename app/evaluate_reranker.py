from sentence_transformers import SentenceTransformer, CrossEncoder

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_pages
from app.retrieval import retrieve

import json


# =========================================================
# 1. Load document
# =========================================================

pages = load_pdf("data/raw/test.pdf")
chunks = chunk_pages(pages)

print(f"Pages loaded: {len(pages)}")
print(f"Chunks created: {len(chunks)}")


# =========================================================
# 2. Load embedding model
# =========================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "BAAI/bge-m3"
)

print("Embedding model loaded.")


# =========================================================
# 3. Load reranker
# =========================================================

print("\nLoading reranker...")

reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3"
)

print("Reranker loaded.")


# =========================================================
# 4. Load evaluation questions
# =========================================================

with open(
    "data/evaluation_questions.json",
    "r",
    encoding="utf-8"
) as f:
    evaluation_questions = json.load(f)


# =========================================================
# 5. Evaluate reranker
# =========================================================

answerable_questions = 0
recall_at_1 = 0
recall_at_3 = 0


for index, item in enumerate(evaluation_questions, start=1):

    question = item["question"]
    expected_answerable = item["answerable"]
    expected_pages = item.get("expected_pages", [])

    print("\n" + "=" * 80)
    print(f"Question {index}: {question}")
    print(f"Expected answerable: {expected_answerable}")
    print(f"Expected pages: {expected_pages}")


    # Semantic retrieval
    # Retrieve a larger candidate pool before reranking.
    # This allows the reranker to choose the best result
    # from more candidates.

    candidate_k = min(10, len(chunks))

    candidates = retrieve(
        query=question,
        chunks=chunks,
        model=embedding_model,
        top_k=candidate_k,
    )


  
    # Rerank semantic candidates
    pairs = [
        [question, candidate["text"]]
        for candidate in candidates
    ]

    reranker_scores = reranker.predict(pairs)


    
    # Attach reranker scores
    reranked_results = []

    for candidate, score in zip(
        candidates,
        reranker_scores
    ):

        result = candidate.copy()

        result["reranker_score"] = float(score)

        reranked_results.append(result)


    # Sort by reranker score
    reranked_results.sort(
        key=lambda item: item["reranker_score"],
        reverse=True
    )


   
    # Recall evaluation
    if expected_answerable:

        answerable_questions += 1

        top_1_pages = {
            result["page_number"]
            for result in reranked_results[:1]
        }

        top_3_pages = {
            result["page_number"]
            for result in reranked_results[:3]
        }

        hit_at_1 = any(
            page in expected_pages
            for page in top_1_pages
        )

        hit_at_3 = any(
            page in expected_pages
            for page in top_3_pages
        )

        if hit_at_1:
            recall_at_1 += 1

        if hit_at_3:
            recall_at_3 += 1

        print(f"Recall@1: {hit_at_1}")
        print(f"Recall@3: {hit_at_3}")

    else:

        print("Unanswerable question")


    
    # Display top reranked results
    print("\nTop reranked results:")

    for rank, result in enumerate(
        reranked_results[:3],
        start=1
    ):

        print(
            f"{rank}. "
            f"reranker={result['reranker_score']:.4f} | "
            f"semantic={result['semantic_score']:.4f} | "
            f"page={result['page_number']} | "
            f"chunk={result['chunk_id']}"
        )

# 6. Final metrics
print("\n" + "=" * 80)
print("RERANKER RETRIEVAL METRICS")

if answerable_questions > 0:

    recall1_percentage = (
        recall_at_1 / answerable_questions
    ) * 100

    recall3_percentage = (
        recall_at_3 / answerable_questions
    ) * 100

    print(
        f"Answerable questions: "
        f"{answerable_questions}"
    )

    print(
        f"Recall@1: "
        f"{recall1_percentage:.2f}%"
    )

    print(
        f"Recall@3: "
        f"{recall3_percentage:.2f}%"
    )

else:

    print("No answerable questions found.")


print("\nReranker evaluation complete.")