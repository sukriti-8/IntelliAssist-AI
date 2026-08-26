from sentence_transformers import CrossEncoder

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_pages


# ---------------------------------------------------------
# 1. Load document
# ---------------------------------------------------------

pages = load_pdf("data/raw/test.pdf")
chunks = chunk_pages(pages)

print(f"Pages loaded: {len(pages)}")
print(f"Chunks created: {len(chunks)}")


# ---------------------------------------------------------
# 2. Load reranker
# ---------------------------------------------------------

print("\nLoading reranker...")

model = CrossEncoder(
    "BAAI/bge-reranker-v2-m3"
)

print("Reranker loaded.")


# ---------------------------------------------------------
# 3. Test query
# ---------------------------------------------------------

query = "What preprocessing techniques are proposed for the sensor data?"


# ---------------------------------------------------------
# 4. Create query-document pairs
# ---------------------------------------------------------

pairs = [
    [query, chunk["text"]]
    for chunk in chunks
]


# ---------------------------------------------------------
# 5. Calculate reranking scores
# ---------------------------------------------------------

scores = model.predict(pairs)


# ---------------------------------------------------------
# 6. Attach scores
# ---------------------------------------------------------

results = []

for chunk, score in zip(chunks, scores):

    result = chunk.copy()
    result["reranker_score"] = float(score)

    results.append(result)


# ---------------------------------------------------------
# 7. Sort by reranker score
# ---------------------------------------------------------

results.sort(
    key=lambda item: item["reranker_score"],
    reverse=True
)


# ---------------------------------------------------------
# 8. Display top results
# ---------------------------------------------------------

print("\nQuery:")
print(query)

print("\nTop reranked results:\n")

for rank, result in enumerate(results[:5], start=1):

    print(
        f"{rank}. "
        f"Score: {result['reranker_score']:.4f} | "
        f"Page: {result['page_number']} | "
        f"Chunk: {result['chunk_id']}"
    )

    print(result["text"][:700])
    print("-" * 80)