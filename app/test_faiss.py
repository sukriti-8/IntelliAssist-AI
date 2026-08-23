import numpy as np
from sentence_transformers import SentenceTransformer

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_pages
from app.vector_store import build_index


pages = load_pdf("data/raw/test.pdf")
chunks = chunk_pages(pages)

model = SentenceTransformer("BAAI/bge-m3")

texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
)

embeddings = np.asarray(embeddings, dtype="float32")

index = build_index(embeddings)

query = "What is the cost of unplanned jet engine failures?"

query_embedding = model.encode(
    [query],
    normalize_embeddings=True,
)

query_embedding = np.asarray(query_embedding, dtype="float32")

scores, indices = index.search(query_embedding, 3)

print(f"Query: {query}\n")

for score, index_position in zip(scores[0], indices[0]):
    chunk = chunks[index_position]

    print(
        f"Score: {score:.4f} | "
        f"Page: {chunk['page_number']} | "
        f"Chunk: {chunk['chunk_id']}"
    )
    print(chunk["text"][:500])
    print("-" * 80)