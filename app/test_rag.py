import numpy as np
from sentence_transformers import SentenceTransformer

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_pages
from app.vector_store import build_index
from app.rag import generate_answer


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

query = "What is the capital of France?"

query_embedding = model.encode(
    [query],
    normalize_embeddings=True,
)

query_embedding = np.asarray(query_embedding, dtype="float32")

scores, indices = index.search(query_embedding, 3)

retrieved_chunks = []

for score, position in zip(scores[0], indices[0]):
    chunk = chunks[position]

    retrieved_chunks.append(
        f"""
Source: {chunk['source']}
Page: {chunk['page_number']}
Chunk: {chunk['chunk_id']}
Similarity: {score:.4f}

{chunk['text']}
"""
    )

context = "\n".join(retrieved_chunks)

answer = generate_answer(
    query=query,
    context=context,
)

print("\nQUESTION:")
print(query)

print("\nRETRIEVED CONTEXT:")
print(context)

print("\nANSWER:")
print(answer)