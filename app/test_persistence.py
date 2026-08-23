import numpy as np
from sentence_transformers import SentenceTransformer

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_pages
from app.vector_store import (
    build_index,
    save_index,
    load_index,
    save_metadata,
    load_metadata,
)


# 1. Load the PDF
pages = load_pdf("data/raw/test.pdf")

# 2. Create chunks
chunks = chunk_pages(pages)

# 3. Load embedding model
model = SentenceTransformer("BAAI/bge-m3")

# 4. Create embeddings for all chunks
texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
)

embeddings = np.asarray(embeddings, dtype="float32")

# 5. Build and save FAISS index
index = build_index(embeddings)
save_index(index)

# 6. Save the metadata using the same chunk order
save_metadata(chunks)

print("Index and metadata saved.")

# 7. Load them again from disk
loaded_index = load_index()
metadata = load_metadata()

print("Index and metadata loaded.")
print("Number of stored vectors:", loaded_index.ntotal)
print("Number of metadata records:", len(metadata))

# 8. Create a query embedding
query = "What is the cost of unplanned jet engine failures?"

query_embedding = model.encode(
    [query],
    normalize_embeddings=True,
)

query_embedding = np.asarray(query_embedding, dtype="float32")

# 9. Search FAISS
scores, indices = loaded_index.search(query_embedding, 3)

# 10. Map FAISS positions back to document metadata
print(f"\nQuery: {query}\n")

for score, position in zip(scores[0], indices[0]):
    chunk = metadata[position]

    print(
        f"Score: {score:.4f} | "
        f"Source: {chunk['source']} | "
        f"Page: {chunk['page_number']} | "
        f"Chunk: {chunk['chunk_id']}"
    )

    print(chunk["text"][:300])
    print("-" * 60)