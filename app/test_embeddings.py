from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_pages
from sentence_transformers import SentenceTransformer


pages = load_pdf("data/raw/test.pdf")
chunks = chunk_pages(pages)

model = SentenceTransformer("BAAI/bge-m3")

text = chunks[0]["text"]
embedding = model.encode(text)

print("Chunk text:")
print(text[:500])

print("\nEmbedding type:", type(embedding))
print("Embedding dimensions:", len(embedding))

print("\nFirst 10 values:")
print(embedding[:10])