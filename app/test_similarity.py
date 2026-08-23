from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_pages
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


pages = load_pdf("data/raw/test.pdf")
chunks = chunk_pages(pages)

model = SentenceTransformer("BAAI/bge-m3")

text_a = chunks[0]["text"]
text_b = chunks[1]["text"]

embedding_a = model.encode([text_a])
embedding_b = model.encode([text_b])

score = cosine_similarity(embedding_a, embedding_b)[0][0]

print("Chunk A:")
print(text_a[:300])

print("\nChunk B:")
print(text_b[:300])

print("\nCosine similarity:", score)