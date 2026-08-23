from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_pages
from app.retrieval import retrieve
from sentence_transformers import SentenceTransformer


pages = load_pdf("data/raw/test.pdf")
chunks = chunk_pages(pages)

model = SentenceTransformer("BAAI/bge-m3")

query = "What is the cost of unplanned jet engine failures?"

results = retrieve(
    query=query,
    chunks=chunks,
    model=model,
    top_k=3,
)

print(f"Query: {query}\n")

for result in results:
    print(
        f"Score: {result['score']:.4f} | "
        f"Page: {result['page_number']} | "
        f"Chunk: {result['chunk_id']}"
    )
    print(result["text"][:500])
    print("-" * 80)