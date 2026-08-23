from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_pages
from app.keyword_search import keyword_search


pages = load_pdf("data/raw/test.pdf")
chunks = chunk_pages(pages)

query = "What sensors are mentioned in the project overview?"

results = keyword_search(
    query=query,
    chunks=chunks,
    top_k=3,
)

print(f"Query: {query}\n")

for result in results:
    print(
        f"Keyword score: {result['keyword_score']:.4f} | "
        f"Page: {result['page_number']} | "
        f"Chunk: {result['chunk_id']}"
    )
    print(result["text"][:500])
    print("-" * 80)