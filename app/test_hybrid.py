from sentence_transformers import SentenceTransformer

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_pages
from app.retrieval import get_semantic_scores
from app.hybrid_search import hybrid_search


pages = load_pdf("data/raw/test.pdf")
chunks = chunk_pages(pages)

model = SentenceTransformer("BAAI/bge-m3")

query = "What sensors are mentioned in the project overview?"

semantic_results = get_semantic_scores(
    query=query,
    chunks=chunks,
    model=model,
)

results = hybrid_search(
    query=query,
    semantic_results=semantic_results,
    top_k=3,
)

print(f"Query: {query}\n")

for rank, result in enumerate(results, start=1):
    print(
        f"{rank}. "
        f"Hybrid: {result['hybrid_score']:.4f} | "
        f"Semantic: {result['semantic_score']:.4f} | "
        f"Keyword: {result['keyword_score']:.4f} | "
        f"Page: {result['page_number']} | "
        f"Chunk: {result['chunk_id']}"
    )

    print(result["text"][:500])
    print("-" * 80)