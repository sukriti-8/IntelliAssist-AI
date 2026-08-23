from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_pages


pages = load_pdf("data/raw/test.pdf")
chunks = chunk_pages(pages)

print("Pages:", len(pages))
print("Chunks:", len(chunks))

for chunk in chunks:
    print(
        f"\n--- Chunk {chunk['chunk_id']} | "
        f"Page {chunk['page_number']} | "
        f"{len(chunk['text'])} chars ---"
    )
    print(chunk["text"][:500])