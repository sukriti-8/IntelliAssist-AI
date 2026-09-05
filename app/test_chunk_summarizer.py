from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_pages
from app.summarizer import summarize_chunk


PDF_PATH = "data/raw/test.pdf"


def main():
    print("=" * 80)
    print("CHUNK SUMMARIZATION TEST")
    print("=" * 80)

    pages = load_pdf(PDF_PATH)
    chunks = chunk_pages(pages)

    print(f"\nTotal chunks: {len(chunks)}")

    first_chunk = chunks[0]

    print(f"Page: {first_chunk['page_number']}")
    print(f"Chunk index: {first_chunk.get('chunk_index', 0)}")
    print(f"Characters: {len(first_chunk['text'])}")

    print("\nOriginal chunk:")
    print("-" * 80)
    print(first_chunk["text"])
    print("-" * 80)

    print("\nGenerating chunk summary...\n")

    summary = summarize_chunk(first_chunk["text"])

    print("Chunk summary:")
    print("-" * 80)
    print(summary)
    print("-" * 80)

    print("\nRESULT: Chunk summarization completed.")


if __name__ == "__main__":
    main()