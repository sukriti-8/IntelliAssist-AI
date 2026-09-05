from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_pages
from app.summarizer import hierarchical_summarize


PDF_PATH = "data/raw/test.pdf"


def main():
    print("=" * 80)
    print("HIERARCHICAL SUMMARIZATION TEST")
    print("=" * 80)

    pages = load_pdf(PDF_PATH)
    chunks = chunk_pages(pages)

    # Use only 2 chunks for this test.
    test_chunks = chunks[:2]

    print(f"\nTotal document chunks: {len(chunks)}")
    print(f"Chunks used for test: {len(test_chunks)}")

    print("\nGenerating hierarchical summary...\n")

    summary = hierarchical_summarize(
        test_chunks,
        batch_size=4,
        delay_seconds=12.0,
    )

    print("\n" + "-" * 80)
    print("FINAL SUMMARY")
    print("-" * 80)
    print(summary)
    print("-" * 80)

    print("\nRESULT: Hierarchical summarization completed.")


if __name__ == "__main__":
    main()