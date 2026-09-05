from app.ingestion.pdf_loader import load_pdf
from app.summarizer import summarize_text


PDF_PATH = "data/raw/test.pdf"


def main():
    print("=" * 80)
    print("DOCUMENT SUMMARIZATION TEST")
    print("=" * 80)

    pages = load_pdf(PDF_PATH)
    full_text = "\n".join(page["text"] for page in pages)

    print(f"\nPDF: {PDF_PATH}")
    print(f"Pages: {len(pages)}")
    print(f"Characters: {len(full_text)}")

    print("\nGenerating summary...\n")

    summary = summarize_text(full_text)

    print("-" * 80)
    print(summary)
    print("-" * 80)

    print("\nRESULT: Summarization completed.")


if __name__ == "__main__":
    main()