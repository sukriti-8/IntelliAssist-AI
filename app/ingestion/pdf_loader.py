from pathlib import Path

from pypdf import PdfReader


def load_pdf(file_path: str) -> list[dict]:
    """Extract text from each PDF page while preserving page metadata."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Expected a PDF file.")

    reader = PdfReader(str(path))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        pages.append(
            {
                "source": path.name,
                "page_number": page_number,
                "text": text.strip(),
            }
        )

    return pages


if __name__ == "__main__":
    pdf_path = "data/raw/test.pdf"

    pages = load_pdf(pdf_path)

    print(f"Extracted {len(pages)} pages.")

    for page in pages[:2]:
        print(f"\n--- {page['source']} | Page {page['page_number']} ---")
        print(page["text"][:1000])