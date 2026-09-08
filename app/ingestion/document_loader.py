from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def load_document(file_path: str) -> list[dict]:
    """
    Extract text from PDF, TXT, or DOCX.

    Returns a list of page/section-like dictionaries so the existing
    chunking and retrieval pipeline can remain unchanged.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if extension == ".pdf":
        return _load_pdf(path)

    if extension == ".docx":
        return _load_docx(path)

    if extension == ".txt":
        return _load_txt(path)

    raise ValueError(f"Unsupported file type: {extension}")


def _load_pdf(path: Path) -> list[dict]:
    """Extract text page-by-page from a PDF."""

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


def _load_docx(path: Path) -> list[dict]:
    """Extract non-empty paragraphs from a DOCX."""

    document = DocxDocument(str(path))

    paragraphs = []

    for paragraph_number, paragraph in enumerate(
        document.paragraphs,
        start=1,
    ):
        text = paragraph.text.strip()

        if not text:
            continue

        paragraphs.append(
            {
                "source": path.name,
                "page_number": paragraph_number,
                "text": text,
            }
        )

    return paragraphs


def _load_txt(path: Path) -> list[dict]:
    """Extract non-empty lines from a TXT file."""

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    lines = text.splitlines()

    sections = []

    for line_number, line in enumerate(lines, start=1):
        line = line.strip()

        if not line:
            continue

        sections.append(
            {
                "source": path.name,
                "page_number": line_number,
                "text": line,
            }
        )

    return sections