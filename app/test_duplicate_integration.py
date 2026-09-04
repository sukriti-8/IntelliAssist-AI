from app.ingestion.pdf_loader import load_pdf
from app.duplicate_detector import (
    calculate_file_hash,
    calculate_content_hash,
)
from app.document_registery import (
    load_registry,
    find_duplicate,
    register_document,
)


PDF_PATH = "data/raw/test.pdf"


def main():
    print("Loading PDF...")

    pages = load_pdf(PDF_PATH)

    full_text = "\n".join(page["text"] for page in pages)

    print(f"Pages extracted: {len(pages)}")
    print(f"Characters extracted: {len(full_text)}")

    file_hash = calculate_file_hash(PDF_PATH)
    content_hash = calculate_content_hash(full_text)

    print(f"\nFile hash: {file_hash}")
    print(f"Content hash: {content_hash}")

    registry = load_registry()

    duplicate = find_duplicate(
        file_hash=file_hash,
        content_hash=content_hash,
        registry=registry,
    )

    if duplicate:
        print("\nDUPLICATE DETECTED")
        print(f"Type: {duplicate['type']}")
        print(f"Existing file: {duplicate['document']['filename']}")
    else:
        print("\nNEW DOCUMENT")
        register_document(
            filename=PDF_PATH.split("/")[-1],
            file_hash=file_hash,
            content_hash=content_hash,
        )
        print("Document registered successfully.")


if __name__ == "__main__":
    main()