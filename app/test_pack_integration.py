from app.ingestion.pdf_loader import load_pdf
from app.duplicate_detector import (
    calculate_file_hash,
    calculate_content_hash,
)
from app.document_registery import (
    register_document,
    load_registry,
)
from app.pack_registry import (
    create_pack,
    find_pack,
    add_document_to_pack,
)


PDF_PATH = "data/raw/test.pdf"


def main():
    print("=" * 80)
    print("DOCUMENT + PACK INTEGRATION TEST")
    print("=" * 80)

    # 1. Extract the real PDF
    pages = load_pdf(PDF_PATH)
    full_text = "\n".join(page["text"] for page in pages)

    print(f"\nPDF: {PDF_PATH}")
    print(f"Pages extracted: {len(pages)}")
    print(f"Characters extracted: {len(full_text)}")

    
    # 2. Calculate hashes
    file_hash = calculate_file_hash(PDF_PATH)
    content_hash = calculate_content_hash(full_text)

    print(f"\nFile hash: {file_hash}")
    print(f"Content hash: {content_hash}")

  
    # 3. Register the real document
    document = register_document(
        filename="test.pdf",
        file_hash=file_hash,
        content_hash=content_hash,
    )

    document_id = document["document_id"]

    print("\nRegistered document:")
    print(document)

  
    # 4. Create a real pack
    pack = create_pack(
        pack_id="pack_001",
        name="Aviation Analytics Study Pack",
        description="Study materials for the aviation analytics project.",
    )

    print("\nCreated pack:")
    print(pack)

   
    # 5. Add the real document to the pack
    updated_pack = add_document_to_pack(
        pack_id="pack_001",
        document_id=document_id,
    )

    print("\nPack after adding document:")
    print(updated_pack)

    # 6. Verify everything
    saved_document = next(
        (
            item
            for item in load_registry()
            if item.get("document_id") == document_id
        ),
        None,
    )

    saved_pack = find_pack("pack_001")

    print("\nSaved document:")
    print(saved_document)

    print("\nSaved pack:")
    print(saved_pack)

    if (
        saved_document is not None
        and saved_document["filename"] == "test.pdf"
        and saved_pack is not None
        and document_id in saved_pack["documents"]
    ):
        print("\n" + "=" * 80)
        print("RESULT: Document-to-pack integration is working.")
        print("=" * 80)
    else:
        print("\nRESULT: Integration test failed.")


if __name__ == "__main__":
    main()