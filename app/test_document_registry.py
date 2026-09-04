from app.document_registery import (
    load_registry,
    save_registry,
    find_duplicate,
    register_document,
)


def main():
    print("=" * 80)
    print("DOCUMENT REGISTRY TEST")
    print("=" * 80)

    # Start with a clean registry
    save_registry([])

    print("\nInitial registry:")
    print(load_registry())

    # Register a document
    document = register_document(
        filename="aviation.pdf",
        file_hash="FILE_HASH_123",
        content_hash="CONTENT_HASH_ABC",
    )

    print("\nRegistered document:")
    print(document)

    # Check that a document ID was generated
    document_id = document.get("document_id")

    print("\nGenerated document ID:")
    print(document_id)

    # Exact file duplicate
    exact_duplicate = find_duplicate(
        file_hash="FILE_HASH_123",
        content_hash="DIFFERENT_CONTENT",
        registry=load_registry(),
    )

    print("\nExact file duplicate test:")
    print(exact_duplicate)

    # Content duplicate
    content_duplicate = find_duplicate(
        file_hash="DIFFERENT_FILE",
        content_hash="CONTENT_HASH_ABC",
        registry=load_registry(),
    )

    print("\nContent duplicate test:")
    print(content_duplicate)

    # New document
    new_document = find_duplicate(
        file_hash="NEW_FILE_HASH",
        content_hash="NEW_CONTENT_HASH",
        registry=load_registry(),
    )

    print("\nNew document test:")
    print(new_document)

    # Validate
    if (
        document_id
        and document_id.startswith("doc_")
        and exact_duplicate is not None
        and exact_duplicate["type"] == "EXACT_FILE_DUPLICATE"
        and content_duplicate is not None
        and content_duplicate["type"] == "CONTENT_DUPLICATE"
        and new_document is None
    ):
        print("\n" + "=" * 80)
        print("RESULT: Document registry with document IDs is working.")
        print("=" * 80)
    else:
        print("\nRESULT: Document registry test failed.")


if __name__ == "__main__":
    main()