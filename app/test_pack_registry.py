from app.pack_registry import (
    load_packs,
    save_packs,
    create_pack,
    find_pack,
    add_document_to_pack,
    remove_document_from_pack,
)


def main():
    print("=" * 80)
    print("PACK REGISTRY TEST")
    print("=" * 80)

    # Start with a clean registry
    save_packs([])

    print("\nInitial packs:")
    print(load_packs())

    # Create a pack
    pack = create_pack(
        pack_id="pack_001",
        name="Machine Learning",
        description="Machine Learning study materials",
    )

    print("\nCreated pack:")
    print(pack)

    # Find the pack
    found_pack = find_pack("pack_001")

    print("\nFind existing pack:")
    print(found_pack)

    # Try to find a non-existing pack
    missing_pack = find_pack("pack_999")

    print("\nFind non-existing pack:")
    print(missing_pack)
        # Add documents to the pack
    updated_pack = add_document_to_pack(
        "pack_001",
        "document_001",
    )

    print("\nAfter adding document_001:")
    print(updated_pack)

    # Try adding the same document again
    updated_pack = add_document_to_pack(
        "pack_001",
        "document_001",
    )

    print("\nAfter adding document_001 again:")
    print(updated_pack)

    # Add another document
    updated_pack = add_document_to_pack(
        "pack_001",
        "document_002",
    )

    print("\nAfter adding document_002:")
    print(updated_pack)

    # Remove document_001
    updated_pack = remove_document_from_pack(
        "pack_001",
        "document_001",
    )

    print("\nAfter removing document_001:")
    print(updated_pack)

    # Validate
    if (
        pack["pack_id"] == "pack_001"
        and pack["name"] == "Machine Learning"
        and found_pack is not None
        and missing_pack is None
        and updated_pack["documents"] == ["document_002"]
    ):
        print("\n" + "=" * 80)
        print("RESULT: Pack registry is working.")
        print("=" * 80)
    else:
        print("\nRESULT: Pack registry test failed.")


if __name__ == "__main__":
    main()