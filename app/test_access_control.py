from app.access_control import can_access_document


def main():
    print("=" * 80)
    print("ACCESS CONTROL TEST")
    print("=" * 80)

    private_document = {
        "document_id": "doc_001",
        "filename": "private.pdf",
        "owner_id": "user_001",
        "access": "private",
        "shared_with": [],
    }

    shared_document = {
        "document_id": "doc_002",
        "filename": "shared.pdf",
        "owner_id": "user_001",
        "access": "shared",
        "shared_with": ["user_002"],
    }

    # Private document
    owner_private = can_access_document(
        private_document,
        "user_001",
    )

    other_private = can_access_document(
        private_document,
        "user_002",
    )

    print("\nPrivate document:")
    print(f"Owner access: {owner_private}")
    print(f"Other user access: {other_private}")

    assert owner_private is True
    assert other_private is False

    # Shared document
    owner_shared = can_access_document(
        shared_document,
        "user_001",
    )

    shared_user = can_access_document(
        shared_document,
        "user_002",
    )

    unrelated_user = can_access_document(
        shared_document,
        "user_003",
    )

    print("\nShared document:")
    print(f"Owner access: {owner_shared}")
    print(f"Shared user access: {shared_user}")
    print(f"Unrelated user access: {unrelated_user}")

    assert owner_shared is True
    assert shared_user is True
    assert unrelated_user is False

    # Empty user ID
    empty_user = can_access_document(
        private_document,
        "",
    )

    print(f"\nEmpty user ID access: {empty_user}")

    assert empty_user is False

    print("\n" + "=" * 80)
    print("RESULT: Access control tests passed.")
    print("=" * 80)


if __name__ == "__main__":
    main()