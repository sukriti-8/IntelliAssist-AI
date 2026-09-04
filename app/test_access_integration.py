from app.document_registery import register_document
from app.access_control import can_access_document


def main():
    print("=" * 80)
    print("DOCUMENT + ACCESS CONTROL INTEGRATION TEST")
    print("=" * 80)

    # Register a real document through the document registry.
    document = register_document(
        filename="access_test.pdf",
        file_hash="ACCESS_TEST_FILE_HASH",
        content_hash="ACCESS_TEST_CONTENT_HASH",
        owner_id="user_001",
        access="shared",
        shared_with=["user_002"],
    )

    print("\nRegistered document:")
    print(document)

    # Test owner
    owner_access = can_access_document(
        document,
        "user_001",
    )

    print(f"\nOwner access: {owner_access}")
    assert owner_access is True

    
    # Test shared user
    shared_user_access = can_access_document(
        document,
        "user_002",
    )

    print(f"Shared user access: {shared_user_access}")
    assert shared_user_access is True

    
    # Test unrelated user
    unrelated_user_access = can_access_document(
        document,
        "user_003",
    )

    print(f"Unrelated user access: {unrelated_user_access}")
    assert unrelated_user_access is False

    print("\n" + "=" * 80)
    print("RESULT: Document + access control integration is working.")
    print("=" * 80)


if __name__ == "__main__":
    main()