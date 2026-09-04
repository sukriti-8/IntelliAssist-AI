from app.document_registery import (
    load_registry,
    save_registry,
    find_duplicate,
    register_document,
)


print("=" * 80)
print("DOCUMENT REGISTRY TEST")
print("=" * 80)


# ---------------------------------------------------------
# Start with a clean test registry
# ---------------------------------------------------------

save_registry([])

print("\nInitial registry:")
print(load_registry())


# ---------------------------------------------------------
# Register first document
# ---------------------------------------------------------

document = register_document(
    filename="aviation.pdf",
    file_hash="FILE_HASH_123",
    content_hash="CONTENT_HASH_ABC",
)

print("\nRegistered document:")
print(document)


# ---------------------------------------------------------
# Test exact file duplicate
# ---------------------------------------------------------

registry = load_registry()

result = find_duplicate(
    file_hash="FILE_HASH_123",
    content_hash="DIFFERENT_CONTENT",
    registry=registry,
)

print("\nExact file duplicate test:")
print(result)


# ---------------------------------------------------------
# Test content duplicate
# ---------------------------------------------------------

result = find_duplicate(
    file_hash="DIFFERENT_FILE",
    content_hash="CONTENT_HASH_ABC",
    registry=registry,
)

print("\nContent duplicate test:")
print(result)


# ---------------------------------------------------------
# Test new document
# ---------------------------------------------------------

result = find_duplicate(
    file_hash="NEW_FILE",
    content_hash="NEW_CONTENT",
    registry=registry,
)

print("\nNew document test:")
print(result)


# ---------------------------------------------------------
# Final result
# ---------------------------------------------------------

print("\n" + "=" * 80)

if (
    load_registry()
    and result is None
):
    print("RESULT: Document registry is working.")

else:
    print("RESULT: Test failed.")

print("=" * 80)