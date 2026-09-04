from app.duplicate_detector import (
    calculate_file_hash,
    calculate_content_hash,
    normalize_text,
)


# ---------------------------------------------------------
# Test file hash
# ---------------------------------------------------------

file_path = "data/raw/test.pdf"

print("=" * 80)
print("DUPLICATE DOCUMENT DETECTION TEST")
print("=" * 80)

file_hash = calculate_file_hash(file_path)

print("\nFile:")
print(file_path)

print("\nSHA-256 file hash:")
print(file_hash)


# ---------------------------------------------------------
# Test content normalization
# ---------------------------------------------------------

text1 = """
Aircraft predictive maintenance
uses sensor data to predict
remaining useful life.
"""

text2 = """
Aircraft predictive maintenance uses sensor data
to predict remaining useful life.
"""

normalized1 = normalize_text(text1)
normalized2 = normalize_text(text2)

print("\n" + "=" * 80)
print("CONTENT NORMALIZATION TEST")
print("=" * 80)

print("\nNormalized text 1:")
print(normalized1)

print("\nNormalized text 2:")
print(normalized2)

print("\nNormalized texts identical:")
print(normalized1 == normalized2)


# ---------------------------------------------------------
# Test content hash
# ---------------------------------------------------------

content_hash1 = calculate_content_hash(text1)
content_hash2 = calculate_content_hash(text2)

print("\n" + "=" * 80)
print("CONTENT HASH TEST")
print("=" * 80)

print("\nContent hash 1:")
print(content_hash1)

print("\nContent hash 2:")
print(content_hash2)

print("\nContent hashes identical:")
print(content_hash1 == content_hash2)


# ---------------------------------------------------------
# Final result
# ---------------------------------------------------------

print("\n" + "=" * 80)

if normalized1 == normalized2 and content_hash1 == content_hash2:
    print("RESULT: Duplicate content detection is working.")
else:
    print("RESULT: Test failed.")

print("=" * 80)