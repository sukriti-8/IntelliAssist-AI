import hashlib
import re
from pathlib import Path


# ---------------------------------------------------------
# Exact file hash
# ---------------------------------------------------------

def calculate_file_hash(file_path: str) -> str:
    """
    Calculate SHA-256 hash of the complete file.

    Identical files will have identical hashes.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(8192), b""):
            sha256.update(block)

    return sha256.hexdigest()


# ---------------------------------------------------------
# Normalize document text
# ---------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize document text before calculating its hash.

    This helps detect documents with identical content even
    when whitespace formatting differs.
    """

    text = text.lower()

    # Replace multiple whitespace characters with one space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------
# Content hash
# ---------------------------------------------------------

def calculate_content_hash(text: str) -> str:
    """
    Calculate SHA-256 hash of normalized document content.
    """

    normalized = normalize_text(text)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()