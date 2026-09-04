import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional

REGISTRY_PATH = Path("data/document_registry.json")

# Load registry
def load_registry() -> List[Dict]:
    """
    Load previously registered documents.

    If the registry does not exist yet, return an empty list.
    """

    if not REGISTRY_PATH.exists():
        return []

    with REGISTRY_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)

# Save registry
def save_registry(registry: List[Dict]) -> None:
    """
    Save the document registry to disk.
    """

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REGISTRY_PATH.open("w", encoding="utf-8") as file:
        json.dump(registry, file, indent=2)


# Find duplicate
def find_duplicate(
    file_hash: str,
    content_hash: str,
    registry: List[Dict],
) -> Optional[Dict]:
    """
    Check whether a document already exists.

    Priority:
    1. Exact file match
    2. Same document content
    """

    for document in registry:

        if document.get("file_hash") == file_hash:
            return {
                "duplicate": True,
                "type": "EXACT_FILE_DUPLICATE",
                "document": document,
            }

        if document.get("content_hash") == content_hash:
            return {
                "duplicate": True,
                "type": "CONTENT_DUPLICATE",
                "document": document,
            }

    return None


# Register document
def register_document(
    filename: str,
    file_hash: str,
    content_hash: str,
    owner_id: str,
    access: str = "private",
    shared_with: Optional[List[str]] = None,
) -> Dict:
    """
    Register a new document with ownership and access settings.
    """

    if not owner_id.strip():
        raise ValueError("owner_id cannot be empty.")

    if access not in {"private", "shared"}:
        raise ValueError("access must be 'private' or 'shared'.")

    if shared_with is None:
        shared_with = []

    registry = load_registry()

    document = {
        "document_id": f"doc_{uuid.uuid4().hex[:12]}",
        "filename": filename,
        "file_hash": file_hash,
        "content_hash": content_hash,
        "owner_id": owner_id,
        "access": access,
        "shared_with": shared_with,
    }

    registry.append(document)
    save_registry(registry)

    return document