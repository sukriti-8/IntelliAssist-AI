import json
from pathlib import Path
from typing import Dict, List, Optional


REGISTRY_PATH = Path("data/pack_registry.json")


def load_packs() -> List[Dict]:
    """Load all document packs from the registry."""

    if not REGISTRY_PATH.exists():
        return []

    with REGISTRY_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_packs(packs: List[Dict]) -> None:
    """Save all document packs to the registry."""

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REGISTRY_PATH.open("w", encoding="utf-8") as file:
        json.dump(packs, file, indent=2)


def create_pack(
    pack_id: str,
    name: str,
    description: str = "",
) -> Dict:
    """Create and save a new document pack."""

    packs = load_packs()

    pack = {
        "pack_id": pack_id,
        "name": name,
        "description": description,
        "documents": [],
    }

    packs.append(pack)
    save_packs(packs)

    return pack


def find_pack(pack_id: str) -> Optional[Dict]:
    """Find a pack by its ID."""

    packs = load_packs()

    for pack in packs:
        if pack.get("pack_id") == pack_id:
            return pack

    return None
def add_document_to_pack(pack_id: str, document_id: str) -> Dict:
    """Add a document to an existing pack."""

    packs = load_packs()

    for pack in packs:
        if pack.get("pack_id") == pack_id:

            if document_id in pack["documents"]:
                return pack

            pack["documents"].append(document_id)
            save_packs(packs)

            return pack

    raise ValueError(f"Pack not found: {pack_id}")


def remove_document_from_pack(pack_id: str, document_id: str) -> Dict:
    """Remove a document from an existing pack."""

    packs = load_packs()

    for pack in packs:
        if pack.get("pack_id") == pack_id:

            if document_id in pack["documents"]:
                pack["documents"].remove(document_id)
                save_packs(packs)

            return pack

    raise ValueError(f"Pack not found: {pack_id}")