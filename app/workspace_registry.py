import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = PROJECT_ROOT / "data" / "workspace_registry.json"


def load_registry():
    """Load all workspaces from the persistent registry."""
    if not REGISTRY_PATH.exists():
        return []

    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    return data if isinstance(data, list) else []


def save_registry(registry):
    """Persist the workspace registry."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2),
        encoding="utf-8",
    )


def find_workspace(workspace_id=None, name=None, registry=None):
    """Find a workspace by ID or exact name."""
    registry = load_registry() if registry is None else registry

    for workspace in registry:
        if workspace_id and workspace.get("workspace_id") == workspace_id:
            return workspace
        if name and workspace.get("name") == name:
            return workspace

    return None


def create_workspace(name, owner_id, registry=None):
    """Create a new workspace with duplicate-name protection per owner."""
    name = name.strip()
    owner_id = owner_id.strip()

    if not name:
        raise ValueError("Workspace name cannot be empty.")
    if not owner_id:
        raise ValueError("Owner ID cannot be empty.")

    registry = load_registry() if registry is None else registry

    for workspace in registry:
        if (
            workspace.get("owner_id") == owner_id
            and workspace.get("name", "").casefold() == name.casefold()
        ):
            raise ValueError("A workspace with this name already exists.")

    workspace = {
        "workspace_id": f"ws_{uuid.uuid4().hex[:12]}",
        "name": name,
        "owner_id": owner_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "document_ids": [],
        "chat_ids": [],
    }

    registry.append(workspace)
    save_registry(registry)
    return workspace


def add_document_to_workspace(workspace_id, document_id, registry=None):
    """Attach a document to a workspace without creating duplicates."""
    registry = load_registry() if registry is None else registry
    workspace = find_workspace(workspace_id=workspace_id, registry=registry)

    if workspace is None:
        raise ValueError("Workspace not found.")

    if document_id not in workspace["document_ids"]:
        workspace["document_ids"].append(document_id)

    save_registry(registry)
    return workspace


def remove_document_from_workspace(workspace_id, document_id, registry=None):
    """Remove a document reference from a workspace."""
    registry = load_registry() if registry is None else registry
    workspace = find_workspace(workspace_id=workspace_id, registry=registry)

    if workspace is None:
        raise ValueError("Workspace not found.")

    workspace["document_ids"] = [
        item for item in workspace.get("document_ids", [])
        if item != document_id
    ]

    save_registry(registry)
    return workspace


def add_chat_to_workspace(workspace_id, chat_id, registry=None):
    """Attach a chat ID to a workspace."""
    registry = load_registry() if registry is None else registry
    workspace = find_workspace(workspace_id=workspace_id, registry=registry)

    if workspace is None:
        raise ValueError("Workspace not found.")

    if chat_id not in workspace["chat_ids"]:
        workspace["chat_ids"].append(chat_id)

    save_registry(registry)
    return workspace


def remove_chat_from_workspace(workspace_id, chat_id, registry=None):
    """Remove a chat reference from a workspace."""
    registry = load_registry() if registry is None else registry
    workspace = find_workspace(workspace_id=workspace_id, registry=registry)

    if workspace is None:
        raise ValueError("Workspace not found.")

    workspace["chat_ids"] = [
        item for item in workspace.get("chat_ids", [])
        if item != chat_id
    ]

    save_registry(registry)
    return workspace


def rename_workspace(workspace_id, new_name, registry=None):
    """Rename a workspace while preserving its ID."""
    new_name = new_name.strip()
    if not new_name:
        raise ValueError("Workspace name cannot be empty.")

    registry = load_registry() if registry is None else registry
    workspace = find_workspace(workspace_id=workspace_id, registry=registry)

    if workspace is None:
        raise ValueError("Workspace not found.")

    for item in registry:
        if (
            item.get("workspace_id") != workspace_id
            and item.get("owner_id") == workspace.get("owner_id")
            and item.get("name", "").casefold() == new_name.casefold()
        ):
            raise ValueError("A workspace with this name already exists.")

    workspace["name"] = new_name
    save_registry(registry)
    return workspace
