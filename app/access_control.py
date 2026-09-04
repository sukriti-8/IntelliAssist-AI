from typing import Dict


def can_access_document(
    document: Dict,
    user_id: str,
) -> bool:
    """
    Check whether a user is allowed to access a document.

    Access rules:
    1. The document owner can always access it.
    2. A shared user can access it when the document is shared.
    3. Everyone else is denied.
    """

    if not user_id.strip():
        return False

    # Owner always has access.
    if document.get("owner_id") == user_id:
        return True

    # Shared users have access only when sharing is enabled.
    if document.get("access") == "shared":
        shared_with = document.get("shared_with", [])
        return user_id in shared_with

    # Private document → non-owner denied.
    return False