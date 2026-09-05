from typing import Dict, List


def add_message(
    history: List[Dict],
    role: str,
    content: str,
) -> List[Dict]:
    """
    Add a user or assistant message to conversation history.
    """

    if role not in {"user", "assistant"}:
        raise ValueError("role must be 'user' or 'assistant'.")

    if not content.strip():
        raise ValueError("content cannot be empty.")

    history.append({
        "role": role,
        "content": content.strip(),
    })

    return history


def get_recent_history(
    history: List[Dict],
    max_messages: int = 6,
) -> List[Dict]:
    """
    Return the most recent messages from the conversation.
    """

    if max_messages < 1:
        raise ValueError("max_messages must be at least 1.")

    return history[-max_messages:]


def format_history(
    history: List[Dict],
    max_messages: int = 6,
) -> str:
    """
    Convert recent conversation history into text
    that can be provided to the language model.
    """

    recent_history = get_recent_history(
        history,
        max_messages=max_messages,
    )

    if not recent_history:
        return "No previous conversation."

    formatted = []

    for message in recent_history:
        role = message["role"].capitalize()
        content = message["content"]

        formatted.append(
            f"{role}: {content}"
        )

    return "\n".join(formatted)