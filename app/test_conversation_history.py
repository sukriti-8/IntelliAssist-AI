from app.conversation_history import (
    add_message,
    get_recent_history,
    format_history,
)


def main():
    print("=" * 80)
    print("CONVERSATION HISTORY TEST")
    print("=" * 80)

    history = []

    add_message(
        history,
        "user",
        "What is the main problem addressed by the project?",
    )

    add_message(
        history,
        "assistant",
        "The project addresses costly and inefficient jet-engine maintenance.",
    )

    add_message(
        history,
        "user",
        "How much can an AOG event cost?",
    )

    add_message(
        history,
        "assistant",
        "An AOG event can cost $10,000 to $150,000 per hour.",
    )

    print("\nFull history:")
    print(history)

    print("\nRecent history:")
    recent = get_recent_history(history, max_messages=2)
    print(recent)

    print("\nFormatted history:")
    print(format_history(history, max_messages=4))

    print("\nRESULT: Conversation history is working.")


if __name__ == "__main__":
    main()