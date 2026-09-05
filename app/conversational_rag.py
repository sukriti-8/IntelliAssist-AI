import os
from typing import Dict, List

from dotenv import load_dotenv
from google import genai

from app.conversation_history import add_message, format_history

load_dotenv()

MODEL_NAME = "gemini-3.6-flash"


def generate_rag_answer(
    question: str,
    retrieved_chunks: List[Dict],
    history: List[Dict],
) -> str:
    """
    Generate a document-grounded answer using retrieved context
    and recent conversation history.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not retrieved_chunks:
        raise ValueError("Retrieved chunks cannot be empty.")

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=api_key)

    document_context = "\n\n".join(
        f"[Page {chunk['page_number']}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    )

    conversation_context = format_history(
        history,
        max_messages=6,
    )

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the provided document context.

Use the conversation history only to understand references
or follow-up questions.

Rules:
- Do not use outside knowledge.
- Do not invent information.
- If the document context does not contain enough evidence,
  say that the answer cannot be determined from the document.
- Give a concise and clear answer.
- Mention the relevant page number when possible.

Conversation history:
{conversation_context}

Document context:
{document_context}

Current question:
{question}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    answer = response.text.strip()

    add_message(history, "user", question)
    add_message(history, "assistant", answer)

    return answer