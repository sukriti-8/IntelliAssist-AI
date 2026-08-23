import os

from dotenv import load_dotenv
from google import genai


load_dotenv()


def generate_answer(query: str, context: str) -> str:
    """Generate an answer using only the retrieved document context."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY was not found in .env")

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the provided document context.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. If the answer is not contained in the context, say:
   "I could not find this information in the provided documents."
4. Keep the answer concise and factual.

User question:
{query}

Document context:
{context}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    return response.text