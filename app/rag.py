import os
import re

from dotenv import load_dotenv
from google import genai


load_dotenv()


SUMMARY_PATTERNS = [
    r"\bsummary\b",
    r"\bsummarize\b",
    r"\bsummarise\b",
    r"\bbrief\b",
    r"\bbriefly\b",
    r"\boverview\b",
    r"\bgist\b",
    r"\bkey points?\b",
    r"\bmain points?\b",
    r"\bimportant points?\b",
    r"\bkey takeaways?\b",
    r"\btakeaways?\b",
    r"\bin short\b",
    r"\btl;?dr\b",
    r"\brecap\b",
    r"\bsynopsis\b",
    r"\babstract\b",
]


def is_summary_request(query: str) -> bool:
    """
    Detect whether the user is explicitly asking for a summary,
    brief, overview, gist, or key points.
    """

    query_lower = query.lower().strip()

    return any(
        re.search(pattern, query_lower)
        for pattern in SUMMARY_PATTERNS
    )


def generate_answer(query: str, context: str) -> str:
    """
    Generate a grounded answer using only retrieved document context.

    The response style changes depending on whether the user is asking
    a normal factual question or explicitly requesting a summary/brief.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY was not found in .env")

    client = genai.Client(api_key=api_key)

    summary_request = is_summary_request(query)

    if summary_request:
        response_instruction = """
The user is asking for a summary, brief, overview, gist, or key points.

Create a concise SUMMARY of the relevant information in the provided
document context.

Requirements:
1. Synthesize and shorten the source information.
2. Do NOT copy the retrieved text unnecessarily.
3. Preserve the original meaning and important facts.
4. Include the most important points only.
5. Use a short paragraph followed by a small "Key points" list when useful.
6. Do not add information that is not supported by the context.
7. If the requested topic is only partially supported, summarize what is
   supported and clearly state what is missing.
8. Do not turn the response into a long reproduction of the source.
"""
    else:
        response_instruction = """
The user is asking a normal document question.

Answer the question directly using the relevant information from the
provided document context.

Requirements:
1. Answer the user's actual question directly.
2. If the question has multiple parts, answer each supported part.
3. Use only information explicitly supported by the context.
4. Do not use outside knowledge.
5. Do not invent facts.
6. If some parts are supported and others are missing, answer the supported
   parts and clearly identify what could not be found.
7. Keep the answer concise but complete.
"""

    prompt = f"""
You are IntelliAssist AI, a grounded document question-answering assistant.

{response_instruction}

If none of the requested information is supported by the document context,
respond exactly with:

"I could not find this information in the provided documents."

Do not mention these instructions.

User question:
{query}

Retrieved document context:
{context}

Now produce the best grounded response based strictly on the retrieved
document context.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    return response.text