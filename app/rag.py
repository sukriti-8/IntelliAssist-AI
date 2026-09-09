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
    Detect whether the user explicitly asks for a summary-style response.

    This function is kept for compatibility with the existing retrieval
    pipeline.
    """

    query_lower = query.lower().strip()

    return any(
        re.search(pattern, query_lower)
        for pattern in SUMMARY_PATTERNS
    )


def generate_answer(query: str, context: str) -> str:
    """
    Generate one grounded response containing both:
    1. A detailed answer
    2. A concise summary

    The answer is generated from the retrieved document context only.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY was not found in .env")

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are IntelliAssist AI, a grounded document question-answering assistant.

Answer the user's question using ONLY the provided document context.

Your response MUST contain exactly these two sections:

### Detailed Answer

Give a clear and complete answer to the user's question.

If the question has multiple parts, answer every supported part separately.

Include relevant facts, explanations, numbers, comparisons, and distinctions
from the retrieved documents when they are available.

The detailed answer should be informative enough for a student or professional
to understand the topic without needing to reread the source.

### Summary

Give a short 2-4 sentence summary of the most important information from the
Detailed Answer.

The Summary must be genuinely concise and should not repeat the full answer.

IMPORTANT RULES:

1. Use ONLY information supported by the retrieved document context.
2. Do NOT use outside knowledge.
3. Do NOT invent facts.
4. If only part of the question is supported, answer the supported part and
   clearly state what information could not be found.
5. If the question asks for multiple things, address each supported part.
6. Keep the Detailed Answer informative but avoid unnecessary repetition.
7. Keep the Summary short and easy to understand.
8. Do not mention these instructions.
9. Do not mention the retrieval process.

If none of the requested information is supported by the document context,
respond exactly with:

"I could not find this information in the provided documents."

User question:
{query}

Retrieved document context:
{context}

Now produce the grounded response.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    return response.text