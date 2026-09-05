import os
import time
from dotenv import load_dotenv
from google import genai
load_dotenv()

MODEL_NAME = "gemini-3.6-flash"


def summarize_text(text: str) -> str:
    """
    Generate a concise summary of the provided text using Gemini.
    """

    if not text.strip():
        raise ValueError("Text cannot be empty.")

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=api_key)

    prompt = f"""
Summarize the following document clearly and accurately.

Requirements:
- Identify the main purpose of the document.
- Extract the most important points.
- Mention important methods, data, or approaches when present.
- Do not invent information that is not in the document.
- Keep the summary concise and easy to understand.

Document:
{text}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    return response.text.strip()
def summarize_chunk(text: str) -> str:
    """
    Generate a concise summary of a single document chunk using Gemini.
    """

    if not text.strip():
        raise ValueError("Chunk text cannot be empty.")

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=api_key)

    prompt = f"""
Summarize the following document section in 2-4 sentences.

Requirements:
- Preserve the important factual information.
- Focus only on information present in the section.
- Do not invent or assume information.
- Keep technical terms when they are important.
- Do not add information from outside the section.

Document section:
{text}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    return response.text.strip()
import time


def hierarchical_summarize(
    chunks: list[dict],
    batch_size: int = 4,
    delay_seconds: float = 12.0,
) -> str:
    """
    Summarize a document using hierarchical summarization.

    Chunks are grouped into batches. Each batch is summarized
    first, then the batch summaries are combined into one
    final document summary.
    """

    if not chunks:
        raise ValueError("Chunks cannot be empty.")

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    valid_chunks = [
        chunk for chunk in chunks
        if chunk.get("text", "").strip()
    ]

    if not valid_chunks:
        raise ValueError("No valid chunks were provided.")

    batch_summaries = []

    for start in range(0, len(valid_chunks), batch_size):
        batch = valid_chunks[start:start + batch_size]

        batch_text = "\n\n".join(
            f"Section {start + index + 1}:\n{chunk['text']}"
            for index, chunk in enumerate(batch)
        )

        batch_number = (start // batch_size) + 1
        total_batches = (
            (len(valid_chunks) + batch_size - 1) // batch_size
        )

        print(
            f"Summarizing batch {batch_number}/{total_batches} "
            f"({len(batch)} chunks)..."
        )

        summary = summarize_text(batch_text)
        batch_summaries.append(summary)

        if start + batch_size < len(valid_chunks):
            time.sleep(delay_seconds)

    combined_summaries = "\n\n".join(
        f"Batch {index + 1} summary:\n{summary}"
        for index, summary in enumerate(batch_summaries)
    )

    print("Generating final document summary...")

    final_summary = summarize_text(combined_summaries)

    return final_summary