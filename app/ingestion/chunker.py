from typing import List, Dict


def chunk_pages(
    pages: List[Dict],
    chunk_size: int = 1000,
    overlap: int = 150,
) -> List[Dict]:
    """
    Create paragraph-aware chunks while preserving page metadata.

    This is our baseline chunking strategy.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"].strip()

        if not text:
            continue

        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n")
            if paragraph.strip()
        ]

        current_chunk = ""

        for paragraph in paragraphs:
            candidate = (
                paragraph
                if not current_chunk
                else f"{current_chunk}\n\n{paragraph}"
            )

            if len(candidate) <= chunk_size:
                current_chunk = candidate
                continue

            if current_chunk:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "source": page["source"],
                        "page_number": page["page_number"],
                        "text": current_chunk,
                    }
                )

                chunk_id += 1

                overlap_text = current_chunk[-overlap:] if overlap else ""

                current_chunk = (
                    f"{overlap_text}\n\n{paragraph}"
                    if overlap_text
                    else paragraph
                )

            else:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "source": page["source"],
                        "page_number": page["page_number"],
                        "text": paragraph,
                    }
                )

                chunk_id += 1
                current_chunk = ""

        if current_chunk:
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source": page["source"],
                    "page_number": page["page_number"],
                    "text": current_chunk,
                }
            )

            chunk_id += 1

    return chunks