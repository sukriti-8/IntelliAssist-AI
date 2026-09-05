from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_pages
from app.retrieval import retrieve
from app.conversation_history import format_history
from app.conversational_rag import generate_rag_answer

from sentence_transformers import SentenceTransformer


PDF_PATH = "data/raw/test.pdf"
MODEL_NAME = "BAAI/bge-m3"


def main():
    print("=" * 80)
    print("CONVERSATIONAL RAG TEST")
    print("=" * 80)

    print("\nLoading document...")
    pages = load_pdf(PDF_PATH)

    print(f"Pages loaded: {len(pages)}")

    chunks = chunk_pages(pages)

    print(f"Chunks created: {len(chunks)}")

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    history = []

    questions = [
        "What is the main problem addressed by the project?",
        "How much can an AOG event cost?",
        "What data is used by the project?",
    ]

    for index, question in enumerate(questions, start=1):
        print("\n" + "-" * 80)
        print(f"QUESTION {index}")
        print("-" * 80)
        print(question)

        retrieved = retrieve(
            query=question,
            chunks=chunks,
            model=model,
            top_k=3,
        )

        answer = generate_rag_answer(
            question=question,
            retrieved_chunks=retrieved,
            history=history,
        )

        print("\nANSWER")
        print(answer)

        print("\nHISTORY AFTER THIS TURN")
        print(format_history(history))

    print("\n" + "=" * 80)
    print("RESULT: Conversational RAG integration completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()