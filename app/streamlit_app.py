import sys
from pathlib import Path

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from app.confidence import assess_evidence
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_pages
from app.duplicate_detector import (
    calculate_file_hash,
    calculate_content_hash,
)
from app.document_registery import (
    load_registry,
    find_duplicate,
    register_document,
)
from app.vector_store import (
    build_index,
    save_index,
    save_metadata,
    load_index,
    load_metadata,
)


st.set_page_config(
    page_title="IntelliAssist AI",
    layout="wide",
)

st.title("IntelliAssist AI")
st.subheader("Smart Document AI Assistant")

st.write(
    "Upload a document and prepare it for document-grounded AI."
)

st.divider()

st.header("Document Upload")

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-m3")


if uploaded_file is not None:

    upload_dir = PROJECT_ROOT / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Use a server-controlled filename instead of trusting
    # the user's uploaded filename as a filesystem path.
    safe_filename = Path(uploaded_file.name).name
    file_path = upload_dir / safe_filename

    with file_path.open("wb") as file:
        file.write(uploaded_file.getbuffer())

    st.success(f"Uploaded: {safe_filename}")

    try:
        
        # 1. PDF EXTRACTION     
        pages = load_pdf(str(file_path))

        full_text = "\n".join(
            page["text"]
            for page in pages
        )

        st.info(
            f"Extracted {len(pages)} pages "
            f"and {len(full_text)} characters."
        )

        
        # 2. DUPLICATE DETECTION       
        file_hash = calculate_file_hash(
            str(file_path)
        )

        content_hash = calculate_content_hash(
            full_text
        )

        registry = load_registry()

        duplicate = find_duplicate(
            file_hash=file_hash,
            content_hash=content_hash,
            registry=registry,
        )

        if duplicate:

            st.warning(
                "Duplicate document detected."
            )

            st.write(
                f"**Type:** {duplicate['type']}"
            )

            existing_document = duplicate["document"]

            st.write(
                f"**Existing file:** "
                f"{existing_document['filename']}"
            )

            st.info(
                "This document is already registered. "
                "Duplicate processing was skipped."
            )

        else:

            # 3. REGISTER DOCUMENT         
            document = register_document(
                filename=safe_filename,
                file_hash=file_hash,
                content_hash=content_hash,
                owner_id="user_001",
            )

            st.success(
                "New document registered."
            )

            st.write(
                f"**Document ID:** "
                f"{document['document_id']}"
            )

           
            # 4. CHUNKING           
            with st.spinner(
                "Creating document chunks..."
            ):

                chunks = chunk_pages(pages)

            st.success(
                f"Created {len(chunks)} document chunks."
            )

            
            # 5. BGE-M3 EMBEDDINGS
            with st.spinner(
                "Generating BGE-M3 embeddings..."
            ):

                model = load_embedding_model()

                texts = [
                    chunk["text"]
                    for chunk in chunks
                ]

                embeddings = model.encode(
                    texts,
                    normalize_embeddings=True,
                )

                embeddings = np.asarray(
                    embeddings,
                    dtype="float32",
                )

            st.success(
                f"Generated embeddings for "
                f"{len(embeddings)} chunks."
            )

            st.write(
                f"**Embedding dimensions:** "
                f"{embeddings.shape[1]}"
            )

            
            # 6. BUILD FAISS INDEX
            with st.spinner(
                "Building FAISS vector index..."
            ):

                index = build_index(
                    embeddings
                )

                save_index(index)
                save_metadata(chunks)

            st.success(
                f"FAISS index created with "
                f"{index.ntotal} vectors."
            )

            st.info(
                "Document processing complete. "
                "The document is now ready for retrieval."
            )

    except Exception as error:

        st.error(
            f"Error processing document: {error}"
        )


st.divider()

st.header("Ask Your Document")

question = st.chat_input(
    "Ask another question about your document..."
)

if question:

    try:
   
        # 1. LOAD SAVED FAISS DATA
        index = load_index()
        chunks = load_metadata()


        # 2. LOAD MODELS 
        embedding_model = load_embedding_model()

        reranker = CrossEncoder(
            "BAAI/bge-reranker-v2-m3"
        )

  
        # 3. CREATE QUERY EMBEDDING
        query_embedding = embedding_model.encode(
            [question],
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        )


        # 4. FAISS CANDIDATE RETRIEVAL
        candidate_k = min(10, index.ntotal)

        scores, indices = index.search(
            query_embedding,
            candidate_k,
        )

        candidates = []

        for score, index_position in zip(
            scores[0],
            indices[0],
        ):

            if index_position < 0:
                continue

            chunk = chunks[index_position].copy()

            chunk["semantic_score"] = float(score)

            candidates.append(chunk)

  
        # 5. RERANK CANDIDATES
        pairs = [
            [question, candidate["text"]]
            for candidate in candidates
        ]

        reranker_scores = reranker.predict(pairs)

        reranked_results = []

        for candidate, score in zip(
            candidates,
            reranker_scores,
        ):

            result = candidate.copy()

            result["reranker_score"] = float(score)

            reranked_results.append(result)

        reranked_results.sort(
            key=lambda item: item["reranker_score"],
            reverse=True,
        )

       
        # 6. EVIDENCE ASSESSMENT   
        evidence_results = reranked_results[:3]

        assessment = assess_evidence(
            evidence_results
        )

        st.subheader("Evidence Assessment")

        st.write(
            f"Confidence: "
            f"{assessment['confidence']:.4f}"
        )

        st.write(
            f"Level: {assessment['level']}"
        )

        st.write(
            f"Decision: {assessment['decision']}"
        )

  
        # 7. DISPLAY RETRIEVED EVIDENCE
        st.subheader("Retrieved Evidence")

        for rank, result in enumerate(
            evidence_results,
            start=1,
        ):

            st.write(
                f"**Result {rank}**"
            )

            st.write(
                f"Page: {result['page_number']} | "
                f"Chunk: {result['chunk_id']} | "
                f"Reranker score: "
                f"{result['reranker_score']:.4f}"
            )

            st.write(result["text"])

            st.divider()

   
        # 8. ANSWER ONLY WHEN EVIDENCE EXISTS   
        if assessment["decision"] == "INSUFFICIENT_EVIDENCE":

            st.warning(
                "I could not find enough information "
                "in the provided document to answer this question."
            )

        else:

            context_parts = []

            for result in evidence_results:

                context_parts.append(
                    f"[Page {result['page_number']}]\n"
                    f"{result['text']}"
                )

            context = "\n\n".join(context_parts)

            from app.rag import generate_answer

            with st.spinner(
                "Generating document-grounded answer..."
            ):

                answer = generate_answer(
                    query=question,
                    context=context,
                )

            st.subheader("Answer")

            st.write(answer)

    except FileNotFoundError:

        st.warning(
            "No processed document is available. "
            "Upload and process a document first."
        )

    except Exception as error:

        st.error(
            f"Error during question answering: {error}"
        )

