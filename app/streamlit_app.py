import sys
from pathlib import Path

import streamlit as st


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.ingestion.pdf_loader import load_pdf
from app.duplicate_detector import (
    calculate_file_hash,
    calculate_content_hash,
)
from app.document_registery import (
    load_registry,
    find_duplicate,
    register_document,
)


st.set_page_config(
    page_title="IntelliAssist AI",
    layout="wide",
)


st.title(" IntelliAssist AI")
st.subheader("Smart Document AI Assistant")

st.write(
    "Upload a document and ask questions using "
    "document-grounded AI."
)


st.divider()


st.header("📄 Document Upload")

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
)


if uploaded_file is not None:

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / uploaded_file.name

    with file_path.open("wb") as file:
        file.write(uploaded_file.getbuffer())

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    try:
        
        # 1. Extract PDF text
        pages = load_pdf(str(file_path))

        full_text = "\n".join(
            page["text"]
            for page in pages
        )

        st.info(
            f"Extracted {len(pages)} pages "
            f"and {len(full_text)} characters."
        )

        
        # 2. Calculate hashes
        file_hash = calculate_file_hash(
            str(file_path)
        )

        content_hash = calculate_content_hash(
            full_text
        )

        
        # 3. Check duplicate registry
        registry = load_registry()

        duplicate = find_duplicate(
            file_hash=file_hash,
            content_hash=content_hash,
            registry=registry,
        )

        if duplicate:

            st.warning(
                "⚠️ Duplicate document detected."
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
                "We will ask for confirmation before "
                "allowing duplicate processing."
            )

        else:

            
            # 4. Register new document
            document = register_document(
                filename=uploaded_file.name,
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

            st.write(
                f"**Pages:** {len(pages)}"
            )

            st.write(
                f"**Characters extracted:** "
                f"{len(full_text)}"
            )

    except Exception as error:

        st.error(
            f"Error processing document: {error}"
        )


st.divider()


st.header(" Ask Your Document")

question = st.text_input(
    "Enter your question:"
)


if question:
    st.info(
        "RAG answering will be connected next."
    )