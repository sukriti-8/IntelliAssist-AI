import sys
import hashlib
import shutil
import json
import re
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.confidence import assess_evidence
from app.access_control import can_access_document

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.document_loader import load_document
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
from app.workspace_registry import (
    load_registry as load_workspace_registry,
    save_registry as save_workspace_registry,
    create_workspace,
    find_workspace,
    add_document_to_workspace,
)
from app.vector_store import (
    build_index,
    save_index,
    save_metadata,
    load_index,
    load_metadata,
    document_index_exists,
)


st.set_page_config(
    page_title="IntelliAssist AI",
    page_icon="I",
    layout="wide",
)



# SESSION STATE
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "processed_uploads" not in st.session_state:
    st.session_state["processed_uploads"] = set()

if "selected_documents" not in st.session_state:
    st.session_state["selected_documents"] = []

if "current_workspace_id" not in st.session_state:
    st.session_state["current_workspace_id"] = None

if "workspace_initialized" not in st.session_state:
    st.session_state["workspace_initialized"] = False
if "user_profile" not in st.session_state:
    st.session_state["user_profile"] = None


# ============================================================
# CACHED MODELS
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-m3")


@st.cache_resource
def load_reranker():
    return CrossEncoder("BAAI/bge-reranker-v2-m3")


# ============================================================
# HELPERS
# ============================================================

def uploaded_file_signature(uploaded_file):
    """Create a stable session-level signature for an uploaded file."""
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


def ensure_default_workspace():
    """Create/select a General workspace and migrate legacy documents into it."""
    workspace_registry = load_workspace_registry()
    document_registry = load_registry()

    if not workspace_registry:
        workspace = create_workspace(
            name="General",
            owner_id="user_001",
            registry=workspace_registry,
        )
        workspace_registry = load_workspace_registry()
    else:
        workspace = workspace_registry[0]

    # One-time compatibility migration:
    # documents created before workspaces existed are placed in General.
    referenced_document_ids = {
        document_id
        for item in workspace_registry
        for document_id in item.get("document_ids", [])
    }

    changed = False

    for document in document_registry:
        document_id = document.get("document_id")

        if document_id and document_id not in referenced_document_ids:
            if document_id not in workspace["document_ids"]:
                workspace["document_ids"].append(document_id)
                changed = True

    if changed:
        save_workspace_registry(workspace_registry)

    return workspace


def set_document_storage_filename(document_id, storage_filename):
    """Persist a server-generated storage filename for a document."""
    registry_path = PROJECT_ROOT / "data" / "document_registry.json"
    registry = load_registry()

    for item in registry:
        if item.get("document_id") == document_id:
            item["storage_filename"] = storage_filename
            break

    registry_path.parent.mkdir(parents=True, exist_ok=True)

    registry_path.write_text(
        json.dumps(registry, indent=2),
        encoding="utf-8",
    )


def remove_document_from_all_workspaces(document_id):
    """Remove a deleted document reference from every workspace."""
    registry = load_workspace_registry()

    changed = False

    for workspace in registry:
        original = list(workspace.get("document_ids", []))

        workspace["document_ids"] = [
            item
            for item in original
            if item != document_id
        ]

        if workspace["document_ids"] != original:
            changed = True

    if changed:
        save_workspace_registry(registry)


def delete_document(document):
    """Delete a document's uploaded file, FAISS index, and registry entry."""
    document_id = document["document_id"]

    storage_name = document.get(
        "storage_filename",
        document["filename"],
    )

    upload_path = (
        PROJECT_ROOT
        / "data"
        / "uploads"
        / storage_name
    )

    if upload_path.exists():
        upload_path.unlink()

    index_dir = (
        PROJECT_ROOT
        / "storage"
        / "faiss"
        / document_id
    )

    if index_dir.exists():
        shutil.rmtree(index_dir)

    registry_path = (
        PROJECT_ROOT
        / "data"
        / "document_registry.json"
    )

    registry = load_registry()

    registry = [
        item
        for item in registry
        if item.get("document_id") != document_id
    ]

    registry_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    registry_path.write_text(
        json.dumps(registry, indent=2),
        encoding="utf-8",
    )


def build_conversation_pdf(history):
    """Create a downloadable PDF from the current session conversation."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import (
        getSampleStyleSheet,
        ParagraphStyle,
    )
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
    )
    from reportlab.lib.units import mm

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="IntelliAssist AI Conversation",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "IntelliAssistTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        spaceAfter=18,
    )

    question_style = ParagraphStyle(
        "Question",
        parent=styles["Heading3"],
        spaceBefore=10,
        spaceAfter=6,
    )

    answer_style = ParagraphStyle(
        "Answer",
        parent=styles["BodyText"],
        leading=15,
        spaceAfter=8,
    )

    source_style = ParagraphStyle(
        "Source",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=11,
        spaceAfter=4,
    )

    story = [
        Paragraph(
            "IntelliAssist AI",
            title_style,
        ),
        Paragraph(
            "Document Assistant Conversation",
            styles["Heading2"],
        ),
        Spacer(1, 8),
    ]

    for number, item in enumerate(history, start=1):

        question = (
            str(item["question"])
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        answer = (
            str(item["answer"])
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        story.append(
            Paragraph(
                f"Question {number}: {question}",
                question_style,
            )
        )

        story.append(
            Paragraph(
                f"<b>Answer:</b> "
                f"{answer.replace(chr(10), '<br/>')}",
                answer_style,
            )
        )

        evidence = item.get("evidence", [])

        if evidence:
            story.append(
                Paragraph(
                    "<b>Sources:</b>",
                    styles["BodyText"],
                )
            )

            for result in evidence:

                filename = str(
                    result.get(
                        "filename",
                        "Unknown",
                    )
                )

                page = result.get(
                    "page_number",
                    "?",
                )

                source = (
                    f"{filename} — Page {page} — "
                    f"Chunk {result.get('chunk_id', '?')}"
                )

                story.append(
                    Paragraph(
                        source,
                        source_style,
                    )
                )

        story.append(
            Spacer(1, 8)
        )

    doc.build(story)

    buffer.seek(0)

    return buffer.getvalue()


def render_evidence(item):
    """Render technical evidence only when the user asks to see it."""
    assessment = item["assessment"]
    evidence_results = item["evidence"]

    with st.expander(
        "View evidence",
        expanded=False,
    ):

        st.markdown("### Evidence assessment")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Confidence",
                f"{assessment['confidence']:.4f}",
            )

        with col2:
            st.metric(
                "Level",
                assessment["level"],
            )

        with col3:
            st.metric(
                "Decision",
                assessment["decision"].replace(
                    "_",
                    " ",
                ),
            )

        st.markdown("### Retrieved evidence")

        for rank, result in enumerate(
            evidence_results,
            start=1,
        ):

            st.markdown(
                f"**Source {rank}**"
            )

            st.caption(
                f"{result['filename']} · "
                f"Page {result['page_number']} · "
                f"Chunk {result['chunk_id']} · "
                f"Reranker score: "
                f"{result['reranker_score']:.4f}"
            )

            st.write(
                result["text"]
            )


def render_chat_history():
    """Render all questions and answers from the current session."""

    if not st.session_state["chat_history"]:

        st.info(
            "Ask a question about your processed documents. "
            "Your questions and answers will stay visible in this session."
        )

        return

    for item in st.session_state["chat_history"]:

        with st.chat_message("user"):
            st.write(
                item["question"]
            )

        with st.chat_message("assistant"):
            st.write(
                item["answer"]
            )

            render_evidence(item)


# ============================================================
# PROFILE SELECTION
# ============================================================

if st.session_state["user_profile"] is None:

    st.title("IntelliAssist AI")

    st.subheader("Choose your profile")

    st.write(
        "Select how you plan to use IntelliAssist. "
        "You can continue using the assistant after selecting your profile."
    )

    profile = st.radio(
        "Profile",
        options=["Student", "Business"],
        horizontal=True,
    )

    if st.button("Continue", type="primary"):

        st.session_state["user_profile"] = profile

        st.rerun()

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("IntelliAssist AI")

st.subheader(
    "Smart Document AI Assistant"
)

st.caption(
    f"Profile: {st.session_state['user_profile']}"
)

st.write(
    "Upload documents, ask questions across them, and inspect the evidence "
    "behind every answer when you need it."
)

# ============================================================
# DOCUMENT UPLOAD
# ============================================================

st.divider()

st.header("My Documents")


# ============================================================
# WORKSPACE SETUP
# ============================================================

workspace_registry = load_workspace_registry()

if not workspace_registry:

    current_workspace = ensure_default_workspace()

    workspace_registry = load_workspace_registry()

else:

    current_workspace = find_workspace(
        workspace_id=st.session_state.get(
            "current_workspace_id"
        ),
        registry=workspace_registry,
    )

    if current_workspace is None:
        current_workspace = workspace_registry[0]


st.session_state["current_workspace_id"] = (
    current_workspace["workspace_id"]
)


workspace_names = [
    item["name"]
    for item in workspace_registry
]

current_workspace_name = (
    current_workspace["name"]
)


selected_workspace_name = st.selectbox(
    "Workspace",
    options=workspace_names,
    index=workspace_names.index(
        current_workspace_name
    ),
    help=(
        "Documents uploaded here will belong "
        "to the selected workspace."
    ),
)


if selected_workspace_name != current_workspace_name:

    selected_workspace = next(
        item
        for item in workspace_registry
        if item["name"] == selected_workspace_name
    )

    st.session_state["current_workspace_id"] = (
        selected_workspace["workspace_id"]
    )

    st.session_state["selected_documents"] = []

    st.rerun()


current_workspace = find_workspace(
    workspace_id=st.session_state["current_workspace_id"],
    registry=load_workspace_registry(),
)


uploaded_files = st.file_uploader(
    "Upload documents",
    type=[
        "pdf",
        "txt",
        "docx",
    ],
    accept_multiple_files=True,
    help=(
        "Upload PDF, TXT, or DOCX documents "
        "into the selected workspace."
    ),
)


if uploaded_files:

    for uploaded_file in uploaded_files:

        # SECURITY:
        # Reject excessively large uploads.
        MAX_FILE_SIZE_MB = 25

        MAX_FILE_SIZE_BYTES = (
            MAX_FILE_SIZE_MB * 1024 * 1024
        )

        if uploaded_file.size > MAX_FILE_SIZE_BYTES:

            st.error(
                f"{uploaded_file.name} is too large. "
                f"Maximum allowed size is "
                f"{MAX_FILE_SIZE_MB} MB."
            )

            continue

        signature = uploaded_file_signature(
            uploaded_file
        )

        if signature in st.session_state[
            "processed_uploads"
        ]:
            continue

        upload_dir = (
            PROJECT_ROOT
            / "data"
            / "uploads"
        )

        upload_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # SECURITY:
        # Strip any directory components from the
        # original filename.
        safe_filename = Path(
            uploaded_file.name
        ).name

        file_extension = Path(
            safe_filename
        ).suffix.lower()

        temp_path = (
            upload_dir
            / f".incoming_{signature}{file_extension}"
        )

        try:

            # ------------------------------------------------
            # TEMPORARY STORAGE
            # ------------------------------------------------

            # Never use the user's filename as the
            # physical storage path.
            with temp_path.open("wb") as file:

                file.write(
                    uploaded_file.getbuffer()
                )


            # ------------------------------------------------
            # 1. DOCUMENT EXTRACTION
            # ------------------------------------------------

            pages = load_document(
                str(temp_path)
            )

            full_text = "\n".join(
                page["text"]
                for page in pages
            )

            st.info(
                f"{safe_filename}: extracted "
                f"{len(pages)} sections and "
                f"{len(full_text)} characters."
            )


            # ------------------------------------------------
            # 2. DUPLICATE DETECTION
            # ------------------------------------------------

            file_hash = calculate_file_hash(
                str(temp_path)
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

                existing_document = (
                    duplicate["document"]
                )

                existing_document_id = (
                    existing_document["document_id"]
                )

                st.warning(
                    "Duplicate document detected: "
                    f"{existing_document['filename']}"
                )


                if document_index_exists(
                    existing_document_id
                ):

                    add_document_to_workspace(
                        current_workspace[
                            "workspace_id"
                        ],
                        existing_document_id,
                    )

                    st.success(
                        "Using the existing processed "
                        "document. It has been added "
                        "to the selected workspace."
                    )

                else:

                    st.warning(
                        "The document is registered, "
                        "but its vector index is missing. "
                        "It needs to be processed again."
                    )


                if temp_path.exists():
                    temp_path.unlink()


            else:

                # ------------------------------------------------
                # 3. REGISTER DOCUMENT
                # ------------------------------------------------

                document = register_document(
                    filename=safe_filename,
                    file_hash=file_hash,
                    content_hash=content_hash,
                    owner_id="user_001",
                )


                # ------------------------------------------------
                # SERVER-GENERATED STORAGE NAME
                # ------------------------------------------------

                storage_filename = (
                    f"{document['document_id']}"
                    f"{file_extension}"
                )

                final_path = (
                    upload_dir
                    / storage_filename
                )

                temp_path.replace(
                    final_path
                )

                set_document_storage_filename(
                    document["document_id"],
                    storage_filename,
                )


                # ------------------------------------------------
                # ATTACH TO WORKSPACE
                # ------------------------------------------------

                add_document_to_workspace(
                    current_workspace[
                        "workspace_id"
                    ],
                    document["document_id"],
                )

                st.success(
                    f"Added {safe_filename} to "
                    f"{current_workspace['name']}."
                )


                # ------------------------------------------------
                # 4. CHUNKING
                # ------------------------------------------------

                with st.spinner(
                    "Creating document chunks..."
                ):

                    chunks = chunk_pages(
                        pages
                    )

                st.success(
                    f"Created {len(chunks)} "
                    f"document chunks."
                )


                # ------------------------------------------------
                # 5. BGE-M3 EMBEDDINGS
                # ------------------------------------------------

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


                # ------------------------------------------------
                # 6. DOCUMENT-SPECIFIC FAISS INDEX
                # ------------------------------------------------

                with st.spinner(
                    "Building FAISS vector index..."
                ):

                    index = build_index(
                        embeddings
                    )

                    save_index(
                        index,
                        document["document_id"],
                    )

                    save_metadata(
                        chunks,
                        document["document_id"],
                    )

                st.success(
                    f"FAISS index created with "
                    f"{index.ntotal} vectors."
                )

                st.info(
                    "Document processing complete. "
                    "The document is now ready for retrieval."
                )


            st.session_state[
                "processed_uploads"
            ].add(signature)


        except Exception as error:

            if temp_path.exists():
                temp_path.unlink()

            st.error(
                f"Error processing "
                f"{safe_filename}: {error}"
            )


# ============================================================
# DOCUMENT EXPLORER
# ============================================================

registry = load_registry()

workspace_registry = load_workspace_registry()

current_workspace = find_workspace(
    workspace_id=st.session_state[
        "current_workspace_id"
    ],
    registry=workspace_registry,
)


workspace_document_ids = set(
    current_workspace.get(
        "document_ids",
        []
    )
    if current_workspace
    else []
)


processed_documents = [
    document
    for document in registry
    if document["document_id"]
    in workspace_document_ids
    and document_index_exists(
        document["document_id"]
    )
]


if processed_documents:

    st.markdown(
        f"### {current_workspace['name']} · Documents"
    )


    # Use document IDs as the internal selection
    # values so duplicate display filenames cannot collide.
    document_labels = {
        document["document_id"]:
            document["filename"]
        for document in processed_documents
    }


    valid_selected_ids = [
        document["document_id"]
        for document
        in st.session_state[
            "selected_documents"
        ]
        if document["document_id"]
        in document_labels
    ]


    if not valid_selected_ids:

        valid_selected_ids = list(
            document_labels.keys()
        )


    selected_ids = st.multiselect(
        "Documents to search",
        options=list(
            document_labels.keys()
        ),
        default=valid_selected_ids,
        format_func=lambda document_id:
            document_labels[document_id],
        help=(
            "Choose which documents in this "
            "workspace IntelliAssist should search."
        ),
    )


    selected_documents = [
        document
        for document in processed_documents
        if document["document_id"]
        in selected_ids
    ]


    st.session_state[
        "selected_documents"
    ] = selected_documents


    st.caption(
        f"{len(processed_documents)} document(s) "
        f"in this workspace · "
        f"{len(selected_documents)} selected for search"
    )


    for document in processed_documents:

        col1, col2, col3 = st.columns(
            [5, 2, 1]
        )

        with col1:

            st.write(
                f"📄 **{document['filename']}**"
            )

        with col2:

            st.caption(
                f"{document.get('access', 'private').capitalize()} · "
                f"{document['document_id']}"
            )

        with col3:

            if st.button(
                "Delete",
                key=f"delete_{document['document_id']}",
            ):

                delete_document(
                    document
                )

                remove_document_from_all_workspaces(
                    document["document_id"]
                )

                st.session_state[
                    "selected_documents"
                ] = [
                    item
                    for item
                    in st.session_state[
                        "selected_documents"
                    ]
                    if item["document_id"]
                    != document["document_id"]
                ]

                st.success(
                    f"Deleted {document['filename']}."
                )

                st.rerun()


else:

    st.info(
        f"No processed documents in "
        f"{current_workspace['name']}. "
        "Upload a PDF above to get started."
    )


# ============================================================
# CONVERSATION
# ============================================================

st.divider()

st.header(
    "💬 Ask Your Documents"
)

render_chat_history()


question = st.chat_input(
    "Ask a question about your selected documents..."
)


# ============================================================
# QUESTION ANSWERING
# ============================================================

if question:

    try:

        # ----------------------------------------------------
        # 1. GET SELECTED DOCUMENTS
        # ----------------------------------------------------

        selected_documents = st.session_state.get(
            "selected_documents",
            [],
        )


        if not selected_documents:

            st.warning(
                "Select at least one processed document "
                "before asking a question."
            )

            st.stop()


        # ----------------------------------------------------
        # SECURITY:
        # Enforce document access before retrieval.
        # ----------------------------------------------------

        current_user_id = "user_001"


        documents = [
            document
            for document in selected_documents
            if can_access_document(
                document,
                current_user_id,
            )
        ]


        if not documents:

            st.error(
                "You do not have access to "
                "the selected documents."
            )

            st.stop()


        # ----------------------------------------------------
        # 2. LOAD MODELS
        # ----------------------------------------------------

        embedding_model = load_embedding_model()

        reranker = load_reranker()


        # ----------------------------------------------------
        # 3. CREATE QUERY / SUB-QUERY LIST
        # ----------------------------------------------------
        #
        # A normal question produces one query.
        #
        # A multi-part question such as:
        #
        # "What is the context of FD001 and FD002
        #  and what columns are used?"
        #
        # is divided into smaller retrieval questions.
        #
        # Each part gets its own retrieval pass.
        # ----------------------------------------------------

        normalized_question = " ".join(
            question.strip().split()
        )


        sub_queries = [
            normalized_question
        ]


        split_parts = re.split(
            r"\s+(?:and|also|along with)\s+",
            normalized_question,
            flags=re.IGNORECASE,
        )


        if len(split_parts) > 1:

            candidate_sub_queries = []

            for part in split_parts:

                part = part.strip(
                    " ,;?.:"
                )

                if len(part.split()) >= 3:

                    candidate_sub_queries.append(
                        part
                    )


            if candidate_sub_queries:

                sub_queries = (
                    candidate_sub_queries
                )


        # Remove duplicate sub-queries
        # while preserving order.
        sub_queries = list(
            dict.fromkeys(
                sub_queries
            )
        )


        # ----------------------------------------------------
        # 4. FAISS RETRIEVAL FOR EACH SUB-QUERY
        # ----------------------------------------------------

        candidates = []


        for sub_query in sub_queries:

            query_embedding = (
                embedding_model.encode(
                    [sub_query],
                    normalize_embeddings=True,
                )
            )


            query_embedding = np.asarray(
                query_embedding,
                dtype="float32",
            )


            for document in documents:

                document_id = (
                    document["document_id"]
                )


                index = load_index(
                    document_id
                )


                chunks = load_metadata(
                    document_id
                )


                candidate_k = min(
                    10,
                    index.ntotal,
                )


                if candidate_k == 0:
                    continue


                scores, indices = index.search(
                    query_embedding,
                    candidate_k,
                )


                for score, index_position in zip(
                    scores[0],
                    indices[0],
                ):

                    if index_position < 0:
                        continue


                    chunk = chunks[
                        index_position
                    ].copy()


                    chunk["document_id"] = (
                        document_id
                    )


                    chunk["filename"] = (
                        document["filename"]
                    )


                    chunk["_chunk_index"] = (
                        int(index_position)
                    )


                    chunk["semantic_score"] = (
                        float(score)
                    )


                    # Remember which sub-query
                    # retrieved this chunk.
                    chunk["_retrieval_query"] = (
                        sub_query
                    )


                    candidates.append(
                        chunk
                    )


        # ----------------------------------------------------
        # NO RETRIEVAL RESULTS
        # ----------------------------------------------------

        if not candidates:

            st.warning(
                "No searchable evidence was found "
                "in the selected documents."
            )

            st.stop()


        # ----------------------------------------------------
        # 5. REMOVE DUPLICATE CANDIDATES
        # ----------------------------------------------------
        #
        # The same chunk may be retrieved by more
        # than one sub-query.
        #
        # Keep the version with the highest semantic score.
        # ----------------------------------------------------

        unique_candidates = {}


        for candidate in candidates:

            key = (
                candidate["document_id"],
                candidate["_chunk_index"],
            )


            existing = unique_candidates.get(
                key
            )


            if (
                existing is None
                or candidate["semantic_score"]
                > existing["semantic_score"]
            ):

                unique_candidates[key] = (
                    candidate
                )


        candidates = list(
            unique_candidates.values()
        )


        # ----------------------------------------------------
        # 6. RERANK CANDIDATES
        # ----------------------------------------------------
        #
        # Each candidate is reranked against
        # the sub-query that retrieved it.
        # ----------------------------------------------------

        pairs = [
            [
                candidate["_retrieval_query"],
                candidate["text"],
            ]
            for candidate in candidates
        ]


        reranker_scores = reranker.predict(
            pairs
        )


        reranked_results = []


        for candidate, score in zip(
            candidates,
            reranker_scores,
        ):

            result = candidate.copy()

            result["reranker_score"] = (
                float(score)
            )

            reranked_results.append(
                result
            )


        reranked_results.sort(
            key=lambda item:
                item["reranker_score"],
            reverse=True,
        )


        # ----------------------------------------------------
        # 7. EVIDENCE SELECTION
        # ----------------------------------------------------
        #
        # First guarantee that each sub-query
        # contributes evidence.
        #
        # Then fill remaining slots with the
        # strongest overall evidence.
        # ----------------------------------------------------

        MAX_EVIDENCE_RESULTS = 8

        evidence_results = []

        seen_keys = set()


        # ----------------------------------------------------
        # FIRST PASS:
        # Strongest result for every sub-query.
        # ----------------------------------------------------

        for sub_query in sub_queries:

            for result in reranked_results:

                if (
                    result["_retrieval_query"]
                    != sub_query
                ):
                    continue


                key = (
                    result["document_id"],
                    result["_chunk_index"],
                )


                if key in seen_keys:
                    continue


                seen_keys.add(key)

                evidence_results.append(
                    result
                )

                break


        # ----------------------------------------------------
        # SECOND PASS:
        # Fill remaining evidence slots.
        # ----------------------------------------------------

        for result in reranked_results:

            if (
                len(evidence_results)
                >= MAX_EVIDENCE_RESULTS
            ):
                break


            key = (
                result["document_id"],
                result["_chunk_index"],
            )


            if key in seen_keys:
                continue


            seen_keys.add(key)

            evidence_results.append(
                result
            )


        # ----------------------------------------------------
        # 8. EVIDENCE ASSESSMENT
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # This was missing in the previous file.
        # assessment was being used before creation.
        # ----------------------------------------------------

        assessment = assess_evidence(
            evidence_results
        )


        # ----------------------------------------------------
        # 9. BUILD EXPANDED CONTEXT
        # ----------------------------------------------------

        if (
            assessment["decision"]
            == "INSUFFICIENT_EVIDENCE"
        ):

            answer = (
                "I could not find enough reliable "
                "evidence in the selected documents "
                "to answer this question."
            )

        else:

            from app.rag import (
                generate_answer,
                is_summary_request,
            )


            # Summary/brief questions need more
            # surrounding context because the
            # relevant section may span several chunks.

            if is_summary_request(
                question
            ):

                context_radius = 2

            else:

                context_radius = 1


            expanded_context = []


            # Keep track of chunks already added
            # so overlapping context windows
            # do not duplicate text.

            added_chunks = set()


            # ------------------------------------------------
            # EXPAND AROUND EACH EVIDENCE RESULT
            # ------------------------------------------------

            for result in evidence_results:

                document_id = (
                    result["document_id"]
                )


                document_chunks = (
                    load_metadata(
                        document_id
                    )
                )


                # The FAISS index position identifies
                # the original chunk inside the document.

                center_index = result.get(
                    "chunk_index"
                )


                # Older metadata may not contain
                # chunk_index.
                #
                # In that case, use the chunk
                # position stored during retrieval.

                if center_index is None:

                    center_index = result.get(
                        "_chunk_index"
                    )


                if center_index is None:
                    continue


                start_index = max(
                    0,
                    center_index
                    - context_radius,
                )


                end_index = min(
                    len(document_chunks),
                    center_index
                    + context_radius
                    + 1,
                )


                for chunk_position in range(
                    start_index,
                    end_index,
                ):

                    chunk_key = (
                        document_id,
                        chunk_position,
                    )


                    if chunk_key in added_chunks:
                        continue


                    added_chunks.add(
                        chunk_key
                    )


                    chunk = document_chunks[
                        chunk_position
                    ]


                    expanded_context.append(
                        {
                            "document_id":
                                document_id,

                            "filename":
                                result["filename"],

                            "chunk_index":
                                chunk_position,

                            "page_number":
                                chunk.get(
                                    "page_number",
                                    "N/A",
                                ),

                            "text":
                                chunk["text"],
                        }
                    )


            # ------------------------------------------------
            # FALLBACK:
            # Use reranked evidence directly.
            # ------------------------------------------------

            if not expanded_context:

                expanded_context = [
                    {
                        "document_id":
                            result["document_id"],

                        "filename":
                            result["filename"],

                        "chunk_index":
                            result.get(
                                "_chunk_index",
                                "N/A",
                            ),

                        "page_number":
                            result.get(
                                "page_number",
                                "N/A",
                            ),

                        "text":
                            result["text"],
                    }

                    for result
                    in evidence_results
                ]


            # ------------------------------------------------
            # BUILD FINAL LLM CONTEXT
            # ------------------------------------------------

            context_parts = []


            for result in expanded_context:

                context_parts.append(
                    f"[Document: "
                    f"{result['filename']} | "
                    f"Page "
                    f"{result['page_number']} | "
                    f"Chunk "
                    f"{result['chunk_index']}]\n"
                    f"{result['text']}"
                )


            context = "\n\n".join(
                context_parts
            )


            # ------------------------------------------------
            # 10. GENERATE GROUNDED ANSWER
            # ------------------------------------------------

            with st.spinner(
                "Generating document-grounded answer..."
            ):

                answer = generate_answer(
                    query=question,
                    context=context,
                )


        # ----------------------------------------------------
        # 11. SAVE COMPLETE Q&A IN SESSION
        # ----------------------------------------------------

        st.session_state[
            "chat_history"
        ].append(
            {
                "question": question,
                "answer": answer,
                "assessment": assessment,
                "evidence": evidence_results,
            }
        )


        # Streamlit reruns after chat input.
        # The newly saved conversation will now
        # render together with all previous Q&A.

        st.rerun()


    except FileNotFoundError:

        st.warning(
            "A processed document is unavailable. "
            "Please upload or reprocess the document."
        )


    except Exception as error:

        st.error(
            f"Error during question answering: "
            f"{error}"
        )


# ============================================================
# EXPORT CURRENT SESSION
# ============================================================

if st.session_state["chat_history"]:

    st.divider()

    st.subheader(
        "📄 Export Conversation"
    )


    try:

        pdf_bytes = build_conversation_pdf(
            st.session_state[
                "chat_history"
            ]
        )


        st.download_button(
            label="Download conversation as PDF",
            data=pdf_bytes,
            file_name="intelliassist_conversation.pdf",
            mime="application/pdf",
        )


    except ImportError:

        st.warning(
            "PDF export requires ReportLab. "
            "Install it with: pip install reportlab"
        )