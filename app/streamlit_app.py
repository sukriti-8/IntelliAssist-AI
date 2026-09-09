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
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from app.confidence import assess_evidence
from app.access_control import can_access_document
from app.sentiment_intent import analyze_sentiment_intent
import os
from dotenv import load_dotenv
from google import genai


load_dotenv(PROJECT_ROOT / ".env")
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set.")
client = genai.Client(api_key=api_key)



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
    initial_sidebar_state="expanded",
)

# ============================================================
# UI THEME
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --ia-bg: #111211;
        --ia-panel: #171817;
        --ia-panel-2: #1c1d1c;
        --ia-border: #353735;
        --ia-text: #f1f1ee;
        --ia-muted: #9b9d99;
        --ia-accent: #2f8df6;
        --ia-success: #19b94b;
        --ia-warning: #7a3b00;
    }
    .stApp { background: var(--ia-bg); color: var(--ia-text); }
    [data-testid="stHeader"] { background: rgba(17,18,17,0.92); }
    [data-testid="stSidebar"] {
        background: #151615;
        border-right: 1px solid var(--ia-border);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }
    .ia-brand {
        font-size: 1.15rem; font-weight: 700; letter-spacing: -0.02em;
        margin: 0 0 1rem 0; color: var(--ia-text);
    }
    .ia-sidebar-label {
        color: var(--ia-muted); font-size: 0.74rem; text-transform: uppercase;
        letter-spacing: 0.08em; margin: 1rem 0 0.35rem 0;
    }
    .ia-divider { height: 1px; background: var(--ia-border); margin: 0.7rem 0; }
    .ia-topbar {
        display:flex; align-items:center; justify-content:space-between;
        margin-bottom: 1rem;
    }
    .ia-title { font-size: 1.12rem; font-weight: 700; }
    .ia-folder {
        border: 1px solid var(--ia-border); border-radius: 9px;
        padding: 0.45rem 0.7rem; color: #c6c8c4; font-size: 0.85rem;
        background: var(--ia-panel);
    }
    .ia-upload-note { color: var(--ia-muted); font-size: 0.78rem; margin-top: -0.45rem; }
    .ia-doc-count { color: var(--ia-muted); font-size: 0.82rem; margin: 0.55rem 0; }
    .ia-doc-row {
        border-top: 1px solid var(--ia-border); padding: 0.55rem 0;
    }
    .ia-ready {
        display:inline-block; background:#073c16; color:#28d455;
        border-radius:999px; padding:0.14rem 0.52rem; font-size:0.72rem;
    }
    .ia-duplicate {
        background:#4a2400; border:1px solid #6b3500; border-radius:9px;
        padding:0.55rem 0.7rem; margin:0.45rem 0 0.8rem 0;
    }
    .ia-question-label, .ia-answer-label {
        color: var(--ia-muted); font-size: 0.78rem; margin-top: 0.8rem;
    }
    .ia-question { font-weight: 650; font-size: 0.96rem; margin-top: 0.15rem; }
    .ia-answer {
        font-size: 0.94rem; line-height: 1.65; margin-top: 0.15rem;
    }
    div[data-testid="stChatInput"] {
        background: transparent;
    }
    div[data-testid="stChatInput"] textarea {
        background: #171817 !important; border: 1px solid var(--ia-border) !important;
        color: var(--ia-text) !important; border-radius: 9px !important;
    }
    .stButton > button, .stDownloadButton > button { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_sidebar(workspace_registry, current_workspace):
    """Render folder/workspace navigation and lightweight recent chats."""
    with st.sidebar:
        st.markdown('<div class="ia-brand">IntelliAssist</div>', unsafe_allow_html=True)
        st.markdown('<div class="ia-sidebar-label">Folders</div>', unsafe_allow_html=True)

        for workspace in workspace_registry:
            active = workspace["workspace_id"] == current_workspace["workspace_id"]
            label = f"📁  {workspace['name']}"
            if st.button(
                label,
                key=f"folder_{workspace['workspace_id']}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                if not active:
                    st.session_state["current_workspace_id"] = workspace["workspace_id"]
                    st.session_state["selected_documents"] = []
                    st.session_state["chat_history"] = []
                    st.rerun()

        st.markdown('<div class="ia-divider"></div>', unsafe_allow_html=True)

        if st.button("＋ New folder", key="new_folder", use_container_width=True):
            st.session_state["show_new_folder"] = True

        if st.session_state.get("show_new_folder"):
            with st.form("new_folder_form", clear_on_submit=True):
                folder_name = st.text_input("Folder name", placeholder="e.g. DBMS")
                c1, c2 = st.columns(2)
                create = c1.form_submit_button("Create", type="primary")
                cancel = c2.form_submit_button("Cancel")
                if cancel:
                    st.session_state["show_new_folder"] = False
                    st.rerun()
                if create:
                    if not folder_name.strip():
                        st.warning("Enter a folder name.")
                    else:
                        create_workspace(
                            name=folder_name.strip(),
                            owner_id=current_profile_owner_id(),
                            registry=workspace_registry,
                        )
                        st.session_state["show_new_folder"] = False
                        st.session_state["current_workspace_id"] = load_workspace_registry()[-1]["workspace_id"]
                        st.session_state["selected_documents"] = []
                        st.rerun()

        if st.session_state.get("chat_history"):
            st.markdown('<div class="ia-sidebar-label">Recent questions</div>', unsafe_allow_html=True)
            for index, item in enumerate(st.session_state["chat_history"][-5:][::-1], start=1):
                question = item["question"].strip()
                short = question if len(question) <= 34 else question[:31] + "..."
                st.caption(f"💬 {short}")

        st.markdown('<div class="ia-divider"></div>', unsafe_allow_html=True)
        st.caption("Current folder")
        st.caption(f"📁 {current_workspace['name']}")




# ============================================================
# SESSION STATE
# ============================================================

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
if "duplicate_notice" not in st.session_state:
    st.session_state["duplicate_notice"] = None


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


BASE_USER_ID = "user_001"
PROFILE_OWNER_IDS = {
    "Student": BASE_USER_ID,
    "Business": f"{BASE_USER_ID}_business",
}


def current_profile_owner_id():
    """Return the storage namespace used by the active profile."""
    profile = st.session_state.get("user_profile")
    return PROFILE_OWNER_IDS.get(profile, BASE_USER_ID)


def workspace_belongs_to_profile(workspace, profile=None):
    """Check whether a workspace belongs to the active profile."""
    profile = profile or st.session_state.get("user_profile")
    owner_id = PROFILE_OWNER_IDS.get(profile, BASE_USER_ID)
    return workspace.get("owner_id") == owner_id


def document_belongs_to_profile(document_id, profile=None, workspace_registry=None):
    """Check whether a document is already attached to the profile namespace."""
    workspace_registry = (
        load_workspace_registry()
        if workspace_registry is None
        else workspace_registry
    )

    return any(
        workspace_belongs_to_profile(workspace, profile)
        and document_id in workspace.get("document_ids", [])
        for workspace in workspace_registry
    )


def ensure_profile_workspace():
    """Create/select a profile-specific General workspace.

    Existing legacy workspaces/documents are treated as Student data so the
    current Student setup is preserved. Business gets a separate namespace.
    """
    workspace_registry = load_workspace_registry()
    document_registry = load_registry()
    profile = st.session_state.get("user_profile", "Student")
    owner_id = PROFILE_OWNER_IDS.get(profile, BASE_USER_ID)

    profile_workspaces = [
        workspace
        for workspace in workspace_registry
        if workspace.get("owner_id") == owner_id
    ]

    # Legacy workspaces were created with user_001, so they remain Student
    # workspaces. Only Student receives the one-time legacy migration.
    if profile == "Student" and not profile_workspaces:
        legacy_workspaces = [
            workspace
            for workspace in workspace_registry
            if workspace.get("owner_id") == BASE_USER_ID
        ]
        profile_workspaces = legacy_workspaces

    if not profile_workspaces:
        workspace = create_workspace(
            name="General",
            owner_id=owner_id,
            registry=workspace_registry,
        )
        workspace_registry = load_workspace_registry()
        profile_workspaces = [workspace]

    workspace = next(
        (
            item
            for item in profile_workspaces
            if item.get("workspace_id") == st.session_state.get("current_workspace_id")
        ),
        profile_workspaces[0],
    )

    # One-time compatibility migration for the existing Student setup:
    # documents not referenced by any workspace are placed in Student General.
    if profile == "Student":
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
    st.write("Select how you plan to use IntelliAssist.")
    profile = st.radio("Profile", options=["Student", "Business"], horizontal=True)
    if st.button("Continue", type="primary"):
        st.session_state["user_profile"] = profile
        st.rerun()
    st.stop()


# HEADER
workspace_registry = load_workspace_registry()
current_workspace = ensure_profile_workspace()
workspace_registry = load_workspace_registry()

# Only show folders belonging to the active profile.
workspace_registry = [
    workspace
    for workspace in workspace_registry
    if workspace_belongs_to_profile(workspace)
]

if current_workspace["workspace_id"] not in {
    workspace["workspace_id"] for workspace in workspace_registry
}:
    current_workspace = workspace_registry[0]

current_workspace = find_workspace(
    workspace_id=current_workspace["workspace_id"],
    registry=workspace_registry,
)
st.session_state["current_workspace_id"] = current_workspace["workspace_id"]
render_sidebar(workspace_registry, current_workspace)

st.markdown(
    f'<div class="ia-topbar"><div class="ia-title">IntelliAssist</div>'
    f'<div class="ia-folder">📁 {current_workspace["name"]}</div></div>',
    unsafe_allow_html=True,
)

# WORKSPACE SETUP
current_workspace = find_workspace(
    workspace_id=st.session_state["current_workspace_id"],
    registry=load_workspace_registry(),
)

st.markdown("### Upload documents")
uploaded_files = st.file_uploader(
    "Upload document",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
    label_visibility="collapsed",
    help="Upload PDF, TXT, or DOCX documents into the selected folder.",
)
st.markdown('<div class="ia-upload-note">PDF, TXT or DOCX · up to 25MB</div>', unsafe_allow_html=True)


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

            # Duplicate reuse is profile-scoped. A Student document must not
            # automatically appear in or be reused by Business, and vice versa.
            if duplicate and not document_belongs_to_profile(
                duplicate["document"]["document_id"],
                workspace_registry=load_workspace_registry(),
            ):
                duplicate = None


            if duplicate:

                existing_document = (
                    duplicate["document"]
                )

                existing_document_id = (
                    existing_document["document_id"]
                )

                st.session_state["duplicate_notice"] = existing_document["filename"]

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

                st.session_state["duplicate_notice"] = None

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
    workspace_id=st.session_state["current_workspace_id"],
    registry=workspace_registry,
)

workspace_document_ids = set(
    current_workspace.get("document_ids", []) if current_workspace else []
)

processed_documents = [
    document
    for document in registry
    if document["document_id"] in workspace_document_ids
    and document_index_exists(document["document_id"])
]

selected_ids = set(
    document.get("document_id")
    for document in st.session_state.get("selected_documents", [])
)

valid_ids = {document["document_id"] for document in processed_documents}
selected_ids &= valid_ids

if not selected_ids and processed_documents:
    selected_ids = valid_ids

if processed_documents:
    st.markdown(
        f'<div class="ia-doc-count">{len(processed_documents)} document(s) · {len(selected_ids)} selected</div>',
        unsafe_allow_html=True,
    )

    for document in processed_documents:
        doc_id = document["document_id"]
        c1, c2, c3 = st.columns([0.45, 7.4, 1.15])
        with c1:
            checked = st.checkbox(
                "",
                value=doc_id in selected_ids,
                key=f"select_{doc_id}",
                label_visibility="collapsed",
            )
        with c2:
            st.markdown(
                f'📄 <strong>{document["filename"]}</strong> &nbsp; <span class="ia-ready">Ready</span>',
                unsafe_allow_html=True,
            )
        with c3:
            if st.button("🗑", key=f"delete_{doc_id}", help="Delete document"):
                delete_document(document)
                remove_document_from_all_workspaces(doc_id)
                st.session_state["selected_documents"] = [
                    item for item in st.session_state["selected_documents"]
                    if item["document_id"] != doc_id
                ]
                st.rerun()
        if checked:
            selected_ids.add(doc_id)
        else:
            selected_ids.discard(doc_id)

    st.session_state["selected_documents"] = [
        document for document in processed_documents
        if document["document_id"] in selected_ids
    ]
else:
    st.markdown(
        '<div class="ia-doc-count">No documents in this folder yet.</div>',
        unsafe_allow_html=True,
    )

# Show a compact duplicate notice when the upload flow detects one.
if st.session_state.get("duplicate_notice"):
    duplicate_name = st.session_state["duplicate_notice"]
    st.markdown(
        f'<div class="ia-duplicate">📋 <strong>{duplicate_name}</strong> looks like a file you already have.</div>',
        unsafe_allow_html=True,
    )

# ============================================================
# CONVERSATION
# ============================================================

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
    with st.expander("Export conversation", expanded=False):
        try:
            pdf_bytes = build_conversation_pdf(st.session_state["chat_history"])
            st.download_button(
                label="Download conversation as PDF",
                data=pdf_bytes,
                file_name="intelliassist_conversation.pdf",
                mime="application/pdf",
            )
        except ImportError:
            st.warning("PDF export requires ReportLab.")


# ============================================================
# BUSINESS SENTIMENT + INTENT ANALYSIS
# ============================================================

if st.session_state.get("user_profile") == "Business":
    with st.expander("Business text analysis", expanded=False):
        analysis_text = st.text_area(
            "Enter text to analyze",
            placeholder="Example: I am disappointed with the service and want my money refunded.",
            height=120,
        )
        if st.button("Analyze Sentiment & Intent", type="primary"):
            if not analysis_text.strip():
                st.warning("Please enter some text to analyze.")
            else:
                try:
                    with st.spinner("Analyzing text..."):
                        result = analyze_sentiment_intent(analysis_text, client)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Sentiment", result["sentiment"])
                        st.caption(f"Confidence: {result['sentiment_confidence']:.2f}")
                    with col2:
                        st.metric("Intent", result["intent"])
                        st.caption(f"Confidence: {result['intent_confidence']:.2f}")
                    st.write(f"**Reason:** {result['reason']}")
                except Exception as error:
                    st.error("Business analysis is temporarily unavailable.")
                    st.caption(f"Technical detail: {error}")
