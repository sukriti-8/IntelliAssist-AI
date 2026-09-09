from pathlib import Path

import streamlit as st

from app.access_control import can_access_document
from app.document_registery import load_registry
from app.summarizer import hierarchical_summarize
from app.vector_store import document_index_exists, load_metadata
from app.workspace_registry import load_registry as load_workspace_registry


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURRENT_USER_ID = "user_001"


st.set_page_config(
    page_title="Summarize · IntelliAssist AI",
    page_icon="I",
    layout="wide",
)


st.title("Document Summaries")
st.caption(
    "Generate concise, document-grounded summaries from your workspace documents."
)


workspace_registry = load_workspace_registry()
document_registry = load_registry()

if not workspace_registry:
    st.warning(
        "No workspace is available yet. Return to IntelliAssist AI "
        "and create or upload a document first."
    )
    st.stop()


workspace_names = [
    workspace["name"]
    for workspace in workspace_registry
]

current_workspace_id = st.session_state.get(
    "current_workspace_id"
)

current_workspace = next(
    (
        workspace
        for workspace in workspace_registry
        if workspace["workspace_id"] == current_workspace_id
    ),
    workspace_registry[0],
)


selected_workspace_name = st.selectbox(
    "Workspace",
    options=workspace_names,
    index=workspace_names.index(
        current_workspace["name"]
    ),
)


current_workspace = next(
    workspace
    for workspace in workspace_registry
    if workspace["name"] == selected_workspace_name
)

st.session_state["current_workspace_id"] = (
    current_workspace["workspace_id"]
)


workspace_document_ids = set(
    current_workspace.get("document_ids", [])
)


available_documents = [
    document
    for document in document_registry
    if document.get("document_id")
    in workspace_document_ids
    and document_index_exists(
        document["document_id"]
    )
    and can_access_document(
        document,
        CURRENT_USER_ID,
    )
]


if not available_documents:
    st.info(
        "No accessible processed documents are "
        "available in this workspace."
    )
    st.stop()


document_labels = {
    document["document_id"]:
        document["filename"]
    for document in available_documents
}


selected_ids = st.multiselect(
    "Documents to summarize",
    options=list(document_labels.keys()),
    format_func=lambda document_id:
        document_labels[document_id],
    default=list(document_labels.keys()),
    help=(
        "Select one or more processed documents. "
        "Each document receives its own summary."
    ),
)


if not selected_ids:
    st.info(
        "Select at least one document to generate "
        "a summary."
    )
    st.stop()


st.caption(
    f"{len(selected_ids)} document(s) selected · "
    "Summaries are generated from the stored "
    "document chunks."
)


if st.button(
    "Generate Summaries",
    type="primary",
):

    st.session_state["document_summaries"] = {}
    st.session_state["summary_error"] = None

    for document_id in selected_ids:

        document = next(
            item
            for item in available_documents
            if item["document_id"] == document_id
        )

        try:

            chunks = load_metadata(
                document_id
            )

            if not chunks:
                raise ValueError(
                    "No processed chunks are "
                    "available for this document."
                )

            with st.spinner(
                f"Generating summary for "
                f"{document['filename']}..."
            ):

                summary = hierarchical_summarize(
                    chunks,
                    batch_size=4,
                    delay_seconds=12.0,
                )

            st.session_state[
                "document_summaries"
            ][document_id] = summary

        except Exception as error:

            st.session_state[
                "summary_error"
            ] = (
                f"Could not summarize "
                f"{document['filename']}: "
                f"{error}"
            )

if st.session_state.get(
    "summary_error"
):

    st.error(
        "Summary generation failed for "
        "one or more documents."
    )
    st.caption(
        st.session_state["summary_error"]
    )

summaries = st.session_state.get(
    "document_summaries",
    {},
)

if summaries:
    st.divider()
    st.subheader(
        "Generated Summaries"
    )
    for document_id, summary in summaries.items():
        filename = document_labels.get(
            document_id,
            document_id,
        )
        with st.container(
            border=True
        ):
            st.markdown(
                f"### {filename}"
            )
            st.write(summary)