"""
pages/3_Document_Library.py
Manage uploaded documents — view details, re-index, delete.
"""
import streamlit as st

st.set_page_config(
    page_title="Document Library — Document Intelligence",
    page_icon="📚",
    layout="wide",
)

from modules.database_manager import get_all_documents, delete_document
from modules.document_processor import delete_document_files
from modules.vector_database import delete_document_vectors

st.markdown("## 📚 Document Library")
st.markdown("Browse and manage all uploaded documents.")
st.markdown("---")

docs = get_all_documents()

if not docs:
    st.info("No documents uploaded yet. Go to **Upload Documents** to add files.")
    st.stop()

# ── Summary strip ─────────────────────────────────────────────────────────────
total = len(docs)
completed = sum(1 for d in docs if d.get("status") == "completed")
failed = sum(1 for d in docs if d.get("status") == "failed")

c1, c2, c3 = st.columns(3)
c1.metric("Total Documents", total)
c2.metric("✅ Completed", completed)
c3.metric("❌ Failed", failed)

st.markdown("---")

# ── Filter ───────────────────────────────────────────────────────────────────
filter_status = st.selectbox(
    "Filter by status",
    ["All", "completed", "processing", "failed", "pending"],
    index=0,
)
if filter_status != "All":
    docs = [d for d in docs if d.get("status") == filter_status]

# ── Document cards ────────────────────────────────────────────────────────────
for doc in docs:
    doc_id = doc["doc_id"]
    status = doc.get("status", "unknown")
    badge_color = {
        "completed": "#3fb950",
        "processing": "#d29922",
        "failed": "#f85149",
        "pending": "#8b949e",
    }.get(status, "#8b949e")

    with st.expander(
        f"📄 **{doc['filename']}** — {doc['file_type'].upper()} "
        f"| {doc.get('file_size', 0) // 1024} KB | {status.upper()}",
        expanded=False,
    ):
        col_info, col_stats, col_action = st.columns([2, 2, 1])

        with col_info:
            st.markdown("**Document Information**")
            st.markdown(f"- **ID:** `{doc_id}`")
            st.markdown(f"- **Filename:** {doc['filename']}")
            st.markdown(f"- **Type:** {doc['file_type'].upper()}")
            size_kb = doc.get("file_size", 0) // 1024
            st.markdown(f"- **Size:** {size_kb:,} KB")
            ts = doc.get("upload_ts", "")[:19].replace("T", " ")
            st.markdown(f"- **Uploaded:** {ts}")

        with col_stats:
            st.markdown("**Extraction Statistics**")
            st.markdown(f"- **Pages:** {doc.get('page_count') or 'N/A'}")
            st.markdown(f"- **Tables:** {doc.get('table_count', 0)}")
            st.markdown(f"- **Images:** {doc.get('image_count', 0)}")
            st.markdown(f"- **Indexed Chunks:** {doc.get('chunk_count', 0)}")
            st.markdown(
                f"- **Status:** <span style='color:{badge_color}'>{status}</span>",
                unsafe_allow_html=True,
            )
            if doc.get("error_message"):
                st.error(f"Error: {doc['error_message'][:200]}")

        with col_action:
            st.markdown("**Actions**")

            # Re-index button
            if st.button(
                "🔄 Re-Index",
                key=f"reidx_{doc_id}",
                help="Re-chunk and generate vector embeddings for this document.",
            ):
                with st.spinner(f"Re-indexing {doc['filename']}…"):
                    from modules.document_processor import process_document
                    from modules.rag_pipeline import index_document
                    from pathlib import Path
                    stored_f = Path(doc.get("stored_path", ""))
                    if stored_f.exists():
                        res = process_document(doc["filename"], stored_f.read_bytes())
                        n, err = index_document(res)
                        if err:
                            st.error(f"Re-indexing failed: {err}")
                        else:
                            st.success(f"Indexed {n} chunks!")
                            st.rerun()
                    else:
                        st.error("Source document file not found on disk.")

            # View extracted markdown
            from modules.config import PROCESSED_DIR
            md_file = PROCESSED_DIR / doc_id / "extracted.md"
            if md_file.exists():
                md_content = md_file.read_text(encoding="utf-8")
                st.download_button(
                    "⬇️ Download Markdown",
                    data=md_content,
                    file_name=f"{doc['filename']}_extracted.md",
                    mime="text/markdown",
                    key=f"dl_md_{doc_id}",
                )

            # Delete button
            if st.button(
                "🗑️ Delete Document",
                key=f"del_{doc_id}",
                help="Removes all files, embeddings, and metadata for this document.",
            ):
                st.session_state[f"confirm_delete_{doc_id}"] = True

            if st.session_state.get(f"confirm_delete_{doc_id}"):
                st.warning(f"⚠️ Delete **{doc['filename']}** and all its data?")
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("Yes, delete", key=f"yes_{doc_id}", type="primary"):
                        # Delete vectors
                        delete_document_vectors(doc_id)
                        # Delete files
                        delete_document_files(
                            doc_id=doc_id,
                            stored_path=doc.get("stored_path", ""),
                            processed_dir=doc.get("processed_dir", ""),
                        )
                        # Delete DB record
                        delete_document(doc_id)
                        st.success(f"Deleted {doc['filename']}.")
                        st.rerun()
                with col_no:
                    if st.button("Cancel", key=f"no_{doc_id}"):
                        st.session_state[f"confirm_delete_{doc_id}"] = False
                        st.rerun()
