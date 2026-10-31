"""
pages/2_Upload_Documents.py
Upload, validate, process, and index documents.
"""
import time
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Upload Documents — Document Intelligence",
    page_icon="📤",
    layout="wide",
)

from modules.config import MAX_FILE_SIZE_MB, SUPPORTED_EXTENSIONS
from modules.database_manager import get_doc_by_hash, compute_file_hash
from modules.document_processor import process_document, ValidationError
from modules.rag_pipeline import index_document

st.markdown("## 📤 Upload Documents")
st.markdown("Upload one or more files to extract, analyse, and index them for AI-powered search.")
st.markdown("---")

# ── Supported formats info ────────────────────────────────────────────────────
with st.expander("ℹ️ Supported file formats & limits", expanded=False):
    st.markdown(f"""
    | Format | Extension | Notes |
    |--------|-----------|-------|
    | PDF | `.pdf` | Text, tables, images, scanned (OCR) |
    | Word | `.docx` | Headings, paragraphs, tables |
    | PowerPoint | `.pptx` | Slides, text, tables |
    | Images | `.png .jpg .jpeg .tiff` | OCR text extraction |

    **Max file size:** {MAX_FILE_SIZE_MB} MB per file.
    """)

# ── Upload widget ─────────────────────────────────────────────────────────────
col_up, col_opts = st.columns([2, 1])

with col_opts:
    st.markdown("#### ⚙️ Processing Options")
    run_vision = st.checkbox(
        "Enable image description (VLM)",
        value=False,
        help="Send extracted images to the vision model for AI descriptions. Requires a VLM like llava:7b installed in Ollama. Can be slow.",
    )
    from modules.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
    chunk_size = st.number_input("Chunk size (chars)", 200, 4000, CHUNK_SIZE, 100)
    chunk_overlap = st.number_input("Chunk overlap (chars)", 0, 500, CHUNK_OVERLAP, 20)

with col_up:
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
        help="Drag and drop or click to browse. Multiple files allowed.",
    )

if not uploaded_files:
    st.info("👆 Select one or more files above to begin.")
    st.stop()

st.markdown(f"**{len(uploaded_files)} file(s) selected**")

# Preview table
preview_data = []
for f in uploaded_files:
    size_mb = len(f.getvalue()) / (1024 * 1024)
    fhash = compute_file_hash(f.getvalue())
    existing = get_doc_by_hash(fhash)
    dup = "⚠️ Already uploaded" if existing else "✅ New"
    preview_data.append({
        "Filename": f.name,
        "Size": f"{size_mb:.2f} MB",
        "Type": Path(f.name).suffix.upper(),
        "Status": dup,
    })

import pandas as pd
st.dataframe(pd.DataFrame(preview_data), use_container_width=True, hide_index=True)

if st.button("🚀 Process & Index All Files", type="primary", use_container_width=False):
    progress_bar = st.progress(0, text="Starting…")
    status_container = st.empty()
    results_log = []
    total = len(uploaded_files)

    for idx, f in enumerate(uploaded_files, 1):
        fname = f.name
        file_bytes = f.getvalue()
        progress_bar.progress(idx / (total * 2), text=f"Processing {fname} ({idx}/{total})…")
        status_container.info(f"⚙️ Processing **{fname}**…")

        t0 = time.perf_counter()

        # Process document
        result = process_document(fname, file_bytes)

        if result.status in ("failed",) and result.error_message.startswith("Duplicate"):
            results_log.append({
                "file": fname,
                "status": "duplicate",
                "detail": result.error_message,
            })
            status_container.warning(f"⚠️ **{fname}** — duplicate, skipped.")
        elif result.status == "failed":
            results_log.append({
                "file": fname,
                "status": "failed",
                "detail": result.error_message[:300],
            })
            status_container.error(f"❌ **{fname}** — processing failed.")
        else:
            # Index
            progress_bar.progress(
                (idx - 0.5) / total, text=f"Indexing {fname}…"
            )
            n_chunks, err = index_document(
                result,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
                run_vision=run_vision,
            )
            elapsed = time.perf_counter() - t0
            if err:
                results_log.append({
                    "file": fname,
                    "status": "index_error",
                    "detail": err,
                })
                status_container.warning(f"⚠️ **{fname}** — processed but indexing failed: {err}")
            else:
                results_log.append({
                    "file": fname,
                    "status": "success",
                    "detail": (
                        f"Pages: {result.page_count} | "
                        f"Tables: {len(result.tables)} | "
                        f"Images: {len(result.images)} | "
                        f"Chunks: {n_chunks} | "
                        f"Time: {elapsed:.1f}s"
                    ),
                })

        progress_bar.progress(idx / total, text=f"Done {idx}/{total}")

    status_container.empty()
    progress_bar.empty()

    st.markdown("### 📋 Processing Summary")
    for log in results_log:
        s = log["status"]
        icon = {"success": "✅", "failed": "❌", "duplicate": "⚠️", "index_error": "⚠️"}.get(s, "ℹ️")
        st.markdown(f"{icon} **{log['file']}** — {log['detail']}")

    success_count = sum(1 for l in results_log if l["status"] == "success")
    if success_count:
        st.success(
            f"✅ {success_count}/{total} document(s) processed and indexed successfully. "
            "Go to **AI Document Assistant** to start asking questions!"
        )
