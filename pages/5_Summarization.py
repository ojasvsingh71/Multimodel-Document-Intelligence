"""
pages/5_Summarization.py
Generate various types of document summaries.
"""
import streamlit as st

st.set_page_config(
    page_title="Summarization — Document Intelligence",
    page_icon="📝",
    layout="wide",
)

from modules.database_manager import get_all_documents
from modules.summarizer import SummaryType, summarize_document
from modules.config import OLLAMA_MODEL

st.markdown("## 📝 Document Summarization")
st.markdown("Generate structured, detailed, or targeted summaries of your documents.")
st.markdown("---")

docs = get_all_documents()
completed_docs = [d for d in docs if d.get("status") == "completed"]

if not completed_docs:
    st.info("No processed documents available. Upload and index documents first.")
    st.stop()

# ── Controls ──────────────────────────────────────────────────────────────────
col_ctrl, col_result = st.columns([1, 2])

with col_ctrl:
    st.markdown("### ⚙️ Configuration")
    selected_filename = st.selectbox(
        "Select document",
        options=[d["filename"] for d in completed_docs],
    )
    selected_doc = next(
        (d for d in completed_docs if d["filename"] == selected_filename), None
    )

    summary_labels = {
        SummaryType.SHORT: "📄 Short Summary (3-5 sentences)",
        SummaryType.DETAILED: "📋 Detailed Summary",
        SummaryType.EXECUTIVE: "💼 Executive Summary",
        SummaryType.SECTION_WISE: "🗂️ Section-wise Summary",
        SummaryType.RESEARCH_PAPER: "🔬 Research Paper Analysis",
        SummaryType.TECHNICAL: "🔧 Technical Summary",
        SummaryType.KEY_FINDINGS: "💡 Key Findings",
        SummaryType.ADVANTAGES_LIMITATIONS: "⚖️ Advantages & Limitations",
        SummaryType.CONCLUSION: "✅ Conclusions",
    }

    selected_label = st.selectbox(
        "Summary type",
        options=list(summary_labels.values()),
    )
    selected_type = next(
        k for k, v in summary_labels.items() if v == selected_label
    )

    model = st.text_input("LLM Model", value=OLLAMA_MODEL)

    if selected_doc:
        st.markdown("**Document info:**")
        st.caption(f"Pages: {selected_doc.get('page_count') or 'N/A'}")
        st.caption(f"Tables: {selected_doc.get('table_count', 0)}")
        st.caption(f"Images: {selected_doc.get('image_count', 0)}")

    generate_btn = st.button("✨ Generate Summary", type="primary", use_container_width=True)

with col_result:
    st.markdown("### 📄 Generated Summary")

    if "last_summary" not in st.session_state:
        st.session_state.last_summary = ""
    if "last_summary_doc" not in st.session_state:
        st.session_state.last_summary_doc = ""

    if generate_btn and selected_doc:
        with st.spinner(f"Generating {selected_label} for **{selected_filename}**…"):
            from modules.llm_handler import is_ollama_running
            from modules.config import OLLAMA_BASE_URL
            if not is_ollama_running(OLLAMA_BASE_URL):
                st.error("❌ Ollama is not running. Start it with: `ollama serve`")
            else:
                summary = summarize_document(
                    doc_id=selected_doc["doc_id"],
                    summary_type=selected_type,
                    model=model,
                )
                st.session_state.last_summary = summary
                st.session_state.last_summary_doc = selected_filename

    if st.session_state.last_summary:
        doc_label = st.session_state.last_summary_doc
        st.markdown(f"*Summary for: **{doc_label}***")
        st.markdown(st.session_state.last_summary)
        st.markdown("---")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "⬇️ Download as Markdown",
                data=st.session_state.last_summary,
                file_name=f"{doc_label}_summary.md",
                mime="text/markdown",
            )
        with col_dl2:
            st.download_button(
                "⬇️ Download as Text",
                data=st.session_state.last_summary,
                file_name=f"{doc_label}_summary.txt",
                mime="text/plain",
            )
    else:
        st.info("Select a document and summary type, then click **Generate Summary**.")
