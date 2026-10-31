"""
pages/8_Document_Comparison.py
Compare two or more documents across research dimensions.
"""
import streamlit as st

st.set_page_config(
    page_title="Document Comparison — Document Intelligence",
    page_icon="🔀",
    layout="wide",
)

import pandas as pd

from modules.database_manager import get_all_documents
from modules.document_comparator import compare_documents, answer_comparison_question
from modules.config import OLLAMA_MODEL

st.markdown("## 🔀 Document Comparison")
st.markdown(
    "Compare two or more documents side by side across key research dimensions. "
    "Results are grounded in the actual document content."
)
st.markdown("---")

docs = get_all_documents()
completed_docs = [d for d in docs if d.get("status") == "completed"]

if len(completed_docs) < 2:
    st.info(
        "You need at least **2 indexed documents** to use this feature. "
        "Upload and process more documents first."
    )
    st.stop()

# ── Configuration ─────────────────────────────────────────────────────────────
col_cfg, col_result = st.columns([1, 2])

with col_cfg:
    st.markdown("### ⚙️ Comparison Settings")

    selected_filenames = st.multiselect(
        "Select documents to compare (2–5)",
        options=[d["filename"] for d in completed_docs],
        default=[d["filename"] for d in completed_docs[:2]],
        max_selections=5,
    )

    CUSTOM_DIMENSIONS = [
        "Research Objective",
        "Problem Statement",
        "Proposed Methodology",
        "Model / Architecture",
        "Dataset",
        "Evaluation Metrics",
        "Results",
        "Limitations",
        "Conclusion",
    ]

    selected_dimensions = st.multiselect(
        "Comparison dimensions",
        options=CUSTOM_DIMENSIONS,
        default=CUSTOM_DIMENSIONS,
    )

    include_narrative = st.checkbox("Generate narrative analysis", value=True)
    model = st.text_input("LLM Model", value=OLLAMA_MODEL)

    compare_btn = st.button(
        "🔀 Compare Documents",
        type="primary",
        use_container_width=True,
        disabled=len(selected_filenames) < 2,
    )

with col_result:
    st.markdown("### 📊 Comparison Results")

    if "comparison_result" not in st.session_state:
        st.session_state.comparison_result = None

    if compare_btn and len(selected_filenames) >= 2:
        selected_docs = [
            {"doc_id": d["doc_id"], "filename": d["filename"]}
            for d in completed_docs
            if d["filename"] in selected_filenames
        ]

        with st.spinner(
            f"Comparing {len(selected_docs)} documents across {len(selected_dimensions)} dimensions… "
            "This may take a few minutes."
        ):
            from modules.llm_handler import is_ollama_running
            from modules.config import OLLAMA_BASE_URL
            if not is_ollama_running(OLLAMA_BASE_URL):
                st.error("❌ Ollama is not running. Start it with: `ollama serve`")
            else:
                result = compare_documents(
                    documents=selected_docs,
                    dimensions=selected_dimensions,
                    model=model,
                    include_narrative=include_narrative,
                )
                st.session_state.comparison_result = result

    if st.session_state.comparison_result:
        result = st.session_state.comparison_result

        if result.get("error"):
            st.error(f"Comparison failed: {result['error']}")
        else:
            table_data = result["table"]
            doc_names = [d["filename"] for d in result["documents"]]

            # Build comparison dataframe
            rows = []
            for dim, values in table_data.items():
                row = {"Dimension": dim}
                for fname in doc_names:
                    row[fname[:25]] = values.get(fname, "Not specified")
                rows.append(row)

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Download comparison table
            csv_data = df.to_csv(index=False)
            st.download_button(
                "⬇️ Export Comparison as CSV",
                data=csv_data,
                file_name="document_comparison.csv",
                mime="text/csv",
            )

            if result.get("narrative"):
                st.markdown("---")
                st.markdown("### 📝 Comparative Analysis")
                st.markdown(result["narrative"])

st.markdown("---")

# ── Free-form comparison questions ────────────────────────────────────────────
st.markdown("### 💬 Ask a Comparison Question")
st.markdown(
    "Ask a free-form question that spans multiple documents. "
    "The system will retrieve evidence from all selected documents."
)

cmp_docs_for_qa = [
    {"doc_id": d["doc_id"], "filename": d["filename"]}
    for d in completed_docs
    if d["filename"] in (selected_filenames if selected_filenames else [])
]

qa_question = st.text_input(
    "Comparison question",
    placeholder="e.g. Which paper proposes a more effective approach and why?",
)
if st.button("Ask", key="cmp_ask") and qa_question and cmp_docs_for_qa:
    doc_ids = [d["doc_id"] for d in cmp_docs_for_qa]
    with st.spinner("Generating comparative answer…"):
        answer = answer_comparison_question(
            question=qa_question,
            doc_ids=doc_ids,
            model=model,
        )
    st.markdown("**Answer:**")
    st.markdown(answer)
