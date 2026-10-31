"""
pages/9_Evaluation.py
Evaluation dashboard for measuring RAG system performance.
"""
import streamlit as st

st.set_page_config(
    page_title="Evaluation — Document Intelligence",
    page_icon="🧪",
    layout="wide",
)

import pandas as pd
import plotly.graph_objects as go

from modules.evaluation import (
    load_dataset,
    run_evaluation,
    aggregate_metrics,
    append_to_dataset,
    results_to_csv,
    save_results,
    ensure_dataset_exists,
)
from modules.database_manager import get_all_documents
from modules.config import TOP_K, OLLAMA_MODEL

st.markdown("## 🧪 Evaluation Dashboard")
st.markdown(
    "Measure RAG system performance using your evaluation dataset. "
    "All metrics are computed from actual system runs — no fabricated results."
)
st.markdown("---")

ensure_dataset_exists()
dataset = load_dataset()

# ── Dataset viewer ────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 Dataset", "▶️ Run Evaluation", "📈 Results"])

with tab1:
    st.markdown("### Evaluation Dataset")
    st.caption(
        "⚠️ Example entries marked as PLACEHOLDER are illustrative. "
        "Add real question-answer pairs verified against your documents."
    )

    if dataset:
        rows = []
        for item in dataset:
            rows.append({
                "Question": item.question,
                "Expected Answer": item.expected_answer[:80] + ("…" if len(item.expected_answer) > 80 else ""),
                "Expected Document": item.expected_document,
                "Expected Page": item.expected_page or "",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No evaluation items found.")

    st.markdown("---")
    st.markdown("### ➕ Add Evaluation Item")

    docs = get_all_documents()
    doc_names = [d["filename"] for d in docs if d.get("status") == "completed"]

    with st.form("add_eval_item"):
        new_q = st.text_area("Question", height=80)
        new_a = st.text_area("Expected Answer (verified against document)", height=100)
        new_doc = st.selectbox("Expected Document", options=doc_names) if doc_names else st.text_input("Expected Document filename")
        new_page = st.number_input("Expected Page (0 = unknown)", 0, 9999, 0)
        submitted = st.form_submit_button("Add Item")
        if submitted and new_q and new_a:
            append_to_dataset(
                question=new_q,
                expected_answer=new_a,
                expected_document=new_doc,
                expected_page=int(new_page) if new_page > 0 else None,
            )
            st.success("Item added.")
            st.rerun()

with tab2:
    st.markdown("### ▶️ Run Evaluation")
    st.markdown(
        "This runs real queries through the RAG pipeline and measures performance. "
        "It requires Ollama to be running and documents to be indexed."
    )

    runnable = [item for item in dataset if "PLACEHOLDER" not in item.expected_answer]
    st.info(f"{len(runnable)} runnable evaluation items (placeholder items are skipped).")

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        eval_top_k = st.slider("Top-K for retrieval", 1, 20, TOP_K)
    with col_cfg2:
        eval_model = st.text_input("LLM Model", value=OLLAMA_MODEL)

    if st.button("▶️ Run Evaluation", type="primary", disabled=not runnable):
        with st.spinner(f"Running {len(runnable)} evaluation items…"):
            from modules.llm_handler import is_ollama_running
            from modules.config import OLLAMA_BASE_URL
            if not is_ollama_running(OLLAMA_BASE_URL):
                st.error("Ollama is not running.")
            else:
                results = run_evaluation(runnable, top_k=eval_top_k, model=eval_model)
                st.session_state["eval_results"] = results
                st.session_state["eval_metrics"] = aggregate_metrics(results)
                save_results(results)
                st.success("Evaluation complete!")
                st.rerun()

with tab3:
    st.markdown("### 📈 Evaluation Results")

    if "eval_results" not in st.session_state or not st.session_state["eval_results"]:
        st.info("No evaluation results yet. Run the evaluation from the **Run Evaluation** tab.")
        st.stop()

    results = st.session_state["eval_results"]
    metrics = st.session_state.get("eval_metrics", {})

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    st.markdown("#### Aggregate Metrics")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Recall@K", f"{metrics.get('avg_recall_at_k', 0):.2%}")
    mc2.metric("Citation Accuracy", f"{metrics.get('avg_citation_accuracy', 0):.2%}")
    mc3.metric("Keyword Overlap (F1)", f"{metrics.get('avg_keyword_overlap', 0):.2%}")
    mc4.metric("Avg Response Time", f"{metrics.get('avg_total_time_ms', 0):.0f} ms")

    # ── Timing breakdown ──────────────────────────────────────────────────────
    st.markdown("#### ⏱️ Average Timing Breakdown")
    timing_fig = go.Figure(go.Bar(
        x=["Retrieval", "Generation", "Total"],
        y=[
            metrics.get("avg_retrieval_time_ms", 0),
            metrics.get("avg_generation_time_ms", 0),
            metrics.get("avg_total_time_ms", 0),
        ],
        marker_color=["#58a6ff", "#3fb950", "#bc8cff"],
    ))
    timing_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6edf3"),
        height=250,
        yaxis_title="Milliseconds",
        margin=dict(t=20, b=10),
    )
    st.plotly_chart(timing_fig, use_container_width=True)

    # ── Per-question results ──────────────────────────────────────────────────
    st.markdown("#### Per-question Results")
    rows = []
    for r in results:
        rows.append({
            "Question": r.question[:60],
            "Recall@K": f"{r.recall_at_k:.0%}",
            "Citation Acc.": f"{r.citation_accuracy:.0%}",
            "KW Overlap": f"{r.keyword_overlap:.0%}",
            "Time (ms)": f"{r.total_time_ms:.0f}",
            "Error": r.error[:40] if r.error else "",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Export ────────────────────────────────────────────────────────────────
    csv_str = results_to_csv(results)
    st.download_button(
        "⬇️ Export Results as CSV",
        data=csv_str,
        file_name="evaluation_results.csv",
        mime="text/csv",
    )

    # ── Experiment notes ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📒 Experiment Notes")
    st.markdown("""
    **Experiment 1 — Keyword vs Semantic retrieval:**
    Run the same questions with semantic retrieval (current) and compare manually to keyword matching.

    **Experiment 2 — Text-only vs Multimodal:**
    Disable VLM image descriptions and re-run to measure the impact on visual questions.

    **Experiment 3 — TOP_K sensitivity:**
    Vary TOP_K between 3, 5, and 10 to observe Recall@K changes.

    **Experiment 4 — Chunk size impact:**
    Re-index with chunk sizes 400, 800, 1200 and compare retrieval quality.

    **Experiment 5 — Model comparison:**
    Switch OLLAMA_MODEL and re-run to compare response quality and latency.
    """)
