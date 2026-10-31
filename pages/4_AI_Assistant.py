"""
pages/4_AI_Assistant.py
Conversational RAG interface for querying documents.
"""
import streamlit as st

st.set_page_config(
    page_title="AI Assistant — Document Intelligence",
    page_icon="🤖",
    layout="wide",
)

from modules.database_manager import get_all_documents
from modules.rag_pipeline import answer_question
from modules.config import OLLAMA_MODEL, TOP_K

st.markdown("## 🤖 AI Document Assistant")
st.markdown("Ask questions about your documents using natural language. Answers are grounded in your uploaded content.")
st.markdown("---")

# ── Session state initialisation ──────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []      # [{role, content}]
if "chat_messages_display" not in st.session_state:
    st.session_state.chat_messages_display = []  # [{role, content, citations}]

# ── Sidebar: document selector & settings ────────────────────────────────────
with st.sidebar:
    st.markdown("### 📄 Document Selection")
    docs = get_all_documents()
    completed_docs = [d for d in docs if d.get("status") == "completed"]

    if not completed_docs:
        st.warning("No indexed documents. Upload and process documents first.")
        selected_doc_ids = []
    else:
        all_selected = st.checkbox("Search all documents", value=True)
        if not all_selected:
            selected_filenames = st.multiselect(
                "Select documents",
                options=[d["filename"] for d in completed_docs],
                default=[d["filename"] for d in completed_docs[:1]],
            )
            selected_doc_ids = [
                d["doc_id"] for d in completed_docs
                if d["filename"] in selected_filenames
            ]
        else:
            selected_doc_ids = [d["doc_id"] for d in completed_docs]

    st.markdown("---")
    st.markdown("### ⚙️ RAG Settings")
    top_k = st.slider("Top-K chunks to retrieve", 1, 20, TOP_K)
    model = st.text_input("LLM Model", value=OLLAMA_MODEL)

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.chat_messages_display = []
        st.rerun()

# ── Suggested starter questions ───────────────────────────────────────────────
if not st.session_state.chat_messages_display:
    st.markdown("#### 💡 Suggested Questions")
    suggestions = [
        "What is the main objective of this research?",
        "What methodology was used in this study?",
        "What dataset was used for evaluation?",
        "Summarise the key findings.",
        "What are the limitations of this work?",
    ]
    cols = st.columns(len(suggestions))
    for col, suggestion in zip(cols, suggestions):
        with col:
            if st.button(suggestion, use_container_width=True, key=f"sug_{suggestion[:15]}"):
                st.session_state["prefill_question"] = suggestion
                st.rerun()

# ── Chat display ──────────────────────────────────────────────────────────────
chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_messages_display:
        role = msg["role"]
        content = msg["content"]
        citations = msg.get("citations", [])

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant"):
                st.markdown(content)
                if citations:
                    with st.expander(f"📚 Sources ({len(citations)} references)", expanded=False):
                        for i, cite in enumerate(citations, 1):
                            page_str = f"p.{cite['page']}" if cite.get("page") else "page unknown"
                            score_str = f"{cite['score']:.2f}"
                            st.markdown(
                                f"**[{i}] {cite['filename']}** — {page_str} "
                                f"| Section: *{cite.get('section', 'N/A')}* "
                                f"| Score: {score_str}"
                            )
                            st.caption(f"> {cite['excerpt'][:200]}…")

# ── Question input ────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill_question", "")
question = st.chat_input(
    placeholder="Ask a question about your documents…",
)
if prefill and not question:
    question = prefill

if question:
    if not selected_doc_ids:
        st.error("No documents selected. Please select at least one indexed document.")
        st.stop()

    # Show user message immediately
    st.session_state.chat_messages_display.append({"role": "user", "content": question})
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.spinner("🔍 Searching documents and generating answer…"):
        response = answer_question(
            question=question,
            doc_ids=selected_doc_ids,
            conversation_history=st.session_state.chat_history[:-1],  # exclude current
            top_k=top_k,
            model=model,
        )

    if response.success:
        # Build citation dicts for display
        citation_dicts = [
            {
                "filename": c.filename,
                "page": c.page,
                "section": c.section,
                "score": c.score,
                "excerpt": c.excerpt,
            }
            for c in response.citations
        ]
        st.session_state.chat_messages_display.append({
            "role": "assistant",
            "content": response.answer,
            "citations": citation_dicts,
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response.answer,
        })

        # Show timing in sidebar
        with st.sidebar:
            st.markdown("### ⏱️ Last Query Timing")
            st.caption(f"Query embed: {response.query_time_ms:.0f} ms")
            st.caption(f"Retrieval: {response.retrieval_time_ms:.0f} ms")
            st.caption(f"Generation: {response.generation_time_ms:.0f} ms")
    else:
        error_msg = f"❌ {response.error}"
        st.session_state.chat_messages_display.append({
            "role": "assistant",
            "content": error_msg,
            "citations": [],
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": error_msg,
        })

    st.rerun()
