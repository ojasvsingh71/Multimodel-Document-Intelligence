"""
app.py — Main entry point for the Multimodal Document Intelligence System.

Run with:
    streamlit run app.py
"""

import streamlit as st

# ── Page config (must be the very first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="Multimodal Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "LLM-Powered Multimodal Document Intelligence System",
    },
)

# ── Bootstrap database ───────────────────────────────────────────────────────
from modules.database_manager import init_database  # noqa: E402
init_database()

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Base ── */
    [data-testid="stAppViewContainer"] { background: #0f1117; }
    [data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }

    /* ── Typography ── */
    h1, h2, h3 { font-family: 'Inter', sans-serif; }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #58a6ff, #bc8cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .sub-title { color: #8b949e; font-size: 1rem; margin-top: 0; }

    /* ── Cards ── */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        text-align: center;
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #58a6ff; }
    .metric-card .label { font-size: 0.85rem; color: #8b949e; margin-top: 0.25rem; }

    /* ── Status badges ── */
    .badge-completed { background:#1a4731; color:#3fb950; padding:3px 10px;
                       border-radius:20px; font-size:0.78rem; }
    .badge-processing { background:#3d2b00; color:#d29922; padding:3px 10px;
                        border-radius:20px; font-size:0.78rem; }
    .badge-failed { background:#4b1a1a; color:#f85149; padding:3px 10px;
                    border-radius:20px; font-size:0.78rem; }
    .badge-pending { background:#1c2128; color:#8b949e; padding:3px 10px;
                     border-radius:20px; font-size:0.78rem; }

    /* ── Chat bubbles ── */
    .user-msg {
        background:#1c2128; border:1px solid #30363d;
        border-radius:10px 10px 2px 10px;
        padding:0.75rem 1rem; margin:0.5rem 0; margin-left:15%;
    }
    .ai-msg {
        background:#161b22; border:1px solid #30363d;
        border-radius:10px 10px 10px 2px;
        padding:0.75rem 1rem; margin:0.5rem 0; margin-right:15%;
    }

    /* ── Source cards ── */
    .source-card {
        background:#0d1117; border:1px solid #30363d; border-radius:8px;
        padding:0.6rem 1rem; margin:0.3rem 0; font-size:0.85rem;
    }

    /* ── Sidebar nav ── */
    [data-testid="stSidebarNav"] { padding-top: 0.5rem; }

    /* ── Divider ── */
    hr { border-color: #30363d; }

    /* ── Button ── */
    .stButton>button {
        background: linear-gradient(135deg, #58a6ff22, #bc8cff22);
        border: 1px solid #58a6ff55;
        color: #58a6ff;
        border-radius: 8px;
        transition: all 0.2s;
    }
    .stButton>button:hover { border-color: #58a6ff; background: #58a6ff33; }

    /* ── Tables ── */
    .dataframe thead th { background: #21262d !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Landing Page Content ─────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🧠 Multimodal Document Intelligence</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload, analyse, and converse with your documents using local AI.</p>', unsafe_allow_html=True)

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="value">📄</div>
        <div class="label">PDF · DOCX · PPTX · Images</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="value">🔍</div>
        <div class="label">Semantic Search & RAG</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="value">👁️</div>
        <div class="label">Vision-Language Understanding</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="value">🔒</div>
        <div class="label">100% Local · No API Keys</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Quick-start guide
col_a, col_b = st.columns([2, 1])

with col_a:
    st.markdown("### 🚀 Quick Start")
    st.markdown("""
    1. **Upload Documents** — Go to **Upload Documents** in the sidebar to upload PDF, DOCX, PPTX, or image files.
    2. **Process & Index** — The system extracts text, tables, and images, then builds a searchable vector index.
    3. **Ask Questions** — Open **AI Document Assistant** to chat with your documents using RAG.
    4. **Summarise** — Use **Summarization** to generate detailed or research-structured summaries.
    5. **Compare** — Select multiple documents in **Document Comparison** to compare methodologies and results.
    6. **Analyse Visuals** — The **Visual Analysis** page shows extracted images with AI-generated descriptions.
    """)

with col_b:
    st.markdown("### ⚡ System Requirements")
    st.markdown("""
    - **Ollama** running locally
    - Recommended model: `qwen2.5:3b`
    - Vision model: `llava:7b` *(optional)*
    - Python 3.10+
    - 8 GB RAM minimum
    """)

    # Quick status indicators
    st.markdown("### 🔌 Status")
    from modules.llm_handler import is_ollama_running
    from modules.vector_database import chroma_status
    from modules.config import OLLAMA_BASE_URL

    ollama_ok = is_ollama_running(OLLAMA_BASE_URL)
    chroma_ok = chroma_status().get("status") == "ok"

    st.markdown(
        f"{'🟢' if ollama_ok else '🔴'} Ollama — {'Running' if ollama_ok else 'Not running'}"
    )
    st.markdown(
        f"{'🟢' if chroma_ok else '🔴'} ChromaDB — {'Ready' if chroma_ok else 'Error'}"
    )

    if not ollama_ok:
        st.warning("Start Ollama: run `ollama serve` in a terminal.")

st.markdown("---")
st.caption(
    "Navigate using the **sidebar** on the left. "
    "All processing happens locally — your documents never leave your machine."
)
