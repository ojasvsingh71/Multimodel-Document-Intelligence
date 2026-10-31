"""
pages/10_Settings.py
System configuration and connection status.
"""
import streamlit as st

st.set_page_config(
    page_title="Settings — Document Intelligence",
    page_icon="⚙️",
    layout="wide",
)

from modules.config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_VISION_MODEL, OLLAMA_TIMEOUT,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,
    MAX_FILE_SIZE_MB, MAX_PAGES_PER_DOC, MAX_IMAGES_PER_DOC, ENABLE_OCR,
    CHROMA_PERSIST_DIR, METADATA_DB_PATH,
)

st.markdown("## ⚙️ Settings & Status")
st.markdown(
    "View current configuration and connection status. "
    "To change settings, edit `.env` in the project root and restart the application."
)
st.markdown("---")

# ── Connection Status ─────────────────────────────────────────────────────────
st.markdown("### 🔌 Connection Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Ollama")
    from modules.llm_handler import ollama_status
    status = ollama_status(OLLAMA_MODEL, OLLAMA_VISION_MODEL, OLLAMA_BASE_URL)

    if status["running"]:
        st.success("✅ Ollama is running")
        st.caption(f"URL: `{OLLAMA_BASE_URL}`")
        st.caption(f"Text model: `{OLLAMA_MODEL}` — {'✅ installed' if status.get('text_model_available') else '❌ not installed'}")
        st.caption(f"Vision model: `{OLLAMA_VISION_MODEL}` — {'✅ installed' if status.get('vision_model_available') else '❌ not installed'}")
        if status.get("vision_supported"):
            st.caption("👁️ Vision capability confirmed")
        installed = status.get("installed_models", [])
        if installed:
            with st.expander("Installed models"):
                for m in installed:
                    st.code(m)
    else:
        st.error("❌ Ollama not running")
        st.caption(status.get("error", ""))
        st.markdown("""
        **To start Ollama:**
        ```
        ollama serve
        ```
        **To install a model:**
        ```
        ollama pull qwen2.5:3b
        ollama pull llava:7b
        ```
        """)

with col2:
    st.markdown("#### ChromaDB")
    from modules.vector_database import chroma_status
    cs = chroma_status()
    if cs.get("status") == "ok":
        st.success("✅ ChromaDB ready")
        st.caption(f"Collection: `{cs['collection_name']}`")
        st.caption(f"Total chunks: {cs['total_chunks']:,}")
        st.caption(f"Storage: `{cs['persist_dir']}`")
    else:
        st.error(f"❌ ChromaDB error: {cs.get('error', 'Unknown')}")

with col3:
    st.markdown("#### Embedding Model")
    from modules.embeddings import embedding_model_status
    es = embedding_model_status(EMBEDDING_MODEL)
    if es.get("status") == "ok":
        st.success("✅ Embedding model loaded")
        st.caption(f"Model: `{es['model']}`")
        st.caption(f"Dimension: {es['dim']}")
    else:
        st.warning(f"⚠️ {es.get('error', 'Model not loaded yet')}")
        st.caption("The model is loaded on first use.")

st.markdown("---")

# ── Current Configuration ─────────────────────────────────────────────────────
st.markdown("### 📋 Current Configuration")
st.caption("These values are loaded from `.env`. Edit `.env` and restart to apply changes.")

col_llm, col_doc, col_rag = st.columns(3)

with col_llm:
    st.markdown("**LLM / Ollama**")
    st.code(f"""
OLLAMA_BASE_URL={OLLAMA_BASE_URL}
OLLAMA_MODEL={OLLAMA_MODEL}
OLLAMA_VISION_MODEL={OLLAMA_VISION_MODEL}
OLLAMA_TIMEOUT={OLLAMA_TIMEOUT}
""")

with col_doc:
    st.markdown("**Document Processing**")
    st.code(f"""
MAX_FILE_SIZE_MB={MAX_FILE_SIZE_MB}
MAX_PAGES_PER_DOC={MAX_PAGES_PER_DOC}
MAX_IMAGES_PER_DOC={MAX_IMAGES_PER_DOC}
ENABLE_OCR={ENABLE_OCR}
EMBEDDING_MODEL={EMBEDDING_MODEL}
""")

with col_rag:
    st.markdown("**RAG / Chunking**")
    st.code(f"""
CHUNK_SIZE={CHUNK_SIZE}
CHUNK_OVERLAP={CHUNK_OVERLAP}
TOP_K={TOP_K}
CHROMA_DIR={CHROMA_PERSIST_DIR}
""")

st.markdown("---")

# ── Ollama Setup Guide ────────────────────────────────────────────────────────
with st.expander("📖 Ollama Setup Guide", expanded=False):
    st.markdown("""
    ### Installing Ollama on Windows

    **1. Download and install:**
    - Visit [ollama.com](https://ollama.com) and download the Windows installer.
    - Run the installer and follow the instructions.

    **2. Start the Ollama server:**
    ```powershell
    ollama serve
    ```

    **3. Pull recommended models:**
    ```powershell
    # Text model (small, fast, CPU-friendly)
    ollama pull qwen2.5:3b

    # Vision model (for image understanding) — larger, ~4GB
    ollama pull llava:7b

    # Alternative lightweight text models
    ollama pull phi3:mini
    ollama pull llama3.2:3b
    ```

    **4. Verify models are installed:**
    ```powershell
    ollama list
    ```

    **5. Test text model:**
    ```powershell
    ollama run qwen2.5:3b "Hello, summarise AI in one sentence."
    ```
    """)

with st.expander("🔧 Troubleshooting", expanded=False):
    st.markdown("""
    | Problem | Solution |
    |---------|----------|
    | Ollama not running | Run `ollama serve` in a terminal |
    | Model not installed | Run `ollama pull <model-name>` |
    | Out of RAM | Use a smaller model like `qwen2.5:1.5b` or `phi3:mini` |
    | Slow responses | Increase `OLLAMA_TIMEOUT` in `.env` |
    | Vision model fails | Check model name with `ollama list` and supports-vision |
    | ChromaDB error | Delete `./database/chroma_db` and re-index documents |
    | Port conflict | Change `OLLAMA_BASE_URL` port if 11434 is in use |
    | Import errors | Run `pip install -r requirements.txt` in your venv |
    """)
