"""
pages/6_Visual_Analysis.py
Display extracted document images with VLM-generated descriptions.
Allow users to ask questions about individual images.
"""
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Visual Analysis — Document Intelligence",
    page_icon="🖼️",
    layout="wide",
)

from modules.database_manager import get_all_documents
from modules.vision_processor import load_image_descriptions
from modules.llm_handler import answer_about_image, is_ollama_running, model_supports_vision
from modules.config import OLLAMA_VISION_MODEL, OLLAMA_BASE_URL

st.markdown("## 🖼️ Visual Analysis")
st.markdown(
    "View images extracted from your documents and their AI-generated descriptions. "
    "Ask questions about individual figures and diagrams."
)
st.markdown("---")

docs = get_all_documents()
completed_docs = [d for d in docs if d.get("status") == "completed" and d.get("image_count", 0) > 0]

if not completed_docs:
    st.info(
        "No documents with extracted images. Upload a PDF with figures or images, "
        "then process it. Ensure **Enable image description (VLM)** is checked during upload."
    )
    st.stop()

# ── Document selector ─────────────────────────────────────────────────────────
selected_filename = st.selectbox(
    "Select document",
    options=[d["filename"] for d in completed_docs],
)
selected_doc = next(d for d in completed_docs if d["filename"] == selected_filename)
doc_id = selected_doc["doc_id"]

# ── Vision model status ───────────────────────────────────────────────────────
ollama_ok = is_ollama_running(OLLAMA_BASE_URL)
vision_ok = ollama_ok and bool(OLLAMA_VISION_MODEL) and model_supports_vision(OLLAMA_VISION_MODEL, OLLAMA_BASE_URL)

if not ollama_ok:
    st.warning("⚠️ Ollama is not running. Image Q&A is unavailable.")
elif not vision_ok:
    st.warning(
        f"⚠️ Vision model `{OLLAMA_VISION_MODEL}` not available or doesn't support images. "
        "Image Q&A is disabled. Configure a VLM in Settings."
    )

# ── Load image descriptions ───────────────────────────────────────────────────
image_metas = load_image_descriptions(doc_id)

if not image_metas:
    st.info(
        "No image descriptions found for this document. "
        "Re-upload with **Enable image description (VLM)** checked, "
        "or use the re-describe button below."
    )
    # Fallback: show raw images without descriptions
    from modules.config import EXTRACTED_IMAGES_DIR
    img_dir = EXTRACTED_IMAGES_DIR / doc_id
    if img_dir.exists():
        raw_images = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
        if raw_images:
            st.markdown(f"**{len(raw_images)} raw image(s) found (no AI descriptions):**")
            cols = st.columns(min(3, len(raw_images)))
            for i, img_path in enumerate(raw_images[:12]):
                with cols[i % 3]:
                    st.image(str(img_path), caption=img_path.name, use_container_width=True)
    st.stop()

st.markdown(f"**{len(image_metas)} image(s) extracted from {selected_filename}**")
st.markdown("---")

# ── Image gallery ─────────────────────────────────────────────────────────────
for img_meta in image_metas:
    img_path = img_meta.get("local_path", "")
    caption = img_meta.get("caption", "Figure")
    page = img_meta.get("page")
    description = img_meta.get("description", "")

    page_str = f"Page {page}" if page else "Page unknown"

    with st.expander(f"🖼️ {caption} — {page_str}", expanded=False):
        col_img, col_desc = st.columns([1, 2])

        with col_img:
            if Path(img_path).exists():
                st.image(img_path, caption=f"{caption} ({page_str})", use_container_width=True)
            else:
                st.warning("Image file not found on disk.")

        with col_desc:
            st.markdown("**AI Description:**")
            if description:
                st.markdown(description)
            else:
                st.info("No description available. Enable VLM processing to generate descriptions.")

            st.markdown("---")
            st.markdown("**Ask about this image:**")
            question_key = f"img_q_{img_meta.get('image_id', img_path)}"
            answer_key = f"img_a_{img_meta.get('image_id', img_path)}"

            user_question = st.text_input(
                "Question",
                key=question_key,
                placeholder="e.g. What does this diagram show?",
                label_visibility="collapsed",
            )
            if st.button("Ask", key=f"ask_{question_key}") and user_question:
                if not vision_ok:
                    st.error("Vision model not available.")
                elif not Path(img_path).exists():
                    st.error("Image file not found.")
                else:
                    with st.spinner("Analysing image…"):
                        answer, success = answer_about_image(
                            image_path=img_path,
                            question=user_question,
                            model=OLLAMA_VISION_MODEL,
                        )
                    if success:
                        st.session_state[answer_key] = answer
                    else:
                        st.error(f"Analysis failed: {answer}")

            if answer_key in st.session_state:
                st.markdown("**Answer:**")
                st.markdown(st.session_state[answer_key])
