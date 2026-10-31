"""
modules/vision_processor.py
============================
Processes document images through a Vision-Language Model (VLM).

Responsibilities:
  - Screen images for meaningfulness (skip tiny/blank images).
  - Send qualifying images to the VLM via llm_handler.describe_image.
  - Persist generated descriptions alongside the image metadata.
  - Update ProcessingResult.images in place.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from modules.config import MAX_IMAGES_PER_DOC, OLLAMA_VISION_MODEL

logger = logging.getLogger(__name__)

# Minimum image dimensions to bother sending to the VLM
_MIN_WIDTH = 50
_MIN_HEIGHT = 50
_MIN_FILE_BYTES = 2_048   # skip near-empty / corrupt images


def _is_meaningful_image(image_path: str) -> bool:
    """
    Return True if the image is large enough and non-trivial.
    Uses PIL for a quick dimension check.
    """
    p = Path(image_path)
    if not p.exists() or p.stat().st_size < _MIN_FILE_BYTES:
        return False
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            w, h = im.size
        return w >= _MIN_WIDTH and h >= _MIN_HEIGHT
    except Exception:
        return False  # unreadable → skip


def _build_context_prompt(image, filename: str) -> str:
    page_str = f"page {image.page}" if image.page else "an unknown page"
    caption = image.caption or "a figure"
    return (
        f"This image is from the document '{filename}', {page_str}. "
        f"It is described as '{caption}'. "
        "Provide a detailed description of everything visible: "
        "diagrams, charts, graphs, architectural elements, labels, text, "
        "arrows, annotations, and any other notable features."
    )


def describe_document_images(
    processing_result,
    vision_model: str = OLLAMA_VISION_MODEL,
    max_images: int = MAX_IMAGES_PER_DOC,
    base_url: Optional[str] = None,
) -> int:
    """
    Iterate over extracted images in a ProcessingResult, send each to the VLM,
    and populate image.description in place.

    Returns the number of images successfully described.
    """
    from modules.llm_handler import describe_image, is_ollama_running, model_supports_vision
    from modules.config import OLLAMA_BASE_URL, PROCESSED_DIR

    _base_url = base_url or OLLAMA_BASE_URL

    if not vision_model:
        logger.warning("OLLAMA_VISION_MODEL is not set — skipping image description.")
        return 0

    if not is_ollama_running(_base_url):
        logger.warning("Ollama is not running — skipping image description.")
        return 0

    if not model_supports_vision(vision_model, _base_url):
        logger.warning(
            "Model '%s' does not support vision — skipping image description.", vision_model
        )
        return 0

    described = 0
    images = processing_result.images or []
    candidates = [img for img in images if _is_meaningful_image(img.local_path)]
    candidates = candidates[:max_images]

    logger.info(
        "Describing %d/%d images for document %s using %s",
        len(candidates), len(images), processing_result.filename, vision_model,
    )

    for img in candidates:
        context = _build_context_prompt(img, processing_result.filename)
        description, success = describe_image(
            image_path=img.local_path,
            context_prompt=context,
            model=vision_model,
            base_url=_base_url,
        )
        if success and description:
            img.description = description
            described += 1
            logger.debug(
                "Image %s described (%d chars).", img.image_id, len(description)
            )
        else:
            logger.warning("Image %s description failed: %s", img.image_id, description)

    # Persist updated image metadata
    try:
        proc_dir = Path(processing_result.processed_dir)
        if proc_dir.exists():
            img_meta = [
                {
                    "image_id": img.image_id,
                    "page": img.page,
                    "caption": img.caption,
                    "local_path": img.local_path,
                    "description": img.description,
                }
                for img in images
            ]
            (proc_dir / "image_descriptions.json").write_text(
                json.dumps(img_meta, indent=2), encoding="utf-8"
            )
    except Exception as exc:
        logger.warning("Could not persist image descriptions: %s", exc)

    return described


def load_image_descriptions(doc_id: str) -> List[dict]:
    """
    Load previously generated image descriptions from disk.
    Returns a list of image metadata dicts.
    """
    from modules.config import PROCESSED_DIR

    desc_file = PROCESSED_DIR / doc_id / "image_descriptions.json"
    if not desc_file.exists():
        return []
    try:
        return json.loads(desc_file.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read image descriptions for %s: %s", doc_id, exc)
        return []
