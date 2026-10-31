"""
modules/embeddings.py
======================
Sentence Transformer embedding generation with:
  - Singleton model caching (loaded once per process).
  - Batch processing to stay within CPU RAM limits.
  - L2-normalisation for cosine similarity via dot product.
  - Graceful error handling and fallback.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from modules.config import EMBEDDING_BATCH_SIZE, EMBEDDING_DIM, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Singleton loader
# ──────────────────────────────────────────────────────────────────────────────

_model = None
_loaded_model_name: str = ""


def get_embedding_model(model_name: str = EMBEDDING_MODEL):
    """Load and cache the Sentence Transformer model."""
    global _model, _loaded_model_name
    if _model is None or _loaded_model_name != model_name:
        logger.info("Loading embedding model: %s", model_name)
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(model_name)
        _loaded_model_name = model_name
        dim = (
            _model.get_embedding_dimension()
            if hasattr(_model, "get_embedding_dimension")
            else _model.get_sentence_embedding_dimension()
        )
        logger.info("Embedding model loaded — dim=%d", dim)
    return _model


def get_embedding_dim(model_name: str = EMBEDDING_MODEL) -> int:
    """Return the output dimension of the configured model."""
    try:
        m = get_embedding_model(model_name)
        if hasattr(m, "get_embedding_dimension"):
            return m.get_embedding_dimension()
        return m.get_sentence_embedding_dimension()
    except Exception:
        return EMBEDDING_DIM


# ──────────────────────────────────────────────────────────────────────────────
# Core embedding functions
# ──────────────────────────────────────────────────────────────────────────────

def embed_texts(
    texts: List[str],
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    normalize: bool = True,
) -> np.ndarray:
    """
    Embed a list of strings and return a float32 numpy array of shape
    (len(texts), embedding_dim).

    Args:
        texts:       List of text strings to embed.
        model_name:  HuggingFace model identifier.
        batch_size:  Number of texts per inference call.
        normalize:   If True, L2-normalise each vector (enables cosine via dot).

    Returns:
        numpy array of shape (N, D).
    """
    if not texts:
        return np.empty((0, get_embedding_dim(model_name)), dtype=np.float32)

    model = get_embedding_model(model_name)
    all_embeddings: List[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start: start + batch_size]
        try:
            vecs = model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
            )
            all_embeddings.append(vecs.astype(np.float32))
        except Exception as exc:
            logger.error("Embedding batch %d failed: %s", start // batch_size, exc)
            # Return zero vectors for the failed batch to avoid data corruption
            dim = get_embedding_dim(model_name)
            zero = np.zeros((len(batch), dim), dtype=np.float32)
            all_embeddings.append(zero)

    return np.vstack(all_embeddings)


def embed_query(
    query: str,
    model_name: str = EMBEDDING_MODEL,
    normalize: bool = True,
) -> List[float]:
    """
    Embed a single query string and return a plain Python list.
    Suitable for passing directly to ChromaDB.
    """
    vec = embed_texts([query], model_name=model_name, normalize=normalize)
    return vec[0].tolist()


def embed_chunks(chunks: list, model_name: str = EMBEDDING_MODEL) -> list:
    """
    Given a list of DocumentChunk objects, add embeddings to them.
    Returns a parallel list of embedding vectors (each a list[float]).
    """
    texts = [c.text for c in chunks]
    if not texts:
        return []
    logger.info("Generating embeddings for %d chunks …", len(texts))
    vecs = embed_texts(texts, model_name=model_name)
    return [v.tolist() for v in vecs]


# ──────────────────────────────────────────────────────────────────────────────
# Status probe
# ──────────────────────────────────────────────────────────────────────────────

def embedding_model_status(model_name: str = EMBEDDING_MODEL) -> dict:
    """Return a status dict for the settings/dashboard page."""
    try:
        m = get_embedding_model(model_name)
        dim = m.get_sentence_embedding_dimension()
        return {"status": "ok", "model": model_name, "dim": dim}
    except Exception as exc:
        return {"status": "error", "model": model_name, "error": str(exc)}
