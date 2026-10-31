"""
modules/vector_database.py
===========================
ChromaDB persistent vector store.

Responsibilities:
  - Create / open a persistent collection.
  - Insert document chunk embeddings with full metadata.
  - Semantic similarity search (optionally filtered by document).
  - Delete all vectors for a document.
  - Collection statistics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from modules.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Client singleton
# ──────────────────────────────────────────────────────────────────────────────

_chroma_client: Optional[Any] = None
_collection = None


def _get_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        logger.info("Opening ChromaDB at %s", CHROMA_PERSIST_DIR)
        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_PERSIST_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _chroma_client


def get_collection():
    """Return (and cache) the document chunks collection."""
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready — %d items",
            CHROMA_COLLECTION_NAME,
            _collection.count(),
        )
    return _collection


def reset_collection_cache() -> None:
    """Force re-open collection on next access (useful after schema changes)."""
    global _collection, _chroma_client
    _collection = None
    _chroma_client = None


# ──────────────────────────────────────────────────────────────────────────────
# Insert
# ──────────────────────────────────────────────────────────────────────────────

def add_chunks(
    chunks: List,            # List[DocumentChunk]
    embeddings: List[List[float]],
) -> int:
    """
    Insert chunks with their embeddings into ChromaDB.

    Returns the number of chunks actually inserted.
    Skips chunks whose chunk_id already exists (idempotent).
    """
    if not chunks or not embeddings:
        return 0
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) counts differ"
        )

    col = get_collection()

    # Filter out already-existing IDs to avoid duplicates
    ids = [c.chunk_id for c in chunks]
    existing = set()
    try:
        existing_result = col.get(ids=ids, include=[])
        existing = set(existing_result.get("ids", []))
    except Exception:
        pass

    new_ids, new_docs, new_embeds, new_metas = [], [], [], []
    for chunk, emb in zip(chunks, embeddings):
        if chunk.chunk_id in existing:
            continue
        new_ids.append(chunk.chunk_id)
        new_docs.append(chunk.text)
        new_embeds.append(emb)
        meta: Dict[str, Any] = {
            "doc_id": chunk.doc_id,
            "filename": chunk.filename,
            "page": chunk.page if chunk.page is not None else -1,
            "section": chunk.section or "",
            "content_type": chunk.content_type,
            "char_count": chunk.char_count,
        }
        new_metas.append(meta)

    if not new_ids:
        logger.debug("All chunks already exist — nothing to insert.")
        return 0

    # ChromaDB recommends batches ≤ 5 000
    BATCH = 500
    inserted = 0
    for i in range(0, len(new_ids), BATCH):
        sl = slice(i, i + BATCH)
        col.add(
            ids=new_ids[sl],
            documents=new_docs[sl],
            embeddings=new_embeds[sl],
            metadatas=new_metas[sl],
        )
        inserted += len(new_ids[sl])

    logger.info("Inserted %d chunks into ChromaDB.", inserted)
    return inserted


# ──────────────────────────────────────────────────────────────────────────────
# Query / Retrieval
# ──────────────────────────────────────────────────────────────────────────────

def search(
    query_embedding: List[float],
    top_k: int = 5,
    doc_ids: Optional[List[str]] = None,   # filter to these documents only
    content_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve the top_k most similar chunks.

    Args:
        query_embedding: Query vector (list[float]).
        top_k:           Number of results to return.
        doc_ids:         If provided, restrict search to these document IDs.
        content_types:   If provided, restrict to these content_type values.

    Returns:
        List of dicts with keys: chunk_id, text, doc_id, filename, page,
        section, content_type, distance, score.
    """
    col = get_collection()
    if col.count() == 0:
        return []

    where: Optional[Dict] = None

    conditions = []
    if doc_ids:
        if len(doc_ids) == 1:
            conditions.append({"doc_id": {"$eq": doc_ids[0]}})
        else:
            conditions.append({"doc_id": {"$in": doc_ids}})
    if content_types:
        if len(content_types) == 1:
            conditions.append({"content_type": {"$eq": content_types[0]}})
        else:
            conditions.append({"content_type": {"$in": content_types}})

    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    try:
        results = col.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, col.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.error("ChromaDB query failed: %s", exc)
        return []

    hits: List[Dict[str, Any]] = []
    ids_list = results.get("ids", [[]])[0]
    docs_list = results.get("documents", [[]])[0]
    metas_list = results.get("metadatas", [[]])[0]
    dists_list = results.get("distances", [[]])[0]

    for chunk_id, text, meta, dist in zip(ids_list, docs_list, metas_list, dists_list):
        # Convert cosine distance → similarity score (0-1)
        score = max(0.0, 1.0 - dist)
        page_raw = meta.get("page", -1)
        hits.append({
            "chunk_id": chunk_id,
            "text": text,
            "doc_id": meta.get("doc_id", ""),
            "filename": meta.get("filename", ""),
            "page": None if page_raw == -1 else int(page_raw),
            "section": meta.get("section", ""),
            "content_type": meta.get("content_type", "text"),
            "distance": dist,
            "score": round(score, 4),
        })

    return hits


# ──────────────────────────────────────────────────────────────────────────────
# Delete
# ──────────────────────────────────────────────────────────────────────────────

def delete_document_vectors(doc_id: str) -> int:
    """
    Delete all vectors belonging to a specific document.
    Returns the count of deleted vectors.
    """
    col = get_collection()
    try:
        # Fetch IDs first
        result = col.get(
            where={"doc_id": {"$eq": doc_id}},
            include=[],
        )
        ids_to_delete = result.get("ids", [])
        if ids_to_delete:
            col.delete(ids=ids_to_delete)
            logger.info("Deleted %d vectors for document %s", len(ids_to_delete), doc_id)
        return len(ids_to_delete)
    except Exception as exc:
        logger.error("Failed to delete vectors for %s: %s", doc_id, exc)
        return 0


# ──────────────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────────────

def collection_stats() -> Dict[str, Any]:
    """Return basic collection statistics for the dashboard."""
    try:
        col = get_collection()
        return {
            "status": "ok",
            "total_chunks": col.count(),
            "collection_name": CHROMA_COLLECTION_NAME,
            "persist_dir": str(CHROMA_PERSIST_DIR),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def chroma_status() -> Dict[str, Any]:
    """Health-check for the settings page."""
    return collection_stats()
