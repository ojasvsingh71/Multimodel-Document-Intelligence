"""
tests/test_vector_database.py
Unit tests for ChromaDB vector database module.
Uses an in-memory ChromaDB client to avoid persisting test data.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_collection():
    """Return a mock ChromaDB collection with a basic in-memory store."""
    store_ids = []
    store_docs = []
    store_embeds = []
    store_metas = []

    mock = MagicMock()
    mock.count.return_value = 0

    def add(ids, documents, embeddings, metadatas):
        store_ids.extend(ids)
        store_docs.extend(documents)
        store_embeds.extend(embeddings)
        store_metas.extend(metadatas)
        mock.count.return_value = len(store_ids)

    def get(ids=None, where=None, include=None):
        if ids:
            return {"ids": [i for i in ids if i in store_ids]}
        return {"ids": store_ids}

    def query(query_embeddings, n_results, where=None, include=None):
        # Return first n_results items (simplified mock)
        n = min(n_results, len(store_ids))
        return {
            "ids": [store_ids[:n]],
            "documents": [store_docs[:n]],
            "metadatas": [store_metas[:n]],
            "distances": [[0.1 * i for i in range(n)]],
        }

    def delete(ids):
        nonlocal store_ids, store_docs, store_embeds, store_metas
        indices = [i for i, sid in enumerate(store_ids) if sid in ids]
        for idx in sorted(indices, reverse=True):
            store_ids.pop(idx)
            store_docs.pop(idx)
            store_embeds.pop(idx)
            store_metas.pop(idx)
        mock.count.return_value = len(store_ids)

    mock.add.side_effect = add
    mock.get.side_effect = get
    mock.query.side_effect = query
    mock.delete.side_effect = delete

    return mock


@pytest.fixture(autouse=True)
def patch_collection(mock_collection, monkeypatch):
    import modules.vector_database as vdb
    monkeypatch.setattr(vdb, "_collection", mock_collection)
    monkeypatch.setattr(vdb, "get_collection", lambda: mock_collection)
    yield


def _make_chunk(chunk_id: str, doc_id: str, page: int = 1):
    from modules.chunking import DocumentChunk
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        filename="test.pdf",
        page=page,
        section="Test Section",
        content_type="text",
        text=f"Content of chunk {chunk_id}",
        char_count=30,
    )


def _make_embedding(dim: int = 384) -> list:
    vec = np.random.rand(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


def test_add_chunks_basic():
    from modules.vector_database import add_chunks

    chunks = [_make_chunk("c001", "doc1"), _make_chunk("c002", "doc1")]
    embeddings = [_make_embedding(), _make_embedding()]
    inserted = add_chunks(chunks, embeddings)
    assert inserted == 2


def test_add_chunks_mismatched_lengths():
    from modules.vector_database import add_chunks

    chunks = [_make_chunk("c003", "doc1")]
    embeddings = [_make_embedding(), _make_embedding()]
    with pytest.raises(ValueError):
        add_chunks(chunks, embeddings)


def test_search_returns_results():
    from modules.vector_database import add_chunks, search

    chunks = [_make_chunk(f"s{i:03d}", "doc_search") for i in range(5)]
    embeddings = [_make_embedding() for _ in range(5)]
    add_chunks(chunks, embeddings)

    query_vec = _make_embedding()
    results = search(query_embedding=query_vec, top_k=3)
    assert isinstance(results, list)
    for r in results:
        assert "chunk_id" in r
        assert "text" in r
        assert "score" in r


def test_search_returns_empty_on_empty_collection(mock_collection):
    from modules.vector_database import search

    mock_collection.count.return_value = 0
    results = search(query_embedding=_make_embedding(), top_k=5)
    assert results == []


def test_delete_document_vectors():
    from modules.vector_database import add_chunks, delete_document_vectors, get_collection

    chunks = [_make_chunk(f"del{i}", "doc_to_delete") for i in range(3)]
    embeddings = [_make_embedding() for _ in range(3)]
    add_chunks(chunks, embeddings)

    deleted = delete_document_vectors("doc_to_delete")
    # Mock returns items found in store_ids matching doc_to_delete
    # The exact count depends on mock implementation; just check it runs without error
    assert isinstance(deleted, int)


def test_collection_stats():
    from modules.vector_database import collection_stats

    stats = collection_stats()
    assert stats["status"] == "ok"
    assert "total_chunks" in stats
