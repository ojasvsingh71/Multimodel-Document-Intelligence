"""
tests/test_embeddings.py
Unit tests for the embedding generation module.
Uses a tiny mock model to avoid downloading real weights during CI.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _make_mock_model(dim: int = 384):
    """Return a mock SentenceTransformer that returns deterministic vectors."""
    mock = MagicMock()
    mock.get_sentence_embedding_dimension.return_value = dim

    def encode(texts, **kwargs):
        # Return unit vectors with a simple hash fingerprint
        result = []
        for t in texts:
            vec = np.zeros(dim, dtype=np.float32)
            for i, ch in enumerate(t[:dim]):
                vec[i % dim] += ord(ch) / 1000.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            result.append(vec)
        return np.array(result, dtype=np.float32)

    mock.encode.side_effect = encode
    return mock


@pytest.fixture(autouse=True)
def patch_sentence_transformer(monkeypatch):
    """Patch get_embedding_model to avoid network/model downloads."""
    mock_model = _make_mock_model()
    import modules.embeddings as emb
    monkeypatch.setattr(emb, "get_embedding_model", lambda *args, **kwargs: mock_model)
    monkeypatch.setattr(emb, "get_embedding_dim", lambda *args, **kwargs: 384)
    yield


def test_embed_texts_shape():
    from modules.embeddings import embed_texts

    texts = ["Hello world", "Document intelligence", "RAG pipeline"]
    vecs = embed_texts(texts)
    assert vecs.shape == (3, 384)
    assert vecs.dtype == np.float32


def test_embed_texts_empty():
    from modules.embeddings import embed_texts

    vecs = embed_texts([])
    assert vecs.shape[0] == 0


def test_embed_query_returns_list():
    from modules.embeddings import embed_query

    vec = embed_query("What is the main objective?")
    assert isinstance(vec, list)
    assert len(vec) == 384
    assert all(isinstance(v, float) for v in vec)


def test_embed_query_different_texts():
    from modules.embeddings import embed_query

    v1 = embed_query("machine learning")
    v2 = embed_query("deep neural networks")
    assert v1 != v2


def test_embed_chunks():
    from modules.embeddings import embed_chunks
    from modules.chunking import DocumentChunk

    chunks = [
        DocumentChunk(
            chunk_id="c1", doc_id="d1", filename="f.pdf",
            page=1, section="Intro", content_type="text",
            text="First chunk content", char_count=19,
        ),
        DocumentChunk(
            chunk_id="c2", doc_id="d1", filename="f.pdf",
            page=2, section="Methods", content_type="text",
            text="Second chunk content", char_count=20,
        ),
    ]
    embeddings = embed_chunks(chunks)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384


def test_embed_texts_batch_processing():
    """Verify that batch processing produces the same result as single-call."""
    from modules.embeddings import embed_texts

    texts = [f"sentence number {i}" for i in range(10)]
    vecs_single = embed_texts(texts, batch_size=100)
    vecs_batched = embed_texts(texts, batch_size=3)
    # Shapes should match
    assert vecs_single.shape == vecs_batched.shape


def test_embed_texts_normalization():
    from modules.embeddings import embed_texts
    import numpy as np

    vecs = embed_texts(["normalise this text"], normalize=True)
    norm = np.linalg.norm(vecs[0])
    assert abs(norm - 1.0) < 1e-5  # should be unit norm
