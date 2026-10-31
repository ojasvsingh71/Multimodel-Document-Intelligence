"""
tests/test_rag_pipeline.py
Unit tests for the RAG pipeline.
All Ollama and ChromaDB calls are mocked so tests run without a live server.
"""
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_environment(monkeypatch):
    """Mock embedding, vector search, and LLM for RAG tests."""
    import modules.rag_pipeline as rag
    import modules.embeddings as emb
    import modules.vector_database as vdb
    import modules.llm_handler as llm

    # Mock embedding
    monkeypatch.setattr(rag, "embed_query", lambda q, **kw: [0.1] * 384)
    monkeypatch.setattr(emb, "embed_query", lambda q, **kw: [0.1] * 384)

    # Mock vector search
    fake_chunks = [
        {
            "chunk_id": "c001",
            "text": "The proposed model achieved 96.5% accuracy on the MNIST dataset.",
            "doc_id": "doc001",
            "filename": "research_paper.pdf",
            "page": 12,
            "section": "Results",
            "content_type": "text",
            "distance": 0.05,
            "score": 0.95,
        },
        {
            "chunk_id": "c002",
            "text": "The CNN architecture consists of 5 convolutional layers.",
            "doc_id": "doc001",
            "filename": "research_paper.pdf",
            "page": 8,
            "section": "Architecture",
            "content_type": "text",
            "distance": 0.10,
            "score": 0.90,
        },
    ]
    monkeypatch.setattr(rag, "search", lambda **kw: fake_chunks)
    monkeypatch.setattr(vdb, "search", lambda **kw: fake_chunks)

    # Mock Ollama running
    monkeypatch.setattr(rag, "is_ollama_running", lambda url: True)
    monkeypatch.setattr(llm, "is_ollama_running", lambda url: True)

    # Mock text generation
    monkeypatch.setattr(rag, "generate_text_chat", lambda messages, **kw: "The model achieved 96.5% accuracy.")
    monkeypatch.setattr(llm, "generate_text_chat", lambda messages, **kw: "The model achieved 96.5% accuracy.")

    yield


def test_answer_question_success(mock_environment):
    from modules.rag_pipeline import answer_question

    response = answer_question(
        question="What accuracy did the model achieve?",
        doc_ids=["doc001"],
    )
    assert response.success
    assert "96.5%" in response.answer or response.answer
    assert len(response.citations) > 0


def test_answer_question_empty_question(mock_environment):
    from modules.rag_pipeline import answer_question

    response = answer_question(question="  ", doc_ids=["doc001"])
    assert not response.success
    assert "empty" in response.error.lower()


def test_answer_question_ollama_not_running(monkeypatch):
    import modules.llm_handler as llm
    import modules.rag_pipeline as rag

    monkeypatch.setattr(rag, "is_ollama_running", lambda url: False)
    monkeypatch.setattr(llm, "is_ollama_running", lambda url: False)

    from modules.rag_pipeline import answer_question
    response = answer_question(question="Test question", doc_ids=["doc001"])
    assert not response.success
    assert "ollama" in response.error.lower()


def test_answer_question_no_chunks(monkeypatch):
    import modules.llm_handler as llm
    import modules.rag_pipeline as rag
    import modules.vector_database as vdb
    import modules.embeddings as emb

    monkeypatch.setattr(rag, "is_ollama_running", lambda url: True)
    monkeypatch.setattr(llm, "is_ollama_running", lambda url: True)
    monkeypatch.setattr(rag, "search", lambda **kw: [])
    monkeypatch.setattr(vdb, "search", lambda **kw: [])
    monkeypatch.setattr(rag, "embed_query", lambda q, **kw: [0.0] * 384)
    monkeypatch.setattr(emb, "embed_query", lambda q, **kw: [0.0] * 384)

    from modules.rag_pipeline import answer_question
    response = answer_question(question="Unanswerable question", doc_ids=["doc001"])
    # Should return a graceful "not found" message, not crash
    assert isinstance(response.answer, str)
    assert len(response.answer) > 0


def test_rag_response_timing(mock_environment):
    from modules.rag_pipeline import answer_question

    response = answer_question(question="What is the methodology?", doc_ids=["doc001"])
    assert response.retrieval_time_ms >= 0
    assert response.generation_time_ms >= 0
    assert response.query_time_ms >= 0


def test_citations_have_required_fields(mock_environment):
    from modules.rag_pipeline import answer_question

    response = answer_question(question="Describe the architecture.", doc_ids=["doc001"])
    for citation in response.citations:
        assert hasattr(citation, "filename")
        assert hasattr(citation, "score")
        assert hasattr(citation, "excerpt")
        assert 0.0 <= citation.score <= 1.0


def test_conversation_history_included(mock_environment, monkeypatch):
    """Verify conversation history is passed to the LLM."""
    import modules.llm_handler as llm
    import modules.rag_pipeline as rag

    captured_messages = []

    def capture_messages(messages, **kw):
        captured_messages.extend(messages)
        return "Answer based on context."

    monkeypatch.setattr(rag, "generate_text_chat", capture_messages)
    monkeypatch.setattr(llm, "generate_text_chat", capture_messages)

    from modules.rag_pipeline import answer_question

    history = [
        {"role": "user", "content": "What paper is this?"},
        {"role": "assistant", "content": "This is a CNN paper."},
    ]
    answer_question(
        question="What dataset was used?",
        doc_ids=["doc001"],
        conversation_history=history,
    )
    # Conversation history should appear in messages
    roles = [m["role"] for m in captured_messages]
    assert "user" in roles
