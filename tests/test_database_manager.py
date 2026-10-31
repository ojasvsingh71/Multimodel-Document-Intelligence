"""
tests/test_database_manager.py
Unit tests for the SQLite metadata database manager.
"""
import os
import tempfile
from pathlib import Path
import pytest


# ── Override DB path for tests so we don't pollute the real database ──────────

@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Redirect database operations to a temp directory."""
    test_db = tmp_path / "test_documents.db"
    monkeypatch.setattr("modules.database_manager.METADATA_DB_PATH", test_db)
    # Re-init the database with the new path
    import modules.database_manager as dbm
    monkeypatch.setattr(dbm, "METADATA_DB_PATH", test_db)
    dbm.init_database()
    yield test_db


def test_insert_and_retrieve_document():
    from modules.database_manager import insert_document, get_document

    doc_id = insert_document(
        filename="test.pdf",
        file_type="pdf",
        file_size=12345,
        file_hash="abc123",
        stored_path="/data/uploads/test.pdf",
    )
    assert doc_id

    doc = get_document(doc_id)
    assert doc is not None
    assert doc["filename"] == "test.pdf"
    assert doc["file_hash"] == "abc123"
    assert doc["status"] == "pending"


def test_update_document_status():
    from modules.database_manager import insert_document, get_document, update_document_status

    doc_id = insert_document(
        filename="status_test.pdf",
        file_type="pdf",
        file_size=100,
        file_hash="hashxyz",
        stored_path="/tmp/status_test.pdf",
    )
    update_document_status(
        doc_id=doc_id,
        status="completed",
        page_count=10,
        table_count=2,
        image_count=3,
        chunk_count=50,
    )
    doc = get_document(doc_id)
    assert doc["status"] == "completed"
    assert doc["page_count"] == 10
    assert doc["table_count"] == 2
    assert doc["chunk_count"] == 50


def test_delete_document():
    from modules.database_manager import insert_document, delete_document, get_document

    doc_id = insert_document(
        filename="delete_me.pdf",
        file_type="pdf",
        file_size=100,
        file_hash="delhash",
        stored_path="/tmp/delete_me.pdf",
    )
    assert get_document(doc_id) is not None
    deleted = delete_document(doc_id)
    assert deleted is True
    assert get_document(doc_id) is None


def test_duplicate_hash_detection():
    from modules.database_manager import insert_document, get_doc_by_hash

    insert_document(
        filename="dup.pdf",
        file_type="pdf",
        file_size=100,
        file_hash="duphash",
        stored_path="/tmp/dup.pdf",
    )
    found = get_doc_by_hash("duphash")
    assert found is not None
    assert found["filename"] == "dup.pdf"

    not_found = get_doc_by_hash("nonexistent")
    assert not_found is None


def test_get_all_documents_order():
    from modules.database_manager import insert_document, get_all_documents
    import time

    insert_document("first.pdf", "pdf", 100, "h1", "/tmp/first.pdf")
    time.sleep(0.01)
    insert_document("second.pdf", "pdf", 100, "h2", "/tmp/second.pdf")

    docs = get_all_documents()
    assert docs[0]["filename"] == "second.pdf"  # most recent first


def test_get_statistics():
    from modules.database_manager import insert_document, update_document_status, get_statistics

    doc_id = insert_document("stat_doc.pdf", "pdf", 100, "stathash", "/tmp/stat_doc.pdf")
    update_document_status(doc_id=doc_id, status="completed", page_count=5, chunk_count=20)

    stats = get_statistics()
    assert stats["total_docs"] >= 1
    assert stats["completed_docs"] >= 1
    assert stats["total_chunks"] >= 20


def test_compute_file_hash_consistency():
    from modules.database_manager import compute_file_hash

    data = b"hello world content"
    h1 = compute_file_hash(data)
    h2 = compute_file_hash(data)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex = 64 chars

    h3 = compute_file_hash(b"different content")
    assert h3 != h1


def test_settings_store():
    from modules.database_manager import get_setting, set_setting

    set_setting("my_key", "my_value")
    assert get_setting("my_key") == "my_value"

    set_setting("my_key", "updated_value")
    assert get_setting("my_key") == "updated_value"

    assert get_setting("missing_key", "default") == "default"
