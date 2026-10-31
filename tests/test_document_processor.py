"""
tests/test_document_processor.py
Unit tests for document processing and validation.
"""
import pytest
from pathlib import Path
from modules.document_processor import validate_upload, ValidationError


def test_validate_upload_valid_pdf():
    """Valid PDF bytes (fake magic bytes) pass validation."""
    # PDF magic bytes header
    fake_pdf = b"%PDF-1.4 " + b"x" * 1024
    validate_upload("test.pdf", fake_pdf)  # should not raise


def test_validate_upload_empty_file():
    with pytest.raises(ValidationError, match="empty"):
        validate_upload("test.pdf", b"")


def test_validate_upload_unsupported_extension():
    with pytest.raises(ValidationError, match="Unsupported"):
        validate_upload("malware.exe", b"x" * 1024)


def test_validate_upload_too_large():
    from modules.config import MAX_FILE_SIZE_BYTES
    oversized = b"x" * (MAX_FILE_SIZE_BYTES + 1)
    with pytest.raises(ValidationError, match="too large"):
        validate_upload("big.pdf", oversized)


def test_validate_upload_supported_extensions():
    fake_content = b"x" * 1024
    for ext in [".pdf", ".docx", ".pptx", ".png", ".jpg", ".jpeg", ".tiff"]:
        validate_upload(f"file{ext}", fake_content)  # should not raise


def test_validate_upload_case_insensitive():
    # Upper-case extension should still work
    validate_upload("DOCUMENT.PDF", b"x" * 1024)


def test_compute_file_hash():
    from modules.database_manager import compute_file_hash

    data = b"test content for hashing"
    h = compute_file_hash(data)
    assert isinstance(h, str)
    assert len(h) == 64


def test_delete_document_files_nonexistent(tmp_path):
    """Deleting non-existent files should not raise exceptions."""
    from modules.document_processor import delete_document_files

    delete_document_files(
        doc_id="nonexistent",
        stored_path=str(tmp_path / "ghost.pdf"),
        processed_dir=str(tmp_path / "ghost_dir"),
    )  # Should complete without error


def test_delete_document_files_real(tmp_path):
    """Verify actual files are removed."""
    from modules.document_processor import delete_document_files

    # Create fake artefacts
    stored = tmp_path / "stored.pdf"
    stored.write_bytes(b"pdf content")
    proc_dir = tmp_path / "processed"
    proc_dir.mkdir()
    (proc_dir / "extracted.md").write_text("content")

    delete_document_files(
        doc_id="testdoc",
        stored_path=str(stored),
        processed_dir=str(proc_dir),
    )

    assert not stored.exists()
    assert not proc_dir.exists()
