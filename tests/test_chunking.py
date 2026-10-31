"""
tests/test_chunking.py
Unit tests for the document chunking module.
"""
import pytest
from modules.chunking import (
    chunk_markdown,
    chunk_table,
    _sliding_window_split,
    DocumentChunk,
)
from modules.document_processor import ExtractedTable


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_MARKDOWN = """# Introduction

This is the introduction section.
It has multiple sentences to test chunking behaviour.

## Related Work

Several prior studies have examined this problem.
Smith et al. (2020) proposed a similar approach.
Jones et al. (2021) extended this work.

## Methodology

The proposed methodology consists of three stages:
1. Data collection from publicly available sources.
2. Pre-processing and normalisation.
3. Model training and evaluation.

## Results

The model achieved an accuracy of 96.5% on the test set.
This represents an improvement of 3.2% over the baseline.

## Conclusion

In this paper we proposed a novel approach that outperforms existing methods.
"""


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_sliding_window_creates_chunks():
    text = "word " * 500  # 2500 chars
    chunks = _sliding_window_split(text, chunk_size=400, overlap=50)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 800  # should not massively exceed chunk_size


def test_chunk_markdown_returns_document_chunks():
    chunks = chunk_markdown(SAMPLE_MARKDOWN, doc_id="doc123", filename="test.pdf")
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    for c in chunks:
        assert isinstance(c, DocumentChunk)
        assert c.doc_id == "doc123"
        assert c.filename == "test.pdf"
        assert c.text.strip()
        assert c.chunk_id


def test_chunk_markdown_preserves_sections():
    chunks = chunk_markdown(SAMPLE_MARKDOWN, doc_id="d1", filename="f.pdf")
    sections = {c.section for c in chunks}
    assert any("Introduction" in s for s in sections)
    assert any("Methodology" in s or "Method" in s for s in sections)


def test_chunk_markdown_content_type_is_text():
    chunks = chunk_markdown(SAMPLE_MARKDOWN, doc_id="d1", filename="f.pdf")
    for c in chunks:
        assert c.content_type == "text"


def test_chunk_markdown_respects_chunk_size():
    chunks = chunk_markdown(
        SAMPLE_MARKDOWN, doc_id="d1", filename="f.pdf", chunk_size=200, overlap=20
    )
    # Each chunk should not be excessively larger than the configured chunk_size
    for c in chunks:
        assert len(c.text) <= 600  # allow some tolerance for heading context


def test_chunk_table():
    table = ExtractedTable(
        table_id="tbl_001",
        page=5,
        caption="Results Table",
        headers=["Method", "Accuracy", "F1"],
        rows=[
            ["CNN", "96.5%", "0.96"],
            ["ResNet", "94.2%", "0.94"],
        ],
        raw_markdown="",
    )
    chunk = chunk_table(table, doc_id="doc1", filename="paper.pdf")
    assert chunk.content_type == "table"
    assert chunk.page == 5
    assert "Results Table" in chunk.text
    assert "CNN" in chunk.text
    assert chunk.doc_id == "doc1"


def test_chunk_table_empty():
    table = ExtractedTable(
        table_id="tbl_empty",
        page=None,
        caption="Empty Table",
        headers=[],
        rows=[],
        raw_markdown="",
    )
    chunk = chunk_table(table, doc_id="doc1", filename="paper.pdf")
    assert isinstance(chunk, DocumentChunk)


def test_empty_markdown_returns_no_chunks():
    chunks = chunk_markdown("", doc_id="d1", filename="empty.pdf")
    assert chunks == []


def test_chunk_ids_are_unique():
    chunks = chunk_markdown(SAMPLE_MARKDOWN, doc_id="d1", filename="f.pdf")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
