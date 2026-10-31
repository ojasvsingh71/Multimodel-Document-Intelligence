"""
modules/chunking.py
====================
Intelligent, section-aware document chunking.

Rather than blindly splitting on character count, this module:
  - Preserves headings and section context.
  - Avoids splitting tables (each table becomes its own chunk).
  - Respects paragraph boundaries.
  - Retains full provenance metadata on every chunk.
  - Falls back to sliding-window splitting only when section structure
    is unavailable.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass
from typing import List, Optional

from modules.config import CHUNK_OVERLAP, CHUNK_SIZE


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    filename: str
    page: Optional[int]       # None when page info is unavailable
    section: str
    content_type: str         # "text" | "table" | "image_description"
    text: str
    char_count: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _make_id() -> str:
    return str(uuid.uuid4())


# ──────────────────────────────────────────────────────────────────────────────
# Heading detection helpers
# ──────────────────────────────────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _detect_heading(line: str) -> Optional[str]:
    m = _HEADING_RE.match(line.strip())
    return m.group(2).strip() if m else None


# ──────────────────────────────────────────────────────────────────────────────
# Sliding-window splitter (fallback)
# ──────────────────────────────────────────────────────────────────────────────

def _sliding_window_split(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Split text into overlapping windows at paragraph boundaries."""
    # Try to split at paragraph boundaries first
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) + 1 > chunk_size and current:
            chunk_text = "\n\n".join(current)
            chunks.append(chunk_text)
            # Overlap: keep last N characters worth of paragraphs
            overlap_text = chunk_text[-overlap:] if overlap else ""
            current = [overlap_text] if overlap_text.strip() else []
            current_len = len(overlap_text)

        current.append(para)
        current_len += len(para) + 2

    if current:
        chunks.append("\n\n".join(current))

    # If a single paragraph is still too large, force-split it
    final: List[str] = []
    for ch in chunks:
        if len(ch) <= chunk_size * 2:
            final.append(ch)
        else:
            # Hard split
            for i in range(0, len(ch), chunk_size - overlap):
                final.append(ch[i: i + chunk_size])
    return [c for c in final if c.strip()]


# ──────────────────────────────────────────────────────────────────────────────
# Section-aware Markdown chunker
# ──────────────────────────────────────────────────────────────────────────────

def chunk_markdown(
    markdown_text: str,
    doc_id: str,
    filename: str,
    page_map: Optional[dict] = None,     # heading -> page_num
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[DocumentChunk]:
    """
    Split Markdown into section-aware chunks.

    Algorithm:
      1. Walk lines and detect headings.
      2. Accumulate lines under the current heading.
      3. When the buffer exceeds chunk_size, flush it (with overlap).
      4. Headings are always included at the start of each child chunk.
    """
    chunks: List[DocumentChunk] = []
    current_section = "Introduction"
    current_lines: List[str] = []
    current_len = 0

    def flush(section: str, lines: List[str], page: Optional[int] = None):
        text = "\n".join(lines).strip()
        if not text:
            return
        if len(text) <= chunk_size:
            chunks.append(DocumentChunk(
                chunk_id=_make_id(),
                doc_id=doc_id,
                filename=filename,
                page=page,
                section=section,
                content_type="text",
                text=text,
                char_count=len(text),
            ))
        else:
            # sliding-window split within this section
            for sub in _sliding_window_split(text, chunk_size, overlap):
                if sub.strip():
                    chunks.append(DocumentChunk(
                        chunk_id=_make_id(),
                        doc_id=doc_id,
                        filename=filename,
                        page=page,
                        section=section,
                        content_type="text",
                        text=sub,
                        char_count=len(sub),
                    ))

    lines = markdown_text.split("\n")
    page: Optional[int] = None

    for line in lines:
        heading = _detect_heading(line)
        if heading:
            # Flush previous section
            flush(current_section, current_lines, page)
            current_section = heading
            current_lines = [line]
            current_len = len(line)
            if page_map:
                page = page_map.get(heading, page)
        else:
            current_lines.append(line)
            current_len += len(line) + 1

            # Flush when buffer is large enough
            if current_len > chunk_size:
                flush(current_section, current_lines, page)
                # Keep section heading for context in next chunk
                current_lines = [f"## {current_section} (continued)"]
                current_len = len(current_lines[0])

    # Flush remainder
    flush(current_section, current_lines, page)

    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Table chunk creator
# ──────────────────────────────────────────────────────────────────────────────

def chunk_table(
    table,          # ExtractedTable from document_processor
    doc_id: str,
    filename: str,
) -> DocumentChunk:
    """Convert an ExtractedTable into a single searchable chunk."""
    # Build a simple text representation
    lines = [f"Table: {table.caption}"]
    if table.headers:
        lines.append("| " + " | ".join(table.headers) + " |")
        lines.append("|" + "---|" * len(table.headers))
    for row in table.rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    text = "\n".join(lines)

    return DocumentChunk(
        chunk_id=_make_id(),
        doc_id=doc_id,
        filename=filename,
        page=table.page,
        section=table.caption,
        content_type="table",
        text=text,
        char_count=len(text),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Image-description chunk creator
# ──────────────────────────────────────────────────────────────────────────────

def chunk_image_description(
    image,          # ExtractedImage from document_processor
    doc_id: str,
    filename: str,
) -> Optional[DocumentChunk]:
    """Convert a VLM-generated image description into a searchable chunk."""
    if not image.description:
        return None
    text = f"Visual content (page {image.page}): {image.description}"
    return DocumentChunk(
        chunk_id=_make_id(),
        doc_id=doc_id,
        filename=filename,
        page=image.page,
        section=image.caption or "Figure",
        content_type="image_description",
        text=text,
        char_count=len(text),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Master chunking entry point
# ──────────────────────────────────────────────────────────────────────────────

def create_chunks(
    processing_result,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[DocumentChunk]:
    """
    Create all chunks from a ProcessingResult.

    Includes: text chunks, table chunks, image-description chunks.
    """
    all_chunks: List[DocumentChunk] = []

    doc_id = processing_result.doc_id
    filename = processing_result.filename

    # Text chunks from markdown
    if processing_result.markdown_text:
        text_chunks = chunk_markdown(
            processing_result.markdown_text,
            doc_id=doc_id,
            filename=filename,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_chunks.extend(text_chunks)

    # Table chunks
    for tbl in processing_result.tables:
        all_chunks.append(chunk_table(tbl, doc_id=doc_id, filename=filename))

    # Image-description chunks (populated after vision_processor runs)
    for img in processing_result.images:
        chunk = chunk_image_description(img, doc_id=doc_id, filename=filename)
        if chunk:
            all_chunks.append(chunk)

    return all_chunks
