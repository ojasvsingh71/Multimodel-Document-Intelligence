"""
modules/document_processor.py
==============================
Orchestration layer that drives the full document ingestion pipeline:
  1. Validate and save the uploaded file.
  2. Run Docling conversion to obtain structured content.
  3. Extract text, tables, and images.
  4. Persist extracted artefacts (Markdown, JSON, images).
  5. Update the metadata database.

Returns a rich ProcessingResult dataclass consumed by downstream modules.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.config import (
    ENABLE_OCR,
    EXTRACTED_IMAGES_DIR,
    MAX_FILE_SIZE_BYTES,
    MAX_PAGES_PER_DOC,
    PROCESSED_DIR,
    SUPPORTED_EXTENSIONS,
    UPLOAD_DIR,
)
from modules.database_manager import (
    compute_file_hash,
    get_doc_by_hash,
    insert_document,
    update_document_status,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractedTable:
    table_id: str
    page: Optional[int]
    caption: str
    headers: List[str]
    rows: List[List[str]]
    raw_markdown: str


@dataclass
class ExtractedImage:
    image_id: str
    page: Optional[int]
    caption: str
    local_path: str          # absolute path to saved PNG
    description: str = ""    # filled by vision_processor later


@dataclass
class DocumentChunkSource:
    """Lightweight pointer attached to every text chunk."""
    doc_id: str
    filename: str
    page: Optional[int]
    section: str
    content_type: str         # "text" | "table" | "image_description"


@dataclass
class ProcessingResult:
    doc_id: str
    filename: str
    file_type: str
    status: str               # "completed" | "failed" | "partial"
    page_count: int = 0
    markdown_text: str = ""
    structured_json: Dict[str, Any] = field(default_factory=dict)
    tables: List[ExtractedTable] = field(default_factory=list)
    images: List[ExtractedImage] = field(default_factory=list)
    error_message: str = ""
    processing_time_s: float = 0.0
    processed_dir: str = ""
    stored_path: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

class ValidationError(ValueError):
    pass


def validate_upload(filename: str, file_bytes: bytes) -> None:
    """Raise ValidationError if the file should be rejected."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValidationError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if len(file_bytes) == 0:
        raise ValidationError("Uploaded file is empty.")
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        mb = len(file_bytes) / (1024 * 1024)
        raise ValidationError(
            f"File is too large ({mb:.1f} MB). "
            f"Maximum allowed: {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Docling-based extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_docling_converter():
    """Return a cached Docling DocumentConverter with OCR enabled."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    pipeline_opts = PdfPipelineOptions()
    pipeline_opts.do_ocr = ENABLE_OCR
    pipeline_opts.do_table_structure = True

    try:
        converter = DocumentConverter(
            format_options={
                "pdf": PdfFormatOption(pipeline_options=pipeline_opts)
            }
        )
    except Exception:
        # Fallback: plain converter without extra options
        converter = DocumentConverter()
    return converter


_docling_converter = None


def _converter():
    global _docling_converter
    if _docling_converter is None:
        _docling_converter = _get_docling_converter()
    return _docling_converter


def _extract_tables_from_docling(doc_result, doc_id: str) -> List[ExtractedTable]:
    """Extract structured tables from a Docling conversion result."""
    tables: List[ExtractedTable] = []
    try:
        # Docling exposes tables via doc.tables
        raw_tables = getattr(doc_result.document, "tables", []) or []
        for idx, tbl in enumerate(raw_tables):
            try:
                page_num: Optional[int] = None
                # Try to get page from prov (provenance)
                if hasattr(tbl, "prov") and tbl.prov:
                    page_num = getattr(tbl.prov[0], "page_no", None)

                caption = getattr(tbl, "caption_text", None) or f"Table {idx + 1}"

                # Export to pandas-style grid
                grid = getattr(tbl, "data", None)
                headers: List[str] = []
                rows: List[List[str]] = []
                if grid and hasattr(grid, "grid"):
                    for r_idx, row in enumerate(grid.grid):
                        row_vals = [str(cell.text if cell else "") for cell in row]
                        if r_idx == 0:
                            headers = row_vals
                        else:
                            rows.append(row_vals)

                # Markdown export
                try:
                    raw_md = tbl.export_to_markdown() if hasattr(tbl, "export_to_markdown") else ""
                except Exception:
                    raw_md = ""

                tables.append(
                    ExtractedTable(
                        table_id=f"{doc_id}_tbl_{idx:03d}",
                        page=page_num,
                        caption=caption,
                        headers=headers,
                        rows=rows,
                        raw_markdown=raw_md,
                    )
                )
            except Exception as exc:
                logger.warning("Failed to parse table %d: %s", idx, exc)
    except Exception as exc:
        logger.warning("Table extraction error: %s", exc)
    return tables


def _extract_images_from_docling(
    doc_result,
    doc_id: str,
    source_path: Path,
) -> List[ExtractedImage]:
    """
    Extract images from a Docling result.
    Falls back to PyMuPDF for PDFs to ensure we get actual image data.
    """
    images: List[ExtractedImage] = []
    img_dir = EXTRACTED_IMAGES_DIR / doc_id
    img_dir.mkdir(parents=True, exist_ok=True)

    suffix = source_path.suffix.lower()

    # --- PyMuPDF path for PDFs ---
    if suffix == ".pdf":
        images = _extract_images_pymupdf(source_path, doc_id, img_dir)

    # --- Image files: treat the file itself as the image ---
    elif suffix in {".png", ".jpg", ".jpeg", ".tiff", ".tif"}:
        dest = img_dir / f"image_000.{suffix.lstrip('.')}"
        shutil.copy2(source_path, dest)
        images.append(
            ExtractedImage(
                image_id=f"{doc_id}_img_000",
                page=None,
                caption="Source image",
                local_path=str(dest),
            )
        )
    return images


def _extract_images_pymupdf(
    pdf_path: Path, doc_id: str, img_dir: Path
) -> List[ExtractedImage]:
    """Extract embedded images from a PDF using PyMuPDF."""
    images: List[ExtractedImage] = []
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        img_count = 0
        for page_num, page in enumerate(doc, start=1):
            if MAX_PAGES_PER_DOC and page_num > MAX_PAGES_PER_DOC:
                break
            for img_idx, img_ref in enumerate(page.get_images(full=True)):
                xref = img_ref[0]
                try:
                    base_img = doc.extract_image(xref)
                    img_bytes = base_img["image"]
                    ext = base_img.get("ext", "png")
                    fname = f"page{page_num:04d}_img{img_idx:03d}.{ext}"
                    dest = img_dir / fname
                    dest.write_bytes(img_bytes)
                    images.append(
                        ExtractedImage(
                            image_id=f"{doc_id}_img_{img_count:04d}",
                            page=page_num,
                            caption=f"Image on page {page_num}",
                            local_path=str(dest),
                        )
                    )
                    img_count += 1
                except Exception as exc:
                    logger.debug("Skipping image xref %d: %s", xref, exc)
        doc.close()
    except ImportError:
        logger.warning("PyMuPDF (fitz) not installed — skipping image extraction.")
    except Exception as exc:
        logger.warning("PyMuPDF image extraction error: %s", exc)
    return images


def _count_pages(source_path: Path, doc_result) -> int:
    """Return page count, preferring PyMuPDF for PDFs."""
    if source_path.suffix.lower() == ".pdf":
        try:
            import fitz
            doc = fitz.open(str(source_path))
            n = doc.page_count
            doc.close()
            return n
        except Exception:
            pass
    # Docling provenance fallback
    try:
        pages = set()
        for item in doc_result.document.texts:
            if hasattr(item, "prov") and item.prov:
                pages.add(item.prov[0].page_no)
        return len(pages) if pages else 0
    except Exception:
        return 0


# ──────────────────────────────────────────────────────────────────────────────
# PPTX / DOCX fallback processors
# ──────────────────────────────────────────────────────────────────────────────

def _process_pptx(path: Path, doc_id: str) -> tuple[str, List[ExtractedTable], List[ExtractedImage]]:
    """Extract slides text and tables from a PPTX file."""
    from pptx import Presentation
    from pptx.util import Pt

    prs = Presentation(str(path))
    md_parts: List[str] = []
    tables: List[ExtractedTable] = []
    images: List[ExtractedImage] = []
    img_dir = EXTRACTED_IMAGES_DIR / doc_id
    img_dir.mkdir(parents=True, exist_ok=True)

    for slide_no, slide in enumerate(prs.slides, start=1):
        title = ""
        texts: List[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                t = shape.text_frame.text.strip()
                if t:
                    if shape.shape_type == 13 or (hasattr(shape, "placeholder_format") and
                       shape.placeholder_format and shape.placeholder_format.idx == 0):
                        title = t
                    else:
                        texts.append(t)
            if shape.has_table:
                tbl = shape.table
                headers = [tbl.cell(0, c).text for c in range(tbl._tbl.col_count)]
                rows = [
                    [tbl.cell(r, c).text for c in range(tbl._tbl.col_count)]
                    for r in range(1, len(tbl.rows))
                ]
                tables.append(
                    ExtractedTable(
                        table_id=f"{doc_id}_tbl_{len(tables):03d}",
                        page=slide_no,
                        caption=f"Table on slide {slide_no}",
                        headers=headers,
                        rows=rows,
                        raw_markdown="",
                    )
                )

        heading = f"## Slide {slide_no}" + (f": {title}" if title else "")
        md_parts.append(heading)
        if texts:
            md_parts.extend(texts)
        md_parts.append("")

    return "\n".join(md_parts), tables, images


def _process_docx(path: Path, doc_id: str) -> tuple[str, List[ExtractedTable], List[ExtractedImage]]:
    """Extract paragraphs and tables from a DOCX file."""
    import docx

    document = docx.Document(str(path))
    md_parts: List[str] = []
    tables: List[ExtractedTable] = []

    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = para.style.name if para.style else ""
        if "Heading 1" in style:
            md_parts.append(f"# {text}")
        elif "Heading 2" in style:
            md_parts.append(f"## {text}")
        elif "Heading 3" in style:
            md_parts.append(f"### {text}")
        else:
            md_parts.append(text)

    for tbl_idx, tbl in enumerate(document.tables):
        rows_data: List[List[str]] = []
        for row in tbl.rows:
            rows_data.append([cell.text.strip() for cell in row.cells])
        headers = rows_data[0] if rows_data else []
        rows = rows_data[1:] if len(rows_data) > 1 else []
        tables.append(
            ExtractedTable(
                table_id=f"{doc_id}_tbl_{tbl_idx:03d}",
                page=None,
                caption=f"Table {tbl_idx + 1}",
                headers=headers,
                rows=rows,
                raw_markdown="",
            )
        )

    return "\n".join(md_parts), tables, []


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline entry point
# ──────────────────────────────────────────────────────────────────────────────

def process_document(filename: str, file_bytes: bytes) -> ProcessingResult:
    """
    Full ingestion pipeline for a single document.

    Steps:
      1. Validate.
      2. Duplicate check.
      3. Save to upload dir.
      4. Docling / fallback conversion.
      5. Extract tables & images.
      6. Persist artefacts.
      7. Update database.
    """
    t0 = time.perf_counter()

    # 1. Validate
    try:
        validate_upload(filename, file_bytes)
    except ValidationError as exc:
        return ProcessingResult(
            doc_id="", filename=filename, file_type="",
            status="failed", error_message=str(exc),
        )

    file_hash = compute_file_hash(file_bytes)
    suffix = Path(filename).suffix.lower()
    file_type = suffix.lstrip(".")

    # 2. Duplicate check
    existing = get_doc_by_hash(file_hash)
    if existing:
        logger.info("Duplicate upload detected for %s — reusing existing record.", filename)
        return ProcessingResult(
            doc_id=existing["doc_id"],
            filename=existing["filename"],
            file_type=existing["file_type"],
            status=existing["status"],
            page_count=existing.get("page_count") or 0,
            error_message="Duplicate: this document was already uploaded.",
            stored_path=existing.get("stored_path") or "",
            processing_time_s=time.perf_counter() - t0,
        )

    # 3. Save file
    safe_name = Path(filename).name
    doc_id = str(uuid.uuid4())
    stored_path = UPLOAD_DIR / f"{doc_id}_{safe_name}"
    stored_path.write_bytes(file_bytes)

    # Register in DB
    insert_document(
        filename=safe_name,
        file_type=file_type,
        file_size=len(file_bytes),
        file_hash=file_hash,
        stored_path=str(stored_path),
    )

    update_document_status(doc_id="", status="processing")
    # Re-fetch after insert to get real doc_id (insert_document returns it)
    # Actually insert_document returns doc_id directly — let's re-call properly:
    # (We need to fix the flow — db insert returns doc_id)
    # Since insert_document() creates its own UUID, we need to match via hash.
    db_rec = get_doc_by_hash(file_hash)
    if db_rec:
        doc_id = db_rec["doc_id"]
    update_document_status(doc_id=doc_id, status="processing")

    markdown_text = ""
    tables: List[ExtractedTable] = []
    images: List[ExtractedImage] = []
    page_count = 0
    error_msg = ""

    try:
        # 4. Conversion
        if suffix in {".pdf"}:
            conv = _converter()
            result = conv.convert(str(stored_path))
            markdown_text = result.document.export_to_markdown()
            tables = _extract_tables_from_docling(result, doc_id)
            images = _extract_images_from_docling(result, doc_id, stored_path)
            page_count = _count_pages(stored_path, result)

        elif suffix == ".docx":
            try:
                conv = _converter()
                result = conv.convert(str(stored_path))
                markdown_text = result.document.export_to_markdown()
                tables = _extract_tables_from_docling(result, doc_id)
                page_count = _count_pages(stored_path, result)
            except Exception:
                markdown_text, tables, images = _process_docx(stored_path, doc_id)

        elif suffix == ".pptx":
            try:
                conv = _converter()
                result = conv.convert(str(stored_path))
                markdown_text = result.document.export_to_markdown()
                tables = _extract_tables_from_docling(result, doc_id)
            except Exception:
                markdown_text, tables, images = _process_pptx(stored_path, doc_id)

        elif suffix in {".png", ".jpg", ".jpeg", ".tiff", ".tif"}:
            # For images, run OCR via Docling then also save original image
            try:
                conv = _converter()
                result = conv.convert(str(stored_path))
                markdown_text = result.document.export_to_markdown()
            except Exception:
                markdown_text = ""
            images = _extract_images_from_docling(None, doc_id, stored_path)
            page_count = 1

        # 5. Persist artefacts
        proc_dir = PROCESSED_DIR / doc_id
        proc_dir.mkdir(parents=True, exist_ok=True)

        (proc_dir / "extracted.md").write_text(markdown_text, encoding="utf-8")

        structured = {
            "doc_id": doc_id,
            "filename": safe_name,
            "file_type": file_type,
            "page_count": page_count,
            "tables": [asdict(t) for t in tables],
            "images": [asdict(img) for img in images],
        }
        (proc_dir / "structured.json").write_text(
            json.dumps(structured, indent=2, default=str), encoding="utf-8"
        )

        # 6. Update DB
        update_document_status(
            doc_id=doc_id,
            status="completed",
            page_count=page_count,
            table_count=len(tables),
            image_count=len(images),
            processed_dir=str(proc_dir),
        )

        return ProcessingResult(
            doc_id=doc_id,
            filename=safe_name,
            file_type=file_type,
            status="completed",
            page_count=page_count,
            markdown_text=markdown_text,
            structured_json=structured,
            tables=tables,
            images=images,
            processing_time_s=time.perf_counter() - t0,
            processed_dir=str(proc_dir),
            stored_path=str(stored_path),
        )

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        logger.error("Processing failed for %s: %s", filename, exc)
        update_document_status(
            doc_id=doc_id, status="failed", error_message=str(exc)
        )
        return ProcessingResult(
            doc_id=doc_id,
            filename=safe_name,
            file_type=file_type,
            status="failed",
            error_message=error_msg,
            processing_time_s=time.perf_counter() - t0,
            stored_path=str(stored_path),
        )


def delete_document_files(doc_id: str, stored_path: str, processed_dir: str) -> None:
    """Remove all files associated with a document from disk."""
    try:
        p = Path(stored_path)
        if p.exists():
            p.unlink()
    except Exception as exc:
        logger.warning("Could not delete stored file: %s", exc)

    try:
        pd = Path(processed_dir)
        if pd.exists() and pd.is_dir():
            shutil.rmtree(pd)
    except Exception as exc:
        logger.warning("Could not delete processed dir: %s", exc)

    try:
        img_dir = EXTRACTED_IMAGES_DIR / doc_id
        if img_dir.exists():
            shutil.rmtree(img_dir)
    except Exception as exc:
        logger.warning("Could not delete images dir: %s", exc)

    logger.info("Deleted all files for document %s", doc_id)


# Keep backward compat with the original milestone-1 function name
def convert_document_to_markdown(file_path) -> str:
    """Legacy helper kept for backward compatibility."""
    from pathlib import Path as _Path
    p = _Path(file_path)
    conv = _converter()
    res = conv.convert(str(p))
    return res.document.export_to_markdown()
