"""
modules/table_extractor.py
===========================
Helpers for working with extracted tables:
  - Load tables from a processed document's JSON artefact.
  - Convert to pandas DataFrame.
  - Export to CSV / Excel bytes.
  - Render as Markdown for display.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def load_tables_for_document(doc_id: str) -> List[Dict[str, Any]]:
    """
    Load table data from the processed document's structured.json.
    Returns a list of table dicts compatible with ExtractedTable.
    """
    from modules.config import PROCESSED_DIR

    json_file = PROCESSED_DIR / doc_id / "structured.json"
    if not json_file.exists():
        return []

    try:
        data = json.loads(json_file.read_text(encoding="utf-8"))
        return data.get("tables", [])
    except Exception as exc:
        logger.warning("Could not load tables for document %s: %s", doc_id, exc)
        return []


def table_to_dataframe(table: Dict[str, Any]):
    """Convert a table dict to a pandas DataFrame."""
    import pandas as pd

    headers = table.get("headers") or []
    rows = table.get("rows") or []

    if not headers and not rows:
        # Try raw_markdown fallback
        md = table.get("raw_markdown", "")
        if md:
            lines = [l for l in md.strip().split("\n") if l.startswith("|")]
            parsed_rows = []
            for line in lines:
                cells = [c.strip() for c in line.strip("|").split("|")]
                if all(set(c) <= {"-", " ", ":"} for c in cells):
                    continue   # separator line
                parsed_rows.append(cells)
            if parsed_rows:
                headers = parsed_rows[0]
                rows = parsed_rows[1:]

    if headers:
        df = pd.DataFrame(rows, columns=headers)
    else:
        df = pd.DataFrame(rows)

    return df


def table_to_csv_bytes(table: Dict[str, Any]) -> bytes:
    """Return the table as UTF-8 CSV bytes."""
    df = table_to_dataframe(table)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def table_to_excel_bytes(table: Dict[str, Any]) -> bytes:
    """Return the table as Excel bytes."""
    import pandas as pd
    import openpyxl  # noqa: F401  required by pandas excel engine

    df = table_to_dataframe(table)
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


def table_to_markdown(table: Dict[str, Any]) -> str:
    """Convert a table dict to a Markdown table string."""
    existing_md = table.get("raw_markdown", "").strip()
    if existing_md:
        return existing_md

    headers = table.get("headers") or []
    rows = table.get("rows") or []

    if not headers and not rows:
        return "_No data_"

    def row_md(cells):
        return "| " + " | ".join(str(c) for c in cells) + " |"

    lines = []
    if headers:
        lines.append(row_md(headers))
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append(row_md(row))

    return "\n".join(lines)
