"""
modules/database_manager.py
============================
SQLite-based persistent metadata store for documents, processing status,
chunks, and activity history.  All important application state lives here
so the document library survives Streamlit restarts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.config import METADATA_DB_PATH

logger = logging.getLogger(__name__)

# One lock per database path — safe for Streamlit's single-process model.
_db_lock = threading.Lock()

# ──────────────────────────────────────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    filename        TEXT NOT NULL,
    file_type       TEXT NOT NULL,
    file_size       INTEGER NOT NULL,
    file_hash       TEXT NOT NULL,
    upload_ts       TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    page_count      INTEGER,
    table_count     INTEGER DEFAULT 0,
    image_count     INTEGER DEFAULT 0,
    chunk_count     INTEGER DEFAULT 0,
    stored_path     TEXT,
    processed_dir   TEXT,
    error_message   TEXT,
    metadata_json   TEXT
);

CREATE TABLE IF NOT EXISTS activity_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    doc_id      TEXT,
    action      TEXT NOT NULL,
    details     TEXT
);

CREATE TABLE IF NOT EXISTS app_settings (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);
"""


# ──────────────────────────────────────────────────────────────────────────────
# Connection helper
# ──────────────────────────────────────────────────────────────────────────────

@contextmanager
def _get_conn():
    """Thread-safe SQLite connection context manager."""
    with _db_lock:
        conn = sqlite3.connect(str(METADATA_DB_PATH), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def init_database() -> None:
    """Create tables if they do not already exist."""
    with _get_conn() as conn:
        conn.executescript(_SCHEMA_SQL)
    logger.info("Database initialised at %s", METADATA_DB_PATH)


# ──────────────────────────────────────────────────────────────────────────────
# Document CRUD
# ──────────────────────────────────────────────────────────────────────────────

def compute_file_hash(file_bytes: bytes) -> str:
    """Return SHA-256 hex digest for duplicate detection."""
    return hashlib.sha256(file_bytes).hexdigest()


def get_doc_by_hash(file_hash: str) -> Optional[Dict[str, Any]]:
    """Return existing document record if this file was already uploaded."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE file_hash = ?", (file_hash,)
        ).fetchone()
    return dict(row) if row else None


def insert_document(
    filename: str,
    file_type: str,
    file_size: int,
    file_hash: str,
    stored_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Insert a new document record and return its generated doc_id."""
    doc_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO documents
              (doc_id, filename, file_type, file_size, file_hash,
               upload_ts, status, stored_path, metadata_json)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                doc_id, filename, file_type, file_size, file_hash,
                now, "pending", stored_path,
                json.dumps(metadata or {}),
            ),
        )
    _log_activity(doc_id=doc_id, action="upload", details=f"Uploaded {filename}")
    logger.info("Document inserted: %s (%s)", filename, doc_id)
    return doc_id


def update_document_status(
    doc_id: str,
    status: str,
    *,
    page_count: Optional[int] = None,
    table_count: Optional[int] = None,
    image_count: Optional[int] = None,
    chunk_count: Optional[int] = None,
    processed_dir: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    """Update processing status and statistics for an existing document."""
    fields: list[str] = ["status = ?"]
    values: list[Any] = [status]
    if page_count is not None:
        fields.append("page_count = ?"); values.append(page_count)
    if table_count is not None:
        fields.append("table_count = ?"); values.append(table_count)
    if image_count is not None:
        fields.append("image_count = ?"); values.append(image_count)
    if chunk_count is not None:
        fields.append("chunk_count = ?"); values.append(chunk_count)
    if processed_dir is not None:
        fields.append("processed_dir = ?"); values.append(processed_dir)
    if error_message is not None:
        fields.append("error_message = ?"); values.append(error_message)
    values.append(doc_id)

    with _get_conn() as conn:
        conn.execute(
            f"UPDATE documents SET {', '.join(fields)} WHERE doc_id = ?",
            values,
        )
    _log_activity(doc_id=doc_id, action="status_update", details=status)


def get_all_documents() -> List[Dict[str, Any]]:
    """Return all document records ordered by upload time descending."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY upload_ts DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_document(doc_id: str) -> Optional[Dict[str, Any]]:
    """Return a single document record or None."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
    return dict(row) if row else None


def delete_document(doc_id: str) -> bool:
    """Delete document record. Returns True if a row was deleted."""
    with _get_conn() as conn:
        cursor = conn.execute(
            "DELETE FROM documents WHERE doc_id = ?", (doc_id,)
        )
    deleted = cursor.rowcount > 0
    if deleted:
        _log_activity(doc_id=doc_id, action="delete", details="Document deleted")
    return deleted


# ──────────────────────────────────────────────────────────────────────────────
# Activity Log
# ──────────────────────────────────────────────────────────────────────────────

def _log_activity(
    action: str,
    doc_id: Optional[str] = None,
    details: Optional[str] = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    try:
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO activity_log (ts, doc_id, action, details) VALUES (?,?,?,?)",
                (now, doc_id, action, details),
            )
    except Exception as exc:
        logger.warning("Failed to write activity log: %s", exc)


def get_recent_activity(limit: int = 20) -> List[Dict[str, Any]]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM activity_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ──────────────────────────────────────────────────────────────────────────────
# App Settings
# ──────────────────────────────────────────────────────────────────────────────

def get_setting(key: str, default: str = "") -> str:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT value FROM app_settings WHERE key = ?", (key,)
        ).fetchone()
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO app_settings (key, value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_statistics() -> Dict[str, int]:
    """Return aggregate counts for the dashboard."""
    with _get_conn() as conn:
        row = conn.execute("""
            SELECT
                COALESCE(COUNT(*), 0) AS total_docs,
                COALESCE(SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END), 0) AS completed_docs,
                COALESCE(SUM(page_count), 0)  AS total_pages,
                COALESCE(SUM(table_count), 0) AS total_tables,
                COALESCE(SUM(image_count), 0) AS total_images,
                COALESCE(SUM(chunk_count), 0) AS total_chunks
            FROM documents
        """).fetchone()
    if not row:
        return {}
    return {k: (int(v) if v is not None else 0) for k, v in dict(row).items()}


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap on import
# ──────────────────────────────────────────────────────────────────────────────

init_database()
