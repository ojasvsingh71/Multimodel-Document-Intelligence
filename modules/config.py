"""
modules/config.py
=================
Central configuration module.
Loads settings from environment variables / .env file and exposes
them as typed constants.  All other modules import from here so that
configuration is never scattered across the codebase.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (one level above this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (ValueError, TypeError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (ValueError, TypeError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key, str(default)).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _resolve_dir(rel_or_abs: str) -> Path:
    """Return an absolute Path, resolving relative paths from the project root."""
    p = Path(rel_or_abs)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Ollama
# ──────────────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_VISION_MODEL: str = os.getenv("OLLAMA_VISION_MODEL", "llava:7b")
OLLAMA_TIMEOUT: int = _env_int("OLLAMA_TIMEOUT", 120)

# ──────────────────────────────────────────────────────────────────────────────
# Embedding Model
# ──────────────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIM: int = 384          # dimension for all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE: int = _env_int("EMBEDDING_BATCH_SIZE", 32)

# ──────────────────────────────────────────────────────────────────────────────
# ChromaDB
# ──────────────────────────────────────────────────────────────────────────────

CHROMA_PERSIST_DIR: Path = _resolve_dir(
    os.getenv("CHROMA_PERSIST_DIR", "./database/chroma_db")
)
CHROMA_COLLECTION_NAME: str = os.getenv(
    "CHROMA_COLLECTION_NAME", "document_chunks"
)

# ──────────────────────────────────────────────────────────────────────────────
# Document Processing
# ──────────────────────────────────────────────────────────────────────────────

MAX_FILE_SIZE_MB: int = _env_int("MAX_FILE_SIZE_MB", 100)
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_PAGES_PER_DOC: int = _env_int("MAX_PAGES_PER_DOC", 0)   # 0 = unlimited
MAX_IMAGES_PER_DOC: int = _env_int("MAX_IMAGES_PER_DOC", 20)
ENABLE_OCR: bool = _env_bool("ENABLE_OCR", True)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".docx", ".pptx", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}
)

# ──────────────────────────────────────────────────────────────────────────────
# RAG / Chunking
# ──────────────────────────────────────────────────────────────────────────────

CHUNK_SIZE: int = _env_int("CHUNK_SIZE", 800)
CHUNK_OVERLAP: int = _env_int("CHUNK_OVERLAP", 120)
TOP_K: int = _env_int("TOP_K", 5)

# ──────────────────────────────────────────────────────────────────────────────
# Storage Paths
# ──────────────────────────────────────────────────────────────────────────────

UPLOAD_DIR: Path = _resolve_dir(os.getenv("UPLOAD_DIR", "./data/uploads"))
PROCESSED_DIR: Path = _resolve_dir(os.getenv("PROCESSED_DIR", "./data/processed"))
EXTRACTED_IMAGES_DIR: Path = _resolve_dir(
    os.getenv("EXTRACTED_IMAGES_DIR", "./data/extracted_images")
)
OUTPUTS_DIR: Path = _resolve_dir(os.getenv("OUTPUTS_DIR", "./data/outputs"))
CACHE_DIR: Path = _resolve_dir(os.getenv("CACHE_DIR", "./data/cache"))

METADATA_DB_PATH: Path = _PROJECT_ROOT / os.getenv(
    "METADATA_DB", "./database/metadata/documents.db"
)
METADATA_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Application
# ──────────────────────────────────────────────────────────────────────────────

APP_ENV: str = os.getenv("APP_ENV", "development")
PROJECT_ROOT: Path = _PROJECT_ROOT

logger.debug("Configuration loaded from %s", _PROJECT_ROOT / ".env")
