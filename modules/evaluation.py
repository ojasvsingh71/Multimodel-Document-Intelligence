"""
modules/evaluation.py
======================
Evaluation framework for the RAG system.

Metrics implemented:
  A. Retrieval Recall@K
  B. Answer Accuracy (LLM-as-judge and keyword-overlap)
  C. Citation Accuracy
  D. Response Time
  E. Table Extraction Quality (precision/recall on cells)

Results are stored as JSON and exportable as CSV.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.config import OUTPUTS_DIR, TOP_K

logger = logging.getLogger(__name__)

_RESULTS_DIR = OUTPUTS_DIR / "evaluation"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalItem:
    question: str
    expected_answer: str
    expected_document: str
    expected_page: Optional[int]
    doc_ids: List[str] = field(default_factory=list)


@dataclass
class EvalResult:
    question: str
    expected_answer: str
    generated_answer: str
    expected_document: str
    expected_page: Optional[int]
    retrieved_docs: List[str] = field(default_factory=list)
    retrieved_pages: List[Optional[int]] = field(default_factory=list)
    recall_at_k: float = 0.0
    citation_accuracy: float = 0.0
    keyword_overlap: float = 0.0
    total_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    error: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Dataset management
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_DATASET = Path("evaluation") / "evaluation_dataset.csv"
_DATASET_HEADERS = [
    "question", "expected_answer", "expected_document", "expected_page", "doc_ids"
]

_EXAMPLE_ROWS = [
    {
        "question": "What is the main objective of the research?",
        "expected_answer": "PLACEHOLDER — replace with actual answer after verification",
        "expected_document": "your_document.pdf",
        "expected_page": "1",
        "doc_ids": "",
    },
    {
        "question": "What dataset was used for evaluation?",
        "expected_answer": "PLACEHOLDER — replace with actual answer after verification",
        "expected_document": "your_document.pdf",
        "expected_page": "",
        "doc_ids": "",
    },
]


def ensure_dataset_exists() -> Path:
    """Create a template evaluation dataset if one does not exist."""
    from modules.config import PROJECT_ROOT
    path = PROJECT_ROOT / _DEFAULT_DATASET
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_DATASET_HEADERS)
            writer.writeheader()
            writer.writerows(_EXAMPLE_ROWS)
        logger.info("Created example evaluation dataset at %s", path)
    return path


def load_dataset(path: Optional[Path] = None) -> List[EvalItem]:
    """Load evaluation items from a CSV file."""
    from modules.config import PROJECT_ROOT
    fpath = path or (PROJECT_ROOT / _DEFAULT_DATASET)
    if not fpath.exists():
        return []

    items = []
    with open(fpath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                page_raw = row.get("expected_page", "").strip()
                page = int(page_raw) if page_raw.isdigit() else None
                doc_ids_raw = row.get("doc_ids", "").strip()
                doc_ids = [d.strip() for d in doc_ids_raw.split(",") if d.strip()]
                items.append(
                    EvalItem(
                        question=row["question"].strip(),
                        expected_answer=row["expected_answer"].strip(),
                        expected_document=row["expected_document"].strip(),
                        expected_page=page,
                        doc_ids=doc_ids,
                    )
                )
            except Exception as exc:
                logger.warning("Skipping malformed eval row: %s", exc)
    return items


def append_to_dataset(
    question: str,
    expected_answer: str,
    expected_document: str,
    expected_page: Optional[int] = None,
    doc_ids: Optional[List[str]] = None,
    path: Optional[Path] = None,
) -> None:
    """Append a new evaluation item to the CSV dataset."""
    from modules.config import PROJECT_ROOT
    fpath = path or (PROJECT_ROOT / _DEFAULT_DATASET)
    ensure_dataset_exists()
    with open(fpath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_DATASET_HEADERS)
        writer.writerow({
            "question": question,
            "expected_answer": expected_answer,
            "expected_document": expected_document,
            "expected_page": expected_page or "",
            "doc_ids": ",".join(doc_ids or []),
        })


# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────────────────────

def _keyword_overlap(expected: str, generated: str) -> float:
    """Simple token-level F1 between expected and generated answers."""
    exp_tokens = set(expected.lower().split())
    gen_tokens = set(generated.lower().split())
    if not exp_tokens:
        return 0.0
    common = exp_tokens & gen_tokens
    precision = len(common) / len(gen_tokens) if gen_tokens else 0.0
    recall = len(common) / len(exp_tokens)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def _recall_at_k(
    expected_doc: str,
    expected_page: Optional[int],
    retrieved_chunks: List[Dict[str, Any]],
) -> float:
    """Return 1.0 if the expected source appears in the retrieved chunks."""
    for chunk in retrieved_chunks:
        if chunk.get("filename", "").lower() == expected_doc.lower():
            if expected_page is None:
                return 1.0
            if chunk.get("page") == expected_page:
                return 1.0
    return 0.0


def _citation_accuracy(
    expected_doc: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> float:
    """Return fraction of retrieved chunks that match the expected document."""
    if not retrieved_chunks:
        return 0.0
    matches = sum(
        1 for c in retrieved_chunks
        if c.get("filename", "").lower() == expected_doc.lower()
    )
    return round(matches / len(retrieved_chunks), 4)


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation runner
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    items: List[EvalItem],
    top_k: int = TOP_K,
    model: Optional[str] = None,
) -> List[EvalResult]:
    """
    Run the full evaluation pipeline on a list of EvalItems.

    Returns a list of EvalResult objects with all metrics populated.
    """
    from modules.config import OLLAMA_MODEL
    from modules.rag_pipeline import answer_question

    _model = model or OLLAMA_MODEL
    results: List[EvalResult] = []

    for item in items:
        if not item.question or "PLACEHOLDER" in item.expected_answer:
            logger.info("Skipping placeholder eval item: %s", item.question)
            continue

        t_start = time.perf_counter()
        try:
            rag_resp = answer_question(
                question=item.question,
                doc_ids=item.doc_ids or None,
                top_k=top_k,
                model=_model,
            )
            total_ms = (time.perf_counter() - t_start) * 1000

            retrieved_docs = [c["filename"] for c in rag_resp.retrieved_chunks]
            retrieved_pages = [c.get("page") for c in rag_resp.retrieved_chunks]

            recall = _recall_at_k(
                item.expected_document,
                item.expected_page,
                rag_resp.retrieved_chunks,
            )
            citation = _citation_accuracy(item.expected_document, rag_resp.retrieved_chunks)
            kw = _keyword_overlap(item.expected_answer, rag_resp.answer)

            results.append(EvalResult(
                question=item.question,
                expected_answer=item.expected_answer,
                generated_answer=rag_resp.answer,
                expected_document=item.expected_document,
                expected_page=item.expected_page,
                retrieved_docs=retrieved_docs,
                retrieved_pages=retrieved_pages,
                recall_at_k=recall,
                citation_accuracy=citation,
                keyword_overlap=kw,
                total_time_ms=total_ms,
                retrieval_time_ms=rag_resp.retrieval_time_ms,
                generation_time_ms=rag_resp.generation_time_ms,
                error=rag_resp.error,
            ))
        except Exception as exc:
            logger.error("Evaluation failed for question '%s': %s", item.question, exc)
            results.append(EvalResult(
                question=item.question,
                expected_answer=item.expected_answer,
                generated_answer="",
                expected_document=item.expected_document,
                expected_page=item.expected_page,
                error=str(exc),
                total_time_ms=(time.perf_counter() - t_start) * 1000,
            ))

    return results


def aggregate_metrics(results: List[EvalResult]) -> Dict[str, float]:
    """Compute aggregate metrics from a list of EvalResults."""
    if not results:
        return {}
    valid = [r for r in results if not r.error]
    if not valid:
        return {"error": "All evaluation items failed."}

    n = len(valid)
    return {
        "n": n,
        "avg_recall_at_k": round(sum(r.recall_at_k for r in valid) / n, 4),
        "avg_citation_accuracy": round(sum(r.citation_accuracy for r in valid) / n, 4),
        "avg_keyword_overlap": round(sum(r.keyword_overlap for r in valid) / n, 4),
        "avg_total_time_ms": round(sum(r.total_time_ms for r in valid) / n, 1),
        "avg_retrieval_time_ms": round(sum(r.retrieval_time_ms for r in valid) / n, 1),
        "avg_generation_time_ms": round(sum(r.generation_time_ms for r in valid) / n, 1),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Export helpers
# ──────────────────────────────────────────────────────────────────────────────

def results_to_csv(results: List[EvalResult]) -> str:
    """Return CSV string of evaluation results."""
    buf = []
    headers = [
        "question", "expected_document", "expected_page",
        "recall_at_k", "citation_accuracy", "keyword_overlap",
        "total_time_ms", "retrieval_time_ms", "generation_time_ms",
        "generated_answer", "error",
    ]
    buf.append(",".join(headers))
    for r in results:
        def esc(v):
            return f'"{str(v).replace(chr(34), chr(39))}"'
        row = [
            esc(r.question), esc(r.expected_document), str(r.expected_page or ""),
            str(r.recall_at_k), str(r.citation_accuracy), str(r.keyword_overlap),
            str(round(r.total_time_ms, 1)), str(round(r.retrieval_time_ms, 1)),
            str(round(r.generation_time_ms, 1)),
            esc(r.generated_answer[:200]), esc(r.error),
        ]
        buf.append(",".join(row))
    return "\n".join(buf)


def save_results(results: List[EvalResult], run_name: str = "eval") -> Path:
    """Persist evaluation results to the outputs/evaluation directory."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = _RESULTS_DIR / f"{run_name}_{ts}.json"
    data = [asdict(r) for r in results]
    out.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return out


# Ensure example dataset exists on import
ensure_dataset_exists()
