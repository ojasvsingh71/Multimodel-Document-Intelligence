"""
modules/document_comparator.py
================================
Compare two or more documents across standard dimensions.
Produces both a structured table and a natural-language narrative.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from modules.config import OLLAMA_MODEL, TOP_K
from modules.embeddings import embed_query
from modules.llm_handler import generate_text, is_ollama_running
from modules.vector_database import search

logger = logging.getLogger(__name__)

_DIMENSIONS = [
    "Research Objective",
    "Problem Statement",
    "Proposed Methodology",
    "Model / Architecture",
    "Dataset",
    "Evaluation Metrics",
    "Results",
    "Limitations",
    "Conclusion",
]

_EXTRACT_PROMPT = """\
Based on the following document excerpts from '{filename}', extract the value for: **{dimension}**

If the information is not present, respond with exactly: Not specified

Excerpts:
{context}

Value for "{dimension}":"""

_COMPARE_PROMPT = """\
You are comparing {n} research documents. Here is a structured comparison table:

{table_json}

Write a clear, evidence-based comparative analysis covering:
1. Major similarities between the documents.
2. Key differences.
3. Which approach appears more effective (if assessable from the evidence) and why.
4. Important caveats and limitations to this comparison.

Be objective, cite the document names, and avoid making unsupported claims.
"""


def _retrieve_context_for_dimension(
    dimension: str,
    doc_id: str,
    top_k: int = 3,
) -> str:
    """Retrieve the most relevant chunks for a specific dimension from one document."""
    try:
        query_vec = embed_query(dimension)
        chunks = search(query_embedding=query_vec, top_k=top_k, doc_ids=[doc_id])
        return "\n\n".join(c["text"] for c in chunks)
    except Exception as exc:
        logger.warning("Context retrieval failed for dimension '%s': %s", dimension, exc)
        return ""


def extract_dimension_value(
    dimension: str,
    doc_id: str,
    filename: str,
    model: str,
    base_url: str,
) -> str:
    """Extract a single dimension value from a document using RAG + LLM."""
    context = _retrieve_context_for_dimension(dimension, doc_id)
    if not context:
        return "Not specified"

    prompt = _EXTRACT_PROMPT.format(
        filename=filename,
        dimension=dimension,
        context=context[:3000],
    )
    try:
        value = generate_text(
            prompt=prompt,
            model=model,
            base_url=base_url,
            temperature=0.1,
            max_tokens=300,
        )
        return value.strip() or "Not specified"
    except Exception as exc:
        logger.warning("Dimension extraction failed (%s, %s): %s", dimension, filename, exc)
        return "Extraction failed"


def compare_documents(
    documents: List[Dict[str, str]],   # [{"doc_id": ..., "filename": ...}, ...]
    dimensions: Optional[List[str]] = None,
    model: str = OLLAMA_MODEL,
    base_url: Optional[str] = None,
    include_narrative: bool = True,
) -> Dict[str, Any]:
    """
    Compare documents across a set of dimensions.

    Returns:
        {
          "table": {dimension: {filename: value, ...}, ...},
          "narrative": "...",
          "documents": [...],
          "error": ""
        }
    """
    from modules.config import OLLAMA_BASE_URL
    _base_url = base_url or OLLAMA_BASE_URL

    if not is_ollama_running(_base_url):
        return {
            "table": {}, "narrative": "",
            "documents": documents,
            "error": "Ollama is not running.",
        }

    _dimensions = dimensions or _DIMENSIONS

    table: Dict[str, Dict[str, str]] = {}

    for dim in _dimensions:
        table[dim] = {}
        for doc in documents:
            value = extract_dimension_value(
                dimension=dim,
                doc_id=doc["doc_id"],
                filename=doc["filename"],
                model=model,
                base_url=_base_url,
            )
            table[dim][doc["filename"]] = value

    narrative = ""
    if include_narrative:
        try:
            table_json = json.dumps(table, indent=2)
            prompt = _COMPARE_PROMPT.format(
                n=len(documents),
                table_json=table_json[:4000],
            )
            narrative = generate_text(
                prompt=prompt,
                model=model,
                base_url=_base_url,
                temperature=0.3,
                max_tokens=2048,
            )
        except Exception as exc:
            logger.warning("Narrative generation failed: %s", exc)
            narrative = "Narrative generation failed. See the table for details."

    return {
        "table": table,
        "narrative": narrative,
        "documents": documents,
        "error": "",
    }


def answer_comparison_question(
    question: str,
    doc_ids: List[str],
    model: str = OLLAMA_MODEL,
    base_url: Optional[str] = None,
) -> str:
    """Answer a free-form comparative question about multiple documents."""
    from modules.rag_pipeline import answer_question
    response = answer_question(
        question=question,
        doc_ids=doc_ids,
        top_k=TOP_K * 2,
        model=model,
        base_url=base_url,
    )
    return response.answer or response.error
