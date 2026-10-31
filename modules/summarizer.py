"""
modules/summarizer.py
======================
Document summarization with multiple summary types.

For documents that exceed the LLM context window, uses a map-reduce
approach: summarize each chunk independently, then combine.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import List, Optional

from modules.config import OLLAMA_MODEL, TOP_K
from modules.llm_handler import generate_text, is_ollama_running

logger = logging.getLogger(__name__)

# Approx tokens a small LLM (~3B) can handle at once
_MAX_CONTEXT_CHARS = 6_000
# Map-reduce: first summarize each chunk, then combine all chunk summaries
_MAP_CHUNK_CHARS = 3_000


class SummaryType(str, Enum):
    SHORT = "short"
    DETAILED = "detailed"
    EXECUTIVE = "executive"
    SECTION_WISE = "section_wise"
    RESEARCH_PAPER = "research_paper"
    TECHNICAL = "technical"
    KEY_FINDINGS = "key_findings"
    ADVANTAGES_LIMITATIONS = "advantages_limitations"
    CONCLUSION = "conclusion"


# ──────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────────────────────────────────────

_PROMPTS = {
    SummaryType.SHORT: (
        "Provide a concise 3-5 sentence summary of the following document excerpt:\n\n{text}"
    ),
    SummaryType.DETAILED: (
        "Provide a comprehensive and detailed summary of the following document. "
        "Cover all major topics, methods, results, and conclusions:\n\n{text}"
    ),
    SummaryType.EXECUTIVE: (
        "Write a one-page executive summary suitable for a business audience. "
        "Focus on key takeaways, recommendations, and outcomes:\n\n{text}"
    ),
    SummaryType.SECTION_WISE: (
        "Summarise this document section by section. "
        "Identify each section by its heading and provide a 2-3 sentence summary:\n\n{text}"
    ),
    SummaryType.RESEARCH_PAPER: (
        "Analyse this research paper and extract the following structured sections. "
        "If a section is not present in the document, state 'Not specified'.\n\n"
        "## Research Objective\n## Problem Statement\n## Motivation\n"
        "## Proposed Methodology\n## Model Architecture\n## Dataset\n"
        "## Experimental Setup\n## Results\n## Evaluation Metrics\n"
        "## Advantages\n## Limitations\n## Conclusion\n\n"
        "Document:\n{text}"
    ),
    SummaryType.TECHNICAL: (
        "Provide a detailed technical summary of this document. "
        "Focus on methods, algorithms, technologies, architectures, and implementation details:\n\n{text}"
    ),
    SummaryType.KEY_FINDINGS: (
        "Extract and list the most important findings, discoveries, or insights "
        "from this document. Present them as numbered bullet points:\n\n{text}"
    ),
    SummaryType.ADVANTAGES_LIMITATIONS: (
        "Identify and clearly describe:\n"
        "**Advantages/Strengths:** What does this work do well?\n"
        "**Limitations/Weaknesses:** What are the shortcomings or gaps?\n\n"
        "Document:\n{text}"
    ),
    SummaryType.CONCLUSION: (
        "Extract and summarise the conclusions and future work from this document. "
        "Be precise and evidence-based:\n\n{text}"
    ),
}

_MAP_PROMPT = "Summarise the following passage in 3-5 sentences:\n\n{text}"

_REDUCE_PROMPT = (
    "You have received multiple partial summaries from a long document. "
    "Synthesise them into a single coherent {summary_type} summary:\n\n"
    "{combined_summaries}"
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _split_for_map(text: str, max_chars: int = _MAP_CHUNK_CHARS) -> List[str]:
    """Split text into chunks suitable for the map step."""
    parts = []
    for i in range(0, len(text), max_chars):
        parts.append(text[i: i + max_chars])
    return parts


def _map_reduce_summarize(
    text: str,
    summary_type: SummaryType,
    model: str,
    base_url: str,
) -> str:
    """Summarize a long document using map-reduce."""
    chunks = _split_for_map(text)
    logger.info("Map-reduce: %d chunks for summary type '%s'", len(chunks), summary_type)

    # Map step
    partial_summaries: List[str] = []
    for i, chunk in enumerate(chunks):
        try:
            s = generate_text(
                prompt=_MAP_PROMPT.format(text=chunk),
                model=model,
                base_url=base_url,
                temperature=0.1,
                max_tokens=512,
            )
            partial_summaries.append(s)
        except Exception as exc:
            logger.warning("Map step chunk %d failed: %s", i, exc)
            partial_summaries.append(chunk[:300] + "…")

    combined = "\n\n---\n\n".join(partial_summaries)

    # Reduce step
    prompt = _REDUCE_PROMPT.format(
        summary_type=summary_type.value.replace("_", " "),
        combined_summaries=combined,
    )
    return generate_text(
        prompt=prompt,
        model=model,
        base_url=base_url,
        temperature=0.2,
        max_tokens=2048,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def summarize_document(
    doc_id: str,
    summary_type: SummaryType = SummaryType.DETAILED,
    model: str = OLLAMA_MODEL,
    base_url: Optional[str] = None,
) -> str:
    """
    Generate a summary for an already-indexed document.

    Reads the extracted Markdown from disk and passes it to the LLM.
    Uses map-reduce for documents that exceed the context window.
    """
    from modules.config import OLLAMA_BASE_URL, PROCESSED_DIR

    _base_url = base_url or OLLAMA_BASE_URL

    if not is_ollama_running(_base_url):
        return "❌ Ollama is not running. Start it with: ollama serve"

    # Load extracted text
    md_file = PROCESSED_DIR / doc_id / "extracted.md"
    if not md_file.exists():
        return "❌ Document has not been processed yet. Please index it first."

    text = md_file.read_text(encoding="utf-8").strip()
    if not text:
        return "❌ No text content found in this document."

    template = _PROMPTS.get(summary_type, _PROMPTS[SummaryType.DETAILED])

    if len(text) <= _MAX_CONTEXT_CHARS:
        # Single-pass summarization
        prompt = template.format(text=text)
        try:
            return generate_text(
                prompt=prompt,
                model=model,
                base_url=_base_url,
                temperature=0.2,
                max_tokens=2048,
            )
        except RuntimeError as exc:
            return f"❌ Summarization failed: {exc}"
    else:
        # Map-reduce for long documents
        try:
            return _map_reduce_summarize(text, summary_type, model, _base_url)
        except RuntimeError as exc:
            return f"❌ Summarization failed: {exc}"


def summarize_text(
    text: str,
    summary_type: SummaryType = SummaryType.SHORT,
    model: str = OLLAMA_MODEL,
    base_url: Optional[str] = None,
) -> str:
    """Summarise arbitrary text (used for table/image content summarisation)."""
    from modules.config import OLLAMA_BASE_URL
    _base_url = base_url or OLLAMA_BASE_URL

    template = _PROMPTS.get(summary_type, _PROMPTS[SummaryType.SHORT])
    if len(text) > _MAX_CONTEXT_CHARS:
        return _map_reduce_summarize(text, summary_type, model, _base_url)
    prompt = template.format(text=text[:_MAX_CONTEXT_CHARS])
    return generate_text(
        prompt=prompt, model=model, base_url=_base_url, temperature=0.2, max_tokens=1024
    )
