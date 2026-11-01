"""
modules/rag_pipeline.py
========================
Custom Retrieval-Augmented Generation pipeline.

Flow:
  1. Validate the user question.
  2. Embed the question.
  3. Retrieve top-K relevant chunks from ChromaDB (filtered by selected docs).
  4. Rank and de-duplicate results.
  5. Build a structured context string.
  6. Construct a grounded system prompt.
  7. Call the LLM with conversation history.
  8. Return the answer + source citations.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from modules.config import OLLAMA_MODEL, TOP_K
from modules.embeddings import embed_query
from modules.llm_handler import generate_text_chat, is_ollama_running
from modules.vector_database import search

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SourceCitation:
    filename: str
    page: Optional[int]
    section: str
    content_type: str
    score: float
    excerpt: str          # first 300 chars of the retrieved chunk


@dataclass
class RAGResponse:
    answer: str
    citations: List[SourceCitation] = field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    query_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    error: str = ""

    @property
    def success(self) -> bool:
        return not self.error and bool(self.answer)


# ──────────────────────────────────────────────────────────────────────────────
# System prompt template
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert document analyst and research assistant. \
You answer questions **exclusively** using the evidence provided below from the user's documents.

Rules you must follow:
1. Answer only from the supplied evidence. Do NOT invent facts.
2. If the evidence does not contain the answer, say so clearly.
3. Cite your sources: mention the document filename and page number when available.
4. If different documents contradict each other, note the contradiction.
5. Distinguish between what the documents state and your own interpretation.
6. Never claim a source supports something it does not contain.
7. Be concise, accurate, and helpful.

---
EVIDENCE FROM DOCUMENTS:
{context}
---
"""


def _build_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a readable context block."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        page_info = f"p.{chunk['page']}" if chunk.get("page") else "page unknown"
        parts.append(
            f"[{i}] Source: {chunk['filename']} ({page_info}) "
            f"| Section: {chunk.get('section', 'N/A')} "
            f"| Type: {chunk.get('content_type', 'text')}\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(parts) if parts else "No relevant content found."


def _deduplicate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove near-duplicate chunks (same doc_id + page + first 100 chars)."""
    seen: set = set()
    unique = []
    for c in chunks:
        key = (c.get("doc_id"), c.get("page"), c.get("text", "")[:100])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


# ──────────────────────────────────────────────────────────────────────────────
# Core RAG function
# ──────────────────────────────────────────────────────────────────────────────

def answer_question(
    question: str,
    doc_ids: Optional[List[str]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    top_k: int = TOP_K,
    model: str = OLLAMA_MODEL,
    base_url: Optional[str] = None,
) -> RAGResponse:
    """
    Full RAG pipeline: question → retrieval → context → LLM → answer.

    Args:
        question:             The user's natural-language question.
        doc_ids:              Limit retrieval to these document IDs (None = all).
        conversation_history: Prior {role, content} messages for memory.
        top_k:                Number of chunks to retrieve.
        model:                Ollama model name.
        base_url:             Ollama base URL (defaults to config).

    Returns:
        RAGResponse with answer and citations.
    """
    from modules.config import OLLAMA_BASE_URL
    _base_url = base_url or OLLAMA_BASE_URL

    t_start = time.perf_counter()

    # Validate
    question = question.strip()
    if not question:
        return RAGResponse(answer="", error="Question cannot be empty.")

    q_clean = question.lower().strip("?!. ")

    # Conversational Greeting
    if q_clean in {"hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "howdy"}:
        from modules.database_manager import get_all_documents
        docs = get_all_documents()
        doc_names = [d["filename"] for d in docs if (d.get("chunk_count") or 0) > 0]
        if doc_names:
            doc_list_str = "\n".join(f"- **{name}**" for name in doc_names)
            return RAGResponse(
                answer=(
                    f"Hello! I am your Multimodal Document Intelligence Assistant.\n\n"
                    f"I have access to the following indexed documents:\n{doc_list_str}\n\n"
                    "Feel free to ask questions about methodologies, models, architectures, results, tables, or figures!"
                )
            )
        else:
            return RAGResponse(
                answer="Hello! I am your Multimodal Document Intelligence Assistant.\n\n"
                       "You have not indexed any documents yet. Please upload files in the **Upload Documents** tab to begin."
            )

    # Document Inventory / Overview query
    if any(t in q_clean for t in ["what information do you have", "what info do you have", "what documents", "what do you know", "what can you do", "help"]):
        from modules.database_manager import get_all_documents
        docs = get_all_documents()
        if docs:
            lines = ["Here are the documents currently loaded in your system:\n"]
            for d in docs:
                chunks = d.get("chunk_count", 0)
                pages = d.get("page_count", "unknown")
                tables = d.get("table_count", 0)
                imgs = d.get("image_count", 0)
                status = "✅ Indexed" if chunks > 0 else "⚠️ Not indexed (0 chunks)"
                lines.append(
                    f"- **{d['filename']}** ({status}) — {pages} pages, {chunks} chunks, {tables} tables, {imgs} images"
                )
            lines.append("\nYou can ask me to summarize them, compare approaches, explain specific tables or figures, or answer specific research questions.")
            return RAGResponse(answer="\n".join(lines))

    if not is_ollama_running(_base_url):
        return RAGResponse(
            answer="",
            error="Ollama is not running. Start it with: ollama serve",
        )

    # 1. Embed query
    t_emb = time.perf_counter()
    try:
        query_vec = embed_query(question)
    except Exception as exc:
        return RAGResponse(answer="", error=f"Embedding failed: {exc}")
    query_time_ms = (time.perf_counter() - t_emb) * 1000

    # 2. Retrieve
    t_ret = time.perf_counter()
    raw_chunks = search(query_embedding=query_vec, top_k=top_k * 2, doc_ids=doc_ids)
    # De-duplicate and take top_k
    chunks = _deduplicate_chunks(raw_chunks)[:top_k]
    retrieval_time_ms = (time.perf_counter() - t_ret) * 1000

    if not chunks:
        from modules.vector_database import get_collection
        total_in_db = get_collection().count()
        if total_in_db == 0:
            msg = (
                "⚠️ **No document chunks are indexed in the vector database.**\n\n"
                "Please go to **Document Library** and click **🔄 Re-Index**, "
                "or upload your documents again via **Upload Documents**."
            )
        else:
            msg = (
                "I could not find relevant content matching your query in the selected documents.\n\n"
                "- Try rephrasing your question or using different keywords.\n"
                "- Ensure the intended document is selected in the sidebar."
            )
        return RAGResponse(
            answer=msg,
            query_time_ms=query_time_ms,
            retrieval_time_ms=retrieval_time_ms,
        )

    # 3. Build context
    context = _build_context(chunks)
    system_content = _SYSTEM_PROMPT.format(context=context)

    # 4. Build message list for multi-turn memory
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_content}]

    # Inject limited conversation history (last 6 turns to stay within context)
    if conversation_history:
        for msg in conversation_history[-6:]:
            if msg.get("role") in {"user", "assistant"}:
                messages.append(msg)

    messages.append({"role": "user", "content": question})

    # 5. Generate
    t_gen = time.perf_counter()
    try:
        answer = generate_text_chat(
            messages=messages,
            model=model,
            base_url=_base_url,
            temperature=0.2,
            max_tokens=2048,
        )
    except RuntimeError as exc:
        return RAGResponse(
            answer="",
            error=str(exc),
            query_time_ms=query_time_ms,
            retrieval_time_ms=retrieval_time_ms,
        )
    generation_time_ms = (time.perf_counter() - t_gen) * 1000

    # 6. Build citations
    citations = [
        SourceCitation(
            filename=c["filename"],
            page=c.get("page"),
            section=c.get("section", ""),
            content_type=c.get("content_type", "text"),
            score=c.get("score", 0.0),
            excerpt=c["text"][:300],
        )
        for c in chunks
    ]

    logger.info(
        "RAG complete — %d chunks | query %.0fms | retrieval %.0fms | gen %.0fms",
        len(chunks), query_time_ms, retrieval_time_ms, generation_time_ms,
    )

    return RAGResponse(
        answer=answer,
        citations=citations,
        retrieved_chunks=chunks,
        query_time_ms=query_time_ms,
        retrieval_time_ms=retrieval_time_ms,
        generation_time_ms=generation_time_ms,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Indexing helper  (called after document processing)
# ──────────────────────────────────────────────────────────────────────────────

def index_document(
    processing_result,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    run_vision: bool = False,
    vision_model: Optional[str] = None,
) -> Tuple[int, str]:
    """
    Chunk → embed → index a ProcessingResult.

    Args:
        processing_result: Output from document_processor.process_document.
        chunk_size:        Override default chunk size.
        overlap:           Override default overlap.
        run_vision:        If True, describe images via VLM first.
        vision_model:      VLM model name override.

    Returns:
        (num_chunks_indexed, error_message)
    """
    from modules.chunking import create_chunks
    from modules.config import CHUNK_OVERLAP, CHUNK_SIZE
    from modules.database_manager import update_document_status
    from modules.embeddings import embed_chunks
    from modules.vector_database import add_chunks

    if processing_result.status not in {"completed", "partial"}:
        return 0, f"Document status is '{processing_result.status}' — skipping indexing."

    _chunk_size = chunk_size or CHUNK_SIZE
    _overlap = overlap or CHUNK_OVERLAP

    # Optional: describe images before chunking so descriptions are included
    if run_vision and processing_result.images:
        from modules.vision_processor import describe_document_images
        from modules.config import OLLAMA_VISION_MODEL
        _vm = vision_model or OLLAMA_VISION_MODEL
        try:
            describe_document_images(processing_result, vision_model=_vm)
        except Exception as exc:
            logger.warning("Vision processing failed (continuing without it): %s", exc)

    # Create chunks
    chunks = create_chunks(processing_result, chunk_size=_chunk_size, overlap=_overlap)
    if not chunks:
        return 0, "No chunks generated — document may be empty."

    # Generate embeddings
    try:
        embeddings = embed_chunks(chunks)
    except Exception as exc:
        return 0, f"Embedding generation failed: {exc}"

    # Insert into ChromaDB
    try:
        inserted = add_chunks(chunks, embeddings)
    except Exception as exc:
        return 0, f"ChromaDB insertion failed: {exc}"

    # Update DB chunk count
    update_document_status(
        doc_id=processing_result.doc_id,
        status="completed",
        chunk_count=inserted,
    )

    return inserted, ""
