# Architecture Documentation

## Multimodal Document Intelligence System — Module Reference

---

## Overview

The system is divided into six logical layers, each implemented as a separate Python module under `modules/`. Data flows downward through these layers; no lower layer imports from a higher one.

```
┌─────────────────────────────────────────────────────┐
│           PRESENTATION LAYER (Streamlit pages)       │
│  app.py · pages/1_Dashboard.py · pages/4_AI_Asst…   │
├─────────────────────────────────────────────────────┤
│             ORCHESTRATION LAYER                      │
│    rag_pipeline.py · document_processor.py           │
├─────────────────────────────────────────────────────┤
│             INTELLIGENCE LAYER                       │
│  llm_handler.py · vision_processor.py · summarizer  │
├─────────────────────────────────────────────────────┤
│             RETRIEVAL LAYER                          │
│       embeddings.py · vector_database.py             │
├─────────────────────────────────────────────────────┤
│             PROCESSING LAYER                         │
│   chunking.py · table_extractor.py · evaluation.py  │
├─────────────────────────────────────────────────────┤
│             FOUNDATION LAYER                         │
│        config.py · database_manager.py               │
└─────────────────────────────────────────────────────┘
```

---

## Module-by-Module Reference

---

### `modules/config.py`

**Purpose:** Central configuration authority.

Loads all settings from `.env` via `python-dotenv`, exposes them as typed module-level constants. Every other module imports from `config.py`. Configuration is never hardcoded elsewhere.

**Key exports:**
- `OLLAMA_MODEL`, `OLLAMA_VISION_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_TIMEOUT`
- `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `EMBEDDING_BATCH_SIZE`
- `CHROMA_PERSIST_DIR`, `CHROMA_COLLECTION_NAME`
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`
- `UPLOAD_DIR`, `PROCESSED_DIR`, `EXTRACTED_IMAGES_DIR`, `METADATA_DB_PATH`

---

### `modules/database_manager.py`

**Purpose:** Persistent SQLite metadata store.

Maintains the document library across Streamlit restarts. All document records, processing statistics, and activity logs live here — not in session state.

**Schema:**
- `documents` — one row per uploaded document with full lifecycle metadata
- `activity_log` — append-only audit log of uploads, updates, and deletions
- `app_settings` — key-value configuration store

**Thread safety:** Uses a `threading.Lock` and WAL journal mode. Safe under Streamlit's single-process model.

**Key functions:**
- `insert_document()` — creates a new document record, returns `doc_id`
- `update_document_status()` — updates status, page count, table/image/chunk counts
- `get_all_documents()` — returns all documents ordered by upload time
- `delete_document()` — removes the database row (caller must clean up files)
- `compute_file_hash()` — SHA-256 for duplicate detection
- `get_statistics()` — aggregate counts for the dashboard

---

### `modules/document_processor.py`

**Purpose:** Full document ingestion orchestration.

The entry point for processing any uploaded file. Validates the upload, saves it, runs Docling conversion, extracts tables and images, saves artefacts, and updates the database.

**Processing flow:**
```
validate_upload()
  → compute_file_hash() → duplicate check
  → save to UPLOAD_DIR
  → insert_document() in SQLite
  → Docling DocumentConverter.convert()
  → _extract_tables_from_docling()
  → _extract_images_pymupdf() (PDF) or copy image (image files)
  → save extracted.md + structured.json to PROCESSED_DIR/doc_id/
  → update_document_status(completed)
  → return ProcessingResult
```

**Fallbacks:**
- PPTX/DOCX that fail Docling fall back to `python-pptx` / `python-docx` parsers
- Image files go through Docling OCR, then the original is saved as the extracted image

**Key dataclasses:**
- `ProcessingResult` — everything downstream modules need
- `ExtractedTable` — structured table with headers, rows, page, caption
- `ExtractedImage` — image metadata with local path (description added later by VLM)

---

### `modules/chunking.py`

**Purpose:** Section-aware document chunking.

Converts `ProcessingResult` into a flat list of `DocumentChunk` objects suitable for embedding and indexing.

**Strategy:**
1. Walk Markdown line-by-line, detecting headings (`#`, `##`, `###`).
2. Accumulate lines under the current heading until buffer exceeds `CHUNK_SIZE`.
3. When flushing, run a sliding-window paragraph-boundary splitter if the section is still too large.
4. Each chunk carries its heading as context so retrieval is section-aware.
5. Tables become individual chunks (`content_type="table"`).
6. VLM image descriptions become individual chunks (`content_type="image_description"`).

**Every chunk contains:**
```json
{
  "chunk_id": "uuid",
  "doc_id": "uuid",
  "filename": "paper.pdf",
  "page": 12,
  "section": "Experimental Results",
  "content_type": "text",
  "text": "The proposed model achieved…",
  "char_count": 342
}
```

---

### `modules/embeddings.py`

**Purpose:** Sentence Transformer embedding generation.

Loads `all-MiniLM-L6-v2` once per process (singleton pattern). Provides batch processing with configurable batch size to stay within CPU RAM limits.

**Key functions:**
- `embed_texts(texts, normalize=True)` → `np.ndarray (N, 384)`
- `embed_query(query)` → `List[float]` (ready for ChromaDB)
- `embed_chunks(chunks)` → `List[List[float]]`
- `embedding_model_status()` → status dict for the settings page

**Normalisation:** All vectors are L2-normalised so that cosine similarity equals the dot product — ChromaDB's native metric for its `hnsw:space=cosine` collection.

---

### `modules/vector_database.py`

**Purpose:** ChromaDB persistent vector store.

Opens a single persistent `PersistentClient` and manages one collection (`document_chunks`). All vectors include metadata that supports filtered retrieval.

**Stored metadata per chunk:**
- `doc_id`, `filename`, `page`, `section`, `content_type`, `char_count`

**Key operations:**
- `add_chunks(chunks, embeddings)` — idempotent insertion (skips existing IDs)
- `search(query_embedding, top_k, doc_ids, content_types)` — filtered similarity search
- `delete_document_vectors(doc_id)` — removes all vectors for a document
- `collection_stats()` — total chunk count and storage path

**Filter enforcement:** Document filters are applied inside the ChromaDB `where` clause — not post-hoc in Python — preventing information leakage between documents.

---

### `modules/llm_handler.py`

**Purpose:** Ollama REST API wrapper.

Handles all communication with the Ollama server. Never assumes a model is running — always checks first and surfaces clear error messages.

**Vision detection:** Uses both a known-model-family prefix check and the Ollama `/api/show` endpoint to determine if a model supports image inputs.

**Key functions:**
- `is_ollama_running()` — health ping
- `list_installed_models()` — returns installed model names
- `model_supports_vision(model_name)` — heuristic + API check
- `generate_text(prompt, system_prompt, …)` — single-turn text generation
- `generate_text_chat(messages, …)` — multi-turn with conversation history
- `describe_image(image_path, context_prompt, …)` → `(description, success)`
- `answer_about_image(image_path, question, …)` → `(answer, success)`

---

### `modules/vision_processor.py`

**Purpose:** Batch VLM image description for document images.

Iterates over `ProcessingResult.images`, screens each image for size/quality, sends qualifying images to the VLM via `llm_handler.describe_image`, and persists results.

**Screening criteria:**
- Minimum 50×50 pixels
- Minimum 2 KB file size

**Persistence:** Descriptions saved to `PROCESSED_DIR/doc_id/image_descriptions.json`. Loaded by the Visual Analysis page and by `chunking.chunk_image_description()`.

---

### `modules/rag_pipeline.py`

**Purpose:** Complete RAG pipeline — question → answer.

**Pipeline steps:**
1. Validate question (reject empty strings).
2. Check Ollama is running.
3. Embed query via `embeddings.embed_query`.
4. Retrieve chunks from ChromaDB via `vector_database.search` (document-filtered).
5. De-duplicate near-identical chunks.
6. Format context string with source labels.
7. Build system prompt with strict grounding rules.
8. Append conversation history (last 6 turns) for multi-turn memory.
9. Call `llm_handler.generate_text_chat`.
10. Package `RAGResponse` with answer + `SourceCitation` objects.

**Also provides `index_document()`:** The single function called after processing to chunk → embed → store a document.

**Grounding rules (encoded in system prompt):**
- Answer only from supplied evidence.
- Never invent facts.
- Cite document + page when metadata is reliable.
- State contradictions across documents explicitly.
- State when information is unavailable.

---

### `modules/summarizer.py`

**Purpose:** Document summarization with 9 summary types.

For documents within `_MAX_CONTEXT_CHARS` (6,000 chars), uses a single LLM call. For longer documents, applies **map-reduce**:

```
Long document
  │
  ▼ (split into 3,000-char chunks)
Map: summarise each chunk in 3-5 sentences
  │
  ▼ (combine partial summaries)
Reduce: synthesise into final summary of chosen type
```

This ensures long documents are never silently truncated.

---

### `modules/document_comparator.py`

**Purpose:** Multi-document comparison engine.

For each combination of `(dimension, document)`, retrieves the top-3 most relevant chunks for that dimension, then asks the LLM to extract the value. Results are assembled into a structured table.

**Dimensions:** Research Objective, Problem Statement, Proposed Methodology, Model/Architecture, Dataset, Evaluation Metrics, Results, Limitations, Conclusion.

Finally, an optional narrative LLM call synthesises the table into a comparative analysis paragraph.

---

### `modules/table_extractor.py`

**Purpose:** Table loading and export utilities.

Reads `structured.json` for a given document, converts table dicts to pandas DataFrames, and provides CSV/Excel/Markdown export. Handles partially-structured tables gracefully using a Markdown fallback.

---

### `modules/evaluation.py`

**Purpose:** Evaluation framework with real (non-fabricated) metrics.

Manages a CSV-based evaluation dataset and runs the full RAG pipeline against each item, measuring:
- **Recall@K** — presence of expected source in retrieved chunks
- **Citation Accuracy** — fraction of retrieved chunks from the expected document
- **Keyword Overlap F1** — token-level overlap between expected and generated answers
- **Response time** — total, retrieval, generation

Results are exportable as CSV and persisted as JSON under `data/outputs/evaluation/`.

---

## Data Flow — End to End

```
Upload (bytes)
│
├─ validate_upload() → reject if invalid
├─ compute_file_hash() → reject if duplicate
├─ save to data/uploads/{doc_id}_{filename}
├─ Docling.convert() → markdown + tables + image refs
├─ PyMuPDF → extract embedded images
├─ [optional] VLM → describe each image
├─ save data/processed/{doc_id}/extracted.md
├─ save data/processed/{doc_id}/structured.json
├─ save data/processed/{doc_id}/image_descriptions.json
│
├─ chunk_markdown() → List[DocumentChunk]
├─ chunk_table() for each table
├─ chunk_image_description() for each described image
│
├─ embed_chunks() → List[List[float]]
├─ add_chunks() → ChromaDB
└─ update_document_status(completed, chunk_count=N)
```

```
Question
│
├─ embed_query() → [0.1, 0.3, …]  (384-dim)
├─ ChromaDB.query(where={doc_id: in selected_docs})
├─ deduplicate results
├─ build context string with [N] Source: … labels
├─ system_prompt + conversation_history + question
├─ Ollama /api/chat → answer text
└─ RAGResponse(answer, citations=[SourceCitation…])
```

---

## Sequence Diagram — Document Upload

```
User → Streamlit → document_processor.process_document()
                        │
                        ├── validate_upload()
                        ├── database_manager.insert_document()
                        ├── Docling.convert()
                        ├── PyMuPDF image extraction
                        ├── [vision_processor.describe_images()]
                        ├── database_manager.update_document_status()
                        │
                        └── rag_pipeline.index_document()
                                │
                                ├── chunking.create_chunks()
                                ├── embeddings.embed_chunks()
                                ├── vector_database.add_chunks()
                                └── database_manager.update(chunk_count)
```

## Sequence Diagram — RAG Question Answering

```
User → Streamlit → rag_pipeline.answer_question()
                        │
                        ├── embeddings.embed_query()
                        ├── vector_database.search(doc_ids=[…])
                        ├── _deduplicate_chunks()
                        ├── _build_context()
                        ├── llm_handler.generate_text_chat()
                        └── RAGResponse → Streamlit displays answer + citations
```
