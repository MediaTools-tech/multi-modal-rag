# Changelog

All notable changes to DeepLens are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Summary-Aware Hybrid Search** — documents are now indexed both as
  fine-grained chunks *and* as a single per-file `summary` vector record
  (`record_type="summary"`), enabling retrieval of whole files from a
  description of their content (e.g. "the story about a man who owns a big mansion").
  - New `search_mode` selector: `chunk`, `summary`, or `hybrid` (default `hybrid`).
  - Hybrid retrieval fuses three ranked lists via Reciprocal Rank Fusion (RRF):
    semantic vector, BM25-style lexical keyword overlap, and document-summary ranking.
  - `FileSearchGroup` results group matches by source file with generated summary,
    best score, and top supporting chunks.
- `DocumentSummarizer` (`ingestion/summarizer.py`) generates per-file summaries at
  ingestion time using the chat engine.
- `ChatEngine.summarize_document()` default method (overridable by concrete engines).
- `search/hybrid.py` module with `hybrid_search()` and `summary_search()` helpers,
  plus lexical (BM25-style) scoring and RRF fusion.
- Config settings: `search_mode`, `hybrid_lexical_weight`, `hybrid_rrf_k`,
  `enable_document_summaries`, `summary_max_chars`.
- New MCP tool `find_documents(description, ...)` that returns matched files with
  their summaries; `semantic_search` now accepts a `search_mode` argument.
- `record_type` / `summary` columns added to both LanceDB and pgvector schemas
  (pgvector migrates automatically; LanceDB migrates best-effort on open).

### Changed

- `VectorRecord` gains `record_type` and `summary` fields; `to_dict`/`from_dict`
  updated accordingly.
- `DocumentRepository.search()` gains an optional `record_types` filter (applied
  in-memory after a larger candidate fetch so it works on pre-summary schemas).
- `SearchPipeline.search()` accepts a `search_mode` override.
- Generator now labels summary records as "Document Summary" and leads with the
  matched-files section when present.
- `IngestionQueue` now accepts a `chat_engine` to build summary records.
- README expanded with a "Summary-Aware Hybrid Search" section (problem statement,
  why RAPTOR-Lite / per-file K-Means clustering was not chosen, mode table, and
  configuration reference).

### Migration

- Enabling summaries on an **existing** index adds the new `record_type` / `summary`
  columns automatically. To populate summaries for already-indexed files, re-index
  the folder.

## [0.1.0] - Initial

- Foundation: abstract interfaces, factory loading, LanceDB / pgvector tables.
- Ingestion & parsing: Docling markdown parser, image EXIF, watchdog monitor.
- Decoupled media chunkers: OpenCV frame extractor, audio WAV converter,
  faster-whisper worker, subtitle parsers.
- Agentic RAG: LangGraph loop (Rewriter → Retriever → Evaluator → Generator).
- MCP tooling: FastMCP stdio server.
- Native client layout: PySide6 window, QThread workers.
