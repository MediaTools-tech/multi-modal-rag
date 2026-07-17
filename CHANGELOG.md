# Changelog

All notable changes to DeepLens are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Summary-Aware Hybrid Search

- **Document summaries** — each text-bearing file is now indexed both as
  fine-grained chunks *and* as a single per-file `summary` vector record
  (`record_type="summary"`), enabling retrieval of whole files from a description
  of their content (e.g. "the story about a man who owns a big mansion").
  - `search_mode` selector: `chunk`, `summary`, or `hybrid` (default `hybrid`).
  - `FileSearchGroup` results group matches by source file with the generated
    summary, best score, and top supporting chunks.
- `DocumentSummarizer` (`ingestion/summarizer.py`) + `ChatEngine.summarize_document()`.
- `search/hybrid.py` — `hybrid_search()` / `summary_search()` fuse vector +
  **corpus-wide lexical** + summary rankings via Reciprocal Rank Fusion (RRF).
- **Corpus-wide lexical / full-text index** (`search_lexical`):
  - `search/lexical.py` — shared `tokenize` + in-Python `bm25_scores`.
  - **pgvector**: `tsvector` generated column + GIN index (`ts_rank_cd`).
  - **LanceDB**: native FTS index, with an in-Python BM25 scan fallback.
- **Optional cross-encoder re-ranker** (`search/reranker.py`) — lazy-loaded,
  cached, **disabled by default** (CPU-friendly). Scores only the top
  `rerank_top_n` fused candidates.
- Config: `search_mode`, `hybrid_rrf_k`, `enable_reranker`, `reranker_model`,
  `rerank_top_n`, `enable_document_summaries`, `summary_max_chars`.
- MCP: new `find_documents(description, ...)` tool; `semantic_search` accepts
  `search_mode`.
- `record_type` / `summary` columns added to both backends (pgvector migrates
  automatically; LanceDB migrates best-effort on open).
- Optional `.[reranker]` dependency group (`sentence-transformers`).

### Changed

- `VectorRecord` gains `record_type` and `summary` fields.
- `DocumentRepository.search()` gains a `record_types` filter (applied in-memory
  after a larger candidate fetch, so it works on pre-summary schemas).
- Hybrid retrieval now uses **corpus-wide lexical fusion** (not query-scoped) and
  optionally a cross-encoder re-ranker on the top fused candidates.
- Generator labels summary records as "Document Summary" and leads with the
  matched-files section.
- `IngestionQueue` now accepts a `chat_engine` to build summary records.
- README expanded with the "Summary-Aware Hybrid Search" section + user guide.

### Notes / Known limitations

- **LanceDB lexical fallback** loads the *entire filtered table* into memory per
  query when the FTS index cannot be built. Fine for thousands of docs; for very
  large local corpora the pgvector/GIN path is the stronger choice.
- **Reranker is opt-in** (`enable_reranker=false` by default). It requires
  `sentence-transformers` (install `.[reranker]`) and a model download on first use.
  On CPU it stays fast only because it scores a small candidate pool (`rerank_top_n`,
  default 15); raising it increases latency.
- Summaries are generated at ingestion via the chat engine; indexing without a
  reachable chat engine (e.g. Ollama stopped) skips summaries gracefully but
  `summary`/`find_documents` modes then return nothing for that index.

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
