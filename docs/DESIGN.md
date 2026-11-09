# Multi-Engine Hybrid RAG — Design & Interview Guide

This document is written for someone **new to RAG, pgvector, LanceDB and Postgres**.
It starts from first principles, then walks the exact implementation, then gives an
interview question bank and a "what you must know" checklist.

---

## 1. First principles: what is RAG?

**Retrieval-Augmented Generation (RAG)** = give an LLM relevant *retrieved* context so it
answers from your data instead of from memorized training weights.

```
question ──▶ embed ──▶ search index (top-k chunks) ──▶ stuff into prompt ──▶ LLM ──▶ grounded answer
```

Why it exists:
- **Grounding / fewer hallucinations** — the model cites concrete source passages.
- **Fresh & private data** — no retraining; update the index instead.
- **Cheaper** — retrieve 5 chunks instead of fine-tuning or stuffing a whole corpus.

Key vocabulary used throughout this project:

| Term | Meaning |
|---|---|
| **Embedding** | A fixed-length float vector representing text/image meaning. Similar meaning → nearby vectors. |
| **Vector DB / ANN** | Storage + *approximate nearest neighbour* search over embeddings. |
| **HNSW** | Graph-based ANN index (fast, memory-heavy). Used by pgvector here. |
| **IVF-PQ** | Inverted-file + product-quantization ANN (compact, lossy). LanceDB option. |
| **Cosine distance** | `1 - cos(θ)`; we use cosine for normalized embeddings. |
| **Chunk** | A slice of a document/transcript/frame used as a retrieval unit. |
| **Lexical / BM25** | Keyword relevance (term frequency + inverse document frequency). |
| **`tsvector`** | Postgres' built-in full-text-search representation. |
| **RRF** | Reciprocal Rank Fusion: merge ranked lists via `1/(k+rank)`. |
| **Reranker** | A slower cross-encoder that re-scores only the top candidates. |
| **VLM** | Vision-Language Model (captions frames). |
| **ASR** | Automatic Speech Recognition (Whisper transcribes audio). |
| **OCR** | Optical Character Recognition (text inside images/scans). |

---

## 2. What this project is

A **dual-engine, hybrid, multimodal RAG framework** with one toggle:

| `SYSTEM_MODE` | Storage engine | Use case |
|---|---|---|
| `DOC_ONLY_RAG` | PostgreSQL + pgvector + `tsvector` | Documents (PDF/DOCX/XLSX/MD/TXT) |
| `MULTI_MODAL_RAG` | Embedded LanceDB (Arrow) | Documents + audio transcripts + video keyframes |

Retrieval is **summary-aware hybrid**:

```
final ranking = RRF( vector(chunks+summaries), corpus-wide lexical, summary-only vector )
                         │
                  optional cross-encoder rerank (top-N)
```

Design goals: CPU-only friendly, **free-tier first**, portable, graceful degradation,
durable ingestion.

---

## 3. High-level architecture

```
┌──────────────────────── PRESENTATION / API ─────────────────────────┐
│  mrag-ingest CLI (queue)  mrag-query CLI  mrag-gui (PySide6)        │
│  Python library (search/hybrid API)                                 │
├──────────────────────── APPLICATION ────────────────────────────────┤
│  pipeline/registry.py  → handler(kind) for {document,audio,video}   │
│  pipeline/context.py   → repo + embedder + VLM + summarizer         │
├──────────────────────── SERVICE ────────────────────────────────────┤
│  ingestion/{watcher,queue,state,worker}  (durable SQLite queue)     │
│  pipeline/{document_parser,audio_transcriber,video_extractor}       │
│  search/{lexical,hybrid,reranker}  utils/embedding_budget.py        │
├──────────────────────── STORAGE (Factory + ABC) ────────────────────┤
│  database/base_interface.py (DocumentRepository)                    │
│    ├─ postgres_provider.py  (pgvector + tsvector)                   │
│    └─ lancedb_provider.py   (LanceDB + FTS + BM25 fallback)         │
├──────────────────────── EXTERNAL ───────────────────────────────────┤
│  Gemini/local embeddings · local LexRank summaries · VLM · SQLite  │
└─────────────────────────────────────────────────────────────────────┘
```

### Ingestion data flow
```
file dropped in data/inbox
   → watcher (watchdog event OR polling scan)
   → queue.enqueue_path (SQLite row: status=discovered)
   → worker.claim_next  (status=processing, atomic BEGIN IMMEDIATE)
   → wait for file stability (size+mtime unchanged N sec)
   → sha256 hash → dedupe check (skip if already ingested)
   → move inbox → processing/
   → handler(path): parse/chunk → embed → (optional summary) → repo.insert
   → when the queue drains: repo.reindex (batched; see §4.2)
   → move processing → processed/  (or failed/), status=processed/failed
```

### Query data flow
```
query text ──▶ embedder.embed_query ──▶ query vector
   vector search (chunks) ┐
   vector search (summary) ├─▶ RRF ─▶ optional rerank ─▶ top-k SearchResults
   corpus-wide lexical    ┘                     └─▶ FileSearchGroups (file-level hits)
```

Retrieval is exposed by the `mrag-query` CLI, the desktop GUI, and the library API (§9).

---

## 4. Storage engines in depth

### 4.1 PostgreSQL + pgvector (`database/postgres_provider.py`)

**What pgvector adds:** a `vector(n)` column type, distance operators (`<=>` cosine,
`<#>` inner product, `<->` L2), and ANN index access methods (`hnsw`, `ivfflat`).

**Schema created by `initialize()`** (all `ADD COLUMN IF NOT EXISTS`, so it migrates):

| Column | Type | Notes |
|---|---|---|
| `id` | `uuid PRIMARY KEY` | from `VectorRecord.id` |
| `embedding` | `vector(EMBEDDING_DIMENSION)` / `NULL` | `NULL` for records indexed in keyword-only mode |
| `content` | `text` | chunk text / caption / transcript |
| `content_tsv` | `tsvector GENERATED ALWAYS AS to_tsvector('english', content) STORED` | auto-maintained full-text column |
| `record_type` | `text` | `chunk` or `summary` |
| `source_path`, `filename`, `parent_directory` | `text` | provenance |
| `file_type`, `mime_type` | `text` | routing/filtering |
| `chunk_index`, `total_chunks` | `integer` | ordering |
| `timestamp_start`, `timestamp_end` | `double precision` | audio/video alignment |
| `created_at`, `file_modified_at`, `file_hash` | `text` | integrity |
| `metadata_json` | `text` | arbitrary JSON (e.g. modality, chunk source offsets) |
| `summary` | `text` | file-level summary |

Text chunks carry `chunk_source_start`, `chunk_source_end`, and
`chunk_overlap_prefix_chars` in `metadata_json`; summary generation uses these offsets to trim
only real overlap. Older records without offsets remain readable and use the word-overlap
fallback.

**Indexes:**
- `USING hnsw (embedding vector_cosine_ops)` — ANN over cosine distance.
- `USING gin (content_tsv)` — fast keyword search across the whole corpus.
- btree on `source_path` and `file_hash` — lookups/dedupe.

**Vector search** (`search`):
```sql
SELECT *, (embedding <=> %s::vector) AS distance
FROM chunks
WHERE record_type = ANY(%s)
ORDER BY distance ASC
LIMIT %s;
```
- `<=>` is **cosine distance** = `1 - cosine_similarity`. We convert score = `1 - distance`, clamped to `[0,1]`.
- The `::vector` cast matters: without it psycopg sends a `float8[]` and Postgres can't find the `<=>` operator (a real bug we hit).
- `register_vector(conn)` from `pgvector.psycopg` registers the adapter for the `vector` type.

**Lexical search** (`search_lexical`):
```sql
SELECT *, ts_rank_cd(content_tsv, plainto_tsquery('english', %s)) AS lex_score
FROM chunks
WHERE content_tsv @@ plainto_tsquery('english', %s)
ORDER BY lex_score DESC LIMIT %s;
```
- `@@` = "does the document match the query?"; `plainto_tsquery` safely parses user text.
- `ts_rank_cd` is **cover-density** ranking (not BM25). Scores are normalised by the max row score so the fused ranks are comparable.
- Because it hits the GIN index, it covers the **entire corpus**, not just the vector top-k. This is what catches rare proper nouns.

`insert` uses `INSERT ... ON CONFLICT (id) DO UPDATE` (idempotent upsert). Rows are converted
back with `_row_to_record` (UUID→str, `metadata_json`→dict).

### 4.2 LanceDB (`database/lancedb_provider.py`)

**What LanceDB is:** an embedded, disk-based vector DB built on the **Lance/Arrow** columnar
format. No server; the DB is just files under `LANCEDB_DIR`.

- Schema is a **PyArrow schema** with `vector = list_(float32, dim)` and `has_vector: bool`.
- `initialize()` connects, opens/creates the table, and best-effort migrates old tables by
  `add_columns({record_type, summary, has_vector})`.
- Keyword-only records store a zero placeholder vector with `has_vector=false`; vector search
  filters `has_vector = true`, so they remain invisible to ANN until backfilled.
- **Critical operational detail:** LanceDB does **not** auto-refresh indexes when rows are
  added. `search(vector)` silently returns *zero* rows if no vector index exists, and a
  pre-ingestion index won't cover new rows. So `reindex()` `drop_index()` + recreates the
  vector index and both FTS indexes (`content` and `source_path`). (The original spec assumed a
  static index; this is the correction.)
- **Batched, not per-file:** rebuilding on *every* insert is O(n²) across a backfill, so
  `insert()` only sets a dirty flag; the `IngestionWorker` calls `reindex()` when the queue
  drains (and on `stop()`), collapsing a 1,000-file import into one rebuild. Postgres's
  `reindex()` is a no-op (HNSW updates incrementally).
- Writes and reindex are guarded by a `threading.Lock` because the ingestion worker is a thread pool.
- `search_lexical`: tries the native **FTS** index (`search(q, query_type="fts")`); if it
  fails, falls back to an **in-Python BM25 scan** (`search/lexical.py`).

### 4.3 Why two engines?

**The short version.** A RAG app has two jobs: *remember* vectors (storage) and
*answer* from them (pipeline + GUI). This project keeps those jobs separate
behind one interface (`DocumentRepository`), and plugs in two different
"remember" backends for two different situations:

- **Postgres + pgvector — the shared filing cabinet.** It runs as a server
  (Docker or native), so several apps/people can use it at once. It has
  decades-old, battle-tested full-text search (`tsvector`) built in, and tools
  like pgAdmin to look inside. Price: you must install, run, and maintain that
  server (it uses RAM even when idle).
- **LanceDB — the notebook you carry.** No server at all; the whole index is
  just files in a folder (`lancedb_lakehouse/`), which is why the portable USB
  build works. Near-zero idle RAM, nothing to administer. Price: one writer at
  a time, fewer admin tools, and its default install needs an **AVX2** CPU.

Neither engine understands documents, images, or video — they all store the
same records (text chunks, summaries, transcripts, captions). So switching
engines never changes *what* the app can do, only *where the data lives and
who can reach it*. That is also why an old machine without AVX2 isn't locked
out: point `ACTIVE_DB_ENGINE=POSTGRES` and everything — including video —
runs there. The AVX2 story is not the *reason* for two engines; it is the
example that proved the separation worthwhile (one setting flip instead of a
redesign).

| | Postgres + pgvector | LanceDB |
|---|---|---|
| Deployment | server (Docker/native) | embedded (no server) |
| Hybrid lexical | native `tsvector` + GIN | native FTS + BM25 fallback |
| RAM | DB-managed | near-zero (memory-mapped Arrow) |
| Best for | doc-only, larger corpora, SQL | local multimodal, zero-ops |
| Tradeoff | needs a running DB | native `pylance` requires **AVX2** |

The **Factory + Abstract Base Class** pattern (`database/base_interface.py`) keeps the rest of
the app engine-agnostic: `get_repository(settings)` returns the right provider. Both providers
also expose batched `iter_records()`; `mrag-db-copy` (in `database/migrate.py`) uses it to copy
vectors, chunks, summaries and metadata across engines without re-parsing or re-embedding. The
GUI also uses a filtered merge on startup: when `state.db` records a different previous engine,
it copies only processed files missing from the current store, so normal engine toggles stay
seamless while deleted queue rows are not resurrected.

---

## 5. Retrieval design (the interesting part)

### 5.1 Two record granularities
Every text-bearing file produces **chunk** records *and* exactly one **summary** record
(`record_type="summary"`). Why:
- Chunks answer *"what does the passage say?"*
- A single per-file summary answers *"which file matches this description?"* — a question
  pure chunk retrieval answers poorly (matches scatter across many chunks).
- By default the summary is generated once at ingestion by **local semantic LexRank** over
  the chunk vectors already computed for retrieval (`SUMMARY_PROVIDER=LOCAL`,
  `SUMMARY_ALGORITHM=SEMANTIC_LEXRANK`, `SUMMARY_SEMANTIC_UNIT=CHUNK`). No LLM call is made and
  no extra embedding call is needed; the selected text is bounded by `SUMMARY_TARGET_WORDS`
  and `SUMMARY_OUTPUT_MAX_CHARS`, then embedded as its own vector pointing back to the same
  `source_path`.
- Before sentence selection, selected chunks are de-overlapped using their stored source
  offsets. Only actual span overlap is trimmed, so non-adjacent selected chunks are kept intact;
  the remaining text is split into sentences. This avoids duplicating the configured chunk
  overlap in the summary.
- For videos with speech, summary candidates use the transcript and drop visual captions;
  captions are used only when no transcript exists. Exact duplicate captions are removed either
  way, so repeated `Visible:`/`Setting:` text cannot crowd out the spoken content.
- `SUMMARY_SEMANTIC_UNIT=SENTENCE` runs LexRank over sentence embeddings (one extra embedding
  pass). `SUMMARY_PROVIDER=LLM` keeps the previous LLM-generated summary, using a head/tail
  excerpt capped at `SUMMARY_MAX_CHARS`. The output budgets are text limits, independent of
  `EMBEDDING_DIMENSION`, but they are clamped to the embedding model's input-token window
  (with a warning) so a summary vector never represents only a truncated prefix of the stored
  summary text.
- Summary records store the provenance needed to audit that path:
  `summary_provider`, `summary_algorithm`, `summary_semantic_unit`, `summary_input_chars`,
  `summary_output_words`, `summary_chunks_considered`, `summary_chunks_selected`,
  `summary_elapsed_ms`, `summary_fallback`, `source_modalities`,
  `summary_embedding_token_budget`, and `summary_embedding_token_clamped`.

> Interview note: this is deliberately **not** RAPTOR/hierarchical clustering. For
> "find the file", one summary per file is the right granularity; cluster trees pay storage
> and re-clustering cost for a task we don't have.

### 5.2 Typo correction (`search/typo.py`)
`TYPO_CORRECTION` applies before retrieval when set to `LOCAL` (default) or `LLM`:
- `LOCAL` compares only out-of-vocabulary words with the backend's indexed vocabulary using
  SymSpell's deletion index plus a bounded Damerau-Levenshtein check. When no spelling candidate
  exists, it completes a unique longer vocabulary prefix (so `authentica` can reach
  `authentication`), but leaves ambiguous prefixes unchanged. It is free, deterministic,
  and keeps names/codes intact by skipping digit-bearing tokens. Backends expose
  `lexical_vocabulary()` (LanceDB scans the table once; Postgres extracts distinct words once,
  both cached until writes invalidate them). The SymSpell index is rebuilt only when the
  vocabulary snapshot changes.
- `LLM` asks the configured text LLM for a one-line corrected query, then falls back to local
  correction if no LLM is configured or the call fails.
- `OFF` leaves the query untouched.

Correction is applied once per request, not once per retrieval stage, so AUTO’s chunk/hybrid
fallback does not pay twice.

### 5.3 Reciprocal Rank Fusion (RRF)
`search/hybrid.py` builds **three ranked lists** and fuses them:
```
1. vector   — ANN over chunks + summaries
2. lexical  — corpus-wide tsvector / FTS ranking
3. summary  — ANN restricted to record_type = summary
score(doc) = Σ_lists 1 / (k + rank_in_list)      # k = HYBRID_RRF_K, default 60
```
RRF uses **rank, not raw score**, so incomparable scales (cosine vs `ts_rank_cd`) don't need
calibration. Higher `k` flattens the influence of rank position.

### 5.4 Optional cross-encoder re-ranker
`search/reranker.py` lazily loads a **`cross-encoder/ms-marco-MiniLM-L-6-v2`** (~80 MB) —
small enough for CPU. It scores only the top `rerank_top_n` fused candidates, is cached
per process, and **falls back to the unchanged list** if the model can't load. It is opt-in
(`RERANKER_PROVIDER=LOCAL`, `RERANKER_ALLOW_CPU=true`). Scores are the model's own sigmoid
probabilities, exposed as-is (no min-max normalization, which used to force the best hit to
100% and collapse every other hit to 0%); the GUI lists passages the answer cited *first*, so
an uncalibrated top hit cannot push an unrelated source to the top of the display.

> Bi-encoder vs cross-encoder: the embedding model is a *bi-encoder* (query and doc encoded
> separately → fast, approximate). A *cross-encoder* processes the pair jointly → accurate
> but O(n) and cannot be pre-indexed, hence "re-rank the shortlist".

### 5.5 BM25 fallback (`search/lexical.py`)
A ~40-line Okapi BM25: `idf = log((N - df + .5)/(df + .5) + 1)`, TF saturation via `k1`, length
normalisation via `b`. **We intentionally don't use the `rank_bm25` module**: it's only used on
the degraded LanceDB path, we already hold the rows in memory, and Postgres/FTS handle the
primary lexical work. Fewer dependencies, and rank fusion makes exact BM25 constants irrelevant.
Numeric grouping is normalized before tokenization, and native-FTS queries try both grouped and
ungrouped spellings, so `6000` matches `6,000` (and vice versa) on all engines.

### 5.6 Keyword-only mode and expansion (`search/keyword.py`)
`SearchMode.KEYWORD` calls `search_lexical()` over chunk records only. It never embeds the query
and never runs the cross-encoder, so it works without loading an embedding model at query time.

Expansion is backend-neutral:

- When `KEYWORD_EXPANSION` is unset, keyword-only/no-embedding searches default to `CORPUS`.
  An explicitly configured value always wins, so exact-name searches can keep `OFF`.
- `CORPUS` selects up to `KEYWORD_EXPANSION_TERMS` terms from the top score band of lexical
  chunks (pseudo-relevance feedback), requiring at least two matching chunks. Terms that are
  common across most of the retrieved pool (and every result in small pools) are dropped so
  generic visual descriptions (for example “setting” or “visible”) do not enter the expansion.
  Each term is searched separately and fused with weighted RRF; the original query list gets
  weight 3, expansion lists weight 1. This keeps query syntax portable across Postgres and
  LanceDB and lets corpus context disambiguate senses.
- `SYNONYMS` uses `assets/synonyms.json.gz`, a compact WordNet-derived map of common English words
  generated offline (about 327 KB gzipped, ~22k roots). It adds no NLTK/WordNet runtime
  dependency and no download. Synonym lists get weight 1.5, below the base query.
- `CORPUS+SYNONYMS` combines both sources, deduplicated and capped by
  `KEYWORD_EXPANSION_TERMS`.

Lexical search also indexes `source_path` tokens: Postgres maintains a generated `path_tsv`
tsvector with a GIN index, while LanceDB keeps a dedicated FTS index on `source_path` and uses
an OR match query to fetch path candidates before exact token-overlap scoring in pandas. If the
path FTS index is unavailable, or newly inserted rows still await `reindex()`, LanceDB falls back
to a pushed-down case-insensitive substring scan. Content and path rankings are fused 3:1, so filename searches such as `scores.xlsx` work
even when the content does not repeat the filename. The degraded BM25 path normalizes NFKC/case, drops
stopwords when meaningful terms remain, and deduplicates query terms. Local typo correction also
completes an unambiguous longer prefix when no edit-distance candidate exists.

Literal searches stay inside the indexed text; we do not grep source files, which may be binary
and whose searchable text may come from OCR/transcripts.

### 5.7 Embedding-optional ingestion and backfill
`EMBEDDING_PROVIDER=NONE` makes the embedder optional end to end. Ingestion still parses, chunks
and summarizes because chunks give keyword search passage precision and citations, and they let a
later embedding run avoid re-parsing. Keyword-only chunks use the same 450-token default target,
without an unused embedding model's window cap. `registry._index_chunks()` stores records with empty
vectors; the providers persist that as `NULL` (Postgres) or zero + `has_vector=false` (LanceDB).
Retrieval modes `keyword` and `auto` work without an embedder; `auto` falls back to keyword.
Explicit `chunk`/`summary`/`hybrid` modes still require one and report an actionable error.

`ingestion/backfill.py` finds records with missing vectors, embeds them in batches, and rewrites
each affected source file through `delete_by_path` + `insert`. This avoids dialect-specific
in-place vector updates and leaves already-embedded records untouched. The CLI command is
`mrag-ingest backfill-embeddings`; the GUI exposes it as **File → Backfill embeddings**. The state
fingerprint omits embedding fields when the provider is `NONE`, so keyword↔vector switches are
treated as backfill rather than as an index rebuild.

---

## 6. Ingestion subsystem (`ingestion/`)

### 6.1 Durable queue (SQLite) — `state.py`
Not an in-memory queue (which loses jobs on crash). A SQLite table `files`:

```
id, path(UNIQUE), sha256, size, mtime, kind,
status ∈ {discovered, processing, processed, failed, skipped},
attempts, error, output, created_at, updated_at
```
- WAL mode + `busy_timeout` for concurrency.
- `claim_next(max_attempts)` uses `BEGIN IMMEDIATE` so two workers can't grab the same row.
- `is_hash_ingested(sha, exclude_id)` gives **content-level dedupe** (a rename/re-drop is a no-op).
- `reset_stale()` requeues rows stuck in `processing` after a crash.

### 6.2 Watcher — `watcher.py`
`watchdog` create/move/delete events **plus** a polling thread (WSL/network shares drop
events). Files moved or deleted out of the inbox drop their queue row and any indexed vectors
(worker-owned `processing` rows are protected from the race). Ignores hidden files and temp
suffixes (`.part`, `.crdownload`, `.tmp`, `.swp`, `.lock`).

### 6.3 Worker — `worker.py`
- Thread pool (`WORKER_COUNT`, default 2 — low for CPU-only Docling/Whisper).
- **Stability wait:** file is only processed after `size`+`mtime` are unchanged for
  `FILE_STABLE_SECONDS` (default 15) — prevents reading a half-copied video.
- Lifecycle with atomic moves: `inbox → processing → processed|failed`, DB path updated each move.
- Dispatch: `handlers[kind]` from `pipeline/registry.py`; unknown kind → skipped.
- Idempotent: re-running on the same file updates records (`delete_by_path` + upsert).

### 6.4 Why SQLite, not Redis/Celery?
Single-machine, CPU-only, zero external service, crash-safe, and enough throughput. Redis/Celery
would add an always-on broker for no benefit at this scale.

---

## 7. Multimodal pipeline

### 7.1 Documents (`pipeline/document_parser.py`)
1. If `DOC_USE_OCR` and suffix is Docling-capable → **Docling** exports layout-aware Markdown
   (tables/headings preserved), with OCR for scans.
2. Otherwise lightweight readers: `pypdf`, `python-docx`, `openpyxl`; HTML tag-stripped; text read directly.
3. Chunk sentence-aware passages at a ~450-token target with embeddings, capped
   by `CHUNK_EMBEDDING_TOKEN_RATIO` (default 0.9) of the embedding model's input
   window. With `EMBEDDING_PROVIDER=NONE` the target remains 450 tokens but is
   independent of any stale/unused embedding model. Word caps (`DOC_CHUNK_SIZE`,
   `DOC_CHUNK_OVERLAP`) remain upper bounds; `DOC_CHUNK_TOKENS` /
   `DOC_CHUNK_OVERLAP_TOKENS` override the target.
   Whole sentences are packed to the budget and only an oversized sentence is
   hard-split. Overlap starts at the previous sentence boundary when that whole
   sentence fits in the chunk headroom, otherwise it falls back to a word
   boundary. This prevents small models (MiniLM/bge-small) from silently
   truncating the tail of a passage. Each chunk records
   `chunk_source_start`, `chunk_source_end`, and `chunk_overlap_prefix_chars`
   so downstream summaries can reconstruct overlap-free text.

### 7.2 Audio (`pipeline/audio_transcriber.py`)
**faster-whisper** (CTranslate2) runs locally on CPU with `int8`. Whisper's short segments are
packed into chunks bounded by **both** audio duration (`WHISPER_CHUNK_MAX_SECONDS`, default
10s) and the embedding token/word budget. This keeps slow speech from becoming one long chunk;
segments over the time cap are split proportionally. Each chunk keeps the start timestamp of its
first segment and the end timestamp of its last, plus source offsets for summary de-overlap.
Model is cached per size/device.

### 7.3 Video (`pipeline/video_extractor.py`)
- **Hybrid keyframe sampling** (not fixed FPS): compute an HSV-histogram **Bhattacharyya
  distance** between sampled frames; keep a frame if it exceeds `MM_SCENE_THRESHOLD` *or* the
  min-interval floor elapsed (`MM_SCENE_MIN_INTERVAL_SEC`, default 2s), capped at
  `MM_MAX_KEYFRAMES_PER_MIN`. This bounds cost regardless of how fast-cutting the video is, and
  still covers static/slow content.
- Frames are resized (`cv2.INTER_AREA` to `MM_TARGET_HEIGHT`) and JPEG-encoded **in memory**
  (`cv2.imencode`) — never written to disk.
- Audio is extracted with **ffmpeg** (16 kHz mono WAV) and transcribed.
- Frames → VLM captions (batched) become chunk content: `[Visual @ 12.34s] {...}`.
- Transcript and visual chunks are independent retrieval units. A transcript chunk keeps the
  start of its first Whisper segment and the end of its last; a visual chunk keeps the sampled
  keyframe timestamp. The GUI seeks to whichever chunk was retrieved, so the displayed time can
  differ between a spoken passage and an on-screen-text keyframe. There is no subtitle-track
  (`.srt`/`.vtt`) ingestion or OCR-to-transcript alignment; speech comes from Whisper and
  on-screen text only from the VLM's frame description.

### 7.4 VLM captioning (`utils/api_clients.py`)
Gemini `generate_content` with multiple inline image parts per request; prompt asks for a
**JSON array** aligned to the images with keys `timestamp, core_actions, on_screen_text,
objects, setting`. JSON is parsed defensively (fallback: raw text for the first frame).
A `RateLimiter` paces requests to `VLM_MAX_RPM` (default 15).

---

## 8. Configuration system (`config.py`)

- **pydantic-settings** validates/coerces `.env` + environment variables.
- **Enums** (`SystemMode`, `DBEngine`, `VLMProvider`, …) make invalid values fail fast.
- Providers are configured **by purpose**, not vendor: each of embeddings / vlm / llm has its own
  provider, model and `*_API_KEY` (`EMBEDDINGS_API_KEY`, `VLM_API_KEY`, `LLM_API_KEY`). These
  `RoleSettings` objects are the single source of truth; consumers never read vendor keys directly.
  Vendor-centric names remain as deprecated fallbacks.
- `SYSTEM_MODE` (content policy) and `ACTIVE_DB_ENGINE` (storage) are **orthogonal**.
  `MODE_TO_ENGINE` only supplies the default engine; an explicit `ACTIVE_DB_ENGINE`
  overrides it (with a warning), so multimodal content may run on Postgres and documents
  on LanceDB.
- `KNOWN_EMBEDDING_DIMS` validates `EMBEDDING_DIMENSION` against the chosen model
  (e.g. `BAAI/bge-small-en-v1.5`=384, `gemini-embedding-001`∈{768,1536,3072}).
- `KNOWN_EMBEDDING_MAX_TOKENS` / `embedding_max_input_tokens()` track the separate
  **input-token window** (MiniLM 256, bge-small/base 512, bge-m3 8192, Gemini and
  EmbeddingGemma 2 2048, Cohere v3 512, unknown models a conservative 512). MRL dimension
  truncation does not change this window. `utils/embedding_budget.py` uses it to clamp summary
  output to `window * 0.8` and to cap chunks at `window * 0.9`; the default chunk target is 450
  tokens, so large-window models still get bounded passages unless `DOC_CHUNK_TOKENS` is raised.
  When `EMBEDDING_PROVIDER=NONE` (keyword-only retrieval) there is no embedding window to
  respect; the default chunk target remains 450 tokens, and `DOC_CHUNK_TOKENS` still overrides it.
  Local models use their Hugging Face tokenizer when it is cached, otherwise a conservative
  character/CJK estimate is used. Cached embedding/reranker weights load with
  `local_files_only=True`, so later startups never check the Hub for changed weights (which
  would silently move the index to a different vector space).
- **`APP_ROOT` / `MRAG_HOME`:** the app root for `.env`, `data/`, `state.db`. Resolution:
  `MRAG_HOME` → else cwd (if a `.env` exists) → else `~/.multimodal_rag`. **All relative
  path settings resolve against `APP_ROOT`, not cwd.**
- **Live GUI settings:** the Settings dialog re-reads `.env` directly with `Settings()` instead of
  a cached `get_settings()`, and the main window polls the file. A confirmed Save reloads the
  backend in-process (`BackendService` and query embedder/reranker), so search/provider/theme
  changes apply without restarting. `ACTIVE_DB_ENGINE` is live and copies processed vectors
  missing from the newly selected engine when `state.db` is shared. Storage-path keys
  (`DATA_DIR`, `INBOX_DIR`, `PROCESSING_DIR`, `PROCESSED_DIR`, `FAILED_DIR`, `LANCEDB_DIR`,
  `STATE_DB_PATH`, `VLM_CACHE_DIR`) are bound at startup and still require a restart. Ingestion
  enqueue is blocked while `_pending_settings` is set or the backend is reloading.
- **Derived properties** keep policy in one place: `has_cuda`/`device` (via `torch` or
  `nvidia-smi`), `vlm_local_fallback_enabled` (only with GPU unless opted in), `embeddings_local_enabled`,
  `reranker_local_enabled`, `whisper_device`/`whisper_compute_type` (`auto` → int8 on CPU / float16 on CUDA).
- `ensure_data_dirs()` creates inbox/processing/processed/failed/caches.

---

## 9. Querying (`mrag-query`)

The query side is a CLI (`multimodal_rag/query.py`) plus the same library API.

```bash
mrag-query "the story about a man who owns a mansion"   # hybrid (vector+lexical+summary) + RRF
mrag-query "mansion" --mode summary -k 5                 # file-level (summary records) only
mrag-query "invoice 4831" --mode keyword                 # FTS/BM25, no query embedding
mrag-query "mansion" --answer                            # synthesize a cited answer [1],[2]...
mrag-query "mansion" --json                              # machine-readable
mrag-query "revenue" --folder /data/processed --no-rerank
mrag-query -i                                            # interactive REPL
```
Flow: `embed_query` → `run_search` (chunk/summary/hybrid) → `print_results`
(+ optional `synthesize_answer` via `LLM_PROVIDER`; `ANSWER_TOP_K` limits how many
retrieved passages reach the prompt, default 3, `0` = all). Defaults come from settings
(`SEARCH_MODE`, `SEARCH_TOP_K`, reranker). In Docker the entrypoint is `mrag-ingest`, so use
`docker compose run --rm --entrypoint mrag-query app "..."`.

Library equivalent:
```python
from multimodal_rag.config import get_settings
from multimodal_rag.database import get_repository
from multimodal_rag.pipeline.embedding import get_embedding_engine
from multimodal_rag.search.hybrid import hybrid_search

settings = get_settings()
repo = get_repository(settings); repo.initialize()
embedder = get_embedding_engine(settings)

query = "the story about a man who owns a mansion"
qv = embedder.embed_query(query)
results, file_groups = hybrid_search(repo, query, qv, top_k=5,
                                     rrf_k=settings.HYBRID_RRF_K,
                                     rerank=settings.active_reranker != "NONE")
for r in results:
    print(round(r.score, 3), r.record.source_path, r.record.timestamp_start)
for g in file_groups:
    print("FILE:", g.filename, "->", g.summary[:80])
```

### Automatic vs manual ingestion
Dropping a file in `data/inbox/` is **automatically** processed while the desktop GUI is open or
a loop is running (`mrag-ingest run` / Docker `app`). The GUI starts the same watcher + worker
after backend initialization. The watcher targets `inbox/` only. With no watcher active,
nothing happens until `mrag-ingest scan` or `add`.

### Desktop GUI (`mrag-gui`)
`multimodal_rag/gui/` is a PySide6 front-end. It ships by default via
`PySide6-Essentials` in `requirements.txt` (QtCore/QtGui/QtWidgets only, ~233 MB — the full
`PySide6` metapackage adds ~417 MB of unused Addons: WebEngine/3D/Multimedia). `.[gui]` is the
equivalent extra for pyproject users. Design:
- `services.py` is a **Qt-free** facade (`BackendService`) over `build_context`,
  `IngestionWorker`, and `query.run_search` — so the GUI holds no business logic.
- `workers.py` runs `BackendService.initialize()` and `search()` (plus optional answer
  synthesis via `generate_rag_answer`) in `QThread`s so the UI never blocks. Our pipeline is
  synchronous, so this is simpler than the old asyncio-in-QThread approach.
- Indexing reuses the durable SQLite queue + `IngestionWorker`; the window polls `state.db`
  with a `QTimer` for progress instead of threading queue callbacks into Qt.
- The GUI auto-starts an inbox watcher and ingestion worker (same mechanism as
  `mrag-ingest run`) after backend initialization, so dropped files are processed while the app
  is open.
- Each panel implements `apply_theme()`, rebuilding its Qt stylesheet from the shared
  `mrag_style` palette when `GUI_THEME` changes; the Settings dialog saves the theme and the
  main window applies it live.
- The GUI follows `SYSTEM_MODE` and shows a warning banner when LanceDB is selected; it does
  not force a mode. `pylance` needs AVX2 at runtime — a documented limitation, **not** a design
  input. On an AVX2-less host, point `ACTIVE_DB_ENGINE=POSTGRES` to keep multimodal content.
- Cross-platform (no OS branching); `app.py` has a **headless guard** that prints a hint and
  exits `2` when there is no `DISPLAY`/`WAYLAND_DISPLAY` (so servers fail gracefully). Not
  runnable in Docker (no display). PyInstaller spec (`mrag-gui.spec`) builds a Windows `.exe`.

---

## 10. Free-tier engineering

- **Embeddings are local by default** (`EMBEDDING_PROVIDER=LOCAL`), batched
  (`EMBEDDING_BATCH_SIZE`) and uncapped by a remote RPM limit. `BAAI/bge-small-en-v1.5` (384-d)
  is the recommended CPU-friendly model; Google embeddings and API pacing remain available for
  hosts that prefer to offload embedding work or use keyword-only/no-embedding mode.
- **VLM (Gemini free = 15 RPM)**: batch N frames per request (`VLM_BATCH_SIZE`, default 8),
  scene-based sampling (fewer frames), `RateLimiter` at `VLM_MAX_RPM`, retries with backoff,
  and a disk cache (`VLM_CACHE_DIR`) so re-runs cost **zero** requests.
- **Cost intuition (Muse Spark 1.3 Contributor, ~$0.1/$0.2 per 1M in/out):** a 1-hour video at
  ~1 keyframe/5s ≈ 720 frames ≈ **$0.05**; at 2 FPS ≈ 7,200 frames ≈ **$0.47**. Scene-based
  sampling keeps you near the low end.
- Postgres FTS, BM25, RRF, Whisper, ffmpeg, Docling, LanceDB all run **locally for free**.

---

## 11. Reliability & graceful degradation

| Missing thing | Behaviour |
|---|---|
| LanceDB not installed | `RuntimeError` with install hint (raised only when that engine initializes) |
| `sentence-transformers` missing | reranker/local embeddings return a safe fallback / clear error |
| VLM not configured | video still indexes audio transcripts; frames skipped with a log |
| Whisper missing | audio/video handler raises a clear "install faster-whisper" error |
| Local summary unavailable | frequency fallback, or summaries skipped; `find-by-description` returns fewer hits |
| ffmpeg missing | audio track skipped (warning); video frames still processed |
| Docling missing | lightweight readers used |

---

## 12. Testing strategy

- **Pure-Python units:** BM25 scores, RRF fusion, chunking.
- **Real Postgres integration:** create table with generated `tsvector`, insert, vector +
  lexical + hybrid search, `file_needs_reindex`, deletes (run against the Docker Postgres).
- **LanceDB:** control flow verified with an in-process **fake backend** (pylance can't run in
  the CI sandbox due to AVX2); plus import/factory and row-mapping tests.
- **Pipeline:** registry wiring with a fake repo/embedder; document handler end-to-end into Postgres.
- **Packaging:** `pip install -e .` + `mrag-ingest` from an unrelated cwd; Docker build + run.

Known limitation: real Gemini/DeepSeek, Docling, faster-whisper and real LanceDB need keys,
hardware, or AVX2 and are not exercised in the sandbox.

---

## 13. Design decisions & tradeoffs (interview gold)

| Decision | Chosen | Alternative | Why |
|---|---|---|---|
| Queue | SQLite durable | Redis/Celery, in-memory | zero-ops, crash-safe, single machine |
| DB abstraction | ABC + Factory | if/else in app | app stays engine-agnostic |
| Two engines | pgvector **or** LanceDB | one engine | server vs embedded tradeoff |
| File-level retrieval | per-file summary record | RAPTOR clusters | right granularity, cheap |
| Fusion | RRF | weighted score sum | scale-free, no calibration |
| Lexical fallback | hand-written BM25 | `rank_bm25` | rare path, fewer deps |
| Reranker default | MiniLM-L-6 (~80MB) | bge-reranker-base (~278MB) | CPU-friendly |
| Embeddings | Google API (free) | local torch | no GPU; local optional |
| Providers sync | psycopg3 / LanceDB sync | asyncpg | matches thread-pool worker |
| Video frames | in-memory JPEG | temp files on disk | speed, cleanliness |
| Frame sampling | hybrid scene+floor+cap | fixed FPS | bounded cost, good coverage |
| GUI Qt package | `PySide6-Essentials` (~233 MB) | full `PySide6` (+417 MB Addons) | we only use Widgets; Docker unaffected (uses extras) |
| GUI + core | separate `gui/` package | embed in core | CLI/headless installs never import Qt |
| Secrets | `.env` | OS keyring | headless/WSL friendly |

---

## 14. Interview question bank

**RAG fundamentals**
1. What problem does RAG solve vs fine-tuning? — grounding, freshness, cost.
2. Explain embeddings and cosine similarity. — vectors, `1-cosθ`, scale-invariant.
3. Why approximate (ANN) instead of exact nearest neighbour? — speed at scale; recall tradeoff.
4. HNSW vs IVF-PQ? — graph (fast/accurate/memory) vs quantized inverted lists (compact/lossy).

**Storage**
5. What is pgvector and what does it add to Postgres? — `vector(n)`, distance ops, HNSW/IVFFlat.
6. What is `tsvector` and how is it kept up to date here? — generated stored column.
7. Why did we cast `%s::vector` in the SQL? — psycopg sends `float8[]` otherwise; operator resolution.
8. Why does a fresh LanceDB search return nothing? — no index; and indexes don't auto-refresh → reindex after insert.
9. How does the factory pattern help here? — swap engines without touching callers.

**Retrieval quality**
10. Why hybrid (vector + lexical)? — semantics + exact rare terms/proper nouns.
11. What is RRF and why rank-based? — scale-free fusion of incomparable scorers.
12. Bi-encoder vs cross-encoder reranking, and why rerank only top-N?
13. Why summary records? — locate *the file* from a description.
14. How is chunk size/overlap chosen? — balance context vs specificity.

**Multimodal**
15. How do you index a video? — keyframes → VLM captions + audio → ASR, all with timestamps.
16. Why scene detection instead of fixed FPS? — bounded, content-adaptive frame count.
17. How do you respect a 15 RPM free tier? — batching, pacing, caching, keyframe reduction.
18. How do you avoid re-captioning on re-runs? — hash-keyed disk cache + content dedupe.

**Engineering**
19. Why a durable queue and how do you dedupe? — SQLite + sha256, `is_hash_ingested`.
20. How do you avoid reading half-copied files? — stability wait on size+mtime.
21. How is the worker safe with multiple threads? — atomic claim, per-backend write lock.
22. Where do you handle failures / retries? — status machine + `attempts`/`MAX_INGEST_RETRIES`.
23. How is the code portable? — package + `MRAG_HOME`, optional extras, Docker; AVX2 caveat.
24. What would you change at 10M documents? — Postgres/partitioning, batching, async, dedicated reranker service.

---

## 15. "What you should know" checklist

- [ ] The ingestion vs query data flows, end to end.
- [ ] Embeddings, cosine distance, ANN, HNSW, IVF-PQ.
- [ ] pgvector column type + operators; `tsvector`/GIN/`ts_rank_cd`; BM25 formula intuition.
- [ ] Why hybrid retrieval and how RRF combines ranks.
- [ ] Summary records vs chunk records and when each wins.
- [ ] Bi-encoder vs cross-encoder; when to rerank.
- [ ] Video keyframe sampling, VLM captions, ASR, timestamp alignment.
- [ ] Durable queue, statuses, dedupe, idempotency, stability wait.
- [ ] Factory/ABC, pydantic-settings, path/app-root resolution, graceful degradation.
- [ ] Free-tier budgeting (RPM/RPD, batching, caching) and rough cost math.
- [ ] Portability constraints (AVX2 for LanceDB, ffmpeg, Docker image sizes).
- [ ] GUI threading model (`QThread` + Qt-free `BackendService`) and why Qt is isolated in `gui/`.
- [ ] Tradeoff table in §13 — be ready to defend each choice.

---

## 16. Limitations & future work
- No HTTP/API server yet (CLI + desktop GUI + library only).
- No GPU path (CPU-first); no AVX2 fallback vector store yet (candidate: `sqlite-vec`).
- Summaries are single-pass, not map-reduce for very long docs.
- LanceDB lexical fallback loads the table into memory when FTS is unavailable.
- Native (non-Docker) Windows users must install ffmpeg themselves.
