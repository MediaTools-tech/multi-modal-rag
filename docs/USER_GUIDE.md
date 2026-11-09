# User Guide

A practical, end-to-end guide to installing, configuring and using the Multi-Engine Hybrid RAG.
No prior experience with Postgres, pgvector or LanceDB is required.

- New to the concepts? Read [`DESIGN.md`](./DESIGN.md) first.
- Just want to run it? Start with the Windows quick start below, then use the rest of this guide
  as a reference.

---

## Quick start — Windows (recommended)

This is the shortest path to a useful local setup. You will run the app, add one free Google AI
Studio key for vision/cited answers, and use a small local embedding model so embeddings run on
your CPU without an embedding API key.

### 1. Get the app running

**Option A — Windows release (no Python):**

1. Download `MultiModalRAG-windows.zip` from the project's GitHub Releases page.
2. Extract it to a **local disk** folder such as `C:\MultiModalRAG` — not OneDrive, a network
   share, or a synced folder.
3. Run `MultiModalRAG.exe`. Windows SmartScreen may warn because the exe is unsigned; choose
   **More info → Run anyway** only if you downloaded it from the official release.
4. On first launch the app creates `.env`, `data/`, and `state.db` next to the exe. Keep this
   folder together when moving the app.

**Option B — Python install:** follow [§3 Install](#3-install), then run `mrag-gui` from the
same environment. Python 3.11 or newer is required.

### 2. Get one free Google AI Studio key (recommended)

1. Open <https://aistudio.google.com/app/apikey>.
2. Sign in with a Google account and create an API key.
3. Copy the key. Free-tier availability and quotas can change; check the current limits in AI
   Studio.

You can use the same key for both the **vision** and **text** roles:

- `VLM_API_KEY` — describes images and sampled video frames, including on-screen text.
- `LLM_API_KEY` — writes the optional cited answer in the chat.

If you only want document search and do not need image/video captions or generated answers, you
can skip the key. Local embeddings and local summaries still work offline.

### 3. Configure it once in the GUI

Open **Settings** from the top menu bar. Use these values for the recommended starter setup:

| Setting | Recommended value | Why |
|---|---|---|
| `SYSTEM_MODE` | `MULTI_MODAL_RAG` | documents + images + audio/video |
| `ACTIVE_DB_ENGINE` | `LANCEDB` | embedded store; no Postgres/Docker needed |
| `EMBEDDING_PROVIDER` | `LOCAL` | free CPU embeddings; no embedding API key |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | small, fast, good quality; 384 dimensions |
| `EMBEDDING_DIMENSION` | `384` | must match bge-small |
| `SUMMARY_PROVIDER` | `LOCAL` | free extractive summaries from stored chunk vectors |
| `SEARCH_MODE` | `auto` | fast chunk search with hybrid fallback |
| `VLM_PROVIDER` | `GEMINI` | captions for images/video frames |
| `VLM_API_KEY` | your AI Studio key | vision role |
| `LLM_PROVIDER` | `GEMINI` | optional cited answers |
| `LLM_API_KEY` | your AI Studio key | text role |

Where to paste the Google key in **Settings**:

- **Vision (VLM)** → set `VLM_PROVIDER` to `GEMINI`, then paste the key into `VLM_API_KEY`.
- **Text LLM** → set `LLM_PROVIDER` to `GEMINI`, then paste the same key into `LLM_API_KEY`.
- **Embeddings** → only if you choose Google embeddings, set `EMBEDDING_PROVIDER=GOOGLE` and
  paste the key into `EMBEDDINGS_API_KEY`. For the recommended local setup, leave this key blank.

Save. Most settings, including an `ACTIVE_DB_ENGINE` switch, apply immediately by reloading the
backend in-process. Only storage-path settings (`DATA_DIR`, `INBOX_DIR`, `LANCEDB_DIR`,
`STATE_DB_PATH`, etc.) require restarting the app. Wait for the status bar to say the backend is
ready before adding files.

> **Important:** `EMBEDDING_MODEL` and `EMBEDDING_DIMENSION` must stay consistent with the index.
> If you change the embedding model or dimension after ingesting files, re-index them. A
> dimension mismatch is blocked with a clear error rather than silently mixing incompatible
> vectors.

### 4. Add files and search

1. Click **Add Folder...** or **Add Files...**, or click **Open Inbox** and copy files into
   `data/inbox/`.
2. Wait for the ingestion table to show `processed`. The status bar shows file/record counts
   and queue activity.
3. Ask a question in the center chat, for example:
   - `Which files discuss the 6,000 computers?`
   - `invoice 4831`
   - `video where a man explains the number of machines`
4. Click a source card to preview it. For video, the preview seeks to the timestamp of the
   retrieved chunk. If the answer used a transcript chunk, the timestamp is the start of that
   transcript chunk; if it used a visual caption, it is the sampled keyframe time.

LanceDB requires an AVX2-capable CPU. If the app crashes with `Illegal instruction`, use
`ACTIVE_DB_ENGINE=POSTGRES` instead (Postgres supports multimodal records too) or run on a newer
CPU.

### Recommended starter profiles

| Goal | Embeddings | Vision / answers | Notes |
|---|---|---|---|
| **Best default** | Local `BAAI/bge-small-en-v1.5` | One Google AI Studio key for Gemini VLM + LLM | Fast, cheap, no embedding API calls |
| **Fully offline documents** | Local bge-small | `VLM_PROVIDER=NONE`, leave LLM blank | No captions or generated answers |
| **Keyword-only, no models** | `EMBEDDING_PROVIDER=NONE` | optional | Search still works; vector/hybrid modes disabled |
| **Google embeddings** | `EMBEDDING_PROVIDER=GOOGLE` + `EMBEDDINGS_API_KEY` | same Google key | Useful on weak CPUs or very large corpora |

The first local embedding run downloads the model from Hugging Face and caches it. Later starts
use the cache without re-downloading. Keep `data/` on a local disk.

---

## 1. What you can do

- Index **folders or individual files** into a durable background queue that survives crashes.
- Search by meaning (vector), by exact words/codes (full-text/BM25), or a fusion of both.
- Find an entire file from a description of its contents (file-level summary retrieval).
- Get an optional cited answer with clickable sources and a preview pane.
- Search without an embedding model using keyword-only mode, then backfill vectors later
  without re-parsing the files.
- Switch between LanceDB (embedded, default) and Postgres/pgvector; switch engines live in the
  GUI and auto-copy missing vectors when `state.db` is shared.
- Edit `.env` from **Settings** while the app is running. Search/query/provider/DB-engine
  changes reload the backend in-process; only storage-path changes need a restart.

Supported inputs:
| Kind | Extensions | Needs |
|---|---|---|
| Documents | `.pdf .docx .doc .xlsx .xls .pptx .ppt .md .txt .html .htm .csv .rtf` | `parsing` deps (Docling optional, better for tables/OCR) |
| Images | `.png .jpg .jpeg .bmp .tiff .tif .webp .gif` | OCR and/or a VLM for captions |
| Audio | `.mp3 .wav .m4a .aac .flac .ogg .opus` | `faster-whisper`; ffmpeg for video/audio extraction |
| Video | `.mp4 .mkv .mov .avi .webm .m4v .mpg .mpeg` | `opencv`, ffmpeg, a VLM for frame captions |

There is currently **no separate `.srt`/`.vtt` ingest path and no subtitle-stream extraction**.
Video speech comes from Whisper. Burned-in/on-screen text is captured only when the VLM describes
a sampled frame.

---

## 2. Prerequisites

| Requirement | Why | How |
|---|---|---|
| **Windows 10/11 x64** | only fully end-to-end-tested platform | Windows release needs no Python |
| **Python ≥ 3.11** | only if running from source | python.org; use 3.12 if unsure |
| **Local disk space** | app, models, index | ~2–4 GB free is comfortable; avoid OneDrive/network drives for `data/` |
| **ffmpeg** | audio/video handling | Windows build on `PATH`; not needed for documents |
| **AVX2-capable CPU** | LanceDB (`pylance`) only | modern x64 CPU; otherwise use `ACTIVE_DB_ENGINE=POSTGRES` |
| **One Google AI Studio key** (optional) | image/video captions + cited answers | free key from <https://aistudio.google.com/app/apikey>; local embeddings/summaries need no key |
| **Docker** (optional) | Postgres and/or the app image | Docker Desktop + WSL integration on Windows |
| **Display** (optional) | `mrag-gui` desktop window | any desktop session; headless uses the CLI |

> **Embeddings and summaries can be fully local and free.** The recommended setup uses
> `BAAI/bge-small-en-v1.5` on CPU and local extractive summaries. A Google key is only needed
> for VLM frame captions and optional LLM-written cited answers. You can run without any key and
> still use document search, or use `EMBEDDING_PROVIDER=NONE` for keyword-only search.
>
> **ffmpeg is only needed for audio/video.** Document-only use does not require it. The code
> degrades gracefully (it logs a warning and skips the audio track if ffmpeg is absent).
>
> **The GUI is optional.** `requirements.txt` includes `PySide6-Essentials` so `mrag-gui` works
> out of the box on a desktop. On a headless server the GUI is unnecessary (and will exit with a
> hint); use `mrag-ingest` / `mrag-query`, or install the leaner `pip install -e ".[lancedb,parsing]"`.
>
> **Local models are cached after the first download.** Subsequent startups load them with
> `local_files_only=True` and do not query Hugging Face for updated weights, so the vector space
> stays stable. A public model needs no token; for gated/private repos set `HF_TOKEN` in the
> process environment or run `hf auth login` (higher rate limits also use a token).

---

## 3. Install

### Option A — Windows executable (no Python)
1. Download the release zip from GitHub Releases.
2. Extract to a local folder such as `C:\MultiModalRAG`.
3. Run `MultiModalRAG.exe`. The app creates `.env` and `data/` next to the exe.
4. Open **Settings** and follow the Windows quick start above.

The exe is unsigned, so SmartScreen/antivirus may warn. Video playback uses VLC if it is
installed; VLC is optional.

### Option B — pip, complete default install (Windows/macOS/Linux)
```powershell
cd multimodal_rag_core
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env      # then edit .env or use the GUI's Settings screen
mrag-ingest status
mrag-gui                         # desktop GUI is included (PySide6-Essentials)
```
On macOS/Linux, use `python3 -m venv .venv && source .venv/bin/activate` and
`cp .env.example .env`.

### Option C — pip extras (choose only what you need)
```bash
pip install -e ".[lancedb,parsing,video,audio,gui]"   # multimodal + GUI
pip install -e ".[postgres,parsing,gui]"              # document-only + GUI
pip install -e ".[lancedb,parsing]"                   # lean headless install
pip install -e ".[all]"                               # everything (Docling, torch, ...)
```
> A bare `pip install -e .` installs only the pure-Python core. The CLIs tell you which extra is
> missing — that is expected, not a bug. Use an editable install when running from source.

### Option D — Docker (most portable, no Python setup)
```bash
# from the repository root
cp multimodal_rag_core/.env.example .env     # compose reads keys from here

# build + run (default: parsing + Docling OCR, with static ffmpeg for audio/video)
MRAG_EXTRAS=parsing,docling docker compose build app
docker compose run --rm app scan
docker compose up -d app

# document-only, smaller image (no ffmpeg; Docling OCR still included)
MRAG_TARGET=runtime-slim MRAG_EXTRAS=parsing,docling docker compose build app

# without OCR (smaller download, but scanned/image-only PDFs extract no text)
MRAG_EXTRAS=parsing docker compose build app

# full ML image (local embeddings/reranker + Whisper/OpenCV on top of Docling) — largest
MRAG_EXTRAS=all docker compose build app
```

**Image sizes (measured before Docling became default):** slim/no-ffmpeg `328 MB` → static ffmpeg `525 MB` → apt ffmpeg `955 MB`.
Adding Docling (torch) grows the image notably (GB range) and the first processed PDF
downloads OCR/layout models (cached under `/data/hf_cache` in Docker, `~/.cache` natively).

We ship static ffmpeg by default. If you prefer a smaller image, use `runtime-slim` and bind-mount
the host binary when you need video:
```bash
docker run --rm -v "$(which ffmpeg):/usr/local/bin/ffmpeg:ro" ...
```

---

## 4. Configure (`.env`)

**Where settings live.** `multimodal_rag/config.py` defines the settings *schema, defaults and
validation* (committed code); `.env` provides *your* values (secrets + deployment choices;
git-ignored). Shell environment variables override `.env`, and `.env` overrides the defaults:
`config.py defaults → .env → environment variables`. So you don't need to touch `config.py` to
pick a mode, engine, or key — copy `.env.example` to `.env` and edit that. (Without a `.env` the
app starts on defaults, but embedding/vision/text calls fail because no API keys are set.)

Create it with `cp .env.example .env` (Windows: `Copy-Item .env.example .env`). The app root is
the directory that has the `.env`, so `data/` and `state.db` are created alongside it. The GUI
creates a default `.env` on first launch if one does not exist.

### Configure through the GUI (easiest)

Open **Settings** from the menu bar. The dialog edits the same `.env`; you do not need to
hand-edit the file. It also shows the effective values from environment variables, so a
shell-level override wins over what you type in the dialog.

- **Most changes apply live.** Search/retrieval, embedding, summary, VLM/LLM, theme, and
  `ACTIVE_DB_ENGINE` changes reload the backend in-process. Wait for the status bar to return to
  ready before adding files.
- **Storage paths need a restart** because open handles are bound at startup: `DATA_DIR`,
  `INBOX_DIR`, `PROCESSING_DIR`, `PROCESSED_DIR`, `FAILED_DIR`, `LANCEDB_DIR`, `STATE_DB_PATH`,
  `VLM_CACHE_DIR`.
- **The database engine switch is live.** When Postgres and LanceDB share `state.db`, the GUI
  copies missing vectors/records to the newly selected engine; it does not re-parse files.
- **Changing the embedding model/dimension invalidates the vector space.** Re-index affected
  files. `mrag-doctor` reports the resolved model, dimension, and whether embeddings are active.

For a fast CPU-friendly setup, choose:

```env
EMBEDDING_PROVIDER=LOCAL
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIMENSION=384
```

`bge-small-en-v1.5` is a good default for CPU-only machines: 384-dimensional vectors, a 512-token
input window, low memory use, and solid English retrieval quality. It needs no API key and
downloads once from Hugging Face. Use `bge-base-en-v1.5` / `768` if you prefer more capacity, or
`EMBEDDING_PROVIDER=NONE` for keyword-only search.

The most important settings:

| Variable | Default | What it does |
|---|---|---|
| `SYSTEM_MODE` | `MULTI_MODAL_RAG` | content policy: `DOC_ONLY_RAG` = documents+images (audio/video skipped), `MULTI_MODAL_RAG` = + audio/video |
| `ACTIVE_DB_ENGINE` | auto | storage engine; default from `SYSTEM_MODE`, set to override |
| `MRAG_HOME` | `~/.multimodal_rag` (or cwd if `.env` present) | app root for `.env`, `data/`, `state.db` |
| `SEARCH_MODE` | `hybrid` | `auto` / `chunk` / `summary` / `hybrid` / `keyword` |
| `SEARCH_TOP_K` | `10` (`.env.example` uses `5`) | passages retrieved before reranking/answer generation |
| `ANSWER_TOP_K` | `3` | passages sent to the answer LLM; `0` = all retrieved |
| `KEYWORD_EXPANSION` | unset → `CORPUS` in keyword-only/no-embedding mode | expansion: `OFF` / `CORPUS` / `SYNONYMS` / `CORPUS+SYNONYMS` |
| `KEYWORD_EXPANSION_TERMS` | `3` | maximum expansion terms added per keyword query |
| `TYPO_CORRECTION` | `LOCAL` | `LOCAL` / `LLM` / `OFF` (local is free, includes unique-prefix completion; LLM rewrites once per search and falls back to local) |
| `SUMMARY_PROVIDER` | `LOCAL` | `LOCAL` / `LLM` / `AUTO` |
| `SUMMARY_SEMANTIC_UNIT` | `CHUNK` | `CHUNK` (no extra embeddings) / `SENTENCE` |
| `SUMMARY_TARGET_WORDS` / `SUMMARY_OUTPUT_MAX_CHARS` | `120` / `900` | summary text budget; independent of embedding dimension, automatically clamped to the embedding model's input-token window |
| `DOC_CHUNK_SIZE` / `DOC_CHUNK_OVERLAP` | `800` / `100` | word-based chunk upper bounds (rarely binding at the token targets) |
| `DOC_CHUNK_TOKENS` / `DOC_CHUNK_OVERLAP_TOKENS` | `0` / `0` | token target; `0` = 450 tokens in both modes (capped by 90% of the window only with embeddings); also sets the word-overlap ratio |
| `GUI_THEME` | `Graphite` | desktop GUI colors: `Midnight` / `Graphite` / `Ocean` / `Daylight` (applies immediately) |
| `RERANKER_PROVIDER` | `LOCAL` | set `NONE` to disable reranking |
| `POSTGRES_HOST/PORT/USER/PASSWORD/DB` | `localhost/5432/postgres/password/rag_db` | Postgres connection (DB auto-created; full `POSTGRES_URL` overrides) |
| `LANCEDB_DIR` | `<app_root>/data/lancedb_lakehouse` | LanceDB store |

Relative paths are resolved against `MRAG_HOME`/app root, **not** your shell's cwd.

Local summaries use semantic LexRank over the chunk vectors already computed by ingestion.
`CHUNK` mode adds no embedding or LLM calls; `SENTENCE` mode adds one sentence-embedding pass.
All summary knobs (`SUMMARY_*`, `ENABLE_DOCUMENT_SUMMARIES`) are in the GUI Settings screen.

Summary length has two layers. `SUMMARY_TARGET_WORDS` and `SUMMARY_OUTPUT_MAX_CHARS` are the
user-facing word/character budgets. Independently of `EMBEDDING_DIMENSION`, the final text is
also clamped to 80% of the configured embedding model's input-token window (for example 204
tokens for MiniLM-L6-v2, 409 for bge-small/base, 1638 for Gemini/EmbeddingGemma 2). If the
clamp changes the text, a warning is logged and the summary record stores
`summary_embedding_token_clamped=true` plus the effective `summary_embedding_token_budget`.
Run `mrag-doctor` to see the resolved `embed_token_cap` and effective chunk token budget.
Dimension and token window are different: MRL truncation (128/256/512/768) changes vector
size, not how much text the encoder reads. Summaries are built from de-overlapped chunk text
using stored source offsets, so adjacent chunk overlap does not duplicate sentences.

Transcript chunks are capped by audio duration (`WHISPER_CHUNK_MAX_SECONDS`, default 10s) as
well as the token/word budget, so slow speech cannot become one huge passage. For video,
summaries prefer the transcript and omit frame captions when speech was transcribed; duplicate
captions are dropped otherwise. Chunks have the same protection and are sentence-aware. `DOC_CHUNK_SIZE` / `DOC_CHUNK_OVERLAP` remain word caps, but embedded chunks target about
450 tokens and are always capped
at 90% of the embedding model's input window. With MiniLM or bge-small, the 450 target is
automatically reduced to roughly 150–340 words so the embedder never silently drops the tail of
a passage. With `EMBEDDING_PROVIDER=NONE` (keyword-only search) there is no embedding window,
so the same 450-token target applies without that cap. Whole sentences are packed to the
budget; only an oversized sentence is hard-split, and overlap prefers a preceding sentence
boundary when it fits. Set `DOC_CHUNK_TOKENS` / `DOC_CHUNK_OVERLAP_TOKENS` to override the
target; explicit values are clamped only when embeddings are enabled. CJK text is split by
tokens/characters rather than whitespace, so
it is covered too. Each chunk stores source offsets and summaries use them to remove overlap
before sentence selection, so boundary text is not repeated. Re-index after changing chunk
settings or upgrading from an older chunker; old records without offsets use the legacy
word-overlap fallback.

### Purpose-first providers (single source of truth)

Configuration is grouped by **purpose**, not vendor. Each purpose has its own provider, model and
key — so there is nothing to cross-reference:

| Purpose | Provider | Model | API key | Base URL |
|---|---|---|---|---|
| Embeddings | `EMBEDDING_PROVIDER` | `EMBEDDING_MODEL` | `EMBEDDINGS_API_KEY` | `EMBEDDING_BASE_URL` |
| Vision (VLM) | `VLM_PROVIDER` | `VLM_MODEL` | `VLM_API_KEY` | `VLM_BASE_URL` |
| Text (LLM) | `LLM_PROVIDER` | `LLM_MODEL` | `LLM_API_KEY` | `LLM_BASE_URL` |

Keys are **blank by default** and are intentionally per-purpose: a Google key is written once per
role that uses it (e.g. `EMBEDDINGS_API_KEY` *and* `VLM_API_KEY`). This keeps each purpose
self-contained and readable.

Example (Google for embeddings + vision, Gemini for text too):
```env
EMBEDDING_PROVIDER=GOOGLE
EMBEDDINGS_API_KEY=AIza...
VLM_PROVIDER=GEMINI
VLM_API_KEY=AIza...
LLM_PROVIDER=GEMINI
LLM_API_KEY=AIza...
```

To get a free key, open <https://aistudio.google.com/app/apikey>, create a key, and paste the
same value into each Google/Gemini role you enable. If you prefer not to repeat it, the legacy
`GOOGLE_API_KEY` value is used as a fallback for all Google/Gemini purposes; the per-purpose
keys stay authoritative when set.

Any purpose without a usable key simply disables that feature (retrieval still works without the
text LLM). Run **`mrag-doctor`** to print each purpose's resolved provider/model/key and whether it
is active.

> **Fallbacks:** the older vendor-centric names (`GOOGLE_API_KEY`, `DEEPSEEK_API_KEY`,
> `TEXT_LLM_PROVIDER`/`VISION_LLM_PROVIDER`, `DEEPSEEK_BASE_URL`) are still honoured when a
> purpose's own field is empty, so existing `.env` files keep working. Prefer the purpose fields.

Minimal `.env` for the recommended Windows setup (local embeddings, Google vision/answers):
```env
SYSTEM_MODE=MULTI_MODAL_RAG
ACTIVE_DB_ENGINE=LANCEDB
EMBEDDING_PROVIDER=LOCAL
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIMENSION=384
SUMMARY_PROVIDER=LOCAL
VLM_PROVIDER=GEMINI
VLM_API_KEY=AIza...
LLM_PROVIDER=GEMINI
LLM_API_KEY=AIza...
```
To use the same key without repeating it, set `GOOGLE_API_KEY=AIza...` and omit the two
per-role keys. Omit the Google values entirely for offline document search.

---

## 5. Start Postgres (only for `DOC_ONLY_RAG` or the Postgres backend)

```bash
# from the repository root
make pg-up        # or:  docker compose up -d postgres
make pg-admin     # optional pgAdmin UI at http://localhost:5050 (admin@local.dev / admin)
make pg-psql      # psql shell
make pg-down      # stop (data kept)
make pg-reset     # DESTRUCTIVE: delete the pgdata volume
```
The image is `pgvector/pgvector:pg16`; `docker/init-pgvector.sql` enables the `vector` extension.
`POSTGRES_URL` in `.env` already points at `localhost:5432`.

---

## 6. Data layout

Everything lives under the **app root** (`MRAG_HOME`, else `~/.multimodal_rag`, else cwd):

```
<app_root>/
├── inbox/             # drop files here
├── processing/        # in-flight (moved automatically)
├── processed/         # success archive
├── failed/            # errors, kept for inspection
├── state.db           # SQLite ingestion queue/state (NOT vectors)
├── lancedb_lakehouse/ # LanceDB vector store (multimodal mode)
├── video_frame_cache/ # GUI video-preview frames
└── vlm_cache/         # cached frame captions
```
With Docker, this maps to the mounted `./data` directory.

> **Ingest state vs. vectors.** `state.db` (the queue/status list shown in the GUI) is **per app root** by default, while the vectors live in the shared database (Postgres/LanceDB). Don't run two app roots (e.g. a source checkout *and* a built exe, or two copies of the app) against the **same** vector DB and expect independent listings — they share vectors and will disagree about the queue. To use one shared record set on purpose, leave `STATE_DB_PATH=` blank so state is keyed by the vector store. For genuinely separate datasets, use a different `POSTGRES_DB` / `LANCEDB_DIR`.

---

## 7. Use it

### 7.1 Index files
```bash
# drop files into the inbox, then:
mrag-ingest run          # watch the inbox continuously and process

# or one-off / explicit:
mrag-ingest add ./my_folder
mrag-ingest add ./report.pdf --force
mrag-ingest scan         # enqueue everything already in the inbox
```
The pipeline: parse → chunk → embed → (local per-file summary, or optional LLM) → store →
rebuild indexes. Local summaries use semantic LexRank over the existing chunk vectors, so
`CHUNK` mode adds no embedding or LLM calls.

### 7.2 Monitor
```bash
mrag-ingest status            # counts per status
mrag-ingest status --failed   # list failures with error messages
mrag-ingest retry all         # requeue failures
mrag-ingest reset-stale       # requeue items stuck in "processing"

# After enabling embeddings, embed records that were indexed without vectors:
mrag-ingest backfill-embeddings
```

### 7.3 Search (`mrag-query`)
```bash
mrag-query "the story about a man who owns a mansion"     # hybrid search (default)
mrag-query "mansion" --mode summary -k 5                   # file-level only
mrag-query "invoice 4831" --mode keyword                   # no query embedding
mrag-query "mansion" --answer                              # + LLM answer with [n] citations
mrag-query "mansion" --json                                # machine-readable output
mrag-query "revenue" --folder /data/processed/reports       # restrict to a path prefix
mrag-query -i                                               # interactive REPL
mrag-query "mansion" --no-rerank                            # skip the cross-encoder
```
Flags: `--mode {auto,chunk,summary,hybrid,keyword}`, `-k/--top-k`, `--folder`, `--answer`,
`--json`, `--rerank/--no-rerank`, `-i/--interactive`. Defaults come from your `.env`
(`SEARCH_MODE`, `SEARCH_TOP_K`, reranker settings).

`keyword` mode never embeds the query and never runs the cross-encoder. When `KEYWORD_EXPANSION`
is unset, keyword-only and no-embedding searches default to `CORPUS`. Set `OFF` to keep the query
literal, which is usually best for exact names and codes. `CORPUS` adds related terms learned from
the top lexical hits (for example furniture near chair); `SYNONYMS` adds generic synonyms from the
bundled compact map; `CORPUS+SYNONYMS` combines both. Everything stays local with no downloads.
Lexical search also matches source paths/filenames, local typo correction completes
unambiguous prefixes, and numeric grouping is normalized so `6000` matches `6,000`. The optional
LLM answer still works with keyword retrieval; `ANSWER_TOP_K` limits how many retrieved passages
are sent to it.

`--answer` calls the configured text LLM (`LLM_PROVIDER`) with the retrieved passages
and asks for a cited answer; without a key it prints the passages and a note instead.

In Docker, the image entrypoint is `mrag-ingest`, so override it for queries:
```bash
docker compose run --rm --entrypoint mrag-query app "mansion garden" --answer
```

#### Library equivalent
```python
from multimodal_rag.config import get_settings
from multimodal_rag.database import get_repository
from multimodal_rag.pipeline.embedding import get_embedding_engine
from multimodal_rag.search.hybrid import hybrid_search

settings = get_settings()
repo = get_repository(settings); repo.initialize()
embedder = get_embedding_engine(settings)

query = "the story about a man who owns a mansion"
results, file_groups = hybrid_search(repo, query, embedder.embed_query(query), top_k=5)

for r in results:
    ts = f" @ {r.record.timestamp_start:.1f}s" if r.record.timestamp_start else ""
    print(f"{r.score:.3f}  {r.record.source_path}{ts}")

for group in file_groups:
    print("FILE:", group.filename, "—", group.summary[:80])
```

### 7.4 Automatic ingestion
Dropping a file into `data/inbox/` is picked up **automatically while the desktop GUI is open**
or while an ingestion loop is running — `mrag-ingest run` (or the Docker `app` service). The GUI
starts the same watcher and worker after the backend initializes. The watcher watches `inbox/`
specifically, not the `data/` root; other subfolders (`processing/`, `processed/`) are managed by
the worker and should not be used as drop targets. With no GUI/watcher active, nothing happens
until you call `mrag-ingest scan` or `add`.

### 7.5 Switch DB engines
`SYSTEM_MODE` controls the default engine and content policy:
```env
# Document-only (Postgres + pgvector + tsvector)
SYSTEM_MODE=DOC_ONLY_RAG

# Multimodal (LanceDB)
SYSTEM_MODE=MULTI_MODAL_RAG
```
Each engine keeps its own store. To avoid re-parsing and re-embedding, copy the existing
index first:
```bash
mrag-db-copy                                  # current engine -> the other engine
mrag-db-copy --from LANCEDB --to POSTGRES --replace
mrag-db-copy --dry-run                        # show source/target counts only
```
Stop the GUI and ingestion worker before copying. The target must use the same embedding
model/dimension. `--replace` is required if the target already has records. The command also
copies engine-specific `state.db` queue/status rows when applicable (`--no-state` skips that).
In the GUI, switching `ACTIVE_DB_ENGINE` performs the copy live when `state.db` is shared; CLI
users run `mrag-db-copy` first. After copying, set `ACTIVE_DB_ENGINE` to the target in
**Settings** (applied live) or `.env`. Re-index instead when
chunking or embedding settings changed, since copied vectors then describe the old chunks.
Engine choice itself is not part of the index fingerprint, so a copied index switches engines
without an "Index settings changed" prompt; that warning is reserved for vector-affecting settings.

### 7.6 Desktop GUI (`mrag-gui`)
A native three-pane PySide6 window (ingest | chat | preview). It is a front-end over the same
backend — indexing runs in the durable queue in background threads, and searches use
`mrag-query`'s hybrid retrieval.

```bash
mrag-gui                     # PySide6-Essentials is already in requirements.txt
# (extras users: pip install -e ".[gui]")
```
What you can do:
- **Add Folder...** / **Add Files...** queue a chosen location for indexing (runs in the background).
- **Open Inbox** opens `data/inbox/`; files dropped there are picked up automatically by the watcher.
- Ask questions in the center chat; click a source to preview it on the right. Image
  results show a thumbnail above their OCR/caption text; click the thumbnail to
  open the original image in your default viewer. Video results show a frame from
  the cited timestamp; click it to play from that moment (VLC when installed,
  otherwise the default player from the start). After each successful query the
  preview pane follows the top cited source (or the best-ranked match when the
  answer has no citations), so it never keeps a previous result on screen.
- The chat input stays disabled while the embedding model and configured reranker
  load and warm up, and while a query is running. Wait for the current answer
  before asking the next question.
- Switching `ACTIVE_DB_ENGINE` copies missing processed vectors from the previous engine
  automatically when `state.db` is shared, so no re-ingestion is needed. The switch now happens
  live from **Settings** as well as at startup. If that copy fails, rows show
  `processed (not in index)`; click **Reindex** to requeue just those files. `mrag-db-copy`
  remains available for explicit full-engine migrations.
- **Settings...** (top-level in the menu bar) edits `.env` (keys, mode, providers, theme, DB).
  Fields are grouped as **General**, **Embeddings**, **Text LLM**, **Vision (VLM)**,
  **Summaries**, **Postgres**, **LanceDB**, and **Document Parsing**. The source line at the
  bottom shows exactly which `.env` will be read/written, and Save creates it if missing.
  Changes apply immediately by rebuilding the backend in-process. DB-engine switches also
  reload live and auto-copy missing vectors when `state.db` is shared; storage-path changes
  are the exception and require an app restart.
- The status bar shows the active mode, indexed file/record counts, queue stats, and the result
  of the last backend reload.

Notes:
- **Cross-platform**: Windows, macOS and Linux desktops. The code has no OS branching.
- **Headless servers**: `mrag-gui` detects a missing `DISPLAY`/`WAYLAND_DISPLAY` and prints a
  hint (exit code `2`) instead of a cryptic Qt error. Use the CLI there.
- The GUI is **not available in Docker** (no display server); run it natively. Minimal Linux
  servers may also need Qt system libs: `apt install libgl1 libegl1 libxkbcommon0 libfontconfig1 libdbus-1-3`.
- LanceDB (`pylance`) needs AVX2 at runtime. On a CPU without it, set
  `ACTIVE_DB_ENGINE=POSTGRES` — multimodal content still works; a warning banner appears when
  LanceDB is selected.
- If embeddings are not configured (or `EMBEDDING_PROVIDER=NONE`), the window still opens in
  keyword-only mode: files are indexed, keyword/auto search works, and vector modes stay disabled.
  Configure an embedding model later and use **File → Backfill embeddings** to vectorize existing
  records without re-parsing them.
- The **"LLM answer"** checkbox (default on) synthesizes a cited answer with the configured
  text LLM; uncheck it for retrieved passages only. Without a key it shows passages and a note.

#### Build a Windows `.exe` (optional)
```bash
cd multimodal_rag_core
pip install -e ".[gui,build]"     # installs PySide6 + PyInstaller
python build_gui.py               # builds and seeds .env / .env.example next to the exe
# -> dist/MultiModalRAG/MultiModalRAG.exe
```
`build_gui.py` copies `.env.example` to `dist/MultiModalRAG/.env` (blank keys, only when missing)
so the distribution is self-contained: the recipient either edits that `.env` or uses
**Settings**. The app watches `.env` and applies changes without a restart, including
`ACTIVE_DB_ENGINE` switches; storage-path changes require restarting the app.
The exe folder is the app root (`MRAG_HOME` defaults there when frozen), regardless of which folder
the app is launched from — so `data/` and `state.db` live beside the app (portable), and the
`.env` next to the exe is the file the app reads and the one **Settings** writes to.
Rebuilding never overwrites an existing `.env`, and overlaying a new release keeps it too.
PyInstaller does **not** cross-compile — build on Windows for a Windows exe.
Set `ONEFILE = True` in `mrag-gui.spec` for a single-file exe (slower startup).

The build embeds product/version metadata (`mrag_gui_version.txt`), uses folder mode (not
onefile) and UPX off — all of which reduce antivirus false positives. PyInstaller output is still
unsigned, and unsigned exes are frequently flagged (especially anything bundling ffmpeg/yt-dlp-like
tools). For public distribution, **code-sign** the exe (`signtool`) and submit any false positive
to Microsoft WDSI and the AV vendor. We deliberately do **not** bundle ffmpeg inside the exe.

---

## 8. Docker details

```bash
make app-build     # build the app image
make app-up        # start the ingestion service (watch + process)
make app-logs      # tail logs
make app-shell     # shell inside the image
make app-down      # stop
```
Useful environment overrides (in the compose `.env` or shell):
| Var | Effect |
|---|---|
| `MRAG_EXTRAS` | pip extras baked in (`parsing`, `lancedb,postgres,parsing`, `all`, ...) |
| `MRAG_TARGET` | `runtime` (with static ffmpeg) or `runtime-slim` (no ffmpeg) |
| `SYSTEM_MODE` | forwarded into the container |
| `GOOGLE_API_KEY`, `DEEPSEEK_API_KEY` | forwarded into the container |

One-off commands inside the image:
```bash
docker compose run --rm app add /data/inbox/video.mp4
docker compose run --rm app status --failed
```

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `LanceDB is not installed...` | missing backend extra | `pip install -e ".[lancedb]"` |
| `Illegal instruction` / crash importing lancedb | CPU lacks **AVX2** | set `ACTIVE_DB_ENGINE=POSTGRES` (keeps multimodal content), or run on a newer CPU |
| `psycopg` import error | Postgres extra missing | `pip install -e ".[postgres]"` |
| `requires GOOGLE_API_KEY` / embedding auth error | no embedding key | set `EMBEDDINGS_API_KEY` (or legacy `GOOGLE_API_KEY`), switch to `EMBEDDING_PROVIDER=LOCAL` + `.[local-embeddings]`, or use keyword-only `EMBEDDING_PROVIDER=NONE` |
| `Embedding dimension mismatch` | model/dimension changed after indexing | use the original model/dimension or re-index with the new one |
| Video audio is not transcribed | ffmpeg not on PATH | install ffmpeg (native), or use the default Docker image |
| Frames produce no captions | VLM not configured | set `VLM_PROVIDER=GEMINI` and `VLM_API_KEY` (or `GOOGLE_API_KEY`) |
| GUI shows a headless hint and exits | no display session | run on a desktop; use the CLIs on servers |
| Settings saved but search behavior did not change | backend still reloading or setting needs restart | wait for the status bar; storage-path settings need an app restart |
| Files stuck in `processing` | crash mid-job | `mrag-ingest reset-stale`, then `run` |
| Same file indexed twice | different content? | dedupe is by content hash; use `--force` to overwrite |
| Postgres won't start on WSL | pgdata on `/mnt/*` | use the named Docker volume (default) |
| Slow/failed writes | `data/` on network/OneDrive | move `data/` to a local disk |

---

## 10. Cost & free-tier tips

- The recommended default is **local bge-small embeddings**, so ingestion and search cost nothing
  beyond CPU time. Google AI Studio is only needed for VLM captions and optional LLM answers.
  Free-tier quotas and model availability can change; check AI Studio.
- **Batch** work: `VLM_BATCH_SIZE` frames per caption request, `EMBEDDING_BATCH_SIZE` texts per embed call.
- **Reduce work**: scene-based keyframe sampling and a disk cache mean re-runs cost nothing.
- Rough VLM cost for a 1-hour video (Muse Spark 1.3 Contributor): ~**$0.05** at ~1 keyframe/5s,
  ~**$0.47** at 2 FPS. Whisper, ffmpeg, Docling, LanceDB, Postgres-FTS and BM25 are all local/free.

---

## 11. FAQ

**Do I need Docker?** No. Docker is only for Postgres and/or reproducible app images. Native
pip installs work for LanceDB mode.

**Do I need a GPU?** No. Everything is CPU-first. A GPU only speeds up optional local models
(sentence-transformers, faster-whisper).

**Which embedder should I use?** Start with `LOCAL` and `BAAI/bge-small-en-v1.5` (384-dim): it
needs no key, runs on CPU, and is fast enough for most personal corpora. Use Google embeddings if
you want to offload CPU work, or `EMBEDDING_PROVIDER=NONE` for keyword-only search.

**Where are my API keys stored?** In `.env` (git-ignored). They are never baked into the image.

**Can I use my own LLM/VLM?** Yes — extend `utils/api_clients.py` and the provider enums in
`config.py`; the rest of the pipeline is provider-agnostic.

**How do I reset everything?** Delete the app-root `data/` (or `make pg-reset` for Postgres).
