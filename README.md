# DeepLens: Privacy-First Multi-Modal Semantic + Keyword File Explorer

> **A modern, cross-platform desktop application that replaces word-for-word filename lookups with deep, multi-modal natural-language search — combining semantic (vector), keyword (full-text/BM25), and document-summary signals via hybrid retrieval, running entirely offline for local privacy or accelerated in the cloud.**

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Desktop Framework](https://img.shields.io/badge/GUI-PySide6%20%2F%20Qt6-6C63FF.svg)](https://doc.qt.io/qtforpython-6/)
[![Agent Framework](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)

---

## 📖 Table of Contents

1. [Product Vision](#-product-vision)
2. [Target Architectures](#-target-architectures)
3. [Key Features](#-key-features)
4. [System Architecture](#-system-architecture)
5. [Security Hardening](#-security-hardening)
6. [Installation & Setup](#-installation--setup)
7. [Usage Guide](#-usage-guide)
8. [Developer Integration (MCP)](#-developer-integration-mcp)
9. [Development Roadmap](#-development-roadmap)

---

## 👁️ Product Vision

Traditional file search engines rely strictly on exact keyword matching on file names. **DeepLens** indexes local directories asynchronously to build a vector store representing documents, images, audio, and video files. Users can query their database using natural, conversational language to locate visual frames, subtitle intervals, audio timestamps, and structured document tables.

- **Privacy Mode (Mode A)**: Indexes and runs inference entirely on-device (CPU-friendly). No cloud telemetry.
- **Enterprise Mode (Mode B)**: Offloads heavy media operations to high-throughput cloud models and scalable SQL infrastructure.

---

## ⚙️ Target Architectures

DeepLens uses a **Repository & Factory Pattern** allowing you to hot-swap the entire backend pipeline via a single configuration flag (`DEEPLENS_MODE` in `.env`).

| Layer | Mode A: Local Privacy Mode (CPU-friendly) | Mode B: Cloud / CV Showcase Mode |
|---|---|---|
| **Vector DB** | **LanceDB** (Embedded, disk-based, zero RAM footprint) | **PostgreSQL + pgvector** (Bundled or remote SQL service) |
| **Embedding Engine** | **Configurable local multi-modal model** (defaults to **Jina-CLIP-v2**, <1B params, unit-normalized 1024-dim vectors — set via `DEEPLENS_LOCAL_EMBEDDING_MODEL`) | **Gemini Text Embedding 004** (3072-dim dense vectors) |
| **Chat LLM / Rewriter** | **Llama-3.2-3B** (or **Phi-4-mini**) running locally via **Ollama** | **Gemini 2.5 Flash** (Generous free tier via AI Studio) |
| **Audio Transcription** | **faster-whisper** (INT8 quantized CPU execution mode) | **Gemini 2.5 Flash** (Native audio modal processing) |
| **Document Parsing** | **Docling** (RT-DETR layout engine + Granite-Docling models) | **Docling** (Layout-aware markdown conversion) |

---

## ✨ Key Features

- 📑 **Layout-Aware Document Parsing**: Utilizes **Docling** to transform PDFs, DOCX, and XLSX sheets into structured Markdown (intact tables, headers, and codeblocks) before slicing them into structure-aware chunks.
- 🔤 **Optical Character Recognition (OCR)**: When enabled, text embedded inside images and scanned PDFs is extracted and indexed so screenshots, invoices, memes, and receipts become fully searchable. The pipeline is **efficient and intelligent** — a cheap heuristic pre-filter plus the OCR engine's own text detector skip recognition on text-free images, and results are cached by file hash to avoid re-computation. Configurable engine (`tesseract` or `easyocr`) and language code, with graceful degradation if an OCR dependency is missing.
- 🎙️ **Decoupled Audio-Visual Pipeline (Local)**:
  - **Visual**: Samples frame streams at 1 FPS using OpenCV, feeding images to Jina-CLIP for visual indexing.
  - **Audio**: Extracts the audio track via `ffmpeg` and runs it through `faster-whisper` for timestamp-aligned transcription.
- 🔀 **Agentic RAG State Machine**: Controlled using a **LangGraph state graph** rather than linear chains. A feedback edge loops back to the Query Rewriter if search results fail relevance filters.
- 📚 **Summary-Aware Hybrid Search**: Documents are indexed both as fine-grained **chunks** *and* as a single per-file **summary** record. Retrieval fuses semantic (vector), lexical (BM25-style keyword), and document-summary signals with **Reciprocal Rank Fusion**, so you can locate *the file* from a description ("the story about a man who owns a big mansion") — not just scattered passages. See [Summary-Aware Hybrid Search](#-summary-aware-hybrid-search).
- 🔌 **Model Context Protocol (MCP)**: Features a built-in FastMCP Server. Exposes semantic retrieval hooks as tools to Cursor, Claude Desktop, and VS Code.
- 💻 **Responsive PySide6 GUI**: Clean 3-panel layout (Left checkbox filter tree, Center chat panel, Right interactive preview panel) using background `QThread` workers to prevent thread blocking.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Presentation Layer                    │
│          PySide6 / Qt6 Desktop GUI + System Tray        │
├─────────────────────────────────────────────────────────┤
│                    Application Layer                     │
│    LangGraph Orchestrator  ·  MCP Server  ·  CLI        │
├─────────────────────────────────────────────────────────┤
│                     Service Layer                        │
│  Ingestion Pipeline  ·  Query Pipeline  ·  File Watcher │
├─────────────────────────────────────────────────────────┤
│                    Repository Layer                      │
│  DocumentRepository (ABC)                                │
│  ├─ LanceDBRepository    (Mode A)                       │
│  └─ PgVectorRepository   (Mode B)                       │
├─────────────────────────────────────────────────────────┤
│                     Engine Layer                         │
│  EmbeddingEngine (ABC)  ·  ChatEngine (ABC)             │
│  ├─ JinaClipEngine      ├─ OllamaEngine                │
│  └─ GeminiEmbedEngine   └─ GeminiChatEngine             │
├─────────────────────────────────────────────────────────┤
│                   Infrastructure                         │
│  Config  ·  Logging  ·  Encryption  ·  Task Queue       │
└─────────────────────────────────────────────────────────┘
```

---

## 🔒 Security Hardening

Security is treated as a first-class citizen in the codebase:

- **Zip Slip Mitigation**: The archive extractor checks and canonicalizes paths, rejecting any paths attempting parent directory traversal (`..`) or writing outside target directories.
- **Path Traversal Protection**: Search scope filters are strictly validated against registered directory roots.
- **Safe Secrets Storage**: API keys are never stored in plaintext configuration files. They are fetched from the OS-level credential store using Python's `keyring` package.
- **Standardized MCP Interface**: Standard MCP tools are strictly read-only; write, overwrite, or delete file operations are not exposed.

---

## 🚀 Installation & Setup

### Prerequisites

You need the following system packages installed and added to your PATH:

- **ffmpeg** (For extracting audio tracks and slicing video clips)
- **unrar** / **7z** (For parsing RAR/7z archives)
- **Tesseract OCR** (Optional — enables OCR of images/scanned PDFs if `DEEPLENS_ENABLE_OCR=true`)
- **PyMuPDF** (Optional — enables rasterizing PDF pages for OCR fallback)

#### macOS

```bash
brew install ffmpeg unrar
```

#### Ubuntu / Debian

```bash
sudo apt update
sudo apt install ffmpeg unrar-free
```

### Installation Steps

1. **Clone the repository**:

   ```bash
   git clone https://github.com/MediaTools-tech/multi-modal-rag.git
   cd multi_modal_rag
   ```

2. **Install in editable dev mode**:
   We recommend using a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"
   ```

3. **Configure Environment Variables**:
   Copy the example environment template and modify it:

   ```bash
   cp .env.example .env
   ```

4. **Register Secrets securely (Cloud Mode)**:
   Use python keyring to store your Gemini API key:

   ```bash
   python3 -c "import keyring; keyring.set_password('deeplens', 'gemini_api_key', 'YOUR_GEMINI_API_KEY')"
   ```

---

## 🖥️ Usage Guide

### Running the Desktop GUI App

Start the main application GUI:

```bash
make run
```

or

```bash
python3 -m deeplens.main
```

### Using Makefile commands

The project includes a `Makefile` for developer lifecycle workflows:

```bash
make dev-install   # Install editable dependencies including pytest/ruff/mypy
make test          # Run unittest and integration suites
make lint          # Check formatting and style rules
make format        # Apply automatic lint formatting fixes
make type-check    # Run strict static mypy type analysis
make audit         # Audit codebase dependencies for CVE vulnerabilities
make clean         # Purge cache directories and temporary wheel files
```

---

## 🔌 Developer Integration (MCP)

DeepLens includes a standard **Model Context Protocol (MCP)** server module. This allows you to expose your indexed folders directly to LLM-capable developer tools like **Cursor**, **Claude Desktop**, or **VS Code**.

### Run the Standalone MCP Server

```bash
make run-mcp
```

or

```bash
python3 -m deeplens.mcp.server
```

### Configuring Claude Desktop

Add this to your Claude Desktop configuration file (typically `~/.profile` or `~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "deeplens": {
      "command": "python3",
      "args": ["-m", "deeplens.mcp.server"],
      "env": {
        "GEMINI_API_KEY": "YOUR_GEMINI_API_KEY"
      }
    }
  }
}
```

### Exposed MCP Tools

- `semantic_search(query, folder_filter?, file_type_filter?, search_mode?, top_k?)`: Run a similarity query over indexed documents and retrieve a citations-enriched answer. `search_mode` is `hybrid` (default), `summary`, or `chunk`.
- `find_documents(description, folder_filter?, file_type_filter?, top_k?)`: Locate whole files whose content matches a natural-language description, returning each match's path and generated summary — ideal for "find the story about a man who owns a big mansion".
- `list_indexed_folders()`: Return a list of all registered directories.
- `get_file_chunks(file_path)`: Return all raw text chunks and corresponding timestamps for a file.
- `get_indexing_status()`: Get current database metrics and counts.

---

## 📚 Summary-Aware Hybrid Search

### The problem

DeepLens stores every file as a set of **chunks** (passages, frames, transcript
segments). That is ideal for *answering* a question grounded in a passage, but it
cannot, by itself, **find a whole document from a description of its content**.
Given thousands of bank statements, letters, books, and stories, a query like
*"story of a man who owns a big mansion"* would scatter top matches across many
chunks; you would get a few relevant passages, not "the file."

The fix is a **document-level signal** in addition to the chunk-level one. DeepLens
generates one concise **summary** per text-bearing file at ingestion time and stores
it as a dedicated `record_type="summary"` vector record that still points back to the
source `absolute_path`. The retriever can therefore match a description to a *file*,
then attach the supporting chunks for grounding.

### Why not RAPTOR-Lite / per-file K-Means clustering?

Recursive clustering (RAPTOR) — summarizing K-Means clusters of a file's chunks into a
tree — is a powerful technique for **answering questions that span an entire long
document** (e.g. a book), because the answer must be aggregated across the whole work.
It is the wrong tool for *locating the file that matches a description*:

- A single per-file summary is the exact-right granularity for "find this document" —
  a cluster summary is not a document summary and does not map 1:1 to "the file".
- Per-file K-Means forces you to choose `k` (even "dynamically" from file size) and to
  re-cluster on every re-index, with no clear payoff for file-finding.
- It multiplies storage and latency for marginal recall gain on the locate-a-file task.

We recommend **summary records + hybrid fusion** as the default, and keep RAPTOR as a
possible *future* answer-aggregation tier (see Roadmap) rather than the retriever.

### How retrieval works

For each query, `search_mode` selects the strategy (default `hybrid`):

| Mode | Behaviour | Best for |
|------|-----------|----------|
| `chunk` | Passage-level ANN only (legacy). | Precise in-document lookups. |
| `summary` | Matches only per-file summary records. | "Find the document that matches this description." |
| `hybrid` | Fuses summary + chunk + lexical signals via RRF. | Mixed / natural-language queries. |

The hybrid path builds three ranked lists and fuses them with **Reciprocal Rank
Fusion** (`1 / (k + rank)`):

1. **Vector** — semantic ANN over all records (chunks + summaries).
2. **Lexical (corpus-wide)** — a real full-text / BM25 search over the *entire*
   index via the repository's `search_lexical` (Postgres `tsvector` + GIN in cloud
   mode; LanceDB FTS, with an in-Python BM25 fallback, in local mode). This catches
   distinctive terms and proper nouns (e.g. *"mansion"*, character names) and —
   crucially — **covers every document, not just the vector top-k**.
3. **Summary** — vector ranking restricted to file-level summary records, giving
   documents a second chance to surface for description-style queries.

### Optional cross-encoder re-ranker (CPU-friendly by design)

Fusion already re-ranks, but for maximum quality you can enable a **cross-encoder
re-ranker** (`enable_reranker=true`). It is **opt-in and off by default** so CPU-only
deployments pay zero cost. When enabled it scores **only the top `rerank_top_n`
fused candidates** (default `15`) — with the tiny default model
(`ms-marco-MiniLM-L-6-v2`, ~80 MB) that is <100 ms even on CPU. GPU/cloud users can
raise `rerank_top_n` for more recall before re-ranking. Install the optional
dependency with `pip install -e ".[reranker]"`.

Results are also grouped into **`FileSearchGroup`** objects (file path, generated
summary, best score, top supporting chunks) so the answer can point you straight at the
document.

### Configuration

| Setting | Env var | Default | Purpose |
|---------|---------|---------|---------|
| `search_mode` | `DEEPLENS_SEARCH_MODE` | `hybrid` | `chunk` / `summary` / `hybrid`. |
| `hybrid_rrf_k` | `DEEPLENS_HYBRID_RRF_K` | `60` | RRF constant (higher = ranks matter less). |
| `enable_reranker` | `DEEPLENS_ENABLE_RERANKER` | `false` | Apply a cross-encoder re-ranker to fused candidates. |
| `reranker_model` | `DEEPLENS_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model id. |
| `rerank_top_n` | `DEEPLENS_RERANK_TOP_N` | `15` | Fused candidates scored by the re-ranker before returning top_k. |
| `enable_document_summaries` | `DEEPLENS_ENABLE_DOCUMENT_SUMMARIES` | `true` | Generate + index per-file summary records. |
| `summary_max_chars` | `DEEPLENS_SUMMARY_MAX_CHARS` | `4000` | Max source text fed to the summarizer. |

> **Re-index note**: enabling summaries on an *existing* index adds new `record_type`
> / `summary` columns (pgvector migrates automatically; LanceDB migrates best-effort at
> open). To populate summaries for already-indexed files, re-index the folder.

---

## 🗺️ Development Roadmap

- [x] **Phase 1: Foundation**: Abstract interfaces, factory loading, and LanceDB/pgvector tables.
- [x] **Phase 2: Ingestion & Parsing**: Docling markdown parser, image EXIF, and watchdog directory monitor.
- [x] **Phase 3: Decoupled Media Chunkers**: OpenCV frame extractor, audio WAV converter, faster-whisper worker, and subtitle parsers.
- [x] **Phase 4: Agentic RAG**: LangGraph loop implementation (Rewriter -> Retriever -> Evaluator -> Generator).
- [x] **Phase 5: MCP Tooling**: FastMCP stdio transport server wrapper.
- [x] **Phase 6: Native Client Layout**: PySide6 QSplitter window design, QThread workers, card overlays, and dark/light styling.
- [ ] **Phase 7: Production Packaging**: standalone executable builds via PyInstaller or Briefcase.
