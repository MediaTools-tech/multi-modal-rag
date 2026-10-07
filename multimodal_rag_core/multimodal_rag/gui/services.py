from __future__ import annotations

import logging
from pathlib import Path

from multimodal_rag.config import IngestionMode, Settings, get_settings
from multimodal_rag.ingestion.queue import IngestQueue
from multimodal_rag.ingestion.state import (
    INDEX_FINGERPRINT_KEYS,
    StateStore,
    fingerprint_of,
)
from multimodal_rag.ingestion.watcher import InboxWatcher
from multimodal_rag.ingestion.worker import IngestionWorker
from multimodal_rag.pipeline.context import PipelineContext, build_context
from multimodal_rag.pipeline.registry import build_handlers
from multimodal_rag.query import run_search
from multimodal_rag.utils.api_clients import generate_rag_answer
from multimodal_rag.utils.token_tracker import TokenTracker

logger = logging.getLogger(__name__)


class BackendService:
    """Thin, Qt-free facade over the repository, embedder, queue and retrieval."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.tracker = TokenTracker()
        self.last_request_usage: dict = {}
        self.context: PipelineContext | None = None
        self.store: StateStore | None = None
        self.queue: IngestQueue | None = None
        self.worker: IngestionWorker | None = None
        self.watcher: InboxWatcher | None = None
        self.last_answer_error: str | None = None
        # Why the embedder is missing (version/download hint), if known.
        # build_context swallows engine-construction errors (lazy weights), so
        # capture the cheap eager checks here for the GUI to display.
        self.init_error: str | None = None

    def initialize(self) -> None:
        self.settings.ensure_data_dirs()
        # Share one tracker so session totals include embeddings, VLM, summaries
        # and answers (build_context would otherwise use its own throwaway one).
        self.context = build_context(self.settings, require_embedder=False, tracker=self.tracker)
        if self.context is not None and self.context.embedder is None:
            try:
                from multimodal_rag.pipeline.embedding import get_embedding_engine

                get_embedding_engine(self.settings, self.tracker)
            except Exception as exc:  # noqa: BLE001
                self.init_error = str(exc)
            else:
                self.init_error = None
        self.store = StateStore(self.settings.STATE_DB_PATH)
        self.store.reset_stale()
        self.queue = IngestQueue(
            self.store,
            self.settings.MAX_INGEST_RETRIES,
            allow_kind=self.settings.allows_kind,
        )
        self.worker = IngestionWorker(
            queue=self.queue,
            handlers=build_handlers(self.context),
            processing_dir=self.settings.PROCESSING_DIR,
            processed_dir=self.settings.PROCESSED_DIR,
            failed_dir=self.settings.FAILED_DIR,
            worker_count=self.settings.WORKER_COUNT,
            file_stable_seconds=self.settings.FILE_STABLE_SECONDS,
            inbox_dir=self.settings.INBOX_DIR,
            allow_kind=self.settings.allows_kind,
            reindex=self.context.repository.reindex if self.context is not None else None,
        )
        # Auto-watch the inbox (same mechanism the CLI `run` uses) so files
        # dropped into INBOX_DIR are picked up without pressing Scan Inbox.
        if self.settings.INGESTION_MODE is IngestionMode.WATCH:
            self.watcher = InboxWatcher(
                self.settings.INBOX_DIR,
                on_file=lambda path: self.queue.enqueue_path(path),
                ignore_extensions=self.settings.ingest_ignore_extensions,
                poll_interval=self.settings.WATCH_POLL_INTERVAL_SEC,
                on_delete=self._on_source_deleted,
            )
            self.watcher.start()

    def _on_source_deleted(self, path) -> None:
        """Drop the queue row and vectors when a tracked inbox file is deleted."""
        if self.store is not None:
            self.store.remove_by_path(path)
        repository = self.context.repository if self.context is not None else None
        if repository is not None:
            try:
                repository.delete_by_path(str(path))
                repository.reindex()
            except Exception as exc:  # noqa: BLE001
                logger.warning("vector delete failed for %s: %s", path, exc)

    @property
    def repository(self):
        return self.context.repository if self.context else None

    @property
    def embedder_ready(self) -> bool:
        return bool(self.context and self.context.embedder is not None)

    def start_worker(self) -> None:
        if self.worker is not None:
            self.worker.start()

    def stop_worker(self) -> None:
        if self.watcher is not None:
            self.watcher.stop()
        if self.worker is not None:
            self.worker.stop()

    def restart_worker(self) -> None:
        if self.worker is not None:
            self.worker.restart()

    def enqueue(self, path: str | Path) -> int:
        if self.queue is None:
            return 0
        target = Path(path)
        if target.is_dir():
            return self.queue.enqueue_directory(
                target, ignore_extensions=self.settings.ingest_ignore_extensions
            )
        if target.is_file():
            return 1 if self.queue.enqueue_path(target) is not None else 0
        return 0

    def search(self, query: str, mode: str | None = None, top_k: int | None = None):
        if self.context is None or self.context.embedder is None:
            raise RuntimeError(
                "Embedding engine not configured. Set EMBEDDING_PROVIDER=GOOGLE + "
                "GOOGLE_API_KEY (or LOCAL + sentence-transformers) and restart."
            )
        mode = mode or self.settings.SEARCH_MODE.value
        top_k = top_k or self.settings.SEARCH_TOP_K
        rerank = self.settings.active_reranker != "NONE"
        logger.info("search query=%r mode=%s top_k=%s rerank=%s", query, mode, top_k, rerank)
        return run_search(
            self.settings,
            self.context.repository,
            self.context.embedder,
            query,
            mode,
            top_k,
            None,
            rerank,
        )

    def answer(self, question: str, results: list) -> str | None:
        self.last_answer_error = None
        try:
            return generate_rag_answer(self.settings, self.tracker, question, results)
        except Exception as exc:  # noqa: BLE001
            self.last_answer_error = f"{type(exc).__name__}: {exc}"
            return None

    def stats(self) -> dict:
        repo_stats: dict = {}
        if self.context and self.context.repository:
            try:
                repo_stats = self.context.repository.get_stats()
            except Exception as exc:
                logger.warning("stats failed: %s", exc)
        return {"repo": repo_stats, "queue": self.store.stats() if self.store else {}}

    def queue_records(self) -> list:
        return self.store.list_by_status() if self.store else []

    def indexed_hashes(self) -> set[str] | None:
        """file_hash values in the vector store, or None if unavailable.

        None (backend down / query error) means "don't reconcile" so we never
        falsely flag everything as missing. An empty set is a valid answer and
        is combined with indexed paths by stale_record_ids.
        """
        if self.context is None or self.context.repository is None:
            return None
        try:
            return self.context.repository.indexed_file_hashes()
        except Exception as exc:  # noqa: BLE001
            logger.warning("indexed_file_hashes failed: %s", exc)
            return None

    def indexed_paths(self) -> set[str] | None:
        """source_path values in the vector store, or None if unavailable."""
        if self.context is None or self.context.repository is None:
            return None
        try:
            return self.context.repository.indexed_paths()
        except Exception as exc:  # noqa: BLE001
            logger.warning("indexed_paths failed: %s", exc)
            return None

    def stale_record_ids(self, records: list) -> set[int]:
        """Processed rows with no trace left in the vector store.

        A row counts as present if its content hash matches, its exact path
        matches, or its filename matches an indexed path (managed inbox files
        move between processing/processed folders, so paths shift while names
        don't). Conservative: only rows matching nothing are stale.
        """
        hashes = self.indexed_hashes()
        paths = self.indexed_paths()
        if hashes is None or paths is None:
            return set()
        basenames = {p.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] for p in paths}
        stale: set[int] = set()
        for record in records:
            if record.status != "processed":
                continue
            if record.sha256 and record.sha256 in hashes:
                continue
            if record.path in paths:
                continue
            name = record.path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            if name and name in basenames:
                continue
            stale.add(record.id)
        return stale

    def remove_missing(self, record_ids: set[int]) -> int:
        """Delete queue rows whose vectors are already gone. Source files kept."""
        if self.store is None:
            return 0
        removed = 0
        for record_id in record_ids:
            removed += self.store.delete(record_id)
        return removed

    @staticmethod
    def _fingerprint_value(settings: Settings, key: str) -> str:
        return fingerprint_of(settings)[key]

    def current_fingerprint(self) -> dict:
        return fingerprint_of(self.settings)

    def check_fingerprint(self) -> dict:
        """Compare stored index fingerprint vs current config.

        Returns {"status": "fresh"|"ok"|"mismatch", "diffs": [...],
        "stored": {...}, "current": {...}}. "fresh" stamps the current config
        (first run / empty state) and needs no action.
        """
        import json

        current = self.current_fingerprint()
        raw = self.store.get_meta("index_fingerprint") if self.store else None
        if not raw:
            has_vectors = False
            try:
                if self.context is not None and self.context.repository is not None:
                    stats = self.context.repository.get_stats() or {}
                    has_vectors = (stats.get("total_records", 0) or 0) > 0
            except Exception as exc:  # noqa: BLE001
                logger.warning("fingerprint stats check failed: %s", exc)
            if has_vectors:
                return {
                    "status": "mismatch",
                    "diffs": [
                        "index predates fingerprinting: vectors exist but were built "
                        "under unknown settings — reset and re-ingest to be sure"
                    ],
                    "stored": {},
                    "current": current,
                }
            if self.store:
                self.store.set_meta("index_fingerprint", json.dumps(current, sort_keys=True))
            return {"status": "fresh", "diffs": [], "stored": {}, "current": current}
        try:
            stored = json.loads(raw)
        except ValueError:
            stored = {}
        if set(stored) < set(INDEX_FINGERPRINT_KEYS):
            # Stamp from an older version: compare the shared subset, and if it
            # matches, silently migrate to the full stamp instead of nagging.
            shared_diffs = [
                f"{key}: index has {stored.get(key)!r}, config now {current[key]!r}"
                for key in INDEX_FINGERPRINT_KEYS
                if key in stored and stored.get(key) != current[key]
            ]
            if shared_diffs:
                return {
                    "status": "mismatch",
                    "diffs": shared_diffs,
                    "stored": stored,
                    "current": current,
                }
            if self.store:
                self.store.set_meta(
                    "index_fingerprint", json.dumps(current, sort_keys=True)
                )
            return {"status": "ok", "diffs": [], "stored": current, "current": current}
        diffs = [
            f"{key}: index has {stored.get(key)!r}, config now {current[key]!r}"
            for key in INDEX_FINGERPRINT_KEYS
            if stored.get(key) != current[key]
        ]
        return {
            "status": "ok" if not diffs else "mismatch",
            "diffs": diffs,
            "stored": stored,
            "current": current,
        }

    def reset_index(self) -> dict:
        """Full clean-slate reset: wipe ALL vectors + queue rows, restamp fingerprint.

        Works even when initialize() failed (e.g. the dimension guard): fresh
        repository/store handles are built instead of reusing the backend ones.
        Source files on disk are kept. Caller restarts the worker/backend.
        """
        import json

        from multimodal_rag.database import get_repository

        repository = None
        fresh = False
        if self.context is not None and self.context.repository is not None:
            repository = self.context.repository
        else:
            # initialize() failed midway (e.g. dimension guard): fresh handles.
            repository = get_repository(self.settings)
            fresh = True
        vectors = 0
        try:
            vectors = repository.clear() or 0
        finally:
            if fresh:
                try:
                    repository.close()
                except Exception:  # noqa: BLE001
                    pass
        store = self.store if self.store is not None else StateStore(self.settings.STATE_DB_PATH)
        queued = store.clear_queue()
        store.set_meta(
            "index_fingerprint", json.dumps(self.current_fingerprint(), sort_keys=True)
        )
        # Keep handles usable for the running session.
        if self.store is None:
            self.store = store
        if self.context is None:
            try:
                self.initialize()
            except Exception as exc:  # noqa: BLE001
                logger.warning("backend re-initialize after reset failed: %s", exc)
        return {"vectors": vectors, "queue": queued}

    def stamp_fingerprint(self) -> dict:
        """Trust the current index: record today's config as its fingerprint.

        Use when the index predates fingerprinting (or drift was reviewed and
        accepted). Wipes nothing; future startups compare against this stamp.
        """
        import json

        current = self.current_fingerprint()
        if self.store:
            self.store.set_meta(
                "index_fingerprint", json.dumps(current, sort_keys=True)
            )
        return current

    def remove_file(self, record) -> dict:
        """Remove a file from the queue list AND its indexed vectors.

        The source file on disk is left untouched. Returns
        ``{"vectors": <deleted chunk count>, "queue": <row removed>}``.
        """
        path = getattr(record, "path", None) or str(record)
        # Vectors may be stored under a different path than the current queue
        # row (inbox files are moved through processing/processed folders after
        # indexing), so sweep every source_path sharing this file's content hash.
        vector_paths = {path}
        sha = getattr(record, "sha256", None)
        if sha and self.context is not None and self.context.repository is not None:
            try:
                for chunk in self.context.repository.get_by_file_hash(sha):
                    if chunk.source_path:
                        vector_paths.add(chunk.source_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("hash lookup failed for %s: %s", path, exc)
        removed_vectors = 0
        if self.context is not None and self.context.repository is not None:
            for source_path in vector_paths:
                try:
                    removed_vectors += (
                        self.context.repository.delete_by_path(source_path) or 0
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("vector delete failed for %s: %s", source_path, exc)
            try:
                self.context.repository.reindex()
            except Exception as exc:  # noqa: BLE001
                logger.warning("reindex failed: %s", exc)
        removed_queue = False
        record_id = getattr(record, "id", None)
        if self.store is not None and record_id is not None:
            removed_queue = self.store.delete(record_id) > 0
        return {"vectors": removed_vectors, "queue": removed_queue}

    def preview_chunks(self, record, limit: int = 3) -> list:
        """Indexed chunks for an ingestion-table row (PDF preview pane).

        Hash first (inbox files move through processing/processed folders, so
        the queue path often differs from the indexed source_path), then exact
        path. Empty on backend-down / not-yet-indexed — never raises, since
        this serves a click preview, not retrieval.
        """
        if self.context is None or self.context.repository is None:
            return []
        repository = self.context.repository
        try:
            sha = getattr(record, "sha256", None)
            if sha:
                chunks = repository.get_by_file_hash(sha)
                if chunks:
                    return list(chunks[:limit])
            path = getattr(record, "path", None) or str(record)
            if path:
                chunks = repository.get_file_chunks(path)
                if chunks:
                    return list(chunks[:limit])
        except Exception as exc:  # noqa: BLE001
            logger.warning("preview lookup failed for %s: %s", getattr(record, "path", record), exc)
        return []
