from __future__ import annotations

import logging
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from .queue import IngestQueue
from .state import FileRecord, sha256_file

logger = logging.getLogger(__name__)

Handler = Callable[[Path, FileRecord], "str | None"]


class IngestionWorker:
    def __init__(
        self,
        queue: IngestQueue,
        handlers: dict[str, Handler],
        processing_dir: str | Path,
        processed_dir: str | Path,
        failed_dir: str | Path,
        worker_count: int = 2,
        file_stable_seconds: float = 15.0,
        idle_poll_seconds: float = 1.0,
        inbox_dir: str | Path | None = None,
        allow_kind: Callable[[str], bool] | None = None,
        reindex: Callable[[], None] | None = None,
    ) -> None:
        self.queue = queue
        self.handlers = handlers
        self.processing_dir = Path(processing_dir)
        self.processed_dir = Path(processed_dir)
        self.failed_dir = Path(failed_dir)
        # Only files under the inbox are "managed" (moved through processing/processed).
        # Files indexed in place (e.g. `mrag-ingest add <dir>` or the GUI) are never moved.
        self.inbox_dir = Path(inbox_dir) if inbox_dir is not None else None
        self.worker_count = max(1, worker_count)
        self.file_stable_seconds = file_stable_seconds
        self.idle_poll_seconds = idle_poll_seconds
        # Content policy safety net (queue already filters; this catches rows
        # queued before a mode switch). None = allow all.
        self.allow_kind = allow_kind
        # Deferred index refresh (LanceDB needs a rebuild after inserts; doing it
        # per file is O(n^2)). Called when the queue is idle and on stop; the
        # provider itself no-ops when nothing changed.
        self.reindex = reindex
        self._stop = threading.Event()
        self._pool: ThreadPoolExecutor | None = None

    def _handler_for(self, kind: str | None) -> Handler | None:
        if kind is not None and kind in self.handlers:
            return self.handlers[kind]
        return self.handlers.get("*")

    def _wait_until_stable(self, path: Path, timeout: float) -> bool:
        deadline = time.time() + timeout
        last: tuple[int, float] | None = None
        stable_since = time.time()
        while time.time() < deadline and not self._stop.is_set():
            try:
                stat = path.stat()
            except FileNotFoundError:
                return False
            signature = (stat.st_size, stat.st_mtime)
            if signature != last:
                last = signature
                stable_since = time.time()
            elif time.time() - stable_since >= self.file_stable_seconds:
                return True
            time.sleep(1.0)
        return False

    @staticmethod
    def _unique_destination(directory: Path, name: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        candidate = directory / name
        if not candidate.exists():
            return candidate
        stem, suffix = candidate.stem, candidate.suffix
        for index in range(1, 10_000):
            candidate = directory / f"{stem}.{index}{suffix}"
            if not candidate.exists():
                return candidate
        raise RuntimeError(f"Could not find a free name in {directory}")

    def _move(self, source: Path, directory: Path) -> Path:
        destination = self._unique_destination(directory, source.name)
        try:
            source.replace(destination)
        except OSError:
            shutil.move(str(source), str(destination))
        return destination

    def _stability_timeout(self, source: Path) -> float:
        base = self.file_stable_seconds * 6 + 30
        try:
            size_mb = source.stat().st_size / (1024 * 1024)
        except OSError:
            size_mb = 0.0
        return base + size_mb * 2  # allow slower copies for large files

    def _is_managed(self, source: Path) -> bool:
        if self.inbox_dir is None:
            return False
        try:
            source.resolve().relative_to(self.inbox_dir.resolve())
            return True
        except ValueError:
            return False

    def _fail_or_retry(
        self, record: FileRecord, message: str, source: Path, managed: bool
    ) -> None:
        if record.attempts < self.queue.max_retries:
            backoff = min(2 ** record.attempts, 30)
            logger.warning(
                "Retry %s (attempt %d/%d): %s",
                record.path,
                record.attempts,
                self.queue.max_retries,
                message,
            )
            self.queue.requeue(record)
            self._stop.wait(backoff)
            return
        if managed and source.exists():
            final = self._move(source, self.failed_dir)
            self.queue.state.update_path(record.id, final)
        self.queue.fail(record, message)

    def _process(self, record: FileRecord) -> None:
        source = Path(record.path)
        if not source.exists():
            self.queue.fail(record, f"source disappeared: {source}")
            return

        if not self._wait_until_stable(source, timeout=self._stability_timeout(source)):
            self._fail_or_retry(
                record,
                "file did not stabilize (still being written?)",
                source,
                self._is_managed(source),
            )
            return

        try:
            digest = sha256_file(source)
            stat = source.stat()
            self.queue.state.update_meta(record.id, digest, stat.st_size, stat.st_mtime)
            # Keep the in-memory record in sync: downstream indexing
            # (_meta -> file_hash) reads it, and update_meta only touches the DB.
            record.sha256 = digest
            record.size = stat.st_size
            record.mtime = stat.st_mtime
            managed = self._is_managed(source)

            if self.queue.is_duplicate(digest, exclude_id=record.id):
                if managed:
                    final = self._move(source, self.processed_dir)
                    self.queue.state.update_path(record.id, final)
                self.queue.skip(record, f"duplicate content sha256={digest}")
                return

            if record.kind is None or self._handler_for(record.kind) is None:
                if managed:
                    final = self._move(source, self.processed_dir)
                    self.queue.state.update_path(record.id, final)
                self.queue.skip(record, f"unsupported file kind={record.kind!r}")
                return

            if (
                record.kind is not None
                and self.allow_kind is not None
                and not self.allow_kind(record.kind)
            ):
                if managed:
                    final = self._move(source, self.processed_dir)
                    self.queue.state.update_path(record.id, final)
                self.queue.skip(
                    record, f"not allowed by SYSTEM_MODE (kind={record.kind})"
                )
                return

            staged = source
            if managed:
                staged = self._move(source, self.processing_dir)
                self.queue.state.update_path(record.id, staged)
            try:
                output = self._handler_for(record.kind)(staged, record)
            except Exception as exc:
                logger.exception("Handler failed for %s", staged)
                self._fail_or_retry(record, f"{type(exc).__name__}: {exc}", staged, managed)
                return

            if managed:
                final = self._move(staged, self.processed_dir)
                self.queue.state.update_path(record.id, final)
            self.queue.complete(record, output=output)
        except Exception as exc:
            logger.exception("Unexpected error processing record %s", record.id)
            self.queue.fail(record, f"unexpected: {type(exc).__name__}: {exc}")

    def _flush_reindex(self) -> None:
        """Rebuild the backend index after a batch of inserts, if needed."""
        if self.reindex is None:
            return
        try:
            self.reindex()
        except Exception:  # noqa: BLE001
            logger.exception("Deferred reindex failed")

    def _loop(self) -> None:
        while not self._stop.is_set():
            record = self.queue.claim()
            if record is None:
                # Queue drained: the safe moment to batch-rebuild indexes.
                self._flush_reindex()
                self._stop.wait(self.idle_poll_seconds)
                continue
            logger.info("Processing id=%s path=%s", record.id, record.path)
            self._process(record)

    def start(self) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=self.worker_count, thread_name_prefix="ingest-worker"
        )
        for _ in range(self.worker_count):
            self._pool.submit(self._loop)
        logger.info("Started %d ingestion worker(s)", self.worker_count)

    def stop(self) -> None:
        self._stop.set()
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        # Flush any pending inserts so the index is searchable after shutdown.
        self._flush_reindex()

    def restart(self) -> None:
        """Stop, then start again (e.g. after a full index reset)."""
        self.stop()
        self._stop.clear()
        self.start()
