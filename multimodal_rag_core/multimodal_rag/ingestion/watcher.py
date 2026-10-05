from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except Exception:
    FileSystemEventHandler = object  # type: ignore[assignment,misc]
    Observer = None  # type: ignore[assignment]
    WATCHDOG_AVAILABLE = False


class InboxWatcher:
    def __init__(
        self,
        inbox_dir: str | Path,
        on_file: Callable[[Path], None],
        ignore_extensions: set[str] | None = None,
        poll_interval: float = 5.0,
        on_delete: Callable[[Path], None] | None = None,
    ) -> None:
        self.inbox_dir = Path(inbox_dir)
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.on_file = on_file
        self.ignore_extensions = ignore_extensions or set()
        self.poll_interval = poll_interval
        # Called when a tracked file leaves the inbox (delete or move-out) so
        # callers can drop its queue row and orphaned vectors.
        self.on_delete = on_delete
        self._seen: dict[str, tuple[int, float]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._observer = None

    def _is_candidate(self, path: Path) -> bool:
        if path.name.startswith("."):
            return False
        if path.suffix.lower() in self.ignore_extensions:
            return False
        return True

    def _is_inside(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.inbox_dir.resolve())
            return True
        except ValueError:
            return False

    def _submit(self, raw_path: str) -> None:
        path = Path(raw_path)
        if not path.is_file() or not self._is_candidate(path):
            return
        # Events can name paths outside the inbox (e.g. the worker moving a file
        # into processing/); only the inbox is our responsibility.
        if not self._is_inside(path):
            return
        try:
            self.on_file(path)
        except Exception:
            logger.exception("Failed to enqueue %s", path)

    def _delete(self, raw_path: str) -> None:
        path = Path(raw_path)
        if self.on_delete is None or not self._is_inside(path):
            return
        try:
            self.on_delete(path)
        except Exception:
            logger.exception("Failed to sync deletion of %s", path)

    def scan_once(self, force: bool = False) -> int:
        count = 0
        for entry in sorted(self.inbox_dir.rglob("*")):
            if entry.is_file() and self._is_candidate(entry):
                self._submit(entry)
                count += 1
        return count

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                for entry in self.inbox_dir.rglob("*"):
                    if not entry.is_file() or not self._is_candidate(entry):
                        continue
                    stat = entry.stat()
                    signature = (stat.st_size, stat.st_mtime)
                    if self._seen.get(str(entry)) != signature:
                        self._seen[str(entry)] = signature
                        self._submit(entry)
            except Exception:
                logger.exception("Polling scan failed")
            self._stop.wait(self.poll_interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._poll_loop, name="inbox-poll", daemon=True)
        self._thread.start()
        if WATCHDOG_AVAILABLE:
            handler = _CallbackHandler(self._submit, self._delete)
            self._observer = Observer()
            self._observer.schedule(handler, str(self.inbox_dir), recursive=True)
            self._observer.start()
            logger.info("Watching %s via watchdog + polling", self.inbox_dir)
        else:
            logger.warning(
                "watchdog not installed; using polling only for %s", self.inbox_dir
            )

    def stop(self) -> None:
        self._stop.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=5)


class _CallbackHandler(FileSystemEventHandler):  # type: ignore[misc]
    def __init__(
        self,
        on_file: Callable[[str], None],
        on_delete: Callable[[str], None],
    ) -> None:
        super().__init__()
        self._on_file = on_file
        self._on_delete = on_delete

    def on_created(self, event) -> None:
        if not event.is_directory:
            self._on_file(event.src_path)

    def on_moved(self, event) -> None:
        if event.is_directory:
            return
        # A rename keeps the old row stale; the new path (if still inside the
        # inbox) is a fresh arrival. Both callbacks are inbox-guarded.
        self._on_file(event.dest_path)
        self._on_delete(event.src_path)

    def on_deleted(self, event) -> None:
        if not event.is_directory:
            self._on_delete(event.src_path)
