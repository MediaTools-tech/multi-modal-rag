"""Directory watcher using watchdog.

Listens for local filesystem changes (creation, modification, deletion) and triggers
callbacks to update the vector database incrementally.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import time
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
import structlog

logger = structlog.get_logger(__name__)


class DebouncedEventHandler(FileSystemEventHandler):
    """Handles filesystem events and groups/debounces them to avoid rapid indexing loops."""

    def __init__(self, callback: Callable[[str, Path], Any], debounce_seconds: float = 0.5) -> None:
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self.loop = asyncio.get_event_loop()
        self._pending_tasks: dict[Path, asyncio.TimerHandle] = {}

    def on_any_event(self, event: FileSystemEvent) -> None:
        """Filter events and trigger callbacks."""
        if event.is_directory:
            return

        event_type = event.event_type
        src_path = Path(event.src_path)
        
        # Resolve path
        try:
            resolved_path = src_path.resolve()
        except Exception:
            resolved_path = src_path

        # Ignore hidden files
        if any(p.startswith(".") for p in resolved_path.parts):
            return

        # Skip folders / directories
        if event_type in ("created", "modified"):
            if resolved_path.exists() and resolved_path.is_dir():
                return
        
        # Debounce logic using asyncio timer
        self.loop.call_soon_threadsafe(self._schedule_callback, event_type, resolved_path)

    def _schedule_callback(self, event_type: str, path: Path) -> None:
        # Cancel previous scheduled task for this path if it exists
        if path in self._pending_tasks:
            self._pending_tasks[path].cancel()

        # Define the actual task to execute
        def _execute() -> None:
            self._pending_tasks.pop(path, None)
            # Run async callback inside loop (or run threadsafe callback)
            if asyncio.iscoroutinefunction(self.callback):
                asyncio.create_task(self.callback(event_type, path))
            else:
                self.callback(event_type, path)

        # Schedule
        handle = self.loop.call_later(self.debounce_seconds, _execute)
        self._pending_tasks[path] = handle


class FileWatcher:
    """Monitors filesystem changes across registered directories."""

    def __init__(self, callback: Callable[[str, Path], Any], debounce_seconds: float = 0.5) -> None:
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self._observer: Observer | None = None
        self._event_handler = DebouncedEventHandler(callback, debounce_seconds)
        self._watched_paths: set[Path] = set()

    def start(self, directories: list[Path]) -> None:
        """Start monitoring directories."""
        if self._observer:
            self.stop()

        self._observer = Observer()
        self._watched_paths = set()

        for dir_path in directories:
            dir_path = Path(dir_path).resolve()
            if dir_path.exists() and dir_path.is_dir():
                self._observer.schedule(self._event_handler, str(dir_path), recursive=True)
                self._watched_paths.add(dir_path)
                logger.info("file_watcher.monitor.directory", path=str(dir_path))
            else:
                logger.warn("file_watcher.monitor.skip_invalid", path=str(dir_path))

        if self._watched_paths:
            self._observer.start()
            logger.info("file_watcher.started", count=len(self._watched_paths))
        else:
            logger.info("file_watcher.no_paths_to_watch")

    def stop(self) -> None:
        """Stop observer thread."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            self._watched_paths.clear()
            logger.info("file_watcher.stopped")
