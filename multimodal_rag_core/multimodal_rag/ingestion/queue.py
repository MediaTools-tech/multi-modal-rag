from __future__ import annotations

from pathlib import Path
from typing import Callable

from .state import FileRecord, StateStore, route_kind


class IngestQueue:
    def __init__(
        self,
        state: StateStore,
        max_retries: int,
        allow_kind: Callable[[str], bool] | None = None,
    ) -> None:
        self.state = state
        self.max_retries = max_retries
        # Content policy (e.g. doc-only mode rejects audio/video). None = allow all.
        self.allow_kind = allow_kind

    def _allowed(self, kind: str) -> bool:
        return self.allow_kind is None or self.allow_kind(kind)

    def enqueue_path(self, path: str | Path, force: bool = False) -> FileRecord | None:
        kind = route_kind(Path(path))
        if kind is not None and not self._allowed(kind):
            return None
        return self.state.enqueue(path, kind=kind, force=force)

    def enqueue_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        force: bool = False,
        ignore_extensions: set[str] | None = None,
    ) -> int:
        directory = Path(directory)
        if not directory.is_dir():
            return 0
        ignore = ignore_extensions or set()
        pattern = "**/*" if recursive else "*"
        count = 0
        for entry in sorted(directory.glob(pattern)):
            if not entry.is_file():
                continue
            if entry.suffix.lower() in ignore:
                continue
            kind = route_kind(entry)
            if kind is None or not self._allowed(kind):
                continue
            self.enqueue_path(entry, force=force)
            count += 1
        return count

    def claim(self) -> FileRecord | None:
        return self.state.claim_next(self.max_retries)

    def requeue(self, record: FileRecord) -> None:
        self.state.requeue(record.id)

    def remove(self, record: FileRecord) -> bool:
        """Permanently drop a queue row (vectors are deleted separately)."""
        return self.state.delete(record.id) > 0

    def complete(self, record: FileRecord, output: str | None = None) -> None:
        self.state.set_status(record.id, "processed", output=output)

    def fail(self, record: FileRecord, error: str) -> None:
        self.state.set_status(record.id, "failed", error=error)

    def skip(self, record: FileRecord, reason: str) -> None:
        self.state.set_status(record.id, "skipped", output=reason)

    def is_duplicate(self, sha256: str, exclude_id: int | None = None) -> bool:
        return self.state.is_hash_ingested(sha256, exclude_id=exclude_id)
