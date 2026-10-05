from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

DISCOVERED = "discovered"
PROCESSING = "processing"
PROCESSED = "processed"
FAILED = "failed"
SKIPPED = "skipped"

TERMINAL_STATUSES = {PROCESSED, SKIPPED}

DOCUMENT_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
    ".md", ".txt", ".html", ".htm", ".csv", ".rtf",
}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".gif"}

EXTENSION_KINDS: dict[str, set[str]] = {
    "document": DOCUMENT_EXTENSIONS,
    "audio": AUDIO_EXTENSIONS,
    "video": VIDEO_EXTENSIONS,
    "image": IMAGE_EXTENSIONS,
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT    NOT NULL UNIQUE,
    sha256      TEXT,
    size        INTEGER,
    mtime       REAL,
    kind        TEXT,
    status      TEXT    NOT NULL,
    attempts    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    output      TEXT,
    created_at  REAL    NOT NULL,
    updated_at  REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);
CREATE INDEX IF NOT EXISTS idx_files_sha ON files(sha256);
CREATE TABLE IF NOT EXISTS meta (
    key         TEXT    PRIMARY KEY,
    value       TEXT    NOT NULL,
    updated_at  REAL    NOT NULL
);
"""

#: Index-time settings baked into every stored vector. Changing any of these
#: orphans the index (see BackendService.check_fingerprint). SYSTEM_MODE is
#: deliberately NOT here: it is a content policy / default-engine preset, not a
#: property of the vectors, so switching doc-only <-> multimodal never needs a
#: re-index (a resulting engine change is caught via ACTIVE_DB_ENGINE).
INDEX_FINGERPRINT_KEYS = (
    "ACTIVE_DB_ENGINE",
    "POSTGRES_DSN",
    "LANCEDB_DIR",
    "DB_TABLE_NAME",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "DOC_CHUNK_SIZE",
    "DOC_CHUNK_OVERLAP",
)


def fingerprint_of(settings) -> dict:
    """Current index-time config as plain strings (works without a backend).

    The database is fingerprinted as the password-masked effective DSN, so
    switching database name, server, user, or URL override is detected while
    no secret is ever stored. SYSTEM_MODE, LLM/VLM keys, models, and rates are
    deliberately excluded: they are policy/query-time and never invalidate
    stored vectors.
    """
    fingerprint = {}
    for key in INDEX_FINGERPRINT_KEYS:
        if key == "POSTGRES_DSN":
            fingerprint[key] = settings.postgres_dsn_sanitized
        else:
            value = getattr(settings, key)
            fingerprint[key] = value.value if hasattr(value, "value") else str(value)
    return fingerprint


def route_kind(path: Path) -> str | None:
    ext = path.suffix.lower()
    for kind, extensions in EXTENSION_KINDS.items():
        if ext in extensions:
            return kind
    return None


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class FileRecord:
    id: int
    path: str
    sha256: str | None
    size: int | None
    mtime: float | None
    kind: str | None
    status: str
    attempts: int
    error: str | None
    output: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "FileRecord":
        return cls(
            id=row["id"],
            path=row["path"],
            sha256=row["sha256"],
            size=row["size"],
            mtime=row["mtime"],
            kind=row["kind"],
            status=row["status"],
            attempts=row["attempts"],
            error=row["error"],
            output=row["output"],
        )


class StateStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.executescript(SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @staticmethod
    def _stat_signature(path: str | Path) -> tuple[int, float] | None:
        try:
            stat = Path(path).stat()
            return (stat.st_size, stat.st_mtime)
        except OSError:
            return None

    def enqueue(self, path: str | Path, kind: str | None = None, force: bool = False) -> FileRecord:
        path = str(path)
        now = time.time()
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM files WHERE path=?", (path,)).fetchone()
            if row is not None and not force:
                signature = self._stat_signature(path)
                changed = (
                    signature is not None
                    and row["status"] not in (DISCOVERED, PROCESSING)
                    and (row["size"] != signature[0] or row["mtime"] != signature[1])
                )
                if not changed:
                    return FileRecord.from_row(row)
                kind = kind or route_kind(Path(path))
                conn.execute(
                    "UPDATE files SET kind=?, status=?, attempts=0, error=NULL, updated_at=? "
                    "WHERE id=?",
                    (kind, DISCOVERED, now, row["id"]),
                )
                conn.commit()
                return self.get(int(row["id"]))
            kind = kind or route_kind(Path(path))
            if row is None:
                cur = conn.execute(
                    "INSERT INTO files (path, kind, status, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (path, kind, DISCOVERED, now, now),
                )
                file_id = int(cur.lastrowid)
            else:
                file_id = int(row["id"])
                conn.execute(
                    "UPDATE files SET kind=?, status=?, error=NULL, attempts=0, updated_at=? WHERE id=?",
                    (kind, DISCOVERED, now, file_id),
                )
            conn.commit()
            return self.get(file_id)
        finally:
            conn.close()

    def get(self, file_id: int) -> FileRecord:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM files WHERE id=?", (file_id,)).fetchone()
            if row is None:
                raise KeyError(f"No ingest record with id={file_id}")
            return FileRecord.from_row(row)
        finally:
            conn.close()

    def is_hash_ingested(self, sha256: str, exclude_id: int | None = None) -> bool:
        conn = self._connect()
        try:
            if exclude_id is None:
                row = conn.execute(
                    "SELECT 1 FROM files WHERE sha256=? AND status IN (?, ?) LIMIT 1",
                    (sha256, PROCESSED, SKIPPED),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT 1 FROM files WHERE sha256=? AND status IN (?, ?) AND id<>? LIMIT 1",
                    (sha256, PROCESSED, SKIPPED, exclude_id),
                ).fetchone()
            return row is not None
        finally:
            conn.close()

    def update_path(self, file_id: int, new_path: str | Path) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE files SET path=?, updated_at=? WHERE id=?",
                (str(new_path), time.time(), file_id),
            )
            conn.commit()
        finally:
            conn.close()

    def claim_next(self, max_attempts: int) -> FileRecord | None:
        conn = self._connect()
        conn.isolation_level = None
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM files WHERE status=? AND attempts < ? ORDER BY id LIMIT 1",
                (DISCOVERED, max_attempts),
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            now = time.time()
            conn.execute(
                "UPDATE files SET status=?, attempts=attempts+1, updated_at=? WHERE id=?",
                (PROCESSING, now, row["id"]),
            )
            conn.execute("COMMIT")
            return self.get(int(row["id"]))
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def update_meta(self, file_id: int, sha256: str, size: int, mtime: float) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE files SET sha256=?, size=?, mtime=?, updated_at=? WHERE id=?",
                (sha256, size, mtime, time.time(), file_id),
            )
            conn.commit()
        finally:
            conn.close()

    def set_status(
        self,
        file_id: int,
        status: str,
        error: str | None = None,
        output: str | None = None,
    ) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE files SET status=?, error=?, output=?, updated_at=? WHERE id=?",
                (status, error, output, time.time(), file_id),
            )
            conn.commit()
        finally:
            conn.close()

    def requeue(self, file_id: int) -> None:
        """Return a row to the queue without clearing its attempt counter."""
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE files SET status=?, error=NULL, updated_at=? WHERE id=?",
                (DISCOVERED, time.time(), file_id),
            )
            conn.commit()
        finally:
            conn.close()

    def reset_stale(self) -> int:
        conn = self._connect()
        try:
            cur = conn.execute(
                "UPDATE files SET status=?, updated_at=? WHERE status=?",
                (DISCOVERED, time.time(), PROCESSING),
            )
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()

    def retry(self, file_id: int | None = None) -> int:
        conn = self._connect()
        try:
            now = time.time()
            if file_id is None:
                cur = conn.execute(
                    "UPDATE files SET status=?, attempts=0, error=NULL, updated_at=? WHERE status=?",
                    (DISCOVERED, now, FAILED),
                )
            else:
                cur = conn.execute(
                    "UPDATE files SET status=?, attempts=0, error=NULL, updated_at=? WHERE id=?",
                    (DISCOVERED, now, file_id),
                )
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()

    def delete(self, file_id: int) -> int:
        """Permanently drop a queue row (used by Remove; vectors are deleted separately)."""
        conn = self._connect()
        try:
            cur = conn.execute("DELETE FROM files WHERE id=?", (file_id,))
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()

    def remove_by_path(self, path: str | Path) -> int:
        """Drop the queue row for ``path`` unless a worker is actively processing it.

        Used when a source file is deleted: a row already in ``processing`` is
        owned by a worker (which moves inbox files during normal ingestion), so
        deleting it here would race the worker. Vectors are deleted separately.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT status FROM files WHERE path=?", (str(path),)
            ).fetchone()
            if row is None or row["status"] == PROCESSING:
                return 0
            cur = conn.execute(
                "DELETE FROM files WHERE path=? AND status<>?",
                (str(path), PROCESSING),
            )
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()

    def clear_queue(self) -> int:
        """Drop ALL queue rows (full reset; vectors are cleared separately)."""
        conn = self._connect()
        try:
            cur = conn.execute("DELETE FROM files")
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()

    def get_meta(self, key: str) -> str | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return row["value"] if row is not None else None
        finally:
            conn.close()

    def set_meta(self, key: str, value: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO meta (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, value, time.time()),
            )
            conn.commit()
        finally:
            conn.close()

    def list_by_status(self, status: str | None = None) -> list[FileRecord]:
        conn = self._connect()
        try:
            if status is None:
                rows = conn.execute("SELECT * FROM files ORDER BY id").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM files WHERE status=? ORDER BY id", (status,)
                ).fetchall()
            return [FileRecord.from_row(row) for row in rows]
        finally:
            conn.close()

    def stats(self) -> dict[str, int]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM files GROUP BY status"
            ).fetchall()
            return {row["status"]: row["n"] for row in rows}
        finally:
            conn.close()
