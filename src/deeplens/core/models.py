"""Core data models and dataclasses used throughout DeepLens."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class FileType(str, Enum):
    """Broad file type categories for routing and filtering."""

    DOCUMENT = "document"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    SUBTITLE = "subtitle"
    ARCHIVE = "archive"
    UNKNOWN = "unknown"


# ── Extension → FileType mapping ─────────────────────────────────────────
EXTENSION_MAP: dict[str, FileType] = {
    # Documents
    ".pdf": FileType.DOCUMENT,
    ".docx": FileType.DOCUMENT,
    ".doc": FileType.DOCUMENT,
    ".pptx": FileType.DOCUMENT,
    ".ppt": FileType.DOCUMENT,
    ".xlsx": FileType.DOCUMENT,
    ".xls": FileType.DOCUMENT,
    ".html": FileType.DOCUMENT,
    ".htm": FileType.DOCUMENT,
    ".md": FileType.DOCUMENT,
    ".txt": FileType.DOCUMENT,
    ".csv": FileType.DOCUMENT,
    ".json": FileType.DOCUMENT,
    ".xml": FileType.DOCUMENT,
    ".epub": FileType.DOCUMENT,
    ".rtf": FileType.DOCUMENT,
    ".rst": FileType.DOCUMENT,
    # Images
    ".jpg": FileType.IMAGE,
    ".jpeg": FileType.IMAGE,
    ".png": FileType.IMAGE,
    ".webp": FileType.IMAGE,
    ".bmp": FileType.IMAGE,
    ".gif": FileType.IMAGE,
    ".tiff": FileType.IMAGE,
    ".tif": FileType.IMAGE,
    ".svg": FileType.IMAGE,
    ".ico": FileType.IMAGE,
    # Audio
    ".mp3": FileType.AUDIO,
    ".wav": FileType.AUDIO,
    ".flac": FileType.AUDIO,
    ".ogg": FileType.AUDIO,
    ".m4a": FileType.AUDIO,
    ".aac": FileType.AUDIO,
    ".wma": FileType.AUDIO,
    # Video
    ".mp4": FileType.VIDEO,
    ".mkv": FileType.VIDEO,
    ".avi": FileType.VIDEO,
    ".mov": FileType.VIDEO,
    ".wmv": FileType.VIDEO,
    ".flv": FileType.VIDEO,
    ".webm": FileType.VIDEO,
    ".m4v": FileType.VIDEO,
    # Subtitles
    ".srt": FileType.SUBTITLE,
    ".vtt": FileType.SUBTITLE,
    ".ass": FileType.SUBTITLE,
    ".ssa": FileType.SUBTITLE,
    # Archives
    ".zip": FileType.ARCHIVE,
    ".rar": FileType.ARCHIVE,
    ".7z": FileType.ARCHIVE,
    ".tar": FileType.ARCHIVE,
    ".gz": FileType.ARCHIVE,
    ".bz2": FileType.ARCHIVE,
    ".xz": FileType.ARCHIVE,
}


def classify_file(path: Path) -> FileType:
    """Classify a file by its extension.

    Args:
        path: Path to the file.

    Returns:
        The FileType category.
    """
    suffix = path.suffix.lower()
    # Handle compound extensions like .tar.gz
    if suffix == ".gz" and path.stem.endswith(".tar"):
        return FileType.ARCHIVE
    return EXTENSION_MAP.get(suffix, FileType.UNKNOWN)


# Record categories stored in the vector database.
class RecordType(str, Enum):
    """Discriminator for what a VectorRecord represents."""

    CHUNK = "chunk"      # A passage / frame / transcript segment (default).
    SUMMARY = "summary"  # A concise whole-file summary used for file-level retrieval.


@dataclass
class VectorRecord:
    """A single vector record stored in the database.

    Each record represents one chunk of a source file — a text passage,
    a single image, a video frame, or an audio transcript segment — or, when
    ``record_type`` is ``SUMMARY``, a concise whole-file summary used for
    document-level ("find this file by its description") retrieval.
    """

    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Vector embedding
    vector: list[float] = field(default_factory=list)

    # Content
    content: str = ""  # Raw text chunk, image caption, transcript segment, or summary

    # Record discriminator
    record_type: str = RecordType.CHUNK.value

    # Source file metadata
    absolute_path: str = ""
    filename: str = ""
    parent_directory: str = ""
    file_type: str = ""  # FileType enum value
    mime_type: str = ""

    # Chunk position
    chunk_index: int = 0
    total_chunks: int = 1

    # Temporal metadata (for audio/video)
    timestamp_start: float | None = None
    timestamp_end: float | None = None

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    file_modified_at: str = ""

    # Integrity
    file_hash: str = ""  # SHA-256 of source file

    # Extensible metadata
    metadata_json: str = "{}"  # JSON blob for EXIF, etc.

    # Document summary (populated when record_type == SUMMARY; mirrors content
    # for convenience so it can be surfaced directly in results).
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a flat dictionary for database insertion."""
        return {
            "id": self.id,
            "vector": self.vector,
            "content": self.content,
            "record_type": self.record_type,
            "absolute_path": self.absolute_path,
            "filename": self.filename,
            "parent_directory": self.parent_directory,
            "file_type": self.file_type,
            "mime_type": self.mime_type,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "created_at": self.created_at,
            "file_modified_at": self.file_modified_at,
            "file_hash": self.file_hash,
            "metadata_json": self.metadata_json,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VectorRecord:
        """Create a VectorRecord from a dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SearchResult:
    """A single search result returned from the vector database."""

    record: VectorRecord
    score: float  # Cosine similarity score (0.0 - 1.0)
    rank: int = 0  # Position in result set


@dataclass
class FileSearchGroup:
    """A file-level grouping of hybrid search results.

    When retrieval is summary- or file-aware, results are clustered by source
    file so the user can be pointed at *the document* (with its summary) rather
    than a scattered set of chunks. ``chunk_results`` holds the supporting
    passage-level hits that ground the match.
    """

    absolute_path: str = ""
    filename: str = ""
    file_type: str = ""
    summary: str = ""
    best_score: float = 0.0
    chunk_results: list[SearchResult] = field(default_factory=list)


@dataclass
class SearchResponse:
    """Complete search response including results and metadata."""

    query: str
    rewritten_query: str
    results: list[SearchResult] = field(default_factory=list)
    answer: str = ""  # LLM-generated answer
    retry_count: int = 0
    total_time_ms: float = 0.0
    # File-level grouping produced by summary / hybrid retrieval.
    file_groups: list[FileSearchGroup] = field(default_factory=list)


@dataclass
class IngestionJob:
    """Represents a file queued for ingestion into the vector database."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_path: Path = field(default_factory=Path)
    file_type: FileType = FileType.UNKNOWN
    status: str = "pending"  # pending, processing, completed, failed
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: str | None = None
    records_created: int = 0


@dataclass
class IndexingProgress:
    """Real-time progress of an indexing operation."""

    folder_path: str = ""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    current_file: str = ""
    status: str = "idle"  # idle, scanning, indexing, completed, error
    eta_seconds: float | None = None

    @property
    def progress_pct(self) -> float:
        """Calculate progress percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100.0
