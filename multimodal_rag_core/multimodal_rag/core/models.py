from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RecordType(str, Enum):
    CHUNK = "chunk"
    SUMMARY = "summary"


@dataclass
class VectorRecord:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    vector: list[float] = field(default_factory=list)
    content: str = ""
    record_type: str = RecordType.CHUNK.value
    source_path: str = ""
    filename: str = ""
    parent_directory: str = ""
    file_type: str = ""
    mime_type: str = ""
    chunk_index: int = 0
    total_chunks: int = 1
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    created_at: str = field(default_factory=utcnow_iso)
    file_modified_at: str = ""
    file_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "vector": self.vector,
            "content": self.content,
            "record_type": self.record_type,
            "source_path": self.source_path,
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
            "metadata": self.metadata,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VectorRecord":
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass
class SearchResult:
    record: VectorRecord
    score: float
    rank: int = 0


@dataclass
class FileSearchGroup:
    source_path: str = ""
    filename: str = ""
    file_type: str = ""
    summary: str = ""
    best_score: float = 0.0
    chunk_results: list[SearchResult] = field(default_factory=list)
