from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    content: str
    index: int
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    metadata: dict = field(default_factory=dict)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = (text or "").split()
    if not words:
        return []
    if chunk_size <= 0:
        return [" ".join(words)]
    overlap = max(0, min(overlap, chunk_size - 1))
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
        if start <= 0:
            start = end
    return chunks


def text_to_chunks(text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    return [
        Chunk(content=content, index=index)
        for index, content in enumerate(chunk_text(text, chunk_size, overlap))
    ]
