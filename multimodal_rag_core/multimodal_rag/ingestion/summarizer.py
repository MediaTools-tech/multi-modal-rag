from __future__ import annotations

import logging
from typing import Callable

from multimodal_rag.core.models import RecordType, VectorRecord

logger = logging.getLogger(__name__)

SummaryFn = Callable[[str, str], str]

TEXTUAL_TYPES = {"document", "subtitle", "audio", "video"}


class DocumentSummarizer:
    """Builds one file-level summary record to enable document-level retrieval."""

    def __init__(self, summarize_fn: SummaryFn | None, max_chars: int = 4000) -> None:
        self.summarize_fn = summarize_fn
        self.max_chars = max_chars

    @staticmethod
    def _representative_text(records: list[VectorRecord], max_chars: int) -> str:
        if not records:
            return ""
        ordered = sorted(records, key=lambda record: record.chunk_index)
        parts: list[str] = []
        total = 0
        for record in ordered:
            snippet = record.content or ""
            if total + len(snippet) > max_chars:
                remaining = max_chars - total
                if remaining > 0:
                    parts.append(snippet[:remaining])
                break
            parts.append(snippet)
            total += len(snippet)
        return "\n\n".join(parts).strip()

    def build_summary_record(self, records: list[VectorRecord]) -> VectorRecord | None:
        if self.summarize_fn is None or not records:
            return None
        base = records[0]
        if base.file_type not in TEXTUAL_TYPES:
            return None
        text = self._representative_text(records, self.max_chars)
        if not text:
            return None
        try:
            summary = self.summarize_fn(text, base.filename)
        except Exception as exc:
            logger.warning("Summary generation failed for %s: %s", base.filename, exc)
            return None
        if not summary:
            return None
        return VectorRecord(
            content=summary,
            record_type=RecordType.SUMMARY.value,
            summary=summary,
            source_path=base.source_path,
            filename=base.filename,
            parent_directory=base.parent_directory,
            file_type=base.file_type,
            mime_type=base.mime_type,
            chunk_index=0,
            total_chunks=1,
            file_modified_at=base.file_modified_at,
            file_hash=base.file_hash,
            metadata={"generated": True},
        )
