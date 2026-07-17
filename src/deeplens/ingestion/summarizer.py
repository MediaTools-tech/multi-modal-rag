"""Per-file summarization used to enable document-level (summary) retrieval.

At ingestion time we generate one concise summary per text-bearing file and
store it as a dedicated ``record_type=SUMMARY`` vector record. This lets the
retriever locate *the file* from a description of its content, instead of only
matching individual chunks.
"""

from __future__ import annotations

from deeplens.config import Settings
from deeplens.core.chat import ChatEngine
from deeplens.core.models import FileType, RecordType, VectorRecord

# File types whose extracted text can be meaningfully summarized.
TEXTUAL_TYPES = {
    FileType.DOCUMENT.value,
    FileType.SUBTITLE.value,
    FileType.AUDIO.value,
    FileType.VIDEO.value,
}


class DocumentSummarizer:
    """Builds a single whole-file summary record from a file's chunk records."""

    def __init__(self, chat_engine: ChatEngine | None, settings: Settings) -> None:
        self.chat_engine = chat_engine
        self.settings = settings

    @staticmethod
    def _representative_text(records: list[VectorRecord], max_chars: int) -> str:
        """Concatenate chunk contents (head-biased) up to ``max_chars``."""
        if not records:
            return ""
        ordered = sorted(records, key=lambda r: r.chunk_index)
        parts: list[str] = []
        total = 0
        # Bias toward the beginning of the document where titles/abstracts live.
        for rec in ordered:
            snippet = rec.content or ""
            if total + len(snippet) > max_chars:
                remaining = max_chars - total
                if remaining > 0:
                    parts.append(snippet[:remaining])
                break
            parts.append(snippet)
            total += len(snippet)
        return "\n\n".join(parts).strip()

    async def build_summary_record(self, records: list[VectorRecord]) -> VectorRecord | None:
        """Generate and return a summary ``VectorRecord`` for ``records``.

        Returns ``None`` when summaries are disabled, the chat engine is
        unavailable, the file has no textual content, or summarization fails.
        """
        if not self.settings.enable_document_summaries:
            return None
        if self.chat_engine is None:
            return None
        if not records:
            return None

        file_type = records[0].file_type
        if file_type not in TEXTUAL_TYPES:
            return None

        text = self._representative_text(records, self.settings.summary_max_chars)
        if not text:
            return None

        try:
            summary = await self.chat_engine.summarize_document(
                text, filename=records[0].filename
            )
        except Exception:
            return None

        if not summary:
            return None

        base = records[0]
        return VectorRecord(
            content=summary,
            record_type=RecordType.SUMMARY.value,
            summary=summary,
            absolute_path=base.absolute_path,
            filename=base.filename,
            parent_directory=base.parent_directory,
            file_type=base.file_type,
            mime_type=base.mime_type,
            chunk_index=0,
            total_chunks=1,
            file_modified_at=base.file_modified_at,
            file_hash=base.file_hash,
        )
