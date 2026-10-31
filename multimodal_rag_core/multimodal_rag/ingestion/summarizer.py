from __future__ import annotations

import logging
import time
from collections.abc import Callable

from multimodal_rag.config import Settings, SummaryProvider
from multimodal_rag.core.models import RecordType, VectorRecord
from multimodal_rag.pipeline.local_summarizer import (
    LocalExtractiveSummarizer,
    LocalSummary,
    clean_summary_text,
    trim_summary_text,
)

logger = logging.getLogger(__name__)

SummaryFn = Callable[[str, str], str]

TEXTUAL_TYPES = {"document", "subtitle", "audio", "video"}
IMAGE_TYPE = "image"


class DocumentSummarizer:
    """Builds one file-level summary record to enable document-level retrieval."""

    def __init__(
        self,
        summarize_fn: SummaryFn | None = None,
        max_chars: int = 4000,
        *,
        settings: Settings | None = None,
        local_summarizer: LocalExtractiveSummarizer | None = None,
    ) -> None:
        self.summarize_fn = summarize_fn
        self.max_chars = max_chars
        self.settings = settings
        self.local_summarizer = local_summarizer
        if settings is not None:
            self.provider = settings.SUMMARY_PROVIDER
            self.include_images = settings.SUMMARY_INCLUDE_IMAGES
            self.target_words = settings.SUMMARY_TARGET_WORDS
            self.output_max_chars = settings.SUMMARY_OUTPUT_MAX_CHARS
            self.local_max_chars = settings.SUMMARY_LOCAL_MAX_CHARS
            self.fallback_to_llm = settings.SUMMARY_FALLBACK_TO_LLM
            self.max_chars = settings.SUMMARY_MAX_CHARS
            if self.provider in (SummaryProvider.LOCAL, SummaryProvider.AUTO):
                if self.local_summarizer is None:
                    self.local_summarizer = LocalExtractiveSummarizer(settings)
        else:
            self.provider = (
                SummaryProvider.LLM if summarize_fn is not None else SummaryProvider.LOCAL
            )
            self.include_images = True
            self.target_words = 120
            self.output_max_chars = 900
            self.local_max_chars = 30000
            self.fallback_to_llm = False

    def _supports(self, file_type: str) -> bool:
        return file_type in TEXTUAL_TYPES or (file_type == IMAGE_TYPE and self.include_images)

    @staticmethod
    def _ordered_records(records: list[VectorRecord]) -> list[VectorRecord]:
        def key(record: VectorRecord) -> tuple[int, float, int]:
            if record.timestamp_start is not None:
                return (0, float(record.timestamp_start), int(record.chunk_index))
            return (1, float(record.chunk_index), 0)

        return sorted(records, key=key)

    @staticmethod
    def _representative_text(records: list[VectorRecord], max_chars: int) -> str:
        if not records or max_chars <= 0:
            return ""
        ordered = DocumentSummarizer._ordered_records(records)
        total = sum(len(clean_summary_text(record.content or "")) for record in ordered)
        if total <= max_chars:
            joined = "\n\n".join(
                part
                for part in (clean_summary_text(record.content or "") for record in ordered)
                if part
            ).strip()
            return trim_summary_text(joined, 0, max_chars)
        tail_chars = max(1, max_chars // 4)
        head_chars = max_chars - tail_chars - 2
        head_parts: list[str] = []
        used = 0
        for record in ordered:
            part = clean_summary_text(record.content or "")
            if not part:
                continue
            remaining = head_chars - used
            if remaining <= 0:
                break
            if len(part) > remaining:
                part = part[:remaining].rstrip()
            if part:
                head_parts.append(part)
                used += len(part) + 2
        tail_parts: list[str] = []
        used = 0
        for record in reversed(ordered):
            part = clean_summary_text(record.content or "")
            if not part:
                continue
            remaining = tail_chars - used
            if remaining <= 0:
                break
            if len(part) > remaining:
                part = part[-remaining:].lstrip()
            if part:
                tail_parts.append(part)
                used += len(part) + 2
        return "\n\n".join(head_parts + list(reversed(tail_parts))).strip()

    def _generate_local(
        self, records: list[VectorRecord], filename: str
    ) -> tuple[str, LocalSummary | None]:
        if self.local_summarizer is None:
            return "", None
        text = self._representative_text(records, self.local_max_chars)
        result = self.local_summarizer.summarize(text, filename, records)
        return result.text, result

    def _generate_llm(self, records: list[VectorRecord], filename: str) -> tuple[str, int]:
        if self.summarize_fn is None:
            return "", 0
        text = self._representative_text(records, self.max_chars)
        if not text:
            return "", 0
        summary = self.summarize_fn(text, filename)
        summary = trim_summary_text(
            clean_summary_text(summary),
            self.target_words,
            self.output_max_chars,
        )
        return summary, len(text)

    def build_summary_record(self, records: list[VectorRecord]) -> VectorRecord | None:
        if not records:
            return None
        base = records[0]
        if not self._supports(base.file_type):
            return None
        if not any((record.content or "").strip() for record in records):
            return None

        started = time.perf_counter()
        provider_used = ""
        algorithm = ""
        semantic_unit = "NONE"
        input_chars = 0
        chunks_considered = len(records)
        chunks_selected = 0
        used_fallback = False
        summary = ""
        local_result: LocalSummary | None = None

        if self.provider in (SummaryProvider.LOCAL, SummaryProvider.AUTO):
            try:
                summary, local_result = self._generate_local(records, base.filename)
            except Exception as exc:
                logger.warning("Local summary generation failed for %s: %s", base.filename, exc)
                summary = ""
            if summary:
                provider_used = SummaryProvider.LOCAL.value
                algorithm = local_result.algorithm if local_result else ""
                semantic_unit = local_result.semantic_unit if local_result else "NONE"
                input_chars = local_result.input_chars if local_result else 0
                chunks_considered = (
                    local_result.chunks_considered if local_result else len(records)
                )
                chunks_selected = local_result.chunks_selected if local_result else 0
                used_fallback = local_result.fallback if local_result else False

        if not summary and (
            self.provider == SummaryProvider.AUTO
            or (self.provider == SummaryProvider.LOCAL and self.fallback_to_llm)
        ):
            try:
                summary, llm_input_chars = self._generate_llm(records, base.filename)
            except Exception as exc:
                logger.warning("LLM summary generation failed for %s: %s", base.filename, exc)
                summary = ""
                llm_input_chars = 0
            if summary:
                provider_used = SummaryProvider.LLM.value
                algorithm = "LLM"
                semantic_unit = "NONE"
                input_chars = llm_input_chars
                chunks_selected = 0
                used_fallback = True

        if not summary and self.provider == SummaryProvider.LLM:
            try:
                summary, llm_input_chars = self._generate_llm(records, base.filename)
            except Exception as exc:
                logger.warning("LLM summary generation failed for %s: %s", base.filename, exc)
                summary = ""
                llm_input_chars = 0
            if summary:
                provider_used = SummaryProvider.LLM.value
                algorithm = "LLM"
                semantic_unit = "NONE"
                input_chars = llm_input_chars
                chunks_selected = 0

        if not summary:
            return None

        modalities = sorted(
            {
                str(record.metadata.get("modality", "")).strip()
                for record in records
                if str(record.metadata.get("modality", "")).strip()
            }
        )
        if not modalities:
            modalities = [base.file_type]
        elapsed_ms = int((time.perf_counter() - started) * 1000)
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
            metadata={
                "generated": True,
                "summary_provider": provider_used,
                "summary_algorithm": algorithm,
                "summary_semantic_unit": semantic_unit,
                "summary_input_chars": input_chars,
                "summary_output_words": len(summary.split()),
                "summary_chunks_considered": chunks_considered,
                "summary_chunks_selected": chunks_selected,
                "summary_elapsed_ms": elapsed_ms,
                "summary_fallback": used_fallback,
                "source_modalities": modalities,
            },
        )
