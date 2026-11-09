from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from multimodal_rag.config import Settings, SummarySemanticUnit, get_settings
from multimodal_rag.database import get_repository
from multimodal_rag.database.base_interface import DocumentRepository
from multimodal_rag.ingestion.summarizer import DocumentSummarizer
from multimodal_rag.pipeline.embedding import EmbeddingEngine, get_embedding_engine
from multimodal_rag.pipeline.local_summarizer import LocalExtractiveSummarizer
from multimodal_rag.utils.api_clients import build_summarize_fn, build_vlm_describer
from multimodal_rag.utils.token_tracker import TokenTracker

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    settings: Settings
    repository: DocumentRepository
    embedder: EmbeddingEngine | None
    summarizer: DocumentSummarizer
    vlm: object | None = None
    transcribe: Callable | None = None
    tracker: TokenTracker = field(default_factory=TokenTracker)


def build_context(
    settings: Settings | None = None,
    repository: DocumentRepository | None = None,
    embedder: EmbeddingEngine | None = None,
    require_embedder: bool = True,
    tracker: TokenTracker | None = None,
) -> PipelineContext:
    settings = settings or get_settings()
    tracker = tracker or TokenTracker()

    if repository is None:
        repository = get_repository(settings)
        repository.initialize()
    if embedder is None:
        try:
            embedder = get_embedding_engine(settings, tracker)
        except Exception as exc:
            if require_embedder:
                raise
            logger.warning("Embedding engine unavailable: %s", exc)
    if embedder is None:
        if require_embedder:
            raise RuntimeError(
                "Embedding engine is disabled or not configured. Set "
                "EMBEDDING_PROVIDER=LOCAL/GOOGLE with a model/key, or use "
                "keyword-only search."
            )
        logger.info("Embedding engine disabled; indexing text for keyword search only")

    vlm = build_vlm_describer(settings, tracker)
    local_summarizer = None
    if settings.summary_uses_local:
        embed_texts = None
        if settings.SUMMARY_SEMANTIC_UNIT is SummarySemanticUnit.SENTENCE:
            if embedder is not None:
                embed_texts = embedder.embed_texts
        local_summarizer = LocalExtractiveSummarizer(settings, embed_texts=embed_texts)
    summarize_fn = build_summarize_fn(settings, tracker) if settings.summary_needs_llm else None
    summarizer = DocumentSummarizer(
        summarize_fn,
        settings.SUMMARY_MAX_CHARS,
        settings=settings,
        local_summarizer=local_summarizer,
    )

    transcribe: Callable | None = None
    if not settings.is_doc_only:
        # Skipped in doc-only mode so faster_whisper (and ctranslate2/PyAV) is
        # never imported on a documents+images-only machine.
        from multimodal_rag.pipeline.audio_transcriber import is_available as whisper_available
        from multimodal_rag.pipeline.audio_transcriber import transcribe_audio

        if whisper_available():
            transcribe = lambda path: transcribe_audio(path, settings)  # noqa: E731

    if vlm is None:
        logger.info("No VLM configured; video frames will not be captioned")
    if transcribe is None:
        logger.info("faster-whisper unavailable; audio/video speech will not be transcribed")

    return PipelineContext(
        settings=settings,
        repository=repository,
        embedder=embedder,
        summarizer=summarizer,
        vlm=vlm,
        transcribe=transcribe,
        tracker=tracker,
    )
