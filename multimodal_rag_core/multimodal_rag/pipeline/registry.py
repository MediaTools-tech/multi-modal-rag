from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable

from multimodal_rag.core.models import VectorRecord
from multimodal_rag.ingestion.state import FileRecord
from multimodal_rag.pipeline.chunking import Chunk
from multimodal_rag.pipeline.context import PipelineContext, build_context

logger = logging.getLogger(__name__)

Handler = Callable[[Path, FileRecord], "str | None"]

MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".mp4": "video/mp4",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def build_handlers(context: PipelineContext | None = None) -> dict[str, Handler]:
    context = context or build_context()
    return {
        "document": _make_document_handler(context),
        "audio": _make_audio_handler(context),
        "video": _make_video_handler(context),
        "image": _make_image_handler(context),
        "*": _unsupported_handler,
    }


def _unsupported_handler(path: Path, record: FileRecord) -> str | None:
    logger.warning("No handler for kind=%s (%s)", record.kind, path.name)
    return None


def _mime(path: Path) -> str:
    return MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")


def _meta(path: Path, record: FileRecord, file_type: str) -> dict:
    stat = path.stat()
    return {
        "source_path": str(path),
        "filename": path.name,
        "parent_directory": str(path.parent),
        "file_type": file_type,
        "mime_type": _mime(path),
        "file_hash": getattr(record, "sha256", None) or "",
        "file_modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def _to_record(chunk: Chunk, meta: dict, total: int) -> VectorRecord:
    return VectorRecord(
        content=chunk.content,
        chunk_index=chunk.index,
        total_chunks=total,
        timestamp_start=chunk.timestamp_start,
        timestamp_end=chunk.timestamp_end,
        metadata=chunk.metadata,
        **meta,
    )


def _index_chunks(
    context: PipelineContext,
    path: Path,
    record: FileRecord,
    chunks: list[Chunk],
    file_type: str,
) -> str:
    if not chunks:
        return f"no content extracted from {path.name}"
    if context.embedder is None:
        raise RuntimeError(
            "Embedding engine is not configured. Set EMBEDDING_PROVIDER/GOOGLE_API_KEY "
            "in .env and restart."
        )

    settings = context.settings
    meta = _meta(path, record, file_type)
    records = [_to_record(chunk, meta, len(chunks)) for chunk in chunks]

    vectors = context.embedder.embed_texts([item.content for item in records])
    if len(vectors) != len(records):
        raise RuntimeError(
            f"Embedding count mismatch for {path.name}: {len(records)} chunk(s) "
            f"but {len(vectors)} vector(s) returned; refusing to index partial data."
        )
    for item, vector in zip(records, vectors):
        item.vector = vector
    if not records:
        return f"embedding produced no vectors for {path.name}"

    if settings.ENABLE_DOCUMENT_SUMMARIES:
        summary = context.summarizer.build_summary_record(records)
        if summary is not None:
            summary.vector = context.embedder.embed_texts([summary.content])[0]
            records.append(summary)

    context.repository.delete_by_path(meta["source_path"])
    inserted = context.repository.insert(records)
    # Index refresh is deferred: the worker flushes reindex() once the queue
    # drains (or on stop), so a backfill rebuilds the LanceDB index once, not
    # once per file. Postgres reindex() is a no-op.
    logger.info("Indexed %s records for %s", inserted, path.name)
    return f"indexed {inserted} record(s) for {path.name}"


def _make_document_handler(context: PipelineContext) -> Handler:
    from multimodal_rag.pipeline.document_parser import parse_document

    def handler(path: Path, record: FileRecord) -> str:
        chunks = parse_document(path, context.settings)
        return _index_chunks(context, path, record, chunks, "document")

    return handler


def _make_image_handler(context: PipelineContext) -> Handler:
    from multimodal_rag.pipeline.image_parser import parse_image

    def handler(path: Path, record: FileRecord) -> str:
        chunks = parse_image(path, context.settings, vlm=context.vlm)
        if not chunks and context.vlm is None:
            return (
                f"no content extracted from {path.name}: no VLM configured "
                "(set VLM_PROVIDER to GEMINI/OPENAI/ANTHROPIC/XAI + VLM_API_KEY, or OLLAMA/LOCAL)"
                " and no OCR text found"
            )
        return _index_chunks(context, path, record, chunks, "image")

    return handler


def _make_audio_handler(context: PipelineContext) -> Handler:
    def handler(path: Path, record: FileRecord) -> str:
        if context.transcribe is None:
            raise RuntimeError(
                "Audio transcription unavailable: install faster-whisper"
            )
        chunks = context.transcribe(path)
        return _index_chunks(context, path, record, chunks, "audio")

    return handler


def _make_video_handler(context: PipelineContext) -> Handler:
    def handler(path: Path, record: FileRecord) -> str:
        settings = context.settings
        chunks: list[Chunk] = []

        if context.transcribe is not None:
            from multimodal_rag.pipeline.video_extractor import extract_audio_track

            audio_path = extract_audio_track(path, settings)
            if audio_path is not None:
                try:
                    for chunk in context.transcribe(audio_path):
                        chunk.metadata["modality"] = "audio"
                        chunks.append(chunk)
                finally:
                    shutil.rmtree(audio_path.parent, ignore_errors=True)

        if context.vlm is not None:
            from multimodal_rag.pipeline.video_extractor import extract_keyframes

            frames = extract_keyframes(path, settings)
            descriptions = context.vlm.describe_frames(
                [(frame.timestamp, frame.jpeg) for frame in frames]
            )
            for frame, description in zip(frames, descriptions):
                if not description:
                    continue
                chunks.append(
                    Chunk(
                        content=f"[Visual @ {frame.timestamp:.2f}s] {description}",
                        index=len(chunks),
                        timestamp_start=frame.timestamp,
                        timestamp_end=frame.timestamp,
                        metadata={"modality": "visual"},
                    )
                )

        return _index_chunks(context, path, record, chunks, "video")

    return handler
