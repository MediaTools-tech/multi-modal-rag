from __future__ import annotations

import logging
from pathlib import Path

from multimodal_rag.config import Settings
from multimodal_rag.pipeline.chunking import (
    CHUNK_OVERLAP_PREFIX,
    CHUNK_SOURCE_END,
    CHUNK_SOURCE_START,
    Chunk,
    token_chunk_spans,
)
from multimodal_rag.utils.embedding_budget import chunk_token_target

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[tuple, object] = {}


def is_available() -> bool:
    try:
        import faster_whisper  # noqa: F401

        return True
    except Exception:
        return False


def _av_has_metadata_errors() -> bool | None:
    """Whether ``av.open()`` accepts ``metadata_errors`` (None = unknown).

    faster-whisper calls ``av.open(..., metadata_errors="ignore")``; PyAV v19
    removed that argument. ``inspect.signature`` is tried first, but ``av.open``
    is a C builtin with no signature on some builds, so fall back to the
    version (present in av 11-18, removed in 19).
    """
    try:
        import av
    except Exception:
        return None
    try:
        import inspect

        return "metadata_errors" in inspect.signature(av.open).parameters
    except (TypeError, ValueError):
        pass
    try:
        major = int(str(getattr(av, "__version__", "0")).split(".")[0])
    except Exception:
        return None
    return major < 19


def _check_av_compat() -> None:
    """Fail fast with an actionable error on the known faster-whisper/av split.

    Without this, an incompatible av fails every transcription with a bare
    ``TypeError`` deep inside retries, invisible in the GUI's Error column.
    """
    if _av_has_metadata_errors() is False:
        raise RuntimeError(
            "Installed 'av' is incompatible with faster-whisper "
            "(av.open() lacks metadata_errors, removed in av 19). "
            'Fix: pip install "av>=11,<19", then retry the file.'
        )


def _get_model(settings: Settings):
    key = (settings.WHISPER_MODEL_SIZE, settings.whisper_device, settings.whisper_compute_type)
    if key not in _MODEL_CACHE:
        from faster_whisper import WhisperModel

        logger.info("Loading faster-whisper model %s on %s", key[0], key[1])
        _MODEL_CACHE[key] = WhisperModel(key[0], device=key[1], compute_type=key[2])
    return _MODEL_CACHE[key]


def _transcript_spans(text: str, settings: Settings):
    provider = getattr(settings.EMBEDDING_PROVIDER, "value", settings.EMBEDDING_PROVIDER)
    embedding_enabled = provider != "NONE"
    overlap_tokens = max(0, int(getattr(settings, "DOC_CHUNK_OVERLAP_TOKENS", 0) or 0))
    overlap_ratio = 0.0
    if overlap_tokens <= 0 and settings.DOC_CHUNK_OVERLAP > 0 and settings.DOC_CHUNK_SIZE > 0:
        overlap_ratio = min(1.0, settings.DOC_CHUNK_OVERLAP / settings.DOC_CHUNK_SIZE)
    return token_chunk_spans(
        text,
        model=settings.EMBEDDING_MODEL,
        local=provider == "LOCAL",
        max_tokens=chunk_token_target(
            settings.EMBEDDING_MODEL,
            int(getattr(settings, "DOC_CHUNK_TOKENS", 0) or 0),
            embedding_enabled=embedding_enabled,
        ),
        overlap_tokens=overlap_tokens,
        overlap_ratio=overlap_ratio,
        max_words=settings.DOC_CHUNK_SIZE,
    )


def _transcript_max_seconds(settings: Settings) -> float:
    return max(0.0, float(getattr(settings, "WHISPER_CHUNK_MAX_SECONDS", 10.0) or 0.0))


def _split_long_segments(
    segments: list[tuple[str, float, float]], max_seconds: float
) -> list[tuple[str, float, float]]:
    """Split a long Whisper segment so every piece fits the time cap."""
    if max_seconds <= 0:
        return segments

    expanded: list[tuple[str, float, float]] = []
    for text, start, end in segments:
        duration = max(0.0, end - start)
        if duration <= max_seconds:
            expanded.append((text, start, end))
            continue
        words = text.split()
        if not words:
            continue
        # Keep each piece at or below the cap using word-position time as a
        # proxy; a single word may still exceed it in extremely slow speech.
        word_limit = max(1, int(max_seconds * len(words) / duration))
        cursor = 0
        while cursor < len(words):
            next_cursor = min(len(words), cursor + word_limit)
            piece_start = start + duration * (cursor / len(words))
            piece_end = start + duration * (next_cursor / len(words))
            expanded.append((" ".join(words[cursor:next_cursor]), piece_start, piece_end))
            cursor = next_cursor
    return expanded


def _time_group_ranges(
    segments: list[tuple[str, float, float]], max_seconds: float
) -> list[tuple[int, int]]:
    """Return ``[start, end)`` segment ranges bounded by audio duration."""
    if not segments:
        return []
    if max_seconds <= 0:
        return [(0, len(segments))]

    groups: list[tuple[int, int]] = []
    first = 0
    for index in range(1, len(segments)):
        if segments[index][2] - segments[first][1] > max_seconds:
            groups.append((first, index))
            first = index
    groups.append((first, len(segments)))
    return groups


def _merge_transcript_segments(
    segments: list[tuple[str, float, float]], settings: Settings
) -> list[Chunk]:
    """Pack short Whisper segments into time- and embedding-bounded chunks.

    Whisper emits one segment per few words, which produces tiny records with
    poor context. Adjacent segments are joined until the next one would exceed
    ``WHISPER_CHUNK_MAX_SECONDS`` or the text would exceed the token/word
    budget, so slow speech cannot turn a whole talk into one chunk. Each output
    chunk keeps the start of its first segment and end of its last.
    """
    if not segments:
        return []

    max_seconds = _transcript_max_seconds(settings)
    segments = _split_long_segments(segments, max_seconds)

    parts: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    offset = 0
    for text, _start, _end in segments:
        if parts:
            parts.append(" ")
            offset += 1
        starts.append(offset)
        parts.append(text)
        offset += len(text)
        ends.append(offset)
    joined = "".join(parts)

    chunks: list[Chunk] = []
    for group_first, group_last in _time_group_ranges(segments, max_seconds):
        group_start = starts[group_first]
        group_end = ends[group_last - 1]
        group_text = joined[group_start:group_end]
        for span in _transcript_spans(group_text, settings):
            start = group_start + span.start
            end = group_start + span.end
            first = next(
                (index for index in range(group_first, group_last) if ends[index] > start),
                group_first,
            )
            last = next(
                (
                    index
                    for index in range(group_last - 1, group_first - 1, -1)
                    if starts[index] < end
                ),
                first,
            )
            chunks.append(
                Chunk(
                    content=joined[start:end],
                    index=len(chunks),
                    timestamp_start=segments[first][1],
                    timestamp_end=segments[last][2],
                    metadata={
                        CHUNK_SOURCE_START: start,
                        CHUNK_SOURCE_END: end,
                        CHUNK_OVERLAP_PREFIX: span.overlap_prefix_chars,
                    },
                )
            )
    return chunks


def transcribe_audio(path: Path, settings: Settings) -> list[Chunk]:
    _check_av_compat()
    model = _get_model(settings)
    segments, _info = model.transcribe(str(path), language=settings.WHISPER_LANGUAGE or None)
    collected: list[tuple[str, float, float]] = []
    for segment in segments:
        text = (segment.text or "").strip()
        if not text:
            continue
        collected.append((text, float(segment.start), float(segment.end)))
    return _merge_transcript_segments(collected, settings)
