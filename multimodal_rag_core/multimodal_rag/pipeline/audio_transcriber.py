from __future__ import annotations

import logging
from pathlib import Path

from multimodal_rag.config import Settings
from multimodal_rag.pipeline.chunking import Chunk

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


def transcribe_audio(path: Path, settings: Settings) -> list[Chunk]:
    _check_av_compat()
    model = _get_model(settings)
    segments, _info = model.transcribe(
        str(path), language=settings.WHISPER_LANGUAGE or None
    )
    chunks: list[Chunk] = []
    for index, segment in enumerate(segments):
        text = (segment.text or "").strip()
        if not text:
            continue
        chunks.append(
            Chunk(
                content=text,
                index=len(chunks),
                timestamp_start=float(segment.start),
                timestamp_end=float(segment.end),
            )
        )
    return chunks
