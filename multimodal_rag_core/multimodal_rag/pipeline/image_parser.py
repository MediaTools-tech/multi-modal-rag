from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from multimodal_rag.config import Settings
from multimodal_rag.pipeline.chunking import Chunk, text_to_chunks

logger = logging.getLogger(__name__)

# Cap the longest side before sending to the VLM (keeps payloads small;
# text in certificates/documents stays legible at this size).
_MAX_SIDE_PIXELS = 1600


def parse_image(path: Path, settings: Settings, vlm: Any | None = None) -> list[Chunk]:
    """Extract searchable text from a standalone image file.

    Combines best-effort OCR text with a VLM caption (reuses the existing
    ``describe_frames`` interface with a single still at ``t=0.0``, so the
    Gemini/Ollama/fallback clients, disk cache and rate limits all apply
    unchanged). Returns ``[]`` when nothing usable is extracted.
    """
    jpeg = _to_jpeg_bytes(path, settings)
    parts: list[str] = []

    ocr_text = _ocr_text(path)
    if ocr_text.strip():
        parts.append(f"[OCR @ 0.00s] {ocr_text.strip()}")

    if vlm is not None and jpeg:
        try:
            descriptions = vlm.describe_frames([(0.0, jpeg)])
        except Exception as exc:  # noqa: BLE001
            logger.warning("VLM caption failed for %s: %s", path.name, exc)
            descriptions = []
        description = (descriptions[0] if descriptions else "").strip()
        if description:
            parts.append(f"[Visual @ 0.00s] {description}")

    if not parts:
        return []
    combined = "\n\n".join(parts)
    chunks = text_to_chunks(combined, settings.DOC_CHUNK_SIZE, settings.DOC_CHUNK_OVERLAP)
    for chunk in chunks:
        chunk.metadata.setdefault("modality", "image")
    return chunks


def _to_jpeg_bytes(path: Path, settings: Settings) -> bytes:
    """Read an image file and return JPEG bytes for the VLM."""
    quality = int(getattr(settings, "MM_JPEG_QUALITY", 80))
    data = path.read_bytes()

    # Fast path: already JPEG and reasonably sized.
    if path.suffix.lower() in {".jpg", ".jpeg"} and len(data) <= 12 * 1024 * 1024:
        return data

    try:
        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("cv2 could not decode image")
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest > _MAX_SIDE_PIXELS:
            scale = _MAX_SIDE_PIXELS / longest
            image = cv2.resize(
                image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
            )
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError("cv2 JPEG encode failed")
        return bytes(buffer)
    except ImportError:
        logger.debug("cv2 not installed; trying PIL for %s", path.name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("cv2 conversion failed for %s: %s", path.name, exc)

    try:
        import io

        from PIL import Image

        image = Image.open(path)
        if image.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            alpha = image.split()[-1] if image.mode in ("RGBA", "LA") else None
            background.paste(image.convert("RGB"), mask=alpha)
            image = background
        else:
            image = image.convert("RGB")
        image.thumbnail((_MAX_SIDE_PIXELS, _MAX_SIDE_PIXELS))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()
    except ImportError:
        logger.debug("PIL not installed; using raw bytes for %s", path.name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("PIL conversion failed for %s: %s", path.name, exc)

    if path.suffix.lower() in {".jpg", ".jpeg"}:
        return data
    raise RuntimeError(
        f"Cannot convert {path.name} to JPEG: install opencv-python-headless "
        "or pillow, or use JPG files."
    )


def _ocr_text(path: Path) -> str:
    """Best-effort OCR; empty string when no OCR backend is available."""
    text = _docling_ocr(path)
    if text.strip():
        return text
    return _tesseract_ocr(path)


def _docling_ocr(path: Path) -> str:
    try:
        from docling.document_converter import DocumentConverter
    except Exception:
        logger.debug("Docling not installed; skipping Docling OCR for %s", path.name)
        return ""
    try:
        result = DocumentConverter().convert(str(path))
        return result.document.export_to_markdown() or ""
    except Exception as exc:  # noqa: BLE001
        logger.debug("Docling OCR failed for %s: %s", path.name, exc)
        return ""


def _tesseract_ocr(path: Path) -> str:
    try:
        import pytesseract  # type: ignore
        from PIL import Image
    except Exception:
        return ""
    try:
        return pytesseract.image_to_string(Image.open(path)) or ""
    except Exception as exc:  # noqa: BLE001
        logger.debug("Tesseract OCR failed for %s: %s", path.name, exc)
        return ""
