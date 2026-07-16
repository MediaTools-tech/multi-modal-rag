"""Image ingestion chunker.

Produces exactly 1 VectorRecord per image. Extracts EXIF tags to append to the metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from PIL import Image
from PIL.ExifTags import TAGS
import structlog

from deeplens.config import Settings
from deeplens.core.models import FileType, VectorRecord
from deeplens.ingestion.router import FileRouter

logger = structlog.get_logger(__name__)


# ── OCR (Optical Character Recognition) ──────────────────────────────
# Efficient, layered OCR pipeline:
#   1) (gate) cheap heuristic pre-filter skips obviously text-free images
#   2) (gate) engine-native text detection — recognition only runs when text exists
#   3) (cache) memoize OCR results by file hash to avoid re-computation
# Every step degrades gracefully: a missing optional dependency means
# "attempt OCR" rather than crashing ingestion.
#
# Why reuse the engine's own detector instead of a separate classifier? The
# detector (EasyOCR/DB/CRAFT) is the *same* stage OCR would run anyway, so it
# answers "is there text?" for free and can never disagree with the recognizer.

# In-memory OCR result cache keyed by file hash (survives re-ingestion per session).
_OCR_CACHE: dict[str, str] = {}
# Cache of loaded EasyOCR readers keyed by language (model load is expensive).
_EASYOCR_READERS: dict[str, Any] = {}

# Heuristic tuning — intentionally conservative so we favour recall over skipping.
_OCR_MIN_TEXT_COMPONENTS = 5
_OCR_MIN_COMPONENT_AREA = 15
_OCR_MAX_COMPONENT_AREA = 5000

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:  # pragma: no cover - optional
    _HAS_CV2 = False


def _to_rgb(image: Image.Image) -> Image.Image:
    """Return an RGB copy of the image (tesseract requires 8-bit RGB/RGBA)."""
    if image.mode in ("RGB", "RGBA"):
        return image
    return image.convert("RGB")


def _image_array(image: Image.Image):
    """Return the image as an RGB numpy array for the OCR engines."""
    import numpy as np

    return np.array(_to_rgb(image))


def _image_has_text_heuristic(image: Image.Image) -> bool:
    """Cheap text-presence pre-filter via edge density + small components.

    Text is characterized by many small high-frequency connected components
    (glyphs); natural photos have few such small components. Returns True when
    text is likely present, or when the heuristic cannot run (stays safe).
    """
    if not _HAS_CV2:
        return True

    import numpy as np

    try:
        gray = cv2.cvtColor(_image_array(image), cv2.COLOR_RGB2GRAY)
        # Binarize via Otsu; text pixels become foreground.
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        small_components = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if _OCR_MIN_COMPONENT_AREA <= area <= _OCR_MAX_COMPONENT_AREA:
                small_components += 1
                if small_components >= _OCR_MIN_TEXT_COMPONENTS:
                    return True
        return False
    except Exception:
        return True


def _get_easyocr_reader(lang: str):
    """Return a cached EasyOCR reader for the given language (avoids reload)."""
    if lang in _EASYOCR_READERS:
        return _EASYOCR_READERS[lang]
    import easyocr  # type: ignore

    reader = easyocr.Reader([lang], gpu=False)
    _EASYOCR_READERS[lang] = reader
    return reader


def _image_has_text_detected(image: Image.Image, settings: Settings) -> bool:
    """Engine-native detection gate, driven by the configured ``ocr_engine``.

    - EasyOCR: runs only its detector (cheap ``reader.detect``) and skips
      recognition when no text regions are found.
    - Tesseract: exposes no detection-only stage, so detection is fused into
      the recognition call (``_tesseract_recognize`` via ``image_to_data``).
      We therefore return True here and let that single fused call decide,
      after the heuristic pre-filter has already dropped text-free images.
    """
    if (settings.ocr_engine or "tesseract").lower() != "easyocr":
        return True
    try:
        reader = _get_easyocr_reader(settings.ocr_language or "eng")
        horizontal, free = reader.detect(_image_array(image))
        return bool(horizontal or free)
    except Exception:
        return True


def _tesseract_recognize(image: Image.Image, lang: str) -> str:
    """Run Tesseract and return reconstructed text with line structure.

    Uses ``image_to_data`` (word boxes + confidence) which performs detection
    and recognition in a single pass. This is the only detection capability
    Tesseract exposes, so it doubles as the engine-native text gate: an image
    with no recognized words yields an empty string (and is skipped).
    """
    import pytesseract  # type: ignore
    from pytesseract import Output

    data = pytesseract.image_to_data(image, lang=lang, output_type=Output.DICT)
    lines: dict[tuple[int, int, int], list[str]] = {}
    texts = data.get("text", [])
    confs = data.get("conf", [])
    for i, raw in enumerate(texts):
        word = (raw or "").strip()
        if not word:
            continue
        if confs[i] == -1:  # -1 means no recognition confidence (non-text)
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
    return "\n".join(" ".join(words) for words in lines.values()).strip()


def _should_run_ocr(image: Image.Image, settings: Settings) -> bool:
    """Combine the cheap heuristic and engine detection gates."""
    if not _image_has_text_heuristic(image):
        return False
    if not _image_has_text_detected(image, settings):
        return False
    return True


def _run_ocr(image: Image.Image, settings: Settings) -> str:
    """Run OCR recognition on an image (assumes gates already passed).

    Supports ``tesseract`` and ``easyocr``. Each engine is imported lazily
    inside a try/except so that a missing optional dependency degrades
    gracefully (returns an empty string) instead of crashing ingestion.

    Args:
        image: A PIL image to run OCR on.
        settings: Application settings (engine selection + language).

    Returns:
        The extracted text, or an empty string if OCR is unavailable/failed.
    """
    if not settings.enable_ocr:
        return ""

    engine = (settings.ocr_engine or "tesseract").lower()
    lang = settings.ocr_language or "eng"

    if engine == "easyocr":
        try:
            reader = _get_easyocr_reader(lang)
            results = reader.readtext(_image_array(image), detail=0, paragraph=True)
            text = " ".join(str(r) for r in results).strip()
            if text:
                logger.info("image_chunker.ocr_success", engine="easyocr")
            return text
        except Exception as e:
            logger.warn("image_chunker.ocr_failed", engine="easyocr", error=str(e))

    # Default to tesseract (also used as fallback when easyocr errors out).
    # Uses image_to_data so detection + recognition happen in one engine call.
    try:
        text = _tesseract_recognize(_to_rgb(image), lang)
        if text:
            logger.info("image_chunker.ocr_success", engine="tesseract")
        return text
    except Exception as e:
        logger.warn("image_chunker.ocr_failed", engine="tesseract", error=str(e))
        return ""


def _get_ocr_text(file_hash: str, image: Image.Image, settings: Settings) -> str:
    """Run the gated, cached OCR pipeline and return the extracted text."""
    if not settings.enable_ocr:
        return ""

    cached = _OCR_CACHE.get(file_hash)
    if cached is not None:
        return cached

    text = _run_ocr(image, settings) if _should_run_ocr(image, settings) else ""
    _OCR_CACHE[file_hash] = text
    return text


class ImageChunker:
    """Chunks image files, extracting structural EXIF tags."""

    @staticmethod
    def chunk(file_path: Path, settings: Settings) -> list[VectorRecord]:
        """Produce 1 VectorRecord for the image, including EXIF attributes."""
        logger.info("image_chunker.start", file=str(file_path))

        meta = FileRouter.get_file_metadata(file_path)
        exif_data = {}
        
        try:
            with Image.open(file_path) as img:
                raw_exif = img._getexif()
                if raw_exif:
                    for tag_id, value in raw_exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        # Clean up byte values or complex structures to prevent JSON errors
                        if isinstance(value, bytes):
                            try:
                                value = value.decode("utf-8", errors="ignore")
                            except Exception:
                                value = str(value)
                        elif not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                            value = str(value)
                        
                        exif_data[str(tag)] = value
        except Exception as e:
            logger.warn("image_chunker.exif_failed", file=str(file_path), error=str(e))

        # Add image dimensions
        try:
            with Image.open(file_path) as img:
                exif_data["width"] = img.width
                exif_data["height"] = img.height
                exif_data["format"] = img.format
        except Exception:
            pass

        # Build a semantic content description for local indexing fallback
        # In cloud mode, this description is overwritten by Gemini output,
        # but in local mode, it provides some basic searchability.
        tags_str = ", ".join(f"{k}: {v}" for k, v in exif_data.items() if k in (
            "Make", "Model", "DateTime", "GPSInfo", "Software", "width", "height", "format"
        ))
        
        content = f"Image file: {file_path.name}"
        if tags_str:
            content += f" ({tags_str})"

        # Run OCR on the image to make any embedded text searchable.
        # This is essential for screenshots, invoices, memes, receipts, etc.
        # The pipeline (heuristic -> detection -> recognition -> cache) only
        # spends the expensive recognition pass when text is actually present.
        ocr_text = ""
        try:
            with Image.open(file_path) as img:
                ocr_text = _get_ocr_text(meta["file_hash"], img, settings)
        except Exception as e:
            logger.warn("image_chunker.ocr_error", file=str(file_path), error=str(e))

        if ocr_text:
            content += f"\n[OCR Text]\n{ocr_text}"

        rec = VectorRecord(
            content=content,
            absolute_path=meta["absolute_path"],
            filename=meta["filename"],
            parent_directory=meta["parent_directory"],
            file_type=FileType.IMAGE.value,
            mime_type=meta["mime_type"],
            chunk_index=0,
            total_chunks=1,
            file_hash=meta["file_hash"],
            file_modified_at=str(meta["file_modified_at"]),
            metadata_json=json.dumps(exif_data),
        )

        logger.info("image_chunker.completed", file=str(file_path))
        return [rec]
