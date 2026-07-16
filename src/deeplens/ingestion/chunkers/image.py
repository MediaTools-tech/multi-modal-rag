"""Image ingestion chunker.

Produces exactly 1 VectorRecord per image. Extracts EXIF tags to append to the metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS
import structlog

from deeplens.config import Settings
from deeplens.core.models import FileType, VectorRecord
from deeplens.ingestion.router import FileRouter

logger = structlog.get_logger(__name__)


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
