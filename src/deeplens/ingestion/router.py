"""File router for scanning, classifying, hashing, and collecting metadata.

Routes files to their respective chunkers based on extension.
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
from pathlib import Path
from typing import Any

import structlog

from deeplens.core.models import FileType, classify_file

logger = structlog.get_logger(__name__)


class FileRouter:
    """Scans directories, identifies file categories, computes hashes, and extracts metadata."""

    @staticmethod
    def route(path: Path) -> FileType:
        """Classify file category using filename suffix."""
        return classify_file(path)

    @classmethod
    def scan_directory(cls, dir_path: Path) -> dict[FileType, list[Path]]:
        """Recursively scan directory and group files by FileType.

        Ignores hidden files and folders (e.g. .git, .DS_Store, .settings).
        """
        grouped: dict[FileType, list[Path]] = {
            FileType.DOCUMENT: [],
            FileType.IMAGE: [],
            FileType.AUDIO: [],
            FileType.VIDEO: [],
            FileType.SUBTITLE: [],
            FileType.ARCHIVE: [],
            FileType.UNKNOWN: [],
        }

        if not dir_path.exists() or not dir_path.is_dir():
            logger.warn("router.scan.invalid_directory", path=str(dir_path))
            return grouped

        logger.info("router.scan.start", path=str(dir_path))

        for root, dirs, files in os.walk(dir_path):
            # In-place modify dirs to skip hidden directories (starting with '.')
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for f in files:
                if f.startswith("."):
                    continue

                fp = Path(root) / f
                ftype = cls.route(fp)
                grouped[ftype].append(fp)

        # Log summary
        counts = {k.value: len(v) for k, v in grouped.items()}
        logger.info("router.scan.completed", counts=counts)
        return grouped

    @staticmethod
    def compute_file_hash(path: Path) -> str:
        """Compute SHA-256 hash of a file for integrity check and deduplication."""
        sha256 = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                # Read in 64KB chunks
                for chunk in iter(lambda: f.read(65536), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            logger.error("router.hash.failed", path=str(path), error=str(e))
            raise

    @classmethod
    def get_file_metadata(cls, path: Path) -> dict[str, Any]:
        """Extract key system metadata for the file."""
        abs_path = str(path.resolve())
        filename = path.name
        parent_dir = str(path.parent.resolve())
        
        # Mime type
        mime_type, _ = mimetypes.guess_type(abs_path)
        if not mime_type:
            mime_type = "application/octet-stream"

        # Modified time
        stat = path.stat()
        modified_at = stat.st_mtime
        
        # Hash
        f_hash = cls.compute_file_hash(path)

        return {
            "absolute_path": abs_path,
            "filename": filename,
            "parent_directory": parent_dir,
            "mime_type": mime_type,
            "file_modified_at": os.path.getmtime(path),
            "file_hash": f_hash,
            "file_size": stat.st_size,
        }
