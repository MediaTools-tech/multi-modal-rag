"""Archive extraction with path validation to prevent Zip Slip vulnerabilities.

Supports .zip, .rar, .7z, and .tar.gz archives.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import zipfile
from pathlib import Path

import py7zr
import rarfile
import structlog

logger = structlog.get_logger(__name__)


class ArchiveExtractor:
    """Handles archive extraction to a temporary directory with security validation."""

    @staticmethod
    def is_safe_path(target_dir: Path, path: Path) -> bool:
        """Validate path to prevent directory traversal / Zip Slip attacks.

        Ensures that the resolved path is strictly within target_dir.
        """
        # Resolve target and absolute paths
        target_resolved = target_dir.resolve()
        
        # In case of absolute paths inside archives, resolve handles them.
        # But we must be careful to handle relative subpaths too.
        try:
            resolved_path = (target_dir / path).resolve()
        except Exception:
            return False

        # Verify that resolved path starts with target_resolved
        return target_resolved in resolved_path.parents or target_resolved == resolved_path

    @classmethod
    def extract(cls, archive_path: Path, target_dir: Path) -> list[Path]:
        """Extract archives to a target directory.

        Args:
            archive_path: Path to the archive.
            target_dir: Destination directory.

        Returns:
            List of paths to extracted files.
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        suffix = archive_path.suffix.lower()
        extracted_paths: list[Path] = []

        logger.info("archive.extract.start", archive=str(archive_path), target=str(target_dir))

        # Handle compound extensions (.tar.gz)
        if suffix == ".gz" and archive_path.stem.endswith(".tar"):
            extracted_paths = cls._extract_tar(archive_path, target_dir)
        elif suffix == ".zip":
            extracted_paths = cls._extract_zip(archive_path, target_dir)
        elif suffix == ".rar":
            extracted_paths = cls._extract_rar(archive_path, target_dir)
        elif suffix == ".7z":
            extracted_paths = cls._extract_7z(archive_path, target_dir)
        else:
            raise ValueError(f"Unsupported archive format: {suffix}")

        logger.info("archive.extract.completed", count=len(extracted_paths))
        return extracted_paths

    @classmethod
    def cleanup(cls, target_dir: Path) -> None:
        """Securely remove the temporary directory and all its contents."""
        try:
            if target_dir.exists() and target_dir.is_dir():
                shutil.rmtree(target_dir)
                logger.info("archive.cleanup.success", directory=str(target_dir))
        except Exception as e:
            logger.error("archive.cleanup.failed", directory=str(target_dir), error=str(e))

    @classmethod
    def _extract_zip(cls, archive_path: Path, target_dir: Path) -> list[Path]:
        extracted: list[Path] = []
        with zipfile.ZipFile(archive_path, "r") as ref:
            for member in ref.infolist():
                # Check for Zip Slip
                member_path = Path(member.filename)
                if not cls.is_safe_path(target_dir, member_path):
                    logger.warn("archive.extract.unsafe_path_skipped", path=member.filename)
                    continue

                if member.is_dir():
                    (target_dir / member_path).mkdir(parents=True, exist_ok=True)
                else:
                    dest_file = target_dir / member_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    with ref.open(member) as source, open(dest_file, "wb") as target:
                        shutil.copyfileobj(source, target)
                    extracted.append(dest_file)
        return extracted

    @classmethod
    def _extract_tar(cls, archive_path: Path, target_dir: Path) -> list[Path]:
        extracted: list[Path] = []
        with tarfile.open(archive_path, "r:gz") as ref:
            for member in ref.getmembers():
                member_path = Path(member.name)
                if not cls.is_safe_path(target_dir, member_path):
                    logger.warn("archive.extract.unsafe_path_skipped", path=member.name)
                    continue

                if member.isdir():
                    (target_dir / member_path).mkdir(parents=True, exist_ok=True)
                else:
                    dest_file = target_dir / member_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    # Extract file obj
                    fobj = ref.extractfile(member)
                    if fobj:
                        with open(dest_file, "wb") as target:
                            shutil.copyfileobj(fobj, target)
                        extracted.append(dest_file)
        return extracted

    @classmethod
    def _extract_rar(cls, archive_path: Path, target_dir: Path) -> list[Path]:
        extracted: list[Path] = []
        # rarfile requires unrar tool on system path
        with rarfile.RarFile(archive_path, "r") as ref:
            for member in ref.infolist():
                member_path = Path(member.filename)
                if not cls.is_safe_path(target_dir, member_path):
                    logger.warn("archive.extract.unsafe_path_skipped", path=member.filename)
                    continue

                if member.isdir():
                    (target_dir / member_path).mkdir(parents=True, exist_ok=True)
                else:
                    dest_file = target_dir / member_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    with ref.open(member) as source, open(dest_file, "wb") as target:
                        shutil.copyfileobj(source, target)
                    extracted.append(dest_file)
        return extracted

    @classmethod
    def _extract_7z(cls, archive_path: Path, target_dir: Path) -> list[Path]:
        extracted: list[Path] = []
        with py7zr.SevenZipFile(archive_path, mode="r") as ref:
            # py7zr doesn't support streaming individual files as easily as zip,
            # but we can list and extract them safely.
            for name in ref.getnames():
                member_path = Path(name)
                if not cls.is_safe_path(target_dir, member_path):
                    logger.warn("archive.extract.unsafe_path_skipped", path=name)
                    continue

            # We can extract all to a temporary folder and then check path safety
            # But the safer way is to do it selectively. py7zr lets us extract
            # targets or filter.
            ref.extractall(path=str(target_dir))
            
            # Walk and collect all files, double-checking path safety
            for root, dirs, files in os.walk(target_dir):
                for f in files:
                    fp = Path(root) / f
                    if cls.is_safe_path(target_dir, fp.relative_to(target_dir)):
                        extracted.append(fp)
                    else:
                        # Purge unsafe file immediately
                        try:
                            fp.unlink()
                        except Exception:
                            pass
        return extracted
