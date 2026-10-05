"""Locate or fetch a static ffmpeg binary for the desktop app.

ffmpeg is only needed to extract the audio track from *video* files (audio-only
files are decoded by PyAV, no ffmpeg required). Rather than bundling ~80 MB into
the PyInstaller zip, the GUI offers to download a static build on first use and
places it next to the executable or in ``~/.multimodal_rag/bin``.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

#: (downloaded_bytes, total_bytes, message); total is 0 when unknown.
ProgressCb = Callable[[int, int, str], None]


def _exe_name() -> str:
    return "ffmpeg.exe" if os.name == "nt" else "ffmpeg"


def install_dir() -> Path:
    """Directory a downloaded ffmpeg should live in.

    Frozen builds put it next to the executable (the runtime hook prepends that
    folder to PATH); source runs use a per-user bin dir.
    """
    if getattr(sys, "frozen", False):
        try:
            return Path(sys.executable).resolve().parent
        except Exception:  # noqa: BLE001
            pass
    return Path.home() / ".multimodal_rag" / "bin"


def _candidate_dirs() -> list[Path]:
    dirs: list[Path] = []
    if getattr(sys, "frozen", False):
        try:
            dirs.append(Path(sys.executable).resolve().parent)
        except Exception:  # noqa: BLE001
            pass
    dirs.append(Path.home() / ".multimodal_rag" / "bin")
    home = os.environ.get("MRAG_HOME")
    if home:
        dirs.append(Path(home) / "bin")
    return dirs


def find_ffmpeg() -> str | None:
    """Absolute path to an ffmpeg binary, or None."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    name = _exe_name()
    for directory in _candidate_dirs():
        candidate = directory / name
        if candidate.is_file():
            return str(candidate)
    return None


def ffmpeg_available() -> bool:
    return find_ffmpeg() is not None


def _download_url() -> str:
    if os.name == "nt" or sys.platform.startswith("win"):
        # Static Windows build (essentials). Direct, stable URL.
        return "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    if sys.platform == "darwin":
        return "https://evermeet.cx/ffmpeg/getrelease/zip"
    return "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"


def _download_to(url: str, target: Path, progress: ProgressCb | None) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "mrag-ffmpeg-fetch/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, open(target, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if progress:
                progress(downloaded, total, "Downloading ffmpeg...")


def _find_zip_member(names: list[str]) -> str | None:
    for candidate in names:
        normalized = candidate.replace("\\", "/").lower()
        if normalized.endswith("/bin/ffmpeg.exe") or normalized.endswith("bin/ffmpeg.exe"):
            return candidate
    for candidate in names:
        if Path(candidate).name.lower() in ("ffmpeg.exe", "ffmpeg"):
            return candidate
    return None


def _find_tar_member(tf: tarfile.TarFile) -> tarfile.TarInfo | None:
    for member in tf.getmembers():
        if member.isfile() and Path(member.name).name == "ffmpeg":
            return member
    return None


def _extract_ffmpeg(archive: Path, dest_dir: Path) -> Path:
    dest = dest_dir / _exe_name()
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            member = _find_zip_member(zf.namelist())
            if member is None:
                raise RuntimeError("ffmpeg executable not found in downloaded archive")
            with zf.open(member) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
    elif name.endswith((".tar.xz", ".txz", ".tar.gz", ".tgz")):
        mode = "r:gz" if name.endswith((".tar.gz", ".tgz")) else "r:xz"
        with tarfile.open(archive, mode) as tf:
            member = _find_tar_member(tf)
            if member is None:
                raise RuntimeError("ffmpeg executable not found in downloaded archive")
            extracted = tf.extractfile(member)
            if extracted is None:
                raise RuntimeError("could not read ffmpeg from downloaded archive")
            with extracted, open(dest, "wb") as out:
                shutil.copyfileobj(extracted, out)
    else:
        shutil.copyfile(archive, dest)
    if os.name != "nt":
        os.chmod(dest, 0o755)
    logger.info("ffmpeg installed at %s", dest)
    return dest


def download_ffmpeg(
    dest_dir: Path | str | None = None, progress: ProgressCb | None = None
) -> Path:
    """Download and extract a static ffmpeg into ``dest_dir``.

    Raises on network/extraction failure. The returned path is the installed
    binary. ``progress`` may raise to abort (used for cancellation).
    """
    target_dir = Path(dest_dir) if dest_dir else install_dir()
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        # e.g. the app lives in a read-only location; fall back to a per-user
        # bin dir, which find_ffmpeg() also searches.
        target_dir = Path.home() / ".multimodal_rag" / "bin"
        target_dir.mkdir(parents=True, exist_ok=True)
    url = _download_url()
    with tempfile.TemporaryDirectory(prefix="mrag-ffmpeg-") as tmp:
        if url.endswith(".zip"):
            suffix = ".zip"
        elif url.endswith(".tar.xz"):
            suffix = ".tar.xz"
        elif url.endswith(".tar.gz"):
            suffix = ".tar.gz"
        else:
            suffix = ""
        archive = Path(tmp) / f"ffmpeg-download{suffix}"
        if progress:
            progress(0, 0, "Connecting to the ffmpeg download server...")
        _download_to(url, archive, progress)
        if progress:
            progress(0, 0, "Extracting ffmpeg...")
        return _extract_ffmpeg(archive, target_dir)
