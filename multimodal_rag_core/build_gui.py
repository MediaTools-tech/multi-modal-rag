"""Build the desktop GUI and seed its distribution config.

    python build_gui.py

Runs PyInstaller (mrag-gui.spec) then refreshes `.env.example` next to the exe
and seeds `.env` from it only if missing (never overwrites live keys), so the
distribution never contains real secrets. The recipient configures via the GUI
(Settings) or by editing the seeded `.env` directly.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    if shutil.which("pyinstaller") is None:
        print(
            "pyinstaller not found. Install the build extra first: "
            'pip install -e ".[gui,build]"  (or: pip install pyinstaller)',
            file=sys.stderr,
        )
        return 1

    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "mrag-gui.spec"],
        cwd=str(HERE),
    )
    if result.returncode != 0:
        return result.returncode

    dist = HERE / "dist" / "MultiModalRAG"
    example = HERE / ".env.example"
    if not dist.is_dir() or not example.is_file():
        print(f"Expected {dist} and {example} to exist after the build.", file=sys.stderr)
        return 1

    shutil.copyfile(example, dist / ".env.example")
    if not (dist / ".env").is_file():
        shutil.copyfile(example, dist / ".env")
        print(f"\nSeeded blank config: {dist / '.env'} (+ .env.example)")
    else:
        print(f"\nKept existing config: {dist / '.env'} (refreshed .env.example)")
    print(
        "Recipient: run MultiModalRAG.exe, set providers/keys in Settings "
        "(or edit .env), then restart. Keep .env next to the exe and launch "
        "from that folder."
    )
    if shutil.which("ffmpeg") is None and not (dist / "ffmpeg.exe").is_file():
        print(
            "\nNOTE: no ffmpeg found — video files will fail to ingest in the exe. "
            "For video support, place a static ffmpeg.exe next to MultiModalRAG.exe "
            "(e.g. from https://www.gyan.dev/ffmpeg/builds/); the app picks it up "
            "from its own folder at startup. Audio-only files need no ffmpeg."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
