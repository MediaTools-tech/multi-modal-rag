"""Subprocess helpers for windowed (PyInstaller) desktop builds.

On Windows, every child process spawned from a ``console=False`` executable
pops a visible terminal window unless explicitly suppressed. The ingestion
worker (ffmpeg audio extraction) and the CUDA probe run in the background, so
a flashing console on every video file looks like a crash. Pass
``**hidden_kwargs()`` to ``subprocess.run``/``Popen`` — a no-op off Windows.
"""

from __future__ import annotations

import os
import subprocess


def hidden_kwargs() -> dict:
    """kwargs suppressing the child console window on Windows."""
    if os.name != "nt":
        return {}
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if flags:
        return {"creationflags": flags}
    info = subprocess.STARTUPINFO()
    info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return {"startupinfo": info}
