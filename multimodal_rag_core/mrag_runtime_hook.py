"""PyInstaller runtime hook.

1. Portable PATH: prepend the exe's folder to PATH when frozen, so an
   ``ffmpeg.exe`` shipped next to the exe is found by ``shutil.which`` for
   video audio extraction. Harmless when absent.
2. Opt-in diagnostics (MRAG_DEBUG=1): force transformers to DEBUG and mirror
   logging to %TEMP%/mrag_startup.log — useful for frozen-app import failures.
   Off by default so normal users get no extra logging or file writes.
"""

import logging
import os
import sys
import tempfile


def _prepend_exe_dir() -> str | None:
    if not getattr(sys, "frozen", False):
        return None
    try:
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        path = os.environ.get("PATH", "")
        if exe_dir and exe_dir not in path.split(os.pathsep):
            os.environ["PATH"] = exe_dir + os.pathsep + path
        return exe_dir
    except Exception:  # never let diagnostics break startup
        return None


_prepend_exe_dir()

if os.environ.get("MRAG_DEBUG"):
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "debug")
    try:
        _path = os.path.join(tempfile.gettempdir(), "mrag_startup.log")
        _handler = logging.FileHandler(_path, encoding="utf-8")
        _handler.setLevel(logging.DEBUG)
        _handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(_handler)
    except Exception:  # never let diagnostics break startup
        pass
