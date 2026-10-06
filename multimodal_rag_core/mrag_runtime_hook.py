"""PyInstaller runtime hook.

1. Portable PATH: prepend the exe's folder to PATH when frozen, so an
   ``ffmpeg.exe`` shipped next to the exe is found by ``shutil.which`` for
   video audio extraction. Harmless when absent.
2. Portable app root: default MRAG_HOME to the exe's folder when frozen, so the
   effective config/data root no longer depends on the *launch* folder. Without
   this, a ``.env`` edited next to the exe is silently ignored when the app is
   started from anywhere else (the root fell back to ``~/.multimodal_rag``).
   An explicitly set MRAG_HOME still wins.
3. First-run seed: copy the bundled ``.env.example`` to ``.env`` next to the exe.
   This must happen here — ``multimodal_rag.config`` freezes APP_ROOT and the
   .env file list at import time, so seeding later in ``gui/app.py`` misses the
   first launch.
4. Opt-in diagnostics (MRAG_DEBUG=1): force transformers to DEBUG and mirror
   logging to %TEMP%/mrag_startup.log — useful for frozen-app import failures.
   Off by default so normal users get no extra logging or file writes.
"""

import logging
import os
import shutil
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


def _anchor_app_root() -> str | None:
    if not getattr(sys, "frozen", False):
        return None
    if os.environ.get("MRAG_HOME"):
        return os.environ["MRAG_HOME"]
    try:
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    except Exception:  # never let diagnostics break startup
        return None
    if exe_dir:
        os.environ["MRAG_HOME"] = exe_dir
        return exe_dir
    return None


def _seed_frozen_env() -> str | None:
    """First-run: seed .env + .env.example next to a frozen exe.

    Never overwrites an existing .env; never raises.
    """
    if not getattr(sys, "frozen", False):
        return None
    try:
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        target = os.path.join(exe_dir, ".env")
        if os.path.isfile(target):
            return target
        candidates = []
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(os.path.join(meipass, ".env.example"))
        candidates.append(os.path.join(exe_dir, "_internal", ".env.example"))
        for src in candidates:
            if os.path.isfile(src):
                shutil.copyfile(src, target)
                example_copy = os.path.join(exe_dir, ".env.example")
                if not os.path.isfile(example_copy):
                    shutil.copyfile(src, example_copy)
                logging.info("seeded first-run config: %s", target)
                return target
        logging.warning("bundled .env.example not found; running without seeded config")
    except Exception as exc:  # noqa: BLE001
        logging.warning("config seeding failed: %s", exc)
    return None


_prepend_exe_dir()
_anchor_app_root()
_seed_frozen_env()

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
