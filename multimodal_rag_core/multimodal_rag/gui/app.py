from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path


def _has_display() -> bool:
    if os.environ.get("QT_QPA_PLATFORM"):
        return True
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _seed_frozen_env() -> Path | None:
    """First-run: seed .env + .env.example next to a frozen exe.

    Without a .env beside the exe the app root falls back to
    ~/.multimodal_rag (fresh state, no keys, stray data dir), so a frozen
    first launch with no config is effectively broken. The bundled
    .env.example (see datas in mrag-gui.spec) is copied to both files.
    Never overwrites an existing .env; never raises.
    """
    if not getattr(sys, "frozen", False):
        return None
    try:
        exe_dir = Path(sys.executable).resolve().parent
        target = exe_dir / ".env"
        if target.is_file():
            return target
        candidates = []
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / ".env.example")
        candidates.append(exe_dir / "_internal" / ".env.example")
        for src in candidates:
            if src.is_file():
                shutil.copyfile(src, target)
                example_copy = exe_dir / ".env.example"
                if not example_copy.is_file():
                    shutil.copyfile(src, example_copy)
                logging.info("seeded first-run config: %s", target)
                return target
        logging.warning("bundled .env.example not found; running without seeded config")
    except Exception as exc:  # noqa: BLE001
        logging.warning("config seeding failed: %s", exc)
    return None


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        from PySide6.QtWidgets import QApplication
    except Exception:
        print(
            'PySide6 is not installed. Install the GUI extra: pip install -e ".[gui]"',
            file=sys.stderr,
        )
        return 1

    if not _has_display():
        print(
            "No display detected. mrag-gui needs a desktop session (X11/Wayland).\n"
            "On a headless server use the CLI instead: `mrag-ingest` / `mrag-query`.",
            file=sys.stderr,
        )
        return 2

    from PySide6.QtCore import QTimer

    from multimodal_rag.config import get_settings
    from multimodal_rag.gui.main_window import MainWindow
    from multimodal_rag.gui.theme import app_qss, get_palette

    _seed_frozen_env()
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Multi-Engine Hybrid RAG")
    app.setStyleSheet(app_qss(get_palette(get_settings().GUI_THEME.value)))
    window = MainWindow()
    window.show()
    # After the window is up, offer to fetch ffmpeg if it is missing.
    QTimer.singleShot(600, window.ensure_ffmpeg)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
