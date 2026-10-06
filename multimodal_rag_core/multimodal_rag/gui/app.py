from __future__ import annotations

import logging
import os
import sys


def _has_display() -> bool:
    if os.environ.get("QT_QPA_PLATFORM"):
        return True
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


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

    # Frozen first-run seeding + MRAG_HOME anchoring already ran in the
    # PyInstaller runtime hook (mrag_runtime_hook.py), before config import.
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
