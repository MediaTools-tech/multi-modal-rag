"""Application instantiation helper for PySide6 GUI."""

from __future__ import annotations

import sys
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from deeplens.config import get_settings
from deeplens.gui.main_window import MainWindow


def create_app() -> tuple[QApplication, MainWindow]:
    """Instantiate QApplication and MainWindow with standard themes."""
    app = QApplication(sys.argv)
    app.setApplicationName("DeepLens")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("DeepLens Team")
    
    # Load stylesheet
    settings = get_settings()
    # Read QSS
    # Check theme from configuration (we load dark.qss by default)
    try:
        from pathlib import Path
        qss_path = Path(__file__).parent / "themes" / "dark.qss"
        if qss_path.exists():
            with open(qss_path, "r", encoding="utf-8") as f:
                app.setStyleSheet(f.read())
    except Exception:
        pass

    window = MainWindow()
    window.show()
    
    return app, window
