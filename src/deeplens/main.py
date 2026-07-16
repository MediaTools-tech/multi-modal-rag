"""Entry point for launching the DeepLens GUI application."""

from __future__ import annotations

import asyncio
import sys
from PySide6.QtCore import QEventLoop

from deeplens.config import get_settings
from deeplens.gui.app import create_app


def main() -> None:
    """Initialize event loop and start GUI."""
    # Ensure settings loaded
    settings = get_settings()

    # PySide6 + asyncio integration
    # To run PySide6 with async tasks safely, we must either run them in QThread
    # (which we do in SearchWorker/IngestWorker) or use a combined event loop.
    # Because we offloaded heavy tasks to QThreads, we can run the standard Qt exec() loop.
    
    app, window = create_app()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
