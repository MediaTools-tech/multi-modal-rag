"""Entry point used by PyInstaller to build the Windows GUI executable."""

import sys

from multimodal_rag.gui.app import main

if __name__ == "__main__":
    sys.exit(main())
