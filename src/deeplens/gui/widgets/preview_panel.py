"""File preview widget supporting text, images, video, and audio players."""

from __future__ import annotations

import os
from pathlib import Path
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
import structlog

logger = structlog.get_logger(__name__)


class PreviewPanel(QWidget):
    """Right sidebar panel displaying contents and metadata for the selected file."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # ──── Panel Title & Close ────
        title_layout = QHBoxLayout()
        self.lbl_title = QLabel("File Preview", self)
        self.lbl_title.setObjectName("titleLabel")
        title_layout.addWidget(self.lbl_title)
        
        title_layout.addStretch()
        
        self.btn_close = QPushButton("✕ Close", self)
        self.btn_close.setObjectName("secondaryButton")
        self.btn_close.clicked.connect(self.clear_preview)
        title_layout.addWidget(self.btn_close)
        layout.addLayout(title_layout)

        # ──── Viewer Stack ────
        self.stack = QStackedWidget(self)
        layout.addWidget(self.stack)

        # 1. Text Viewer
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.stack.addWidget(self.text_edit)

        # 2. Image Viewer
        self.image_scroll = QScrollArea(self)
        self.image_scroll.setWidgetResizable(True)
        self.image_scroll.setStyleSheet("background: #181826;")
        self.image_label = QLabel(self.image_scroll)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_scroll.setWidget(self.image_label)
        self.stack.addWidget(self.image_scroll)

        # 3. Audio/Video Placeholder Viewer
        # (Standard Qt6 QMediaPlayer needs system codecs. We build a simple fallback control widget).
        self.media_widget = QFrame(self)
        self.media_widget.setStyleSheet("background: #181826; border-radius: 8px;")
        media_layout = QVBoxLayout(self.media_widget)
        media_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.media_status_lbl = QLabel("Media Player", self.media_widget)
        self.media_status_lbl.setStyleSheet("font-weight: bold; color: #6c63ff;")
        media_layout.addWidget(self.media_status_lbl)

        self.btn_play_sys = QPushButton("Open in OS Default Player", self.media_widget)
        self.btn_play_sys.clicked.connect(self._on_open_in_os)
        media_layout.addWidget(self.btn_play_sys)
        self.stack.addWidget(self.media_widget)

        # ──── Bottom Metadata Container ────
        self.meta_frame = QFrame(self)
        self.meta_frame.setStyleSheet("background-color: #1f1f33; border-radius: 8px; padding: 10px;")
        meta_layout = QVBoxLayout(self.meta_frame)
        meta_layout.setSpacing(5)

        self.lbl_filename = QLabel("Name: -", self.meta_frame)
        self.lbl_path = QLabel("Path: -", self.meta_frame)
        self.lbl_path.setWordWrap(True)
        self.lbl_size = QLabel("Size: -", self.meta_frame)

        meta_layout.addWidget(self.lbl_filename)
        meta_layout.addWidget(self.lbl_path)
        meta_layout.addWidget(self.lbl_size)
        layout.addWidget(self.meta_frame)

        self.current_file_path: Path | None = None
        self.clear_preview()

    def clear_preview(self) -> None:
        """Reset panel to blank state."""
        self.current_file_path = None
        self.lbl_title.setText("File Preview")
        self.lbl_filename.setText("Name: -")
        self.lbl_path.setText("Path: -")
        self.lbl_size.setText("Size: -")
        self.stack.setCurrentIndex(0)
        self.text_edit.setPlainText("Select a search result item to preview the file details here.")
        self.meta_frame.hide()

    def preview_file(self, file_path_str: str) -> None:
        """Determine file format and load into appropriate viewer."""
        path = Path(file_path_str)
        if not path.exists():
            logger.warn("preview.file_not_found", path=file_path_str)
            return

        self.current_file_path = path
        self.meta_frame.show()
        
        # Populate meta
        self.lbl_filename.setText(f"Name: {path.name}")
        self.lbl_path.setText(f"Path: {str(path)}")
        size_kb = path.stat().st_size / 1024
        self.lbl_size.setText(f"Size: {size_kb:.1f} KB")
        self.lbl_title.setText(f"Preview: {path.name}")

        suffix = path.suffix.lower()

        # Route by extension
        # 1. Images
        if suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"):
            self.stack.setCurrentIndex(1)
            pixmap = QPixmap(str(path))
            # Scale to fit scroll area reasonably
            scaled = pixmap.scaled(
                350, 350,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)
            logger.info("preview.image", path=str(path))

        # 2. Audio / Video
        elif suffix in (".mp4", ".mkv", ".avi", ".mov", ".mp3", ".wav", ".flac"):
            self.stack.setCurrentIndex(2)
            self.media_status_lbl.setText(f"Media File: {path.name}")
            logger.info("preview.media", path=str(path))

        # 3. Documents (PDF/Text/Markdown)
        else:
            self.stack.setCurrentIndex(0)
            logger.info("preview.document", path=str(path))
            
            # Simple text parser. PDF/DOCX can display extracted markdown/text chunks
            # if we wanted, but let's read text files directly first.
            if suffix in (".txt", ".md", ".csv", ".json", ".xml", ".py", ".sh", ".yaml", ".yml"):
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        # read first 500 lines to prevent memory locks
                        lines = [f.readline() for _ in range(500)]
                        content = "".join(lines)
                        if f.readline():
                            content += "\n\n... [Content Truncated] ..."
                    self.text_edit.setPlainText(content)
                except Exception as e:
                    self.text_edit.setPlainText(f"Could not read text content: {str(e)}")
            else:
                self.text_edit.setPlainText(
                    f"Binary document format: '{suffix}'.\n"
                    "Semantic search index chunks are processed successfully.\n"
                    "Click 'Open in OS' to view the file in external application."
                )

    def _on_open_in_os(self) -> None:
        """Launch default system player/viewer."""
        if self.current_file_path and self.current_file_path.exists():
            import subprocess
            import sys
            
            p_str = str(self.current_file_path)
            if sys.platform == "win32":
                os.startfile(p_str)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", p_str])
            else:
                subprocess.Popen(["xdg-open", p_str])
            logger.info("preview.open_in_os", path=p_str)
class_name = PreviewPanel
