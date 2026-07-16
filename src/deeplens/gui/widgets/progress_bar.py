"""Progress bar widget for showing file indexing status in the GUI."""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget
import structlog

logger = structlog.get_logger(__name__)


class IndexingProgressWidget(QWidget):
    """Progress widget representing file ingestion queues."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Labels
        self.label = QLabel("Ingestion Idle", self)
        self.label.setStyleSheet("font-weight: bold; font-size: 11px;")
        
        self.detail_label = QLabel("", self)
        self.detail_label.setStyleSheet("color: #a0a0b0; font-size: 11px;")

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(12)

        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.detail_label)

    def update_progress(self, progress_data: dict) -> None:
        """Update values displayed by the progress bar.

        Expects dictionary keys matching IndexingProgress fields:
        - folder_path, total_files, processed_files, current_file, status, eta_seconds
        """
        status = progress_data.get("status", "idle")
        total = progress_data.get("total_files", 0)
        processed = progress_data.get("processed_files", 0)
        current = progress_data.get("current_file", "")
        eta = progress_data.get("eta_seconds")

        # Set title
        if status == "scanning":
            self.label.setText("Scanning folder structures...")
            self.progress_bar.setRange(0, 0)  # Indeterminate
        elif status == "indexing":
            pct = int((processed / total) * 100) if total > 0 else 0
            self.label.setText(f"Indexing: {processed}/{total} files ({pct}%)")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(pct)
        elif status == "completed":
            self.label.setText("Indexing completed")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
            self.detail_label.setText("")
            return
        elif status == "error":
            self.label.setText("Indexing failed")
            self.progress_bar.setValue(0)
            return
        else:
            self.label.setText("System Idle")
            self.progress_bar.setValue(0)
            self.detail_label.setText("")
            return

        # Detailed label
        detail = ""
        if current:
            # truncate filename if too long
            filename = current[:30] + "..." if len(current) > 33 else current
            detail = f"Processing: {filename}"
        
        if eta is not None:
            m, s = divmod(int(eta), 60)
            eta_str = f"{m}m {s}s" if m > 0 else f"{s}s"
            detail += f" — ETA: {eta_str}"

        self.detail_label.setText(detail)
