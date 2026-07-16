"""Chat and search panel widget for submitting queries and showing results."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
import structlog

logger = structlog.get_logger(__name__)


class ClickableLabel(QLabel):
    """QLabel that emits a clicked signal, styled like a link."""

    clicked = Signal(str)

    def __init__(self, text: str, path: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.path = path
        self.setStyleSheet("color: #6c63ff; text-decoration: underline; font-weight: bold;")
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self.path)


class ChatPanel(QWidget):
    """Center UI panel handling conversation messages and RAG results."""

    query_submitted = Signal(str)
    result_clicked = Signal(str)  # absolute path of result file

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # ──── Top Filter Row ────
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter Type:", self))
        
        self.filter_combo = QComboBox(self)
        self.filter_combo.addItems(["All", "Documents", "Images", "Audio", "Video"])
        filter_layout.addWidget(self.filter_combo)
        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        # ──── Center Conversation Scroll Area ────
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background-color: #181826; border: none; border-radius: 8px;")

        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_layout.setSpacing(15)
        
        self.scroll_area.setWidget(self.scroll_widget)
        layout.addWidget(self.scroll_area)

        # ──── Bottom Input Row ────
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText("Search files or ask a question... (Ctrl+K)")
        self.input_field.returnPressed.connect(self._on_send_clicked)
        input_layout.addWidget(self.input_field)

        self.btn_send = QPushButton("Send", self)
        self.btn_send.clicked.connect(self._on_send_clicked)
        input_layout.addWidget(self.btn_send)
        layout.addLayout(input_layout)

        # Add initial greeting bubble
        self.add_assistant_message(
            "Hello! I am DeepLens. Register a folder on the left, and start searching "
            "or querying your files semantically.",
            []
        )

    def _on_send_clicked(self) -> None:
        text = self.input_field.text().strip()
        if text:
            self.input_field.clear()
            self.add_user_message(text)
            self.query_submitted.emit(text)

    def get_active_type_filter(self) -> str | None:
        """Return the active file type filter mapped to FileType string values."""
        val = self.filter_combo.currentText().lower()
        if val == "all":
            return None
        # Maps "documents" -> "document", "images" -> "image" etc.
        if val.endswith("s"):
            return val[:-1]
        return val

    def add_user_message(self, text: str) -> None:
        """Add user query speech bubble aligned to the right."""
        bubble = QFrame(self)
        bubble.setStyleSheet(
            "background-color: #6c63ff; color: #ffffff; border-radius: 12px; padding: 10px;"
        )
        bubble.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        
        lbl = QLabel(text, bubble)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color: #ffffff;")
        
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.addWidget(lbl)

        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addStretch()
        row_layout.addWidget(bubble)

        self.scroll_layout.addWidget(row_widget)
        self._scroll_to_bottom()

    def add_assistant_message(self, text: str, results_data: list) -> None:
        """Add AI response bubble with search result cards on the left."""
        # AI bubble container
        bubble = QFrame(self)
        bubble.setStyleSheet(
            "background-color: #252538; color: #e0e0e0; border-radius: 12px; padding: 12px;"
        )
        bubble.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred)
        
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setSpacing(10)

        # Answer body
        lbl_text = QLabel(text, bubble)
        lbl_text.setWordWrap(True)
        lbl_text.setTextFormat(Qt.TextFormat.MarkdownText)
        lbl_text.setStyleSheet("color: #e0e0e0; line-height: 1.4;")
        bubble_layout.addWidget(lbl_text)

        # If we have vector search matches, show clickable result card items
        if results_data:
            lbl_results_title = QLabel("Relevant Sources Found:", bubble)
            lbl_results_title.setStyleSheet("font-weight: bold; color: #a0a0b0; font-size: 11px;")
            bubble_layout.addWidget(lbl_results_title)

            for item in results_data:
                # Expects dict with score, filename, path, snippet
                score = item.get("score", 0.0)
                name = item.get("filename", "")
                path = item.get("absolute_path", "")
                snippet = item.get("content", "")

                card = QFrame(bubble)
                card.setStyleSheet("background-color: #1a1a2e; border-radius: 6px; padding: 8px;")
                card_layout = QVBoxLayout(card)
                card_layout.setSpacing(4)

                # Header with score and filename link
                header_layout = QHBoxLayout()
                
                score_badge = QLabel(f"{int(score*100)}% match", card)
                score_badge.setStyleSheet(
                    "background-color: #3e3e5c; color: #6c63ff; border-radius: 4px; "
                    "padding: 2px 6px; font-size: 10px; font-weight: bold;"
                )
                header_layout.addWidget(score_badge)

                link = ClickableLabel(name, path, card)
                link.clicked.connect(self.result_clicked.emit)
                header_layout.addWidget(link)
                header_layout.addStretch()
                card_layout.addLayout(header_layout)

                # Snippet
                if snippet:
                    snip_lbl = QLabel(snippet[:150] + "..." if len(snippet) > 150 else snippet, card)
                    snip_lbl.setWordWrap(True)
                    snip_lbl.setStyleSheet("color: #a0a0b0; font-size: 11px;")
                    card_layout.addWidget(snip_lbl)

                bubble_layout.addWidget(card)

        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(bubble)
        row_layout.addStretch()

        self.scroll_layout.addWidget(row_widget)
        self._scroll_to_bottom()

    def show_loading(self) -> None:
        """Display a temporary loading/typing indicator."""
        self.loading_widget = QWidget(self)
        lbl = QLabel("DeepLens is thinking...", self.loading_widget)
        lbl.setStyleSheet("color: #6c63ff; font-style: italic;")
        
        l_layout = QHBoxLayout(self.loading_widget)
        l_layout.setContentsMargins(15, 5, 0, 5)
        l_layout.addWidget(lbl)
        
        self.scroll_layout.addWidget(self.loading_widget)
        self._scroll_to_bottom()

    def hide_loading(self) -> None:
        """Remove the loading indicator."""
        if hasattr(self, "loading_widget") and self.loading_widget:
            self.scroll_layout.removeWidget(self.loading_widget)
            self.loading_widget.deleteLater()
            self.loading_widget = None

    def _scroll_to_bottom(self) -> None:
        # Use QScrollBar control
        scrollbar = self.scroll_area.verticalScrollBar()
        # Single shot timer or direct call (we call directly, then defer)
        scrollbar.setValue(scrollbar.maximum())
        from PySide6.QtCore import QTimer
        QTimer.singleShot(50, lambda: scrollbar.setValue(scrollbar.maximum()))
