from __future__ import annotations

import re

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QCheckBox,
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

from multimodal_rag.config import get_settings
from multimodal_rag.gui.theme import get_palette


# Matches bare http(s):// URLs that are not already part of a markdown link.
_BARE_URL_RE = re.compile(r"(?<![\(\[<\"'=])(https?://[^\s<\)\]]+)")

_TRIM_TRAILING = ".,;:!?)]}'\""


def linkify_bare_urls(text: str) -> str:
    """Turn bare https://... URLs into markdown links so QLabel MarkdownText renders them clickable."""

    def _repl(match: re.Match[str]) -> str:
        url = match.group(1)
        # Avoid double-linking URLs already inside a markdown link destination.
        start = match.start(1)
        prefix = match.string[max(0, start - 1) : start]
        if prefix == "]" and "](" in match.string[max(0, start - 300) : start]:
            return url
        trailing = ""
        while url and url[-1] in _TRIM_TRAILING:
            trailing = url[-1] + trailing
            url = url[:-1]
        if not url:
            return match.group(1)
        return f"[{url}]({url}){trailing}"

    return _BARE_URL_RE.sub(_repl, text)


class ClickableLabel(QLabel):
    clicked = Signal(str)

    def __init__(
        self, text: str, payload: str, color: str, parent: QWidget | None = None
    ) -> None:
        super().__init__(text, parent)
        self.payload = payload
        self.setStyleSheet(f"color: {color}; text-decoration: underline; font-weight: bold;")
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.clicked.emit(self.payload)


class ChatPanel(QWidget):
    query_submitted = Signal(str)
    result_clicked = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loading: QWidget | None = None
        self.palette = get_palette(get_settings().GUI_THEME.value)
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        top = QHBoxLayout()
        top.addWidget(QLabel("Mode:", self))
        self.mode_combo = QComboBox(self)
        # Display labels describe granularity (what you get back); the stored
        # userData keeps the SearchMode values the backend expects, so labels
        # can be renamed freely without breaking search.
        self.mode_combo.addItem("auto", "auto")
        self.mode_combo.addItem("hybrid", "hybrid")
        self.mode_combo.addItem("summary", "summary")
        self.mode_combo.addItem("chunk", "chunk")
        # Default to the .env configured mode so the chat dropdown and the
        # Settings panel stay in sync (previously this always opened on "auto",
        # so queries ran as auto even when SEARCH_MODE=hybrid in .env).
        try:
            default_mode = get_settings().SEARCH_MODE.value
        except Exception:  # noqa: BLE001
            default_mode = "auto"
        idx = self.mode_combo.findData(default_mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)
        self.mode_combo.setToolTip(
            "auto: try chunk first; if retrieval is empty or the answer declines, "
            "retry once with hybrid (provenance shown with the answer).\n"
            "hybrid: chunks + summaries + keywords, fused and reranked (best recall).\n"
            "summary: one file-level match per file (which files are about this?).\n"
            "chunk: pure vector search over passages, no keywords (fast, literal)."
        )
        top.addWidget(self.mode_combo)
        self.answer_check = QCheckBox("LLM answer", self)
        self.answer_check.setChecked(True)
        self.answer_check.setToolTip(
            "Synthesize a cited answer with the configured text LLM (TEXT_LLM_PROVIDER). "
            "Uncheck for retrieved passages only."
        )
        top.addWidget(self.answer_check)
        top.addStretch()
        layout.addLayout(top)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet(
            f"background-color: {self.palette.surface_bg}; border: none; border-radius: 8px;"
        )
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_layout.setSpacing(14)
        self.scroll.setWidget(self.scroll_widget)
        layout.addWidget(self.scroll, 1)
        # Stick to the newest message: the layout pass (word-wrap height) lands
        # after addWidget returns, so a single immediate setValue(maximum) races
        # it and the view can end up above the latest bubble.
        self.scroll.verticalScrollBar().rangeChanged.connect(self._on_scroll_range_changed)

        bottom = QHBoxLayout()
        self.input = QLineEdit(self)
        self.input.setPlaceholderText("Ask a question about your files... (Enter to send)")
        self.input.returnPressed.connect(self._on_send)
        bottom.addWidget(self.input)
        self.send_btn = QPushButton("Send", self)
        self.send_btn.clicked.connect(self._on_send)
        bottom.addWidget(self.send_btn)
        layout.addLayout(bottom)

        self.set_backend_state(
            False, "Waiting for the database and index to load..."
        )
        self.add_assistant_message(
            "Starting up — connecting to the database and loading the search index. "
            "Questions are enabled as soon as it's ready."
        )

    def set_backend_state(self, ready: bool, reason: str = "") -> None:
        """Gate querying: only enable the input once the backend is usable.

        Prevents the misleading "embedding engine not configured" error when a
        question is sent while the database/index is still loading at startup.
        """
        self.input.setEnabled(ready)
        self.send_btn.setEnabled(ready)
        if ready:
            self.input.setPlaceholderText(
                "Ask a question about your files... (Enter to send)"
            )
            self.input.setToolTip("")
        else:
            hint = reason or "Waiting for the database and index to load..."
            self.input.setPlaceholderText(hint)
            self.input.setToolTip(hint)

    def _on_send(self) -> None:
        if not self.input.isEnabled():
            return
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        self.add_user_message(text)
        self.query_submitted.emit(text)

    def active_mode(self) -> str:
        data = self.mode_combo.currentData()
        if isinstance(data, str) and data:
            return data
        # Fallback for any combo without userData (never the case above).
        return self.mode_combo.currentText()

    def set_mode(self, mode: str) -> None:
        """Select a mode in the dropdown (used after Settings saves .env)."""
        if not mode:
            return
        idx = self.mode_combo.findData(mode)
        if idx < 0:
            idx = self.mode_combo.findText(mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)

    def answer_enabled(self) -> bool:
        return self.answer_check.isChecked()

    def add_user_message(self, text: str) -> None:
        p = self.palette
        bubble = QFrame()
        bubble.setStyleSheet(
            f"background-color: {p.user_bubble_bg}; color: {p.user_bubble_text}; "
            "border-radius: 12px; padding: 10px;"
        )
        bubble.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        inner = QVBoxLayout(bubble)
        label = QLabel(text)
        label.setWordWrap(True)
        # Selectable so the user can copy their own prompt.
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        label.setCursor(QCursor(Qt.CursorShape.IBeamCursor))
        label.setStyleSheet(f"color: {p.user_bubble_text};")
        inner.addWidget(label)
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addStretch()
        row_layout.addWidget(bubble)
        self.scroll_layout.addWidget(row)
        self._scroll_bottom()

    def add_assistant_message(
        self, text: str, results: list | None = None, cited: set[int] | list[int] | None = None
    ) -> None:
        p = self.palette
        bubble = QFrame()
        bubble.setStyleSheet(
            f"background-color: {p.assistant_bubble_bg}; "
            f"color: {p.assistant_bubble_text}; border-radius: 12px; padding: 12px;"
        )
        inner = QVBoxLayout(bubble)
        inner.setSpacing(8)

        body = QLabel(linkify_bare_urls(text or "(no answer)"))
        body.setWordWrap(True)
        body.setTextFormat(Qt.TextFormat.MarkdownText)
        # Selectable + clickable links (was non-interactive, so text behaved like an image).
        body.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        body.setOpenExternalLinks(True)
        body.setStyleSheet(f"color: {p.assistant_bubble_text};")
        inner.addWidget(body)

        items = list(results or [])
        if cited:
            # The passages the answer cites are what the user wants to see.
            # Reranking can still place a keyword-matching but unrelated hit
            # (e.g. a video frame) on top, so surface cited passages first while
            # preserving each card's original [N] citation number. Cited cards
            # follow the answer's mention order (first [3] then [2] -> card [3]
            # first), uncited cards keep rank order after the divider.
            if isinstance(cited, (list, tuple)):
                order: dict = {}
                for idx, rank in enumerate(cited):
                    if rank not in order:
                        order[rank] = idx
                cited_set = set(cited)
                items.sort(
                    key=lambda it: (
                        (0, order.get(it.get("rank"), 10**9))
                        if it.get("rank") in cited_set
                        else (1, it.get("rank") or 0)
                    )
                )
            else:
                items.sort(
                    key=lambda it: (it.get("rank") not in cited, it.get("rank") or 0)
                )
        divider_pending = bool(cited) and any(
            it.get("rank") not in cited for it in items
        )

        for item in items:
            result = item["result"]
            record = result.record
            rank = item.get("rank")
            is_cited = bool(cited and rank in cited)
            if divider_pending and not is_cited:
                divider_pending = False
                divider = QLabel("Other matches (not cited by the answer)")
                divider.setStyleSheet(
                    f"color: {p.text_muted}; font-size: 10px; padding-top: 2px;"
                )
                inner.addWidget(divider)
            card = QFrame()
            card.setStyleSheet(
                f"background-color: {p.card_bg}; border-radius: 6px; padding: 8px;"
            )
            card_layout = QVBoxLayout(card)
            card_layout.setSpacing(4)
            header = QHBoxLayout()
            # A cited card shows only its citation number: uncalibrated rerank
            # scores commonly read ~0% for exactly the passages the answer used,
            # so pairing such a score with a "cited by answer" tag is misleading.
            if is_cited:
                badge_text = f"[{item.get('rank', '?')}]"
            else:
                badge_text = (
                    f"[{item.get('rank', '?')}] · "
                    f"{int(max(result.score, 0.0) * 100)}%"
                )
            badge = QLabel(badge_text)
            badge.setStyleSheet(
                f"background-color: {p.badge_bg}; color: {p.badge_text}; "
                "border-radius: 4px; padding: 2px 6px; font-size: 10px; font-weight: bold;"
            )
            header.addWidget(badge)
            if is_cited:
                tag = QLabel("cited by answer")
                tag.setStyleSheet(
                    f"background-color: {p.accent}; color: {p.accent_text}; "
                    "border-radius: 4px; padding: 2px 6px; font-size: 10px; font-weight: bold;"
                )
                header.addWidget(tag)
            stamp = (
                f" @ {record.timestamp_start:.1f}s"
                if record.timestamp_start is not None
                else ""
            )
            link = ClickableLabel(f"{record.filename}{stamp}", item["key"], p.link)
            link.clicked.connect(lambda key: self.result_clicked.emit(key))
            header.addWidget(link)
            header.addStretch()
            card_layout.addLayout(header)
            snippet = (record.content or "").replace("\n", " ")
            snippet = snippet[:160] + "..." if len(snippet) > 160 else snippet
            snip = QLabel(snippet)
            snip.setWordWrap(True)
            snip.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
                | Qt.TextInteractionFlag.TextSelectableByKeyboard
            )
            snip.setCursor(QCursor(Qt.CursorShape.IBeamCursor))
            snip.setStyleSheet(f"color: {p.text_muted}; font-size: 11px;")
            card_layout.addWidget(snip)
            inner.addWidget(card)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(bubble)
        row_layout.addStretch()
        self.scroll_layout.addWidget(row)
        self._scroll_bottom()

    def show_loading(self) -> None:
        self._loading = QWidget()
        layout = QHBoxLayout(self._loading)
        layout.setContentsMargins(15, 5, 0, 5)
        label = QLabel("Searching...")
        label.setStyleSheet(f"color: {self.palette.link}; font-style: italic;")
        layout.addWidget(label)
        self.scroll_layout.addWidget(self._loading)
        self._scroll_bottom()

    def hide_loading(self) -> None:
        if self._loading is not None:
            self.scroll_layout.removeWidget(self._loading)
            self._loading.deleteLater()
            self._loading = None

    def _scroll_bottom(self) -> None:
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())
        QTimer.singleShot(50, lambda: bar.setValue(bar.maximum()))
        QTimer.singleShot(300, lambda: bar.setValue(bar.maximum()))

    def _on_scroll_range_changed(self) -> None:
        # Fires once the layout settles on its final height: snap to it so the
        # newest message is always visible without manual scrolling.
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())
