from __future__ import annotations

import html
import os
import shutil
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import QLabel, QMessageBox, QPushButton, QStyle, QTextBrowser, QVBoxLayout, QWidget

from multimodal_rag.config import get_settings
from multimodal_rag.gui.theme import get_palette

_TEXT_SUFFIXES = {".txt", ".md", ".csv", ".json", ".log", ".rst", ".xml", ".html", ".htm"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".gif"}
_VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg"}
_PREVIEW_MAX_CHARS = 4000


def find_vlc() -> str | None:
    found = shutil.which("vlc") or shutil.which("vlc.exe")
    if found:
        return found
    # Fall back to the standard desktop install locations: a Start-menu
    # shortcut or a cmd run from VLC's own folder works for manual use but
    # leaves PATH empty, which is exactly the "starts from 0" symptom.
    if sys.platform.startswith("win"):
        for candidate in (
            r"C:\Program Files\VideoLAN\VLC\vlc.exe",
            r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe",
        ):
            if os.path.isfile(candidate):
                return candidate
    return None


class PreviewPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.palette = get_palette(get_settings().GUI_THEME.value)
        self._play_path: Path | None = None
        self._play_ts: float | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(QLabel("Preview", self))
        self.view = QTextBrowser(self)
        self.view.setOpenExternalLinks(False)
        self.view.setStyleSheet(
            f"background-color: {self.palette.surface_bg}; border: none; border-radius: 8px;"
        )
        self.view.setHtml(
            f"<p style='color:{self.palette.text_muted}'>"
            "Select a source to preview it here.</p>"
        )
        layout.addWidget(self.view, 1)

        self.play_btn = QPushButton(self)
        self.play_btn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.play_btn.setVisible(False)
        self.play_btn.clicked.connect(self._on_play)
        layout.addWidget(self.play_btn)

        self.usage_last = QLabel("Last request: —", self)
        self.usage_last.setWordWrap(True)
        self.usage_last.setStyleSheet(f"color: {self.palette.text_muted}; font-size: 11px;")
        self.usage_last.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.usage_last)
        self.usage_session = QLabel("Session: —", self)
        self.usage_session.setWordWrap(True)
        self.usage_session.setStyleSheet(f"color: {self.palette.link}; font-size: 11px;")
        self.usage_session.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.usage_session)

    def set_last_usage(self, text: str) -> None:
        self.usage_last.setText(f"Last request: {text}")

    def set_session_usage(self, text: str) -> None:
        self.usage_session.setText(f"Session: {text}")

    def show_result(self, result) -> None:
        record = result.record
        stamp = (
            f"<p><b>Timestamp:</b> {record.timestamp_start:.2f}s"
            + (f" – {record.timestamp_end:.2f}s" if record.timestamp_end else "")
            + "</p>"
            if record.timestamp_start is not None
            else ""
        )
        body = (
            f"<h3>{html.escape(record.filename)}</h3>"
            f"<p><b>Score:</b> {result.score:.3f} &nbsp; <b>Type:</b> {html.escape(record.record_type)}</p>"
            f"<p><b>Path:</b> {html.escape(record.source_path)}</p>"
            f"{stamp}"
            f"<hr/>"
            f"<pre style='white-space:pre-wrap'>{html.escape(record.content or '')}</pre>"
        )
        self.view.setHtml(body)
        if record.file_type == "video" and record.timestamp_start is not None:
            self._arm_play(Path(record.source_path), record.timestamp_start)
        else:
            self._hide_play()

    def show_file_record(self, record) -> None:
        """Preview a file selected in the ingestion table (queue state + content)."""
        path = Path(getattr(record, "path", "") or "")
        name = path.name or str(getattr(record, "path", ""))
        suffix = path.suffix.lower()
        status = html.escape(str(getattr(record, "status", "") or ""))
        kind = html.escape(str(getattr(record, "kind", "") or ""))
        error = html.escape(str(getattr(record, "error", "") or ""))
        output = html.escape(str(getattr(record, "output", "") or ""))

        header = (
            f"<h3>{html.escape(name)}</h3>"
            f"<p><b>Status:</b> {status} &nbsp; <b>Kind:</b> {kind}</p>"
            f"<p><b>Path:</b> {html.escape(str(path))}</p>"
        )
        if error:
            header += f"<p><b>Error:</b> {error}</p>"
        if output:
            header += f"<p style='color:{self.palette.text_muted}'>{output}</p>"
        header += "<hr/>"

        if not path.is_file():
            self.view.setHtml(
                header
                + f"<p style='color:{self.palette.text_muted}'>File not found at this path "
                "(inbox files are moved through processing/processed folders).</p>"
            )
            self._hide_play()
            return

        if suffix in _VIDEO_SUFFIXES:
            self.view.setHtml(
                header
                + f"<p style='color:{self.palette.text_muted}'>Video file — "
                "ask a question in chat, then click a timestamped source link to preview it here.</p>"
            )
            self._hide_play()
            return

        if suffix in _IMAGE_SUFFIXES:
            url = QUrl.fromLocalFile(str(path)).toString()
            self.view.setHtml(
                header + f"<img src='{html.escape(url)}' width='280' />"
            )
            self._hide_play()
            return

        if suffix in _TEXT_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")[:_PREVIEW_MAX_CHARS]
            except OSError as exc:
                self.view.setHtml(
                    header
                    + f"<p style='color:{self.palette.text_muted}'>Cannot read file: "
                    f"{html.escape(str(exc))}</p>"
                )
                self._hide_play()
                return
            self.view.setHtml(header + f"<pre style='white-space:pre-wrap'>{html.escape(text)}</pre>")
            self._hide_play()
            return

        self.view.setHtml(
            header
            + f"<p style='color:{self.palette.text_muted}'>No inline preview for this file type. "
            "Ask a question in chat, then click a source link to see its indexed passages here.</p>"
        )
        self._hide_play()

    def clear(self) -> None:
        self.view.setHtml(
            f"<p style='color:{self.palette.text_muted}'>"
            "Select a source to preview it here.</p>"
        )
        self._hide_play()

    def _hide_play(self) -> None:
        self._play_path = None
        self._play_ts = None
        self.play_btn.setVisible(False)

    def _arm_play(self, path: Path, timestamp: float | None) -> None:
        if not path.is_file():
            self._hide_play()
            return
        self._play_path = path
        self._play_ts = timestamp
        has_vlc = find_vlc() is not None
        if timestamp:
            label = f"Play from {timestamp:.0f}s"
            if not has_vlc:
                label += " (default player, from start)"
        else:
            label = "Open video"
            if not has_vlc:
                label += " (default player)"
        self.play_btn.setText(label)
        self.play_btn.setToolTip(
            "Opens in VLC at this timestamp when VLC is installed; "
            "otherwise opens in the default player from the start."
        )
        self.play_btn.setVisible(True)

    def _on_play(self) -> None:
        path, timestamp = self._play_path, self._play_ts
        if path is None or not path.is_file():
            QMessageBox.warning(self, "Cannot play", "The video file no longer exists at:\n" f"{path}")
            return
        try:
            vlc = find_vlc()
            print(f"[play] ts={timestamp} vlc={vlc}", flush=True)
            if vlc is not None:
                # --no-one-instance: a running VLC would otherwise take over the
                # new launch and silently drop --start-time (starts from 0).
                argv = [vlc, "--no-one-instance", "--start-time",
                        str(int(timestamp or 0)), str(path)]
                print(f"[play] argv={argv}", flush=True)
                subprocess.Popen(argv)
            elif sys.platform.startswith("win"):
                print(f"[play] no VLC found; default player for {path}", flush=True)
                os.startfile(str(path))  # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Cannot play", f"Could not open the video:\n{exc}")
