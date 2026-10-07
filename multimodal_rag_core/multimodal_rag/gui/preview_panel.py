from __future__ import annotations

import html
import os
import shutil
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QLabel, QMessageBox, QPushButton, QStyle, QTextBrowser, QVBoxLayout, QWidget

from multimodal_rag.config import get_settings
from multimodal_rag.gui.theme import get_palette

_TEXT_SUFFIXES = {".txt", ".md", ".csv", ".json", ".log", ".rst", ".xml", ".html", ".htm"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".gif"}
_VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg"}
_AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
# Office docs have no cheap direct reader here (no python-docx/openpyxl import
# in the GUI layer) — like PDFs, they preview from indexed passages.
_OFFICE_SUFFIXES = {".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".rtf"}
_PREVIEW_MAX_CHARS = 4000
# Indexed passages shown for files with no direct inline preview (same text
# search actually retrieves — no live re-extraction, so no mojibake and no
# extra format dependencies in the GUI process).
_INDEXED_PREVIEW_CHUNKS = 3
_INDEXED_PREVIEW_CHARS = 1500


def _looks_garbled(text: str) -> bool:
    """True if the text is likely font-encoding mojibake, not real content.

    Only counts characters that never occur in legitimate text in ANY language:
    U+FFFD replacements, C0/C1 controls (outside tab/newline), and Private Use
    Area codepoints (where custom PDF font encodings surface). Non-Latin
    scripts pass untouched — this test cannot misfire on them.
    """
    if not text:
        return True
    # Threshold note: real extractions often carry a few damaged chars (e.g.
    # curly quotes surfacing as U+FFFD — see the joke-book chunks, ~2% bad —
    # and must still preview). Only heavy damage (>15%) counts as mojibake.
    bad = 0
    for char in text:
        point = ord(char)
        if char == "�":
            bad += 3  # never legitimate
        elif point < 32 and char not in ("\t", "\n", "\r"):
            bad += 2
        elif 0xE000 <= point <= 0xF8FF or 0xF0000 <= point <= 0xFFFFF:
            bad += 1
    return bad / len(text) > 0.15


def _indexed_passages_html(
    chunks: list | None, label: str, muted: str
) -> tuple[str | None, bool]:
    """(html, garbled) for the first indexed passages.

    Garbled (font-encoding mojibake) passages are skipped; ``garbled`` is True
    when chunks exist but none were readable — callers then explain instead of
    showing nonsense.
    """
    items = list(chunks or [])[:_INDEXED_PREVIEW_CHUNKS]
    texts = []
    garbled = False
    for chunk in items:
        if isinstance(chunk, dict):
            content = chunk.get("content", "") or ""
        else:
            content = getattr(chunk, "content", "") or ""
        content = content.strip()[:_INDEXED_PREVIEW_CHARS]
        if not content:
            continue
        if _looks_garbled(content):
            garbled = True
            continue
        texts.append(content)
    if not texts:
        return None, garbled
    joined = "<hr/>".join(
        f"<pre style='white-space:pre-wrap'>{html.escape(text)}</pre>" for text in texts
    )
    return (
        f"<p style='color:{muted}'>{label} — first {len(texts)} indexed "
        f"passage(s), same text search retrieves.</p>" + joined,
        False,
    )


def _garbled_hint(name: str, muted: str) -> str:
    """Explanation shown instead of font-encoding mojibake."""
    return (
        f"<p style='color:{muted}'>{html.escape(name)} — its indexed text looks "
        "garbled (PDF custom font encoding, or scanned pages ingested without OCR). "
        "Search hits from this file will read the same way. To fix: install Docling "
        "with <i>DOC_USE_OCR=true</i>, then File → Reset index and re-ingest. "
        "The file itself is fine — click the Path link above to open it.</p>"
    )


def _path_link(path_str: str, link_color: str | None = None) -> str:
    """Clickable file-path HTML: opens the file with its default app.

    Empty paths render as an em-dash (no dead link). The color is applied
    inline because QTextBrowser's default anchor blue is unreadable on the
    dark themes — callers pass their palette's link color.
    """
    if not (path_str or "").strip():
        return "—"
    url = QUrl.fromLocalFile(path_str).toString()
    style = f" style=\"color:{link_color};\"" if link_color else ""
    return (
        f"<a href=\"{html.escape(url, quote=True)}\"{style}>"
        f"{html.escape(path_str)}</a>"
    )


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
        # CRITICAL: openLinks must also be False. With it left at its True
        # default, clicking a file:// link BOTH emits anchorClicked (our
        # handler opens the file externally — correct) AND navigates the
        # browser itself to the file URL, replacing the pane content with the
        # raw file bytes (readable for .txt, garbage mojibake for video).
        # False means anchor clicks only emit anchorClicked: open externally,
        # pane untouched.
        self.view.setOpenLinks(False)
        self.view.anchorClicked.connect(self._on_anchor)
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

    def _on_anchor(self, url: QUrl) -> None:
        """Open preview-pane file links with the system default app.

        QTextBrowser has setOpenExternalLinks(False), so clicks arrive here
        instead of opening silently. Missing files (inbox items moved through
        processing/processed folders) get a warning, not silence.
        """
        try:
            path_str = url.toLocalFile() if url.isLocalFile() else url.toString()
        except Exception:
            path_str = ""
        if url.isLocalFile() and path_str and not Path(path_str).exists():
            QMessageBox.warning(
                self,
                "File not found",
                "The file no longer exists at:\n"
                f"{path_str}\n\n"
                "(Inbox files move through processing/processed folders after indexing.)",
            )
            return
        QDesktopServices.openUrl(url)

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
            f"<p><b>Path:</b> {_path_link(record.source_path, self.palette.link)}</p>"
            f"{stamp}"
            f"<hr/>"
        )
        content = record.content or ""
        if not content.strip():
            body += (
                f"<p style='color:{self.palette.text_muted}'>(empty passage)</p>"
            )
        elif _looks_garbled(content):
            # Same guard as show_file_record: whisper hallucinations on
            # non-speech audio (music/silence) can index unreadable passages.
            # Explain instead of rendering mojibake.
            body += _garbled_hint(record.filename, self.palette.text_muted)
        else:
            body += f"<pre style='white-space:pre-wrap'>{html.escape(content)}</pre>"
        self.view.setHtml(body)
        if record.file_type == "video":
            # Summary records carry timestamp_start=None (file-level) and the
            # first keyframe/audio segment carries 0.0 (falsy but valid). Both
            # mean "from the start" and must still arm the play button.
            self._arm_play(Path(record.source_path), record.timestamp_start)
        else:
            self._hide_play()

    def show_file_record(self, record, chunks: list | None = None) -> None:
        """Preview a file selected in the ingestion table (queue state + content).

        ``chunks`` are indexed passages for the file (fetched by the caller via
        BackendService.preview_chunks); used for the PDF branch so the pane
        shows exactly what search retrieves instead of a live re-extraction.
        """
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
            f"<p><b>Path:</b> {_path_link(str(path), self.palette.link)}</p>"
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
            passages, garbled = _indexed_passages_html(
                chunks, "Video transcript / keyframe captions", self.palette.text_muted
            )
            self.view.setHtml(
                header
                + f"<p style='color:{self.palette.text_muted}'>Video file — "
                "use the Play button below to open it from the start, or "
                "ask a question in chat, then click a timestamped source link to preview it here.</p>"
                + (passages or "")
                + (_garbled_hint(name, self.palette.text_muted) if garbled else "")
            )
            self._arm_play(path, None)
            return

        if suffix in _IMAGE_SUFFIXES:
            url = QUrl.fromLocalFile(str(path)).toString()
            self.view.setHtml(
                header + f"<img src='{html.escape(url)}' width='280' />"
            )
            self._hide_play()
            return

        if suffix == ".pdf" or suffix in _OFFICE_SUFFIXES or suffix in _AUDIO_SUFFIXES:
            passages, garbled = _indexed_passages_html(chunks, name, self.palette.text_muted)
            if passages:
                self.view.setHtml(
                    header + passages
                    + f"<p style='color:{self.palette.text_muted}'>Click the Path link "
                    "above to open the full file.</p>"
                )
            elif garbled:
                self.view.setHtml(header + _garbled_hint(name, self.palette.text_muted))
            else:
                self.view.setHtml(
                    header
                    + f"<p style='color:{self.palette.text_muted}'>{html.escape(name)} — no indexed "
                    "passages yet (still queued, failed, or skipped). "
                    "Click the Path link above to open the file.</p>"
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

        passages, garbled = _indexed_passages_html(chunks, name, self.palette.text_muted)
        if passages:
            self.view.setHtml(
                header + passages
                + f"<p style='color:{self.palette.text_muted}'>Click the Path link "
                "above to open the full file.</p>"
            )
        elif garbled:
            self.view.setHtml(header + _garbled_hint(name, self.palette.text_muted))
        else:
            self.view.setHtml(
                header
                + f"<p style='color:{self.palette.text_muted}'>No inline preview for this file type "
                "and no indexed passages yet. Ask a question in chat, then click a source link "
                "to see its indexed passages here — or open the file via the Path link above.</p>"
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
        # NOTE: use `is not None`, not truthiness — 0.0 is a valid timestamp
        # (first frame/segment) and must show "Play from 0s", not "Open video".
        if timestamp is not None:
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
                        str(int(0 if timestamp is None else timestamp)), str(path)]
                print(f"[play] argv={argv}", flush=True)
                from multimodal_rag.utils.subproc import hidden_kwargs

                subprocess.Popen(argv, **hidden_kwargs())
            elif sys.platform.startswith("win"):
                print(f"[play] no VLC found; default player for {path}", flush=True)
                os.startfile(str(path))  # noqa: S606
            elif sys.platform == "darwin":
                from multimodal_rag.utils.subproc import hidden_kwargs

                subprocess.Popen(["open", str(path)], **hidden_kwargs())
            else:
                from multimodal_rag.utils.subproc import hidden_kwargs

                subprocess.Popen(["xdg-open", str(path)], **hidden_kwargs())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Cannot play", f"Could not open the video:\n{exc}")
