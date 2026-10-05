from __future__ import annotations

import logging
import re

from PySide6.QtCore import Qt, QTimer, QUrl, Slot
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QSplitter,
    QStatusBar,
)

from multimodal_rag.config import get_settings
from multimodal_rag.gui.chat_panel import ChatPanel
from multimodal_rag.gui.ingest_panel import IngestPanel
from multimodal_rag.gui.preview_panel import PreviewPanel
from multimodal_rag.gui.services import BackendService
from multimodal_rag.gui.settings_dialog import SettingsDialog
from multimodal_rag.gui.workers import InitWorker, SearchWorker

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()
        self.service = BackendService(self.settings)
        self._results: dict[str, object] = {}
        self._timer: QTimer | None = None
        self._stale_ids: set[int] = set()

        self.setWindowTitle("Multi-Engine Hybrid RAG")
        self.resize(1200, 760)
        self._build_ui()
        self._start_backend()

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.setCentralWidget(splitter)

        self.ingest_panel = IngestPanel(self)
        self.chat_panel = ChatPanel(self)
        self.preview_panel = PreviewPanel(self)
        splitter.addWidget(self.ingest_panel)
        splitter.addWidget(self.chat_panel)
        splitter.addWidget(self.preview_panel)
        splitter.setSizes([330, 570, 300])

        self.setStatusBar(QStatusBar(self))

        file_menu = self.menuBar().addMenu("File")
        act_add = QAction("Add Folder...", self)
        act_add.triggered.connect(self._on_add_folder)
        act_add_files = QAction("Add Files...", self)
        act_add_files.triggered.connect(self._on_add_files)
        act_reset = QAction("Reset index...", self)
        act_reset.triggered.connect(self._on_reset_index)
        act_ffmpeg = QAction("Download ffmpeg...", self)
        act_ffmpeg.setToolTip("Fetch a static ffmpeg for video audio extraction")
        act_ffmpeg.triggered.connect(self._download_ffmpeg)
        act_settings = QAction("Settings...", self)
        act_settings.triggered.connect(self._on_settings)
        act_exit = QAction("Exit", self)
        act_exit.triggered.connect(self.close)
        for action in (act_add, act_add_files, act_reset, act_ffmpeg, act_exit):
            file_menu.addAction(action)
        # Top-level "Settings" in the same menu-bar row as File, so it is
        # immediately visible (no extra toolbar row).
        self.menuBar().addAction(act_settings)

        self.chat_panel.query_submitted.connect(self._on_query)
        self.chat_panel.result_clicked.connect(self._on_result_clicked)
        self.ingest_panel.add_folder.connect(self._on_add_folder)
        self.ingest_panel.add_files.connect(self._on_add_files)
        self.ingest_panel.open_inbox.connect(self._on_open_inbox)
        self.ingest_panel.refresh.connect(self._on_refresh)
        self.ingest_panel.file_selected.connect(self._on_file_selected)
        self.ingest_panel.remove_file.connect(self._on_remove_file)

        # Make the inbox location discoverable.
        self.ingest_panel.btn_open_inbox.setToolTip(
            f"Open the inbox folder in your file manager:\n{self.settings.INBOX_DIR}"
        )

    def _start_backend(self) -> None:
        self.statusBar().showMessage("Initializing backend...")
        self.chat_panel.set_backend_state(
            False, "Waiting for the database and index to load..."
        )
        self._init_worker = InitWorker(self.service)
        self._init_worker.ready.connect(self._on_ready)
        self._init_worker.start()

    @Slot(str)
    def _on_ready(self, error: str) -> None:
        if error:
            self.statusBar().showMessage("Backend error")
            self.chat_panel.set_backend_state(
                False, "Backend failed to start — see the message above."
            )
            if "has embedding" in error and "EMBEDDING_DIMENSION" in error:
                self._on_dimension_blocked(error)
                return
            QMessageBox.critical(
                self,
                "Backend failed",
                f"Could not initialize the backend:\n{error}\n\n"
                "Check .env, Postgres/Docker, and the selected engine. "
                "You can still edit Settings and restart.",
            )
            self.chat_panel.add_assistant_message(f"Backend failed to start: {error}")
            return
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(2000)
        if not self.service.embedder_ready:
            self.ingest_panel.set_warning(
                "No embedding engine configured — indexing and search are disabled until "
                "you set keys and restart."
            )
            self.chat_panel.set_backend_state(
                False, "Embeddings not configured — set keys in Settings and restart."
            )
            self.chat_panel.add_assistant_message(
                "Backend is up, but embeddings are not configured. Open Settings, set "
                "GOOGLE_API_KEY (or EMBEDDING_PROVIDER=LOCAL), then restart."
            )
        else:
            self.chat_panel.set_backend_state(True)
            if not getattr(self, "_greeted", False):
                self._greeted = True
                self.chat_panel.add_assistant_message(
                    "Ready. Ask a question about your files, or add a files/folder on the left to index one."
                )
        self._check_index_fingerprint()
        self._refresh(reconcile=True)

    def _on_dimension_blocked(self, error: str) -> None:
        """Startup blocked by vector-dimension change: offer rebuild + restart."""
        answer = QMessageBox.question(
            self,
            "Embedding dimension changed",
            f"Backend cannot start:\n{error}\n\n"
            "Rebuild the table at the new dimension and clear the queue, "
            "then restart the backend? Old vectors will be deleted "
            "(source files on disk are kept; you will re-ingest them).",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self.chat_panel.add_assistant_message(
                f"Backend blocked: {error}\nPoint EMBEDDING_MODEL/DIMENSION back "
                "at the old values, or restart and choose Yes to rebuild."
            )
            return
        try:
            result = self.service.reset_index()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Rebuild failed", str(exc))
            return
        self.chat_panel.add_assistant_message(
            f"Table rebuilt: {result['vectors']} old vector(s) and "
            f"{result['queue']} queue entr(ies) deleted. Restarting backend..."
        )
        self._start_backend()

    def _check_index_fingerprint(self) -> None:
        try:
            result = self.service.check_fingerprint()
        except Exception as exc:  # noqa: BLE001
            logger.warning("fingerprint check failed: %s", exc)
            return
        if result["status"] != "mismatch":
            return
        details = "\n".join(f"• {line}" for line in result["diffs"])
        box = QMessageBox(self)
        box.setWindowTitle("Index settings changed")
        box.setText(
            "The stored index was built with different index-time settings:\n\n"
            f"{details}\n\n"
            "Queries against it may be wrong or fail.\n"
            "If you changed nothing, your index predates fingerprinting and is "
            "probably fine — choose Trust below."
        )
        btn_reset = box.addButton(
            "Reset & start fresh", QMessageBox.ButtonRole.DestructiveRole
        )
        btn_trust = box.addButton(
            "Trust current index", QMessageBox.ButtonRole.AcceptRole
        )
        btn_later = box.addButton("Decide later", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(btn_trust)
        box.exec()
        clicked = box.clickedButton()
        if clicked == btn_reset:
            self._do_reset_index()
        elif clicked == btn_trust:
            self.service.stamp_fingerprint()
            self.chat_panel.add_assistant_message(
                "Current settings stamped as this index's fingerprint. "
                "Future startups will warn only on real changes."
            )
        else:
            self.chat_panel.add_assistant_message(
                "Index settings differ from the stored index "
                f"({'; '.join(result['diffs'])}). Queries may be wrong until you "
                "use File → Reset index... and re-ingest."
            )

    def _do_reset_index(self) -> None:
        self.service.stop_worker()
        try:
            result = self.service.reset_index()
        finally:
            self.service.restart_worker()
        self.preview_panel.clear()
        self._refresh()
        # The backend stayed up, so querying can be re-enabled immediately.
        self.chat_panel.set_backend_state(self.service.embedder_ready)
        self.chat_panel.add_assistant_message(
            f"Index reset: {result['vectors']} vector record(s) and "
            f"{result['queue']} queue entr(ies) deleted. Re-ingest your files to start fresh."
        )

    def _on_reset_index(self) -> None:
        answer = QMessageBox.question(
            self,
            "Reset index",
            "Delete ALL indexed records and the queue list, and start fresh?\n\n"
            "Source files on disk are kept. This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._do_reset_index()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Reset failed", str(exc))

    def _refresh(self, reconcile: bool = False) -> None:
        try:
            stats = self.service.stats()
            repo = stats.get("repo", {})
            queue = stats.get("queue", {})
            self.statusBar().showMessage(
                f"Mode: {self.settings.SYSTEM_MODE.value} | "
                f"{repo.get('total_files', 0)} files / {repo.get('total_records', 0)} records | "
                f"queue: {queue}"
            )
            records = self.service.queue_records()
            if reconcile:
                # Cross-check the queue against the vector store: a row marked
                # processed whose vectors are gone (deleted elsewhere) is stale.
                self._stale_ids = self.service.stale_record_ids(records)
            self.ingest_panel.set_records(records, stale_ids=self._stale_ids)
            self._update_session_usage()
        except Exception as exc:  # noqa: BLE001
            logger.warning("refresh failed: %s", exc)

    def _on_refresh(self) -> None:
        # Sync the list with the actual index: drop entries whose vectors are gone.
        self._refresh(reconcile=True)
        stale = set(self._stale_ids)
        if stale:
            removed = self.service.remove_missing(stale)
            self._stale_ids = set()
            self._refresh(reconcile=True)
            self.chat_panel.add_assistant_message(
                f"Removed {removed} queue entr(ies) missing from the index "
                "(vectors were deleted elsewhere; source files kept)."
            )

    def _update_session_usage(self) -> None:
        from multimodal_rag.utils.token_tracker import TokenTracker

        self.preview_panel.set_session_usage(
            TokenTracker.format_usage(self.service.tracker.snapshot())
        )

    def _on_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select a folder to index")
        if not folder:
            return
        try:
            count = self.service.enqueue(folder)
            self.chat_panel.add_assistant_message(
                f"Queued {count} file(s) from `{folder}`. Indexing runs in the background."
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Enqueue failed", str(exc))

    def _supported_files_filter(self) -> str:
        from multimodal_rag.ingestion.state import EXTENSION_KINDS

        exts = sorted(
            {
                ext.lstrip(".")
                for kind, kind_exts in EXTENSION_KINDS.items()
                if self.settings.allows_kind(kind)
                for ext in kind_exts
            }
        )
        patterns = " ".join(f"*.{ext}" for ext in exts)
        return f"Supported files ({patterns});;All files (*)"

    def _on_add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select file(s) to index", "", self._supported_files_filter()
        )
        if not files:
            return
        try:
            count = 0
            for file in files:
                count += self.service.enqueue(file)
            self.chat_panel.add_assistant_message(
                f"Queued {count} file(s). Indexing runs in the background."
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Enqueue failed", str(exc))

    def _on_open_inbox(self) -> None:
        try:
            self.settings.ensure_data_dirs()
        except Exception:  # noqa: BLE001
            pass
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.settings.INBOX_DIR))

    def _on_query(self, query: str) -> None:
        self.chat_panel.show_loading()
        self._last_with_answer = self.chat_panel.answer_enabled()
        self._search_worker = SearchWorker(
            self.service,
            query,
            self.chat_panel.active_mode(),
            self.settings.SEARCH_TOP_K,
            with_answer=self.chat_panel.answer_enabled(),
        )
        self._search_worker.completed.connect(self._on_search_done)
        self._search_worker.start()

    @Slot(object, object, str, str)
    def _on_search_done(self, results, groups, answer: str, error: str) -> None:
        from multimodal_rag.utils.token_tracker import TokenTracker

        self.chat_panel.hide_loading()
        self.preview_panel.set_last_usage(
            TokenTracker.format_usage(self.service.last_request_usage)
            if self.service.last_request_usage
            else "—"
        )
        self._update_session_usage()
        if error:
            self.chat_panel.add_assistant_message(f"Search error: {error}")
            return
        self._results = {}
        cards = []
        for index, result in enumerate(results):
            key = str(index)
            self._results[key] = result
            # 1-based rank doubles as the [N] citation number: format_context
            # numbers results in this same pool order, so card N == citation [N].
            cards.append({"key": key, "result": result, "rank": index + 1})
        if answer:
            text = answer
        elif results:
            lines = [f"{r.score:.3f} — {r.record.filename}" for r in results[:5]]
            text = "Top matches:\n" + "\n".join(lines)
            if getattr(self, "_last_with_answer", False):
                reason = self.service.last_answer_error
                if reason:
                    text += f"\n\n_Answer generation failed: {reason}_"
                else:
                    text += (
                        "\n\n_No synthesized answer: no text LLM configured. Set "
                        "TEXT_LLM_PROVIDER=GEMINI (reuse GOOGLE_API_KEY) or DEEPSEEK_API_KEY, "
                        "then restart._"
                    )
        else:
            text = "No matches found."
        if groups:
            text += "\n\nMatching files:\n" + "\n".join(f"- {g.filename}" for g in groups)
        # The answer cites the context by [N]; surface which cards it used so the
        # top-ranked (most query-relevant) card can't be mistaken for the source.
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer or "")}
        self.chat_panel.add_assistant_message(text, cards, cited)

    def _on_result_clicked(self, key: str) -> None:
        result = self._results.get(key)
        if result is not None:
            self.preview_panel.show_result(result)

    def _on_file_selected(self, record: object) -> None:
        self.preview_panel.show_file_record(record)

    def _on_remove_file(self, record: object) -> None:
        name = getattr(record, "path", str(record))
        answer = QMessageBox.question(
            self,
            "Remove file",
            f"Remove `{name}` from the list and delete its indexed records?\n\n"
            "The source file on disk is kept.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            result = self.service.remove_file(record)
            self.preview_panel.clear()
            self._refresh()
            self.chat_panel.add_assistant_message(
                f"Removed `{name}`: {result['vectors']} indexed record(s) deleted, "
                "queue entry removed."
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Remove failed", str(exc))

    def ensure_ffmpeg(self) -> None:
        """Offer to fetch ffmpeg once at startup when it is not installed."""
        from multimodal_rag.utils.ffmpeg_tool import ffmpeg_available

        if ffmpeg_available() or getattr(self, "_ffmpeg_worker", None) is not None:
            return
        answer = QMessageBox.question(
            self,
            "ffmpeg not found",
            "ffmpeg is needed to extract audio from video files.\n\n"
            "Download a static ffmpeg build next to the application now? "
            "(one-time, ~80 MB). Audio and document search work without it.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._download_ffmpeg()

    def _download_ffmpeg(self) -> None:
        from multimodal_rag.gui.workers import FfmpegDownloadWorker
        from multimodal_rag.utils.ffmpeg_tool import ffmpeg_available

        if ffmpeg_available():
            self.chat_panel.add_assistant_message("ffmpeg is already available.")
            return
        worker = getattr(self, "_ffmpeg_worker", None)
        if worker is not None and worker.isRunning():
            return

        dialog = QProgressDialog("Preparing ffmpeg download...", "Cancel", 0, 0, self)
        dialog.setWindowTitle("Downloading ffmpeg")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)

        worker = FfmpegDownloadWorker(self)
        self._ffmpeg_worker = worker
        worker.progress.connect(lambda d, t, m: self._on_ffmpeg_progress(dialog, d, t, m))
        worker.done.connect(lambda ok, info: self._on_ffmpeg_done(dialog, ok, info))
        dialog.canceled.connect(worker.requestInterruption)
        worker.start()

    @staticmethod
    def _on_ffmpeg_progress(
        dialog: QProgressDialog, downloaded: int, total: int, message: str
    ) -> None:
        if total > 0:
            dialog.setRange(0, 100)
            dialog.setValue(min(100, int(downloaded * 100 / total)))
        else:
            dialog.setRange(0, 0)  # indeterminate (connecting/extracting)
        dialog.setLabelText(message)

    def _on_ffmpeg_done(self, dialog: QProgressDialog, ok: bool, info: str) -> None:
        dialog.reset()
        dialog.close()
        self._ffmpeg_worker = None
        if ok:
            self.chat_panel.add_assistant_message(
                f"ffmpeg downloaded to `{info}`. Video audio extraction is now enabled."
            )
        elif info == "cancelled":
            self.chat_panel.add_assistant_message("ffmpeg download cancelled.")
        else:
            QMessageBox.warning(
                self,
                "ffmpeg download failed",
                f"Could not download ffmpeg:\n{info}\n\n"
                "You can install it system-wide and add it to PATH, then restart.",
            )

    def _on_settings(self) -> None:
        dialog = SettingsDialog(self)
        if dialog.exec() != SettingsDialog.DialogCode.Accepted:
            return
        choice = QMessageBox.question(
            self,
            "Restart required",
            "Settings saved to .env.\n\n"
            "The app needs to be restarted for the new settings to take effect.\n\n"
            "Exit app?.\n\n",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if choice == QMessageBox.StandardButton.Yes:
            self.close()

    def closeEvent(self, event) -> None:  # noqa: N802
        try:
            self.service.stop_worker()
        except Exception:  # noqa: BLE001
            pass
        event.accept()
