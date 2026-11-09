from __future__ import annotations

import logging
import re
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl, Slot
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QSplitter,
    QStatusBar,
)

from multimodal_rag.config import SearchMode, Settings, get_settings
from multimodal_rag.gui.chat_panel import ChatPanel
from multimodal_rag.gui.ingest_panel import IngestPanel
from multimodal_rag.gui.preview_panel import PreviewPanel
from multimodal_rag.gui.services import BackendService
from multimodal_rag.gui.settings_dialog import SettingsDialog
from multimodal_rag.gui.theme import app_qss, get_palette
from multimodal_rag.gui.workers import (
    BackfillWorker,
    InitWorker,
    SearchWorker,
    WarmupWorker,
)

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    HOT_QUERY_KEYS = frozenset(
        {
            "SEARCH_MODE",
            "SEARCH_TOP_K",
            "ANSWER_TOP_K",
            "KEYWORD_EXPANSION",
            "KEYWORD_EXPANSION_TERMS",
            "TYPO_CORRECTION",
        }
    )
    RESTART_REQUIRED_KEYS = frozenset(
        {
            "LANCEDB_DIR",
            "STATE_DB_PATH",
            "DATA_DIR",
            "INBOX_DIR",
            "PROCESSING_DIR",
            "PROCESSED_DIR",
            "FAILED_DIR",
            "VLM_CACHE_DIR",
        }
    )

    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()
        self.service = BackendService(self.settings)
        self._results: dict[str, object] = {}
        self._result_seq = 0
        self._timer: QTimer | None = None
        self._stale_ids: set[int] = set()
        self._init_worker: InitWorker | None = None
        self._warmup_worker: WarmupWorker | None = None
        self._search_worker: SearchWorker | None = None
        self._backfill_worker: BackfillWorker | None = None
        self._models_warmed = False
        self._env_path = Path(self.settings.app_root) / ".env"
        self._env_stamp = self._current_env_stamp()
        self._settings_dialog_open = False
        self._pending_settings: tuple | None = None

        self.setWindowTitle("Multi-Engine Hybrid RAG")
        self.resize(1200, 760)
        self._build_ui()
        self._env_timer = QTimer(self)
        self._env_timer.timeout.connect(self._check_env_changes)
        self._env_timer.start(1000)
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
        self.act_backfill = QAction("Backfill embeddings...", self)
        self.act_backfill.setToolTip(
            "Embed records that were indexed while embeddings were disabled"
        )
        self.act_backfill.triggered.connect(self._on_backfill_embeddings)
        act_ffmpeg = QAction("Download ffmpeg...", self)
        act_ffmpeg.setToolTip("Fetch a static ffmpeg for video audio extraction")
        act_ffmpeg.triggered.connect(self._download_ffmpeg)
        act_settings = QAction("Settings...", self)
        act_settings.triggered.connect(self._on_settings)
        act_exit = QAction("Exit", self)
        act_exit.triggered.connect(self.close)
        for action in (
            act_add,
            act_add_files,
            act_reset,
            self.act_backfill,
            act_ffmpeg,
            act_exit,
        ):
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
        self.ingest_panel.reindex_missing.connect(self._on_reindex_missing)
        self.ingest_panel.file_selected.connect(self._on_file_selected)
        self.ingest_panel.remove_file.connect(self._on_remove_file)

        # Make the inbox location discoverable.
        self.ingest_panel.btn_open_inbox.setToolTip(
            f"Open the inbox folder in your file manager:\n{self.settings.INBOX_DIR}"
        )

    def _start_backend(self) -> None:
        self._models_warmed = False
        self._warmup_worker = None
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
            if "EMBEDDING_MODEL_NOT_CACHED" in error or "EMBEDDING_MODEL_UNSUPPORTED" in error:
                self._on_embedding_model_blocked(error)
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
        sync = self.service.engine_sync or {}
        if sync.get("copied"):
            self.chat_panel.add_assistant_message(
                f"Copied {sync['copied']} vector record(s) from {sync['from']} "
                f"to {sync['to']} after the DB engine switch."
            )
        if self.service.engine_sync_error:
            self.chat_panel.add_assistant_message(
                "Could not sync the previous DB engine automatically: "
                f"{self.service.engine_sync_error}\n"
                "Use Reindex for any rows marked processed (not in index)."
            )
        if not self.service.embedder_ready:
            init_error = getattr(self.service, "init_error", None)
            if init_error and (
                "EMBEDDING_MODEL_NOT_CACHED" in init_error
                or "EMBEDDING_MODEL_UNSUPPORTED" in init_error
            ):
                self._on_embedding_model_blocked(init_error)
            elif init_error and not self.service.embedding_disabled:
                self.chat_panel.add_assistant_message(
                    "Embeddings unavailable, so keyword search will be used: "
                    f"{init_error}"
                )
            self._enter_keyword_only()
            self._check_index_fingerprint()
            self._refresh(reconcile=True)
            return
        self.ingest_panel.set_warning("")
        # Keep querying disabled until both models are loaded and warmed up.
        self.chat_panel.set_backend_state(
            False, "Loading and warming up local models..."
        )
        self._warmup_worker = WarmupWorker(self.service)
        self._warmup_worker.warmed.connect(self._on_warmup_done)
        self._warmup_worker.start()
        self._check_index_fingerprint()
        self._refresh(reconcile=True)

    def _enter_keyword_only(self) -> None:
        """Keep the app usable when no embedding engine is configured."""
        self._models_warmed = True
        self.chat_panel.set_mode("keyword")
        self.chat_panel.set_backend_state(True)
        self.chat_panel.focus_input()
        self.ingest_panel.set_warning(
            "No embedding engine — indexing stores chunks/summaries for keyword "
            "search only. Enable embeddings later and use File → Backfill embeddings."
        )
        if not getattr(self, "_greeted", False):
            self._greeted = True
            self.chat_panel.add_assistant_message(
                "Keyword-only search is ready. Files are indexed and searchable "
                "without embeddings; vector/hybrid modes are disabled until you "
                "configure an embedding model. Existing unembedded records can be "
                "embedded later with File → Backfill embeddings."
            )

    @Slot(str)
    def _on_warmup_done(self, error: str) -> None:
        if not self.service.embedder_ready:
            return
        if error:
            self._models_warmed = False
            self.statusBar().showMessage("Model warmup failed")
            self.chat_panel.set_backend_state(
                False, "Model warmup failed — search is disabled."
            )
            self.chat_panel.add_assistant_message(
                "Search is disabled because model warmup failed:\n"
                f"{error}\n\nFix the embedding/reranker setup, then restart."
            )
            return
        self._models_warmed = True
        self.chat_panel.set_backend_state(True)
        self.chat_panel.focus_input()
        if not getattr(self, "_greeted", False):
            self._greeted = True
            self.chat_panel.add_assistant_message(
                "Ready. Ask a question about your files, or add a files/folder "
                "on the left to index one."
            )
        self._notify_missing_vectors()

    def _notify_missing_vectors(self) -> None:
        try:
            missing = int(self.service.stats().get("repo", {}).get("missing_vectors", 0) or 0)
        except Exception:  # noqa: BLE001
            return
        if missing:
            self.chat_panel.add_assistant_message(
                f"{missing} indexed record(s) have no embeddings yet. "
                "Use File → Backfill embeddings to enable vector/hybrid search "
                "for them."
            )

    def _on_embedding_model_blocked(self, error: str) -> None:
        """Local embedding checkpoint missing: show download steps, stay open.

        The model is NOT bundled (hundreds of MB) — first use downloads it
        from HuggingFace into the HF cache. The full hint (license + login +
        pre-download command) is also posted to chat so it stays copyable
        after the dialog closes.
        """
        if "EMBEDDING_MODEL_UNSUPPORTED" in error:
            suffix = (
                "Upgrade transformers in the app's venv as the message describes, "
                "then restart. You can still edit Settings without restarting."
            )
        else:
            suffix = (
                "Pick a smaller variant (Settings → EMBEDDING_MODEL → "
                "...:text-only 270M) or a smaller MRL dimension (256/128) if disk/RAM "
                "is tight, then restart with internet access. You can still edit "
                "Settings without restarting."
            )
        QMessageBox.warning(
            self,
            "Embedding model not downloaded",
            f"{error}\n\n{suffix}",
        )
        self.ingest_panel.set_warning(
            "Embedding model not downloaded — see the download steps in chat, "
            "then restart with internet access."
        )
        self.chat_panel.set_backend_state(
            False, "Embedding model not downloaded — see chat for steps."
        )
        self.chat_panel.add_assistant_message(f"Backend blocked: {error}")

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
        box.addButton("Decide later", QMessageBox.ButtonRole.RejectRole)
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
            self.service.restart_ingestion()
        self.preview_panel.clear()
        self._refresh()
        # The backend stayed up, but only re-enable queries if the models were
        # already warmed successfully (or keyword-only mode is active).
        self.chat_panel.set_backend_state(self._models_warmed)
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
            missing = int(repo.get("missing_vectors", 0) or 0)
            missing_note = f" | missing embeddings: {missing}" if missing else ""
            self.statusBar().showMessage(
                f"Mode: {self.settings.SYSTEM_MODE.value} | "
                f"{repo.get('total_files', 0)} files / {repo.get('total_records', 0)} records"
                f"{missing_note} | queue: {queue}"
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
        # Re-check the queue against the current index without deleting anything.
        self._refresh(reconcile=True)

    def _on_reindex_missing(self) -> None:
        self._refresh(reconcile=True)
        stale = set(self._stale_ids)
        if not stale:
            self.chat_panel.add_assistant_message(
                "No processed files are missing from the current index."
            )
            return
        requeued = self.service.reindex_missing(stale)
        self._stale_ids = set()
        self._refresh(reconcile=True)
        self.chat_panel.add_assistant_message(
            f"Requeued {requeued} file(s) missing from the current index. "
            "Indexing starts in the background; this is expected after switching "
            "DB engines without mrag-db-copy."
        )

    def _on_backfill_embeddings(self) -> None:
        if self._backfill_worker is not None:
            return
        if not self.service.embedder_ready:
            self.chat_panel.add_assistant_message(
                "Backfill needs an embedding model. Set EMBEDDING_PROVIDER=LOCAL "
                "(or GOOGLE with a key), restart, then try again."
            )
            return
        answer = QMessageBox.question(
            self,
            "Backfill embeddings",
            "Embed all indexed records that currently have no vectors?\n\n"
            "Indexing pauses while this runs. Existing vectors are kept.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.service.stop_worker()
        self.chat_panel.set_backend_state(False, "Backfilling embeddings...")
        self.act_backfill.setEnabled(False)
        self._backfill_worker = BackfillWorker(self.service)
        self._backfill_worker.done.connect(self._on_backfill_done)
        self._backfill_worker.start()

    @Slot(object, str)
    def _on_backfill_done(self, result, error: str) -> None:
        self._backfill_worker = None
        self.act_backfill.setEnabled(True)
        self.service.restart_ingestion()
        self.chat_panel.set_backend_state(True)
        self.chat_panel.focus_input()
        if error:
            self.chat_panel.add_assistant_message(f"Embedding backfill failed: {error}")
        else:
            self.chat_panel.add_assistant_message(
                f"Embedding backfill complete: {result.get('records', 0)} record(s) "
                f"in {result.get('files', 0)} file(s) embedded."
            )
        self._refresh(reconcile=True)

    def _update_session_usage(self) -> None:
        from multimodal_rag.utils.token_tracker import TokenTracker

        self.preview_panel.set_session_usage(
            TokenTracker.format_usage(self.service.tracker.snapshot())
        )

    def _backend_ready_for_enqueue(self) -> bool:
        if self.service.queue is not None and self._pending_settings is None:
            return True
        QMessageBox.information(
            self,
            "Backend is loading",
            "The backend is still initializing, switching databases, or "
            "applying saved settings. Wait for it to finish, then add files again.",
        )
        return False

    def _on_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select a folder to index")
        if not folder or not self._backend_ready_for_enqueue():
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
        if not files or not self._backend_ready_for_enqueue():
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
        if self._search_worker is not None:
            return
        if not self._models_warmed:
            return
        mode = self.chat_panel.active_mode()
        if (
            not self.service.embedder_ready
            and mode not in (SearchMode.KEYWORD.value, SearchMode.AUTO.value)
        ):
            self.chat_panel.add_assistant_message(
                "Vector search needs an embedding model. Choose keyword (or auto) "
                "mode, or configure embeddings and restart."
            )
            return
        self.chat_panel.show_loading()
        self.chat_panel.set_busy(True)
        self._last_with_answer = self.chat_panel.answer_enabled()
        worker = SearchWorker(
            self.service,
            query,
            self.chat_panel.active_mode(),
            self.settings.SEARCH_TOP_K,
            with_answer=self.chat_panel.answer_enabled(),
        )
        self._search_worker = worker
        worker.completed.connect(self._on_search_done)
        worker.finished.connect(self._on_search_finished)
        worker.start()

    @Slot()
    def _on_search_finished(self) -> None:
        worker = self.sender()
        if worker is self._search_worker:
            self._search_worker = None
        self.chat_panel.set_busy(False)
        self.chat_panel.focus_input()

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
            self.preview_panel.clear("Search failed; no source preview.")
            return
        # Keep every previous answer's cards resolvable. Reusing rank keys
        # (0, 1, 2, ...) and clearing this map made clicks on older answers
        # show the newest result, or nothing when the new answer had fewer
        # hits. A session-unique key preserves timestamps/chunks per card.
        cards = []
        for index, result in enumerate(results):
            self._result_seq += 1
            key = f"result-{self._result_seq}"
            self._results[key] = result
            # 1-based rank doubles as the [N] citation number: format_context
            # numbers results in this same pool order, so card N == citation [N].
            cards.append({"key": key, "result": result, "rank": index + 1})
        if answer:
            text = answer
            # AUTO provenance: say which evidence the citations come from, so
            # a hybrid retry is visible instead of masquerading as chunk output.
            requested = getattr(getattr(self, "_search_worker", None), "mode", "")
            if requested == "auto" and self.service.last_mode_used == "hybrid":
                text = (
                    "_Auto: chunk was inconclusive, retried with hybrid._\n\n" + text
                )
            elif requested == "auto" and self.service.last_mode_used == "keyword":
                text = (
                    "_Auto: embeddings unavailable, used keyword search._\n\n" + text
                )
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
        # The answer cites the context by [N]; surface cards in the same order
        # the answer mentions them (e.g. [3] then [2] -> card [3] first), so the
        # visual list matches the reading order. Uncited cards keep rank order
        # after a divider. Deduplicate preserving first-mention order.
        cited = list(dict.fromkeys(int(n) for n in re.findall(r"\[(\d+)\]", answer or "")))
        # Keep the preview pane in sync with the answer the user just received.
        # The UI puts cited cards first (in answer mention order), so preview the
        # same top card; with no citations, fall back to the best-ranked result.
        top_result = None
        for rank in cited:
            if 1 <= rank <= len(results):
                top_result = results[rank - 1]
                break
        if top_result is None and results:
            top_result = results[0]
        if top_result is not None:
            self.preview_panel.show_result(top_result)
        else:
            self.preview_panel.clear("No source matched this query.")
        self.chat_panel.add_assistant_message(text, cards, cited)

    def _on_result_clicked(self, key: str) -> None:
        result = self._results.get(key)
        if result is not None:
            self.preview_panel.show_result(result)

    def _on_file_selected(self, record: object) -> None:
        try:
            chunks = self.service.preview_chunks(record)
        except Exception:  # noqa: BLE001
            chunks = []
        self.preview_panel.show_file_record(record, chunks)

    def _on_remove_file(self, record: object) -> None:
        records = list(record) if isinstance(record, list) else [record]
        if not records:
            return
        if len(records) == 1:
            prompt = (
                f"Remove `{getattr(records[0], 'path', str(records[0]))}` from the list "
                "and delete its indexed records?\n\nThe source file on disk is kept."
            )
        else:
            prompt = (
                f"Remove {len(records)} files from the list and delete their "
                "indexed records?\n\nThe source files on disk are kept."
            )
        answer = QMessageBox.question(
            self,
            "Remove file" if len(records) == 1 else f"Remove {len(records)} files",
            prompt,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            total_vectors = 0
            for item in records:
                result = self.service.remove_file(item)
                total_vectors += result["vectors"]
            self.preview_panel.clear()
            self._refresh()
            if len(records) == 1:
                name = getattr(records[0], "path", str(records[0]))
                text = (
                    f"Removed `{name}`: {total_vectors} indexed record(s) deleted, "
                    "queue entry removed."
                )
            else:
                text = (
                    f"Removed {len(records)} file(s): {total_vectors} indexed "
                    "record(s) deleted, queue entries removed."
                )
            self.chat_panel.add_assistant_message(text)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Remove failed", str(exc))

    def ensure_ffmpeg(self) -> None:
        """Offer to fetch ffmpeg once at startup when it is not installed."""
        from multimodal_rag.utils.ffmpeg_tool import find_ffmpeg

        resolved = find_ffmpeg()
        logger.info("ffmpeg check: %s", resolved or "not found")
        if resolved is not None or getattr(self, "_ffmpeg_worker", None) is not None:
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
        before = self.settings
        dialog = SettingsDialog(self)
        self._settings_dialog_open = True
        try:
            accepted = dialog.exec() == SettingsDialog.DialogCode.Accepted
        finally:
            self._settings_dialog_open = False
        if not accepted:
            return
        self._env_stamp = self._current_env_stamp()
        try:
            new_settings = Settings()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self,
                "Settings could not be reloaded",
                f"The saved .env is not valid:\n{exc}\n\n"
                "Fix it in the Settings dialog or a text editor.",
            )
            return
        changed = self._changed_setting_fields(before, new_settings)
        if not changed:
            self.statusBar().showMessage("No settings changed.", 4000)
            return
        self._queue_settings_apply(new_settings, changed, "Settings")

    def _current_env_stamp(self):
        try:
            stat = self._env_path.stat()
        except OSError:
            return None
        return (stat.st_mtime_ns, stat.st_size)

    @staticmethod
    def _changed_setting_fields(before, after) -> set[str]:
        old = before.model_dump()
        new = after.model_dump()
        return {
            key for key in old.keys() | new.keys() if old.get(key) != new.get(key)
        }

    def _check_env_changes(self) -> None:
        if self._settings_dialog_open:
            return
        stamp = self._current_env_stamp()
        if stamp == self._env_stamp:
            return
        self._env_stamp = stamp
        if stamp is None:
            return
        try:
            new_settings = Settings()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not reload settings from %s: %s", self._env_path, exc)
            self.statusBar().showMessage(f".env reload failed: {exc}", 8000)
            return
        if new_settings == self.settings:
            return
        changed = self._changed_setting_fields(self.settings, new_settings)
        if changed:
            self._queue_settings_apply(new_settings, changed, ".env")

    def _queue_settings_apply(self, new_settings, changed: set[str], source: str) -> None:
        if changed & self.RESTART_REQUIRED_KEYS:
            self.statusBar().showMessage(
                "Storage-path settings changed; restart the app to apply them.",
                8000,
            )
            return
        if self._settings_busy():
            self._pending_settings = (new_settings, changed, source)
            self.statusBar().showMessage(
                "Settings saved; applying when the current task finishes.", 6000
            )
            QTimer.singleShot(500, self._maybe_apply_pending_settings)
            return
        self._apply_settings(new_settings, changed, source)

    def _settings_busy(self) -> bool:
        workers = (
            self._init_worker,
            self._warmup_worker,
            self._search_worker,
            self._backfill_worker,
        )
        return any(worker is not None and worker.isRunning() for worker in workers)

    def _maybe_apply_pending_settings(self) -> None:
        if self._pending_settings is None:
            return
        if self._settings_busy():
            QTimer.singleShot(500, self._maybe_apply_pending_settings)
            return
        new_settings, changed, source = self._pending_settings
        self._pending_settings = None
        self._apply_settings(new_settings, changed, source)

    def _apply_settings(
        self, new_settings, changed: set[str], source: str
    ) -> None:
        get_settings.cache_clear()
        if "GUI_THEME" in changed:
            self._apply_theme(new_settings.GUI_THEME.value)
        if changed <= self.HOT_QUERY_KEYS | {"GUI_THEME"}:
            self.settings = new_settings
            self.service.settings = new_settings
            if self.service.context is not None:
                self.service.context.settings = new_settings
            if "SEARCH_MODE" in changed:
                self.chat_panel.set_mode(new_settings.SEARCH_MODE.value)
            self._update_path_tooltips()
            self.statusBar().showMessage(f"Settings applied from {source}.", 6000)
            return
        if "ACTIVE_DB_ENGINE" in changed:
            engine = new_settings.ACTIVE_DB_ENGINE
            self.statusBar().showMessage(
                f"Switching database to {engine.value if engine else 'default'}..."
            )
        self._reload_backend(new_settings)
        self.statusBar().showMessage(
            f"Settings applied from {source}; backend reloaded.", 8000
        )

    def _reload_backend(self, new_settings) -> None:
        logger.info(
            "Reloading backend: %s -> %s, embeddings=%s",
            self.settings.ACTIVE_DB_ENGINE,
            new_settings.ACTIVE_DB_ENGINE,
            new_settings.EMBEDDING_PROVIDER.value,
        )
        self._models_warmed = False
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        old_service = self.service
        try:
            old_service.stop_worker()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not stop previous ingestion worker: %s", exc)
        repository = old_service.context.repository if old_service.context else None
        if repository is not None:
            try:
                repository.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not close previous repository: %s", exc)
        self.settings = new_settings
        self.service = BackendService(new_settings)
        self._stale_ids = set()
        self._update_path_tooltips()
        self.chat_panel.set_mode(new_settings.SEARCH_MODE.value)
        self._start_backend()

    def _apply_theme(self, theme: str) -> None:
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(app_qss(get_palette(theme)))
        for panel in (self.ingest_panel, self.chat_panel, self.preview_panel):
            apply_theme = getattr(panel, "apply_theme", None)
            if apply_theme is not None:
                apply_theme(theme)

    def _update_path_tooltips(self) -> None:
        self.ingest_panel.btn_open_inbox.setToolTip(
            f"Open the inbox folder in your file manager:\n{self.settings.INBOX_DIR}"
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._backfill_worker is not None and self._backfill_worker.isRunning():
            QMessageBox.information(
                self,
                "Backfill in progress",
                "Embedding backfill is still running. Wait for it to finish before "
                "closing the application.",
            )
            event.ignore()
            return
        try:
            self._env_timer.stop()
            self.service.stop_worker()
        except Exception:  # noqa: BLE001
            pass
        event.accept()
