"""MainWindow container class for the DeepLens desktop layout."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)
import structlog

from deeplens.config import AppMode, Settings, get_settings
from deeplens.core.chat import configure_summary_budget
from deeplens.core.factory import BackendFactory
from deeplens.core.models import IndexingProgress, SearchResponse
from deeplens.ingestion.task_queue import IngestionQueue
from deeplens.search.graph import SearchPipeline

# Import widgets
from deeplens.gui.widgets.chat_panel import ChatPanel
from deeplens.gui.widgets.folder_tree import FolderTreeWidget
from deeplens.gui.widgets.preview_panel import PreviewPanel
from deeplens.gui.widgets.settings_dialog import SettingsDialog

logger = structlog.get_logger(__name__)


class SearchWorker(QThread):
    """Background worker thread for running RAG searches to prevent GUI lockup."""

    finished = Signal(SearchResponse)

    def __init__(
        self,
        pipeline: SearchPipeline,
        query: str,
        folder_filter: str | None = None,
        file_type_filter: str | None = None
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.query = query
        self.folder_filter = folder_filter
        self.file_type_filter = file_type_filter

    def run(self) -> None:
        """Run search pipeline inside thread and emit result."""
        # Create an event loop inside this thread to run async pipelines
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Run the search
            result = loop.run_until_complete(
                self.pipeline.search(
                    query=self.query,
                    folder_filter=self.folder_filter,
                    file_type_filter=self.file_type_filter,
                )
            )
            self.finished.emit(result)
        except Exception as e:
            logger.error("search_worker.error", error=str(e))
        finally:
            loop.close()


class IngestWorker(QThread):
    """Background worker thread running the ingestion queue loop."""

    progress = Signal(dict)

    def __init__(self, queue: IngestionQueue) -> None:
        super().__init__()
        self.queue = queue

    def run(self) -> None:
        """Run the asyncio loop to process items."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Override progress callback to emit PySide6 signals
        def _prog_callback(data: IndexingProgress) -> None:
            # Emit dictionary payload thread-safely
            self.progress.emit({
                "folder_path": data.folder_path,
                "total_files": data.total_files,
                "processed_files": data.processed_files,
                "current_file": data.current_file,
                "status": data.status,
                "eta_seconds": data.eta_seconds,
            })

        self.queue.progress_callback = _prog_callback

        try:
            # Start queue workers
            loop.run_until_complete(self.queue.start())
            # Keep loop alive while thread is running
            while not self.isInterruptionRequested():
                loop.run_until_complete(asyncio.sleep(0.1))
            # Stop workers
            loop.run_until_complete(self.queue.stop())
        except Exception as e:
            logger.error("ingest_worker.error", error=str(e))
        finally:
            loop.close()


class MainWindow(QMainWindow):
    """DeepLens main window container containing the 3-panel UI."""

    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()
        self.setWindowTitle("DeepLens — Privacy-First Semantic File Explorer")
        self.resize(1100, 700)
        
        # Core backend pipelines
        self.repo = None
        self.embedder = None
        self.chat = None
        self.pipeline = None
        self.ingest_queue = None
        
        self.ingest_thread = None
        self.search_thread = None
        
        # Folder filter constraint
        self.folder_filters: list[str] = []

        self._init_backend()
        self._init_ui()
        self._init_systray()
        self._wire_signals()
        self._init_shortcuts()

    def _init_backend(self) -> None:
        """Instantiate swappable repositories and pipelines."""
        try:
            factory = BackendFactory(self.settings)
            self.repo = factory.create_repository()
            self.embedder = factory.create_embedding_engine()
            self.chat = factory.create_chat_engine()
            
            # Initializing repositories and embedding weights is slow.
            # Usually we want to run async, but on startup we execute a quick runner:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running (e.g. within an async stack)
                asyncio.create_task(self._async_init())
            else:
                loop.run_until_complete(self._async_init())

        except Exception as e:
            logger.error("main_window.init_backend.failed", error=str(e))
            QMessageBox.critical(
                self,
                "Backend Failure",
                f"Failed to initialize swappable backend interfaces:\n{str(e)}\n\n"
                "Please verify your configuration and restart."
            )

    async def _async_init(self) -> None:
        await self.repo.initialize()
        await self.embedder.initialize()
        await self.chat.initialize()

        # Size the summarization character budget to the chat model's real
        # context window so long-context models use more of each file.
        summary_chars = configure_summary_budget(self.settings, self.chat)
        logger.info("main_window.summary_budget", summary_max_chars=summary_chars)
        

        self.pipeline = SearchPipeline(self.repo, self.embedder, self.chat, self.settings)
        self.ingest_queue = IngestionQueue(
            self.repo, self.embedder, self.settings, chat_engine=self.chat
        )
        
        # Launch background ingestion worker loop
        self.ingest_thread = IngestWorker(self.ingest_queue)
        self.ingest_thread.progress.connect(self._on_ingestion_progress)
        self.ingest_thread.start()

        # Update stats
        self._update_db_stats()

    def _init_ui(self) -> None:
        # Create core splitter
        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.setCentralWidget(self.splitter)

        # 1. Left Sidebar (Folder Tree)
        self.sidebar = FolderTreeWidget(self)
        self.splitter.addWidget(self.sidebar)

        # 2. Center Panel (Chat search)
        self.chat_panel = ChatPanel(self)
        self.splitter.addWidget(self.chat_panel)

        # 3. Right Sidebar (Preview)
        self.preview_panel = PreviewPanel(self)
        self.splitter.addWidget(self.preview_panel)

        # Splitter initial sizes
        self.splitter.setSizes([260, 520, 320])

        # Status Bar
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        
        # Indicator badge
        mode_text = "LOCAL (Offline)" if self.settings.mode == AppMode.LOCAL else "CLOUD (Accelerated)"
        self.status_bar.showMessage(f"Mode: {mode_text}  |  Ready")

        # Menu bar
        self._init_menu()

    def _init_menu(self) -> None:
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu("File")
        
        act_add_folder = QAction("Add Folder...", self)
        act_add_folder.triggered.connect(self.sidebar._on_add_clicked)
        file_menu.addAction(act_add_folder)

        file_menu.addSeparator()

        act_settings = QAction("Settings...", self)
        act_settings.triggered.connect(self._on_settings_triggered)
        file_menu.addAction(act_settings)

        act_exit = QAction("Exit", self)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # View Menu
        view_menu = menubar.addMenu("View")
        
        act_toggle_left = QAction("Toggle Left Panel", self)
        act_toggle_left.triggered.connect(lambda: self._toggle_panel(0))
        view_menu.addAction(act_toggle_left)

        act_toggle_right = QAction("Toggle Right Panel", self)
        act_toggle_right.triggered.connect(lambda: self._toggle_panel(2))
        view_menu.addAction(act_toggle_right)

        # Help Menu
        help_menu = menubar.addMenu("Help")
        act_about = QAction("About DeepLens", self)
        act_about.triggered.connect(self._on_about_triggered)
        help_menu.addAction(act_about)

    def _init_systray(self) -> None:
        """Create a system tray icon for running headless in the background."""
        self.tray = QSystemTrayIcon(self)
        # Check system icon
        self.tray.setIcon(QIcon.fromTheme("system-search", QIcon()))
        
        tray_menu = QMenu(self)
        show_action = QAction("Show MainWindow", self)
        show_action.triggered.connect(self.showNormal)
        tray_menu.addAction(show_action)

        exit_action = QAction("Exit App", self)
        exit_action.triggered.connect(self._on_exit_from_tray)
        tray_menu.addAction(exit_action)

        self.tray.setContextMenu(tray_menu)
        self.tray.show()

    def _wire_signals(self) -> None:
        # Chat queries
        self.chat_panel.query_submitted.connect(self._on_search_submitted)
        self.chat_panel.result_clicked.connect(self.preview_panel.preview_file)

        # Sidebar selections
        self.sidebar.folder_added.connect(self._on_folder_added)
        self.sidebar.selection_changed.connect(self._on_scope_changed)

    def _init_shortcuts(self) -> None:
        # Ctrl+K focus search
        self.sc_search = QShortcut(QKeySequence("Ctrl+K"), self)
        self.sc_search.activated.connect(self.chat_panel.input_field.setFocus)

        # Ctrl+, settings
        self.sc_settings = QShortcut(QKeySequence("Ctrl+,"), self)
        self.sc_settings.activated.connect(self._on_settings_triggered)

    def _toggle_panel(self, idx: int) -> None:
        widget = self.splitter.widget(idx)
        if widget.isVisible():
            widget.hide()
        else:
            widget.show()

    def _on_folder_added(self, folder_path: str) -> None:
        """Queue folder for indexing."""
        if self.ingest_queue:
            # We submit to async queue inside main thread
            loop = asyncio.get_event_loop()
            loop.create_task(self.ingest_queue.submit(Path(folder_path)))

    def _on_scope_changed(self, selected_paths: list) -> None:
        """Set parent folder filter constraint list."""
        self.folder_filters = [str(p) for p in selected_paths]

    @Slot(dict)
    def _on_ingestion_progress(self, progress_data: dict) -> None:
        """Pass progress stats to the sidebar UI."""
        self.sidebar.update_progress(progress_data)
        
        # On complete, update DB record counts
        if progress_data.get("status") == "completed":
            self._update_db_stats()

    def _update_db_stats(self) -> None:
        """Retrieve total record metrics and display on statusbar."""
        if self.repo:
            async def _get() -> None:
                stats = await self.repo.get_stats()
                tot = stats.get("total_records", 0)
                files = stats.get("total_files", 0)
                
                mode_text = "LOCAL (Offline)" if self.settings.mode == AppMode.LOCAL else "CLOUD (Accelerated)"
                self.status_bar.showMessage(
                    f"Mode: {mode_text}  |  Indexed Files: {files} ({tot} chunks)  |  Ready"
                )
            
            asyncio.get_event_loop().create_task(_get())

    def _on_search_submitted(self, query: str) -> None:
        """Trigger background search worker."""
        if not self.pipeline:
            self.chat_panel.add_assistant_message("Error: RAG search pipeline not initialized yet.", [])
            return

        self.chat_panel.show_loading()

        # Select first folder filter for search constraint if present (or handle multiple in repo)
        folder_filter = self.folder_filters[0] if self.folder_filters else None
        file_type_filter = self.chat_panel.get_active_type_filter()

        # Create worker
        self.search_thread = SearchWorker(
            pipeline=self.pipeline,
            query=query,
            folder_filter=folder_filter,
            file_type_filter=file_type_filter
        )
        self.search_thread.finished.connect(self._on_search_completed)
        self.search_thread.start()

    @Slot(SearchResponse)
    def _on_search_completed(self, response: SearchResponse) -> None:
        """Receive pipeline search updates and append assistant response bubble."""
        self.chat_panel.hide_loading()
        
        # Convert SearchResults to list of dicts for GUI cards
        results_data = []
        for r in response.results:
            results_data.append({
                "score": r.score,
                "filename": r.record.filename,
                "absolute_path": r.record.absolute_path,
                "content": r.record.content,
            })

        self.chat_panel.add_assistant_message(response.answer, results_data)
        
        # update stats
        self._update_db_stats()

    def _on_settings_triggered(self) -> None:
        dlg = SettingsDialog(self)
        if dlg.exec() == SettingsDialog.DialogCode.Accepted:
            # Config was updated. Prompt restart to re-bind backends.
            QMessageBox.information(
                self,
                "Settings Saved",
                "Your settings have been saved. Please restart DeepLens to re-initialize the backend stacks."
            )

    def _on_about_triggered(self) -> None:
        QMessageBox.about(
            self,
            "About DeepLens",
            "<h3>DeepLens v0.1.0</h3>"
            "<p>Privacy-First Multi-Modal Semantic File Explorer.</p>"
            "<p>Built using PySide6, LangGraph, LanceDB, and Google Gemini.</p>"
            "<p>Developed for high-performance offline indexing and CV showcase.</p>"
        )

    def _on_exit_from_tray(self) -> None:
        self.tray.hide()
        QApplication.quit()

    def closeEvent(self, event) -> None:
        """Intercept close to run cleanups."""
        logger.info("main_window.closing")
        
        # Stop background threads
        if self.ingest_thread:
            self.ingest_thread.requestInterruption()
            self.ingest_thread.wait()

        # Release database locks
        if self.repo:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.repo.close())
            else:
                loop.run_until_complete(self.repo.close())
                
        event.accept()
