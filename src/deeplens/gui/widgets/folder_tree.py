"""Folder tree sidebar widget for directory management and scope filtering."""

from __future__ import annotations

import os
from pathlib import Path
from PySide6.QtCore import QModelIndex, Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHeaderView,
    QPushButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)
import structlog

from deeplens.gui.widgets.progress_bar import IndexingProgressWidget

logger = structlog.get_logger(__name__)


class FolderTreeWidget(QWidget):
    """Left sidebar widget allowing registering folders and setting query boundaries."""

    folder_added = Signal(str)
    folder_removed = Signal(str)
    selection_changed = Signal(list)  # list of absolute paths

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Add Folder Button
        self.btn_add = QPushButton("+ Add Folder", self)
        self.btn_add.clicked.connect(self._on_add_clicked)
        layout.addWidget(self.btn_add)

        # Tree view with checkboxes
        self.tree_view = QTreeView(self)
        self.tree_view.setHeaderHidden(True)
        self.tree_view.setAcceptDrops(True)
        self.tree_view.setDragEnabled(False)
        self.tree_view.setDropIndicatorShown(True)
        self.tree_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        
        # Override drag drop events
        self.tree_view.dragEnterEvent = self.dragEnterEvent
        self.tree_view.dropEvent = self.dropEvent

        self.model = QStandardItemModel(self)
        self.tree_view.setModel(self.model)
        self.model.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.tree_view)

        # Indexing Progress Widget
        self.progress_widget = IndexingProgressWidget(self)
        layout.addWidget(self.progress_widget)

        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept drop if it contains folders."""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                p = Path(url.toLocalFile())
                if p.exists() and p.is_dir():
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        """Process dropped folders."""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                p = Path(url.toLocalFile())
                if p.exists() and p.is_dir():
                    self.add_registered_folder(str(p.resolve()))
            event.acceptProposedAction()

    def _on_add_clicked(self) -> None:
        """Open file dialog to choose folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Index")
        if folder:
            self.add_registered_folder(folder)

    def add_registered_folder(self, folder_path: str) -> None:
        """Add folder to tree and trigger indexing."""
        # Check if already present
        for row in range(self.model.rowCount()):
            item = self.model.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == folder_path:
                return

        logger.info("folder_tree.add_folder", path=folder_path)
        
        # Root node for the folder
        item = QStandardItem(os.path.basename(folder_path))
        item.setData(folder_path, Qt.ItemDataRole.UserRole)
        item.setCheckable(True)
        item.setCheckState(Qt.CheckState.Checked)
        item.setEditable(False)

        # Populate child files/dirs (collapsed by default)
        self._populate_item(item, Path(folder_path))
        
        self.model.appendRow(item)
        self.folder_added.emit(folder_path)
        self._emit_selection()

    def _populate_item(self, parent_item: QStandardItem, path: Path) -> None:
        """Scan child elements and add sub-nodes."""
        try:
            for entry in path.iterdir():
                if entry.name.startswith("."):
                    continue
                
                child_item = QStandardItem(entry.name)
                child_item.setData(str(entry.resolve()), Qt.ItemDataRole.UserRole)
                child_item.setCheckable(True)
                child_item.setCheckState(Qt.CheckState.Checked)
                child_item.setEditable(False)

                if entry.is_dir():
                    self._populate_item(child_item, entry)
                
                parent_item.appendRow(child_item)
        except Exception:
            pass

    def _on_item_changed(self, item: QStandardItem) -> None:
        """Sync child checkboxes and trigger filter updates."""
        # Block signals temporarily to prevent infinite recursion
        self.model.itemChanged.disconnect(self._on_item_changed)

        state = item.checkState()
        # Set all child items to match parent state
        self._set_children_state(item, state)
        
        # If checked, ensure parent is checked too
        if state == Qt.CheckState.Checked:
            parent = item.parent()
            while parent:
                parent.setCheckState(Qt.CheckState.Checked)
                parent = parent.parent()

        self.model.itemChanged.connect(self._on_item_changed)
        self._emit_selection()

    def _set_children_state(self, parent_item: QStandardItem, state: Qt.CheckState) -> None:
        for i in range(parent_item.rowCount()):
            child = parent_item.child(i)
            child.setCheckState(state)
            if child.hasChildren():
                self._set_children_state(child, state)

    def _emit_selection(self) -> None:
        """Gather all checklisted root paths and emit."""
        selected_roots = []
        for row in range(self.model.rowCount()):
            item = self.model.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                selected_roots.append(item.data(Qt.ItemDataRole.UserRole))
        self.selection_changed.emit(selected_roots)

    def update_progress(self, progress_data: dict) -> None:
        """Forward indexing status to the progress widget."""
        self.progress_widget.update_progress(progress_data)
