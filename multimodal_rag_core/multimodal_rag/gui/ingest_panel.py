from __future__ import annotations

from PySide6.QtCore import QItemSelection, QItemSelectionModel, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from multimodal_rag.config import get_settings
from multimodal_rag.gui.theme import get_palette


class IngestPanel(QWidget):
    add_folder = Signal()
    add_files = Signal()
    open_inbox = Signal()
    refresh = Signal()
    file_selected = Signal(object)
    remove_file = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._displayed: list = []
        self._last_sig: list = []
        self.palette = get_palette(get_settings().GUI_THEME.value)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(QLabel("Ingestion", self))

        row1 = QHBoxLayout()
        self.btn_add = QPushButton("Add Folder...")
        self.btn_add.clicked.connect(self.add_folder.emit)
        self.btn_add_files = QPushButton("Add Files...")
        self.btn_add_files.clicked.connect(self.add_files.emit)
        self.btn_open_inbox = QPushButton("Open Inbox")
        self.btn_open_inbox.clicked.connect(self.open_inbox.emit)
        row1.addWidget(self.btn_add)
        row1.addWidget(self.btn_add_files)
        row1.addWidget(self.btn_open_inbox)
        row1.addStretch()
        layout.addLayout(row1)

        self.hint = QLabel("")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet(f"color: {self.palette.warn}; font-size: 11px;")
        layout.addWidget(self.hint)

        self.table = QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(["File", "Status", "Error"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.itemSelectionChanged.connect(self._on_selection)
        layout.addWidget(self.table, 1)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh.emit)
        self.btn_remove = QPushButton("Remove selected")
        self.btn_remove.setToolTip(
            "Remove the selected file(s) from the list and delete their indexed records. "
            "Tip: Ctrl+click / Shift+click to select several rows. "
            "The source files on disk are kept."
        )
        self.btn_remove.setEnabled(False)
        self.btn_remove.clicked.connect(self._on_remove_clicked)
        bottom = QHBoxLayout()
        bottom.addWidget(self.btn_refresh)
        bottom.addWidget(self.btn_remove)
        layout.addLayout(bottom)

    def set_warning(self, message: str) -> None:
        self.hint.setText(message)

    def set_records(self, records: list, stale_ids: set[int] | None = None) -> None:
        stale = stale_ids or set()
        ordered = list(reversed(records[-500:]))
        sig = [
            (record.id, record.path, record.status, record.error or "", record.id in stale)
            for record in ordered
        ]
        if sig == self._last_sig:
            # Nothing changed: leave the table (and the user's in-progress
            # Ctrl+click multi-selection) completely untouched. Rebuilding here
            # every 2s is what collapsed multi-selections down to one row.
            self.btn_remove.setEnabled(bool(self.selected_records()))
            return
        self._last_sig = sig
        # Snapshot the selection before tearing the rows down.
        selected_paths: set[str] = set()
        for row in self.table.selectionModel().selectedRows() if self.table.selectionModel() else []:
            if 0 <= row.row() < len(self._displayed):
                selected_paths.add(self._displayed[row.row()].path)
        self._displayed = ordered
        restore_rows: list[int] = []
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(0)
            for record in self._displayed:
                row = self.table.rowCount()
                self.table.insertRow(row)
                name = record.path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                self.table.setItem(row, 0, QTableWidgetItem(name))
                status_item = QTableWidgetItem(record.status)
                if record.id in stale:
                    status_item.setText(f"{record.status} (not in index)")
                    status_item.setForeground(QBrush(QColor(self.palette.warn)))
                self.table.setItem(row, 1, status_item)
                self.table.setItem(row, 2, QTableWidgetItem(record.error or ""))
                if record.path in selected_paths:
                    restore_rows.append(row)
            # Restore in ONE selection operation: per-row selectRow() calls
            # collapse an ExtendedSelection back to a single row.
            if restore_rows:
                model = self.table.selectionModel()
                tbl_model = self.table.model()
                last_col = max(0, self.table.columnCount() - 1)
                sel = QItemSelection()
                for row in restore_rows:
                    sel.select(tbl_model.index(row, 0), tbl_model.index(row, last_col))
                model.select(sel, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
        finally:
            self.table.blockSignals(False)
        self.btn_remove.setEnabled(bool(self.selected_records()))

    def _on_selection(self) -> None:
        records = self.selected_records()
        self.btn_remove.setEnabled(bool(records))
        # Preview follows a single selection; a multi-select leaves the
        # current preview untouched instead of guessing which file to show.
        if len(records) == 1:
            self.file_selected.emit(records[0])

    def _on_remove_clicked(self) -> None:
        records = self.selected_records()
        if records:
            self.remove_file.emit(records)

    def selected_records(self):
        """Currently selected queue records (possibly empty)."""
        model = self.table.selectionModel()
        if model is None:
            return []
        out = []
        for row in model.selectedRows():
            index = row.row()
            if 0 <= index < len(self._displayed):
                out.append(self._displayed[index])
        return out

    def selected_record(self):
        """Single selected queue record, or None (kept for compatibility)."""
        records = self.selected_records()
        return records[0] if len(records) == 1 else None
