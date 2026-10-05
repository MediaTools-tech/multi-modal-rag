from __future__ import annotations

from PySide6.QtCore import Signal
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
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_selection)
        layout.addWidget(self.table, 1)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh.emit)
        self.btn_remove = QPushButton("Remove selected")
        self.btn_remove.setToolTip(
            "Remove the selected file from the list and delete its indexed records. "
            "The source file on disk is kept."
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
        # Preserve the selected file across the periodic refresh rebuild.
        selected_path: str | None = None
        for row in self.table.selectionModel().selectedRows() if self.table.selectionModel() else []:
            if 0 <= row.row() < len(self._displayed):
                selected_path = self._displayed[row.row()].path
                break
        self._displayed = list(reversed(records[-500:]))
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(0)
            for record in self._displayed:
                row = self.table.rowCount()
                self.table.insertRow(row)
                name = record.path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                self.table.setItem(row, 0, QTableWidgetItem(name))
                status_item = QTableWidgetItem(record.status)
                if stale_ids and record.id in stale_ids:
                    status_item.setText(f"{record.status} (not in index)")
                    status_item.setForeground(QBrush(QColor(self.palette.warn)))
                self.table.setItem(row, 1, status_item)
                self.table.setItem(row, 2, QTableWidgetItem(record.error or ""))
                if selected_path is not None and record.path == selected_path:
                    self.table.selectRow(row)
        finally:
            self.table.blockSignals(False)
        self.btn_remove.setEnabled(self.selected_record() is not None)

    def _on_selection(self) -> None:
        model = self.table.selectionModel()
        if model is None:
            return
        rows = model.selectedRows()
        if len(rows) != 1:
            self.btn_remove.setEnabled(False)
            return
        index = rows[0].row()
        if 0 <= index < len(self._displayed):
            self.btn_remove.setEnabled(True)
            self.file_selected.emit(self._displayed[index])
        else:
            self.btn_remove.setEnabled(False)

    def _on_remove_clicked(self) -> None:
        model = self.table.selectionModel()
        if model is None:
            return
        rows = model.selectedRows()
        if len(rows) != 1:
            return
        index = rows[0].row()
        if 0 <= index < len(self._displayed):
            self.remove_file.emit(self._displayed[index])

    def selected_record(self):
        """Currently selected queue record, or None."""
        model = self.table.selectionModel()
        if model is None:
            return None
        rows = model.selectedRows()
        if len(rows) != 1:
            return None
        index = rows[0].row()
        if 0 <= index < len(self._displayed):
            return self._displayed[index]
        return None
