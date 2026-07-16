"""Settings configuration dialog for modifying backend flags and keys."""

from __future__ import annotations

import os
from pathlib import Path
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
import structlog

from deeplens.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class SettingsDialog(QDialog):
    """Configuration dialog for the DeepLens application settings."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = get_settings()
        self.setWindowTitle("Settings")
        self.resize(500, 450)
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        self.tabs = QTabWidget(self)
        main_layout.addWidget(self.tabs)

        # ──── Tab 1: General ────
        self.tab_general = QWidget()
        self.tabs.addTab(self.tab_general, "General")
        layout_gen = QFormLayout(self.tab_general)

        self.combo_mode = QComboBox(self)
        self.combo_mode.addItems(["local", "cloud"])
        self.combo_mode.setCurrentText(self.settings.mode.value)
        layout_gen.addRow("Backend Mode:", self.combo_mode)

        self.line_data_dir = QLineEdit(str(self.settings.data_dir), self)
        btn_browse = QPushButton("Browse", self)
        btn_browse.clicked.connect(self._on_browse_data_dir)
        
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.line_data_dir)
        dir_layout.addWidget(btn_browse)
        layout_gen.addRow("Data Directory:", dir_layout)

        # ──── Tab 2: Local Mode Settings ────
        self.tab_local = QWidget()
        self.tabs.addTab(self.tab_local, "Local Mode")
        layout_loc = QFormLayout(self.tab_local)

        self.line_ollama_host = QLineEdit(self.settings.ollama_host, self)
        layout_loc.addRow("Ollama Host URL:", self.line_ollama_host)

        self.line_ollama_chat = QLineEdit(self.settings.ollama_chat_model, self)
        layout_loc.addRow("Chat Model:", self.line_ollama_chat)

        self.line_jina_clip = QLineEdit(self.settings.jina_clip_model, self)
        layout_loc.addRow("Jina CLIP model:", self.line_jina_clip)

        self.combo_whisper = QComboBox(self)
        self.combo_whisper.addItems(["tiny", "base", "small", "medium", "large"])
        self.combo_whisper.setCurrentText(self.settings.whisper_model_size)
        layout_loc.addRow("Whisper Model Size:", self.combo_whisper)

        # ──── Tab 3: Cloud Mode Settings ────
        self.tab_cloud = QWidget()
        self.tabs.addTab(self.tab_cloud, "Cloud Mode")
        layout_cld = QFormLayout(self.tab_cloud)

        # Retrieve API key securely or check env
        current_api_key = os.environ.get("GEMINI_API_KEY", "")
        try:
            import keyring
            key = keyring.get_password("deeplens", "gemini_api_key")
            if key:
                current_api_key = key
        except Exception:
            pass

        self.line_api_key = QLineEdit(current_api_key, self)
        self.line_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        layout_cld.addRow("Gemini API Key:", self.line_api_key)

        self.line_gemini_chat = QLineEdit(self.settings.gemini_chat_model, self)
        layout_cld.addRow("Gemini Chat Model:", self.line_gemini_chat)

        # Postgres settings
        self.line_pg_host = QLineEdit(self.settings.postgres_host, self)
        layout_cld.addRow("Postgres Host:", self.line_pg_host)

        self.spin_pg_port = QSpinBox(self)
        self.spin_pg_port.setRange(1, 65535)
        self.spin_pg_port.setValue(self.settings.postgres_port)
        layout_cld.addRow("Postgres Port:", self.spin_pg_port)

        self.line_pg_db = QLineEdit(self.settings.postgres_db, self)
        layout_cld.addRow("Postgres DB:", self.line_pg_db)

        self.line_pg_user = QLineEdit(self.settings.postgres_user, self)
        layout_cld.addRow("Postgres User:", self.line_pg_user)

        self.line_pg_pass = QLineEdit(self.settings.postgres_password, self)
        self.line_pg_pass.setEchoMode(QLineEdit.EchoMode.Password)
        layout_cld.addRow("Postgres Password:", self.line_pg_pass)

        # ──── Tab 4: Advanced Settings ────
        self.tab_adv = QWidget()
        self.tabs.addTab(self.tab_adv, "Advanced")
        layout_adv = QFormLayout(self.tab_adv)

        self.spin_chunk = QSpinBox(self)
        self.spin_chunk.setRange(100, 2000)
        self.spin_chunk.setValue(self.settings.chunk_size)
        layout_adv.addRow("Chunk Size (words):", self.spin_chunk)

        # Overlap Slider
        self.slide_overlap = QSlider(Qt.Orientation.Horizontal, self)
        self.slide_overlap.setRange(0, 50)  # 0% to 50%
        self.slide_overlap.setValue(int(self.settings.chunk_overlap * 100))
        self.lbl_overlap = QLabel(f"{self.slide_overlap.value()}%", self)
        self.slide_overlap.valueChanged.connect(self._on_overlap_changed)
        
        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(self.slide_overlap)
        overlap_layout.addWidget(self.lbl_overlap)
        layout_adv.addRow("Chunk Overlap:", overlap_layout)

        self.spin_workers = QSpinBox(self)
        self.spin_workers.setRange(1, 8)
        self.spin_workers.setValue(self.settings.ingestion_workers)
        layout_adv.addRow("Ingestion Workers:", self.spin_workers)

        self.spin_top_k = QSpinBox(self)
        self.spin_top_k.setRange(1, 50)
        self.spin_top_k.setValue(self.settings.search_top_k)
        layout_adv.addRow("Search Top K:", self.spin_top_k)

        # ──── Bottom Save/Cancel Row ────
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        
        btn_cancel = QPushButton("Cancel", self)
        btn_cancel.setObjectName("secondaryButton")
        btn_cancel.clicked.connect(self.reject)
        buttons_layout.addWidget(btn_cancel)

        btn_save = QPushButton("Save Settings", self)
        btn_save.clicked.connect(self._on_save_clicked)
        buttons_layout.addWidget(btn_save)

        main_layout.addLayout(buttons_layout)

    def _on_browse_data_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Choose Data Directory")
        if d:
            self.line_data_dir.setText(d)

    def _on_overlap_changed(self, value: int) -> None:
        self.lbl_overlap.setText(f"{value}%")

    def _on_save_clicked(self) -> None:
        """Write all parameters back to .env file and settings singleton."""
        mode = self.combo_mode.currentText()
        data_dir = self.line_data_dir.text()
        ollama_host = self.line_ollama_host.text()
        ollama_chat = self.line_ollama_chat.text()
        jina_clip = self.line_jina_clip.text()
        whisper_size = self.combo_whisper.currentText()
        
        api_key = self.line_api_key.text().strip()
        gemini_chat = self.line_gemini_chat.text()
        pg_host = self.line_pg_host.text()
        pg_port = self.spin_pg_port.value()
        pg_db = self.line_pg_db.text()
        pg_user = self.line_pg_user.text()
        pg_pass = self.line_pg_pass.text()

        chunk_size = self.spin_chunk.value()
        chunk_overlap = self.slide_overlap.value() / 100.0
        workers = self.spin_workers.value()
        top_k = self.spin_top_k.value()

        # Update secure Keyring for Gemini Key
        if api_key:
            try:
                import keyring
                keyring.set_password("deeplens", "gemini_api_key", api_key)
            except Exception as e:
                logger.warn("settings.keyring.failed", error=str(e))
                # Set as environment variable
                os.environ["GEMINI_API_KEY"] = api_key

        # Construct .env file lines
        env_content = f"""# Auto-generated DeepLens settings
DEEPLENS_MODE={mode}
DEEPLENS_DATA_DIR={data_dir}
DEEPLENS_INGESTION_WORKERS={workers}
DEEPLENS_CHUNK_SIZE={chunk_size}
DEEPLENS_CHUNK_OVERLAP={chunk_overlap}
DEEPLENS_SEARCH_TOP_K={top_k}

# Local Mode
OLLAMA_HOST={ollama_host}
OLLAMA_CHAT_MODEL={ollama_chat}
OLLAMA_REWRITER_MODEL={ollama_chat}
JINA_CLIP_MODEL={jina_clip}
DEEPLENS_WHISPER_MODEL_SIZE={whisper_size}

# Cloud Mode
GEMINI_CHAT_MODEL={gemini_chat}
POSTGRES_HOST={pg_host}
POSTGRES_PORT={pg_port}
POSTGRES_DB={pg_db}
POSTGRES_USER={pg_user}
POSTGRES_PASSWORD={pg_pass}
"""
        try:
            with open(".env", "w", encoding="utf-8") as f:
                f.write(env_content)
            logger.info("settings.saved_env_file")
        except Exception as e:
            logger.error("settings.write_env_failed", error=str(e))

        # Dynamically update singleton fields
        self.settings.mode = mode
        self.settings.data_dir = Path(data_dir)
        self.settings.ollama_host = ollama_host
        self.settings.ollama_chat_model = ollama_chat
        self.settings.ollama_rewriter_model = ollama_chat
        self.settings.jina_clip_model = jina_clip
        self.settings.whisper_model_size = whisper_size
        self.settings.gemini_chat_model = gemini_chat
        self.settings.postgres_host = pg_host
        self.settings.postgres_port = pg_port
        self.settings.postgres_db = pg_db
        self.settings.postgres_user = pg_user
        self.settings.postgres_password = pg_pass
        self.settings.chunk_size = chunk_size
        self.settings.chunk_overlap = chunk_overlap
        self.settings.ingestion_workers = workers
        self.settings.search_top_k = top_k

        # Recheck directories
        self.settings.ensure_directories()

        self.accept()
