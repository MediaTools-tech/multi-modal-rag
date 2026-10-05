from __future__ import annotations

from enum import Enum
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from multimodal_rag.config import KNOWN_EMBEDDING_DIMS, get_settings

ENUM_CHOICES = {
    "GUI_THEME": ["Midnight", "Graphite", "Ocean", "Daylight"],
    "SYSTEM_MODE": ["DOC_ONLY_RAG", "MULTI_MODAL_RAG"],
    "SEARCH_MODE": ["hybrid", "summary", "chunk"],
    "ACTIVE_DB_ENGINE": ["POSTGRES", "LANCEDB"],
    "EMBEDDING_PROVIDER": ["GOOGLE", "LOCAL"],
    "LLM_PROVIDER": ["DEEPSEEK", "GEMINI", "OPENAI", "ANTHROPIC", "XAI", "OLLAMA", "LOCAL"],
    "VLM_PROVIDER": ["GEMINI", "DEEPSEEK", "OPENAI", "ANTHROPIC", "XAI", "OLLAMA", "NONE", "LOCAL"],
    "VLM_FALLBACK_PROVIDER": ["GEMINI", "OLLAMA", "NONE", "LOCAL", "OPENAI", "ANTHROPIC", "XAI", "DEEPSEEK"],
    "LANCEDB_INDEX_TYPE": ["IVF_FLAT", "IVF_PQ", "FLAT"],
    "RERANKER_PROVIDER": ["LOCAL", "NONE"],
}

#: Display order: (section title, ordered keys). Every key must be a real
#: Settings field and match its .env name exactly; keys in ENUM_CHOICES render
#: as dropdowns, MODEL_PRESETS/STATIC_MODEL_LISTS keys as editable dropdowns,
#: everything else as text fields. DB_TABLE_NAME is intentionally absent: the
#: table is fixed from .env, and POSTGRES_DB / LANCEDB_DIR are the GUI knobs
#: for separate stores.
SECTIONS: list[tuple[str, list[str]]] = [
    ("General", ["GUI_THEME", "SYSTEM_MODE", "SEARCH_MODE", "ACTIVE_DB_ENGINE"]),
    (
        "Embeddings",
        [
            "EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDINGS_API_KEY",
            "EMBEDDING_BASE_URL", "EMBEDDING_DIMENSION", "EMBEDDING_BATCH_SIZE",
            "EMBEDDING_NORMALIZE", "EMBEDDING_TASK_TYPE", "EMBEDDING_MAX_RPM",
            "EMBEDDING_LOCAL_FALLBACK", "EMBEDDING_INPUT_COST_PER_1M",
            "EMBEDDING_OUTPUT_COST_PER_1M",
            "RERANKER_PROVIDER", "RERANKER_MODEL", "RERANKER_TOP_N", "RERANKER_ALLOW_CPU",
        ],
    ),
    (
        "Text LLM",
        [
            "LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL",
            "LLM_LOCAL_MODEL", "LLM_INPUT_COST_PER_1M", "LLM_OUTPUT_COST_PER_1M",
        ],
    ),
    (
        "Vision (VLM)",
        [
            "VLM_PROVIDER", "VLM_MODEL", "VLM_API_KEY", "VLM_BASE_URL",
            "VLM_LOCAL_MODEL", "VLM_FALLBACK_PROVIDER", "VLM_FALLBACK_MODEL",
            "VLM_CPU_FALLBACK_MODEL", "VLM_ENABLE_LOCAL_FALLBACK",
            "VLM_ALLOW_CPU_FALLBACK", "VLM_BATCH_SIZE", "VLM_MAX_RPM",
            "VLM_MAX_RPD", "VLM_RETRY_MAX", "VLM_BACKOFF_BASE",
            "VLM_INPUT_COST_PER_1M", "VLM_OUTPUT_COST_PER_1M", "VLM_CACHE_DIR",
        ],
    ),
    (
        "Postgres",
        [
            "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_USER",
            "POSTGRES_PASSWORD", "POSTGRES_DB", "POSTGRES_URL",
        ],
    ),
    (
        "LanceDB",
        ["LANCEDB_DIR", "LANCEDB_INDEX_TYPE", "LANCEDB_NUM_PARTITIONS"],
    ),
]


#: Model presets per provider for the editable model dropdowns. Anything not
#: listed can still be typed in (including a fully custom id) — the typed text
#: is what gets saved.
MODEL_PRESETS: dict[str, dict[str, list[str]]] = {
    "EMBEDDING_MODEL": {
        "GOOGLE": ["models/gemini-embedding-001", "models/text-embedding-004"],
        "LOCAL": [
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5",
            "BAAI/bge-m3",
        ],
    },
    "LLM_MODEL": {
        "DEEPSEEK": ["deepseek-chat", "deepseek-reasoner", "deepseek-flash"],
        "GEMINI": ["gemini-3.8-flash", "gemini-2.5-flash"],
        "OPENAI": ["gpt-4o-mini", "gpt-4o"],
        "ANTHROPIC": ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"],
        "XAI": ["grok-3-mini", "grok-3"],
        "OLLAMA": ["llama3.1:8b", "mistral:7b", "qwen2.5:7b"],
        "LOCAL": ["llama3.1:8b", "mistral:7b", "qwen2.5:7b"],
    },
    "VLM_MODEL": {
        "GEMINI": ["gemini-3.8-flash", "gemini-2.5-flash"],
        "DEEPSEEK": ["deepseek-flash"],
        "OPENAI": ["gpt-4o-mini", "gpt-4o"],
        "ANTHROPIC": ["claude-3-5-sonnet-latest"],
        "XAI": ["grok-3-mini"],
        "OLLAMA": ["qwen2-vl:7b", "llava:7b", "llama3.2-vision:11b"],
        "LOCAL": ["qwen2-vl:7b", "llava:7b", "llama3.2-vision:11b"],
        "NONE": [],
    },
}

#: Which provider dropdown drives each model dropdown.
MODEL_PROVIDER_KEY = {
    "EMBEDDING_MODEL": "EMBEDDING_PROVIDER",
    "LLM_MODEL": "LLM_PROVIDER",
    "VLM_MODEL": "VLM_PROVIDER",
}

#: Fixed preset lists (no provider dependency) for the local-model fields.
STATIC_MODEL_LISTS = {
    "LLM_LOCAL_MODEL": ["llama3.1:8b", "mistral:7b", "qwen2.5:7b"],
    "VLM_LOCAL_MODEL": ["qwen2-vl:7b", "llava:7b"],
    "VLM_FALLBACK_MODEL": ["qwen2-vl:7b", "llava:7b"],
    "RERANKER_MODEL": [
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "BAAI/bge-reranker-base",
    ],
}


class SettingsDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = get_settings()
        self.setWindowTitle("Settings")
        self.resize(620, 560)
        self.setMinimumSize(480, 320)
        self.widgets: dict[str, QWidget] = {}

        layout = QVBoxLayout(self)
        # Scrollable form: the field list outgrew fixed dialog heights, which
        # pushed Save/Cancel off-screen. The button box stays pinned below.
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.verticalScrollBar().setSingleStep(15)
        form_host = QWidget(scroll)
        form = QFormLayout(form_host)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(6)
        scroll.setWidget(form_host)
        layout.addWidget(scroll, 1)

        for title, keys in SECTIONS:
            header = QLabel(f"── {title} ──", form_host)
            header.setStyleSheet("font-weight: bold;")
            form.addRow(header)
            for key in keys:
                if key in ENUM_CHOICES:
                    combo = QComboBox()
                    items = list(ENUM_CHOICES[key])
                    current = self._enum_value(key)
                    if current not in items:
                        # Unset in .env (e.g. blank provider = use fallback):
                        # show it as blank, not as the first preset.
                        items = [""] + items
                        current = ""
                    combo.addItems(items)
                    combo.setCurrentText(current)
                    form.addRow(key, combo)
                    self.widgets[key] = combo
                elif key in MODEL_PROVIDER_KEY or key in STATIC_MODEL_LISTS:
                    combo = self._make_model_combo(key)
                    form.addRow(key, combo)
                    self.widgets[key] = combo
                else:
                    field = QLineEdit(self._text_value(key))
                    if "API_KEY" in key or "PASSWORD" in key:
                        field.setEchoMode(QLineEdit.EchoMode.Password)
                    form.addRow(key, field)
                    self.widgets[key] = field

        for model_key, prov_key in MODEL_PROVIDER_KEY.items():
            prov = self.widgets.get(prov_key)
            if isinstance(prov, QComboBox):
                prov.currentTextChanged.connect(
                    lambda _text, mk=model_key: self._on_provider_changed(mk)
                )
        emb_model = self.widgets.get("EMBEDDING_MODEL")
        if isinstance(emb_model, QComboBox):
            emb_model.currentTextChanged.connect(lambda _text: self._maybe_fix_dimension())
            emb_model.currentTextChanged.connect(
                lambda _text: self._sync_reranker_from_embedding()
            )

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        note = QLabel(
            "Saves to .env. The app will not restart on its own — close and "
            "reopen it to apply changes.",
            self,
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #a0a0b0; font-size: 11px;")
        layout.addWidget(note)

        # Show the exact file being read/written: the app root depends on the
        # launch folder (or MRAG_HOME), so without this it is easy to edit one
        # .env while the GUI reads another.
        env_path = Path(self.settings.app_root) / ".env"
        src = QLabel(
            f"Source: {env_path}"
            + ("" if env_path.is_file() else " (will be created on Save)"),
            self,
        )
        src.setWordWrap(True)
        src.setStyleSheet("color: #a0a0b0; font-size: 11px;")
        src.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(src)

    def _enum_value(self, key: str) -> str:
        # Raw configured values, NOT role-resolved fallbacks: the dialog must
        # show exactly what is in .env (blank stays blank).
        if key == "VLM_PROVIDER":
            provider = self.settings.VLM_PROVIDER
            return provider.value if provider is not None else ""
        if key == "LLM_PROVIDER":
            provider = self.settings.LLM_PROVIDER
            return provider.value if provider is not None else ""
        value = getattr(self.settings, key)
        return value.value if isinstance(value, Enum) else str(value)

    def _text_value(self, key: str) -> str:
        # Raw values (see above); role fallbacks would show guessed content.
        if key == "VLM_MODEL":
            return self.settings.VLM_MODEL or ""
        if key == "LLM_MODEL":
            return self.settings.LLM_MODEL or ""
        return str(getattr(self.settings, key) or "")

    def _initial_provider(self, model_key: str) -> str:
        if model_key == "EMBEDDING_MODEL":
            value = self.settings.EMBEDDING_PROVIDER
            return value.value if isinstance(value, Enum) else str(value)
        if model_key == "LLM_MODEL":
            return self.settings.llm_role.provider
        if model_key == "VLM_MODEL":
            return self.settings.vlm_role.provider
        return ""

    def _provider_of(self, model_key: str) -> str:
        prov = self.widgets.get(MODEL_PROVIDER_KEY.get(model_key, ""))
        if isinstance(prov, QComboBox):
            return prov.currentText()
        return self._initial_provider(model_key)

    @staticmethod
    def _fill_model_combo(combo: QComboBox, items: list[str], current: str) -> None:
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItems(items)
            if current and current in items:
                combo.setCurrentText(current)
            elif current:
                combo.setEditText(current)  # keep a custom (non-preset) value
            else:
                combo.setCurrentIndex(-1)  # blank stays blank (nothing committed)
        finally:
            combo.blockSignals(False)

    def _make_model_combo(self, key: str) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        combo.setToolTip("Pick a preset, or type any custom model id — the typed text is saved.")
        if key in MODEL_PROVIDER_KEY:
            presets = MODEL_PRESETS.get(key, {}).get(self._initial_provider(key), [])
        else:
            presets = STATIC_MODEL_LISTS.get(key, [])
        self._fill_model_combo(combo, presets, self._text_value(key))
        return combo

    def _refill_model_combo(self, model_key: str) -> None:
        combo = self.widgets.get(model_key)
        if not isinstance(combo, QComboBox):
            return
        previous = combo.currentText()
        presets = MODEL_PRESETS.get(model_key, {}).get(self._provider_of(model_key), [])
        if previous and previous not in self._union_presets(model_key):
            current = previous  # custom typed id: never clobbered
        elif previous in presets:
            current = previous  # still valid for the new provider
        else:
            current = presets[0] if presets else ""
        self._fill_model_combo(combo, presets, current)
        if model_key == "EMBEDDING_MODEL":
            self._maybe_fix_dimension()

    def _on_provider_changed(self, model_key: str) -> None:
        self._refill_model_combo(model_key)
        if model_key == "EMBEDDING_MODEL":
            self._sync_reranker_from_embedding()

    #: Reranker ids the dialog itself manages (auto-filled); anything else typed
    #: in is treated as the user's custom choice and never overwritten.
    AUTO_RERANKER_VALUES = {
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "BAAI/bge-reranker-base",
    }

    @staticmethod
    def _reranker_preset_for_embedding(model: str) -> str | None:
        name = (model or "").lower()
        if "bge" in name:
            return "BAAI/bge-reranker-base"
        if "minilm" in name or "mini-lm" in name:
            return "cross-encoder/ms-marco-MiniLM-L-6-v2"
        return None

    def _sync_reranker_from_embedding(self) -> None:
        """Keep the reranker fields consistent with the embedding choice.

        Local embeddings -> RERANKER_PROVIDER=LOCAL plus the matching reranker
        preset. API embeddings -> blank reranker model, left for the user to
        fill. A custom (non-preset) reranker id is never overwritten.
        """
        emb_prov_w = self.widgets.get("EMBEDDING_PROVIDER")
        emb_model_w = self.widgets.get("EMBEDDING_MODEL")
        rprov_w = self.widgets.get("RERANKER_PROVIDER")
        rmodel_w = self.widgets.get("RERANKER_MODEL")
        if emb_prov_w is None or emb_model_w is None or rprov_w is None or rmodel_w is None:
            return
        emb_prov = emb_prov_w.currentText() if isinstance(emb_prov_w, QComboBox) else ""
        emb_model = emb_model_w.currentText() if isinstance(emb_model_w, QComboBox) else ""
        if not isinstance(rmodel_w, QComboBox):
            return
        current = rmodel_w.currentText()
        if emb_prov == "LOCAL":
            if isinstance(rprov_w, QComboBox):
                rprov_w.setCurrentText("LOCAL")
            if not current or current in self.AUTO_RERANKER_VALUES:
                mapped = self._reranker_preset_for_embedding(emb_model)
                if mapped:
                    rmodel_w.blockSignals(True)
                    try:
                        if mapped in self._combo_items(rmodel_w):
                            rmodel_w.setCurrentText(mapped)
                        else:
                            rmodel_w.setEditText(mapped)
                    finally:
                        rmodel_w.blockSignals(False)
        elif current in self.AUTO_RERANKER_VALUES:
            rmodel_w.blockSignals(True)
            try:
                rmodel_w.setEditText("")
                rmodel_w.setCurrentIndex(-1)
            finally:
                rmodel_w.blockSignals(False)

    @staticmethod
    def _combo_items(combo: QComboBox) -> list[str]:
        return [combo.itemText(i) for i in range(combo.count())]

    @staticmethod
    def _union_presets(model_key: str) -> set[str]:
        out: set[str] = set()
        for plist in MODEL_PRESETS.get(model_key, {}).values():
            out.update(plist)
        return out

    def _maybe_fix_dimension(self) -> None:
        """Auto-populate EMBEDDING_DIMENSION from the selected model.

        Only fires when the model dictates exactly one dimension and the
        current value is not valid for it; valid or ambiguous values and
        unknown models are left untouched.
        """
        model_combo = self.widgets.get("EMBEDDING_MODEL")
        dim_widget = self.widgets.get("EMBEDDING_DIMENSION")
        if model_combo is None or dim_widget is None:
            return
        model = str(model_combo.currentText()).strip()
        valid = KNOWN_EMBEDDING_DIMS.get(model)
        if not valid or len(valid) != 1:
            return
        try:
            current_dim = int(str(dim_widget.text()).strip())
        except (TypeError, ValueError):
            current_dim = None
        if current_dim not in valid:
            dim_widget.setText(str(next(iter(valid))))

    def _values(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, widget in self.widgets.items():
            out[key] = widget.currentText() if isinstance(widget, QComboBox) else widget.text()
        return out

    def _save(self) -> None:
        env_path = Path(self.settings.app_root) / ".env"
        existing = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
        updates = self._values()
        seen: set[str] = set()
        lines: list[str] = []
        for line in existing:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key = stripped.split("=", 1)[0].strip()
                if key in updates:
                    lines.append(f"{key}={updates[key]}")
                    seen.add(key)
                    continue
            lines.append(line)
        for key, value in updates.items():
            if key not in seen:
                lines.append(f"{key}={value}")
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.accept()
