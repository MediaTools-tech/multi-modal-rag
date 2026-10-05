from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from enum import Enum
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

from pydantic import BaseModel, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_HOME_ENV = "MRAG_HOME"


def resolve_app_root() -> Path:
    """Cwd-independent application root for .env, state and data files."""
    env_root = os.environ.get(DEFAULT_HOME_ENV)
    if env_root:
        return Path(env_root).expanduser().resolve()
    if (Path.cwd() / ".env").is_file():
        return Path.cwd().resolve()
    return (Path.home() / ".multimodal_rag").resolve()


APP_ROOT = resolve_app_root()


def _env_files() -> list[Path]:
    """Lowest -> highest precedence. Missing files are ignored by pydantic."""
    candidates = [Path.home() / ".multimodal_rag" / ".env", Path.cwd() / ".env", APP_ROOT / ".env"]
    ordered: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved not in ordered and resolved.is_file():
            ordered.append(resolved)
    return ordered


class SystemMode(str, Enum):
    DOC_ONLY_RAG = "DOC_ONLY_RAG"
    MULTI_MODAL_RAG = "MULTI_MODAL_RAG"


class DBEngine(str, Enum):
    POSTGRES = "POSTGRES"
    LANCEDB = "LANCEDB"


class HardwareDevice(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class LLMProvider(str, Enum):
    DEEPSEEK = "DEEPSEEK"
    GEMINI = "GEMINI"
    OLLAMA = "OLLAMA"
    LOCAL = "LOCAL"
    OPENAI = "OPENAI"
    ANTHROPIC = "ANTHROPIC"
    XAI = "XAI"


class VLMProvider(str, Enum):
    GEMINI = "GEMINI"
    OLLAMA = "OLLAMA"
    NONE = "NONE"
    LOCAL = "LOCAL"
    OPENAI = "OPENAI"
    ANTHROPIC = "ANTHROPIC"
    XAI = "XAI"
    DEEPSEEK = "DEEPSEEK"


class EmbeddingProvider(str, Enum):
    LOCAL = "LOCAL"
    GOOGLE = "GOOGLE"


class RerankerProvider(str, Enum):
    LOCAL = "LOCAL"
    NONE = "NONE"


class FrameStrategy(str, Enum):
    HYBRID = "HYBRID"
    SCENE = "SCENE"
    FPS = "FPS"


class LanceIndexType(str, Enum):
    IVF_FLAT = "IVF_FLAT"
    IVF_PQ = "IVF_PQ"
    FLAT = "FLAT"


class IngestionMode(str, Enum):
    WATCH = "watch"
    SCAN = "scan"
    MANUAL = "manual"


class SearchMode(str, Enum):
    CHUNK = "chunk"
    SUMMARY = "summary"
    HYBRID = "hybrid"


class GuiTheme(str, Enum):
    MIDNIGHT = "Midnight"
    GRAPHITE = "Graphite"
    OCEAN = "Ocean"
    # SUNSET = "Sunset"  # disabled for now
    DAYLIGHT = "Daylight"


MODE_TO_ENGINE = {
    SystemMode.DOC_ONLY_RAG: DBEngine.POSTGRES,
    SystemMode.MULTI_MODAL_RAG: DBEngine.LANCEDB,
}

KNOWN_EMBEDDING_DIMS: dict[str, set[int]] = {
    "BAAI/bge-small-en-v1.5": {384},
    "BAAI/bge-base-en-v1.5": {768},
    "sentence-transformers/all-MiniLM-L6-v2": {384},
    "BAAI/bge-m3": {1024},
    "models/gemini-embedding-001": {768, 1536, 3072},
    "gemini-embedding-001": {768, 1536, 3072},
    "models/text-embedding-004": {256, 512, 768},
    "text-embedding-004": {256, 512, 768},
    "embed-english-v3.0": {1024},
    "embed-multilingual-v3.0": {1024},
}


class RoleSettings(BaseModel):
    """One purpose (embeddings / vlm / llm) with its provider, model and credential.

    This is the single source of truth consumers use; they never read vendor keys
    or provider selectors directly.
    """

    purpose: str
    provider: str
    model: str
    api_key: str = ""
    base_url: str = ""


@lru_cache(maxsize=1)
def detect_cuda() -> bool:
    try:
        import torch

        if torch.cuda.is_available():
            return True
    except Exception:
        pass

    if shutil.which("nvidia-smi"):
        try:
            subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=True,
            )
            return True
        except Exception:
            pass
    return False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    SYSTEM_MODE: SystemMode = SystemMode.MULTI_MODAL_RAG
    ACTIVE_DB_ENGINE: DBEngine | None = None
    # Desktop GUI color scheme (see gui/theme.py). Applied at startup.
    GUI_THEME: GuiTheme = GuiTheme.GRAPHITE

    # ── Purpose-first roles (single source of truth) ──────────────────────
    # Each purpose has its own provider, model and key. Keys are blank by default
    # and intentionally per-purpose (a Google key is written for each role it serves).
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Embeddings
    EMBEDDING_PROVIDER: EmbeddingProvider = EmbeddingProvider.GOOGLE
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    EMBEDDINGS_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = ""
    EMBEDDING_DIMENSION: int = 768
    EMBEDDING_BATCH_SIZE: int = 64
    EMBEDDING_NORMALIZE: bool = True
    EMBEDDING_TASK_TYPE: str = "RETRIEVAL_DOCUMENT"
    EMBEDDING_MAX_RPM: int = 1000
    EMBEDDING_LOCAL_FALLBACK: bool = True
    EMBEDDING_INPUT_COST_PER_1M: float = 0.0
    EMBEDDING_OUTPUT_COST_PER_1M: float = 0.0

    # Vision (standalone image captions + video keyframe captions)
    VLM_PROVIDER: VLMProvider | None = None
    VLM_MODEL: str | None = None
    VLM_API_KEY: str = ""
    VLM_BASE_URL: str = ""
    VLM_FALLBACK_PROVIDER: VLMProvider = VLMProvider.OLLAMA
    VLM_FALLBACK_MODEL: str = "qwen2-vl:7b"
    VLM_CPU_FALLBACK_MODEL: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    VLM_ENABLE_LOCAL_FALLBACK: bool = True
    VLM_ALLOW_CPU_FALLBACK: bool = False
    # Default model when VLM_PROVIDER=LOCAL and VLM_MODEL is blank.
    # Must be pulled in Ollama first: `ollama pull qwen2-vl:7b`.
    VLM_LOCAL_MODEL: str = "qwen2-vl:7b"
    VLM_INPUT_COST_PER_1M: float = 0.0
    VLM_OUTPUT_COST_PER_1M: float = 0.0

    # Text (per-file summaries + synthesized answers)
    LLM_PROVIDER: LLMProvider | None = None
    LLM_MODEL: str | None = None
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = ""
    # Default model when LLM_PROVIDER=LOCAL and LLM_MODEL is blank.
    # Must be pulled in Ollama first: `ollama pull llama3.1:8b`.
    LLM_LOCAL_MODEL: str = "llama3.1:8b"
    LLM_INPUT_COST_PER_1M: float = 0.0
    LLM_OUTPUT_COST_PER_1M: float = 0.0

    # ── Deprecated provider-centric names (fallbacks; prefer the role fields) ──
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    GOOGLE_API_KEY: str = ""
    # Shared Google endpoint used by any Google/Gemini role when its own base URL is blank.
    # Blank = SDK default (also https://generativelanguage.googleapis.com).
    GOOGLE_BASE_URL: str = ""
    TEXT_LLM_PROVIDER: LLMProvider = LLMProvider.DEEPSEEK
    TEXT_LLM_MODEL: str = "deepseek-chat"
    VISION_LLM_PROVIDER: VLMProvider = VLMProvider.GEMINI
    VISION_LLM_MODEL: str = "gemini-3.8-flash"
    # Vendor endpoints + default models for the newer providers. Purpose-scoped
    # LLM_*/VLM_* keys and models take precedence when set; check each vendor's
    # docs for current model names (old ones get retired).
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4o-mini"
    ANTHROPIC_BASE_URL: str = "https://api.anthropic.com"
    ANTHROPIC_MODEL: str = "claude-3-5-sonnet-latest"
    XAI_BASE_URL: str = "https://api.x.ai/v1"
    XAI_MODEL: str = "grok-3-mini"

    RERANKER_PROVIDER: RerankerProvider = RerankerProvider.LOCAL
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_TOP_N: int = 15
    RERANKER_ALLOW_CPU: bool = True

    SEARCH_TOP_K: int = 10
    SEARCH_MODE: SearchMode = SearchMode.HYBRID
    HYBRID_RRF_K: int = 60
    ENABLE_DOCUMENT_SUMMARIES: bool = True
    SUMMARY_MAX_CHARS: int = 4000

    HARDWARE_DEVICE: HardwareDevice = HardwareDevice.AUTO

    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "password"
    POSTGRES_DB: str = "rag_db"
    # Optional full-DSN override (managed DBs, sockets, extra query params).
    # Blank (default) = built from the components above.
    POSTGRES_URL: str = ""

    @property
    def postgres_dsn(self) -> str:
        """Effective Postgres DSN: explicit URL, else built from components."""
        if self.POSTGRES_URL.strip():
            return self.POSTGRES_URL.strip()
        from urllib.parse import quote

        user = quote(self.POSTGRES_USER, safe="")
        password = quote(self.POSTGRES_PASSWORD, safe="")
        return (
            f"postgresql://{user}:{password}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def postgres_dsn_sanitized(self) -> str:
        """Effective DSN with the password masked (safe to store/log)."""
        from urllib.parse import urlparse, urlunparse

        parts = urlparse(self.postgres_dsn)
        userinfo, _, hostinfo = parts.netloc.rpartition("@")
        if not userinfo:
            return self.postgres_dsn
        user, _, _ = userinfo.partition(":")
        masked = f"{user}:***@{hostinfo}" if hostinfo else f"{user}:***"
        return urlunparse(parts._replace(netloc=masked))
    LANCEDB_DIR: str = "./data/lancedb_lakehouse"
    DB_TABLE_NAME: str = "chunks"
    LANCEDB_INDEX_TYPE: LanceIndexType = LanceIndexType.IVF_FLAT
    LANCEDB_NUM_PARTITIONS: int = 256

    DOC_USE_OCR: bool = True
    DOC_CHUNK_SIZE: int = 800
    DOC_CHUNK_OVERLAP: int = 100

    MM_FRAME_STRATEGY: FrameStrategy = FrameStrategy.HYBRID
    MM_FRAME_SAMPLING_FPS: int = 2
    MM_SCENE_THRESHOLD: float = 0.30
    MM_SCENE_MIN_INTERVAL_SEC: float = 8.0
    MM_MAX_KEYFRAMES_PER_MIN: int = 30
    MM_DEDUP_HASH_THRESHOLD: int = 6
    MM_RESIZE_FRAME: bool = True
    MM_TARGET_HEIGHT: int = 360
    MM_JPEG_QUALITY: int = 80

    VLM_BATCH_SIZE: int = 8
    VLM_MAX_RPM: int = 15
    VLM_MAX_RPD: int = 1500
    VLM_RETRY_MAX: int = 5
    VLM_BACKOFF_BASE: float = 2.0
    ENABLE_CACHE: bool = True
    VLM_CACHE_DIR: str = "./data/vlm_cache"

    DATA_DIR: str = "./data"
    INBOX_DIR: str = "./data/inbox"
    PROCESSING_DIR: str = "./data/processing"
    PROCESSED_DIR: str = "./data/processed"
    FAILED_DIR: str = "./data/failed"
    # Ingest queue/state database. Default = ./data/state.db under the app root
    # (self-contained: one queue per app folder, travels with a portable build).
    # Leave BLANK ("") to instead key one per-user file by the active vector
    # store (Postgres DSN+table, or LanceDB dir+table), so several app folders
    # pointing at the same DB share one record set. Relative paths resolve
    # against the app root.
    STATE_DB_PATH: str = "./data/state.db"
    INGESTION_MODE: IngestionMode = IngestionMode.WATCH
    WATCH_POLL_INTERVAL_SEC: float = 5.0
    FILE_STABLE_SECONDS: float = 15.0
    WORKER_COUNT: int = 2
    MAX_INGEST_RETRIES: int = 3
    INGEST_IGNORE_EXTENSIONS: str = ".part,.crdownload,.tmp,.swp,.lock"

    WHISPER_MODEL_SIZE: str = "base"
    WHISPER_DEVICE: str = "auto"
    WHISPER_COMPUTE_TYPE: str = "auto"
    WHISPER_LANGUAGE: str | None = None

    DEEPSEEK_INPUT_COST_PER_1M: float = 0.27
    DEEPSEEK_OUTPUT_COST_PER_1M: float = 1.10
    GEMINI_INPUT_COST_PER_1M: float = 0.0
    GEMINI_OUTPUT_COST_PER_1M: float = 0.0
    # Per-model token rates in USD per 1M tokens (JSON in .env), e.g.
    # MODEL_TOKEN_RATES={"gemini-3.8-flash": {"input": 0.075, "output": 0.30}}
    # A ["in", "out"] list per model is accepted as shorthand. These take
    # precedence over the provider-level rates above; models with neither show
    # "rate missing" (local Ollama models show "local" instead).
    MODEL_TOKEN_RATES: dict = {}

    @field_validator("ACTIVE_DB_ENGINE", "VLM_PROVIDER", "LLM_PROVIDER", mode="before")
    @classmethod
    def _blank_enum_is_none(cls, value):
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @field_validator("MODEL_TOKEN_RATES", mode="before")
    @classmethod
    def _parse_model_rates(cls, value):
        import json

        if value is None or value == "":
            return {}
        if isinstance(value, str):
            value = json.loads(value) if value.strip() else {}
        normalized: dict[str, dict[str, float]] = {}
        for model, rates in (value or {}).items():
            if isinstance(rates, (list, tuple)) and len(rates) == 2:
                normalized[str(model)] = {"input": float(rates[0]), "output": float(rates[1])}
            elif isinstance(rates, dict):
                normalized[str(model)] = {
                    "input": float(rates.get("input", 0.0)),
                    "output": float(rates.get("output", 0.0)),
                }
        return normalized

    @field_validator("DB_TABLE_NAME")
    @classmethod
    def _valid_table_name(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value or ""):
            raise ValueError(
                "DB_TABLE_NAME must be a simple SQL identifier "
                "(letters, digits, underscore; cannot start with a digit)."
            )
        return value

    @model_validator(mode="before")
    @classmethod
    def _blank_strings_use_defaults(cls, data):
        # A blank "KEY=" in .env (left by hand-editing or the Settings dialog)
        # must fall back to the default, not crash float/bool/int parsing and
        # kill startup with no recovery path in the GUI.
        if isinstance(data, dict):
            return {
                key: value
                for key, value in data.items()
                if not (isinstance(value, str) and value.strip() == "")
            }
        return data

    @model_validator(mode="after")
    def _resolve_engine(self) -> "Settings":
        # SYSTEM_MODE is a content-policy preset that only suggests a default engine.
        # ACTIVE_DB_ENGINE is authoritative if set explicitly: the two are orthogonal
        # (e.g. multimodal content stored in Postgres, or documents in LanceDB).
        default_engine = MODE_TO_ENGINE[self.SYSTEM_MODE]
        if self.ACTIVE_DB_ENGINE is None:
            self.ACTIVE_DB_ENGINE = default_engine
        elif self.ACTIVE_DB_ENGINE != default_engine:
            logger.warning(
                "ACTIVE_DB_ENGINE=%s overrides the default engine for %s (%s).",
                self.ACTIVE_DB_ENGINE.value,
                self.SYSTEM_MODE.value,
                default_engine.value,
            )
        return self

    @model_validator(mode="after")
    def _validate_embedding_dim(self) -> "Settings":
        valid = KNOWN_EMBEDDING_DIMS.get(self.EMBEDDING_MODEL)
        if valid is None or self.EMBEDDING_DIMENSION in valid:
            return self
        if len(valid) == 1:
            # Model dictates exactly one dimension (e.g. MiniLM/bge-small = 384):
            # snap to it so a GUI-only setup works without editing .env.
            chosen = next(iter(valid))
            logger.warning(
                "EMBEDDING_DIMENSION=%d invalid for EMBEDDING_MODEL=%s; using %d",
                self.EMBEDDING_DIMENSION,
                self.EMBEDDING_MODEL,
                chosen,
            )
            self.EMBEDDING_DIMENSION = chosen
            return self
        raise ValueError(
            f"EMBEDDING_DIMENSION={self.EMBEDDING_DIMENSION} invalid for "
            f"EMBEDDING_MODEL={self.EMBEDDING_MODEL} (valid: {sorted(valid)})."
        )

    @model_validator(mode="after")
    def _resolve_relative_paths(self) -> "Settings":
        for name in (
            "DATA_DIR",
            "INBOX_DIR",
            "PROCESSING_DIR",
            "PROCESSED_DIR",
            "FAILED_DIR",
            "VLM_CACHE_DIR",
            "LANCEDB_DIR",
        ):
            value = getattr(self, name)
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = APP_ROOT / path
            setattr(self, name, str(path.resolve()))
        # Ingest state follows the vector store identity (see _default_state_db_path),
        # not the launch folder, so every entry point sees the same records.
        if not self.STATE_DB_PATH.strip():
            self.STATE_DB_PATH = str(self._default_state_db_path())
        else:
            path = Path(self.STATE_DB_PATH).expanduser()
            if not path.is_absolute():
                path = APP_ROOT / path
            self.STATE_DB_PATH = str(path.resolve())
        return self

    def _default_state_db_path(self) -> Path:
        """Per-user state.db keyed by the active vector store's identity."""
        import hashlib
        import os

        base_env = os.environ.get("MRAG_HOME")
        base = Path(base_env).expanduser() if base_env else Path.home() / ".multimodal_rag"
        if self.ACTIVE_DB_ENGINE is DBEngine.POSTGRES:
            identity = f"POSTGRES|{self.postgres_dsn_sanitized}|{self.DB_TABLE_NAME}"
        else:
            identity = f"LANCEDB|{self.LANCEDB_DIR}|{self.DB_TABLE_NAME}"
        key = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
        return base / "state" / f"state_{key}.db"

    @property
    def app_root(self) -> Path:
        return APP_ROOT

    @property
    def embedding_role(self) -> RoleSettings:
        provider = self.EMBEDDING_PROVIDER.value
        if provider == "GOOGLE":
            base_url = self.EMBEDDING_BASE_URL or self.GOOGLE_BASE_URL
        else:
            base_url = self.EMBEDDING_BASE_URL
        return RoleSettings(
            purpose="embeddings",
            provider=provider,
            model=self.EMBEDDING_MODEL,
            api_key=self.EMBEDDINGS_API_KEY
            or (self.GOOGLE_API_KEY if provider == "GOOGLE" else ""),
            base_url=base_url,
        )

    def _vendor_default(self, provider: str) -> tuple[str, str]:
        """(default model, default base URL) for the newer API vendors."""
        if provider == "OPENAI":
            return self.OPENAI_MODEL, self.OPENAI_BASE_URL
        if provider == "ANTHROPIC":
            return self.ANTHROPIC_MODEL, self.ANTHROPIC_BASE_URL
        if provider == "XAI":
            return self.XAI_MODEL, self.XAI_BASE_URL
        raise ValueError(f"No vendor default for provider {provider!r}")

    @property
    def vlm_role(self) -> RoleSettings:
        provider = (self.VLM_PROVIDER or self.VISION_LLM_PROVIDER).value
        if provider == "LOCAL":
            # All-local indicator: runs on the local Ollama runner, no API key.
            return RoleSettings(
                purpose="vlm",
                provider=provider,
                model=self.VLM_MODEL or self.VLM_LOCAL_MODEL,
                api_key="",
                base_url=self.VLM_BASE_URL or self.OLLAMA_BASE_URL,
            )
        model = self.VLM_MODEL or self.VISION_LLM_MODEL
        if provider == "GEMINI":
            base_url = self.VLM_BASE_URL or self.GOOGLE_BASE_URL
        elif provider == "OLLAMA":
            base_url = self.VLM_BASE_URL or self.OLLAMA_BASE_URL
        elif provider in ("OPENAI", "ANTHROPIC", "XAI"):
            default_model, default_url = self._vendor_default(provider)
            model = self.VLM_MODEL or default_model
            base_url = self.VLM_BASE_URL or default_url
        elif provider == "DEEPSEEK":
            model = self.VLM_MODEL or "deepseek-flash"
            base_url = self.VLM_BASE_URL or "https://api.deepseek.com"
        else:
            base_url = self.VLM_BASE_URL
        return RoleSettings(
            purpose="vlm",
            provider=provider,
            model=model,
            api_key=self.VLM_API_KEY or (self.GOOGLE_API_KEY if provider == "GEMINI" else ""),
            base_url=base_url,
        )

    @property
    def llm_role(self) -> RoleSettings:
        provider = (self.LLM_PROVIDER or self.TEXT_LLM_PROVIDER).value
        if provider == "LOCAL":
            # All-local indicator: runs on the local Ollama runner, no API key.
            return RoleSettings(
                purpose="llm",
                provider=provider,
                model=self.LLM_MODEL or self.LLM_LOCAL_MODEL,
                api_key="",
                base_url=self.LLM_BASE_URL or self.OLLAMA_BASE_URL,
            )
        if provider == "GEMINI":
            # Use a Gemini model even if a leftover non-Gemini model name is configured.
            if self.LLM_MODEL and "gemini" in self.LLM_MODEL.lower():
                model = self.LLM_MODEL
            elif "gemini" in (self.TEXT_LLM_MODEL or "").lower():
                model = self.TEXT_LLM_MODEL
            else:
                model = self.VLM_MODEL or self.VISION_LLM_MODEL
        elif provider in ("OPENAI", "ANTHROPIC", "XAI"):
            default_model, _ = self._vendor_default(provider)
            model = self.LLM_MODEL or default_model
        else:
            model = self.LLM_MODEL or self.TEXT_LLM_MODEL
        if provider == "DEEPSEEK":
            api_key = self.LLM_API_KEY or self.DEEPSEEK_API_KEY
            base_url = self.LLM_BASE_URL or self.DEEPSEEK_BASE_URL
        elif provider == "GEMINI":
            api_key = self.LLM_API_KEY or self.GOOGLE_API_KEY
            base_url = self.LLM_BASE_URL or self.GOOGLE_BASE_URL
        elif provider in ("OPENAI", "ANTHROPIC", "XAI"):
            _, default_url = self._vendor_default(provider)
            api_key = self.LLM_API_KEY
            base_url = self.LLM_BASE_URL or default_url
        else:  # OLLAMA
            api_key = self.LLM_API_KEY
            base_url = self.LLM_BASE_URL or self.OLLAMA_BASE_URL
        return RoleSettings(
            purpose="llm", provider=provider, model=model, api_key=api_key, base_url=base_url
        )

    # Deprecated convenience views (prefer the *_role objects).
    @property
    def text_llm_api_key(self) -> str:
        return self.llm_role.api_key

    @property
    def vision_llm_api_key(self) -> str:
        return self.vlm_role.api_key

    @property
    def embedding_api_key(self) -> str:
        return self.embedding_role.api_key

    @property
    def text_model(self) -> str:
        return self.llm_role.model

    def cost_for(
        self,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        purpose: str = "",
    ) -> tuple[float, str]:
        """(cost_usd, status) for one call; status is rated/local/missing.

        Rate precedence: per-model MODEL_TOKEN_RATES → per-purpose
        VLM/LLM/EMBEDDING_*_COST_PER_1M → legacy provider DEEPSEEK_*/GEMINI_*.
        """
        from multimodal_rag.utils.token_tracker import compute_model_cost

        return compute_model_cost(
            self.MODEL_TOKEN_RATES,
            {
                "DEEPSEEK": (self.DEEPSEEK_INPUT_COST_PER_1M, self.DEEPSEEK_OUTPUT_COST_PER_1M),
                "GEMINI": (self.GEMINI_INPUT_COST_PER_1M, self.GEMINI_OUTPUT_COST_PER_1M),
            },
            model,
            provider,
            prompt_tokens,
            completion_tokens,
            purpose=purpose,
            purpose_rates={
                "vlm": (self.VLM_INPUT_COST_PER_1M, self.VLM_OUTPUT_COST_PER_1M),
                "llm": (self.LLM_INPUT_COST_PER_1M, self.LLM_OUTPUT_COST_PER_1M),
                "embedding": (
                    self.EMBEDDING_INPUT_COST_PER_1M,
                    self.EMBEDDING_OUTPUT_COST_PER_1M,
                ),
            },
        )

    @property
    def has_cuda(self) -> bool:
        if self.HARDWARE_DEVICE is HardwareDevice.CPU:
            return False
        return detect_cuda()

    @property
    def device(self) -> str:
        if self.HARDWARE_DEVICE is HardwareDevice.CUDA and not detect_cuda():
            return "cpu"
        return "cuda" if self.has_cuda else "cpu"

    @property
    def is_doc_only(self) -> bool:
        return self.SYSTEM_MODE is SystemMode.DOC_ONLY_RAG

    @property
    def is_multimodal(self) -> bool:
        return self.SYSTEM_MODE is SystemMode.MULTI_MODAL_RAG

    def allows_kind(self, kind: str) -> bool:
        """Content policy. DOC_ONLY_RAG ingests documents + images only;
        MULTI_MODAL_RAG adds time-based media (audio + video)."""
        if not self.is_doc_only:
            return True
        return kind in ("document", "image")

    @property
    def vlm_batching_enabled(self) -> bool:
        return self.VISION_LLM_PROVIDER is not VLMProvider.NONE and self.VLM_BATCH_SIZE > 1

    @property
    def vlm_local_fallback_enabled(self) -> bool:
        if not self.VLM_ENABLE_LOCAL_FALLBACK:
            return False
        if self.VLM_FALLBACK_PROVIDER is VLMProvider.NONE:
            return False
        return self.has_cuda or self.VLM_ALLOW_CPU_FALLBACK

    @property
    def vlm_local_fallback_model(self) -> str | None:
        if not self.vlm_local_fallback_enabled:
            return None
        if self.has_cuda:
            return self.VLM_FALLBACK_MODEL
        return self.VLM_CPU_FALLBACK_MODEL

    @property
    def embeddings_local_enabled(self) -> bool:
        if self.EMBEDDING_PROVIDER is EmbeddingProvider.LOCAL:
            return True
        return self.EMBEDDING_LOCAL_FALLBACK and self.has_cuda

    @property
    def reranker_local_enabled(self) -> bool:
        if self.RERANKER_PROVIDER is not RerankerProvider.LOCAL:
            return False
        return self.has_cuda or self.RERANKER_ALLOW_CPU

    @property
    def active_reranker(self) -> str:
        if self.RERANKER_PROVIDER is RerankerProvider.NONE:
            return "NONE"
        if not self.reranker_local_enabled:
            return "NONE"
        return self.RERANKER_PROVIDER.value

    @property
    def whisper_device(self) -> str:
        if self.WHISPER_DEVICE != "auto":
            return self.WHISPER_DEVICE
        return "cuda" if self.has_cuda else "cpu"

    @property
    def whisper_compute_type(self) -> str:
        if self.WHISPER_COMPUTE_TYPE != "auto":
            return self.WHISPER_COMPUTE_TYPE
        return "float16" if self.whisper_device == "cuda" else "int8"

    @property
    def ingest_ignore_extensions(self) -> set[str]:
        return {
            ext.strip().lower()
            for ext in self.INGEST_IGNORE_EXTENSIONS.split(",")
            if ext.strip()
        }

    def ensure_data_dirs(self) -> None:
        for directory in (
            self.DATA_DIR,
            self.INBOX_DIR,
            self.PROCESSING_DIR,
            self.PROCESSED_DIR,
            self.FAILED_DIR,
            self.VLM_CACHE_DIR,
            self.LANCEDB_DIR,
        ):
            Path(directory).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
