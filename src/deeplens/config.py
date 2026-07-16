"""DeepLens application configuration.

Centralizes all settings with environment variable overrides. Uses pydantic-settings
for validation and type coercion. Secrets (API keys) are stored in the OS keyring,
never in plaintext config files.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppMode(str, Enum):
    """Application mode selector — single flag to swap the entire backend."""

    LOCAL = "local"
    CLOUD = "cloud"


class Settings(BaseSettings):
    """Global application settings, loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_prefix="DEEPLENS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Mode ──────────────────────────────────────────────────────────────
    mode: AppMode = Field(default=AppMode.LOCAL, description="Backend mode: local or cloud")

    # ── Paths ─────────────────────────────────────────────────────────────
    data_dir: Path = Field(
        default=Path("~/.deeplens"),
        description="Root directory for application data (DB files, logs, temp)",
    )

    # ── Ingestion ─────────────────────────────────────────────────────────
    ingestion_workers: int = Field(default=2, ge=1, le=8, description="Concurrent ingestion workers")
    chunk_size: int = Field(default=500, ge=100, le=2000, description="Target chunk size in words")
    chunk_overlap: float = Field(default=0.15, ge=0.0, le=0.5, description="Chunk overlap ratio")
    video_chunk_seconds: int = Field(default=20, ge=5, le=60, description="Video segment length")
    video_overlap_seconds: int = Field(default=10, ge=0, le=30, description="Video segment overlap")
    video_fps_sample: int = Field(default=1, ge=1, le=5, description="Frames per second to sample")

    # ── Search ────────────────────────────────────────────────────────────
    search_top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    max_retries: int = Field(default=2, ge=0, le=5, description="Max LangGraph retry loops")
    relevance_threshold: float = Field(
        default=0.65, ge=0.0, le=1.0, description="Min cosine similarity for context pass"
    )

    # ── Local Mode ────────────────────────────────────────────────────────
    ollama_host: str = Field(default="http://localhost:11434", description="Ollama server URL")
    ollama_chat_model: str = Field(default="llama3.2:3b", description="Ollama chat model name")
    ollama_rewriter_model: str = Field(
        default="llama3.2:3b", description="Ollama rewriter model name"
    )
    jina_clip_model: str = Field(
        default="jinaai/jina-clip-v2", description="Jina CLIP model identifier"
    )
    whisper_model_size: str = Field(
        default="base", description="faster-whisper model size (tiny/base/small/medium/large)"
    )

    # ── Cloud Mode ────────────────────────────────────────────────────────
    gemini_embedding_model: str = Field(
        default="models/text-embedding-004", description="Gemini embedding model"
    )
    gemini_chat_model: str = Field(
        default="models/gemini-2.5-flash", description="Gemini chat model"
    )
    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_db: str = Field(default="deeplens", description="PostgreSQL database name")
    postgres_user: str = Field(default="deeplens", description="PostgreSQL username")
    postgres_password: str = Field(default="", description="PostgreSQL password")

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", description="Logging level")
    enable_langsmith: bool = Field(default=False, description="Enable LangSmith tracing (cloud)")

    # ── Vector dimensions (derived, not user-set) ─────────────────────────
    @property
    def vector_dim(self) -> int:
        """Return embedding dimension based on active mode."""
        return 1024 if self.mode == AppMode.LOCAL else 3072

    @field_validator("data_dir", mode="before")
    @classmethod
    def expand_data_dir(cls, v: Any) -> Path:
        """Expand ~ in data directory path."""
        return Path(os.path.expanduser(str(v)))

    @model_validator(mode="after")
    def ensure_directories(self) -> "Settings":
        """Create required data directories on startup."""
        for subdir in ("db", "logs", "temp", "cache"):
            (self.data_dir / subdir).mkdir(parents=True, exist_ok=True)
        return self

    def get_gemini_api_key(self) -> str:
        """Retrieve Gemini API key from OS keyring (never stored in config)."""
        import keyring

        key = keyring.get_password("deeplens", "gemini_api_key")
        if not key:
            # Fallback to environment variable for CI / Docker
            key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise ValueError(
                "Gemini API key not found. Set it with:\n"
                '  python -c "import keyring; keyring.set_password(\'deeplens\', '
                "'gemini_api_key', 'YOUR_KEY')\"\n"
                "  or set GEMINI_API_KEY environment variable."
            )
        return key

    @property
    def postgres_dsn(self) -> str:
        """Build PostgreSQL connection string."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def async_postgres_dsn(self) -> str:
        """Build async PostgreSQL connection string."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def lancedb_path(self) -> Path:
        """Path to the LanceDB database directory."""
        return self.data_dir / "db" / "lancedb"

    @property
    def temp_dir(self) -> Path:
        """Temporary directory for archive extraction, frame dumps, etc."""
        return self.data_dir / "temp"

    @property
    def log_dir(self) -> Path:
        """Directory for structured log files."""
        return self.data_dir / "logs"


# ── Singleton access ──────────────────────────────────────────────────────
_settings: Settings | None = None


def get_settings(**overrides: Any) -> Settings:
    """Get or create the global settings singleton.

    Args:
        **overrides: Field overrides, useful for testing.

    Returns:
        The application settings instance.
    """
    global _settings
    if _settings is None or overrides:
        _settings = Settings(**overrides)
    return _settings
