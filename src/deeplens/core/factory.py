"""Backend factory — creates repository and engine instances based on the active mode.

This is the single point where the local/cloud decision is made. All other code
depends only on the abstract interfaces, never on concrete implementations.
"""

from __future__ import annotations

import structlog

from deeplens.config import AppMode, Settings, get_settings
from deeplens.core.chat import ChatEngine
from deeplens.core.embedding import EmbeddingEngine
from deeplens.core.repository import DocumentRepository

logger = structlog.get_logger(__name__)


class BackendFactory:
    """Factory for creating backend instances based on application mode.

    Usage:
        factory = BackendFactory(settings)
        repo = factory.create_repository()
        embedder = factory.create_embedding_engine()
        chat = factory.create_chat_engine()
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        logger.info(
            "backend_factory.init",
            mode=self.settings.mode.value,
            vector_dim=self.settings.vector_dim,
        )

    def create_repository(self) -> DocumentRepository:
        """Create the vector database repository for the current mode."""
        if self.settings.mode == AppMode.LOCAL:
            from deeplens.repositories.lancedb_repo import LanceDBRepository

            return LanceDBRepository(self.settings)
        else:
            from deeplens.repositories.pgvector_repo import PgVectorRepository

            return PgVectorRepository(self.settings)

    def create_embedding_engine(self) -> EmbeddingEngine:
        """Create the embedding engine for the current mode."""
        if self.settings.mode == AppMode.LOCAL:
            from deeplens.engines.jina_clip import JinaClipEngine

            return JinaClipEngine(self.settings)
        else:
            from deeplens.engines.gemini_embed import GeminiEmbeddingEngine

            return GeminiEmbeddingEngine(self.settings)

    def create_chat_engine(self) -> ChatEngine:
        """Create the chat / LLM engine for the current mode."""
        if self.settings.mode == AppMode.LOCAL:
            from deeplens.engines.ollama_chat import OllamaChatEngine

            return OllamaChatEngine(self.settings)
        else:
            from deeplens.engines.gemini_chat import GeminiChatEngine

            return GeminiChatEngine(self.settings)
