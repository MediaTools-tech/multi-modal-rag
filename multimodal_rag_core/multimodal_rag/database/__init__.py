from __future__ import annotations

from multimodal_rag.config import DBEngine, Settings, get_settings
from multimodal_rag.database.base_interface import DocumentRepository


def get_repository(settings: Settings | None = None) -> DocumentRepository:
    settings = settings or get_settings()
    if settings.ACTIVE_DB_ENGINE is DBEngine.POSTGRES:
        from multimodal_rag.database.postgres_provider import PostgresProvider

        return PostgresProvider(settings)
    from multimodal_rag.database.lancedb_provider import LanceDBProvider

    return LanceDBProvider(settings)


__all__ = ["DocumentRepository", "get_repository"]
