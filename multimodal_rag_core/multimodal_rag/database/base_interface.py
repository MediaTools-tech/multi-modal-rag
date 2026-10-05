from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any

from multimodal_rag.core.models import SearchResult, VectorRecord


class DocumentRepository(ABC):
    """Storage-agnostic interface for chunk/summary storage and retrieval."""

    @abstractmethod
    def initialize(self) -> None:
        ...

    @abstractmethod
    def insert(self, records: list[VectorRecord]) -> int:
        ...

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        record_types: list[str] | None = None,
        folder_prefix: str | None = None,
    ) -> list[SearchResult]:
        ...

    @abstractmethod
    def search_lexical(
        self,
        query: str,
        top_k: int = 50,
        record_types: list[str] | None = None,
        folder_prefix: str | None = None,
    ) -> list[SearchResult]:
        ...

    @abstractmethod
    def get_by_file_hash(self, file_hash: str) -> list[VectorRecord]:
        ...

    @abstractmethod
    def get_file_chunks(self, source_path: str) -> list[VectorRecord]:
        ...

    @abstractmethod
    def delete_by_path(self, source_path: str) -> int:
        ...

    @abstractmethod
    def delete_by_folder(self, folder_path: str) -> int:
        ...

    @abstractmethod
    def clear(self) -> int:
        """Delete ALL records (full reset). Returns the deleted count."""
        ...

    @abstractmethod
    def indexed_file_hashes(self) -> set[str]:
        """Distinct non-empty file_hash values present in the store."""
        ...

    @abstractmethod
    def indexed_paths(self) -> set[str]:
        """Distinct source_path values present in the store."""
        ...

    @abstractmethod
    def get_stats(self) -> dict[str, int]:
        ...

    @abstractmethod
    def reindex(self) -> None:
        ...

    def close(self) -> None:
        return None

    def insert_chunk(
        self,
        chunk_id: str,
        vector: list[float],
        text_content: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        try:
            normalized_id = str(uuid.UUID(str(chunk_id)))
        except (ValueError, AttributeError, TypeError):
            # Non-UUID ids are namespaced so uuid-keyed backends still accept them.
            normalized_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(chunk_id)))
        record = VectorRecord(
            id=normalized_id,
            vector=vector,
            content=text_content,
            metadata=metadata or {},
        )
        return self.insert([record])

    def file_needs_reindex(self, source_path: str, file_hash: str) -> bool:
        existing = self.get_file_chunks(source_path)
        if not existing:
            return True
        return existing[0].file_hash != file_hash

    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        limit: int = 10,
        **kwargs: Any,
    ) -> list[SearchResult]:
        from multimodal_rag.search.hybrid import hybrid_search

        results, _ = hybrid_search(
            self, query_text, query_vector, top_k=limit, **kwargs
        )
        return results

    def hybrid_search_grouped(
        self,
        query_text: str,
        query_vector: list[float],
        limit: int = 10,
        **kwargs: Any,
    ):
        from multimodal_rag.search.hybrid import hybrid_search

        return hybrid_search(self, query_text, query_vector, top_k=limit, **kwargs)
