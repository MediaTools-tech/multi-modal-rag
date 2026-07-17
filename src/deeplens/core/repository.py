"""Abstract base class for vector database repositories.

All database operations go through this interface, allowing seamless
swapping between LanceDB (local) and pgvector (cloud) backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from deeplens.core.models import SearchResult, VectorRecord


class DocumentRepository(ABC):
    """Abstract vector database repository.

    Implementations must handle:
    - Table/collection creation with dynamic vector dimensions
    - CRUD operations for VectorRecord entries
    - ANN (approximate nearest neighbor) search with metadata filtering
    - Deduplication via file_hash lookups
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the database connection and create tables/collections if needed."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the database connection and release resources."""
        ...

    @abstractmethod
    async def insert(self, records: list[VectorRecord]) -> int:
        """Insert vector records into the database.

        Args:
            records: List of VectorRecord objects to insert.

        Returns:
            Number of records successfully inserted.
        """
        ...

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        folder_filter: str | None = None,
        file_type_filter: str | None = None,
        record_types: list[str] | None = None,
    ) -> list[SearchResult]:
        """Perform ANN search with optional metadata filtering.

        Args:
            query_vector: The query embedding vector.
            top_k: Number of results to return.
            folder_filter: If set, only search within this directory subtree.
            file_type_filter: If set, only search this file type (document/image/audio/video).
            record_types: If set, only return records whose ``record_type`` is in
                this list (e.g. ``["summary"]`` for document-level retrieval).

        Returns:
            List of SearchResult objects sorted by relevance.
        """
        ...

    @abstractmethod
    async def get_by_file_hash(self, file_hash: str) -> list[VectorRecord]:
        """Look up records by source file hash (for deduplication).

        Args:
            file_hash: SHA-256 hash of the source file.

        Returns:
            List of existing records for this file hash.
        """
        ...

    @abstractmethod
    async def delete_by_path(self, absolute_path: str) -> int:
        """Delete all records associated with a source file path.

        Args:
            absolute_path: The absolute path of the source file.

        Returns:
            Number of records deleted.
        """
        ...

    @abstractmethod
    async def delete_by_folder(self, folder_path: str) -> int:
        """Delete all records within a folder subtree.

        Args:
            folder_path: The folder path prefix to match.

        Returns:
            Number of records deleted.
        """
        ...

    @abstractmethod
    async def list_indexed_folders(self) -> list[str]:
        """Return all unique parent directories that have been indexed.

        Returns:
            Sorted list of unique parent directory paths.
        """
        ...

    @abstractmethod
    async def search_lexical(
        self,
        query: str,
        top_k: int = 50,
        folder_filter: str | None = None,
        file_type_filter: str | None = None,
    ) -> list[SearchResult]:
        """Corpus-wide lexical / full-text search.

        Unlike vector search (which only sees its own top-k), this ranks *every*
        indexed document by keyword relevance, enabling keyword queries to surface
        documents the vector search would not. Returns ranked candidates with a
        normalized lexical relevance in ``score`` (higher = better).

        Args:
            query: The keyword query.
            top_k: Maximum number of lexical candidates to return.
            folder_filter: Optional directory subtree to scope the search.
            file_type_filter: Optional file-type category filter.

        Returns:
            List of SearchResult objects ranked by lexical relevance.
        """
        ...

    @abstractmethod
    async def get_file_chunks(self, absolute_path: str) -> list[VectorRecord]:
        """Return all chunks for a specific file, ordered by chunk_index.

        Args:
            absolute_path: Path to the source file.

        Returns:
            List of VectorRecord objects ordered by chunk_index.
        """
        ...

    @abstractmethod
    async def get_stats(self) -> dict[str, int]:
        """Return database statistics.

        Returns:
            Dictionary with keys: total_records, total_files, total_folders,
            and counts per file_type.
        """
        ...

    async def file_needs_reindex(self, absolute_path: str, file_hash: str) -> bool:
        """Check whether a file needs (re-)indexing.

        A file needs indexing if:
        - No records exist for this file_hash, or
        - The existing records have a different file_hash (file was modified).

        Args:
            absolute_path: Path to the file.
            file_hash: Current SHA-256 hash of the file.

        Returns:
            True if the file should be (re-)indexed.
        """
        existing = await self.get_by_file_hash(file_hash)
        if not existing:
            # Check if there are stale records for this path
            stale = await self.get_file_chunks(absolute_path)
            return True  # Either new file or content changed
        return False
