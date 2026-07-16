"""LanceDB implementation of the DocumentRepository.

Uses LanceDB as an embedded, disk-based vector store. Suitable for local-only deployment
with low memory overhead.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
import structlog

from deeplens.config import Settings
from deeplens.core.models import SearchResult, VectorRecord
from deeplens.core.repository import DocumentRepository

logger = structlog.get_logger(__name__)


class LanceDBRepository(DocumentRepository):
    """LanceDB vector repository."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db_path = settings.lancedb_path
        self._conn: lancedb.DBConnection | None = None
        self._table: lancedb.table.Table | None = None
        self._table_name = "documents"

    def _get_schema(self) -> pa.Schema:
        """Create PyArrow schema matching VectorRecord."""
        dim = self.settings.vector_dim
        return pa.schema([
            pa.field("id", pa.string(), nullable=False),
            pa.field("vector", pa.list_(pa.float32(), dim), nullable=False),
            pa.field("content", pa.string()),
            pa.field("absolute_path", pa.string()),
            pa.field("filename", pa.string()),
            pa.field("parent_directory", pa.string()),
            pa.field("file_type", pa.string()),
            pa.field("mime_type", pa.string()),
            pa.field("chunk_index", pa.int64()),
            pa.field("total_chunks", pa.int64()),
            pa.field("timestamp_start", pa.float64(), nullable=True),
            pa.field("timestamp_end", pa.float64(), nullable=True),
            pa.field("created_at", pa.string()),
            pa.field("file_modified_at", pa.string()),
            pa.field("file_hash", pa.string()),
            pa.field("metadata_json", pa.string()),
        ])

    async def initialize(self) -> None:
        """Connect to LanceDB and ensure the table exists."""
        def _init() -> None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = lancedb.connect(str(self.db_path))
            
            if self._table_name in self._conn.table_names():
                self._table = self._conn.open_table(self._table_name)
                logger.info("lancedb.initialize.open_existing", table=self._table_name)
            else:
                schema = self._get_schema()
                self._table = self._conn.create_table(
                    self._table_name,
                    schema=schema,
                    mode="create"
                )
                logger.info("lancedb.initialize.create_new", table=self._table_name)

        await asyncio.to_thread(_init)

    async def close(self) -> None:
        """Release LanceDB resources."""
        # LanceDB connections don't require explicit close in python client usually,
        # but we set variables to None to release file locks.
        self._table = None
        self._conn = None
        logger.info("lancedb.close")

    async def insert(self, records: list[VectorRecord]) -> int:
        """Insert records into LanceDB."""
        if not records:
            return 0
            
        if self._table is None:
            raise RuntimeError("Repository not initialized. Call initialize() first.")

        data = [r.to_dict() for r in records]

        def _add() -> int:
            self._table.add(data)
            return len(data)

        count = await asyncio.to_thread(_add)
        logger.info("lancedb.insert", count=count)
        return count

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        folder_filter: str | None = None,
        file_type_filter: str | None = None,
    ) -> list[SearchResult]:
        """Search vector table with optional filters."""
        if self._table is None:
            raise RuntimeError("Repository not initialized.")

        # Construct SQL where clause for LanceDB filtering
        where_clauses = []
        if folder_filter:
            # Matches prefix folder path
            where_clauses.append(f"parent_directory LIKE '{folder_filter}%'")
        if file_type_filter:
            where_clauses.append(f"file_type = '{file_type_filter}'")

        where_expr = " AND ".join(where_clauses) if where_clauses else None

        def _query() -> list[SearchResult]:
            q = self._table.search(query_vector).metric("cosine")
            if where_expr:
                q = q.where(where_expr)
            
            # Execute search and fetch pandas DataFrame
            results_df = q.limit(top_k).to_pandas()
            
            output = []
            for i, row in results_df.iterrows():
                # LanceDB cosine search returns distance (usually L2 or cosine distance where 0 is identical)
                # Cosine similarity = 1 - cosine distance
                dist = row.get("_distance", 1.0)
                score = max(0.0, min(1.0, 1.0 - float(dist)))
                
                # Reconstruct VectorRecord
                record_dict = row.to_dict()
                # Clean up distance column
                record_dict.pop("_distance", None)
                # Convert vector back to float list
                record_dict["vector"] = list(row["vector"])
                
                # Check for None values for float fields
                for f in ("timestamp_start", "timestamp_end"):
                    if record_dict.get(f) is not None and (
                        isinstance(record_dict[f], float) and os.isnan(record_dict[f])
                    ):
                        record_dict[f] = None

                rec = VectorRecord.from_dict(record_dict)
                output.append(SearchResult(record=rec, score=score, rank=len(output) + 1))
            return output

        return await asyncio.to_thread(_query)

    async def get_by_file_hash(self, file_hash: str) -> list[VectorRecord]:
        """Look up records by source file hash."""
        if self._table is None:
            return []

        def _query() -> list[VectorRecord]:
            df = self._table.search().where(f"file_hash = '{file_hash}'").to_pandas()
            return [VectorRecord.from_dict(row.to_dict()) for _, row in df.iterrows()]

        return await asyncio.to_thread(_query)

    async def delete_by_path(self, absolute_path: str) -> int:
        """Delete all records associated with a file path."""
        if self._table is None:
            return 0

        def _delete() -> None:
            # LanceDB supports deleting via filter expression
            # Double escape single quotes in paths
            safe_path = absolute_path.replace("'", "''")
            self._table.delete(f"absolute_path = '{safe_path}'")

        await asyncio.to_thread(_delete)
        logger.info("lancedb.delete_by_path", path=absolute_path)
        return 1

    async def delete_by_folder(self, folder_path: str) -> int:
        """Delete all records within a folder subtree."""
        if self._table is None:
            return 0

        def _delete() -> None:
            safe_folder = folder_path.replace("'", "''")
            self._table.delete(f"parent_directory LIKE '{safe_folder}%' OR absolute_path LIKE '{safe_folder}%'")

        await asyncio.to_thread(_delete)
        logger.info("lancedb.delete_by_folder", folder=folder_path)
        return 1

    async def list_indexed_folders(self) -> list[str]:
        """Return all unique parent directories indexed."""
        if self._table is None:
            return []

        def _list() -> list[str]:
            df = self._table.to_pandas()
            if df.empty:
                return []
            return sorted(df["parent_directory"].unique().tolist())

        return await asyncio.to_thread(_list)

    async def get_file_chunks(self, absolute_path: str) -> list[VectorRecord]:
        """Return all chunks for a specific file, ordered by chunk_index."""
        if self._table is None:
            return []

        def _query() -> list[VectorRecord]:
            safe_path = absolute_path.replace("'", "''")
            df = self._table.search().where(f"absolute_path = '{safe_path}'").to_pandas()
            if df.empty:
                return []
            df = df.sort_values(by="chunk_index")
            return [VectorRecord.from_dict(row.to_dict()) for _, row in df.iterrows()]

        return await asyncio.to_thread(_query)

    async def get_stats(self) -> dict[str, int]:
        """Return database statistics."""
        if self._table is None:
            return {"total_records": 0, "total_files": 0, "total_folders": 0}

        def _stats() -> dict[str, int]:
            df = self._table.to_pandas()
            if df.empty:
                return {"total_records": 0, "total_files": 0, "total_folders": 0}
            
            stats = {
                "total_records": len(df),
                "total_files": df["absolute_path"].nunique(),
                "total_folders": df["parent_directory"].nunique(),
            }
            # Add counts per file_type
            for ft in df["file_type"].unique():
                stats[f"count_{ft}"] = len(df[df["file_type"] == ft])
            return stats

        return await asyncio.to_thread(_stats)
