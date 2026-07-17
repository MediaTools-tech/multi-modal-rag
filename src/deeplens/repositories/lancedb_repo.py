"""LanceDB implementation of the DocumentRepository.

Uses LanceDB as an embedded, disk-based vector store. Suitable for local-only deployment
with low memory overhead.
"""

from __future__ import annotations

import asyncio
from math import isnan
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
import structlog

from deeplens.config import Settings
from deeplens.core.models import SearchResult, VectorRecord
from deeplens.core.repository import DocumentRepository
from deeplens.search.lexical import bm25_scores, tokenize

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
            pa.field("record_type", pa.string()),
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
            pa.field("summary", pa.string()),
        ])

    def _ensure_summary_columns(self) -> None:
        """Best-effort migration: add record_type/summary columns to existing tables."""
        if self._table is None:
            return
        existing = {f.name for f in self._table.schema}
        missing = [c for c in ("record_type", "summary") if c not in existing]
        if not missing:
            return
        try:
            # LanceDB add_columns: name -> default literal value.
            defaults = {c: ("" if c == "summary" else "chunk") for c in missing}
            self._table.add_columns(defaults)  # type: ignore
            logger.info("lancedb.migrate.added_columns", columns=missing)
        except Exception as e:
            logger.warn(
                "lancedb.migrate.failed",
                columns=missing,
                error=str(e),
                hint="Re-index the folder to add summary support.",
            )

    def _ensure_fts_index(self) -> None:
        """Best-effort creation of a LanceDB full-text search index on ``content``."""
        if self._table is None:
            return
        try:
            self._table.create_fts_index("content")
            logger.info("lancedb.fts_index.ready")
        except Exception as e:
            logger.warn(
                "lancedb.fts_index.failed",
                error=str(e),
                hint="Will fall back to an in-Python BM25 scan for lexical search.",
            )

    def _row_to_search_result(self, row: Any, score: float, *, drop_extra: tuple[str, ...] = ("_distance", "_score")) -> SearchResult:
        """Convert a LanceDB result row (pandas Series / mapping) to a SearchResult."""
        record_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        for col in drop_extra:
            record_dict.pop(col, None)
        record_dict["vector"] = list(record_dict.get("vector") or [])
        for f in ("timestamp_start", "timestamp_end"):
            v = record_dict.get(f)
            if v is not None and isinstance(v, float) and isnan(v):
                record_dict[f] = None
        rec = VectorRecord.from_dict(record_dict)
        return SearchResult(record=rec, score=score, rank=0)

    async def initialize(self) -> None:
        """Connect to LanceDB and ensure the table exists."""
        def _init() -> None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = lancedb.connect(str(self.db_path))
            
            if self._table_name in self._conn.table_names():
                self._table = self._conn.open_table(self._table_name)
                logger.info("lancedb.initialize.open_existing", table=self._table_name)
                self._ensure_summary_columns()
                self._ensure_fts_index()
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
        record_types: list[str] | None = None,
    ) -> list[SearchResult]:
        """Search vector table with optional filters.

        Args:
            query_vector: Query embedding.
            top_k: Max number of results to return.
            folder_filter: Limit to a directory subtree.
            file_type_filter: Limit to a file type category.
            record_types: If provided, only return records whose ``record_type``
                is in this list (e.g. ``["summary"]`` for file-level retrieval).
                Filtering is done in-memory after a larger candidate fetch so the
                method works even on schemas created before summary columns existed.
        """
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

        # When filtering by record_type we over-fetch then trim in Python so we
        # still return up to top_k matches.
        fetch_limit = top_k * 5 if record_types else top_k

        def _query() -> list[SearchResult]:
            q = self._table.search(query_vector).metric("cosine")
            if where_expr:
                q = q.where(where_expr)

            # Execute search and fetch pandas DataFrame
            results_df = q.limit(fetch_limit).to_pandas()

            output = []
            for i, row in results_df.iterrows():
                # LanceDB cosine search returns distance (usually L2 or cosine distance where 0 is identical)
                # Cosine similarity = 1 - cosine distance
                dist = row.get("_distance", 1.0)
                score = max(0.0, min(1.0, 1.0 - float(dist)))

                rec = self._row_to_search_result(row, score)

                # In-memory record_type filter (tolerant of pre-summary schemas).
                if record_types is not None:
                    if rec.record.record_type not in record_types:
                        continue

                output.append(rec)
                rec.rank = len(output)
                if len(output) >= top_k:
                    break
            return output

        return await asyncio.to_thread(_query)

    async def search_lexical(
        self,
        query: str,
        top_k: int = 50,
        folder_filter: str | None = None,
        file_type_filter: str | None = None,
    ) -> list[SearchResult]:
        """Corpus-wide lexical search.

        Uses the LanceDB FTS index when available; falls back to an in-Python
        BM25 scan over the (filtered) table so keyword search still covers every
        document when the FTS index could not be built.
        """
        if self._table is None:
            raise RuntimeError("Repository not initialized.")

        where_clauses = []
        if folder_filter:
            where_clauses.append(f"parent_directory LIKE '{folder_filter}%'")
        if file_type_filter:
            where_clauses.append(f"file_type = '{file_type_filter}'")
        where_expr = " AND ".join(where_clauses) if where_clauses else None

        def _query() -> list[SearchResult]:
            # Preferred path: native full-text search index.
            try:
                q = self._table.search(query, query_type="fts")
                if where_expr:
                    q = q.where(where_expr)
                df = q.limit(top_k).to_pandas()
                results: list[SearchResult] = []
                for _, row in df.iterrows():
                    raw = float(row.get("_score", 0.0) or 0.0)
                    rec = self._row_to_search_result(row, abs(raw))
                    results.append(rec)
                max_score = max((r.score for r in results), default=1.0) or 1.0
                for r in results:
                    r.score = r.score / max_score if max_score > 0 else 0.0
                results.sort(key=lambda r: r.score, reverse=True)
                for i, r in enumerate(results):
                    r.rank = i + 1
                return results
            except Exception as e:
                logger.warn("lancedb.search_lexical.fts_failed", error=str(e))

            # Fallback: load filtered rows and score with in-Python BM25.
            q = self._table.search()
            if where_expr:
                q = q.where(where_expr)
            df = q.to_pandas()
            if df.empty:
                return []

            docs = [(str(row["id"]), str(row.get("content") or "")) for _, row in df.iterrows()]
            scores = bm25_scores(tokenize(query), docs)
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            max_score = ranked[0][1] if ranked and ranked[0][1] > 0 else 1.0

            by_id = {str(row["id"]): row for _, row in df.iterrows()}
            results = []
            for doc_id, sc in ranked:
                if sc <= 0:
                    continue
                rec = self._row_to_search_result(by_id[doc_id], sc / max_score)
                results.append(rec)
            for i, r in enumerate(results):
                r.rank = i + 1
            return results

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
