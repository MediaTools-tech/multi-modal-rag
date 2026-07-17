"""PostgreSQL + pgvector implementation of the DocumentRepository.

Exposes high-performance, enterprise-ready vector storage. Suitable for cloud showcase.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg
import structlog

from deeplens.config import Settings
from deeplens.core.models import SearchResult, VectorRecord
from deeplens.core.repository import DocumentRepository

logger = structlog.get_logger(__name__)


class PgVectorRepository(DocumentRepository):
    """PgVector database repository using asyncpg."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pool: asyncpg.Pool | None = None
        self._table_name = "documents"

    async def initialize(self) -> None:
        """Create pool and create database table with pgvector extension."""
        logger.info("pgvector.initialize.start", dsn=self.settings.postgres_dsn)
        
        # Create connection pool
        self._pool = await asyncpg.create_pool(
            host=self.settings.postgres_host,
            port=self.settings.postgres_port,
            database=self.settings.postgres_db,
            user=self.settings.postgres_user,
            password=self.settings.postgres_password,
            min_size=1,
            max_size=10,
        )

        async with self._pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create table with vector column using halfvec for memory efficiency (pgvector >= 0.5)
            # Falling back to regular vector if halfvec is not fully supported in target setup,
            # but let's define it as `vector(dim)` or `halfvec(dim)` based on standard.
            # Using halfvec type (vector of 2-byte floats):
            dim = self.settings.vector_dim
            
            # Check pgvector version or capabilities to see if halfvec is supported
            # halfvec was introduced in pgvector 0.5.0
            try:
                await conn.execute(f"CREATE TABLE IF NOT EXISTS {self._table_name} (id UUID PRIMARY KEY);")
                # Add vector column if it doesn't exist
                await conn.execute(
                    f"ALTER TABLE {self._table_name} ADD COLUMN IF NOT EXISTS vector halfvec({dim});"
                )
                logger.info("pgvector.initialize.halfvec_supported", dim=dim)
            except Exception:
                # Fallback to float4 vector if halfvec isn't supported or fails
                logger.warn("pgvector.initialize.halfvec_failed_falling_back_to_vector")
                await conn.execute(
                    f"ALTER TABLE {self._table_name} ADD COLUMN IF NOT EXISTS vector vector({dim});"
                )

            # Create other fields
            await conn.execute(f"""
                ALTER TABLE {self._table_name}
                ADD COLUMN IF NOT EXISTS content TEXT,
                ADD COLUMN IF NOT EXISTS record_type TEXT,
                ADD COLUMN IF NOT EXISTS absolute_path TEXT,
                ADD COLUMN IF NOT EXISTS filename TEXT,
                ADD COLUMN IF NOT EXISTS parent_directory TEXT,
                ADD COLUMN IF NOT EXISTS file_type TEXT,
                ADD COLUMN IF NOT EXISTS mime_type TEXT,
                ADD COLUMN IF NOT EXISTS chunk_index INT,
                ADD COLUMN IF NOT EXISTS total_chunks INT,
                ADD COLUMN IF NOT EXISTS timestamp_start DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS timestamp_end DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS created_at TEXT,
                ADD COLUMN IF NOT EXISTS file_modified_at TEXT,
                ADD COLUMN IF NOT EXISTS file_hash TEXT,
                ADD COLUMN IF NOT EXISTS metadata_json TEXT,
                ADD COLUMN IF NOT EXISTS summary TEXT;
            """)

            # Create HNSW indexes for fast approximate search
            # We use cosine distance index (<=> operator)
            try:
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS documents_vector_idx ON {self._table_name} "
                    f"USING hnsw (vector halfvec_cosine_ops);"
                )
            except Exception:
                # Fallback index
                try:
                    await conn.execute(
                        f"CREATE INDEX IF NOT EXISTS documents_vector_idx ON {self._table_name} "
                        f"USING hnsw (vector vector_cosine_ops);"
                    )
                except Exception as e:
                    logger.warn("pgvector.initialize.index_creation_failed", error=str(e))

            # Full-text search: a stored tsvector column + GIN index gives a
            # real corpus-wide keyword index (covers every document, not just
            # the vector top-k).
            await conn.execute(f"""
                ALTER TABLE {self._table_name}
                ADD COLUMN IF NOT EXISTS content_tsv tsvector
                GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED;
            """)
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_docs_tsv ON {self._table_name} USING GIN(content_tsv);"
            )

            # Regular indexes for metadata filters
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_docs_parent_dir ON {self._table_name} (parent_directory);"
            )
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_docs_file_hash ON {self._table_name} (file_hash);"
            )
            
        logger.info("pgvector.initialize.success")

    async def close(self) -> None:
        """Close pool connection."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("pgvector.close")

    async def insert(self, records: list[VectorRecord]) -> int:
        """Insert records using COPY or batch insert."""
        if not records or not self._pool:
            return 0

        async with self._pool.acquire() as conn:
            # Prepare rows
            rows = []
            for r in records:
                # pgvector expects vectors as string '[1, 2, 3...]' or list
                # asyncpg supports passing list directly or formatted string
                rows.append((
                    r.id,
                    r.vector,
                    r.content,
                    r.record_type,
                    r.absolute_path,
                    r.filename,
                    r.parent_directory,
                    r.file_type,
                    r.mime_type,
                    r.chunk_index,
                    r.total_chunks,
                    r.timestamp_start,
                    r.timestamp_end,
                    r.created_at,
                    r.file_modified_at,
                    r.file_hash,
                    r.metadata_json,
                    r.summary,
                ))

            # Batch insert using executemany
            query = f"""
                INSERT INTO {self._table_name} (
                    id, vector, content, record_type, absolute_path, filename, parent_directory, file_type, mime_type,
                    chunk_index, total_chunks, timestamp_start, timestamp_end, created_at, file_modified_at, file_hash,
                    metadata_json, summary
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
                ON CONFLICT (id) DO UPDATE SET
                    vector = EXCLUDED.vector,
                    content = EXCLUDED.content,
                    record_type = EXCLUDED.record_type,
                    absolute_path = EXCLUDED.absolute_path,
                    filename = EXCLUDED.filename,
                    parent_directory = EXCLUDED.parent_directory,
                    file_type = EXCLUDED.file_type,
                    mime_type = EXCLUDED.mime_type,
                    chunk_index = EXCLUDED.chunk_index,
                    total_chunks = EXCLUDED.total_chunks,
                    timestamp_start = EXCLUDED.timestamp_start,
                    timestamp_end = EXCLUDED.timestamp_end,
                    created_at = EXCLUDED.created_at,
                    file_modified_at = EXCLUDED.file_modified_at,
                    file_hash = EXCLUDED.file_hash,
                    metadata_json = EXCLUDED.metadata_json,
                    summary = EXCLUDED.summary;
            """
            await conn.executemany(query, rows)
            
        logger.info("pgvector.insert", count=len(records))
        return len(records)

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        folder_filter: str | None = None,
        file_type_filter: str | None = None,
        record_types: list[str] | None = None,
    ) -> list[SearchResult]:
        """Perform similarity search using cosine distance (<=> operator in pgvector).

        See :meth:`LanceDBRepository.search` for the meaning of ``record_types``.
        """
        if not self._pool:
            return []

        # pgvector query: SELECT *, vector <=> $1 as distance FROM documents ... ORDER BY distance LIMIT $2
        # Cosine distance <=> yields values from 0 (same direction) to 2 (opposite).
        # Cosine similarity = 1 - (vector <=> query_vector)
        where_clauses = []
        params: list[Any] = [query_vector]

        param_counter = 2
        if folder_filter:
            where_clauses.append(f"parent_directory LIKE ${param_counter}")
            params.append(f"{folder_filter}%")
            param_counter += 1
        if file_type_filter:
            where_clauses.append(f"file_type = ${param_counter}")
            params.append(file_type_filter)
            param_counter += 1
        if record_types:
            where_clauses.append(f"record_type = ANY(${param_counter})")
            params.append(record_types)
            param_counter += 1

        where_expr = ""
        if where_clauses:
            where_expr = "WHERE " + " AND ".join(where_clauses)

        query = f"""
            SELECT *, (vector <=> $1) as distance
            FROM {self._table_name}
            {where_expr}
            ORDER BY distance ASC
            LIMIT ${param_counter};
        """
        params.append(top_k)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        results = []
        for i, row in enumerate(rows):
            # Parse row into record
            rec_dict = dict(row)
            # Remove distance from record args
            distance = rec_dict.pop("distance")
            # Convert halfvec/vector format back to list of floats if it comes back as string or array
            vec_val = rec_dict["vector"]
            # asyncpg might return custom string format or list. Normalize:
            if isinstance(vec_val, str):
                rec_dict["vector"] = [float(x) for x in vec_val.strip("[]").split(",")]
            elif isinstance(vec_val, list):
                rec_dict["vector"] = [float(x) for x in vec_val]
            else:
                # If pgvector casts to array / list
                rec_dict["vector"] = list(vec_val)

            # Cosine distance to Cosine similarity (values are between 0 and 2, mapped to similarity [0, 1])
            score = max(0.0, min(1.0, 1.0 - float(distance)))

            # Handle UUID conversions
            rec_dict["id"] = str(rec_dict["id"])

            rec = VectorRecord.from_dict(rec_dict)
            results.append(SearchResult(record=rec, score=score, rank=i + 1))

        return results

    async def search_lexical(
        self,
        query: str,
        top_k: int = 50,
        folder_filter: str | None = None,
        file_type_filter: str | None = None,
    ) -> list[SearchResult]:
        """Corpus-wide lexical search using Postgres full-text search (tsvector)."""
        if not self._pool:
            return []

        params: list[Any] = [query]  # $1: the keyword query
        where_clauses = ["content_tsv @@ plainto_tsquery('english', $1)"]
        param_counter = 2
        if folder_filter:
            where_clauses.append(f"parent_directory LIKE ${param_counter}")
            params.append(f"{folder_filter}%")
            param_counter += 1
        if file_type_filter:
            where_clauses.append(f"file_type = ${param_counter}")
            params.append(file_type_filter)
            param_counter += 1

        where_expr = " AND ".join(where_clauses)
        query_sql = f"""
            SELECT *, ts_rank_cd(content_tsv, plainto_tsquery('english', $1)) AS lex_score
            FROM {self._table_name}
            WHERE {where_expr}
            ORDER BY lex_score DESC
            LIMIT ${param_counter};
        """
        params.append(top_k)

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query_sql, *params)
        except Exception as e:
            logger.warn("pgvector.search_lexical.failed", error=str(e))
            return []

        results: list[SearchResult] = []
        raw_scores = [float(row["lex_score"] or 0.0) for row in rows]
        max_score = max(raw_scores) if raw_scores else 1.0
        for i, row in enumerate(rows):
            rec_dict = dict(row)
            rec_dict.pop("lex_score", None)
            vec_val = rec_dict["vector"]
            if isinstance(vec_val, str):
                rec_dict["vector"] = [float(x) for x in vec_val.strip("[]").split(",")]
            elif isinstance(vec_val, list):
                rec_dict["vector"] = [float(x) for x in vec_val]
            else:
                rec_dict["vector"] = list(vec_val)
            rec_dict["id"] = str(rec_dict["id"])
            rec = VectorRecord.from_dict(rec_dict)
            # Normalize ts_rank to [0, 1] for consistent ranking semantics.
            score = (float(row["lex_score"] or 0.0) / max_score) if max_score > 0 else 0.0
            results.append(SearchResult(record=rec, score=score, rank=i + 1))
        return results

    async def get_by_file_hash(self, file_hash: str) -> list[VectorRecord]:
        """Look up records by source file hash."""
        if not self._pool:
            return []

        query = f"SELECT * FROM {self._table_name} WHERE file_hash = $1;"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, file_hash)
            
        return [self._row_to_record(row) for row in rows]

    async def delete_by_path(self, absolute_path: str) -> int:
        """Delete records associated with absolute file path."""
        if not self._pool:
            return 0

        query = f"DELETE FROM {self._table_name} WHERE absolute_path = $1;"
        async with self._pool.acquire() as conn:
            result = await conn.execute(query, absolute_path)
            
        # extract count from command tag e.g. "DELETE 5"
        count = 0
        if result and result.startswith("DELETE"):
            count = int(result.split()[1])
            
        logger.info("pgvector.delete_by_path", path=absolute_path, deleted=count)
        return count

    async def delete_by_folder(self, folder_path: str) -> int:
        """Delete records within a folder subtree."""
        if not self._pool:
            return 0

        query = f"""
            DELETE FROM {self._table_name}
            WHERE parent_directory LIKE $1 OR absolute_path LIKE $1;
        """
        async with self._pool.acquire() as conn:
            result = await conn.execute(query, f"{folder_path}%")
            
        count = 0
        if result and result.startswith("DELETE"):
            count = int(result.split()[1])
            
        logger.info("pgvector.delete_by_folder", folder=folder_path, deleted=count)
        return count

    async def list_indexed_folders(self) -> list[str]:
        """Return unique indexed parent directories."""
        if not self._pool:
            return []

        query = f"SELECT DISTINCT parent_directory FROM {self._table_name} ORDER BY parent_directory;"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query)
            
        return [row["parent_directory"] for row in rows if row["parent_directory"]]

    async def get_file_chunks(self, absolute_path: str) -> list[VectorRecord]:
        """Return all chunks for a specific file, ordered by chunk_index."""
        if not self._pool:
            return []

        query = f"SELECT * FROM {self._table_name} WHERE absolute_path = $1 ORDER BY chunk_index;"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, absolute_path)
            
        return [self._row_to_record(row) for row in rows]

    async def get_stats(self) -> dict[str, int]:
        """Return stats."""
        if not self._pool:
            return {"total_records": 0, "total_files": 0, "total_folders": 0}

        async with self._pool.acquire() as conn:
            tot_rec = await conn.fetchval(f"SELECT COUNT(*) FROM {self._table_name};")
            tot_files = await conn.fetchval(f"SELECT COUNT(DISTINCT absolute_path) FROM {self._table_name};")
            tot_folders = await conn.fetchval(f"SELECT COUNT(DISTINCT parent_directory) FROM {self._table_name};")
            
            stats = {
                "total_records": tot_rec or 0,
                "total_files": tot_files or 0,
                "total_folders": tot_folders or 0,
            }

            # counts per file_type
            type_rows = await conn.fetch(
                f"SELECT file_type, COUNT(*) as cnt FROM {self._table_name} GROUP BY file_type;"
            )
            for row in type_rows:
                if row["file_type"]:
                    stats[f"count_{row['file_type']}"] = row["cnt"]

            return stats

    def _row_to_record(self, row: asyncpg.Record) -> VectorRecord:
        """Convert an asyncpg Record to a VectorRecord."""
        rec_dict = dict(row)
        rec_dict["id"] = str(rec_dict["id"])
        vec_val = rec_dict["vector"]
        if isinstance(vec_val, str):
            rec_dict["vector"] = [float(x) for x in vec_val.strip("[]").split(",")]
        elif isinstance(vec_val, list):
            rec_dict["vector"] = [float(x) for x in vec_val]
        else:
            rec_dict["vector"] = list(vec_val)
        return VectorRecord.from_dict(rec_dict)
