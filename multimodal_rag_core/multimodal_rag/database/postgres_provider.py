from __future__ import annotations

import json
import logging
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from multimodal_rag.config import Settings
from multimodal_rag.core.models import SearchResult, VectorRecord
from multimodal_rag.database.base_interface import DocumentRepository

logger = logging.getLogger(__name__)


def _like_escape(value: str) -> str:
    """Escape LIKE wildcards so folder prefixes match literally."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class PostgresProvider(DocumentRepository):
    """PostgreSQL + pgvector + native tsvector full-text search."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dsn = settings.postgres_dsn
        self.table = settings.DB_TABLE_NAME
        self.dim = settings.EMBEDDING_DIMENSION

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self.dsn, row_factory=dict_row)
        register_vector(conn)
        return conn

    @staticmethod
    def _quote_ident(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    def _ensure_database(self) -> None:
        """Create the configured database on first run (fresh Postgres)."""
        from urllib.parse import urlparse

        dbname = urlparse(self.dsn).path.lstrip("/").split("?", 1)[0]
        if not dbname:
            return
        admin_dsn = f"{self.dsn.rsplit('/', 1)[0]}/postgres"
        conn = psycopg.connect(admin_dsn, row_factory=dict_row)
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (dbname,))
                if cur.fetchone() is None:
                    cur.execute(f"CREATE DATABASE {self._quote_ident(dbname)};")
                    logger.info("Created Postgres database %r", dbname)
        finally:
            conn.close()

    def initialize(self) -> None:
        self._ensure_database()
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(
                    f"CREATE TABLE IF NOT EXISTS {self.table} (id uuid PRIMARY KEY);"
                )
                cur.execute(
                    f"ALTER TABLE {self.table} "
                    f"ADD COLUMN IF NOT EXISTS embedding vector({self.dim});"
                )
                cur.execute(
                    f"""
                    ALTER TABLE {self.table}
                        ADD COLUMN IF NOT EXISTS content text,
                        ADD COLUMN IF NOT EXISTS record_type text,
                        ADD COLUMN IF NOT EXISTS source_path text,
                        ADD COLUMN IF NOT EXISTS filename text,
                        ADD COLUMN IF NOT EXISTS parent_directory text,
                        ADD COLUMN IF NOT EXISTS file_type text,
                        ADD COLUMN IF NOT EXISTS mime_type text,
                        ADD COLUMN IF NOT EXISTS chunk_index integer,
                        ADD COLUMN IF NOT EXISTS total_chunks integer,
                        ADD COLUMN IF NOT EXISTS timestamp_start double precision,
                        ADD COLUMN IF NOT EXISTS timestamp_end double precision,
                        ADD COLUMN IF NOT EXISTS created_at text,
                        ADD COLUMN IF NOT EXISTS file_modified_at text,
                        ADD COLUMN IF NOT EXISTS file_hash text,
                        ADD COLUMN IF NOT EXISTS metadata_json text,
                        ADD COLUMN IF NOT EXISTS summary text;
                    """
                )
                cur.execute(
                    f"""
                    ALTER TABLE {self.table}
                        ADD COLUMN IF NOT EXISTS content_tsv tsvector
                        GENERATED ALWAYS AS (
                            to_tsvector('english', coalesce(content, ''))
                        ) STORED;
                    """
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {self.table}_embedding_idx "
                    f"ON {self.table} USING hnsw (embedding vector_cosine_ops);"
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {self.table}_tsv_idx "
                    f"ON {self.table} USING gin (content_tsv);"
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {self.table}_path_idx "
                    f"ON {self.table} (source_path);"
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {self.table}_hash_idx "
                    f"ON {self.table} (file_hash);"
                )
                cur.execute(
                    "SELECT format_type(atttypid, atttypmod) AS type "
                    "FROM pg_attribute "
                    "WHERE attrelid = %s::regclass AND attname = 'embedding';",
                    (self.table,),
                )
                row = cur.fetchone()
            conn.commit()
            actual = row["type"] if row else None
            expected = f"vector({self.dim})"
            if actual and actual != expected:
                raise ValueError(
                    f"Table {self.table!r} has embedding {actual}, configured "
                    f"EMBEDDING_DIMENSION expects {expected}. Reset the index "
                    f"(GUI: File -> Reset index... / CLI: mrag-ingest reset-index) "
                    f"to rebuild the table, or use a new DB_TABLE_NAME."
                )
        finally:
            conn.close()
        logger.info("PostgresProvider initialized (table=%s, dim=%s)", self.table, self.dim)

    def insert(self, records: list[VectorRecord]) -> int:
        if not records:
            return 0
        rows = [
            (
                record.id,
                record.vector,
                record.content,
                record.record_type,
                record.source_path,
                record.filename,
                record.parent_directory,
                record.file_type,
                record.mime_type,
                record.chunk_index,
                record.total_chunks,
                record.timestamp_start,
                record.timestamp_end,
                record.created_at,
                record.file_modified_at,
                record.file_hash,
                json.dumps(record.metadata),
                record.summary,
            )
            for record in records
        ]
        query = f"""
            INSERT INTO {self.table} (
                id, embedding, content, record_type, source_path, filename,
                parent_directory, file_type, mime_type, chunk_index, total_chunks,
                timestamp_start, timestamp_end, created_at, file_modified_at,
                file_hash, metadata_json, summary
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                content = EXCLUDED.content,
                record_type = EXCLUDED.record_type,
                source_path = EXCLUDED.source_path,
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
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.executemany(query, rows)
            conn.commit()
        finally:
            conn.close()
        return len(rows)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        record_types: list[str] | None = None,
        folder_prefix: str | None = None,
    ) -> list[SearchResult]:
        where: list[str] = []
        params: list[Any] = [query_vector]
        if record_types:
            where.append("record_type = ANY(%s)")
            params.append(record_types)
        if folder_prefix:
            where.append("source_path LIKE %s")
            params.append(f"{_like_escape(folder_prefix)}%")
        where_expr = f"WHERE {' AND '.join(where)}" if where else ""
        params.append(top_k)

        query = f"""
            SELECT *, (embedding <=> %s::vector) AS distance
            FROM {self.table}
            {where_expr}
            ORDER BY distance ASC
            LIMIT %s;
        """
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        finally:
            conn.close()

        results: list[SearchResult] = []
        for rank, row in enumerate(rows):
            distance = float(row.pop("distance"))
            score = max(0.0, min(1.0, 1.0 - distance))
            results.append(
                SearchResult(record=self._row_to_record(row), score=score, rank=rank + 1)
            )
        return results

    def search_lexical(
        self,
        query: str,
        top_k: int = 50,
        record_types: list[str] | None = None,
        folder_prefix: str | None = None,
    ) -> list[SearchResult]:
        where = ["content_tsv @@ plainto_tsquery('english', %s)"]
        params: list[Any] = [query, query]
        if record_types:
            where.append("record_type = ANY(%s)")
            params.append(record_types)
        if folder_prefix:
            where.append("source_path LIKE %s")
            params.append(f"{_like_escape(folder_prefix)}%")
        params.append(top_k)

        sql = f"""
            SELECT *, ts_rank_cd(content_tsv, plainto_tsquery('english', %s)) AS lex_score
            FROM {self.table}
            WHERE {' AND '.join(where)}
            ORDER BY lex_score DESC
            LIMIT %s;
        """
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        finally:
            conn.close()

        raw = [float(row["lex_score"] or 0.0) for row in rows]
        max_score = max(raw) if raw else 1.0
        results: list[SearchResult] = []
        for rank, row in enumerate(rows):
            score = float(row.pop("lex_score") or 0.0)
            score = score / max_score if max_score > 0 else 0.0
            results.append(
                SearchResult(record=self._row_to_record(row), score=score, rank=rank + 1)
            )
        return results

    def get_by_file_hash(self, file_hash: str) -> list[VectorRecord]:
        return self._fetch_records("WHERE file_hash = %s", (file_hash,))

    def get_file_chunks(self, source_path: str) -> list[VectorRecord]:
        return self._fetch_records(
            "WHERE source_path = %s ORDER BY chunk_index", (source_path,)
        )

    def delete_by_path(self, source_path: str) -> int:
        return self._delete("WHERE source_path = %s", (source_path,))

    def delete_by_folder(self, folder_path: str) -> int:
        like = f"{_like_escape(folder_path)}%"
        return self._delete(
            "WHERE parent_directory LIKE %s OR source_path LIKE %s", (like, like)
        )

    def clear(self) -> int:
        """Drop and recreate the table (full reset, also fixes dimension changes)."""
        try:
            total = self.get_stats().get("total_records", 0)
        except Exception:  # noqa: BLE001
            total = 0
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {self.table};")
            conn.commit()
        finally:
            conn.close()
        self.initialize()
        return int(total)

    def get_stats(self) -> dict[str, int]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) AS n FROM {self.table};")
                total = int(cur.fetchone()["n"])
                cur.execute(
                    f"SELECT COUNT(DISTINCT source_path) AS n FROM {self.table};"
                )
                files = int(cur.fetchone()["n"])
        finally:
            conn.close()
        return {"total_records": total, "total_files": files}

    def indexed_file_hashes(self) -> set[str]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT DISTINCT file_hash FROM {self.table} "
                    "WHERE file_hash IS NOT NULL AND file_hash <> '';"
                )
                return {str(row["file_hash"]) for row in cur.fetchall()}
        finally:
            conn.close()

    def indexed_paths(self) -> set[str]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT DISTINCT source_path FROM {self.table} "
                    "WHERE source_path IS NOT NULL AND source_path <> '';"
                )
                return {str(row["source_path"]) for row in cur.fetchall()}
        finally:
            conn.close()

    def reindex(self) -> None:
        # pgvector updates HNSW incrementally on insert; nothing to rebuild.
        return None

    def _fetch_records(self, where_sql: str, params: tuple[Any, ...]) -> list[VectorRecord]:
        query = f"SELECT * FROM {self.table} {where_sql};"
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        finally:
            conn.close()
        return [self._row_to_record(dict(row)) for row in rows]

    def _delete(self, where_sql: str, params: tuple[Any, ...]) -> int:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self.table} {where_sql};", params)
                deleted = cur.rowcount
            conn.commit()
        finally:
            conn.close()
        return int(deleted)

    @staticmethod
    def _row_to_record(row: dict[str, Any]) -> VectorRecord:
        data = dict(row)
        data.pop("content_tsv", None)
        data.pop("distance", None)
        data.pop("lex_score", None)
        if data.get("id") is not None:
            data["id"] = str(data["id"])
        metadata = data.pop("metadata_json", None)
        if isinstance(metadata, str) and metadata:
            try:
                data["metadata"] = json.loads(metadata)
            except json.JSONDecodeError:
                data["metadata"] = {}
        return VectorRecord.from_dict(data)
