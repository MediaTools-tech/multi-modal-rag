from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from multimodal_rag.config import Settings
from multimodal_rag.core.models import SearchResult, VectorRecord
from multimodal_rag.database.base_interface import DocumentRepository
from multimodal_rag.search.lexical import bm25_scores, tokenize

logger = logging.getLogger(__name__)


def _sql_escape(value: str) -> str:
    return value.replace("'", "''")


class LanceDBProvider(DocumentRepository):
    """Embedded LanceDB store with FTS and an in-Python BM25 fallback."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db_path = Path(settings.LANCEDB_DIR)
        self.table_name = settings.DB_TABLE_NAME
        self.dim = settings.EMBEDDING_DIMENSION
        self._conn: Any = None
        self._table: Any = None
        self._write_lock = threading.Lock()
        self._dirty = False

    def _schema(self) -> Any:
        import pyarrow as pa

        return pa.schema(
            [
                pa.field("id", pa.string(), nullable=False),
                pa.field("vector", pa.list_(pa.float32(), self.dim), nullable=False),
                pa.field("content", pa.string()),
                pa.field("record_type", pa.string()),
                pa.field("source_path", pa.string()),
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
            ]
        )

    def initialize(self) -> None:
        try:
            import lancedb
        except ImportError as exc:
            raise RuntimeError(
                "LanceDB is not installed. Install it with `pip install lancedb pyarrow` "
                "or set SYSTEM_MODE=DOC_ONLY_RAG / ACTIVE_DB_ENGINE=POSTGRES."
            ) from exc
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._conn = lancedb.connect(str(self.db_path))
        if self.table_name in self._conn.table_names():
            self._table = self._conn.open_table(self.table_name)
            self._ensure_columns()
        else:
            self._table = self._conn.create_table(
                self.table_name, schema=self._schema(), mode="create"
            )
        self._ensure_fts_index()
        self._ensure_vector_index()
        logger.info(
            "LanceDBProvider initialized (dir=%s, table=%s)", self.db_path, self.table_name
        )

    def _ensure_columns(self) -> None:
        existing = {field.name for field in self._table.schema}
        defaults = {}
        if "record_type" not in existing:
            defaults["record_type"] = "chunk"
        if "summary" not in existing:
            defaults["summary"] = ""
        if defaults:
            try:
                self._table.add_columns(defaults)
            except Exception as exc:
                logger.warning("LanceDB column migration failed: %s", exc)

    def _create_vector_index(self, index_type: str, num_partitions: int) -> None:
        """Create the ANN index, tolerating old and new lancedb APIs.

        New unified API (recommended): column first + ``config`` object.
        Legacy API: metric first (``create_index(metric, vector_column_name, ...)``),
        so the old call ``create_index("vector", metric="cosine", ...)`` bound
        "vector" to ``metric`` and raised
        ``got multiple values for argument 'metric'`` on new versions.
        """
        try:
            from lancedb.index import IvfFlat, IvfPq
        except ImportError:
            # Old lancedb: metric and column must both be keywords.
            kwargs: dict = {
                "metric": "cosine",
                "vector_column_name": "vector",
                "num_partitions": num_partitions,
            }
            try:
                self._table.create_index(index_type=index_type, **kwargs)
            except TypeError:
                self._table.create_index(**kwargs)
            return
        if index_type == "IVF_PQ":
            config = IvfPq(distance_type="cosine", num_partitions=num_partitions)
        else:
            config = IvfFlat(distance_type="cosine", num_partitions=num_partitions)
        self._table.create_index("vector", config=config)

    def _ensure_vector_index(self) -> None:
        index_type = self.settings.LANCEDB_INDEX_TYPE.value
        if index_type == "FLAT":
            return  # brute-force scan by design, no index to build
        try:
            row_count = self._table.count_rows()
        except Exception:
            row_count = None
        num_partitions = self.settings.LANCEDB_NUM_PARTITIONS
        if row_count is not None and row_count < max(1, num_partitions):
            # IVF training needs at least one row per partition; tiny tables
            # (e.g. 36 rows / 256 partitions) cannot train and don't need ANN
            # anyway — brute-force scan is exact and faster. The index builds
            # on a later reindex once the table outgrows the partition count.
            logger.info(
                "Skipping LanceDB %s index: %s rows < %s partitions (brute-force scan)",
                index_type,
                row_count,
                num_partitions,
            )
            return
        try:
            self._create_vector_index(index_type, num_partitions)
        except Exception as exc:
            if "already exists" in str(exc).lower():
                return
            logger.warning(
                "LanceDB index %s failed (%s); retrying default index", index_type, exc
            )
            try:
                self._create_vector_index("IVF_FLAT", num_partitions)
            except Exception as exc2:
                if "already exists" not in str(exc2).lower():
                    logger.warning("LanceDB default vector index failed: %s", exc2)

    def _ensure_fts_index(self) -> None:
        try:
            self._table.create_fts_index("content")
        except Exception as exc:
            if "already exists" not in str(exc).lower():
                logger.warning(
                    "LanceDB FTS index creation failed (will use BM25 fallback): %s", exc
                )

    def reindex(self) -> None:
        # Skip rebuilding when nothing changed since the last rebuild. The dirty
        # check is inside the lock so two workers flushing at the idle boundary
        # cannot both rebuild the same pending batch.
        with self._write_lock:
            if not self._dirty:
                return
            self._reindex_locked()

    def _reindex_locked(self) -> None:
        try:
            self._table.drop_index()
        except Exception:
            pass
        self._ensure_vector_index()
        try:
            self._table.create_fts_index("content", replace=True)
        except Exception as exc:
            logger.warning("LanceDB FTS reindex failed: %s", exc)
        self._dirty = False

    def insert(self, records: list[VectorRecord]) -> int:
        if not records:
            return 0
        data = []
        for record in records:
            payload = record.to_dict()
            payload["metadata_json"] = json.dumps(payload.pop("metadata", {}))
            data.append(payload)
        with self._write_lock:
            self._table.add(data)
            # LanceDB does not refresh indexes on insert, but rebuilding the
            # whole vector + FTS index on every file is O(n^2) across a backfill.
            # Mark dirty and let reindex() run once the ingestion queue drains
            # (IngestionWorker flushes on idle/stop), so a 1,000-file import does
            # one rebuild instead of 1,000.
            self._dirty = True
        return len(data)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        record_types: list[str] | None = None,
        folder_prefix: str | None = None,
    ) -> list[SearchResult]:
        where = None
        if folder_prefix:
            where = f"source_path LIKE '{_sql_escape(folder_prefix)}%'"
        fetch_limit = top_k * 5 if record_types else top_k

        query = self._table.search(query_vector).metric("cosine")
        if where:
            query = query.where(where)
        frame = query.limit(fetch_limit).to_pandas()

        results: list[SearchResult] = []
        for _, row in frame.iterrows():
            data = row.to_dict()
            record = self._row_to_record(data)
            if record_types is not None and record.record_type not in record_types:
                continue
            distance = float(data.get("_distance", 1.0) or 0.0)
            score = max(0.0, min(1.0, 1.0 - distance))
            results.append(SearchResult(record=record, score=score, rank=len(results) + 1))
            if len(results) >= top_k:
                break
        return results

    def search_lexical(
        self,
        query: str,
        top_k: int = 50,
        record_types: list[str] | None = None,
        folder_prefix: str | None = None,
    ) -> list[SearchResult]:
        where = None
        if folder_prefix:
            where = f"source_path LIKE '{_sql_escape(folder_prefix)}%'"

        try:
            fts = self._table.search(query, query_type="fts")
            if where:
                fts = fts.where(where)
            frame = fts.limit(top_k).to_pandas()
            results: list[SearchResult] = []
            for _, row in frame.iterrows():
                data = row.to_dict()
                record = self._row_to_record(data)
                if record_types is not None and record.record_type not in record_types:
                    continue
                results.append(
                    SearchResult(record=record, score=abs(float(data.get("_score", 0.0) or 0.0)))
                )
            if results:
                max_score = max(r.score for r in results) or 1.0
                for result in results:
                    result.score = result.score / max_score
                results.sort(key=lambda r: r.score, reverse=True)
                for rank, result in enumerate(results, start=1):
                    result.rank = rank
                return results
        except Exception as exc:
            logger.warning("LanceDB FTS failed, falling back to BM25: %s", exc)

        frame = self._table.to_pandas()
        if frame.empty:
            return []
        if folder_prefix:
            frame = frame[
                frame["source_path"].fillna("").str.startswith(folder_prefix)
            ]
        docs = [
            (str(row["id"]), str(row.get("content") or "")) for _, row in frame.iterrows()
        ]
        scores = bm25_scores(tokenize(query), docs)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        max_score = ranked[0][1] if ranked and ranked[0][1] > 0 else 1.0
        by_id = {str(row["id"]): row for _, row in frame.iterrows()}
        results = []
        for doc_id, raw_score in ranked:
            if raw_score <= 0:
                continue
            record = self._row_to_record(by_id[doc_id].to_dict())
            if record_types is not None and record.record_type not in record_types:
                continue
            results.append(
                SearchResult(record=record, score=raw_score / max_score, rank=len(results) + 1)
            )
            if len(results) >= top_k:
                break
        return results

    def get_by_file_hash(self, file_hash: str) -> list[VectorRecord]:
        frame = self._table.to_pandas()
        if frame.empty:
            return []
        subset = frame[frame["file_hash"] == file_hash]
        return [self._row_to_record(row.to_dict()) for _, row in subset.iterrows()]

    def get_file_chunks(self, source_path: str) -> list[VectorRecord]:
        frame = self._table.to_pandas()
        if frame.empty:
            return []
        subset = frame[frame["source_path"] == source_path].sort_values("chunk_index")
        return [self._row_to_record(row.to_dict()) for _, row in subset.iterrows()]

    def _count(self, predicate) -> int:
        frame = self._table.to_pandas()
        if frame.empty:
            return 0
        return int(predicate(frame).sum())

    def delete_by_path(self, source_path: str) -> int:
        deleted = self._count(lambda f: f["source_path"] == source_path)
        with self._write_lock:
            self._table.delete(f"source_path = '{_sql_escape(source_path)}'")
        return deleted

    def delete_by_folder(self, folder_path: str) -> int:
        safe = _sql_escape(folder_path)
        deleted = self._count(
            lambda f: f["source_path"].fillna("").str.startswith(folder_path)
            | f["parent_directory"].fillna("").str.startswith(folder_path)
        )
        with self._write_lock:
            self._table.delete(
                f"parent_directory LIKE '{safe}%' OR source_path LIKE '{safe}%'"
            )
        return deleted

    def clear(self) -> int:
        """Drop and recreate the table (full reset)."""
        total = self.get_stats().get("total_records", 0)
        with self._write_lock:
            if self.table_name in self._conn.table_names():
                self._conn.drop_table(self.table_name)
        self.initialize()
        return int(total)

    def get_stats(self) -> dict[str, int]:
        frame = self._table.to_pandas()
        if frame.empty:
            return {"total_records": 0, "total_files": 0}
        return {
            "total_records": int(len(frame)),
            "total_files": int(frame["source_path"].nunique()),
        }

    def indexed_file_hashes(self) -> set[str]:
        frame = self._table.to_pandas()
        if frame.empty or "file_hash" not in frame:
            return set()
        return {str(h) for h in frame["file_hash"].dropna().unique() if str(h)}

    def indexed_paths(self) -> set[str]:
        frame = self._table.to_pandas()
        if frame.empty or "source_path" not in frame:
            return set()
        return {str(p) for p in frame["source_path"].dropna().unique() if str(p)}

    def close(self) -> None:
        self._table = None
        self._conn = None

    @staticmethod
    def _row_to_record(data: dict[str, Any]) -> VectorRecord:
        record = dict(data)
        record.pop("_distance", None)
        record.pop("_score", None)
        metadata = record.pop("metadata_json", None)
        if isinstance(metadata, str) and metadata:
            try:
                record["metadata"] = json.loads(metadata)
            except json.JSONDecodeError:
                record["metadata"] = {}
        for field_name in ("timestamp_start", "timestamp_end"):
            value = record.get(field_name)
            if value is not None and isinstance(value, float) and value != value:
                record[field_name] = None
        return VectorRecord.from_dict(record)
