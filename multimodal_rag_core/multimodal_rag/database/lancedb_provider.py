from __future__ import annotations

import json
import logging
import re
import threading
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from multimodal_rag.config import Settings
from multimodal_rag.core.models import SearchResult, VectorRecord
from multimodal_rag.database.base_interface import DocumentRepository
from multimodal_rag.search.lexical import (
    STOPWORDS,
    bm25_scores,
    normalize_text,
    numeric_query_variants,
    tokenize,
)

logger = logging.getLogger(__name__)


def _sql_escape(value: str) -> str:
    return value.replace("'", "''")


def _path_tokens(value: Any) -> set[str]:
    """Split a path/filename on separators, dots, underscores, and hyphens."""
    if value is None or (isinstance(value, float) and value != value):
        text = ""
    else:
        text = str(value)
    return {
        token
        for token in re.split(r"[\W_]+", normalize_text(text).casefold(), flags=re.UNICODE)
        if token
    }


def _path_query_terms(query: str, limit: int = 16) -> list[str]:
    terms = [term for term in _path_tokens(query) if len(term) >= 2 and term not in STOPWORDS]
    # Keep the pushed-down SQL predicate bounded even for very long queries.
    return sorted(terms)[:limit]


def _path_scope_filter(
    record_types: list[str] | None,
    folder_prefix: str | None,
) -> str | None:
    """Build the record-type/folder predicate shared by path lookups."""
    if record_types == []:
        return None
    clauses = []
    if folder_prefix:
        clauses.append(f"source_path LIKE '{_sql_escape(folder_prefix)}%'")
    if record_types is not None:
        values = ", ".join(f"'{_sql_escape(str(value))}'" for value in record_types)
        clauses.append(f"record_type IN ({values})")
    return " AND ".join(clauses) or None


def _path_filter(
    terms: list[str],
    record_types: list[str] | None,
    folder_prefix: str | None,
) -> str | None:
    """Build a LanceDB SQL predicate for path/filename token candidates."""
    if not terms or record_types == []:
        return None
    clauses = [
        f"(lower(source_path) LIKE '%{_sql_escape(term)}%' "
        f"OR lower(filename) LIKE '%{_sql_escape(term)}%')"
        for term in terms
    ]
    where = f"({' OR '.join(clauses)})"
    scope = _path_scope_filter(record_types, folder_prefix)
    if scope:
        where += f" AND {scope}"
    return where


#: Keep path results as complete records; the row filter already avoids a full scan.
_PATH_COLUMNS = [
    "id",
    "vector",
    "content",
    "record_type",
    "source_path",
    "filename",
    "parent_directory",
    "file_type",
    "mime_type",
    "chunk_index",
    "total_chunks",
    "timestamp_start",
    "timestamp_end",
    "created_at",
    "file_modified_at",
    "file_hash",
    "metadata_json",
    "summary",
    "has_vector",
]

#: FTS auto-projects `_score`; selecting it explicitly avoids LanceDB's
#: deprecation warning while keeping path-column fetches narrow.
_PATH_FTS_COLUMNS = [*_PATH_COLUMNS, "_score"]

#: Match the Python path scorer: split punctuation, keep case/stopword handling
#: in Python, and do not stem (the exact token overlap is what gets scored).
_PATH_FTS_OPTIONS = {
    "base_tokenizer": "simple",
    "lower_case": True,
    "ascii_folding": True,
    "stem": False,
    "remove_stop_words": False,
    "max_token_length": 128,
}


class LanceDBProvider(DocumentRepository):
    """Embedded LanceDB store with FTS and an in-Python BM25 fallback."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db_path = Path(settings.LANCEDB_DIR)
        self.table_name = settings.DB_TABLE_NAME
        self.dim = settings.EMBEDDING_DIMENSION
        # The table's actual vector width can differ from the current setting
        # (e.g. keyword-only mode after changing embedding models). Placeholder
        # vectors must match the stored column, not the configured dimension.
        self._storage_dim = self.dim
        self._path_pushdown_supported = True
        self._path_fts_supported = False
        self._conn: Any = None
        self._table: Any = None
        self._write_lock = threading.Lock()
        self._dirty = False
        self._vocab_cache: dict[str, int] | None = None

    def _schema(self) -> Any:
        import pyarrow as pa

        return pa.schema(
            [
                pa.field("id", pa.string(), nullable=False),
                pa.field("vector", pa.list_(pa.float32(), self.dim), nullable=False),
                pa.field("has_vector", pa.bool_(), nullable=False),
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
        self._storage_dim = self._read_vector_dim()
        if self._storage_dim != self.dim:
            if self.settings.EMBEDDING_PROVIDER.value == "NONE":
                logger.warning(
                    "LanceDB vector column is %d-dimensional but "
                    "EMBEDDING_DIMENSION=%d; keyword-only placeholders will use "
                    "the table dimension",
                    self._storage_dim,
                    self.dim,
                )
            else:
                raise RuntimeError(
                    f"LanceDB table {self.table_name!r} stores "
                    f"{self._storage_dim}-dimensional vectors but "
                    f"EMBEDDING_DIMENSION={self.dim}. Reset the index "
                    "(GUI: File -> Reset index... / CLI: mrag-ingest reset-index) "
                    "or use a new LANCEDB_DIR."
                )
        self._ensure_fts_index()
        self._ensure_vector_index()
        self._vocab_cache = None
        logger.info(
            "LanceDBProvider initialized (dir=%s, table=%s)", self.db_path, self.table_name
        )

    def _read_vector_dim(self) -> int:
        field = self._table.schema.field("vector")
        size = getattr(field.type, "list_size", None)
        if size is None:
            raise RuntimeError(
                "LanceDB vector column is not a fixed-size list; reset the index."
            )
        return int(size)

    def _ensure_columns(self) -> None:
        existing = {field.name for field in self._table.schema}
        defaults = {}
        if "record_type" not in existing:
            defaults["record_type"] = "cast('chunk' as string)"
        if "summary" not in existing:
            defaults["summary"] = "cast('' as string)"
        if "has_vector" not in existing:
            # Existing rows were created before embedding-free indexing existed
            # and therefore have real vectors.
            defaults["has_vector"] = "cast(true as boolean)"
        if defaults:
            # LanceDB evaluates these as SQL expressions per row, not as Python
            # values. Passing a bool raises "not an instance of 'str'".
            try:
                self._table.add_columns(defaults)
            except Exception as exc:
                logger.warning("LanceDB column migration failed: %s", exc)
            missing = set(defaults) - {field.name for field in self._table.schema}
            if missing:
                raise RuntimeError(
                    "LanceDB could not migrate the table to the current schema "
                    f"(missing: {', '.join(sorted(missing))}). Reset the index "
                    "(GUI: File -> Reset index... / CLI: mrag-ingest reset-index) "
                    "or use a new LANCEDB_DIR."
                ) from None

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

    def _create_fts_index(self, field: str, *, replace: bool = False, **options) -> bool:
        try:
            self._table.create_fts_index(field, replace=replace, **options)
            return True
        except Exception as exc:
            if "already exists" in str(exc).lower():
                return True
            logger.warning("LanceDB FTS index on %s failed: %s", field, exc)
            return False

    def _ensure_fts_index(self) -> None:
        self._create_fts_index("content")
        self._path_fts_supported = self._create_fts_index(
            "source_path", **_PATH_FTS_OPTIONS
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
        self._create_fts_index("content", replace=True)
        self._path_fts_supported = self._create_fts_index(
            "source_path", replace=True, **_PATH_FTS_OPTIONS
        )
        self._dirty = False

    def insert(self, records: list[VectorRecord]) -> int:
        if not records:
            return 0
        data = []
        for record in records:
            payload = record.to_dict()
            has_vector = record.vector is not None and len(record.vector) > 0
            payload["has_vector"] = has_vector
            if not has_vector:
                # LanceDB's fixed-size vector column is non-null; store a
                # placeholder and exclude it from vector search via has_vector.
                payload["vector"] = [0.0] * self._storage_dim
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
            self._vocab_cache = None
        return len(data)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        record_types: list[str] | None = None,
        folder_prefix: str | None = None,
    ) -> list[SearchResult]:
        conditions = ["has_vector = true"]
        if folder_prefix:
            conditions.append(f"source_path LIKE '{_sql_escape(folder_prefix)}%'")
        where = " AND ".join(conditions)
        fetch_limit = top_k * 5 if record_types else top_k

        query = self._table.search(query_vector).metric("cosine")
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

    def _path_fts_frame(
        self,
        terms: list[str],
        limit: int,
        record_types: list[str] | None,
        folder_prefix: str | None,
    ) -> Any:
        """Fetch path candidates from the ``source_path`` FTS index."""
        from lancedb.query import FullTextOperator, MatchQuery

        match = MatchQuery(
            " ".join(terms),
            "source_path",
            operator=FullTextOperator.OR,
            fuzziness=0,
        )
        query = self._table.search(match, query_type="fts")
        scope = _path_scope_filter(record_types, folder_prefix)
        if scope:
            query = query.where(scope)
        # Overfetch because the FTS score is only a candidate filter; exact
        # term-count scoring happens below.
        return query.limit(limit).select(_PATH_FTS_COLUMNS).to_pandas()

    def _path_results(
        self,
        query: str,
        top_k: int,
        record_types: list[str] | None,
        folder_prefix: str | None,
        frame=None,
    ) -> list[SearchResult]:
        """Rank chunks by filename/source-path token overlap."""
        terms = _path_query_terms(query)
        if not terms or record_types == []:
            return []
        if frame is None:
            # Inserted rows are not in the FTS index until reindex() runs, so
            # use the live filtered scan while the table is dirty.
            if getattr(self, "_path_fts_supported", False) and not getattr(
                self, "_dirty", False
            ):
                try:
                    frame = self._path_fts_frame(
                        terms,
                        max(top_k * 100, 1000),
                        record_types,
                        folder_prefix,
                    )
                except Exception as exc:
                    self._path_fts_supported = False
                    logger.warning(
                        "LanceDB path FTS failed, falling back to filtered scan: %s",
                        exc,
                    )
            if frame is None:
                where = _path_filter(terms, record_types, folder_prefix)
                if where is None:
                    return []
                if getattr(self, "_path_pushdown_supported", True):
                    try:
                        # Plain SQL scan (no vector/FTS) with the substring predicate
                        # pushed down: only candidate rows cross into pandas.
                        frame = (
                            self._table.search(None)
                            .where(where)
                            .select(_PATH_COLUMNS)
                            .to_pandas()
                        )
                    except Exception as exc:
                        self._path_pushdown_supported = False
                        logger.warning(
                            "LanceDB path filter failed, falling back to table scans: %s",
                            exc,
                        )
                        frame = self._table.to_pandas()
                else:
                    frame = self._table.to_pandas()
        if frame.empty:
            return []
        if folder_prefix:
            frame = frame[
                frame["source_path"].fillna("").str.startswith(folder_prefix)
            ]
        if record_types:
            frame = frame[frame["record_type"].isin(record_types)]
        if frame.empty:
            return []

        scored: list[tuple[float, Any]] = []
        for _, row in frame.iterrows():
            path_terms = _path_tokens(row.get("source_path"))
            name_terms = _path_tokens(row.get("filename"))
            matched = [
                term for term in terms if term in path_terms or term in name_terms
            ]
            if not matched:
                continue
            # A filename match is stronger than an incidental parent-directory
            # match, and matching more query terms is better than matching one.
            score = len(matched) + 0.5 * sum(
                1 for term in terms if term in name_terms
            )
            scored.append((score, row))
        if not scored:
            return []

        scored.sort(key=lambda item: (-item[0], str(item[1].get("source_path") or "")))
        max_score = scored[0][0] or 1.0
        results: list[SearchResult] = []
        for rank, (score, row) in enumerate(scored[:top_k], start=1):
            record = self._row_to_record(row.to_dict())
            results.append(
                SearchResult(record=record, score=score / max_score, rank=rank)
            )
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

        content_results: list[SearchResult] = []
        full_frame = None
        variants = numeric_query_variants(query)
        for variant in variants:
            variant_results: list[SearchResult] = []
            try:
                fts = self._table.search(variant, query_type="fts")
                if where:
                    fts = fts.where(where)
                # record_type is filtered below, so overfetch when a caller asks
                # for chunks only; otherwise summary rows can crowd chunks out
                # of the pre-filter limit.
                fetch_limit = top_k * 5 if record_types else top_k
                frame = fts.limit(fetch_limit).to_pandas()
                for _, row in frame.iterrows():
                    data = row.to_dict()
                    record = self._row_to_record(data)
                    if record_types is not None and record.record_type not in record_types:
                        continue
                    variant_results.append(
                        SearchResult(
                            record=record,
                            score=abs(float(data.get("_score", 0.0) or 0.0)),
                        )
                    )
            except Exception as exc:
                logger.warning(
                    "LanceDB FTS failed for %r, falling back to BM25: %s", variant, exc
                )
                continue
            if not variant_results:
                continue
            variant_max = max(result.score for result in variant_results) or 1.0
            for result in variant_results:
                result.score = result.score / variant_max
            content_results.extend(variant_results)

        if content_results:
            by_id: dict[str, SearchResult] = {}
            for result in content_results:
                existing = by_id.get(result.record.id)
                if existing is None or result.score >= existing.score:
                    by_id[result.record.id] = result
            content_results = list(by_id.values())
            content_results.sort(key=lambda r: r.score, reverse=True)
            for rank, result in enumerate(content_results, start=1):
                result.rank = rank

        if not content_results:
            full_frame = self._table.to_pandas()
            if full_frame.empty:
                return []
            if folder_prefix:
                full_frame = full_frame[
                    full_frame["source_path"].fillna("").str.startswith(folder_prefix)
                ]
            docs = [
                (str(row["id"]), str(row.get("content") or ""))
                for _, row in full_frame.iterrows()
            ]
            scores = bm25_scores(tokenize(query), docs)
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            max_score = ranked[0][1] if ranked and ranked[0][1] > 0 else 1.0
            by_id = {str(row["id"]): row for _, row in full_frame.iterrows()}
            for doc_id, raw_score in ranked:
                if raw_score <= 0:
                    continue
                record = self._row_to_record(by_id[doc_id].to_dict())
                if record_types is not None and record.record_type not in record_types:
                    continue
                content_results.append(
                    SearchResult(
                        record=record,
                        score=raw_score / max_score,
                        rank=len(content_results) + 1,
                    )
                )
                if len(content_results) >= top_k:
                    break

        path_results = self._path_results(
            query, top_k, record_types, folder_prefix, full_frame
        )
        if not content_results:
            return path_results[:top_k]
        if not path_results:
            return content_results[:top_k]
        from multimodal_rag.search.hybrid import rrf_fuse

        fused = rrf_fuse(
            [content_results, path_results], k=60, weights=[3.0, 1.0]
        )[:top_k]
        for rank, result in enumerate(fused, start=1):
            result.rank = rank
        return fused

    def lexical_vocabulary(self, max_terms: int = 50000) -> dict[str, int]:
        if self._vocab_cache is not None:
            return self._vocab_cache
        frame = self._table.to_pandas()
        if frame.empty or "content" not in frame:
            self._vocab_cache = {}
            return self._vocab_cache
        counts: Counter[str] = Counter()
        for text in frame["content"].dropna():
            counts.update(tokenize(str(text)))
        self._vocab_cache = dict(counts.most_common(max_terms))
        return self._vocab_cache

    def iter_records(self, batch_size: int = 500) -> Iterator[list[VectorRecord]]:
        frame = self._table.to_pandas()
        batch: list[VectorRecord] = []
        for _, row in frame.iterrows():
            batch.append(self._row_to_record(row.to_dict()))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

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
            self._vocab_cache = None
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
            self._vocab_cache = None
        return deleted

    def clear(self) -> int:
        """Drop and recreate the table (full reset)."""
        total = self.get_stats().get("total_records", 0)
        with self._write_lock:
            if self.table_name in self._conn.table_names():
                self._conn.drop_table(self.table_name)
        self.initialize()
        self._vocab_cache = None
        return int(total)

    def get_stats(self) -> dict[str, int]:
        frame = self._table.to_pandas()
        if frame.empty:
            return {"total_records": 0, "total_files": 0, "missing_vectors": 0}
        if "has_vector" in frame:
            missing = int((~frame["has_vector"].fillna(True).astype(bool)).sum())
        else:
            missing = 0
        return {
            "total_records": int(len(frame)),
            "total_files": int(frame["source_path"].nunique()),
            "missing_vectors": missing,
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
        has_vector = record.pop("has_vector", True)
        if has_vector is False or str(has_vector).lower() in ("false", "0"):
            record["vector"] = []
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
