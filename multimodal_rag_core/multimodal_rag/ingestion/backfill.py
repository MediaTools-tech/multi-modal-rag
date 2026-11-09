from __future__ import annotations

import logging
from collections.abc import Callable

from multimodal_rag.core.models import VectorRecord
from multimodal_rag.pipeline.embedding import EmbeddingEngine

logger = logging.getLogger(__name__)

ProgressFn = Callable[[int, int, int], None]


def _embed_records(
    embedder: EmbeddingEngine,
    records: list[VectorRecord],
    batch_size: int,
) -> int:
    embedded = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        vectors = embedder.embed_texts([record.content for record in batch])
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"Embedding count mismatch during backfill: {len(batch)} record(s) "
                f"but {len(vectors)} vector(s) returned."
            )
        for record, vector in zip(batch, vectors, strict=False):
            record.vector = vector
        embedded += len(batch)
    return embedded


def backfill_embeddings(
    repository,
    embedder: EmbeddingEngine,
    *,
    batch_size: int = 64,
    progress: ProgressFn | None = None,
) -> dict[str, int]:
    """Embed records that were indexed while embeddings were disabled.

    Files are replaced as a unit (delete_by_path + insert) because LanceDB has
    no portable in-place vector update. Existing vectors in the same file are
    kept as-is; only records with an empty vector are embedded.
    """
    missing = [record for batch in repository.iter_missing_vectors() for record in batch]
    paths = sorted({record.source_path for record in missing if record.source_path})
    total = len(paths)
    files = 0
    embedded = 0
    skipped = 0
    unassigned = sum(1 for record in missing if not record.source_path)

    for index, source_path in enumerate(paths, start=1):
        records = repository.get_file_chunks(source_path)
        pending = [
            record
            for record in records
            if (record.vector is None or len(record.vector) == 0)
            and (record.content or "").strip()
        ]
        skipped += sum(
            1
            for record in records
            if (record.vector is None or len(record.vector) == 0)
            and not (record.content or "").strip()
        )
        if pending:
            embedded += _embed_records(embedder, pending, max(1, batch_size))
            repository.delete_by_path(source_path)
            repository.insert(records)
            files += 1
        if progress is not None:
            progress(index, total, embedded)

    if files:
        repository.reindex()
    result = {
        "files": files,
        "records": embedded,
        "skipped": skipped,
        "unassigned": unassigned,
        "missing_files": total,
    }
    logger.info(
        "embedding backfill: %d file(s), %d record(s) embedded, %d skipped",
        files,
        embedded,
        skipped,
    )
    return result
