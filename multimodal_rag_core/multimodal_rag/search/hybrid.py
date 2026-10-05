from __future__ import annotations

import logging
from collections import defaultdict

from multimodal_rag.core.models import FileSearchGroup, RecordType, SearchResult
from multimodal_rag.search.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


def rrf_fuse(ranked_lists: list[list[SearchResult]], k: int = 60) -> list[SearchResult]:
    """Reciprocal Rank Fusion of several ranked result lists.

    Returns new ``SearchResult`` objects whose ``score`` is the fused RRF value.
    Previously the fused score was computed only for ordering and discarded, so
    callers (and the logs below) saw a stale pre-fusion score from whichever
    list ranked the record first. The source results are left untouched.
    """
    fused: dict[str, float] = defaultdict(float)
    order: dict[str, SearchResult] = {}
    for ranked in ranked_lists:
        for rank, result in enumerate(ranked):
            record_id = result.record.id
            fused[record_id] += 1.0 / (k + rank + 1)
            order.setdefault(record_id, result)
    fused_results = [
        SearchResult(record=result.record, score=fused[record_id], rank=result.rank)
        for record_id, result in order.items()
    ]
    fused_results.sort(key=lambda r: r.score, reverse=True)
    return fused_results


def _build_file_groups(
    summary_results: list[SearchResult],
    chunk_results: list[SearchResult],
    max_groups: int = 10,
) -> list[FileSearchGroup]:
    chunks_by_path: dict[str, list[SearchResult]] = defaultdict(list)
    for chunk in chunk_results:
        chunks_by_path[chunk.record.source_path].append(chunk)

    groups: list[FileSearchGroup] = []
    seen: set[str] = set()
    for summary in summary_results:
        path = summary.record.source_path
        if path in seen:
            continue
        seen.add(path)
        record = summary.record
        groups.append(
            FileSearchGroup(
                source_path=record.source_path,
                filename=record.filename,
                file_type=record.file_type,
                summary=record.summary or record.content,
                best_score=summary.score,
                chunk_results=chunks_by_path.get(path, [])[:3],
            )
        )
        if len(groups) >= max_groups:
            break
    return groups


def hybrid_search(
    repo,
    query_text: str,
    query_vector: list[float],
    top_k: int = 10,
    rrf_k: int = 60,
    reranker: CrossEncoderReranker | None = None,
    rerank: bool = False,
    rerank_top_n: int = 15,
    include_summaries: bool = True,
    folder_prefix: str | None = None,
) -> tuple[list[SearchResult], list[FileSearchGroup]]:
    """Fuse vector, corpus-wide lexical and summary rankings with RRF."""
    chunk_results = repo.search(
        query_vector,
        top_k=max(top_k * 5, 25),
        record_types=[RecordType.CHUNK.value],
        folder_prefix=folder_prefix,
    )

    summary_results: list[SearchResult] = []
    if include_summaries:
        summary_results = repo.search(
            query_vector,
            top_k=top_k,
            record_types=[RecordType.SUMMARY.value],
            folder_prefix=folder_prefix,
        )

    lexical_results = repo.search_lexical(
        query_text,
        top_k=max(top_k * 10, 100),
        folder_prefix=folder_prefix,
    )

    vector_ranked = sorted(
        chunk_results + summary_results, key=lambda r: r.score, reverse=True
    )
    summary_ranked = sorted(summary_results, key=lambda r: r.score, reverse=True)

    fused = rrf_fuse([vector_ranked, lexical_results, summary_ranked], k=rrf_k)
    logger.info(
        "hybrid fused top scores: %s",
        [f"{r.score:.4f}" for r in fused[:5]],
    )

    pool_size = max(top_k, rerank_top_n) if rerank else top_k
    pool = fused[:pool_size]

    if rerank:
        if reranker is None:
            reranker = CrossEncoderReranker(top_n=rerank_top_n)
        pool = reranker.rerank(query_text, pool, top_k)
        logger.info(
            "hybrid post-rerank top scores: %s",
            [f"{r.score:.4f}" for r in pool[:5]],
        )
    else:
        logger.info("hybrid rerank skipped (rerank=False)")
        pool = pool[:top_k]

    for rank, result in enumerate(pool):
        result.rank = rank + 1

    file_groups = _build_file_groups(summary_results, chunk_results)
    return pool, file_groups
