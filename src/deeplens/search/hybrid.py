"""Summary-aware hybrid retrieval.

Combines three complementary signals so a query like *"story of a man who owns
a big mansion"* can both **locate the file** (via per-document summary records)
and **ground the answer** (via passage chunks), while a **corpus-wide** lexical
signal boosts precision on distinctive keywords / proper nouns.

The three ranked lists are fused with Reciprocal Rank Fusion (RRF):

    1. Vector ranking  — semantic ANN over all records (chunks + summaries).
    2. Lexical ranking — full-text / BM25 search across the *entire* corpus
       (via the repository's ``search_lexical``), not just the vector top-k.
    3. Summary ranking — vector ranking restricted to file-level summary records,
       giving documents a second chance to surface for description-style queries.

An optional cross-encoder re-ranker then re-scores only the top fused
candidates (see ``search/reranker.py``) for maximum quality.
"""

from __future__ import annotations

from collections import defaultdict

import structlog

from deeplens.config import Settings
from deeplens.core.models import FileSearchGroup, RecordType, SearchResult
from deeplens.core.repository import DocumentRepository
from deeplens.search.reranker import CrossEncoderReranker

logger = structlog.get_logger(__name__)


def _rrf_ranked(lists: list[list[SearchResult]], k: int) -> list[SearchResult]:
    """Fuse multiple ranked lists via Reciprocal Rank Fusion.

    Returns a single ranked list keyed by record id (each id appears once,
    keeping its highest-relevance occurrence).
    """
    fused: dict[str, float] = defaultdict(float)
    order: dict[str, SearchResult] = {}
    for ranked in lists:
        for rank, res in enumerate(ranked):
            fused[res.record.id] += 1.0 / (k + rank + 1)
            order.setdefault(res.record.id, res)
    return sorted(order.values(), key=lambda r: fused[r.record.id], reverse=True)


def _build_file_groups(
    summary_results: list[SearchResult],
    chunk_results: list[SearchResult],
    max_groups: int = 10,
) -> list[FileSearchGroup]:
    """Group summary matches, attaching their top supporting chunks."""
    chunks_by_path: dict[str, list[SearchResult]] = defaultdict(list)
    for cr in chunk_results:
        chunks_by_path[cr.record.absolute_path].append(cr)

    groups: list[FileSearchGroup] = []
    seen: set[str] = set()
    for sr in summary_results:
        path = sr.record.absolute_path
        if path in seen:
            continue
        seen.add(path)
        rec = sr.record
        groups.append(
            FileSearchGroup(
                absolute_path=rec.absolute_path,
                filename=rec.filename,
                file_type=rec.file_type,
                summary=rec.summary or rec.content,
                best_score=sr.score,
                chunk_results=chunks_by_path.get(path, [])[:3],
            )
        )
        if len(groups) >= max_groups:
            break
    return groups


async def hybrid_search(
    repo: DocumentRepository,
    query: str,
    query_vector: list[float],
    settings: Settings,
    top_k: int = 10,
    folder_filter: str | None = None,
    file_type_filter: str | None = None,
    reranker: CrossEncoderReranker | None = None,
) -> tuple[list[SearchResult], list[FileSearchGroup]]:
    """Run summary-aware hybrid retrieval with corpus-wide lexical fusion.

    Returns the fused ranked ``SearchResult`` list and a file-level grouping
    (``FileSearchGroup``) that points the user at whole documents.
    """
    rrf_k = settings.hybrid_rrf_k

    # 1. Vector candidates (chunks + summaries).
    chunk_results = await repo.search(
        query_vector,
        top_k=max(top_k * 5, 25),
        folder_filter=folder_filter,
        file_type_filter=file_type_filter,
        record_types=[RecordType.CHUNK.value],
    )
    summary_results = await repo.search(
        query_vector,
        top_k=top_k,
        folder_filter=folder_filter,
        file_type_filter=file_type_filter,
        record_types=[RecordType.SUMMARY.value],
    )

    # 2. Corpus-wide lexical ranking (covers ALL documents, not just vector top-k).
    lexical_results = await repo.search_lexical(
        query,
        top_k=max(top_k * 10, 100),
        folder_filter=folder_filter,
        file_type_filter=file_type_filter,
    )

    # 3. Build the three ranked lists and fuse with RRF.
    vector_ranked = sorted(
        chunk_results + summary_results, key=lambda r: r.score, reverse=True
    )
    summary_ranked = sorted(summary_results, key=lambda r: r.score, reverse=True)
    lexical_ranked = lexical_results

    fused = _rrf_ranked([vector_ranked, lexical_ranked, summary_ranked], k=rrf_k)

    # 4. Optional cross-encoder re-rank over a small fused candidate pool.
    if reranker is None and settings.enable_reranker:
        reranker = CrossEncoderReranker(settings)

    pool_size = max(top_k, settings.rerank_top_n)
    pool = fused[:pool_size]
    if reranker is not None:
        pool = await reranker.rerank(query, pool, top_k)
    else:
        pool = pool[:top_k]

    for i, r in enumerate(pool):
        r.rank = i + 1

    file_groups = _build_file_groups(summary_results, chunk_results)
    return pool, file_groups


async def summary_search(
    repo: DocumentRepository,
    query_vector: list[float],
    settings: Settings,
    top_k: int = 10,
    folder_filter: str | None = None,
    file_type_filter: str | None = None,
    query: str | None = None,
    reranker: CrossEncoderReranker | None = None,
) -> tuple[list[SearchResult], list[FileSearchGroup]]:
    """Pure document-level retrieval: match only per-file summary records.

    Best for "find the file that matches this description" queries. Returns the
    summary matches plus a file-level grouping with supporting chunk grounding.
    """
    summary_results = await repo.search(
        query_vector,
        top_k=top_k,
        folder_filter=folder_filter,
        file_type_filter=file_type_filter,
        record_types=[RecordType.SUMMARY.value],
    )
    if not summary_results:
        return [], []

    chunk_results = await repo.search(
        query_vector,
        top_k=max(top_k * 5, 25),
        folder_filter=folder_filter,
        file_type_filter=file_type_filter,
        record_types=[RecordType.CHUNK.value],
    )

    if reranker is None and settings.enable_reranker:
        reranker = CrossEncoderReranker(settings)
    if reranker is not None:
        # Use the original text query when available; otherwise fall back to a
        # lexical proxy derived from the top chunk.
        rerank_query = query or _summary_query_hint(chunk_results, summary_results)
        summary_results = await reranker.rerank(rerank_query, summary_results, top_k)
    else:
        for i, r in enumerate(summary_results[:top_k]):
            r.rank = i + 1

    file_groups = _build_file_groups(summary_results, chunk_results)
    return summary_results[:top_k], file_groups


def _summary_query_hint(
    chunk_results: list[SearchResult], summary_results: list[SearchResult]
) -> str:
    """Build a query proxy for re-ranking summaries when only a vector is known.

    Callers that have the original text query should pass it directly instead.
    """
    # Prefer the highest-scoring chunk text as a lexical proxy for the query.
    if chunk_results:
        return max(chunk_results, key=lambda r: r.score).record.content
    return summary_results[0].record.content if summary_results else ""
