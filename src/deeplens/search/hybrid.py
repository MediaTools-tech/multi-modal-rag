"""Summary-aware hybrid retrieval.

Combines three complementary signals so a query like *"story of a man who owns
a big mansion"* can both **locate the file** (via per-document summary records)
and **ground the answer** (via passage chunks), while a lexical signal boosts
precision on distinctive keywords / proper nouns.

The three ranked lists are fused with Reciprocal Rank Fusion (RRF):

    1. Vector ranking  — semantic ANN over all records (chunks + summaries).
    2. Lexical ranking — BM25-style keyword overlap (handles "mansion", names).
    3. Summary ranking — vector ranking restricted to file-level summary records,
       giving documents a second chance to surface for description-style queries.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

import structlog

from deeplens.core.models import (
    FileSearchGroup,
    RecordType,
    SearchResult,
)
from deeplens.core.repository import DocumentRepository

logger = structlog.get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9'+-]*")


def tokenize(text: str) -> list[str]:
    """Lower-cased tokenization for lexical scoring."""
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


@dataclass
class _Candidate:
    """Internal fusion candidate carrying its multiple relevance signals."""

    result: SearchResult
    vector_score: float = 0.0
    lexical_score: float = 0.0


def _lexical_scores(query_tokens: list[str], candidates: list[_Candidate]) -> None:
    """Compute a relative BM25-style lexical score for each candidate in place."""
    if not query_tokens:
        return

    # Document frequency across the candidate set (cheap, query-scoped IDF).
    df: dict[str, int] = defaultdict(int)
    for c in candidates:
        doc_tokens = set(tokenize(c.result.record.content))
        for qt in query_tokens:
            if qt in doc_tokens:
                df[qt] += 1

    n = max(1, len(candidates))
    idf = {qt: math.log((n - df[qt] + 0.5) / (df[qt] + 0.5) + 1.0) for qt in query_tokens}

    raw: list[float] = []
    for c in candidates:
        doc_tokens = tokenize(c.result.record.content)
        tf = defaultdict(int)
        for t in doc_tokens:
            tf[t] += 1
        doc_len = max(1, len(doc_tokens))
        score = 0.0
        for qt in query_tokens:
            if tf[qt] > 0:
                # BM25 saturation term, length-normalized.
                score += idf[qt] * (tf[qt] * (1.2 + 1)) / (tf[qt] + 1.2 * doc_len)
        c.lexical_score = score
        raw.append(score)

    max_raw = max(raw) if raw else 0.0
    if max_raw > 0:
        for c in candidates:
            c.lexical_score = c.lexical_score / max_raw  # normalize to [0, 1]


def _reciprocal_rank_fusion(
    ranked_lists: list[list[_Candidate]], k: int
) -> dict[str, float]:
    """Fuse multiple ranked lists via RRF; returns candidate-id -> fused score."""
    fused: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, cand in enumerate(ranked):
            fused[cand.result.record.id] += 1.0 / (k + rank + 1)
    return fused


def _by_vector(cands: list[_Candidate]) -> list[_Candidate]:
    return sorted(cands, key=lambda c: c.vector_score, reverse=True)


def _by_lexical(cands: list[_Candidate]) -> list[_Candidate]:
    return sorted(cands, key=lambda c: c.lexical_score, reverse=True)


async def hybrid_search(
    repo: DocumentRepository,
    query: str,
    query_vector: list[float],
    top_k: int = 10,
    folder_filter: str | None = None,
    file_type_filter: str | None = None,
    lexical_weight: float = 0.3,
    rrf_k: int = 60,
) -> tuple[list[SearchResult], list[FileSearchGroup]]:
    """Run summary-aware hybrid retrieval.

    Returns the fused ranked ``SearchResult`` list and a file-level grouping
    (``FileSearchGroup``) that points the user at whole documents.
    """
    # 1. Candidate retrieval. Over-fetch chunks so lexical re-ranking has signal.
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

    candidates: list[_Candidate] = []
    for r in chunk_results:
        candidates.append(_Candidate(result=r, vector_score=r.score))
    for r in summary_results:
        candidates.append(_Candidate(result=r, vector_score=r.score))

    if not candidates:
        return [], []

    # 2. Lexical signal.
    query_tokens = tokenize(query)
    _lexical_scores(query_tokens, candidates)

    # Blend lexical weight into the vector ranking so a single fused list can be
    # produced: effective = (1 - w) * vector + w * lexical.
    for c in candidates:
        c.vector_score = (1.0 - lexical_weight) * c.vector_score + lexical_weight * c.lexical_score

    # 3. Build the three ranked lists and fuse with RRF.
    vector_ranked = _by_vector(candidates)
    lexical_ranked = _by_lexical(candidates)
    summary_ranked = _by_vector(
        [c for c in candidates if c.result.record.record_type == RecordType.SUMMARY.value]
    )

    fused = _reciprocal_rank_fusion(
        [vector_ranked, lexical_ranked, summary_ranked], k=rrf_k
    )

    ranked = sorted(candidates, key=lambda c: fused[c.result.record.id], reverse=True)
    results = []
    for i, c in enumerate(ranked[:top_k]):
        c.result.rank = i + 1
        results.append(c.result)

    # 4. File-level grouping from the summary matches (document-level hits).
    file_groups = _build_file_groups(summary_results, chunk_results)
    return results, file_groups


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


async def summary_search(
    repo: DocumentRepository,
    query_vector: list[float],
    top_k: int = 10,
    folder_filter: str | None = None,
    file_type_filter: str | None = None,
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

    file_groups = _build_file_groups(summary_results, chunk_results)
    for i, r in enumerate(summary_results[:top_k]):
        r.rank = i + 1
    return summary_results[:top_k], file_groups
