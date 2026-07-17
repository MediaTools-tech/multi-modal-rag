"""Shared lexical / full-text scoring utilities.

These power the *corpus-wide* keyword ranking (as opposed to the earlier
query-scoped lexical signal) so keyword search covers every indexed document,
not just the vector top-k. ``bm25_scores`` runs in memory over a retrieved
corpus subset and is used as the LanceDB fallback when a native FTS index is
unavailable.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

_TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9'+-]*")


def tokenize(text: str) -> list[str]:
    """Lower-cased tokenization for lexical scoring."""
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def bm25_scores(
    query_tokens: list[str],
    docs: list[tuple[str, str]],
    k1: float = 1.2,
    b: float = 0.75,
) -> dict[str, float]:
    """Compute BM25 scores over an in-memory corpus subset.

    Args:
        query_tokens: Tokenized query.
        docs: List of ``(doc_id, text)`` pairs.
        k1, b: Standard BM25 free parameters.

    Returns:
        Mapping of ``doc_id`` -> raw BM25 score (unnormalized).
    """
    if not query_tokens or not docs:
        return {}

    df: dict[str, int] = defaultdict(int)
    doc_tokens: dict[str, list[str]] = {}
    lengths: dict[str, int] = {}

    for doc_id, text in docs:
        toks = tokenize(text)
        doc_tokens[doc_id] = toks
        lengths[doc_id] = len(toks)
        present = set(toks)
        for qt in query_tokens:
            if qt in present:
                df[qt] += 1

    n = max(1, len(doc_tokens))
    avgdl = sum(lengths.values()) / n if lengths else 1.0

    scores: dict[str, float] = {}
    for doc_id, toks in doc_tokens.items():
        tf = defaultdict(int)
        for t in toks:
            tf[t] += 1
        dl = max(1, lengths[doc_id])
        score = 0.0
        for qt in query_tokens:
            if tf[qt] == 0:
                continue
            idf = math.log((n - df[qt] + 0.5) / (df[qt] + 0.5) + 1.0)
            score += idf * (tf[qt] * (k1 + 1)) / (tf[qt] + k1 * (1 - b + b * dl / avgdl))
        scores[doc_id] = score
    return scores
