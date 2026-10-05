from __future__ import annotations

import math
import re
from collections import defaultdict

_TOKEN_RE = re.compile(r"[^\W_][\w'+-]*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


def bm25_scores(
    query_tokens: list[str],
    docs: list[tuple[str, str]],
    k1: float = 1.2,
    b: float = 0.75,
) -> dict[str, float]:
    if not query_tokens or not docs:
        return {}

    df: dict[str, int] = defaultdict(int)
    doc_tokens: dict[str, list[str]] = {}
    lengths: dict[str, int] = {}

    for doc_id, text in docs:
        tokens = tokenize(text)
        doc_tokens[doc_id] = tokens
        lengths[doc_id] = len(tokens)
        present = set(tokens)
        for query_token in query_tokens:
            if query_token in present:
                df[query_token] += 1

    n = max(1, len(doc_tokens))
    avgdl = sum(lengths.values()) / n if lengths else 1.0

    scores: dict[str, float] = {}
    for doc_id, tokens in doc_tokens.items():
        tf: dict[str, int] = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        dl = max(1, lengths[doc_id])
        score = 0.0
        for query_token in query_tokens:
            if tf[query_token] == 0:
                continue
            idf = math.log((n - df[query_token] + 0.5) / (df[query_token] + 0.5) + 1.0)
            score += idf * (tf[query_token] * (k1 + 1)) / (
                tf[query_token] + k1 * (1 - b + b * dl / avgdl)
            )
        scores[doc_id] = score
    return scores
