from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict

_TOKEN_RE = re.compile(r"[^\W_][\w'+-]*", re.UNICODE)
_GROUPED_NUMBER_RE = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")
_BARE_NUMBER_RE = re.compile(r"(?<![\d.])(\d{4,})(?![\d.])")

#: Low-information English words. They are kept in the index, but removed from
#: query/document scoring when a query has more meaningful terms.
STOPWORDS = frozenset(
    """
    a about after again against all also am an and any are as at be because been
    before being below between both but by can could did do does doing down during
    each few for from further had has have having he her here hers herself him
    himself his how i if in into is it its itself just me more most my myself no
    nor not now of off on once only or other our ours ourselves out over own same
    she should so some such than that the their theirs them themselves then there
    these they this those through to too under until up very was we were what
    when where which while who whom why will with would you your yours yourself
    yourselves
    """.split()
)


def normalize_text(text: str) -> str:
    """Normalize compatibility forms and numeric grouping so variants match."""
    normalized = unicodedata.normalize("NFKC", text or "")
    return _GROUPED_NUMBER_RE.sub("", normalized)


def _group_digits(match: re.Match[str]) -> str:
    value = match.group(1)
    groups: list[str] = []
    while len(value) > 3:
        groups.append(value[-3:])
        value = value[:-3]
    groups.append(value)
    return ",".join(reversed(groups))


def numeric_query_variants(text: str) -> list[str]:
    """Return grouped and ungrouped spellings of numeric terms in a query."""
    original = unicodedata.normalize("NFKC", text or "").strip()
    ungrouped = _GROUPED_NUMBER_RE.sub("", original)
    grouped = _BARE_NUMBER_RE.sub(_group_digits, ungrouped)
    variants: list[str] = []
    for variant in (original, ungrouped, grouped):
        if variant and variant not in variants:
            variants.append(variant)
    return variants


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(normalize_text(text))]


def _scoring_tokens(text: str) -> list[str]:
    tokens = tokenize(text)
    meaningful = [token for token in tokens if token not in STOPWORDS]
    return meaningful or tokens


def bm25_scores(
    query_tokens: list[str],
    docs: list[tuple[str, str]],
    k1: float = 1.2,
    b: float = 0.75,
) -> dict[str, float]:
    if not query_tokens or not docs:
        return {}

    # Deduplicate repeated query words and drop stopwords when real terms remain.
    terms = [token.casefold() for token in query_tokens if token]
    terms = [token for token in terms if token not in STOPWORDS] or terms
    terms = list(dict.fromkeys(terms))
    if not terms:
        return {}

    df: dict[str, int] = defaultdict(int)
    doc_tokens: dict[str, list[str]] = {}
    lengths: dict[str, int] = {}

    for doc_id, text in docs:
        tokens = _scoring_tokens(text)
        doc_tokens[doc_id] = tokens
        lengths[doc_id] = len(tokens)
        present = set(tokens)
        for query_token in terms:
            if query_token in present:
                df[query_token] += 1

    n = max(1, len(doc_tokens))
    avgdl = max(1.0, sum(lengths.values()) / n if lengths else 1.0)

    scores: dict[str, float] = {}
    for doc_id, tokens in doc_tokens.items():
        tf: dict[str, int] = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        dl = max(1, lengths[doc_id])
        score = 0.0
        for query_token in terms:
            if tf[query_token] == 0:
                continue
            idf = math.log((n - df[query_token] + 0.5) / (df[query_token] + 0.5) + 1.0)
            score += idf * (tf[query_token] * (k1 + 1)) / (
                tf[query_token] + k1 * (1 - b + b * dl / avgdl)
            )
        scores[doc_id] = score
    return scores
