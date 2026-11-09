from __future__ import annotations

import gzip
import json
import logging
from collections import Counter
from collections.abc import Mapping
from functools import lru_cache
from importlib import resources
from pathlib import Path

from multimodal_rag.config import KeywordExpansion, Settings
from multimodal_rag.core.models import RecordType, SearchResult
from multimodal_rag.search.hybrid import rrf_fuse
from multimodal_rag.search.lexical import STOPWORDS, tokenize

logger = logging.getLogger(__name__)


def _valid_term(term: str, query_terms: set[str]) -> bool:
    return (
        3 <= len(term) <= 24
        and term not in query_terms
        and term not in STOPWORDS
        and term.isalpha()
    )


@lru_cache(maxsize=1)
def synonym_map() -> Mapping[str, tuple[str, ...]]:
    """Load the bundled, compact WordNet-derived synonym map once per process."""
    raw: bytes | None = None
    try:
        resource = resources.files("multimodal_rag").joinpath("assets/synonyms.json.gz")
        raw = resource.read_bytes()
    except Exception:  # pragma: no cover - fallback for unusual installers
        path = Path(__file__).resolve().parent.parent / "assets" / "synonyms.json.gz"
        try:
            raw = path.read_bytes()
        except OSError:
            logger.warning("bundled synonym map not found; SYNONYMS expansion disabled")
            return {}
    try:
        data = json.loads(gzip.decompress(raw).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("could not read bundled synonym map: %s", exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("bundled synonym map has an unexpected format")
        return {}
    return {
        str(term): tuple(str(synonym) for synonym in synonyms)
        for term, synonyms in data.items()
        if isinstance(synonyms, list)
    }


def synonym_expansion_terms(query: str, max_terms: int = 3) -> list[str]:
    """Return generic synonyms for the query's content words.

    The map is intentionally small (common English words only), so names,
    product codes, and domain-specific identifiers are left untouched.
    """
    if max_terms <= 0:
        return []
    query_terms = [
        term
        for term in tokenize(query)
        if term.isalpha() and len(term) >= 3 and term not in STOPWORDS
    ]
    if not query_terms:
        return []
    mapping = synonym_map()
    seen = set(query_terms)
    terms: list[str] = []
    for query_term in query_terms:
        for synonym in mapping.get(query_term, ()):
            if synonym in seen or not synonym.isalpha():
                continue
            seen.add(synonym)
            terms.append(synonym)
            if len(terms) >= max_terms:
                return terms
    return terms


def corpus_expansion_terms(
    query: str,
    results: list[SearchResult],
    max_terms: int = 3,
    min_doc_frequency: int = 2,
) -> list[str]:
    """Pick salient terms from the strongest lexical hits.

    Pseudo-relevance feedback is only useful when the top hits are coherent.
    Weak tail hits (often generic visual captions such as "visible", "setting",
    "text") would otherwise contribute boilerplate terms that flood the fused
    ranking. Seeds are therefore limited to the top score band, and terms that
    are common across the whole retrieved pool are treated as background noise.
    """
    if max_terms <= 0 or len(results) < min_doc_frequency:
        return []
    query_terms = set(tokenize(query))
    best_score = max(result.score for result in results)
    seeds = [
        result for result in results if result.score >= best_score * 0.25
    ][:10]

    background_frequency: Counter[str] = Counter()
    for result in results:
        terms = {
            term
            for term in tokenize(result.record.content)
            if _valid_term(term, query_terms)
        }
        background_frequency.update(terms)

    doc_frequency: Counter[str] = Counter()
    term_frequency: Counter[str] = Counter()
    for result in seeds:
        terms = [
            term
            for term in tokenize(result.record.content)
            if _valid_term(term, query_terms)
        ]
        doc_frequency.update(set(terms))
        term_frequency.update(terms)

    ranked = [
        (term, count)
        for term, count in doc_frequency.items()
        if count >= min_doc_frequency
    ]
    ranked.sort(key=lambda item: (-item[1], -term_frequency[item[0]], item[0]))
    background_limit = min(
        len(results) - 1,
        max(min_doc_frequency, len(results) // 2),
    )
    ranked = [
        (term, count)
        for term, count in ranked
        if background_frequency[term] <= background_limit
    ]
    return [term for term, _ in ranked[:max_terms]]


def keyword_search(
    repo,
    query: str,
    top_k: int = 10,
    folder_prefix: str | None = None,
    expansion: str = KeywordExpansion.OFF.value,
    max_expansion_terms: int = 3,
) -> list[SearchResult]:
    """Lexical-only retrieval with optional query expansion.

    No query embedding and no cross-encoder are used. Expansion terms are searched
    separately and fused with the original ranking, so the base query always keeps
    the strongest weight and backend-specific OR/FTS syntax is not required.
    """
    expansion = getattr(expansion, "value", expansion)
    max_expansion_terms = max(0, min(max_expansion_terms, 10))
    pool_size = max(top_k * 4, 20)
    base = repo.search_lexical(
        query,
        top_k=pool_size,
        record_types=[RecordType.CHUNK.value],
        folder_prefix=folder_prefix,
    )

    synonym_terms: list[str] = []
    corpus_terms: list[str] = []
    if expansion in {
        KeywordExpansion.SYNONYMS.value,
        KeywordExpansion.CORPUS_SYNONYMS.value,
    }:
        synonym_terms = synonym_expansion_terms(query, max_expansion_terms)
    # KEYWORD_EXPANSION_TERMS is the total budget, not a per-source budget.
    remaining = max_expansion_terms - len(synonym_terms)
    if expansion in {
        KeywordExpansion.CORPUS.value,
        KeywordExpansion.CORPUS_SYNONYMS.value,
    } and base and remaining > 0:
        corpus_terms = corpus_expansion_terms(query, base, remaining)

    expansions: list[tuple[str, float]] = []
    seen = set(tokenize(query))
    for term in synonym_terms:
        if term not in seen:
            seen.add(term)
            expansions.append((term, 1.5))
    for term in corpus_terms:
        if term not in seen:
            seen.add(term)
            expansions.append((term, 1.0))

    if not base and not expansions:
        return []
    if not expansions:
        return base[:top_k]

    ranked_lists: list[list[SearchResult]] = []
    weights: list[float] = []
    if base:
        ranked_lists.append(base)
        weights.append(3.0)
    for term, weight in expansions:
        expanded = repo.search_lexical(
            term,
            top_k=pool_size,
            record_types=[RecordType.CHUNK.value],
            folder_prefix=folder_prefix,
        )
        if expanded:
            ranked_lists.append(expanded)
            weights.append(weight)
    if not ranked_lists:
        return []
    if len(ranked_lists) == 1 and base:
        return base[:top_k]

    fused = rrf_fuse(ranked_lists, k=60, weights=weights)[:top_k]
    for rank, result in enumerate(fused, start=1):
        result.rank = rank
    logger.info(
        "keyword expansion terms: %s",
        [f"{term} ({weight:g})" for term, weight in expansions],
    )
    return fused


def run_keyword_search(
    settings: Settings,
    repo,
    query: str,
    top_k: int,
    folder: str | None,
):
    """Settings-aware wrapper used by the CLI/GUI search paths."""
    return (
        keyword_search(
            repo,
            query,
            top_k=top_k,
            folder_prefix=folder,
            expansion=settings.KEYWORD_EXPANSION.value,
            max_expansion_terms=settings.KEYWORD_EXPANSION_TERMS,
        ),
        [],
    )
