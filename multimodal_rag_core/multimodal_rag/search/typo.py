from __future__ import annotations

import logging
import re
import threading
from collections.abc import Mapping
from difflib import SequenceMatcher
from typing import Any

from multimodal_rag.config import Settings, TypoCorrectionMode

logger = logging.getLogger(__name__)

_SPELL_CACHE: tuple[Mapping[str, int], Any] | None = None
_SPELL_LOCK = threading.Lock()

_TOKEN_RE = re.compile(r"[^\W_][\w'+-]*", re.UNICODE)
TYPO_CORRECTION_PROMPT = (
    "Fix spelling mistakes in the search query below. Return ONLY the corrected "
    "query on one line, with no quotes or explanation. Preserve names, numbers, "
    "product codes, language, and intent. If no word is misspelled, return the "
    "query unchanged.\n\nQUERY: {query}"
)


def correct_query_local(
    query: str,
    vocabulary: Mapping[str, int] | None,
    *,
    min_word_length: int = 3,
) -> str:
    """Correct unknown query words against indexed word forms.

    Only out-of-vocabulary words are touched. SymSpell's deletion index provides
    fast candidate lookup over the corpus vocabulary; candidates must still be
    within a small Damerau-Levenshtein distance, so names and codes are left
    alone unless they are genuinely close to an indexed word. If symspellpy is
    unavailable, the small built-in matcher keeps local correction working.
    """
    if not query or not vocabulary:
        return query

    spell = _symspell_for(vocabulary)
    buckets: dict[int, list[tuple[str, int]]] = {}
    if spell is None:
        for word, frequency in vocabulary.items():
            buckets.setdefault(len(word), []).append((word, int(frequency)))

    replacements: list[tuple[int, int, str]] = []
    for match in _TOKEN_RE.finditer(query):
        original = match.group(0)
        word = original.lower()
        if len(word) < min_word_length or word in vocabulary:
            continue
        if any(character.isdigit() for character in word):
            continue
        max_edits = 1 if len(word) <= 6 else 2
        candidate = (
            _best_symspell_candidate(spell, word, max_edits)
            if spell is not None
            else _best_candidate(word, buckets, max_edits)
        )
        if candidate and candidate != word:
            replacements.append(
                (match.start(), match.end(), _match_case(original, candidate))
            )

    if not replacements:
        return query
    parts: list[str] = []
    cursor = 0
    for start, end, replacement in replacements:
        parts.append(query[cursor:start])
        parts.append(replacement)
        cursor = end
    parts.append(query[cursor:])
    return "".join(parts)


def correct_query_llm(
    settings: Settings,
    query: str,
    tracker=None,
) -> str | None:
    """Return an LLM-rewritten query, or None when no LLM is available/failed."""
    try:
        from multimodal_rag.utils.api_clients import build_chat_client

        client = build_chat_client(settings, tracker)
        if client is None:
            return None
        answer = client.generate(
            TYPO_CORRECTION_PROMPT.format(query=query), temperature=0.0
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM typo correction failed: %s", exc)
        return None
    text = (answer or "").strip().strip("`\"'")
    if "\n" in text:
        text = text.splitlines()[0].strip()
    text = re.sub(r"^(corrected( query)?|query)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("`\"'")
    if not text or len(text) > max(200, len(query) * 4):
        return None
    return text


def prepare_query(settings: Settings, repo, query: str, tracker=None) -> str:
    """Apply the configured typo-correction mode to a search query."""
    mode = getattr(settings, "TYPO_CORRECTION", TypoCorrectionMode.LOCAL)
    mode = getattr(mode, "value", mode)
    if mode == TypoCorrectionMode.OFF.value:
        return query
    if mode == TypoCorrectionMode.LLM.value:
        corrected = correct_query_llm(settings, query, tracker)
        if corrected is not None:
            if corrected != query:
                logger.info("typo correction (LLM): %r -> %r", query, corrected)
            return corrected
        logger.warning("LLM typo correction unavailable; using local correction")

    vocabulary: Mapping[str, int] = {}
    try:
        if hasattr(repo, "lexical_vocabulary"):
            vocabulary = repo.lexical_vocabulary()
    except Exception as exc:  # noqa: BLE001
        logger.warning("local typo vocabulary unavailable: %s", exc)
        return query
    corrected = correct_query_local(query, vocabulary)
    if corrected != query:
        logger.info("typo correction (local): %r -> %r", query, corrected)
    return corrected


def _symspell_for(vocabulary: Mapping[str, int]):
    """Build/reuse a SymSpell index for one backend vocabulary snapshot."""
    global _SPELL_CACHE
    with _SPELL_LOCK:
        cached = _SPELL_CACHE
    if cached is not None and cached[0] is vocabulary:
        return cached[1]
    try:
        from symspellpy import SymSpell
    except ImportError:
        logger.warning("symspellpy is not installed; using built-in typo matcher")
        return None
    spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    for word, frequency in vocabulary.items():
        count = int(frequency)
        if word and count > 0:
            spell.create_dictionary_entry(word, count)
    with _SPELL_LOCK:
        _SPELL_CACHE = (vocabulary, spell)
    return spell


def _best_symspell_candidate(spell: Any, word: str, max_edits: int) -> str | None:
    from symspellpy import Verbosity

    suggestions = spell.lookup(
        word,
        Verbosity.CLOSEST,
        max_edit_distance=max_edits,
        include_unknown=False,
    )
    best: tuple[int, int, int, str] | None = None
    for suggestion in suggestions:
        candidate = suggestion.term
        if candidate == word:
            continue
        if len(word) <= 4 and SequenceMatcher(None, word, candidate).ratio() <= 0.75:
            continue
        distance = int(suggestion.distance)
        if distance > max_edits:
            continue
        score = (
            distance,
            0 if candidate[:1] == word[:1] else 1,
            -int(suggestion.count),
            candidate,
        )
        if best is None or score < best:
            best = score
    return best[3] if best is not None else None


def _best_candidate(
    word: str,
    buckets: dict[int, list[tuple[str, int]]],
    max_edits: int,
) -> str | None:
    candidates: list[tuple[str, int]] = []
    for length in range(len(word) - max_edits, len(word) + max_edits + 1):
        candidates.extend(buckets.get(length, ()))
    best: tuple[int, int, int, str] | None = None
    for candidate, frequency in candidates:
        if candidate == word:
            continue
        if len(word) <= 4 and SequenceMatcher(None, word, candidate).ratio() <= 0.75:
            continue
        distance = _damerau_levenshtein(word, candidate, max_edits)
        if distance > max_edits:
            continue
        score = (
            distance,
            0 if candidate[:1] == word[:1] else 1,
            -frequency,
            candidate,
        )
        if best is None or score < best:
            best = score
    return best[3] if best is not None else None


def _damerau_levenshtein(a: str, b: str, max_distance: int) -> int:
    if abs(len(a) - len(b)) > max_distance:
        return max_distance + 1
    previous_previous: list[int] | None = None
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, right in enumerate(b, start=1):
            cost = 0 if left == right else 1
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
            if (
                i > 1
                and j > 1
                and left == b[j - 2]
                and a[i - 2] == right
            ):
                current[j] = min(current[j], previous_previous[j - 2] + 1)  # type: ignore[index]
        previous_previous, previous = previous, current
    return previous[-1]


def _match_case(original: str, corrected: str) -> str:
    if original.isupper():
        return corrected.upper()
    if original[:1].isupper():
        return corrected[:1].upper() + corrected[1:]
    return corrected
