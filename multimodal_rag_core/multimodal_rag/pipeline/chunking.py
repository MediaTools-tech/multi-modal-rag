from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from multimodal_rag.config import embedding_max_input_tokens
from multimodal_rag.utils.embedding_budget import (
    chunk_token_budget,
    chunk_token_target,
    count_tokens,
)

if TYPE_CHECKING:
    from multimodal_rag.config import Settings

logger = logging.getLogger(__name__)

CHUNK_SOURCE_START = "chunk_source_start"
CHUNK_SOURCE_END = "chunk_source_end"
CHUNK_OVERLAP_PREFIX = "chunk_overlap_prefix_chars"

_WARNED_CHUNK_BUDGETS: set[tuple[str, int, int]] = set()
_SENTENCE_END_RE = re.compile(r"[.!?。！？]+[\"'\u2019\u201d)\]]*\s*|\n{2,}", re.UNICODE)
_ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "u.s",
    "u.k",
    "fig",
    "no",
}


@dataclass
class Chunk:
    content: str
    index: int
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class _Span:
    start: int
    end: int
    overlap_prefix_chars: int = 0


def chunk_source_span(metadata: dict | None) -> tuple[int, int] | None:
    if not metadata:
        return None
    start = metadata.get(CHUNK_SOURCE_START)
    end = metadata.get(CHUNK_SOURCE_END)
    if isinstance(start, int) and isinstance(end, int) and 0 <= start < end:
        return start, end
    return None


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = (text or "").split()
    if not words:
        return []
    if chunk_size <= 0:
        return [" ".join(words)]
    overlap = max(0, min(overlap, chunk_size - 1))
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
        if start <= 0:
            start = end
    return chunks


def token_chunk_text(
    text: str,
    *,
    model: str,
    local: bool,
    max_tokens: int,
    overlap_tokens: int = 0,
    overlap_ratio: float = 0.0,
    max_words: int = 0,
) -> list[str]:
    spans = token_chunk_spans(
        text,
        model=model,
        local=local,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        overlap_ratio=overlap_ratio,
        max_words=max_words,
    )
    source = (text or "").strip()
    return [source[span.start : span.end] for span in spans]


def token_chunk_spans(
    text: str,
    *,
    model: str,
    local: bool,
    max_tokens: int,
    overlap_tokens: int = 0,
    overlap_ratio: float = 0.0,
    max_words: int = 0,
) -> list[_Span]:
    text = (text or "").strip()
    if not text:
        return []

    max_tokens = max(1, int(max_tokens))

    def counter(value: str) -> int:
        return count_tokens(value, model, local=local)

    overlap_budget = overlap_tokens
    if overlap_budget <= 0 and overlap_ratio > 0:
        overlap_budget = int(max_tokens * overlap_ratio)
    overlap_budget = max(0, min(int(overlap_budget), max_tokens - 1))

    overlap_words = 0
    if overlap_tokens <= 0 and overlap_ratio > 0 and max_words > 0:
        overlap_words = max(0, int(max_words * overlap_ratio))

    base_spans = _sentence_packed_spans(
        text,
        max_tokens=max_tokens,
        overlap_budget=overlap_budget,
        max_words=max_words,
        overlap_words=overlap_words,
        counter=counter,
    )

    spans: list[_Span] = []
    previous_base_start: int | None = None
    previous_base_end: int | None = None
    previous_stored_end: int | None = None
    for base_start, base_end in base_spans:
        start = base_start
        if previous_base_start is not None and overlap_budget > 0:
            start = _overlap_start_for_base(
                text,
                previous_base_start,
                previous_base_end or base_start,
                base_start,
                base_end,
                overlap_budget,
                max_tokens,
                max_words,
                counter,
            )
        start, end = _strip_span(text, start, base_end)
        if end <= start:
            previous_base_start, previous_base_end = base_start, base_end
            continue
        overlap_prefix = 0
        if previous_stored_end is not None and start < previous_stored_end:
            overlap_prefix = previous_stored_end - start
        spans.append(_Span(start=start, end=end, overlap_prefix_chars=overlap_prefix))
        previous_base_start, previous_base_end = base_start, base_end
        previous_stored_end = end
    return spans


def text_to_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
    settings: Settings | None = None,
) -> list[Chunk]:
    if settings is None:
        contents = chunk_text(text, chunk_size, overlap)
        return [Chunk(content=content, index=index) for index, content in enumerate(contents)]

    source = (text or "").strip()
    if not source:
        return []
    spans = _model_aware_chunk_spans(source, chunk_size, overlap, settings)
    return [
        Chunk(
            content=source[span.start : span.end],
            index=index,
            metadata={
                CHUNK_SOURCE_START: span.start,
                CHUNK_SOURCE_END: span.end,
                CHUNK_OVERLAP_PREFIX: span.overlap_prefix_chars,
            },
        )
        for index, span in enumerate(spans)
    ]


def _model_aware_chunk_spans(
    text: str,
    chunk_size: int,
    overlap: int,
    settings: Settings,
) -> list[_Span]:
    model = settings.EMBEDDING_MODEL
    provider = getattr(settings.EMBEDDING_PROVIDER, "value", settings.EMBEDDING_PROVIDER)
    local = provider == "LOCAL"
    window = embedding_max_input_tokens(model)
    safe_budget = chunk_token_budget(model)
    configured = int(getattr(settings, "DOC_CHUNK_TOKENS", 0) or 0)
    max_tokens = chunk_token_target(model, configured)
    if configured > safe_budget:
        key = (model, configured, safe_budget)
        if key not in _WARNED_CHUNK_BUDGETS:
            _WARNED_CHUNK_BUDGETS.add(key)
            logger.warning(
                "DOC_CHUNK_TOKENS=%d exceeds the safe budget for %s; using %d "
                "tokens (model window %d).",
                configured,
                model,
                max_tokens,
                window,
            )

    configured_overlap = int(getattr(settings, "DOC_CHUNK_OVERLAP_TOKENS", 0) or 0)
    overlap_tokens = 0
    overlap_ratio = 0.0
    if configured_overlap > 0:
        overlap_tokens = min(configured_overlap, max(0, max_tokens - 1))
    elif overlap > 0 and chunk_size > 0:
        overlap_ratio = min(1.0, overlap / chunk_size)

    return token_chunk_spans(
        text,
        model=model,
        local=local,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        overlap_ratio=overlap_ratio,
        max_words=chunk_size,
    )


def _sentence_packed_spans(
    text: str,
    *,
    max_tokens: int,
    overlap_budget: int,
    max_words: int,
    overlap_words: int,
    counter: Callable[[str], int],
) -> list[tuple[int, int]]:
    sentence_spans = _sentence_spans(text)
    if not sentence_spans:
        return []

    base_budget = max(1, max_tokens - overlap_budget)
    base_word_cap = max_words
    if max_words > 0:
        base_word_cap = max(1, max_words - overlap_words)

    spans: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end: int | None = None
    budget = max_tokens
    word_cap = max_words
    for start, end in sentence_spans:
        if not _fits(text, start, end, budget, word_cap, counter):
            if current_start is not None:
                spans.append((current_start, current_end or current_start))
                current_start = None
                current_end = None
                budget = base_budget
                word_cap = base_word_cap
            pieces = _split_span_by_budget(
                text,
                start,
                end,
                budget,
                word_cap,
                counter,
                rest_max_tokens=base_budget,
                rest_max_words=base_word_cap,
            )
            spans.extend(pieces)
            budget = base_budget
            word_cap = base_word_cap
            continue
        if current_start is None:
            current_start, current_end = start, end
        elif _fits(text, current_start, end, budget, word_cap, counter):
            current_end = end
        else:
            spans.append((current_start, current_end or current_start))
            budget = base_budget
            word_cap = base_word_cap
            current_start, current_end = start, end
    if current_start is not None:
        spans.append((current_start, current_end or current_start))
    return [(start, end) for start, end in spans if text[start:end].strip()]


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END_RE.finditer(text):
        if _is_abbreviation(text, match.start()):
            continue
        end = match.end()
        if end <= start:
            continue
        spans.append((start, end))
        start = end
    if start < len(text):
        spans.append((start, len(text)))
    return spans or [(0, len(text))]


def _is_abbreviation(text: str, punct_start: int) -> bool:
    match = re.search(r"([A-Za-z.]+)\.$", text[: punct_start + 1])
    if not match:
        return False
    token = match.group(1).lower()
    return token in _ABBREVIATIONS or (len(token) == 1 and token.isalpha())


def _previous_sentence_start(text: str, position: int, floor: int) -> int:
    if position <= floor:
        return floor
    last_end: int | None = None
    for match in _SENTENCE_END_RE.finditer(text, floor, position):
        if _is_abbreviation(text, match.start()):
            continue
        last_end = match.end()
    return last_end if last_end is not None and last_end > floor else floor


def _fits(
    text: str,
    start: int,
    end: int,
    max_tokens: int,
    max_words: int,
    counter: Callable[[str], int],
) -> bool:
    if end <= start:
        return False
    if max_words > 0 and len(text[start:end].split()) > max_words:
        return False
    return counter(text[start:end]) <= max_tokens


def _split_span_by_budget(
    text: str,
    start: int,
    end: int,
    max_tokens: int,
    max_words: int,
    counter: Callable[[str], int],
    *,
    rest_max_tokens: int | None = None,
    rest_max_words: int | None = None,
) -> list[tuple[int, int]]:
    pieces: list[tuple[int, int]] = []
    piece_tokens = max_tokens
    piece_words = max_words
    while start < end:
        piece_end = _max_prefix_end(text, start, piece_tokens, counter, limit=end)
        piece_end = _apply_word_cap(text, start, piece_end, piece_words)
        piece_end = _trim_to_boundary(text, start, piece_end)
        while piece_end > start + 1 and counter(text[start:piece_end]) > piece_tokens:
            piece_end -= 1
        if piece_end <= start:
            piece_end = min(end, start + 1)
        pieces.append((start, piece_end))
        start = piece_end
        if rest_max_tokens is not None:
            piece_tokens = rest_max_tokens
        if rest_max_words is not None:
            piece_words = rest_max_words
    return pieces


def _overlap_start_for_base(
    text: str,
    previous_start: int,
    previous_end: int,
    base_start: int,
    base_end: int,
    overlap_budget: int,
    max_tokens: int,
    max_words: int,
    counter: Callable[[str], int],
) -> int:
    if previous_end <= previous_start or overlap_budget <= 0:
        return base_start

    low, high = previous_start, base_start
    while low < high:
        middle = (low + high) // 2
        if counter(text[middle:base_start]) <= overlap_budget:
            high = middle
        else:
            low = middle + 1
    candidate = low
    sentence_start = _previous_sentence_start(text, candidate, previous_start)
    if sentence_start < candidate and _fits(
        text, sentence_start, base_end, max_tokens, max_words, counter
    ):
        candidate = sentence_start
    else:
        candidate = _next_word_start(text, candidate, base_start)

    while candidate < base_start and not _fits(
        text, candidate, base_end, max_tokens, max_words, counter
    ):
        candidate = _next_word_start(text, candidate, base_start)
    return candidate if candidate < base_start else base_start


def _next_word_start(text: str, start: int, limit: int) -> int:
    if start >= limit:
        return limit
    match = re.search(r"\s+", text[start:limit])
    if not match:
        return start + 1
    return start + match.end()


def _strip_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _max_prefix_end(
    text: str,
    start: int,
    max_tokens: int,
    counter: Callable[[str], int],
    *,
    limit: int | None = None,
) -> int:
    high_limit = len(text) if limit is None else min(limit, len(text))
    if counter(text[start : start + 1]) > max_tokens:
        return start + 1
    low, high = start + 1, high_limit
    while low < high:
        middle = (low + high + 1) // 2
        if counter(text[start:middle]) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return low


def _apply_word_cap(text: str, start: int, end: int, max_words: int) -> int:
    if max_words <= 0:
        return end
    seen = 0
    for match in re.finditer(r"\S+", text[start:end]):
        seen += 1
        if seen >= max_words:
            return min(end, start + match.end())
    return end


def _trim_to_boundary(text: str, start: int, end: int) -> int:
    if end >= len(text):
        return end
    if text[end - 1].isspace() or text[end].isspace():
        return end
    boundary = max(
        text.rfind("\n", start, end),
        text.rfind(" ", start, end),
    )
    if boundary > start and boundary >= start + max(1, (end - start) // 2):
        return boundary
    return end
