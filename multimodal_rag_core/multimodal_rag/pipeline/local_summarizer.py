from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from multimodal_rag.config import Settings, SummaryAlgorithm, SummarySemanticUnit
from multimodal_rag.core.models import RecordType, VectorRecord
from multimodal_rag.pipeline.chunking import chunk_source_span

logger = logging.getLogger(__name__)

EmbedTexts = Callable[[list[str]], list[list[float]]]

_MARKER_RE = re.compile(r"\[(?:Visual|OCR|Transcript|Audio|Speech)[^\]]*\]", re.IGNORECASE)
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+", re.UNICODE)
_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")

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

_ENGLISH_STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


@dataclass(frozen=True)
class LocalSummary:
    text: str
    algorithm: str
    semantic_unit: str
    input_chars: int
    output_words: int
    chunks_considered: int
    chunks_selected: int
    elapsed_ms: int
    fallback: bool = False


@dataclass
class _Candidate:
    record: VectorRecord
    text: str
    vector: list[float]


@dataclass
class _Sentence:
    text: str
    order: int
    parent_score: float


def clean_summary_text(text: str) -> str:
    text = _MARKER_RE.sub(" ", text or "")
    text = text.replace("\x00", " ")
    return _WHITESPACE_RE.sub(" ", text).strip()


def trim_summary_text(text: str, max_words: int, max_chars: int) -> str:
    words = text.split()
    if max_words > 0 and len(words) > max_words:
        text = " ".join(words[:max_words])
    if max_chars > 0 and len(text) > max_chars:
        clipped = text[:max_chars].rstrip()
        boundary = clipped.rfind(" ")
        if boundary >= max(1, max_chars // 2):
            clipped = clipped[:boundary]
        text = clipped.rstrip(" ,;:-")
    return text.strip()


def deoverlap_record_texts(records: Sequence[VectorRecord]) -> list[str]:
    parts: list[str] = []
    previous_end: int | None = None
    for record in records:
        raw = record.content or ""
        span = chunk_source_span(record.metadata)
        if span is not None:
            start, end = span
            if previous_end is not None and start < previous_end:
                raw = raw[min(len(raw), previous_end - start) :]
            previous_end = max(previous_end, end) if previous_end is not None else end
        else:
            previous_end = None
        part = clean_summary_text(raw)
        if part:
            parts.append(part)
    return parts


def _normalize_scores(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [1.0 if high > 0 else 0.0 for _ in values]
    span = high - low
    return [(float(value) - low) / span for value in values]


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    tail_chars = max(1, max_chars // 4)
    head_chars = max_chars - tail_chars - 1
    return f"{text[:head_chars].rstrip()} {text[-tail_chars:].lstrip()}".strip()


def _even_sample(items: Sequence, limit: int) -> list:
    if limit <= 0 or len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[0]]
    indices: list[int] = []
    for position in range(limit):
        index = round(position * (len(items) - 1) / (limit - 1))
        if not indices or index != indices[-1]:
            indices.append(index)
    return [items[index] for index in indices]


def _word_overlap(
    previous: str, following: str, max_overlap: int | None = None
) -> int:
    previous_words = previous.split()
    following_words = following.split()
    if max_overlap is None:
        max_overlap = max(80, min(len(previous_words), len(following_words)) // 2)
    limit = min(len(previous_words), len(following_words), max_overlap)
    for size in range(limit, 0, -1):
        if previous_words[-size:] == following_words[:size]:
            return size
    return 0


def _stitch_texts(texts: Sequence[str]) -> str:
    output: list[str] = []
    previous = ""
    previous_words: list[str] = []
    for text in texts:
        text = text.strip()
        if not text:
            continue
        if not output:
            output.append(text)
            previous = text
            previous_words = text.split()
            continue
        overlap = _word_overlap(previous, text)
        following_words = text.split()
        if overlap:
            output.append(" ".join(following_words[overlap:]))
            previous_words.extend(following_words[overlap:])
            previous = " ".join(previous_words)
        else:
            output.append(text)
            previous = text
            previous_words = following_words
    return " ".join(part for part in output if part).strip()


def _sentence_key(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _split_sentences(text: str) -> list[str]:
    text = clean_summary_text(text)
    if not text:
        return []
    pieces = _SPLIT_RE.split(text)
    sentences: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if sentences and _ends_with_abbreviation(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return sentences


def _ends_with_abbreviation(sentence: str) -> bool:
    match = re.search(r"([A-Za-z.]+)\.$", sentence.strip())
    if not match:
        return False
    token = match.group(1).lower()
    return token in _ABBREVIATIONS or (len(token) == 1 and token.isalpha())


def _sentence_similarity(left: str, right: str) -> float:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def _record_sort_key(record: VectorRecord) -> tuple[int, float, int]:
    if record.timestamp_start is not None:
        return (0, float(record.timestamp_start), int(record.chunk_index))
    return (1, float(record.chunk_index), 0)


class LocalExtractiveSummarizer:
    def __init__(
        self,
        settings: Settings | None = None,
        embed_texts: EmbedTexts | None = None,
    ) -> None:
        self.settings = settings
        self.embed_texts = embed_texts
        self.target_words = max(20, int(getattr(settings, "SUMMARY_TARGET_WORDS", 120)))
        self.output_max_chars = max(200, int(getattr(settings, "SUMMARY_OUTPUT_MAX_CHARS", 900)))
        self.local_max_chars = max(1000, int(getattr(settings, "SUMMARY_LOCAL_MAX_CHARS", 30000)))
        self.max_chunks = max(8, int(getattr(settings, "SUMMARY_MAX_CHUNKS", 400)))
        self.max_sentences = max(20, int(getattr(settings, "SUMMARY_MAX_SENTENCES", 400)))
        self.mmr_lambda = min(1.0, max(0.0, float(getattr(settings, "SUMMARY_MMR_LAMBDA", 0.7))))
        self.similarity_threshold = min(
            1.0, max(0.0, float(getattr(settings, "SUMMARY_SIMILARITY_THRESHOLD", 0.10)))
        )
        algorithm = getattr(settings, "SUMMARY_ALGORITHM", SummaryAlgorithm.SEMANTIC_LEXRANK)
        self.algorithm = algorithm.value if hasattr(algorithm, "value") else str(algorithm).upper()
        unit = getattr(settings, "SUMMARY_SEMANTIC_UNIT", SummarySemanticUnit.CHUNK)
        self.semantic_unit = unit.value if hasattr(unit, "value") else str(unit).upper()
        language = str(getattr(settings, "SUMMARY_LANGUAGE", "english") or "english").lower()
        self.stopwords = _ENGLISH_STOPWORDS if language.startswith("en") else set()

    def summarize(
        self,
        text: str,
        filename: str,
        records: list[VectorRecord] | None = None,
    ) -> LocalSummary:
        started = time.perf_counter()
        ordered = self._ordered_records(records or [])
        candidates = self._candidates(ordered)
        fallback_text = clean_summary_text(text) or self._records_text(ordered)

        if self.algorithm != SummaryAlgorithm.FREQUENCY.value:
            if self.semantic_unit == SummarySemanticUnit.SENTENCE.value:
                if candidates and self.embed_texts is not None:
                    try:
                        result = self._sentence_summary(ordered, candidates, started)
                        if result.text:
                            return result
                    except Exception as exc:
                        logger.warning(
                            "Sentence-level semantic summary failed for %s: %s", filename, exc
                        )
            if candidates:
                try:
                    result = self._chunk_summary(candidates, started)
                    if result.text:
                        return result
                except Exception as exc:
                    logger.warning("Semantic summary failed for %s: %s", filename, exc)

        return self._frequency_summary(
            fallback_text,
            ordered,
            started,
            fallback=self.algorithm != SummaryAlgorithm.FREQUENCY.value,
        )

    def _ordered_records(self, records: Sequence[VectorRecord]) -> list[VectorRecord]:
        usable = [
            record
            for record in records
            if record.record_type != RecordType.SUMMARY.value and (record.content or "").strip()
        ]
        return sorted(usable, key=_record_sort_key)

    def _candidates(self, records: Sequence[VectorRecord]) -> list[_Candidate]:
        chosen = _even_sample(records, self.max_chunks)
        candidates: list[_Candidate] = []
        for record in chosen:
            content = clean_summary_text(record.content)
            if not content:
                continue
            vector: list[float] = []
            for value in record.vector or []:
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(numeric):
                    vector.append(numeric)
            candidates.append(_Candidate(record=record, text=content, vector=vector))
        return candidates

    def _records_text(self, records: Sequence[VectorRecord]) -> str:
        chosen = _even_sample(records, self.max_chunks)
        if chosen and all(chunk_source_span(record.metadata) for record in chosen):
            return _truncate_text(
                " ".join(deoverlap_record_texts(chosen)), self.local_max_chars
            )
        texts = [clean_summary_text(record.content) for record in chosen]
        return _truncate_text(_stitch_texts(texts), self.local_max_chars)

    def _chunk_summary(self, candidates: list[_Candidate], started: float) -> LocalSummary:
        try:
            import numpy as np
        except Exception:
            return self._empty_summary(candidates, "FREQUENCY", "NONE", started, fallback=True)

        vectors = [candidate.vector for candidate in candidates]
        graph = self._semantic_graph(vectors, np)
        if graph is None:
            return self._empty_summary(
                candidates, "FREQUENCY", "NONE", started, fallback=True
            )
        active_indices, similarity, centrality = graph
        active = [candidates[index] for index in active_indices]
        if not active:
            return self._empty_summary(candidates, "FREQUENCY", "NONE", started, fallback=True)

        selected = self._select_chunks(active, similarity, centrality)
        if not selected:
            return self._empty_summary(candidates, "FREQUENCY", "NONE", started, fallback=True)

        selected_sorted = sorted(selected)
        sentences: list[_Sentence] = []
        seen: set[str] = set()
        previous_end: int | None = None
        selected_text_chars = 0
        for index in selected_sorted:
            parent_score = float(centrality[index])
            record = active[index].record
            span = chunk_source_span(record.metadata)
            if span is not None:
                start, end = span
                raw = record.content or ""
                if previous_end is not None and start < previous_end:
                    raw = raw[min(len(raw), previous_end - start) :]
                previous_end = max(previous_end, end) if previous_end is not None else end
                text = clean_summary_text(raw)
            else:
                previous_end = None
                text = active[index].text
            selected_text_chars += len(text)
            for sentence in _split_sentences(text):
                key = _sentence_key(sentence)
                if not key or key in seen:
                    continue
                seen.add(key)
                sentences.append(
                    _Sentence(
                        text=sentence,
                        order=len(sentences),
                        parent_score=parent_score,
                    )
                )
        if not sentences:
            return self._empty_summary(candidates, "FREQUENCY", "NONE", started, fallback=True)

        sentences = _even_sample(sentences, self.max_sentences)
        for order, sentence in enumerate(sentences):
            sentence.order = order
        chosen = self._select_sentences(sentences)
        summary = self._render(sentences, chosen)
        if not summary:
            return self._empty_summary(candidates, "FREQUENCY", "NONE", started, fallback=True)

        return LocalSummary(
            text=summary,
            algorithm=SummaryAlgorithm.SEMANTIC_LEXRANK.value,
            semantic_unit=SummarySemanticUnit.CHUNK.value,
            input_chars=selected_text_chars,
            output_words=len(summary.split()),
            chunks_considered=len(candidates),
            chunks_selected=len(selected),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    def _sentence_summary(
        self,
        ordered: Sequence[VectorRecord],
        candidates: Sequence[_Candidate],
        started: float,
    ) -> LocalSummary:
        try:
            import numpy as np
        except Exception:
            return self._empty_summary(
                list(candidates), "FREQUENCY", "NONE", started, fallback=True
            )

        text = self._records_text(ordered)
        if not text:
            return self._empty_summary(
                list(candidates), "FREQUENCY", "NONE", started, fallback=True
            )
        sentences = [
            _Sentence(text=sentence, order=index, parent_score=0.0)
            for index, sentence in enumerate(_split_sentences(text))
        ]
        sentences = _even_sample(sentences, self.max_sentences)
        for order, sentence in enumerate(sentences):
            sentence.order = order
        if len(sentences) < 2:
            return self._empty_summary(
                list(candidates), "FREQUENCY", "NONE", started, fallback=True
            )

        vectors = (
            self.embed_texts([sentence.text for sentence in sentences])
            if self.embed_texts
            else []
        )
        if len(vectors) != len(sentences):
            raise RuntimeError(
                f"sentence embedder returned {len(vectors)} vectors for {len(sentences)} sentences"
            )
        graph = self._semantic_graph(vectors, np)
        if graph is None:
            return self._empty_summary(
                list(candidates), "FREQUENCY", "NONE", started, fallback=True
            )
        active_indices, similarity, centrality = graph
        active_sentences = [sentences[index] for index in active_indices]
        for order, sentence in enumerate(active_sentences):
            sentence.order = order
        chosen = self._select_sentences(active_sentences, similarity, centrality)
        summary = self._render(active_sentences, chosen)
        if not summary:
            return self._empty_summary(
                list(candidates), "FREQUENCY", "NONE", started, fallback=True
            )
        return LocalSummary(
            text=summary,
            algorithm=SummaryAlgorithm.SEMANTIC_LEXRANK.value,
            semantic_unit=SummarySemanticUnit.SENTENCE.value,
            input_chars=len(text),
            output_words=len(summary.split()),
            chunks_considered=len(candidates),
            chunks_selected=0,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    def _frequency_summary(
        self,
        text: str,
        records: Sequence[VectorRecord],
        started: float,
        *,
        fallback: bool = True,
    ) -> LocalSummary:
        text = clean_summary_text(text) or self._records_text(records)
        text = _truncate_text(text, self.local_max_chars)
        if not text:
            return LocalSummary(
                text="",
                algorithm=SummaryAlgorithm.FREQUENCY.value,
                semantic_unit="NONE",
                input_chars=0,
                output_words=0,
                chunks_considered=len(records),
                chunks_selected=0,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                fallback=fallback,
            )
        sentences = [
            _Sentence(text=sentence, order=index, parent_score=0.0)
            for index, sentence in enumerate(_split_sentences(text))
        ]
        sentences = _even_sample(sentences, self.max_sentences)
        for order, sentence in enumerate(sentences):
            sentence.order = order
        chosen = self._select_sentences(sentences)
        summary = self._render(sentences, chosen)
        return LocalSummary(
            text=summary,
            algorithm=SummaryAlgorithm.FREQUENCY.value,
            semantic_unit="NONE",
            input_chars=len(text),
            output_words=len(summary.split()),
            chunks_considered=len(records),
            chunks_selected=0,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            fallback=fallback,
        )

    def _semantic_graph(self, vectors: Sequence[Sequence[float]], np):
        groups = Counter(
            len(vector) for vector in vectors if vector is not None and len(vector) > 0
        )
        if not groups:
            return None
        dimension = max(groups, key=lambda size: (groups[size], size))
        rows = []
        active_indices: list[int] = []
        for index, vector in enumerate(vectors):
            if len(vector) != dimension:
                continue
            try:
                row = np.asarray(vector, dtype=np.float32)
            except (TypeError, ValueError):
                continue
            norm = float(np.linalg.norm(row))
            if not math.isfinite(norm) or norm <= 0:
                continue
            rows.append(row / norm)
            active_indices.append(index)
        if not rows:
            return None
        matrix = np.stack(rows)
        similarity = matrix @ matrix.T
        np.fill_diagonal(similarity, 0.0)
        similarity = np.maximum(similarity - self.similarity_threshold, 0.0)
        centrality = self._lexrank(similarity, np)
        return active_indices, similarity, centrality

    def _lexrank(self, similarity, np):
        size = similarity.shape[0]
        if size == 0:
            return np.zeros(0, dtype=np.float32)
        row_sums = similarity.sum(axis=1)
        transition = np.zeros_like(similarity)
        nonzero = row_sums > 0
        transition[nonzero] = similarity[nonzero] / row_sums[nonzero, None]
        transition[~nonzero] = 1.0 / size
        scores = np.full(size, 1.0 / size, dtype=np.float32)
        damping = 0.85
        for _ in range(100):
            new_scores = (1.0 - damping) / size + damping * transition.T.dot(scores)
            if float(np.abs(new_scores - scores).sum()) < 1e-6:
                return new_scores
            scores = new_scores
        return scores

    def _select_chunks(self, candidates: Sequence[_Candidate], similarity, centrality) -> list[int]:
        size = len(candidates)
        if size == 0:
            return []
        centrality_values = [float(value) for value in centrality]
        normalized = _normalize_scores(centrality_values)
        position_bonus = [0.0] * size
        position_bonus[0] = 0.06
        if size > 1:
            position_bonus[-1] = 0.03
        selected: list[int] = []
        remaining = set(range(size))
        selected_words = 0
        minimum = min(size, 3)
        maximum = min(size, 16)
        target_input_words = max(self.target_words * 4, 300)
        while remaining and len(selected) < maximum:
            if selected and selected_words >= target_input_words and len(selected) >= minimum:
                break
            if not selected:
                scores = [
                    self.mmr_lambda * normalized[index] + position_bonus[index]
                    for index in range(size)
                ]
            else:
                scores = []
                for index in range(size):
                    redundancy = max(float(similarity[index][chosen]) for chosen in selected)
                    scores.append(
                        self.mmr_lambda * normalized[index]
                        + position_bonus[index]
                        - (1.0 - self.mmr_lambda) * redundancy
                    )
            best = max(remaining, key=lambda index: (scores[index], -index))
            selected.append(best)
            remaining.remove(best)
            selected_words += len(candidates[best].text.split())
        return selected

    def _select_sentences(
        self,
        sentences: Sequence[_Sentence],
        similarity=None,
        centrality=None,
    ) -> list[int]:
        size = len(sentences)
        if size == 0:
            return []
        if centrality is not None:
            raw_scores = [float(value) for value in centrality]
        else:
            raw_scores = self._sentence_scores(sentences)
        scores = _normalize_scores(raw_scores)
        selected: list[int] = []
        remaining = set(range(size))
        selected_words = 0
        target_words = max(self.target_words, 20)
        while remaining and selected_words < target_words:
            if not selected:
                mmr = scores
            else:
                mmr = []
                for index in range(size):
                    if similarity is not None:
                        redundancy = max(float(similarity[index][chosen]) for chosen in selected)
                    else:
                        redundancy = max(
                            _sentence_similarity(
                                sentences[index].text, sentences[chosen].text
                            )
                            for chosen in selected
                        )
                    mmr.append(
                        self.mmr_lambda * scores[index]
                        - (1.0 - self.mmr_lambda) * redundancy
                    )
            best = max(remaining, key=lambda index: (mmr[index], -index))
            selected.append(best)
            remaining.remove(best)
            selected_words += len(sentences[best].text.split())
        return selected

    def _sentence_scores(self, sentences: Sequence[_Sentence]) -> list[float]:
        counts: Counter[str] = Counter()
        tokenized: list[list[str]] = []
        for sentence in sentences:
            tokens = _tokens(sentence.text)
            tokenized.append(tokens)
            counts.update(
                token
                for token in tokens
                if len(token) > 2 and token not in self.stopwords
            )
        if not counts:
            counts.update(token for tokens in tokenized for token in tokens)
        highest = max(counts.values()) if counts else 1
        scores: list[float] = []
        for index, (sentence, tokens) in enumerate(zip(sentences, tokenized, strict=True)):
            frequency = sum(counts[token] for token in tokens if token in counts)
            frequency = frequency / math.sqrt(max(1, len(tokens))) / highest
            position = 0.10 if index == 0 else 0.05 if index == len(sentences) - 1 else 0.0
            scores.append(0.55 * frequency + 0.35 * sentence.parent_score + position)
        return scores

    def _render(self, sentences: Sequence[_Sentence], selected: Sequence[int]) -> str:
        if not selected:
            return ""
        ordered = sorted(selected, key=lambda index: sentences[index].order)
        parts: list[str] = []
        words = 0
        chars = 0
        for index in ordered:
            text = sentences[index].text.strip()
            if not text:
                continue
            sentence_words = len(text.split())
            projected_chars = chars + (1 if parts else 0) + len(text)
            over_chars = projected_chars > self.output_max_chars
            over_words = words + sentence_words > self.target_words
            if over_chars or over_words:
                if not parts:
                    parts.append(
                        trim_summary_text(text, self.target_words, self.output_max_chars)
                    )
                break
            parts.append(text)
            words += sentence_words
            chars = projected_chars
        return " ".join(parts).strip()

    def _empty_summary(
        self,
        candidates: Sequence[_Candidate],
        algorithm: str,
        unit: str,
        started: float,
        fallback: bool,
    ) -> LocalSummary:
        return LocalSummary(
            text="",
            algorithm=algorithm,
            semantic_unit=unit,
            input_chars=0,
            output_words=0,
            chunks_considered=len(candidates),
            chunks_selected=0,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            fallback=fallback,
        )
