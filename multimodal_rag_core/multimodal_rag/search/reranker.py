from __future__ import annotations

import logging
import math
from typing import Any

from multimodal_rag.core.models import SearchResult

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, Any] = {}


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        top_n: int = 15,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.top_n = top_n
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        cache_key = f"{self.model_name}::{self.device}"
        if cache_key in _MODEL_CACHE:
            self._model = _MODEL_CACHE[cache_key]
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:
            logger.warning("Reranker unavailable (sentence-transformers missing): %s", exc)
            return None
        try:
            model = CrossEncoder(self.model_name, device=self.device)
            _MODEL_CACHE[cache_key] = model
            self._model = model
            return model
        except Exception as exc:
            logger.warning("Failed to load reranker %s: %s", self.model_name, exc)
            return None

    def rerank(
        self, query: str, candidates: list[SearchResult], top_n: int | None = None
    ) -> list[SearchResult]:
        if not candidates:
            return []
        limit = top_n or self.top_n
        model = self._ensure_model()
        if model is None:
            logger.info("rerank skipped: cross-encoder model unavailable")
            return candidates[:limit]

        pairs = [(query, candidate.record.content or "") for candidate in candidates]
        try:
            raw_scores = model.predict(pairs)
        except Exception as exc:
            logger.warning("Reranker inference failed: %s", exc)
            return candidates[:limit]

        scores: list[float] = []
        for score in raw_scores:
            if isinstance(score, (list, tuple)):
                scores.append(float(score[0]))
            else:
                scores.append(float(score))

        # sentence-transformers CrossEncoder applies sigmoid by default for
        # single-label rerankers, so predict() already returns probabilities in
        # [0, 1]. Only custom/older models return unbounded logits, so we apply
        # sigmoid just in that case. The previous code always sigmoided again
        # AND min-maxed the result set, which forced the top hit to exactly 100%
        # and collapsed every other hit (even ones the answer cited) to 0%.
        if any(value < 0.0 or value > 1.0 for value in scores):
            scores = [1.0 / (1.0 + math.exp(-value)) for value in scores]

        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        logger.info(
            "rerank model=%s probs min=%.4f max=%.4f n=%d",
            self.model_name,
            min(scores) if scores else 0.0,
            max(scores) if scores else 0.0,
            len(scores),
        )
        reranked: list[SearchResult] = []
        for rank, index in enumerate(order[:limit]):
            candidate = candidates[index]
            # Expose the model's calibrated probability as-is (ordering is the
            # real signal; the number is not a percentage of "correctness").
            candidate.score = max(0.0, min(1.0, scores[index]))
            candidate.rank = rank + 1
            reranked.append(candidate)
        return reranked
