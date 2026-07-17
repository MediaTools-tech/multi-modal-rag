"""Optional cross-encoder re-ranker for the fused candidate set.

Design goals (per deployment class):
  * CPU-only  -> disabled by default (``enable_reranker=False``); the RRF fusion
    already re-ranks, so these users pay zero cost and latency.
  * GPU/cloud  -> enable and (optionally) raise ``rerank_top_n`` for more recall.

When enabled, only the top ``rerank_top_n`` fused candidates are scored — a tiny
model (``ms-marco-MiniLM-L-6-v2``, ~80MB) handles ~15 pairs in well under 100ms
even on CPU, so latency stays bounded regardless of corpus size.

The model is imported lazily and cached per model name so it is loaded at most
once per process.
"""

from __future__ import annotations

from typing import Any

import asyncio
import structlog

from deeplens.config import Settings
from deeplens.core.models import SearchResult

logger = structlog.get_logger(__name__)

# Process-wide cache: model name -> loaded CrossEncoder instance.
_MODEL_CACHE: dict[str, Any] = {}


class CrossEncoderReranker:
    """Re-ranks ``(query, document)`` pairs with a cross-encoder."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_name = settings.reranker_model
        self._model: Any = None

    def _ensure_model(self) -> Any:
        """Load (or fetch cached) the cross-encoder; return None if unavailable."""
        if self._model is not None:
            return self._model
        if self.model_name in _MODEL_CACHE:
            self._model = _MODEL_CACHE[self.model_name]
            return self._model
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception as e:
            logger.warn("reranker.import_failed", error=str(e))
            return None
        try:
            model = CrossEncoder(self.model_name)
            _MODEL_CACHE[self.model_name] = model
            self._model = model
            return model
        except Exception as e:
            logger.warn("reranker.load_failed", model=self.model_name, error=str(e))
            return None

    async def rerank(
        self, query: str, candidates: list[SearchResult], top_n: int
    ) -> list[SearchResult]:
        """Re-rank ``candidates`` by cross-encoder relevance; return top ``top_n``.

        Falls back to the unchanged candidate list (truncated to ``top_n``) if the
        model cannot be loaded or inference fails, so the pipeline degrades safely.
        """
        if not candidates:
            return []
        model = self._ensure_model()
        if model is None:
            return candidates[:top_n]

        pairs = [(query, c.record.content or "") for c in candidates]
        try:
            scores = await asyncio.to_thread(model.predict, pairs)
        except Exception as e:
            logger.warn("reranker.predict_failed", error=str(e))
            return candidates[:top_n]

        flat: list[float] = []
        for s in scores:
            if isinstance(s, (list, tuple)):
                flat.append(float(s[0]))
            else:
                flat.append(float(s))

        order = sorted(range(len(candidates)), key=lambda i: flat[i], reverse=True)
        reranked: list[SearchResult] = []
        for rank, i in enumerate(order[:top_n]):
            c = candidates[i]
            c.score = float(flat[i])
            c.rank = rank + 1
            reranked.append(c)
        return reranked
