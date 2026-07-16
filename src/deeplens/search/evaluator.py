"""Context evaluator node for the LangGraph search pipeline."""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


async def evaluate_context(state: dict) -> dict:
    """Evaluate quality and relevance of retrieved results."""
    results = state.get("results", [])
    retry_count = state.get("retry_count", 0)
    settings = state["settings"]

    threshold = settings.relevance_threshold
    max_retries = settings.max_retries

    logger.info("search.node.evaluator.start", count=len(results), threshold=threshold)

    # Check if we have results above relevance threshold
    has_relevant = False
    max_score = 0.0
    
    if results:
        max_score = max(r.score for r in results)
        if max_score >= threshold:
            has_relevant = True

    if has_relevant:
        logger.info("search.node.evaluator.pass", max_score=max_score)
        return {"context_quality": "pass"}
    
    # Retry logic if not relevant
    if retry_count < max_retries:
        next_retry = retry_count + 1
        logger.warn(
            "search.node.evaluator.fail_retry",
            max_score=max_score,
            current_retry=retry_count,
            next_retry=next_retry,
        )
        return {"context_quality": "fail", "retry_count": next_retry}
    else:
        logger.warn("search.node.evaluator.exhausted", max_score=max_score, retries_attempted=retry_count)
        return {"context_quality": "exhausted"}
