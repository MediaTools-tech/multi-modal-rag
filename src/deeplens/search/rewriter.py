"""Query rewriter node for the LangGraph search pipeline."""

from __future__ import annotations

import structlog

from deeplens.core.chat import ChatMessage

logger = structlog.get_logger(__name__)


async def rewrite_query(state: dict) -> dict:
    """Rewrite conversational query to dense keywords.

    Resolves pronouns based on conversation history.
    """
    query = state.get("query", "")
    history = state.get("conversation_history", [])
    chat_engine = state["chat_engine"]

    logger.info("search.node.rewriter.start", query=query)

    # Convert conversation history dicts to ChatMessage objects
    chat_history = []
    for msg in history:
        chat_history.append(ChatMessage(role=msg["role"], content=msg["content"]))

    try:
        rewritten = await chat_engine.rewrite_query(query, chat_history)
        logger.info("search.node.rewriter.success", original=query, rewritten=rewritten)
        return {"rewritten_query": rewritten}
    except Exception as e:
        logger.error("search.node.rewriter.failed", error=str(e))
        # Fallback to original query
        return {"rewritten_query": query}
