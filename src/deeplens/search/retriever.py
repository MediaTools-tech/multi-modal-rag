"""Vector retriever node for the LangGraph search pipeline."""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


async def retrieve(state: dict) -> dict:
    """Embed query and search vector repository."""
    rewritten_query = state.get("rewritten_query", "")
    embedder = state["embedder"]
    repo = state["repo"]
    settings = state["settings"]
    
    folder_filter = state.get("folder_filter")
    file_type_filter = state.get("file_type_filter")

    logger.info("search.node.retriever.start", query=rewritten_query)

    try:
        # Embed rewritten query
        query_vector = await embedder.embed_text(rewritten_query)
        
        # Search repo
        top_k = settings.search_top_k
        results = await repo.search(
            query_vector=query_vector,
            top_k=top_k,
            folder_filter=folder_filter,
            file_type_filter=file_type_filter,
        )

        logger.info("search.node.retriever.success", count=len(results))
        return {"results": results}

    except Exception as e:
        logger.error("search.node.retriever.failed", error=str(e))
        return {"results": []}
