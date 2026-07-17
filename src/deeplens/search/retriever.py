"""Vector retriever node for the LangGraph search pipeline."""

from __future__ import annotations

import structlog

from deeplens.core.models import RecordType
from deeplens.search.hybrid import hybrid_search, summary_search

logger = structlog.get_logger(__name__)


async def retrieve(state: dict) -> dict:
    """Embed query and search the vector repository.

    Honors ``state["search_mode"]`` (defaults to ``settings.search_mode``):
      - ``"chunk"``   : passage-level ANN only (legacy behaviour).
      - ``"summary"`` : document-level summary matching (find the file).
      - ``"hybrid"``  : fused summary + chunk + corpus-wide lexical retrieval (recommended).
    """
    rewritten_query = state.get("rewritten_query", "")
    embedder = state["embedder"]
    repo = state["repo"]
    settings = state["settings"]

    folder_filter = state.get("folder_filter")
    file_type_filter = state.get("file_type_filter")
    search_mode = state.get("search_mode") or settings.search_mode

    logger.info("search.node.retriever.start", query=rewritten_query, mode=search_mode)

    try:
        query_vector = await embedder.embed_text(rewritten_query)
        top_k = settings.search_top_k

        if search_mode == "chunk":
            results = await repo.search(
                query_vector=query_vector,
                top_k=top_k,
                folder_filter=folder_filter,
                file_type_filter=file_type_filter,
                record_types=[RecordType.CHUNK.value],
            )
            return {"results": results, "file_groups": []}

        if search_mode == "summary":
            results, file_groups = await summary_search(
                repo=repo,
                query_vector=query_vector,
                settings=settings,
                top_k=top_k,
                folder_filter=folder_filter,
                file_type_filter=file_type_filter,
                query=rewritten_query,
            )
            return {"results": results, "file_groups": file_groups}

        # Default: hybrid
        results, file_groups = await hybrid_search(
            repo=repo,
            query=rewritten_query,
            query_vector=query_vector,
            settings=settings,
            top_k=top_k,
            folder_filter=folder_filter,
            file_type_filter=file_type_filter,
        )
        return {"results": results, "file_groups": file_groups}

    except Exception as e:
        logger.error("search.node.retriever.failed", error=str(e))
        return {"results": [], "file_groups": []}
