"""LangGraph RAG pipeline orchestration module."""

from __future__ import annotations

import time
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
import structlog

from deeplens.config import Settings
from deeplens.core.chat import ChatEngine
from deeplens.core.embedding import EmbeddingEngine
from deeplens.core.models import SearchResponse, SearchResult
from deeplens.core.repository import DocumentRepository

# Import nodes
from deeplens.search.rewriter import rewrite_query
from deeplens.search.retriever import retrieve
from deeplens.search.evaluator import evaluate_context
from deeplens.search.generator import generate_response

logger = structlog.get_logger(__name__)


class SearchState(TypedDict, total=False):
    """LangGraph State representation for the search pipeline."""

    query: str
    rewritten_query: str
    results: list[SearchResult]
    answer: str
    context_quality: Literal["pass", "fail", "exhausted"]
    retry_count: int
    folder_filter: str | None
    file_type_filter: str | None
    conversation_history: list[dict[str, str]]
    total_time_ms: float
    
    # Executable components passed through state
    settings: Settings
    repo: DocumentRepository
    embedder: EmbeddingEngine
    chat_engine: ChatEngine


def _route_after_evaluation(state: SearchState) -> str:
    """Determine the next step based on context evaluation."""
    quality = state.get("context_quality", "pass")
    if quality == "pass":
        return "generator"
    elif quality == "fail":
        return "rewriter"
    else:  # exhausted
        return "generator"


class SearchPipeline:
    """Orchestrates RAG retrieval using a compiled LangGraph state machine."""

    def __init__(
        self,
        repository: DocumentRepository,
        embedding_engine: EmbeddingEngine,
        chat_engine: ChatEngine,
        settings: Settings
    ) -> None:
        self.repository = repository
        self.embedding_engine = embedding_engine
        self.chat_engine = chat_engine
        self.settings = settings
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Define and compile the LangGraph workflow."""
        workflow = StateGraph(SearchState)

        # Register nodes
        workflow.add_node("rewriter", rewrite_query)
        workflow.add_node("retriever", retrieve)
        workflow.add_node("evaluator", evaluate_context)
        workflow.add_node("generator", generate_response)

        # Wire edges
        workflow.add_edge(START, "rewriter")
        workflow.add_edge("rewriter", "retriever")
        workflow.add_edge("retriever", "evaluator")
        
        # Conditional branching
        workflow.add_conditional_edges(
            "evaluator",
            _route_after_evaluation,
            {
                "generator": "generator",
                "rewriter": "rewriter"
            }
        )
        
        workflow.add_edge("generator", END)

        # Compile
        return workflow.compile()

    async def search(
        self,
        query: str,
        folder_filter: str | None = None,
        file_type_filter: str | None = None,
        history: list[dict[str, str]] | None = None
    ) -> SearchResponse:
        """Execute the LangGraph search pipeline."""
        start_time = time.perf_counter()

        initial_state: SearchState = {
            "query": query,
            "rewritten_query": query,
            "results": [],
            "answer": "",
            "context_quality": "pass",
            "retry_count": 0,
            "folder_filter": folder_filter,
            "file_type_filter": file_type_filter,
            "conversation_history": history or [],
            "settings": self.settings,
            "repo": self.repository,
            "embedder": self.embedding_engine,
            "chat_engine": self.chat_engine,
        }

        logger.info("search.pipeline.start", query=query)
        
        try:
            # Execute graph synchronously or asynchronously (compiled graph.ainvoke)
            final_state = await self._graph.ainvoke(initial_state)
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            
            logger.info("search.pipeline.completed", time_ms=elapsed_ms)
            
            return SearchResponse(
                query=query,
                rewritten_query=final_state.get("rewritten_query", query),
                results=final_state.get("results", []),
                answer=final_state.get("answer", ""),
                retry_count=final_state.get("retry_count", 0),
                total_time_ms=elapsed_ms
            )

        except Exception as e:
            logger.error("search.pipeline.failed", error=str(e))
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SearchResponse(
                query=query,
                rewritten_query=query,
                results=[],
                answer=f"Error running search pipeline: {str(e)}",
                total_time_ms=elapsed_ms
            )
