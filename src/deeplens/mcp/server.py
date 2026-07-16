"""Model Context Protocol (MCP) server for DeepLens.

Exposes semantic retrieval and file chunk hooks to third-party tools like Cursor
or Claude Desktop. Runs on standard input/output (stdio) transport.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP
import structlog

from deeplens.config import get_settings
from deeplens.core.factory import BackendFactory
from deeplens.search.graph import SearchPipeline

logger = structlog.get_logger(__name__)

# Initialize FastMCP Server
mcp = FastMCP("deeplens")

# Global pipeline instances to avoid re-initializing on each request
_repo: Any = None
_embedder: Any = None
_chat: Any = None
_pipeline: SearchPipeline | None = None


async def get_pipeline() -> SearchPipeline:
    """Retrieve or initialize search pipeline components."""
    global _pipeline, _repo, _embedder, _chat
    if _pipeline is None:
        settings = get_settings()
        factory = BackendFactory(settings)
        
        # Instantiate
        _repo = factory.create_repository()
        _embedder = factory.create_embedding_engine()
        _chat = factory.create_chat_engine()
        
        # Initialize
        await _repo.initialize()
        await _embedder.initialize()
        await _chat.initialize()
        
        _pipeline = SearchPipeline(_repo, _embedder, _chat, settings)
    return _pipeline


@mcp.tool()
async def semantic_search(
    query: str,
    folder_filter: str | None = None,
    file_type_filter: str | None = None,
    top_k: int = 10
) -> str:
    """Perform a conversational semantic query across registered local files.

    Args:
        query: The conversational search query.
        folder_filter: Optional directory path subtree to limit search.
        file_type_filter: Optional file category (document, image, audio, video).
        top_k: Number of search results to evaluate.
    """
    try:
        pipeline = await get_pipeline()
        
        # Limit top_k to safe boundaries
        top_k = max(1, min(50, top_k))
        
        # Temp overrides
        pipeline.settings.search_top_k = top_k
        
        response = await pipeline.search(
            query=query,
            folder_filter=folder_filter,
            file_type_filter=file_type_filter,
        )
        
        # Format response as text for LLM client
        return response.answer

    except Exception as e:
        logger.error("mcp.tool.semantic_search.failed", error=str(e))
        return f"Error executing semantic search: {str(e)}"


@mcp.tool()
async def list_indexed_folders() -> str:
    """List all registered parent directories that have been indexed."""
    try:
        await get_pipeline()
        folders = await _repo.list_indexed_folders()
        return json.dumps(folders, indent=2)
    except Exception as e:
        logger.error("mcp.tool.list_indexed_folders.failed", error=str(e))
        return f"Error listing folders: {str(e)}"


@mcp.tool()
async def get_file_chunks(file_path: str) -> str:
    """Retrieve all raw text chunk contents and timestamps for a specific file.

    Args:
        file_path: The absolute path of the indexed file.
    """
    try:
        await get_pipeline()
        chunks = await _repo.get_file_chunks(file_path)
        
        # Format output
        output_chunks = []
        for c in chunks:
            output_chunks.append({
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
                "content": c.content,
                "timestamp_start": c.timestamp_start,
                "timestamp_end": c.timestamp_end,
            })
            
        return json.dumps(output_chunks, indent=2)

    except Exception as e:
        logger.error("mcp.tool.get_file_chunks.failed", file=file_path, error=str(e))
        return f"Error retrieving file chunks: {str(e)}"


@mcp.tool()
async def get_indexing_status() -> str:
    """Retrieve vector database statistics and count of indexed files."""
    try:
        await get_pipeline()
        stats = await _repo.get_stats()
        return json.dumps(stats, indent=2)
    except Exception as e:
        logger.error("mcp.tool.get_indexing_status.failed", error=str(e))
        return f"Error getting status: {str(e)}"


async def shutdown() -> None:
    """Graceful cleanup on exit."""
    global _repo, _embedder, _chat
    logger.info("mcp.shutdown.start")
    if _repo:
        await _repo.close()
    if _embedder:
        await _embedder.close()
    if _chat:
        await _chat.close()
    logger.info("mcp.shutdown.completed")


def main() -> None:
    """MCP server entry point."""
    structlog.configure(
        processors=[
            structlog.processors.JSONRenderer()
        ],
        logger_factory=structlog.PrintLoggerFactory(sys.stderr)  # stdio logs go to stderr
    )
    
    logger.info("mcp.server.starting")
    try:
        # Run stdio transport
        mcp.run(transport="stdio")
    finally:
        # Run async shutdown
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(shutdown())
            else:
                loop.run_until_complete(shutdown())
        except Exception:
            pass


if __name__ == "__main__":
    main()
