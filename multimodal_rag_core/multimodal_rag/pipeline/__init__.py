from __future__ import annotations

from .chunking import Chunk, chunk_text, text_to_chunks

__all__ = [
    "Chunk",
    "PipelineContext",
    "build_context",
    "build_handlers",
    "chunk_text",
    "text_to_chunks",
]


def __getattr__(name: str):
    if name in ("PipelineContext", "build_context"):
        from .context import PipelineContext, build_context

        return {"PipelineContext": PipelineContext, "build_context": build_context}[name]
    if name == "build_handlers":
        from .registry import build_handlers

        return build_handlers
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
