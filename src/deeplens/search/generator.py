"""Generative response generator node for the LangGraph search pipeline."""

from __future__ import annotations

import structlog

from deeplens.core.chat import ChatMessage
from deeplens.core.models import RecordType

logger = structlog.get_logger(__name__)


async def generate_response(state: dict) -> dict:
    """Generate final response text citing search results."""
    query = state.get("query", "")
    rewritten_query = state.get("rewritten_query", "")
    results = state.get("results", [])
    file_groups = state.get("file_groups", [])
    context_quality = state.get("context_quality", "pass")
    chat_engine = state["chat_engine"]

    logger.info("search.node.generator.start", quality=context_quality)

    if context_quality == "exhausted":
        # Graceful fallback response
        answer = (
            "I searched your files but couldn't find any documents or media that directly "
            f"match your query: '{query}'.\n\n"
            "Suggestions:\n"
            "- Try checking if the target folder is registered and fully indexed.\n"
            "- Simplify your query or try different keywords.\n"
            "- Check the folder tree selection to verify you aren't filtering too narrowly."
        )
        return {"answer": answer}

    # File-level summary (best for "find the document that matches this description").
    file_section = ""
    if file_groups:
        lines = ["Matched files (by document summary):"]
        for g in file_groups:
            lines.append(
                f"- {g.filename}  (score {g.best_score:.2f})\n"
                f"  Path: {g.absolute_path}\n"
                f"  Summary: {g.summary}"
            )
        file_section = "\n".join(lines)

    # Format context passages with citations
    context_blocks = []
    for i, res in enumerate(results):
        rec = res.record
        time_info = ""
        if rec.timestamp_start is not None and rec.timestamp_end is not None:
            time_info = f" [Time: {rec.timestamp_start:.1f}s - {rec.timestamp_end:.1f}s]"

        if rec.record_type == RecordType.SUMMARY.value:
            label = f"Document Summary [{i+1}]"
        else:
            label = f"Source [{i+1}]"

        block = (
            f"{label}: {rec.filename}{time_info}\n"
            f"Path: {rec.absolute_path}\n"
            f"Content: {rec.content}\n"
            "---"
        )
        context_blocks.append(block)

    context_str = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an expert desktop search assistant. Answer the user's query based strictly on the "
        "provided file content passages. Be helpful, concise, and professional.\n\n"
        "Rules:\n"
        "1. Answer based ONLY on the provided context. If the answer cannot be found in the context, say so.\n"
        "2. Cite your sources clearly using [Source N] / [Document Summary N] format when referencing.\n"
        "3. Provide direct path links when discussing specific files.\n"
        "4. For audio/video files, mention the relevant timestamps where applicable.\n"
        "5. When a 'Matched files' section is present, lead with the most relevant file(s).\n"
        "6. Output clean, readable Markdown."
    )

    file_block = f"\n\nMatched files:\n{file_section}" if file_section else ""

    prompt = (
        f"User Query: {query}\n\n"
        f"Search Term: {rewritten_query}\n"
        f"{file_block}\n\n"
        f"Indexed Context Passages:\n{context_str}\n\n"
        "Answer:"
    )

    try:
        response = await chat_engine.generate(prompt, system_prompt=system_prompt)
        return {"answer": response.content}
    except Exception as e:
        logger.error("search.node.generator.failed", error=str(e))
        return {"answer": f"Error: Generative response generation failed: {str(e)}"}
