"""Ollama implementation of the ChatEngine.

Runs local chat models (e.g. Llama-3.2-3B or Phi-4) entirely offline via Ollama.
"""

from __future__ import annotations

from typing import Any

import ollama
import structlog

from deeplens.config import Settings
from deeplens.core.chat import ChatEngine, ChatMessage, ChatResponse

logger = structlog.get_logger(__name__)


class OllamaChatEngine(ChatEngine):
    """Ollama local chat engine."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_name = settings.ollama_chat_model
        self._client: ollama.AsyncClient | None = None

    async def initialize(self) -> None:
        """Initialize AsyncClient and verify Ollama is running."""
        logger.info("ollama_chat.initialize.start", host=self.settings.ollama_host, model=self.model_name)
        self._client = ollama.AsyncClient(host=self.settings.ollama_host)
        
        try:
            # Check connection
            await self._client.list()
            logger.info("ollama_chat.initialize.success")
        except Exception as e:
            logger.error("ollama_chat.initialize.failed", error=str(e))
            raise RuntimeError(
                f"Could not connect to Ollama server at {self.settings.ollama_host}.\n"
                "Please verify Ollama is installed and running (`ollama serve`)."
            ) from e

    async def close(self) -> None:
        """Release client reference."""
        self._client = None
        logger.info("ollama_chat.close")

    async def generate(self, prompt: str, system_prompt: str | None = None) -> ChatResponse:
        """Generate single completion."""
        messages = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))
        return await self.chat(messages)

    async def chat(
        self, messages: list[ChatMessage], system_prompt: str | None = None
    ) -> ChatResponse:
        """Multi-turn conversation."""
        if self._client is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        # Map to Ollama format
        ollama_messages = []
        if system_prompt:
            ollama_messages.append({"role": "system", "content": system_prompt})
            
        for msg in messages:
            ollama_messages.append({"role": msg.role, "content": msg.content})

        try:
            response = await self._client.chat(
                model=self.model_name,
                messages=ollama_messages,
            )
            
            content = response.message.content or ""
            
            # Map usage stats if available
            usage = {
                "prompt_tokens": response.get("prompt_eval_count", 0),
                "completion_tokens": response.get("eval_count", 0),
            }

            return ChatResponse(
                content=content,
                model=self.model_name,
                usage=usage
            )

        except Exception as e:
            logger.error("ollama_chat.chat.failed", error=str(e))
            return ChatResponse(
                content="Error: Local LLM generation failed. Verify Ollama is running and model is loaded.",
                model=self.model_name
            )
