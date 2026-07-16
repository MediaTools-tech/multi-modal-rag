"""Gemini API implementation of the ChatEngine.

Runs cloud-accelerated chat models (e.g. Gemini 2.5 Flash) via the official google-genai SDK.
"""

from __future__ import annotations

from google import genai
from google.genai import types
import structlog

from deeplens.config import Settings
from deeplens.core.chat import ChatEngine, ChatMessage, ChatResponse

logger = structlog.get_logger(__name__)


class GeminiChatEngine(ChatEngine):
    """Google Gemini chat engine."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_name = settings.gemini_chat_model
        self._client: genai.Client | None = None

    async def initialize(self) -> None:
        """Initialize Google GenAI client."""
        logger.info("gemini_chat.initialize.start", model=self.model_name)
        
        # Will raise ValueError if API key is missing
        api_key = self.settings.get_gemini_api_key()
        
        self._client = genai.Client(api_key=api_key)
        logger.info("gemini_chat.initialize.success")

    async def close(self) -> None:
        """Release client reference."""
        self._client = None
        logger.info("gemini_chat.close")

    async def generate(self, prompt: str, system_prompt: str | None = None) -> ChatResponse:
        """Generate single completion."""
        if self._client is None:
            raise RuntimeError("Engine not initialized.")

        config = types.GenerateContentConfig()
        if system_prompt:
            config.system_instruction = system_prompt

        try:
            # We run in default thread executor since google-genai is sync
            import asyncio
            
            def _call() -> Any:
                return self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )

            response = await asyncio.to_thread(_call)
            content = response.text or ""
            
            usage = {}
            if response.usage_metadata:
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                    "completion_tokens": response.usage_metadata.candidates_token_count or 0,
                }

            return ChatResponse(
                content=content,
                model=self.model_name,
                usage=usage,
            )
        except Exception as e:
            logger.error("gemini_chat.generate.failed", error=str(e))
            return ChatResponse(
                content=f"Error: Gemini API call failed: {str(e)}",
                model=self.model_name,
            )

    async def chat(
        self, messages: list[ChatMessage], system_prompt: str | None = None
    ) -> ChatResponse:
        """Multi-turn conversation."""
        if self._client is None:
            raise RuntimeError("Engine not initialized.")

        config = types.GenerateContentConfig()
        if system_prompt:
            config.system_instruction = system_prompt

        # Convert ChatMessage list to Gemini Contents
        contents = []
        for msg in messages:
            # Map role: "user" -> "user", "assistant" -> "model", "system" -> config.system_instruction
            gemini_role = "model" if msg.role == "assistant" else "user"
            contents.append(
                types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=msg.content)]
                )
            )

        try:
            import asyncio
            
            def _call() -> Any:
                return self._client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )

            response = await asyncio.to_thread(_call)
            content = response.text or ""

            usage = {}
            if response.usage_metadata:
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                    "completion_tokens": response.usage_metadata.candidates_token_count or 0,
                }

            return ChatResponse(
                content=content,
                model=self.model_name,
                usage=usage,
            )
        except Exception as e:
            logger.error("gemini_chat.chat.failed", error=str(e))
            return ChatResponse(
                content=f"Error: Gemini chat API call failed: {str(e)}",
                model=self.model_name,
            )
