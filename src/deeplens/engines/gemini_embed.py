"""Gemini API implementation of the EmbeddingEngine.

Leverages Google Gemini cloud embedding model (e.g. text-embedding-004) for
high-quality, multilingual 3072-dimensional embeddings.
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from PIL import Image
import structlog

from deeplens.config import Settings
from deeplens.core.embedding import EmbeddingEngine

logger = structlog.get_logger(__name__)


class GeminiEmbeddingEngine(EmbeddingEngine):
    """Google Gemini embedding engine."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_name = settings.gemini_embedding_model
        self._client: genai.Client | None = None

    @property
    def dimension(self) -> int:
        """Gemini Embedding 004 returns 3072 dimensions."""
        return 3072

    async def initialize(self) -> None:
        """Initialize Google GenAI client."""
        logger.info("gemini_embed.initialize.start", model=self.model_name)
        
        # Will raise ValueError if API key is missing
        api_key = self.settings.get_gemini_api_key()
        
        # Initialize client
        self._client = genai.Client(api_key=api_key)
        logger.info("gemini_embed.initialize.success")

    async def close(self) -> None:
        """Release client reference."""
        self._client = None
        logger.info("gemini_embed.close")

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single string."""
        vectors = await self.embed_texts([text])
        return vectors[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch embed texts with exponential backoff on rate limits."""
        if not texts:
            return []

        if self._client is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        # Gemini supports batch embedding
        max_retries = 3
        backoff = 1.0

        for attempt in range(max_retries):
            try:
                # Call embedding API
                response = self._client.models.embed_content(
                    model=self.model_name,
                    contents=texts,
                )
                
                # Extract vectors
                vectors = []
                for embedding in response.embeddings:
                    vectors.append([float(x) for x in embedding.values])
                return vectors

            except Exception as e:
                # Check for rate limit or network errors
                logger.warn("gemini_embed.embed_texts.failed", attempt=attempt + 1, error=str(e))
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(backoff)
                backoff *= 2.0

        return []

    async def embed_image(self, image: Image.Image) -> list[float]:
        """Embed PIL Image using Gemini.

        Note: Gemini Embedding models (like text-embedding-004) are text-only.
        To handle multi-modal image indexing in cloud mode:
        1. We use Gemini 2.5 Flash to generate a detailed textual description of the image.
        2. We embed that description using text-embedding-004.
        This provides high-quality semantic search over images in cloud mode.
        """
        if self._client is None:
            raise RuntimeError("Engine not initialized.")

        # Save image to bytes buffer
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        img_bytes = img_byte_arr.getvalue()

        # Step 1: Generate description with Gemini 2.5 Flash
        # Use AI Studio client
        logger.info("gemini_embed.embed_image.describe_image")
        
        # Describe image prompt
        prompt = (
            "Analyze this image in detail. Describe all objects, actions, text, colors, "
            "mood, setting, and any semantic details. Output a clean, dense textual "
            "description that would be ideal for semantic search indexing. Keep it under 200 words."
        )

        description = ""
        try:
            # We call 2.5-flash
            response = self._client.models.generate_content(
                model=self.settings.gemini_chat_model,
                contents=[
                    types.Part.from_bytes(
                        data=img_bytes,
                        mime_type="image/jpeg",
                    ),
                    prompt
                ]
            )
            description = response.text or ""
            logger.info("gemini_embed.embed_image.described", len_desc=len(description))
        except Exception as e:
            logger.error("gemini_embed.embed_image.description_failed", error=str(e))
            # Fallback description
            description = "An image containing visual elements."

        # Step 2: Embed the description
        return await self.embed_text(description)
