"""Jina-CLIP-v2 implementation of the EmbeddingEngine.

Local-only, CPU-efficient multi-modal model returning 1024-dimensional embeddings.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from PIL import Image
import structlog

from deeplens.config import Settings
from deeplens.core.embedding import EmbeddingEngine

logger = structlog.get_logger(__name__)


class JinaClipEngine(EmbeddingEngine):
    """Jina-CLIP-v2 local embedding engine."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_name = settings.local_embedding_model
        self._model: Any = None
        self._processor: Any = None
        self._device = "cpu"

    @property
    def dimension(self) -> int:
        """Jina-CLIP-v2 returns 1024-dimensional embeddings."""
        return 1024

    async def initialize(self) -> None:
        """Load weights using HuggingFace transformers in a background thread."""
        logger.info("jina_clip.initialize.start", model=self.model_name)

        def _load() -> None:
            import torch
            from transformers import AutoModel, AutoProcessor

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("jina_clip.initialize.device", device=self._device)

            # Jina-CLIP-v2 requires trust_remote_code=True. Pass torch_dtype as a
            # *string*: the model's config does `hasattr(torch, torch_dtype)`,
            # which crashes with "attribute name must be string, not
            # 'torch.dtype'" if a torch.dtype object is passed.
            self._model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype="float32" if self._device == "cpu" else "float16"
            ).to(self._device)
            self._model.eval()

            # Processor handles image preprocessing & text tokenization
            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

        await asyncio.to_thread(_load)
        logger.info("jina_clip.initialize.success")

    async def close(self) -> None:
        """Unload model from RAM/VRAM."""
        self._model = None
        self._processor = None
        logger.info("jina_clip.close")

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single string."""
        vectors = await self.embed_texts([text])
        return vectors[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch embed texts."""
        if not texts:
            return []

        if self._model is None or self._processor is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        def _encode() -> list[list[float]]:
            import torch

            # Tokenize using processor
            inputs = self._processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                # model.get_text_features or direct call depending on transformers integration
                # Jina CLIP v2 exposes encode_text function natively
                embeddings = self._model.encode_text(texts)
                
                # If embeddings is a PyTorch tensor, normalize it and move to CPU
                if hasattr(embeddings, "cpu"):
                    # Normalize to unit length for cosine similarity
                    norm = torch.nn.functional.normalize(torch.tensor(embeddings), p=2, dim=-1)
                    return norm.cpu().tolist()
                else:
                    # If it returns a numpy array, normalize manually
                    import numpy as np
                    embeddings = np.array(embeddings)
                    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
                    normalized = embeddings / (norms + 1e-9)
                    return normalized.tolist()

        return await asyncio.to_thread(_encode)

    async def embed_image(self, image: Image.Image) -> list[float]:
        """Embed PIL Image."""
        if self._model is None or self._processor is None:
            raise RuntimeError("Engine not initialized.")

        def _encode() -> list[float]:
            import torch

            # Jina CLIP v2 supports encode_image method natively or via processor
            with torch.no_grad():
                embeddings = self._model.encode_image(image)
                
                if hasattr(embeddings, "cpu"):
                    norm = torch.nn.functional.normalize(torch.tensor(embeddings), p=2, dim=-1)
                    return norm.cpu().tolist()[0]
                else:
                    import numpy as np
                    embeddings = np.array(embeddings)
                    norm = embeddings / (np.linalg.norm(embeddings) + 1e-9)
                    return norm.flatten().tolist()

        return await asyncio.to_thread(_encode)
