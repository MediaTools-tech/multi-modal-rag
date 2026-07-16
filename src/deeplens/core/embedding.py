"""Abstract base class for embedding engines.

Embedding engines convert text, images, and other media into dense vector
representations for storage and similarity search.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image


class EmbeddingEngine(ABC):
    """Abstract embedding engine.

    Implementations must handle text and image embeddings at minimum.
    The vector dimension is determined by the concrete engine.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the output embedding dimension."""
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """Load model weights and prepare for inference."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release model resources."""
        ...

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Embed a text string into a dense vector.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        ...

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings (batch operation).

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors.
        """
        ...

    @abstractmethod
    async def embed_image(self, image: Image.Image) -> list[float]:
        """Embed an image into a dense vector.

        Args:
            image: A PIL Image object.

        Returns:
            A list of floats representing the embedding vector.
        """
        ...

    async def embed_image_from_path(self, path: Path) -> list[float]:
        """Embed an image from a file path.

        Args:
            path: Path to the image file.

        Returns:
            A list of floats representing the embedding vector.
        """
        image = Image.open(path).convert("RGB")
        return await self.embed_image(image)
