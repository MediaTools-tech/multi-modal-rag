from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod

from multimodal_rag.config import Settings
from multimodal_rag.utils.rate_limit import RateLimiter
from multimodal_rag.utils.token_tracker import TokenTracker

logger = logging.getLogger(__name__)


class EmbeddingEngine(ABC):
    dimension: int

    @abstractmethod
    def embed_texts(self, texts: list[str], *, task_type: str | None = None) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        # Queries must use the query-side task type for asymmetric embedders
        # (e.g. Google text embeddings); documents use RETRIEVAL_DOCUMENT.
        return self.embed_texts([text], task_type="RETRIEVAL_QUERY")[0]


class GoogleEmbeddingEngine(EmbeddingEngine):
    def __init__(self, settings: Settings, tracker: TokenTracker | None = None) -> None:
        self.settings = settings
        role = settings.embedding_role
        self.dimension = settings.EMBEDDING_DIMENSION
        self.model = role.model
        self.task_type = settings.EMBEDDING_TASK_TYPE
        self.batch_size = max(1, settings.EMBEDDING_BATCH_SIZE)
        self.api_key = role.api_key
        self.base_url = role.base_url
        self.tracker = tracker or TokenTracker()
        self.limiter = RateLimiter(settings.EMBEDDING_MAX_RPM)
        self._client = None
        # The genai client is not safe to create concurrently nor to reuse after
        # it has been closed (e.g. by a racing thread). Guard creation + calls and
        # recreate on failure.
        self._guard = threading.RLock()

    def _ensure(self):
        with self._guard:
            if self._client is None:
                from google import genai

                kwargs = {"api_key": self.api_key}
                if self.base_url:
                    from google.genai import types

                    kwargs["http_options"] = types.HttpOptions(base_url=self.base_url)
                self._client = genai.Client(**kwargs)
            return self._client

    def _reset_client(self) -> None:
        with self._guard:
            self._client = None

    def _embed_batch(self, batch: list[str], effective_task: str):
        from google.genai import types

        for attempt in range(2):
            try:
                with self._guard:
                    return self._ensure().models.embed_content(
                        model=self.model,
                        contents=batch,
                        config=types.EmbedContentConfig(
                            task_type=effective_task,
                            output_dimensionality=self.dimension,
                        ),
                    )
            except Exception as exc:  # noqa: BLE001
                if attempt == 0:
                    logger.warning("Embedding call failed (%s); recreating client", exc)
                    self._reset_client()
                else:
                    raise

    def embed_texts(
        self, texts: list[str], *, task_type: str | None = None
    ) -> list[list[float]]:
        effective_task = task_type or self.task_type
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            self.limiter.wait()
            response = self._embed_batch(batch, effective_task)
            vectors.extend([list(embedding.values) for embedding in response.embeddings])
            # The embed API returns no usage metadata: record a ~4 chars/token
            # estimate so session totals include the (usually largest) consumer.
            estimated = sum(max(1, len(text) // 4) for text in batch)
            cost, status = self.settings.cost_for(
                self.model, "GEMINI", estimated, 0, purpose="embedding"
            )
            self.tracker.record(
                "GEMINI", estimated, 0, cost, model=self.model,
                unrated=(status == "missing"),
            )
        return vectors


class LocalEmbeddingEngine(EmbeddingEngine):
    def __init__(self, settings: Settings, tracker: TokenTracker | None = None) -> None:
        self.settings = settings
        self.dimension = settings.EMBEDDING_DIMENSION
        self.model_name = settings.EMBEDDING_MODEL
        self.batch_size = max(1, settings.EMBEDDING_BATCH_SIZE)
        self.tracker = tracker or TokenTracker()
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.settings.device)
        return self._model

    def embed_texts(
        self, texts: list[str], *, task_type: str | None = None
    ) -> list[list[float]]:
        # Local sentence-transformers models are symmetric; task_type is ignored.
        model = self._ensure()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.settings.EMBEDDING_NORMALIZE,
            convert_to_numpy=True,
        )
        # Offline: no API spend. Record estimated tokens so the model shows up
        # as "local" on the usage panel instead of being invisible.
        estimated = sum(max(1, len(text) // 4) for text in texts)
        self.tracker.record("LOCAL", estimated, 0, 0.0, model=self.model_name, local=True)
        return [list(map(float, vector)) for vector in embeddings]


def get_embedding_engine(settings: Settings, tracker: TokenTracker | None = None) -> EmbeddingEngine:
    role = settings.embedding_role
    if role.provider == "GOOGLE":
        if not role.api_key:
            raise RuntimeError(
                "Embeddings need an API key: set EMBEDDINGS_API_KEY (or GOOGLE_API_KEY) in .env"
            )
        return GoogleEmbeddingEngine(settings, tracker)
    if role.provider == "LOCAL":
        return LocalEmbeddingEngine(settings, tracker)
    raise NotImplementedError(f"Embedding provider {role.provider!r} not supported yet")
