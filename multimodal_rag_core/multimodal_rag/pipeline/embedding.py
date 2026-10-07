from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod

from multimodal_rag.config import (
    EMBEDDINGGEMMA_MODEL_ID,
    EMBEDDINGGEMMA_VARIANTS,
    Settings,
    is_embeddinggemma_spec,
    parse_embeddinggemma_spec,
)
from multimodal_rag.utils.rate_limit import RateLimiter
from multimodal_rag.utils.token_tracker import TokenTracker

logger = logging.getLogger(__name__)

#: sentence-transformers version that first supports EmbeddingGemma 2.
GEMMA2_MIN_ST_VERSION = "6.1.0"


def _type_name(exc: BaseException) -> str:
    return type(exc).__name__


def _caused_by_unsupported_arch(exc: BaseException) -> bool:
    """True when the weights fetched fine but the installed `transformers`
    predates the checkpoint's architecture (e.g. `embedding_gemma2`).

    Walks the cause chain because sentence-transformers wraps the original
    KeyError/ValueError from transformers' CONFIG_MAPPING.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = str(current).lower()
        if "does not recognize this architecture" in text or "embedding_gemma2" in text:
            return True
        current = current.__cause__ or current.__context__
    return False


def _gemma_arch_hint(model_name: str, variant: str, exc: BaseException) -> str:
    info = EMBEDDINGGEMMA_VARIANTS.get(variant, {})
    label = info.get("label", variant)
    return (
        f"EMBEDDING_MODEL_UNSUPPORTED: {model_name!r} "
        f"(EmbeddingGemma 2 — {label}) downloaded fine, but the installed "
        f"'transformers' library predates its architecture "
        f"({_type_name(exc)}). This is NOT a download problem — no re-download helps.\n"
        f"Fix (venv active):\n"
        f"  pip install -U transformers \"sentence-transformers>={GEMMA2_MIN_ST_VERSION}\"\n"
        f"then restart the app. If it still fails (launch-day architecture with "
        f"no release support yet):\n"
        f"  pip install git+https://github.com/huggingface/transformers.git\n"
        f"Note: files that already burned all their ingest retries while the loader "
        f"was broken stay 'failed' — requeue with `mrag-ingest retry all`, then restart."
    )


def _gemma_download_hint(model_name: str, variant: str) -> str:
    info = EMBEDDINGGEMMA_VARIANTS.get(variant, {})
    label = info.get("label", variant)
    params = info.get("params", "")
    return (
        f"EMBEDDING_MODEL_NOT_CACHED: local embedding model {model_name!r} "
        f"(EmbeddingGemma 2 — {label}, {params}) is not downloaded on this machine.\n"
        f"The model is NOT bundled with the app / Docker image / exe (hundreds of MB) — "
        f"it downloads once from HuggingFace on first use and is then cached.\n"
        f"To get it:\n"
        f"  1. Accept the license at https://huggingface.co/{EMBEDDINGGEMMA_MODEL_ID}\n"
        f"  2. pip install -U \"sentence-transformers>={GEMMA2_MIN_ST_VERSION}\" transformers huggingface_hub\n"
        f"  3. huggingface-cli login   (needed: gated repo)\n"
        f"  4. Pre-download (or just restart the app with internet — it auto-downloads):\n"
        f"     huggingface-cli download {EMBEDDINGGEMMA_MODEL_ID}\n"
        f"Cache: $HF_HOME (Docker: /data/hf_cache, persisted on the /data volume) "
        f"or ~/.cache/huggingface.\n"
        f"Tip: pick a smaller variant (text-only 270M) or a 256/128 MRL dimension "
        f"if RAM/disk is tight; queries and docs must share one dimension."
    )


def _generic_download_hint(model_name: str) -> str:
    return (
        f"EMBEDDING_MODEL_NOT_CACHED: local embedding model {model_name!r} "
        f"could not be loaded (not cached and download failed).\n"
        f"The model is NOT bundled with the app — it downloads once from "
        f"HuggingFace on first use.\n"
        f"Check the model id spelling and internet, then pre-download:\n"
        f"  pip install -U sentence-transformers huggingface_hub\n"
        f"  huggingface-cli download {model_name}\n"
        f"(Gated repos additionally need: accept the license on its HF page + "
        f"huggingface-cli login.)"
    )


class EmbeddingEngine(ABC):
    dimension: int

    @abstractmethod
    def embed_texts(self, texts: list[str], *, task_type: str | None = None) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        # Queries must use the query-side task type for asymmetric embedders
        # (e.g. Google text embeddings); documents use RETRIEVAL_DOCUMENT.
        return self.embed_texts([text], task_type="RETRIEVAL_QUERY")[0]

    def warmup(self) -> None:
        """Pre-load the model so the first real query is fast.

        Default: no-op (API engines have nothing to warm). Local engines
        override to load the checkpoint and run one throwaway inference so
        tokenizer/kernels/threadpools are ready off the request path.
        """


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
        self._lock = threading.Lock()
        if is_embeddinggemma_spec(self.model_name):
            # Fail fast at backend init (before any download): Gemma 2 needs
            # sentence-transformers>=6.1.0. The weights themselves stay lazy
            # (first query/ingest downloads them); that failure surfaces with
            # the EMBEDDING_MODEL_NOT_CACHED download hint.
            try:
                from importlib.metadata import version
                from packaging.version import Version
            except Exception:
                version = None  # type: ignore[assignment]
                Version = None  # type: ignore[assignment]
            if version is not None and Version is not None:
                try:
                    installed = version("sentence-transformers")
                    if Version(installed) < Version(GEMMA2_MIN_ST_VERSION):
                        raise RuntimeError(
                            f"EMBEDDING_MODEL_NOT_CACHED: {self.model_name!r} needs "
                            f"sentence-transformers>={GEMMA2_MIN_ST_VERSION} "
                            f"(installed {installed}). Upgrade: "
                            f"pip install -U \"sentence-transformers>={GEMMA2_MIN_ST_VERSION}\""
                        )
                except Exception as exc:
                    if "EMBEDDING_MODEL_NOT_CACHED" in str(exc):
                        raise
                    logger.debug("st version check skipped: %s", exc)

    def _load_kwargs(self) -> dict:
        """SentenceTransformer kwargs for the configured EMBEDDING_MODEL.

        EmbeddingGemma 2 shares one checkpoint; the :suffix picks encoders
        (same 768d vector space) and EMBEDDING_DIMENSION maps to MRL
        truncate_dim. Legacy models (MiniLM/bge) get no extra kwargs.
        """
        if is_embeddinggemma_spec(self.model_name):
            _, variant, config_kwargs = parse_embeddinggemma_spec(self.model_name)
            kwargs: dict = {"config_kwargs": config_kwargs}
            if self.dimension != 768:
                # MRL truncation (128/256/512); validated by Settings.
                kwargs["truncate_dim"] = self.dimension
            return kwargs
        return {}

    def _base_model_id(self) -> str:
        if is_embeddinggemma_spec(self.model_name):
            base, _, _ = parse_embeddinggemma_spec(self.model_name)
            return base
        return self.model_name

    def _ensure(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:
                raise RuntimeError(
                    "Local embeddings need sentence-transformers: "
                    f"pip install -U \"sentence-transformers>={GEMMA2_MIN_ST_VERSION}\" "
                    f"({_type_name(exc)}: {exc})"
                ) from exc
            load_kwargs = self._load_kwargs()
            try:
                self._model = SentenceTransformer(
                    self._base_model_id(), device=self.settings.device, **load_kwargs
                )
            except Exception as exc:
                if is_embeddinggemma_spec(self.model_name):
                    _, variant, _ = parse_embeddinggemma_spec(self.model_name)
                    if _caused_by_unsupported_arch(exc):
                        raise RuntimeError(
                            _gemma_arch_hint(self.model_name, variant, exc)
                        ) from exc
                    raise RuntimeError(
                        f"{_gemma_download_hint(self.model_name, variant)}\n"
                        f"Underlying error: {_type_name(exc)}: {exc}"
                    ) from exc
                raise RuntimeError(
                    f"{_generic_download_hint(self.model_name)}\n"
                    f"Underlying error: {_type_name(exc)}: {exc}"
                ) from exc
        return self._model

    def warmup(self) -> None:
        # Run one throwaway query through the same code path the first real
        # search will use (bge query prefix / gemma task config handled by
        # embed_query-level settings), so checkpoint load + tokenizer init +
        # threadpool/kernel warmup happen off the request path.
        model = self._ensure()
        model.encode(
            ["warmup"],
            batch_size=1,
            normalize_embeddings=self.settings.EMBEDDING_NORMALIZE,
            convert_to_numpy=True,
        )

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
