from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import threading
import time
import urllib.request
from pathlib import Path

from multimodal_rag.config import Settings
from multimodal_rag.utils.rate_limit import RateLimiter
from multimodal_rag.utils.token_tracker import TokenTracker

logger = logging.getLogger(__name__)

FRAME_PROMPT = (
    "You are captioning consecutive keyframes of a video for a search index. "
    "Return ONLY a JSON array of exactly {n} objects, in the same order as the "
    'images. Each object must have the keys "timestamp", "core_actions", '
    '"on_screen_text", "objects", and "setting".\n'
    "IMPORTANT: preserve any on-screen text verbatim in on_screen_text.\n"
    "Timestamps: {timestamps}"
)

SUMMARY_PROMPT = (
    "Write one concise paragraph (max 120 words) summarizing the content of the "
    "document titled {filename!r}. Capture the main subject, entities, and purpose "
    "so the file can later be found from a natural-language description. "
    "Do not add commentary.\n\nDOCUMENT:\n{text}"
)

ANSWER_PROMPT = (
    "You are a helpful assistant answering strictly from the provided context. "
    "Cite sources as [1], [2], ... matching the context numbers. If the context "
    "does not contain the answer, say you don't know.\n\n"
    "QUESTION: {question}\n\nCONTEXT:\n{context}\n\nANSWER:"
)


def _parse_descriptions(text: str, count: int) -> list[str]:
    match = re.search(r"\[.*\]", text or "", re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            out: list[str] = []
            for item in parsed[:count]:
                if isinstance(item, dict):
                    out.append(json.dumps(item, ensure_ascii=False))
                else:
                    out.append(str(item))
            out.extend([""] * (count - len(out)))
            return out
    logger.warning("VLM returned non-JSON batch; using raw text for first frame")
    return [text.strip()] + [""] * (count - 1)


class _FrameCache:
    """Disk cache of per-frame descriptions keyed by image hash."""

    def __init__(self, directory: str, enabled: bool) -> None:
        self.dir = Path(directory) if enabled else None

    @staticmethod
    def key(jpeg: bytes) -> str:
        return hashlib.sha256(jpeg).hexdigest()

    def get(self, key: str) -> str | None:
        if self.dir is None:
            return None
        path = self.dir / key[:2] / key
        if path.exists():
            try:
                return path.read_text(encoding="utf-8")
            except OSError:
                return None
        return None

    def put(self, key: str, text: str) -> None:
        if self.dir is None or not text:
            return
        path = self.dir / key[:2] / key
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except OSError:
            pass


class GeminiClient:
    """Gemini vision + text, with retry/backoff, disk cache and a daily cap."""

    provider = "GEMINI"

    def __init__(
        self,
        settings: Settings,
        tracker: TokenTracker | None = None,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        purpose: str = "",
    ) -> None:
        self.settings = settings
        self.api_key = api_key or settings.GOOGLE_API_KEY
        self.model = model or settings.VISION_LLM_MODEL
        self.base_url = base_url or ""
        self.purpose = purpose
        self.tracker = tracker or TokenTracker()
        self.limiter = RateLimiter(settings.VLM_MAX_RPM)
        self.cache = _FrameCache(settings.VLM_CACHE_DIR, settings.ENABLE_CACHE)
        self._client = None
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

    def _check_daily_cap(self) -> None:
        if self.settings.VLM_MAX_RPD <= 0:
            return
        path = Path(self.settings.VLM_CACHE_DIR) / f"rpd_{time.strftime('%Y%m%d')}.txt"
        try:
            count = int(path.read_text()) if path.exists() else 0
        except (OSError, ValueError):
            count = 0
        if count >= self.settings.VLM_MAX_RPD:
            raise RuntimeError(
                f"Daily VLM request cap reached ({self.settings.VLM_MAX_RPD}); "
                "raise VLM_MAX_RPD or retry tomorrow."
            )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(count + 1))
        except OSError:
            pass

    def _generate(
        self, contents: list, json_mode: bool = False, temperature: float | None = None
    ) -> str:
        attempts = max(1, self.settings.VLM_RETRY_MAX)
        last_exc: Exception | None = None
        for attempt in range(attempts):
            self.limiter.wait()
            self._check_daily_cap()
            try:
                client = self._ensure()
                kwargs = {"model": self.model, "contents": contents}
                if json_mode or temperature is not None:
                    from google.genai import types

                    config_kwargs: dict = {}
                    if json_mode:
                        config_kwargs["response_mime_type"] = "application/json"
                    if temperature is not None:
                        config_kwargs["temperature"] = temperature
                    kwargs["config"] = types.GenerateContentConfig(**config_kwargs)
                response = client.models.generate_content(**kwargs)
                usage = getattr(response, "usage_metadata", None)
                prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
                completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
                cost, status = self.settings.cost_for(
                    self.model, "GEMINI", prompt_tokens, completion_tokens,
                    purpose=self.purpose,
                )
                self.tracker.record(
                    "GEMINI",
                    prompt_tokens,
                    completion_tokens,
                    cost,
                    model=self.model,
                    unrated=(status == "missing"),
                )
                return response.text or ""
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                self._reset_client()  # a closed/stale client must be recreated
                if attempt < attempts - 1:
                    delay = self.settings.VLM_BACKOFF_BASE ** attempt
                    logger.warning("Gemini call failed (%s); retrying in %.1fs", exc, delay)
                    time.sleep(delay)
        raise RuntimeError(f"Gemini call failed after {attempts} attempts: {last_exc}")

    def describe_frames(self, frames: list[tuple[float, bytes]]) -> list[str]:
        from google.genai import types

        if not frames:
            return []
        results: list[str | None] = [None] * len(frames)
        pending: list[tuple[int, float, bytes, str]] = []
        for index, (timestamp, jpeg) in enumerate(frames):
            key = _FrameCache.key(jpeg)
            cached = self.cache.get(key)
            if cached is not None:
                results[index] = cached
            else:
                pending.append((index, timestamp, jpeg, key))

        batch_size = max(1, self.settings.VLM_BATCH_SIZE)
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            parts = [
                types.Part.from_bytes(data=jpeg, mime_type="image/jpeg")
                for _, _, jpeg, _ in batch
            ]
            prompt = FRAME_PROMPT.format(
                n=len(batch),
                timestamps=[round(ts, 2) for _, ts, _, _ in batch],
            )
            text = self._generate([prompt, *parts], json_mode=True)
            for (index, _, _, key), description in zip(
                batch, _parse_descriptions(text, len(batch))
            ):
                results[index] = description
                self.cache.put(key, description)
        return [item or "" for item in results]

    def summarize(self, text: str, filename: str) -> str:
        return self._generate(
            [SUMMARY_PROMPT.format(filename=filename, text=text)]
        ).strip()

    def generate(self, prompt: str, temperature: float | None = None) -> str:
        return self._generate([prompt], temperature=temperature).strip()


class OllamaClient:
    """Local Ollama text + vision client (no API key, offline)."""

    provider = "OLLAMA"

    def __init__(
        self,
        settings: Settings,
        model: str,
        tracker: TokenTracker | None = None,
        base_url: str | None = None,
    ) -> None:
        self.settings = settings
        self.model = model
        self.base = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.tracker = tracker or TokenTracker()

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read().decode("utf-8"))

    def generate(self, prompt: str, temperature: float | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if temperature is not None:
            payload["options"] = {"temperature": temperature}
        data = self._post("/api/chat", payload)
        self.tracker.record(
            "OLLAMA",
            int(data.get("prompt_eval_count", 0) or 0),
            int(data.get("eval_count", 0) or 0),
            0.0,
            model=self.model,
            local=True,
        )
        return (data.get("message") or {}).get("content", "")

    def describe_frames(self, frames: list[tuple[float, bytes]]) -> list[str]:
        if not frames:
            return []
        descriptions: list[str] = []
        batch_size = max(1, self.settings.VLM_BATCH_SIZE)
        for start in range(0, len(frames), batch_size):
            batch = frames[start : start + batch_size]
            images = [base64.b64encode(jpeg).decode("ascii") for _, jpeg in batch]
            prompt = FRAME_PROMPT.format(
                n=len(batch), timestamps=[round(ts, 2) for ts, _ in batch]
            )
            data = self._post(
                "/api/chat",
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt, "images": images}],
                    "stream": False,
                },
            )
            text = (data.get("message") or {}).get("content", "")
            descriptions.extend(_parse_descriptions(text, len(batch)))
            self.tracker.record(
                "OLLAMA",
                int(data.get("prompt_eval_count", 0) or 0),
                int(data.get("eval_count", 0) or 0),
                0.0,
                model=self.model,
                local=True,
            )
        return descriptions

    def summarize(self, text: str, filename: str) -> str:
        return self.generate(SUMMARY_PROMPT.format(filename=filename, text=text)).strip()


class DeepSeekClient:
    """DeepSeek text + vision (OpenAI-compatible image_url parts)."""

    provider = "DEEPSEEK"

    def __init__(
        self,
        settings: Settings,
        tracker: TokenTracker | None = None,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        purpose: str = "llm",
    ) -> None:
        self.settings = settings
        role = settings.llm_role
        self.api_key = api_key if api_key is not None else role.api_key
        self.model = model or role.model
        self.base_url = base_url or role.base_url or "https://api.deepseek.com"
        self.purpose = purpose
        self.tracker = tracker or TokenTracker()
        self._client = None

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def _record(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        cost, status = self.settings.cost_for(
            model, "DEEPSEEK", prompt_tokens, completion_tokens, purpose=self.purpose
        )
        self.tracker.record(
            "DEEPSEEK",
            prompt_tokens,
            completion_tokens,
            cost,
            model=model,
            unrated=(status == "missing"),
        )

    def generate(
        self, prompt: str, model: str | None = None, temperature: float | None = None
    ) -> str:
        client = self._ensure()
        create_kwargs: dict = {
            "model": model or self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        response = client.chat.completions.create(**create_kwargs)
        usage = getattr(response, "usage", None)
        used_model = model or self.model
        self._record(
            used_model,
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
        )
        return response.choices[0].message.content or ""

    def describe_frames(self, frames: list[tuple[float, bytes]]) -> list[str]:
        if not frames:
            return []
        descriptions: list[str] = []
        batch_size = max(1, self.settings.VLM_BATCH_SIZE)
        for start in range(0, len(frames), batch_size):
            batch = frames[start : start + batch_size]
            content: list[dict] = [
                {
                    "type": "text",
                    "text": FRAME_PROMPT.format(
                        n=len(batch), timestamps=[round(ts, 2) for ts, _ in batch]
                    ),
                }
            ]
            for _, jpeg in batch:
                encoded = base64.b64encode(jpeg).decode("ascii")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    }
                )
            client = self._ensure()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
            )
            usage = getattr(response, "usage", None)
            self._record(
                self.model,
                int(getattr(usage, "prompt_tokens", 0) or 0),
                int(getattr(usage, "completion_tokens", 0) or 0),
            )
            text = response.choices[0].message.content or ""
            descriptions.extend(_parse_descriptions(text, len(batch)))
        return descriptions

    def summarize(self, text: str, filename: str) -> str:
        return self.generate(
            SUMMARY_PROMPT.format(filename=filename, text=text)
        ).strip()


class OpenAIClient:
    """OpenAI chat + vision. Also serves xAI (Grok), whose API is OpenAI-compatible."""

    def __init__(
        self,
        settings: Settings,
        tracker: TokenTracker | None = None,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        purpose: str = "",
        provider: str = "OPENAI",
    ) -> None:
        self.settings = settings
        self.provider = provider
        role = settings.llm_role
        self.api_key = api_key if api_key is not None else role.api_key
        self.model = model or role.model
        self.base_url = base_url or role.base_url or "https://api.openai.com/v1"
        self.purpose = purpose
        self.tracker = tracker or TokenTracker()
        self._client = None

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def _record(self, prompt_tokens: int, completion_tokens: int) -> None:
        cost, status = self.settings.cost_for(
            self.model, self.provider, prompt_tokens, completion_tokens,
            purpose=self.purpose,
        )
        self.tracker.record(
            self.provider, prompt_tokens, completion_tokens, cost,
            model=self.model, unrated=(status == "missing"),
        )

    def generate(self, prompt: str, temperature: float | None = None) -> str:
        client = self._ensure()
        create_kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        response = client.chat.completions.create(**create_kwargs)
        usage = getattr(response, "usage", None)
        self._record(
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
        )
        return response.choices[0].message.content or ""

    def describe_frames(self, frames: list[tuple[float, bytes]]) -> list[str]:
        if not frames:
            return []
        descriptions: list[str] = []
        batch_size = max(1, self.settings.VLM_BATCH_SIZE)
        for start in range(0, len(frames), batch_size):
            batch = frames[start : start + batch_size]
            content: list[dict] = [
                {
                    "type": "text",
                    "text": FRAME_PROMPT.format(
                        n=len(batch), timestamps=[round(ts, 2) for ts, _ in batch]
                    ),
                }
            ]
            for _, jpeg in batch:
                encoded = base64.b64encode(jpeg).decode("ascii")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    }
                )
            client = self._ensure()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
            )
            usage = getattr(response, "usage", None)
            self._record(
                int(getattr(usage, "prompt_tokens", 0) or 0),
                int(getattr(usage, "completion_tokens", 0) or 0),
            )
            text = response.choices[0].message.content or ""
            descriptions.extend(_parse_descriptions(text, len(batch)))
        return descriptions

    def summarize(self, text: str, filename: str) -> str:
        return self.generate(
            SUMMARY_PROMPT.format(filename=filename, text=text)
        ).strip()


class AnthropicClient:
    """Anthropic Claude chat + vision via the Messages API."""

    provider = "ANTHROPIC"
    MAX_TOKENS = 2048

    def __init__(
        self,
        settings: Settings,
        tracker: TokenTracker | None = None,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        purpose: str = "",
    ) -> None:
        self.settings = settings
        role = settings.llm_role
        self.api_key = api_key if api_key is not None else role.api_key
        self.model = model or role.model
        self.base_url = base_url or role.base_url or "https://api.anthropic.com"
        self.purpose = purpose
        self.tracker = tracker or TokenTracker()
        self._client = None

    def _ensure(self):
        if self._client is None:
            import anthropic

            kwargs: dict = {"api_key": self.api_key}
            if self.base_url and self.base_url != "https://api.anthropic.com":
                kwargs["base_url"] = self.base_url
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def _record(self, prompt_tokens: int, completion_tokens: int) -> None:
        cost, status = self.settings.cost_for(
            self.model, self.provider, prompt_tokens, completion_tokens,
            purpose=self.purpose,
        )
        self.tracker.record(
            self.provider, prompt_tokens, completion_tokens, cost,
            model=self.model, unrated=(status == "missing"),
        )

    @staticmethod
    def _text(response) -> str:
        return "".join(
            block.text for block in (response.content or []) if getattr(block, "type", "") == "text"
        )

    def generate(self, prompt: str, temperature: float | None = None) -> str:
        client = self._ensure()
        create_kwargs: dict = {
            "model": self.model,
            "max_tokens": self.MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        response = client.messages.create(**create_kwargs)
        usage = getattr(response, "usage", None)
        self._record(
            int(getattr(usage, "input_tokens", 0) or 0),
            int(getattr(usage, "output_tokens", 0) or 0),
        )
        return self._text(response).strip()

    def describe_frames(self, frames: list[tuple[float, bytes]]) -> list[str]:
        if not frames:
            return []
        descriptions: list[str] = []
        batch_size = max(1, self.settings.VLM_BATCH_SIZE)
        for start in range(0, len(frames), batch_size):
            batch = frames[start : start + batch_size]
            blocks: list[dict] = [
                {
                    "type": "text",
                    "text": FRAME_PROMPT.format(
                        n=len(batch), timestamps=[round(ts, 2) for ts, _ in batch]
                    ),
                }
            ]
            for _, jpeg in batch:
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64.b64encode(jpeg).decode("ascii"),
                        },
                    }
                )
            client = self._ensure()
            response = client.messages.create(
                model=self.model,
                max_tokens=self.MAX_TOKENS,
                messages=[{"role": "user", "content": blocks}],
            )
            usage = getattr(response, "usage", None)
            self._record(
                int(getattr(usage, "input_tokens", 0) or 0),
                int(getattr(usage, "output_tokens", 0) or 0),
            )
            descriptions.extend(_parse_descriptions(self._text(response), len(batch)))
        return descriptions

    def summarize(self, text: str, filename: str) -> str:
        return self.generate(
            SUMMARY_PROMPT.format(filename=filename, text=text)
        ).strip()


class FallbackDescriber:
    """Try the primary VLM; on failure fall back to a local one."""

    def __init__(self, primary, fallback) -> None:
        self.primary = primary
        self.fallback = fallback

    def describe_frames(self, frames):
        try:
            return self.primary.describe_frames(frames)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Primary VLM failed (%s); using local fallback", exc)
            return self.fallback.describe_frames(frames)


def _llm_client(settings: Settings, tracker: TokenTracker):
    role = settings.llm_role
    if role.provider == "DEEPSEEK" and role.api_key:
        return DeepSeekClient(settings, tracker)
    if role.provider == "GEMINI" and role.api_key:
        return GeminiClient(
            settings, tracker, api_key=role.api_key, model=role.model,
            base_url=role.base_url, purpose="llm",
        )
    if role.provider == "OPENAI" and role.api_key:
        return OpenAIClient(
            settings, tracker, api_key=role.api_key, model=role.model,
            base_url=role.base_url, purpose="llm", provider="OPENAI",
        )
    if role.provider == "XAI" and role.api_key:
        # xAI (Grok) is OpenAI-compatible; only the endpoint/key differ.
        return OpenAIClient(
            settings, tracker, api_key=role.api_key, model=role.model,
            base_url=role.base_url, purpose="llm", provider="XAI",
        )
    if role.provider == "ANTHROPIC" and role.api_key:
        return AnthropicClient(
            settings, tracker, api_key=role.api_key, model=role.model,
            base_url=role.base_url, purpose="llm",
        )
    if role.provider in ("OLLAMA", "LOCAL"):
        return OllamaClient(settings, role.model, tracker, base_url=role.base_url)
    return None


def build_summarize_fn(settings: Settings, tracker: TokenTracker):
    try:
        client = _llm_client(settings, tracker)
        return client.summarize if client is not None else None
    except Exception as exc:
        logger.warning("Summarizer unavailable: %s", exc)
        return None


def build_vlm_describer(settings: Settings, tracker: TokenTracker):
    role = settings.vlm_role
    primary = None
    if role.provider == "GEMINI" and role.api_key:
        primary = GeminiClient(
            settings, tracker, api_key=role.api_key, model=role.model,
            base_url=role.base_url, purpose="vlm",
        )
    elif role.provider == "OPENAI" and role.api_key:
        primary = OpenAIClient(
            settings, tracker, api_key=role.api_key, model=role.model,
            base_url=role.base_url, purpose="vlm", provider="OPENAI",
        )
    elif role.provider == "XAI" and role.api_key:
        primary = OpenAIClient(
            settings, tracker, api_key=role.api_key, model=role.model,
            base_url=role.base_url, purpose="vlm", provider="XAI",
        )
    elif role.provider == "ANTHROPIC" and role.api_key:
        primary = AnthropicClient(
            settings, tracker, api_key=role.api_key, model=role.model,
            base_url=role.base_url, purpose="vlm",
        )
    elif role.provider == "DEEPSEEK" and role.api_key:
        primary = DeepSeekClient(
            settings, tracker, api_key=role.api_key, model=role.model,
            base_url=role.base_url or "https://api.deepseek.com", purpose="vlm",
        )
    elif role.provider in ("OLLAMA", "LOCAL"):
        primary = OllamaClient(settings, role.model, tracker, base_url=role.base_url)
    if primary is None:
        return None
    if settings.vlm_local_fallback_enabled:
        fallback_model = settings.vlm_local_fallback_model
        if fallback_model:
            return FallbackDescriber(
                primary, OllamaClient(settings, fallback_model, tracker)
            )
    return primary


def build_chat_client(settings: Settings, tracker: TokenTracker):
    """Text-only client for RAG answer synthesis (`.generate(prompt)`)."""
    return _llm_client(settings, tracker)


def format_context(results: list) -> str:
    lines = []
    for index, result in enumerate(results, start=1):
        record = result.record
        location = record.source_path
        if record.timestamp_start is not None:
            location += f" @ {record.timestamp_start:.1f}s"
        lines.append(f"[{index}] ({location}) {record.content}")
    return "\n".join(lines)


def generate_rag_answer(
    settings: Settings,
    tracker: TokenTracker,
    question: str,
    results: list,
) -> str | None:
    """Synthesize a cited answer from retrieved results; None if no text LLM."""
    if not results:
        return None
    client = build_chat_client(settings, tracker)
    if client is None:
        return None
    prompt = ANSWER_PROMPT.format(question=question, context=format_context(results))
    # Factual synthesis over retrieved context: temperature comes from
    # LLM_TEMPERATURE (default 0.0 = deterministic verdicts, same evidence →
    # same answer). Summaries and captions keep their defaults (their generate
    # paths pass no temperature).
    # The two log lines below are the flip-flop diagnostic: identical
    # prompt_sha across runs with different verdicts == generation variance;
    # differing prompt_sha == retrieval variance. Absence of these lines means
    # this code is not the build actually running.
    temperature = float(getattr(settings, "LLM_TEMPERATURE", 0.0))
    prompt_sha = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
    logger.info(
        "rag_answer q=%r n_results=%d prompt_chars=%d prompt_sha=%s temperature=%s",
        question,
        len(results),
        len(prompt),
        prompt_sha,
        temperature,
    )
    answer = client.generate(prompt, temperature=temperature)
    logger.info(
        "rag_answer done prompt_sha=%s completion_chars=%d head=%r",
        prompt_sha,
        len(answer or ""),
        (answer or "")[:200],
    )
    return answer
