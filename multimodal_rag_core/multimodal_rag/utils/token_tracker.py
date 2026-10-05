from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ModelUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    local: bool = False
    unrated: bool = False


def compute_model_cost(
    model_rates: dict,
    provider_fallbacks: dict,
    model: str,
    provider: str,
    prompt_tokens: int,
    completion_tokens: int,
    purpose: str = "",
    purpose_rates: dict | None = None,
) -> tuple[float, str]:
    """Resolve (cost_usd, status) for one model call.

    Precedence: per-model ``MODEL_TOKEN_RATES`` entry (exact name, then the
    part after ``/`` so ``models/gemini-embedding-001`` matches
    ``gemini-embedding-001``) → per-purpose rates (``vlm`` / ``llm`` /
    ``embedding`` pairs, counted only when non-zero) → provider-level
    fallback rates (same non-zero rule) → ``"missing"``.
    Local providers (Ollama / local engines) are always ``"local"``.
    """
    provider = (provider or "").upper()
    if provider in ("OLLAMA", "LOCAL"):
        return 0.0, "local"

    def _priced(amount: int, rate: float) -> float:
        return amount / 1_000_000 * float(rate)

    entry = (model_rates or {}).get(model or "")
    if entry is None and model and "/" in model:
        entry = (model_rates or {}).get(model.rsplit("/", 1)[-1])
    if isinstance(entry, (list, tuple)) and len(entry) == 2:
        entry = {"input": entry[0], "output": entry[1]}
    if entry is not None:
        return (
            _priced(prompt_tokens, entry.get("input", 0.0))
            + _priced(completion_tokens, entry.get("output", 0.0)),
            "rated",
        )
    key = (purpose or "").lower()
    pair = (purpose_rates or {}).get(key) if key else None
    if pair is not None and (float(pair[0]) > 0 or float(pair[1]) > 0):
        return (
            _priced(prompt_tokens, pair[0]) + _priced(completion_tokens, pair[1]),
            "rated",
        )
    fallback = provider_fallbacks.get(provider, (0.0, 0.0))
    if float(fallback[0]) > 0 or float(fallback[1]) > 0:
        return (
            _priced(prompt_tokens, fallback[0]) + _priced(completion_tokens, fallback[1]),
            "rated",
        )
    return 0.0, "missing"


@dataclass
class TokenTracker:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    per_provider: dict[str, int] = field(default_factory=dict)
    per_model: dict[str, ModelUsage] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(
        self,
        provider: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
        model: str = "",
        local: bool = False,
        unrated: bool = False,
    ) -> None:
        prompt_tokens = int(prompt_tokens)
        completion_tokens = int(completion_tokens)
        cost_usd = float(cost_usd)
        with self._lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.cost_usd += cost_usd
            self.calls += 1
            self.per_provider[provider] = self.per_provider.get(provider, 0) + 1
            key = model or provider
            entry = self.per_model.get(key)
            if entry is None:
                entry = self.per_model[key] = ModelUsage()
            entry.prompt_tokens += prompt_tokens
            entry.completion_tokens += completion_tokens
            entry.cost_usd += cost_usd
            entry.calls += 1
            if local:
                entry.local = True
                entry.unrated = False
            elif unrated:
                entry.unrated = True
            else:
                entry.unrated = False
        logger.info(
            "tokens provider=%s model=%s prompt=%d completion=%d est_cost=$%.6f",
            provider,
            key,
            prompt_tokens,
            completion_tokens,
            cost_usd,
        )

    @staticmethod
    def cost(
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        input_rate_per_1m: float,
        output_rate_per_1m: float,
    ) -> float:
        return (
            prompt_tokens / 1_000_000 * input_rate_per_1m
            + completion_tokens / 1_000_000 * output_rate_per_1m
        )

    def snapshot(self) -> dict:
        """Point-in-time counters; pass to :meth:`delta` for per-request usage."""
        with self._lock:
            return {
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
                "cost": self.cost_usd,
                "calls": self.calls,
                "models": {
                    key: {
                        "prompt": entry.prompt_tokens,
                        "completion": entry.completion_tokens,
                        "cost": entry.cost_usd,
                        "calls": entry.calls,
                        "local": entry.local,
                        "unrated": entry.unrated,
                    }
                    for key, entry in self.per_model.items()
                },
            }

    @staticmethod
    def delta(before: dict, after: dict) -> dict:
        """Usage accumulated between two snapshots (per-request view)."""
        models: dict[str, dict] = {}
        for key, later in after.get("models", {}).items():
            earlier = before.get("models", {}).get(key, {})
            prompt = later.get("prompt", 0) - earlier.get("prompt", 0)
            completion = later.get("completion", 0) - earlier.get("completion", 0)
            cost = later.get("cost", 0.0) - earlier.get("cost", 0.0)
            calls = later.get("calls", 0) - earlier.get("calls", 0)
            if prompt or completion or cost or calls:
                models[key] = {
                    "prompt": prompt,
                    "completion": completion,
                    "cost": cost,
                    "calls": calls,
                    "local": bool(later.get("local", False)),
                    "unrated": bool(later.get("unrated", False)),
                }
        return {
            "prompt": after.get("prompt", 0) - before.get("prompt", 0),
            "completion": after.get("completion", 0) - before.get("completion", 0),
            "cost": after.get("cost", 0.0) - before.get("cost", 0.0),
            "calls": after.get("calls", 0) - before.get("calls", 0),
            "models": models,
        }

    @staticmethod
    def _model_suffix(stats: dict) -> str:
        if stats.get("local"):
            return "local"
        if stats.get("unrated"):
            return "rate missing"
        return f"${stats.get('cost', 0.0):.6f}"

    @staticmethod
    def format_usage(usage: dict) -> str:
        """One-line summary, e.g. for console output or GUI labels."""
        bits = [
            f"{usage.get('prompt', 0):,} in / {usage.get('completion', 0):,} out",
            f"${usage.get('cost', 0.0):.6f}",
        ]
        models = usage.get("models", {})
        if models:
            top = sorted(models.items(), key=lambda item: item[1].get("cost", 0.0), reverse=True)[:3]
            bits.append(
                ", ".join(
                    f"{key} ({stats.get('calls', 0)}x, "
                    f"{stats.get('prompt', 0):,}/{stats.get('completion', 0):,}, "
                    f"{TokenTracker._model_suffix(stats)})"
                    for key, stats in top
                )
            )
        return " · ".join(bits)

    def report(self) -> str:
        lines = [
            f"calls={self.calls} prompt_tokens={self.prompt_tokens} "
            f"completion_tokens={self.completion_tokens} est_cost=${self.cost_usd:.6f}"
        ]
        for key in sorted(self.per_model):
            entry = self.per_model[key]
            lines.append(
                f"  {key}: calls={entry.calls} prompt={entry.prompt_tokens} "
                f"completion={entry.completion_tokens} "
                f"est_cost=${entry.cost_usd:.6f}"
                + (" [local]" if entry.local else "")
                + (" [rate missing]" if entry.unrated else "")
            )
        return "\n".join(lines)
