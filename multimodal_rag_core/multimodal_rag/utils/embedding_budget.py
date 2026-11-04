from __future__ import annotations

import logging
from collections.abc import Callable

from multimodal_rag.config import (
    CHUNK_DEFAULT_TOKENS,
    CHUNK_EMBEDDING_TOKEN_RATIO,
    SUMMARY_EMBEDDING_TOKEN_RATIO,
    embedding_max_input_tokens,
    is_embeddinggemma_spec,
    parse_embeddinggemma_spec,
)

logger = logging.getLogger(__name__)

_TOKENIZER_CACHE: dict[str, Callable[[str], int] | None] = {}
_WARNED_BUDGETS: set[tuple[str, int]] = set()


def summary_token_budget(model: str) -> int:
    """Tokens a summary may use, leaving headroom for embedder prefixes."""
    window = embedding_max_input_tokens(model)
    return max(1, int(window * SUMMARY_EMBEDDING_TOKEN_RATIO))


def chunk_token_budget(model: str) -> int:
    """Maximum safe token budget for a passage chunk, with embedder headroom."""
    window = embedding_max_input_tokens(model)
    return max(1, int(window * CHUNK_EMBEDDING_TOKEN_RATIO))


def chunk_token_target(model: str, configured: int = 0) -> int:
    """Effective chunk target: configured value or 450, capped by the model."""
    target = int(configured) if int(configured) > 0 else CHUNK_DEFAULT_TOKENS
    return max(1, min(target, chunk_token_budget(model)))


def count_tokens(text: str, model: str, *, local: bool) -> int:
    """Count tokens with the local tokenizer when available, else estimate."""
    return _token_counter(model, local=local)(text)


def _is_cjk(character: str) -> bool:
    code = ord(character)
    return (
        0x3040 <= code <= 0x30FF  # Hiragana / Katakana
        or 0x3400 <= code <= 0x4DBF  # CJK Extension A
        or 0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0xAC00 <= code <= 0xD7AF  # Hangul syllables
        or 0xF900 <= code <= 0xFAFF  # CJK Compatibility Ideographs
    )


def estimate_tokens(text: str) -> int:
    """Conservative fallback when no local tokenizer is available.

    CJK characters are counted one-for-one; other text uses the usual ~4
    characters/token estimate. Over-estimating is intentional: a summary must
    never be silently truncated by the embedder.
    """
    if not text:
        return 0
    cjk = sum(1 for character in text if _is_cjk(character))
    return cjk + max(1, (len(text) - cjk) // 4)


def _base_model_id(model: str) -> str:
    if is_embeddinggemma_spec(model):
        base, _, _ = parse_embeddinggemma_spec(model)
        return base
    return model


def _token_counter(model: str, *, local: bool) -> Callable[[str], int]:
    if not local:
        return estimate_tokens
    if model in _TOKENIZER_CACHE:
        cached = _TOKENIZER_CACHE[model]
        return cached or estimate_tokens
    counter: Callable[[str], int] | None = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(_base_model_id(model), local_files_only=True)

        def counter(text: str) -> int:
            try:
                return len(tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                return estimate_tokens(text)

    except Exception as exc:
        logger.debug("Tokenizer unavailable for %s (%s); using char estimate", model, exc)
    _TOKENIZER_CACHE[model] = counter
    return counter or estimate_tokens


def fit_summary_to_embedding_limit(
    text: str,
    model: str,
    *,
    local: bool,
) -> tuple[str, int, bool]:
    """Clamp ``text`` to the embedding model's input window.

    Returns ``(text, budget_tokens, was_clamped)``. The user's word/char
    budgets still apply first; this is only the model-aware safety bound.
    """
    budget = summary_token_budget(model)
    if not text:
        return text, budget, False
    counter = _token_counter(model, local=local)
    if counter(text) <= budget:
        return text, budget, False

    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if counter(text[:middle]) <= budget:
            low = middle
        else:
            high = middle - 1
    clipped = text[:low].rstrip()
    boundary = clipped.rfind(" ")
    if boundary >= max(1, len(clipped) // 2):
        clipped = clipped[:boundary]
    clipped = clipped.rstrip(" ,;:-") or text[:1]

    key = (model, budget)
    if key not in _WARNED_BUDGETS:
        _WARNED_BUDGETS.add(key)
        logger.warning(
            "Summary exceeded the embedding input window for %s; clamping output "
            "to ~%d tokens (model limit %d). Use a model with a larger window or "
            "lower SUMMARY_TARGET_WORDS/SUMMARY_OUTPUT_MAX_CHARS to silence this.",
            model,
            budget,
            embedding_max_input_tokens(model),
        )
    else:
        logger.debug("Summary clamped to %d tokens for %s", budget, model)
    return clipped, budget, True
