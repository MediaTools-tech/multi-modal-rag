from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def is_local_model_path(model_id: str) -> bool:
    """True when model_id points at an existing local directory."""
    if not model_id:
        return False
    try:
        return Path(model_id).expanduser().exists()
    except OSError:
        return False


def model_is_cached(model_id: str) -> bool:
    """True when a complete-looking Hugging Face snapshot exists locally.

    Checks the Hub cache directly instead of asking the Hub. SentenceTransformer
    and transformers download individual files (not snapshot trees), so we look
    for a snapshot containing both config metadata and model weights. Callers
    then pass ``local_files_only=True``: an already-cached model is never
    re-checked for upstream weight changes on startup.
    """
    if not model_id or is_local_model_path(model_id):
        return is_local_model_path(model_id)
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        repo_dir = Path(HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}"
        snapshots = repo_dir / "snapshots"
        if not snapshots.is_dir():
            return False
        # Prefer the revision transformers would resolve (refs/main). If its
        # snapshot is partial, report "not cached" so the caller may fetch the
        # missing files instead of failing a local-only load.
        main_ref = repo_dir / "refs" / "main"
        if main_ref.is_file():
            commit = main_ref.read_text(encoding="utf-8").strip()
            snapshot = snapshots / commit
            return snapshot.is_dir() and _snapshot_has_config_and_weights(snapshot)
        return any(
            _snapshot_has_config_and_weights(snapshot)
            for snapshot in snapshots.iterdir()
            if snapshot.is_dir()
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not inspect Hugging Face cache for %s: %s", model_id, exc)
        return False


def _snapshot_has_config_and_weights(snapshot: Path) -> bool:
    try:
        names = {entry.name for entry in snapshot.iterdir() if entry.is_file()}
    except OSError:
        return False
    has_config = bool(names & {"config.json", "modules.json"})
    has_weights = any(_is_weight_file(name) for name in names)
    return has_config and has_weights


def _is_weight_file(name: str) -> bool:
    if name in {"model.safetensors", "pytorch_model.bin"}:
        return True
    prefix_match = name.startswith(("model-", "pytorch_model-"))
    suffix_match = name.endswith((".safetensors", ".bin"))
    return prefix_match and suffix_match
