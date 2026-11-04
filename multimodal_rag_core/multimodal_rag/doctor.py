from __future__ import annotations

from multimodal_rag.config import RoleSettings, embedding_max_input_tokens, get_settings
from multimodal_rag.utils.embedding_budget import chunk_token_target, summary_token_budget


def _state(role: RoleSettings, needs_key: bool) -> str:
    if not needs_key:
        return "n/a (local)"
    return "set" if role.api_key else "MISSING"


def _needs_key(provider: str) -> bool:
    return provider not in ("LOCAL", "OLLAMA", "NONE")


def _bool(ok: bool) -> str:
    return "OK" if ok else "MISSING"


def _print_summaries(settings) -> None:
    import importlib.util

    numpy_state = "OK" if importlib.util.find_spec("numpy") else "MISSING (frequency fallback)"
    provider = settings.SUMMARY_PROVIDER.value
    algorithm = settings.SUMMARY_ALGORITHM.value
    unit = settings.SUMMARY_SEMANTIC_UNIT.value
    print("SUMMARIES   file-level summary records")
    print(f"  enabled  : {settings.ENABLE_DOCUMENT_SUMMARIES}")
    print(f"  provider : {provider}   algorithm={algorithm}   semantic_unit={unit}")
    print(
        f"  output   : target_words={settings.SUMMARY_TARGET_WORDS}"
        f"   max_chars={settings.SUMMARY_OUTPUT_MAX_CHARS}"
        f"   embed_token_cap={summary_token_budget(settings.EMBEDDING_MODEL)}"
    )
    print(
        f"  input    : llm_max_chars={settings.SUMMARY_MAX_CHARS}"
        f"   local_max_chars={settings.SUMMARY_LOCAL_MAX_CHARS}"
    )
    print(
        f"  graph    : max_chunks={settings.SUMMARY_MAX_CHUNKS}"
        f"   max_sentences={settings.SUMMARY_MAX_SENTENCES}"
        f"   numpy={numpy_state}"
    )
    print(
        f"  images   : {settings.SUMMARY_INCLUDE_IMAGES}"
        f"   fallback_to_llm={settings.SUMMARY_FALLBACK_TO_LLM}"
    )
    print()


def _print_chunking(settings) -> None:
    model = settings.EMBEDDING_MODEL
    window = embedding_max_input_tokens(model)
    configured = int(settings.DOC_CHUNK_TOKENS or 0)
    effective = chunk_token_target(model, configured)
    overlap = int(settings.DOC_CHUNK_OVERLAP_TOKENS or 0)
    print("CHUNKING    sentence-aware token-budgeted passage splitting")
    print(f"  words    : size={settings.DOC_CHUNK_SIZE}   overlap={settings.DOC_CHUNK_OVERLAP}")
    print(
        f"  tokens   : configured={configured or 'auto'}   effective={effective}"
        f"   model_window={window}   overlap={overlap or 'auto'}"
    )
    print()


def _print_fingerprint(settings) -> None:
    """Show the stored index fingerprint vs current index-time config."""
    import json

    from multimodal_rag.ingestion.state import (
        INDEX_FINGERPRINT_KEYS,
        StateStore,
        fingerprint_of,
    )

    print("INDEX       fingerprint (index-time settings baked into stored vectors)")
    try:
        store = StateStore(settings.STATE_DB_PATH)
    except Exception as exc:
        print(f"  state.db unreadable: {exc}")
        return
    raw = store.get_meta("index_fingerprint")
    if not raw:
        print("  no fingerprint yet (nothing ingested since this feature was added)")
        return
    try:
        stored = json.loads(raw)
    except ValueError:
        stored = {}
    current_values = fingerprint_of(settings)
    drift = False
    for key in INDEX_FINGERPRINT_KEYS:
        current = current_values[key]
        mark = "" if stored.get(key) == current else "   <-- DIFFERS"
        drift = drift or bool(mark)
        print(f"  {key:<20} stored={stored.get(key)!r} current={current!r}{mark}")
    if drift:
        print("  WARNING: index-time settings changed — reset the index and re-ingest")
        print("  (GUI: File -> Reset index... / CLI: mrag-ingest reset-index)")


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()

    print("Multi-Engine Hybrid RAG - effective configuration")
    print("=" * 60)
    engine = settings.ACTIVE_DB_ENGINE.value
    print(f"SYSTEM_MODE : {settings.SYSTEM_MODE.value}  ->  engine {engine}")
    print(f"App root    : {settings.app_root}  (MRAG_HOME)")
    print(f"Ingest state: {settings.STATE_DB_PATH}")
    print()

    roles = [
        (settings.embedding_role, "indexing + query vectors"),
        (settings.vlm_role, "video keyframe captions"),
        (settings.llm_role, "synthesized answers + optional LLM summaries"),
    ]
    for role, description in roles:
        needs = _needs_key(role.provider)
        ok = (not needs) or bool(role.api_key)
        if role.provider == "NONE":
            ok = False
        print(f"{role.purpose.upper():<11} ({description})")
        print(f"  provider : {role.provider}")
        print(f"  model    : {role.model}")
        print(f"  api_key  : {_state(role, needs)}")
        if role.base_url:
            print(f"  base_url : {role.base_url}")
        print(f"  effective: {_bool(ok)}")
        print()

    _print_summaries(settings)
    _print_chunking(settings)
    print("RERANKER    cross-encoder over fused candidates")
    print(f"  provider : {settings.RERANKER_PROVIDER.value}   model={settings.RERANKER_MODEL}"
          f"   active={settings.active_reranker}")
    print()
    print(f"TYPO        query spelling correction: {settings.TYPO_CORRECTION.value}")
    print()
    _print_fingerprint(settings)
    print()
    print("HOW IT WORKS")
    print("  Each purpose (EMBEDDINGS / VLM / LLM) has its own provider, model and key.")
    print("  Set <ROLE>_API_KEY. A Google key is used once per role that needs it")
    print("  (explicit duplication), e.g. EMBEDDINGS_API_KEY and VLM_API_KEY.")
    print("  Deprecated vendor keys (GOOGLE_API_KEY / DEEPSEEK_API_KEY) still work as")
    print("  fallbacks but prefer the purpose-scoped names.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
