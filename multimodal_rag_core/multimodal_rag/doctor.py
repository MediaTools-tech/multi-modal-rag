from __future__ import annotations

from multimodal_rag.config import RoleSettings, get_settings


def _state(role: RoleSettings, needs_key: bool) -> str:
    if not needs_key:
        return "n/a (local)"
    return "set" if role.api_key else "MISSING"


def _needs_key(provider: str) -> bool:
    return provider not in ("LOCAL", "OLLAMA", "NONE")


def _bool(ok: bool) -> str:
    return "OK" if ok else "MISSING"


def _print_fingerprint(settings) -> None:
    """Show the stored index fingerprint vs current index-time config."""
    import json

    from multimodal_rag.ingestion.state import INDEX_FINGERPRINT_KEYS, StateStore

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
    drift = False
    for key in INDEX_FINGERPRINT_KEYS:
        value = getattr(settings, key)
        current = value.value if hasattr(value, "value") else str(value)
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
    print(f"SYSTEM_MODE : {settings.SYSTEM_MODE.value}  ->  engine {settings.ACTIVE_DB_ENGINE.value}")
    print(f"App root    : {settings.app_root}  (MRAG_HOME)")
    print(f"Ingest state: {settings.STATE_DB_PATH}")
    print()

    roles = [
        (settings.embedding_role, "indexing + query vectors"),
        (settings.vlm_role, "video keyframe captions"),
        (settings.llm_role, "per-file summaries + synthesized answers"),
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

    print("RERANKER    cross-encoder over fused candidates")
    print(f"  provider : {settings.RERANKER_PROVIDER.value}   model={settings.RERANKER_MODEL}"
          f"   active={settings.active_reranker}")
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
