from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

from multimodal_rag.config import DBEngine, Settings, get_settings
from multimodal_rag.core.models import VectorRecord
from multimodal_rag.ingestion.state import StateStore, fingerprint_of

logger = logging.getLogger("mrag-db-copy")


def _other_engine(engine: DBEngine) -> DBEngine:
    return DBEngine.LANCEDB if engine is DBEngine.POSTGRES else DBEngine.POSTGRES


def _make_repo(settings: Settings, engine: DBEngine):
    scoped = settings.model_copy(update={"ACTIVE_DB_ENGINE": engine})
    if engine is DBEngine.LANCEDB:
        from multimodal_rag.database.lancedb_provider import LanceDBProvider

        return LanceDBProvider(scoped)
    if engine is DBEngine.POSTGRES:
        from multimodal_rag.database.postgres_provider import PostgresProvider

        return PostgresProvider(scoped)
    raise ValueError(f"Unsupported engine: {engine}")


def _is_uuid(value: object) -> bool:
    try:
        uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return False
    return True


def _prepare_batch(records: list[VectorRecord], target: DBEngine) -> list[VectorRecord]:
    """Postgres ids are uuid columns; namespace any legacy non-UUID ids."""
    if target is not DBEngine.POSTGRES:
        return records
    prepared: list[VectorRecord] = []
    for record in records:
        if _is_uuid(record.id):
            prepared.append(record)
            continue
        data = record.to_dict()
        data["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, str(record.id)))
        prepared.append(VectorRecord.from_dict(data))
    return prepared


def copy_records(source, target, target_engine: DBEngine, batch_size: int = 500) -> int:
    """Copy every source record into target; returns the number inserted."""
    copied = 0
    for batch in source.iter_records(batch_size):
        if not batch:
            continue
        copied += target.insert(_prepare_batch(batch, target_engine))
    return copied


def copy_missing_records(
    source,
    target,
    target_engine: DBEngine,
    batch_size: int = 500,
    *,
    allowed_hashes: set[str] | None = None,
    allowed_paths: set[str] | None = None,
) -> int:
    """Merge source records absent from target, preserving existing target data.

    Matching uses file hash first and source path second, so switching engines
    copies only files that exist in the previous engine but not the current one.
    ``allowed_*`` limits copies to files still marked processed in state.db.
    """
    filter_by_state = allowed_hashes is not None or allowed_paths is not None
    hashes_filter = allowed_hashes or set()
    paths_filter = allowed_paths or set()
    target_hashes = target.indexed_file_hashes()
    target_paths = target.indexed_paths()
    copied = 0
    for batch in source.iter_records(batch_size):
        pending: list[VectorRecord] = []
        for record in batch:
            if filter_by_state:
                by_hash = bool(record.file_hash) and record.file_hash in hashes_filter
                by_path = bool(record.source_path) and record.source_path in paths_filter
                if not by_hash and not by_path:
                    continue
            if record.file_hash and record.file_hash in target_hashes:
                continue
            if record.source_path and record.source_path in target_paths:
                continue
            pending.append(record)
        if not pending:
            continue
        copied += target.insert(_prepare_batch(pending, target_engine))
        for record in pending:
            if record.file_hash:
                target_hashes.add(record.file_hash)
            if record.source_path:
                target_paths.add(record.source_path)
    return copied


def copy_missing_between(
    settings: Settings,
    source_engine: DBEngine,
    target_engine: DBEngine,
    target,
    batch_size: int = 500,
    *,
    allowed_hashes: set[str] | None = None,
    allowed_paths: set[str] | None = None,
) -> int:
    """Open ``source_engine`` and merge its missing records into ``target``."""
    source = _make_repo(settings, source_engine)
    try:
        source.initialize()
        return copy_missing_records(
            source,
            target,
            target_engine,
            batch_size,
            allowed_hashes=allowed_hashes,
            allowed_paths=allowed_paths,
        )
    finally:
        source.close()


def build_parser() -> argparse.ArgumentParser:
    choices = [engine.value for engine in DBEngine]
    parser = argparse.ArgumentParser(
        prog="mrag-db-copy",
        description=(
            "Copy indexed vectors between LanceDB and Postgres without re-parsing "
            "or re-embedding source files."
        ),
    )
    parser.add_argument(
        "--from",
        dest="source",
        type=str.upper,
        choices=choices,
        default=None,
        help="source engine (default: current ACTIVE_DB_ENGINE)",
    )
    parser.add_argument(
        "--to",
        dest="target",
        type=str.upper,
        choices=choices,
        default=None,
        help="target engine (default: the other engine)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="clear the target first (required when it already has records)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show source/target counts and exit without copying",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="records per insert batch (default: 500)",
    )
    parser.add_argument(
        "--no-state",
        action="store_true",
        help="do not copy the engine-specific state.db queue/status rows",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()
    source_engine = DBEngine(args.source) if args.source else settings.ACTIVE_DB_ENGINE
    if source_engine is None:
        print("No active DB engine configured.", file=sys.stderr)
        return 1
    target_engine = (
        DBEngine(args.target) if args.target else _other_engine(source_engine)
    )
    if source_engine is target_engine:
        print("Source and target engines must differ.", file=sys.stderr)
        return 1

    target_settings = settings.model_copy(update={"ACTIVE_DB_ENGINE": target_engine})
    source_state = Path(settings.STATE_DB_PATH)
    source_default_state = Path(settings._default_state_db_path())
    target_state = (
        Path(target_settings._default_state_db_path())
        if source_state.resolve() == source_default_state.resolve()
        else source_state
    )

    print(f"copying {source_engine.value} -> {target_engine.value}")
    print("stop the GUI/ingestion worker before copying.")
    source = _make_repo(settings, source_engine)
    target = _make_repo(settings, target_engine)
    try:
        source.initialize()
        target.initialize()
        source_count = int(source.get_stats().get("total_records", 0))
        target_count = int(target.get_stats().get("total_records", 0))
        print(f"source records: {source_count}")
        print(f"target records: {target_count}")
        if source_count == 0:
            print("source store is empty; nothing to copy.", file=sys.stderr)
            return 1
        if target_count and not args.replace:
            print(
                "target store is not empty; rerun with --replace to clear it first.",
                file=sys.stderr,
            )
            return 2
        if args.dry_run:
            print("dry run: no records copied.")
            return 0
        if args.replace:
            removed = target.clear()
            print(f"cleared {removed} target record(s)")

        copied = copy_records(source, target, target_engine, args.batch_size)
        target.reindex()
        final_count = int(target.get_stats().get("total_records", 0))
        print(f"copied {copied} record(s); target now has {final_count}")
        if copied != source_count or final_count != source_count:
            print(
                "record count mismatch (source changed during copy, or duplicate "
                "ids collapsed in Postgres).",
                file=sys.stderr,
            )
            return 1

        if not args.no_state:
            target_store = StateStore(target_state)
            if target_state.resolve() != source_state.resolve():
                state_count = target_store.copy_from(
                    source_state, replace=args.replace
                )
                target_store.reset_stale()
                print(f"copied {state_count} state.db queue row(s)")
            else:
                print("state.db is shared; queue rows already present")
            target_store.set_meta(
                "index_fingerprint",
                json.dumps(fingerprint_of(target_settings), sort_keys=True),
            )
    finally:
        try:
            source.close()
        finally:
            target.close()

    print(
        f"done. Set ACTIVE_DB_ENGINE={target_engine.value} and restart the app; "
        "no source files need to be re-ingested."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
