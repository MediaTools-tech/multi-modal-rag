from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from multimodal_rag.config import IngestionMode, Settings, get_settings

from .queue import IngestQueue
from .state import FAILED, StateStore
from .watcher import InboxWatcher
from .worker import Handler, IngestionWorker

logger = logging.getLogger("ingestion")


def _build(settings: Settings) -> tuple[StateStore, IngestQueue]:
    settings.ensure_data_dirs()
    state = StateStore(settings.STATE_DB_PATH)
    queue = IngestQueue(
        state, max_retries=settings.MAX_INGEST_RETRIES, allow_kind=settings.allows_kind
    )
    return state, queue


def _build_context(settings: Settings):
    """Shared pipeline context (repository + embedder + VLM) or None if unavailable."""
    try:
        from multimodal_rag.pipeline.context import build_context

        return build_context(settings, require_embedder=False)
    except Exception as exc:
        logger.warning(
            "Could not build pipeline context (%s); queued files will be marked "
            "failed until the pipeline is configured.",
            exc,
        )
        return None


def _load_handlers(context) -> dict[str, Handler]:
    if context is None:
        return {}
    from multimodal_rag.pipeline.registry import build_handlers

    return build_handlers(context)


def _make_delete_callback(state: StateStore, context):
    """Drop the queue row and any indexed vectors when a source file disappears."""

    def on_delete(path: Path) -> None:
        removed = state.remove_by_path(path)
        repository = context.repository if context is not None else None
        if repository is not None:
            try:
                repository.delete_by_path(str(path))
                repository.reindex()
            except Exception:  # noqa: BLE001
                logger.exception("Vector deletion failed for deleted source %s", path)
        if removed:
            logger.info("Dropped queue row for deleted source %s", path)

    return on_delete


def _build_worker(
    settings: Settings,
    queue: IngestQueue,
    handlers: dict[str, Handler],
    reindex=None,
) -> IngestionWorker:
    return IngestionWorker(
        queue=queue,
        handlers=handlers,
        processing_dir=settings.PROCESSING_DIR,
        processed_dir=settings.PROCESSED_DIR,
        failed_dir=settings.FAILED_DIR,
        worker_count=settings.WORKER_COUNT,
        file_stable_seconds=settings.FILE_STABLE_SECONDS,
        inbox_dir=settings.INBOX_DIR,
        allow_kind=settings.allows_kind,
        reindex=reindex,
    )


def cmd_add(settings: Settings, args: argparse.Namespace) -> int:
    state, queue = _build(settings)
    total = 0
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            total += queue.enqueue_directory(
                path, force=args.force, ignore_extensions=settings.ingest_ignore_extensions
            )
        elif path.is_file():
            if queue.enqueue_path(path, force=args.force) is not None:
                total += 1
        else:
            print(f"skip (not found): {path}", file=sys.stderr)
    print(f"enqueued {total} file(s)")
    return 0


def cmd_scan(settings: Settings, args: argparse.Namespace) -> int:
    state, queue = _build(settings)
    count = queue.enqueue_directory(
        settings.INBOX_DIR,
        recursive=True,
        force=args.force,
        ignore_extensions=settings.ingest_ignore_extensions,
    )
    print(f"scanned inbox, enqueued {count} file(s)")
    return 0


def cmd_watch(settings: Settings, args: argparse.Namespace) -> int:
    state, queue = _build(settings)
    context = _build_context(settings)
    watcher = InboxWatcher(
        settings.INBOX_DIR,
        on_file=lambda path: queue.enqueue_path(path),
        ignore_extensions=settings.ingest_ignore_extensions,
        poll_interval=settings.WATCH_POLL_INTERVAL_SEC,
        on_delete=_make_delete_callback(state, context),
    )
    watcher.start()
    print(f"watching {settings.INBOX_DIR} (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        watcher.stop()
    return 0


def cmd_worker(settings: Settings, args: argparse.Namespace) -> int:
    state, queue = _build(settings)
    context = _build_context(settings)
    reindex = context.repository.reindex if context is not None else None
    worker = _build_worker(settings, queue, _load_handlers(context), reindex=reindex)
    worker.start()
    print(f"worker(s) running: {settings.WORKER_COUNT} (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
    return 0


def cmd_run(settings: Settings, args: argparse.Namespace) -> int:
    state, queue = _build(settings)
    reset = state.reset_stale()
    if reset:
        print(f"reset {reset} stale processing record(s)")
    context = _build_context(settings)
    reindex = context.repository.reindex if context is not None else None
    worker = _build_worker(settings, queue, _load_handlers(context), reindex=reindex)
    worker.start()

    watcher = None
    if settings.INGESTION_MODE is IngestionMode.WATCH:
        watcher = InboxWatcher(
            settings.INBOX_DIR,
            on_file=lambda path: queue.enqueue_path(path),
            ignore_extensions=settings.ingest_ignore_extensions,
            poll_interval=settings.WATCH_POLL_INTERVAL_SEC,
            on_delete=_make_delete_callback(state, context),
        )
        watcher.start()
    else:
        queue.enqueue_directory(
            settings.INBOX_DIR, ignore_extensions=settings.ingest_ignore_extensions
        )

    print("ingestion service running (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if watcher is not None:
            watcher.stop()
        worker.stop()
    return 0


def cmd_status(settings: Settings, args: argparse.Namespace) -> int:
    state, _ = _build(settings)
    stats = state.stats()
    if not stats:
        print("no ingest records")
    for status, count in sorted(stats.items()):
        print(f"{status:12s} {count}")
    if args.failed:
        for record in state.list_by_status(FAILED):
            print(f"  [id={record.id}] {record.path}\n      error: {record.error}")
    return 0


def cmd_retry(settings: Settings, args: argparse.Namespace) -> int:
    state, _ = _build(settings)
    file_id = None if args.id in (None, "all") else int(args.id)
    count = state.retry(file_id)
    print(f"requeued {count} record(s)")
    return 0


def cmd_reset_stale(settings: Settings, args: argparse.Namespace) -> int:
    state, _ = _build(settings)
    count = state.reset_stale()
    print(f"reset {count} stale record(s)")
    return 0


def cmd_reset_index(settings: Settings, args: argparse.Namespace) -> int:
    from multimodal_rag.gui.services import BackendService

    if not args.yes:
        answer = input(
            "Delete ALL indexed records and the queue list, and start fresh? "
            "Source files are kept. [y/N] "
        ).strip().lower()
        if answer not in ("y", "yes"):
            print("aborted")
            return 1
    service = BackendService(settings)
    try:
        result = service.reset_index()
    finally:
        repository = service.repository
        if repository is not None:
            try:
                repository.close()
            except Exception:  # noqa: BLE001
                pass
    print(
        f"index reset: {result['vectors']} vector record(s) and "
        f"{result['queue']} queue entr(ies) deleted; fingerprint restamped"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mrag-ingest", description="RAG ingestion queue CLI")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="enqueue files or directories")
    p_add.add_argument("paths", nargs="+")
    p_add.add_argument("--force", action="store_true", help="re-ingest even if seen")
    p_add.set_defaults(func=cmd_add)

    p_scan = sub.add_parser("scan", help="enqueue everything currently in the inbox")
    p_scan.add_argument("--force", action="store_true")
    p_scan.set_defaults(func=cmd_scan)

    sub.add_parser("watch", help="watch inbox and enqueue (no processing)").set_defaults(func=cmd_watch)

    sub.add_parser("worker", help="process queued items").set_defaults(func=cmd_worker)

    sub.add_parser("run", help="watch + process").set_defaults(func=cmd_run)

    p_status = sub.add_parser("status", help="show queue stats")
    p_status.add_argument("--failed", action="store_true", help="list failed records with errors")
    p_status.set_defaults(func=cmd_status)

    p_retry = sub.add_parser("retry", help="requeue failed records")
    p_retry.add_argument("id", nargs="?", default="all")
    p_retry.set_defaults(func=cmd_retry)

    sub.add_parser("reset-stale", help="requeue records stuck in processing").set_defaults(
        func=cmd_reset_stale
    )

    p_reset = sub.add_parser(
        "reset-index", help="DESTRUCTIVE: delete all vectors + queue rows, restamp fingerprint"
    )
    p_reset.add_argument("--yes", action="store_true", help="skip confirmation prompt")
    p_reset.set_defaults(func=cmd_reset_index)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()
    return args.func(settings, args)


if __name__ == "__main__":
    raise SystemExit(main())
