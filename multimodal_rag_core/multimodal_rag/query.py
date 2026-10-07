from __future__ import annotations

import argparse
import json
import logging

from multimodal_rag.config import SearchMode, get_settings
from multimodal_rag.core.models import SearchResult
from multimodal_rag.database import get_repository
from multimodal_rag.pipeline.embedding import get_embedding_engine
from multimodal_rag.search.hybrid import hybrid_search
from multimodal_rag.search.reranker import CrossEncoderReranker
from multimodal_rag.utils.api_clients import generate_rag_answer
from multimodal_rag.utils.token_tracker import TokenTracker

logger = logging.getLogger("mrag-query")


def run_search(
    settings,
    repo,
    embedder,
    query: str,
    mode: str,
    top_k: int,
    folder: str | None,
    rerank: bool,
):
    query_vector = embedder.embed_query(query)
    if mode == SearchMode.CHUNK.value:
        results = repo.search(
            query_vector, top_k=top_k, record_types=["chunk"], folder_prefix=folder
        )
        return results, []
    if mode == SearchMode.SUMMARY.value:
        results = repo.search(
            query_vector, top_k=top_k, record_types=["summary"], folder_prefix=folder
        )
        return results, []
    reranker = None
    if rerank:
        reranker = CrossEncoderReranker(
            model_name=settings.RERANKER_MODEL,
            device=settings.device,
            top_n=settings.RERANKER_TOP_N,
        )
    return hybrid_search(
        repo,
        query,
        query_vector,
        top_k=top_k,
        rrf_k=settings.HYBRID_RRF_K,
        reranker=reranker,
        rerank=rerank,
        rerank_top_n=settings.RERANKER_TOP_N,
        folder_prefix=folder,
    )


def run_auto_search(
    settings,
    repo,
    embedder,
    query: str,
    top_k: int,
    folder: str | None,
    rerank: bool,
):
    """AUTO retrieval: chunk first, empty results fall back to hybrid.

    L1 of the AUTO strategy — costs no LLM call. Returns
    (results, groups, mode_used).
    """
    results, groups = run_search(
        settings, repo, embedder, query, SearchMode.CHUNK.value, top_k, folder, False
    )
    if results:
        return results, groups, SearchMode.CHUNK.value
    logger.info("auto: chunk empty, falling back to hybrid (no answer spent)")
    results, groups = run_search(
        settings, repo, embedder, query, SearchMode.HYBRID.value, top_k, folder, rerank
    )
    return results, groups, SearchMode.HYBRID.value


def run_auto_query(
    settings,
    repo,
    embedder,
    query: str,
    top_k: int,
    folder: str | None,
    rerank: bool,
    answer_fn=None,
):
    """AUTO end-to-end: L1 (empty chunk -> hybrid) plus L2 (abstaining chunk
    answer -> one hybrid retry). answer_fn(results) -> str|None; None skips
    answering. Returns (results, groups, answer, mode_used).
    """
    from multimodal_rag.utils.api_clients import is_abstention_answer

    results, groups, used = run_auto_search(
        settings, repo, embedder, query, top_k, folder, rerank
    )
    answer = ""
    if answer_fn is not None and results:
        answer = answer_fn(results) or ""
        if used == SearchMode.CHUNK.value and answer and is_abstention_answer(answer):
            logger.info("auto: chunk answer abstained, retrying once with hybrid")
            results, groups = run_search(
                settings, repo, embedder, query, SearchMode.HYBRID.value, top_k, folder, rerank
            )
            answer = ""
            if results:
                answer = answer_fn(results) or ""
            used = SearchMode.HYBRID.value
    return results, groups, answer, used


def print_results(results: list[SearchResult], groups, as_json: bool) -> None:
    if as_json:
        print(
            json.dumps(
                {
                    "results": [
                        {
                            "score": result.score,
                            "record_type": result.record.record_type,
                            "source_path": result.record.source_path,
                            "filename": result.record.filename,
                            "timestamp_start": result.record.timestamp_start,
                            "timestamp_end": result.record.timestamp_end,
                            "content": result.record.content,
                        }
                        for result in results
                    ],
                    "files": [
                        {
                            "source_path": group.source_path,
                            "filename": group.filename,
                            "summary": group.summary,
                            "score": group.best_score,
                        }
                        for group in groups
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if not results:
        print("No results.")
        return
    for index, result in enumerate(results, start=1):
        record = result.record
        stamp = f" @ {record.timestamp_start:.2f}s" if record.timestamp_start is not None else ""
        snippet = record.content.replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        print(f"{index:>2}. [{result.score:.3f}] {record.record_type}  {record.filename}{stamp}")
        print(f"    {snippet}")
    if groups:
        print("\nMatched files:")
        for group in groups:
            print(f"  - {group.filename}: {(group.summary or '')[:100]}")


def synthesize_answer(settings, tracker, question: str, results: list[SearchResult]):
    if not results:
        return None
    try:
        answer = generate_rag_answer(settings, tracker, question, results)
    except Exception as exc:  # noqa: BLE001
        print(f"\n[answer failed: {type(exc).__name__}: {exc}]")
        return None
    if answer is None:
        print("\n[answer skipped: no TEXT_LLM_PROVIDER/API key configured]")
    return answer


def _handle(settings, repo, embedder, tracker, args, query: str) -> int:
    before = tracker.snapshot()
    answer = ""
    if args.mode == SearchMode.AUTO.value:
        answer_fn = None
        if args.answer:
            answer_fn = lambda res: synthesize_answer(settings, tracker, query, res)
        results, groups, answer, used = run_auto_query(
            settings, repo, embedder, query, args.top_k, args.folder, args.rerank, answer_fn
        )
        print(f"[auto] retrieval mode used: {used}")
    else:
        results, groups = run_search(
            settings, repo, embedder, query, args.mode, args.top_k, args.folder, args.rerank
        )
        if args.answer and not args.json:
            answer = synthesize_answer(settings, tracker, query, results) or ""
    print_results(results, groups, args.json)
    if answer and not args.json:
        print("\nAnswer:\n" + answer)
    usage = TokenTracker.delta(before, tracker.snapshot())
    print(f"\n[tokens] request: {TokenTracker.format_usage(usage)}")
    print(f"[tokens] session: {TokenTracker.format_usage(tracker.snapshot())}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mrag-query", description="Query the RAG index")
    parser.add_argument("query", nargs="?", help="question / search text")
    parser.add_argument("-i", "--interactive", action="store_true", help="interactive REPL")
    parser.add_argument(
        "--mode", choices=["auto", "chunk", "summary", "hybrid"], default=None
    )
    parser.add_argument("-k", "--top-k", type=int, default=None)
    parser.add_argument("--folder", default=None, help="restrict to a source_path prefix")
    parser.add_argument("--answer", action="store_true", help="synthesize an answer via the text LLM")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--rerank", dest="rerank", action="store_true", default=None)
    parser.add_argument("--no-rerank", dest="rerank", action="store_false")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()
    if args.mode is None:
        args.mode = settings.SEARCH_MODE.value
    if args.top_k is None:
        args.top_k = settings.SEARCH_TOP_K
    if args.rerank is None:
        args.rerank = settings.active_reranker != "NONE"

    tracker = TokenTracker()
    repo = get_repository(settings)
    repo.initialize()
    embedder = get_embedding_engine(settings, tracker)

    try:
        if args.interactive:
            print("Interactive query mode. Type 'exit' to quit.")
            while True:
                try:
                    query = input("query> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not query:
                    continue
                if query.lower() in {"exit", "quit", ":q"}:
                    break
                _handle(settings, repo, embedder, tracker, args, query)
            return 0
        if not args.query:
            build_parser().print_help()
            return 2
        return _handle(settings, repo, embedder, tracker, args, args.query)
    finally:
        repo.close()


if __name__ == "__main__":
    raise SystemExit(main())
