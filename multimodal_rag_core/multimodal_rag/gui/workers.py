from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from multimodal_rag.gui.services import BackendService


class InitWorker(QThread):
    ready = Signal(str)  # empty string on success, error message otherwise

    def __init__(self, service: BackendService) -> None:
        super().__init__()
        self.service = service

    def run(self) -> None:
        try:
            self.service.initialize()
            self.service.start_worker()
            self.ready.emit("")
        except Exception as exc:  # noqa: BLE001
            self.ready.emit(f"{type(exc).__name__}: {exc}")


class FfmpegDownloadWorker(QThread):
    # downloaded bytes, total bytes (0 if unknown), message
    progress = Signal(int, int, str)
    # success, installed path or error/cancel marker
    done = Signal(bool, str)

    def run(self) -> None:
        from multimodal_rag.utils.ffmpeg_tool import download_ffmpeg

        def report(downloaded: int, total: int, message: str) -> None:
            if self.isInterruptionRequested():
                raise RuntimeError("cancelled")
            self.progress.emit(downloaded, total, message)

        try:
            path = download_ffmpeg(progress=report)
            self.done.emit(True, str(path))
        except RuntimeError as exc:
            if "cancelled" in str(exc):
                self.done.emit(False, "cancelled")
            else:
                self.done.emit(False, f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            self.done.emit(False, f"{type(exc).__name__}: {exc}")


class SearchWorker(QThread):
    # results, groups, answer, error
    completed = Signal(object, object, str, str)

    def __init__(
        self,
        service: BackendService,
        query: str,
        mode: str,
        top_k: int,
        with_answer: bool = False,
    ) -> None:
        super().__init__()
        self.service = service
        self.query = query
        self.mode = mode
        self.top_k = top_k
        self.with_answer = with_answer

    def run(self) -> None:
        from multimodal_rag.utils.token_tracker import TokenTracker

        before = self.service.tracker.snapshot()
        try:
            results, groups = self.service.search(self.query, self.mode, self.top_k)
            answer = ""
            if self.with_answer and results:
                answer = self.service.answer(self.query, results) or ""
            usage = TokenTracker.delta(before, self.service.tracker.snapshot())
            self.service.last_request_usage = usage
            print(f"[tokens] last request: {TokenTracker.format_usage(usage)}", flush=True)
            print(
                f"[tokens] session: {TokenTracker.format_usage(self.service.tracker.snapshot())}",
                flush=True,
            )
            self.completed.emit(results, groups, answer, "")
        except Exception as exc:  # noqa: BLE001
            self.completed.emit([], [], "", f"{type(exc).__name__}: {exc}")
