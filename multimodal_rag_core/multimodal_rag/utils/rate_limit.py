from __future__ import annotations

import threading
import time


class RateLimiter:
    """Simple client-side pacing so we never exceed a per-minute quota."""

    def __init__(self, max_per_minute: int) -> None:
        self.min_interval = 60.0 / max_per_minute if max_per_minute > 0 else 0.0
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            elapsed = time.monotonic() - self._last
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last = time.monotonic()
