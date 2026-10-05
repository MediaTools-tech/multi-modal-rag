from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from multimodal_rag.config import Settings
from multimodal_rag.utils.ffmpeg_tool import find_ffmpeg

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    timestamp: float
    jpeg: bytes


def is_available() -> bool:
    try:
        import cv2  # noqa: F401

        return True
    except Exception:
        return False


def extract_keyframes(path: Path, settings: Settings) -> list[Frame]:
    """Keyframe sampler honoring MM_FRAME_STRATEGY (FPS / SCENE / HYBRID)."""
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    strategy = settings.MM_FRAME_STRATEGY.value
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    sample_fps = max(1, settings.MM_FRAME_SAMPLING_FPS)
    step = max(1, int(round(fps / sample_fps)))
    min_interval = settings.MM_SCENE_MIN_INTERVAL_SEC
    max_per_min = max(1, settings.MM_MAX_KEYFRAMES_PER_MIN)
    threshold = settings.MM_SCENE_THRESHOLD
    dedup_threshold = max(0, settings.MM_DEDUP_HASH_THRESHOLD)

    frames: list[Frame] = []
    previous_hist = None
    previous_hash: int | None = None
    last_kept = -1e9
    minute_bucket = -1
    minute_count = 0
    index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step != 0:
                index += 1
                continue

            timestamp = index / fps
            hist = _histogram(frame, cv2, np)
            diff = (
                _histogram_distance(previous_hist, hist, cv2)
                if previous_hist is not None
                else 1.0
            )
            previous_hist = hist

            minute = int(timestamp // 60)
            if minute != minute_bucket:
                minute_bucket = minute
                minute_count = 0

            scene_hit = diff >= threshold and (timestamp - last_kept) >= 1.0
            floor_hit = (timestamp - last_kept) >= min_interval
            if strategy == "FPS":
                keep = True
            elif strategy == "SCENE":
                keep = scene_hit
            else:  # HYBRID
                keep = scene_hit or floor_hit

            if keep and scene_hit and previous_hash is not None:
                frame_hash = _average_hash(frame, cv2, np)
                if _hamming(frame_hash, previous_hash) <= dedup_threshold:
                    keep = False  # near-duplicate caused by motion, not a new shot

            if keep and minute_count < max_per_min:
                frames.append(Frame(timestamp=timestamp, jpeg=_encode(frame, settings, cv2)))
                previous_hash = _average_hash(frame, cv2, np)
                last_kept = timestamp
                minute_count += 1
            index += 1
    finally:
        capture.release()

    logger.info("Extracted %d keyframes from %s (%s)", len(frames), path.name, strategy)
    return frames


def extract_audio_track(path: Path, settings: Settings) -> Path | None:
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        return None
    target = Path(tempfile.mkdtemp()) / "audio.wav"
    proc = subprocess.run(
        [
            ffmpeg, "-y", "-i", str(path),
            "-ar", "16000", "-ac", "1", "-vn", str(target),
        ],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not target.exists():
        logger.warning("ffmpeg audio extraction failed for %s", path)
        return None
    return target


def _histogram(frame, cv2, np):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram, 0, 1, cv2.NORM_MINMAX)
    return histogram


def _histogram_distance(previous, current, cv2) -> float:
    return float(cv2.compareHist(previous, current, cv2.HISTCMP_BHATTACHARYYA))


def _average_hash(frame, cv2, np) -> int:
    """64-bit average hash used to drop near-duplicate frames."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    bits = small > small.mean()
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    return value


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _encode(frame, settings: Settings, cv2) -> bytes:
    working = frame
    if settings.MM_RESIZE_FRAME:
        height, width = frame.shape[:2]
        target = settings.MM_TARGET_HEIGHT
        if height > target and height > 0:
            scale = target / height
            working = cv2.resize(
                frame, (max(1, int(width * scale)), target), interpolation=cv2.INTER_AREA
            )
    ok, buffer = cv2.imencode(
        ".jpg", working, [int(cv2.IMWRITE_JPEG_QUALITY), settings.MM_JPEG_QUALITY]
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes()
