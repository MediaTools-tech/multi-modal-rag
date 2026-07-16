"""Video ingestion chunker.

Handles:
- Local Mode: Split visual frame sampling (OpenCV @ 1 FPS) + audio transcription (faster-whisper).
- Cloud Mode: Splitting videos into 15-30s chunks via ffmpeg for Gemini indexing.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import ffmpeg
import structlog

# We import cv2 dynamically to avoid headless import issues if missing,
# but we can import normally as we have it in dependencies
import cv2

from deeplens.config import Settings
from deeplens.core.models import FileType, VectorRecord
from deeplens.ingestion.router import FileRouter
from deeplens.ingestion.chunkers.audio import AudioChunker

logger = structlog.get_logger(__name__)


class VideoChunker:
    """Processes video files by separating or segmenting visual and audio tracks."""

    @classmethod
    def chunk(cls, file_path: Path, settings: Settings) -> list[VectorRecord]:
        """Route to local or cloud video chunker based on configuration."""
        from deeplens.config import AppMode
        
        if settings.mode == AppMode.LOCAL:
            return cls.chunk_local(file_path, settings)
        else:
            return cls.chunk_cloud(file_path, settings)

    @classmethod
    def chunk_local(cls, file_path: Path, settings: Settings) -> list[VectorRecord]:
        """Local Mode video chunking.

        1. Extracts and transcribes the audio track (yields text records).
        2. Extracts frames at 1 FPS using OpenCV (yields image records).
        3. Clean up temp WAV/image files.
        """
        logger.info("video_chunker.local.start", file=str(file_path))
        meta = FileRouter.get_file_metadata(file_path)
        records: list[VectorRecord] = []

        temp_audio_path = None
        extracted_frames_dir = None

        try:
            # ──── Step 1: Process Audio ────
            # Extract audio to temp .wav and run AudioChunker on it
            fd, temp_audio_str = tempfile.mkstemp(suffix=".wav", dir=str(settings.temp_dir))
            os.close(fd)
            temp_audio_path = Path(temp_audio_str)

            logger.info("video_chunker.local.extract_audio", target=str(temp_audio_path))
            try:
                (
                    ffmpeg.input(str(file_path))
                    .output(str(temp_audio_path), ar="16000", ac="1", format="wav")
                    .overwrite_output()
                    .run(quiet=True)
                )
                
                # Transcribe using AudioChunker logic
                audio_records = AudioChunker.chunk(temp_audio_path, settings)
                
                # Re-map absolute_path and file_type to indicate they belong to the video
                for r in audio_records:
                    r.absolute_path = meta["absolute_path"]
                    r.filename = meta["filename"]
                    r.parent_directory = meta["parent_directory"]
                    r.file_type = FileType.VIDEO.value  # Mark as video chunk
                    r.content = f"[Audio Transcript] {r.content}"
                    r.file_hash = meta["file_hash"]
                    r.file_modified_at = str(meta["file_modified_at"])
                
                records.extend(audio_records)
                logger.info("video_chunker.local.audio_done", count=len(audio_records))
            except Exception as audio_err:
                logger.error("video_chunker.local.audio_failed", error=str(audio_err))

            # ──── Step 2: Visual Frame Extraction ────
            # Create a temp dir for frame extraction
            extracted_frames_dir = Path(tempfile.mkdtemp(dir=str(settings.temp_dir)))
            logger.info("video_chunker.local.extract_frames", dir=str(extracted_frames_dir))
            
            # Read video with OpenCV
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                raise RuntimeError("Could not open video file via OpenCV.")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 25.0
            
            # Frame sampling interval (1 FPS)
            sample_interval = int(round(fps))
            
            frame_count = 0
            saved_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_interval == 0:
                    timestamp = frame_count / fps
                    frame_name = f"frame_{saved_count:06d}_t_{timestamp:.2f}.jpg"
                    frame_path = extracted_frames_dir / frame_name
                    
                    # Save frame
                    cv2.imwrite(str(frame_path), frame)
                    
                    # Create VectorRecord
                    # Note: local embedding engine (Jina CLIP) will encode this image on insert
                    rec = VectorRecord(
                        # Describe the visual frame containing video path and timestamp
                        content=f"[Visual Frame] Video: {file_path.name} at {timestamp:.2f}s",
                        absolute_path=meta["absolute_path"],
                        filename=meta["filename"],
                        parent_directory=meta["parent_directory"],
                        file_type=FileType.VIDEO.value,
                        mime_type="image/jpeg",
                        chunk_index=saved_count,
                        timestamp_start=float(timestamp),
                        timestamp_end=float(timestamp + 1.0),
                        file_hash=meta["file_hash"],
                        file_modified_at=str(meta["file_modified_at"]),
                        # Store physical frame path in metadata temporarily so the ingestion pipeline
                        # can load and encode the image before saving to DB.
                        metadata_json=f'{{"temp_frame_path": "{str(frame_path)}"}}'
                    )
                    records.append(rec)
                    saved_count += 1
                
                frame_count += 1

            cap.release()
            logger.info("video_chunker.local.visual_done", count=saved_count)

            # ──── Step 3: Check for Subtitles ────
            # Attempt to extract embedded subtitle track via ffmpeg
            try:
                sub_records = cls._extract_embedded_subtitles(file_path, settings, meta)
                records.extend(sub_records)
                logger.info("video_chunker.local.subtitles_done", count=len(sub_records))
            except Exception as sub_err:
                logger.debug("video_chunker.local.subtitles_failed_or_none", error=str(sub_err))

        except Exception as e:
            logger.error("video_chunker.local.failed", file=str(file_path), error=str(e))
        finally:
            # Clean up audio WAV
            if temp_audio_path and temp_audio_path.exists():
                try:
                    temp_audio_path.unlink()
                except Exception:
                    pass

        return records

    @classmethod
    def chunk_cloud(cls, file_path: Path, settings: Settings) -> list[VectorRecord]:
        """Cloud Mode video chunking.

        Slices video into 15–30 second overlapping segments. In Cloud Mode,
        we upload these short clips directly to Gemini for multimodal reasoning.
        """
        logger.info("video_chunker.cloud.start", file=str(file_path))
        meta = FileRouter.get_file_metadata(file_path)
        records: list[VectorRecord] = []

        chunk_sec = settings.video_chunk_seconds
        overlap_sec = settings.video_overlap_seconds

        # Get video duration via ffprobe
        try:
            probe = ffmpeg.probe(str(file_path))
            duration = float(probe["format"]["duration"])
        except Exception as e:
            logger.error("video_chunker.cloud.probe_failed", error=str(e))
            duration = 300.0  # Fallback 5 mins

        # Slice video
        start = 0.0
        chunk_idx = 0
        
        extracted_clips_dir = Path(tempfile.mkdtemp(dir=str(settings.temp_dir)))

        while start < duration:
            end = min(duration, start + chunk_sec)
            clip_name = f"clip_{chunk_idx:04d}_s_{start:.2f}_e_{end:.2f}.mp4"
            clip_path = extracted_clips_dir / clip_name

            logger.info("video_chunker.cloud.slice", start=start, end=end, target=str(clip_path))
            try:
                # ffmpeg slice
                (
                    ffmpeg.input(str(file_path), ss=start, to=end)
                    .output(str(clip_path), c="copy", format="mp4")
                    .overwrite_output()
                    .run(quiet=True)
                )

                # VectorRecord representing this video segment
                rec = VectorRecord(
                    content=f"[Video Segment] {file_path.name} from {start:.1f}s to {end:.1f}s",
                    absolute_path=meta["absolute_path"],
                    filename=meta["filename"],
                    parent_directory=meta["parent_directory"],
                    file_type=FileType.VIDEO.value,
                    mime_type="video/mp4",
                    chunk_index=chunk_idx,
                    timestamp_start=float(start),
                    timestamp_end=float(end),
                    file_hash=meta["file_hash"],
                    file_modified_at=str(meta["file_modified_at"]),
                    # Save path to local slice so Gemini can load it during embedding
                    metadata_json=f'{{"temp_clip_path": "{str(clip_path)}"}}'
                )
                records.append(rec)
                chunk_idx += 1
            except Exception as slice_err:
                logger.error("video_chunker.cloud.slice_failed", start=start, end=end, error=str(slice_err))

            # Move forward with overlap
            start += (chunk_sec - overlap_sec)
            if start >= duration - 1.0:
                break

        return records

    @classmethod
    def _extract_embedded_subtitles(cls, file_path: Path, settings: Settings, meta: dict) -> list[VectorRecord]:
        """Try to extract embedded subtitles using ffmpeg and index them."""
        # Check if subtitle track exists
        try:
            probe = ffmpeg.probe(str(file_path))
            sub_streams = [s for s in probe["streams"] if s["codec_type"] == "subtitle"]
            if not sub_streams:
                return []
        except Exception:
            return []

        records = []
        # Extract first subtitle track as SRT
        fd, temp_srt_str = tempfile.mkstemp(suffix=".srt", dir=str(settings.temp_dir))
        os.close(fd)
        temp_srt_path = Path(temp_srt_str)

        try:
            # Extract subtitle track
            (
                ffmpeg.input(str(file_path))
                .output(str(temp_srt_path), map="0:s:0")
                .overwrite_output()
                .run(quiet=True)
            )

            # Use SubtitleChunker to parse it
            from deeplens.ingestion.chunkers.subtitle import SubtitleChunker
            sub_records = SubtitleChunker.chunk(temp_srt_path, settings)
            
            # Map back to video metadata
            for r in sub_records:
                r.absolute_path = meta["absolute_path"]
                r.filename = meta["filename"]
                r.parent_directory = meta["parent_directory"]
                r.file_type = FileType.VIDEO.value
                r.content = f"[Subtitle] {r.content}"
                r.file_hash = meta["file_hash"]
                r.file_modified_at = str(meta["file_modified_at"])
            records.extend(sub_records)

        except Exception as e:
            logger.debug("video_chunker.subtitles.extract_failed", error=str(e))
        finally:
            if temp_srt_path.exists():
                try:
                    temp_srt_path.unlink()
                except Exception:
                    pass
        return records
