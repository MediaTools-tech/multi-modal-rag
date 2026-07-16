"""Audio ingestion chunker.

Extracts transcripts using faster-whisper (INT8 quantized CPU mode) with timestamp alignment.
Converts input audio to standard WAV format via ffmpeg-python if necessary.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import ffmpeg
import structlog

from deeplens.config import Settings
from deeplens.core.models import FileType, VectorRecord
from deeplens.ingestion.router import FileRouter

logger = structlog.get_logger(__name__)


class AudioChunker:
    """Transcribes audio files into timestamped text chunks."""

    @staticmethod
    def chunk(file_path: Path, settings: Settings) -> list[VectorRecord]:
        """Convert to WAV, transcribe with faster-whisper, and yield VectorRecords."""
        logger.info("audio_chunker.start", file=str(file_path))

        meta = FileRouter.get_file_metadata(file_path)
        records: list[VectorRecord] = []
        
        # Temp dir for WAV conversion
        temp_wav_path = None

        try:
            # Step 1: Convert to WAV if not already a WAV file (or convert to normalize format)
            # Standard WAV: mono, 16kHz
            suffix = file_path.suffix.lower()
            if suffix == ".wav":
                temp_wav_path = file_path
            else:
                fd, temp_wav_str = tempfile.mkstemp(suffix=".wav", dir=str(settings.temp_dir))
                os.close(fd)
                temp_wav_path = Path(temp_wav_str)

                logger.info("audio_chunker.convert_wav", source=str(file_path), target=str(temp_wav_path))
                (
                    ffmpeg.input(str(file_path))
                    .output(str(temp_wav_path), ar="16000", ac="1", format="wav")
                    .overwrite_output()
                    .run(quiet=True)
                )

            # Step 2: Transcribe via faster-whisper
            from faster_whisper import WhisperModel
            
            # Load model (CPU, INT8 quantization)
            # Use model size from settings
            model_size = settings.whisper_model_size
            logger.info("audio_chunker.load_whisper", size=model_size)
            
            model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
                download_root=str(settings.data_dir / "cache" / "whisper")
            )

            # Transcribe
            # beam_size = 5 by default
            segments, info = model.transcribe(str(temp_wav_path), beam_size=5)
            
            logger.info(
                "audio_chunker.transcribing",
                language=info.language,
                duration=info.duration
            )

            # Collect transcribed segments
            raw_segments = list(segments)
            
            # Map segments to VectorRecords
            for i, seg in enumerate(raw_segments):
                rec = VectorRecord(
                    content=seg.text.strip(),
                    absolute_path=meta["absolute_path"],
                    filename=meta["filename"],
                    parent_directory=meta["parent_directory"],
                    file_type=FileType.AUDIO.value,
                    mime_type=meta["mime_type"],
                    chunk_index=i,
                    total_chunks=len(raw_segments),
                    timestamp_start=float(seg.start),
                    timestamp_end=float(seg.end),
                    file_hash=meta["file_hash"],
                    file_modified_at=str(meta["file_modified_at"]),
                )
                records.append(rec)

            logger.info("audio_chunker.completed", file=str(file_path), segments=len(records))

        except Exception as e:
            logger.error("audio_chunker.failed", file=str(file_path), error=str(e))
        finally:
            # Clean up temporary WAV if created
            if temp_wav_path and temp_wav_path != file_path and temp_wav_path.exists():
                try:
                    temp_wav_path.unlink()
                    logger.info("audio_chunker.cleanup_temp_wav", path=str(temp_wav_path))
                except Exception as cleanup_err:
                    logger.warn("audio_chunker.cleanup_temp_wav_failed", error=str(cleanup_err))

        return records
