"""Subtitle parser and chunker for SRT/VTT files."""

from __future__ import annotations

from pathlib import Path
import re
import structlog

from deeplens.config import Settings
from deeplens.core.models import FileType, VectorRecord
from deeplens.ingestion.router import FileRouter

logger = structlog.get_logger(__name__)


class SubtitleChunker:
    """Parses subtitle tracks and chunks them by time windows."""

    @staticmethod
    def chunk(file_path: Path, settings: Settings) -> list[VectorRecord]:
        """Parse standalone .srt or .vtt subtitle file and group entries by time."""
        logger.info("subtitle_chunker.start", file=str(file_path))

        meta = FileRouter.get_file_metadata(file_path)
        suffix = file_path.suffix.lower()
        records: list[VectorRecord] = []

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            entries = []
            if suffix == ".srt":
                entries = SubtitleChunker._parse_srt(content)
            elif suffix == ".vtt":
                entries = SubtitleChunker._parse_vtt(content)
            else:
                logger.warn("subtitle_chunker.unsupported_format", format=suffix)
                return []

            if not entries:
                return []

            # Group subtitle entries into time-window chunks (e.g. 30 second blocks)
            window_size_sec = 30.0
            overlap_sec = 5.0
            
            chunk_idx = 0
            current_chunk_entries = []
            chunk_start = entries[0]["start"]
            chunk_end = chunk_start + window_size_sec

            # Helper to create a record from grouped entries
            def _create_record(group: list[dict], index: int) -> VectorRecord:
                combined_text = " ".join(e["text"] for e in group)
                start_t = group[0]["start"]
                end_t = group[-1]["end"]
                return VectorRecord(
                    content=combined_text,
                    absolute_path=meta["absolute_path"],
                    filename=meta["filename"],
                    parent_directory=meta["parent_directory"],
                    file_type=FileType.SUBTITLE.value,
                    mime_type=meta["mime_type"],
                    chunk_index=index,
                    total_chunks=0,  # Updated dynamically later
                    timestamp_start=float(start_t),
                    timestamp_end=float(end_t),
                    file_hash=meta["file_hash"],
                    file_modified_at=str(meta["file_modified_at"]),
                )

            for entry in entries:
                if entry["start"] > chunk_end:
                    # Flush
                    if current_chunk_entries:
                        records.append(_create_record(current_chunk_entries, chunk_idx))
                        chunk_idx += 1
                    
                    # Setup next chunk with overlap
                    # Retain entries starting in overlap region
                    overlap_start = chunk_end - overlap_sec
                    current_chunk_entries = [e for e in current_chunk_entries if e["start"] >= overlap_start]
                    current_chunk_entries.append(entry)
                    chunk_start = current_chunk_entries[0]["start"] if current_chunk_entries else entry["start"]
                    chunk_end = chunk_start + window_size_sec
                else:
                    current_chunk_entries.append(entry)

            if current_chunk_entries:
                records.append(_create_record(current_chunk_entries, chunk_idx))

            # Update total chunks
            total = len(records)
            for r in records:
                r.total_chunks = total

            logger.info("subtitle_chunker.completed", file=str(file_path), chunks=total)

        except Exception as e:
            logger.error("subtitle_chunker.failed", file=str(file_path), error=str(e))

        return records

    @staticmethod
    def _parse_time(time_str: str) -> float:
        """Convert HH:MM:SS,mmm or MM:SS.mmm to seconds float."""
        time_str = time_str.replace(",", ".")
        parts = re.split(r"[:.]", time_str)
        if len(parts) == 4:  # HH:MM:SS.mmm
            h, m, s, ms = parts
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
        elif len(parts) == 3:  # MM:SS.mmm
            m, s, ms = parts
            return int(m) * 60 + int(s) + int(ms) / 1000.0
        return 0.0

    @classmethod
    def _parse_srt(cls, content: str) -> list[dict]:
        """SRT Parser."""
        entries = []
        # Split by blank lines
        blocks = re.split(r"\n\s*\n", content.strip())
        for block in blocks:
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            if len(lines) >= 3:
                # Line 0: Index, Line 1: Timestamps, Line 2+: Subtitle text
                time_match = re.match(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})", lines[1])
                if time_match:
                    start_t = cls._parse_time(time_match.group(1))
                    end_t = cls._parse_time(time_match.group(2))
                    text = " ".join(lines[2:])
                    entries.append({"start": start_t, "end": end_t, "text": text})
        return entries

    @classmethod
    def _parse_vtt(cls, content: str) -> list[dict]:
        """VTT Parser."""
        entries = []
        # Remove WEBVTT header
        content = re.sub(r"^WEBVTT.*?\n", "", content, flags=re.IGNORECASE)
        blocks = re.split(r"\n\s*\n", content.strip())
        for block in blocks:
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            if len(lines) >= 2:
                # Line 0 might be index/identifier, or timestamps
                t_idx = 0
                time_match = re.match(r"(\d{2}:\d{2}.*?)\s*-->\s*(\d{2}:\d{2}.*?)", lines[0])
                if not time_match and len(lines) >= 3:
                    time_match = re.match(r"(\d{2}:\d{2}.*?)\s*-->\s*(\d{2}:\d{2}.*?)", lines[1])
                    t_idx = 1
                
                if time_match:
                    start_t = cls._parse_time(time_match.group(1).split()[0])
                    end_t = cls._parse_time(time_match.group(2).split()[0])
                    text = " ".join(lines[t_idx+1:])
                    entries.append({"start": start_t, "end": end_t, "text": text})
        return entries
