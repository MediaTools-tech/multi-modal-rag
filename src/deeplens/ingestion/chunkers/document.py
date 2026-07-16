"""Document parsing chunker using Docling.

Extracts layout-aware structures (tables, multi-column text, headers) and chunks them
with structure boundaries intact. Falls back to plaintext for standard text/markdown files.
"""

from __future__ import annotations

from pathlib import Path
import structlog

from deeplens.config import Settings
from deeplens.core.models import FileType, VectorRecord
from deeplens.ingestion.router import FileRouter

logger = structlog.get_logger(__name__)


class DocumentChunker:
    """Chunks documents into layout-preserving textual passages."""

    @staticmethod
    def chunk(file_path: Path, settings: Settings) -> list[VectorRecord]:
        """Parse doc into structure-preserving chunks (~500 words)."""
        logger.info("document_chunker.start", file=str(file_path))

        meta = FileRouter.get_file_metadata(file_path)
        suffix = file_path.suffix.lower()
        records: list[VectorRecord] = []

        # Read content based on format
        try:
            content_md = ""
            
            # For simple files (.txt, .md, .csv), read directly to avoid Docling overhead
            if suffix in (".txt", ".md", ".csv", ".json", ".xml"):
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    content_md = f.read()
            else:
                # Use Docling layout parser
                from docling.document_converter import DocumentConverter
                
                converter = DocumentConverter()
                result = converter.convert(str(file_path))
                # Export to clean Markdown representation
                content_md = result.document.export_to_markdown()

            # Perform structure-aware chunking on content_md
            chunks = DocumentChunker._split_markdown(content_md, settings.chunk_size, settings.chunk_overlap)
            
            for i, chunk_text in enumerate(chunks):
                rec = VectorRecord(
                    content=chunk_text,
                    absolute_path=meta["absolute_path"],
                    filename=meta["filename"],
                    parent_directory=meta["parent_directory"],
                    file_type=FileType.DOCUMENT.value,
                    mime_type=meta["mime_type"],
                    chunk_index=i,
                    total_chunks=len(chunks),
                    file_hash=meta["file_hash"],
                    file_modified_at=str(meta["file_modified_at"]),
                )
                records.append(rec)

            logger.info("document_chunker.completed", file=str(file_path), chunks=len(records))

        except Exception as e:
            logger.error("document_chunker.failed", file=str(file_path), error=str(e))
            # Fallback to direct plaintext read
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                chunks = DocumentChunker._split_plaintext(content, settings.chunk_size, settings.chunk_overlap)
                for i, chunk_text in enumerate(chunks):
                    rec = VectorRecord(
                        content=chunk_text,
                        absolute_path=meta["absolute_path"],
                        filename=meta["filename"],
                        parent_directory=meta["parent_directory"],
                        file_type=FileType.DOCUMENT.value,
                        mime_type=meta["mime_type"],
                        chunk_index=i,
                        total_chunks=len(chunks),
                        file_hash=meta["file_hash"],
                        file_modified_at=str(meta["file_modified_at"]),
                    )
                    records.append(rec)
            except Exception as inner_e:
                logger.error("document_chunker.fallback_failed", file=str(file_path), error=str(inner_e))

        return records

    @staticmethod
    def _split_markdown(text: str, chunk_size: int, overlap: float) -> list[str]:
        """Split markdown text, attempting to preserve tables, list and paragraph blocks."""
        # Simple implementation splitting by paragraphs and tables
        # ensuring no mid-table splits where possible.
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_len = 0
        overlap_size = int(chunk_size * overlap)

        for para in paragraphs:
            # Word count
            para_len = len(para.split())
            
            # If a single paragraph is larger than chunk_size, split it by sentences
            if para_len > chunk_size:
                # Flush current
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                
                # Split large paragraph
                sentences = para.replace(". ", ".\n").split("\n")
                sub_chunk = []
                sub_len = 0
                for sent in sentences:
                    sent_len = len(sent.split())
                    if sub_len + sent_len > chunk_size:
                        chunks.append(" ".join(sub_chunk))
                        # Keep overlap sentences
                        sub_chunk = sub_chunk[-max(1, int(len(sub_chunk)*overlap)):]
                        sub_len = sum(len(s.split()) for s in sub_chunk)
                    sub_chunk.append(sent)
                    sub_len += sent_len
                if sub_chunk:
                    chunks.append(" ".join(sub_chunk))
                continue

            if current_len + para_len > chunk_size:
                chunks.append("\n\n".join(current_chunk))
                
                # Calculate overlap: keep some paragraphs
                overlap_paras = []
                overlap_len = 0
                for p in reversed(current_chunk):
                    p_len = len(p.split())
                    if overlap_len + p_len > overlap_size:
                        break
                    overlap_paras.insert(0, p)
                    overlap_len += p_len
                
                current_chunk = overlap_paras
                current_len = overlap_len

            current_chunk.append(para)
            current_len += para_len

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return [c.strip() for c in chunks if c.strip()]

    @staticmethod
    def _split_plaintext(text: str, chunk_size: int, overlap: float) -> list[str]:
        """Split plain text by word counts."""
        words = text.split()
        chunks = []
        step = int(chunk_size * (1 - overlap))
        for i in range(0, len(words), step):
            chunk_words = words[i : i + chunk_size]
            chunks.append(" ".join(chunk_words))
            if i + chunk_size >= len(words):
                break
        return chunks
