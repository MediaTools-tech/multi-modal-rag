from __future__ import annotations

import logging
import re
from pathlib import Path

from multimodal_rag.config import Settings
from multimodal_rag.pipeline.chunking import Chunk, text_to_chunks

logger = logging.getLogger(__name__)

DOCLING_SUFFIXES = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".md"}

# Warn-once flag: a missing Docling install otherwise degrades silently
# (per-file debug logs are invisible at the default INFO level).
_DOCLING_MISSING_WARNED = False


def parse_document(path: Path, settings: Settings) -> list[Chunk]:
    text = _extract_text(path, settings)
    if not text or not text.strip():
        return []
    return text_to_chunks(
        text,
        settings.DOC_CHUNK_SIZE,
        settings.DOC_CHUNK_OVERLAP,
        settings=settings,
    )


def _extract_text(path: Path, settings: Settings) -> str:
    if settings.DOC_USE_OCR and path.suffix.lower() in DOCLING_SUFFIXES:
        markdown = _docling_markdown(path)
        if markdown:
            return markdown
    return _fallback_text(path)


def _docling_markdown(path: Path) -> str:
    global _DOCLING_MISSING_WARNED
    try:
        from docling.document_converter import DocumentConverter
    except Exception:
        if not _DOCLING_MISSING_WARNED:
            _DOCLING_MISSING_WARNED = True
            logger.warning(
                "Docling is not installed, so DOC_USE_OCR has no effect: "
                "scanned/image-only PDFs will extract no text. "
                "Install it with `pip install docling` (pulls torch + OCR models)."
            )
        return ""
    try:
        result = DocumentConverter().convert(str(path))
        return result.document.export_to_markdown()
    except Exception as exc:
        logger.warning("Docling failed for %s: %s", path, exc)
        return ""


def _fallback_text(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix in {".txt", ".md", ".csv", ".json", ".log", ".rst", ".xml"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix in {".html", ".htm"}:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            return re.sub(r"<[^>]+>", " ", raw)
        if suffix == ".pdf":
            return _pdf_text(path)
        if suffix == ".docx":
            import docx

            document = docx.Document(str(path))
            return "\n".join(paragraph.text for paragraph in document.paragraphs)
        if suffix in {".doc", ".rtf"}:
            # Legacy Word (.doc is OLE binary or RTF-in-disguise) and .rtf
            # have no Docling/python-docx reader. Raw-decoding them leaks RTF
            # control words and \pict image hex blobs into the index.
            return _legacy_word_text(path)
        if suffix in {".xlsx", ".xls"}:
            return _spreadsheet_text(path)
    except Exception as exc:
        logger.warning("Fallback parse failed for %s: %s", path, exc)
    try:
        data = path.read_bytes()
    except Exception:
        return ""
    if _looks_binary(data):
        # .ppt, legacy .xls that openpyxl rejects, unknown binaries, ...:
        # index extracted text runs, not raw bytes.
        return _strings_text(data)
    return data.decode("utf-8", errors="ignore")


def _pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        from PyPDF2 import PdfReader  # type: ignore
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _looks_binary(data: bytes) -> bool:
    """True when bytes are not plain text (null bytes or mostly non-printable)."""
    if not data:
        return False
    if b"\x00" in data:
        return True
    sample = data[:8192]
    printable = sum(1 for byte in sample if byte in b"\t\n\r" or 32 <= byte < 127)
    return printable / len(sample) < 0.70


def _strings_text(data: bytes) -> str:
    """Best-effort text from a binary file (legacy OLE .doc, .ppt, ...).

    Picks up ASCII runs and UTF-16LE runs (how OLE Word stores text) in file
    order. Noisy but searchable — strictly better than indexing raw bytes.
    """
    parts: list[tuple[int, str]] = []
    for match in re.finditer(rb"[\x20-\x7e\t]{4,}", data):
        parts.append((match.start(), match.group().decode("ascii")))
    for match in re.finditer(rb"(?:[\x20-\x7e]\x00){4,}", data):
        try:
            text = match.group().decode("utf-16-le")
        except UnicodeDecodeError:
            continue
        parts.append((match.start(), text))
    parts.sort(key=lambda item: item[0])
    return "\n".join(text for _, text in parts)


def _legacy_word_text(path: Path) -> str:
    """Text from .doc/.rtf without a real Word reader."""
    try:
        data = path.read_bytes()
    except Exception:
        return ""
    stripped = data.lstrip(b"\xef\xbb\xbf \t\r\n")
    if stripped.startswith(b"{\\rtf"):
        return _strip_rtf(stripped.decode("utf-8", errors="ignore"))
    if _looks_binary(data):
        return _strings_text(data)
    return data.decode("utf-8", errors="ignore")


def _strip_rtf(raw: str) -> str:
    """Plain text from RTF markup.

    Destination groups that carry no readable text (images, font/color
    tables, metadata) are skipped whole — notably {\\pict ...} image hex
    blobs, which otherwise index megabytes of hex. Handles \'hh and \\uN
    escapes, \\par/\\line breaks, and literal \\{ \\} \\\\.
    """
    skip_destinations = {"pict", "fonttbl", "colortbl", "stylesheet", "info", "xmlnstbl"}
    out: list[str] = []
    i, n = 0, len(raw)
    # Stack of "this group (or an ancestor) is skipped".
    skip_stack: list[bool] = [False]
    while i < n:
        char = raw[i]
        if char == "{":
            skip_stack.append(skip_stack[-1])
            i += 1
        elif char == "}":
            if len(skip_stack) > 1:
                skip_stack.pop()
            i += 1
        elif char == "\\" and i + 1 < n:
            nxt = raw[i + 1]
            if nxt in "{}\\":
                if not skip_stack[-1]:
                    out.append(nxt)
                i += 2
            elif nxt == "'":
                hexpart = raw[i + 2 : i + 4]
                try:
                    if not skip_stack[-1]:
                        out.append(bytes.fromhex(hexpart).decode("cp1252"))
                except ValueError:
                    pass
                i += 4
            elif nxt == "~":
                if not skip_stack[-1]:
                    out.append(" ")
                i += 2
            elif nxt in ("-", "_"):
                i += 2
            elif nxt == "*":
                # Ignorable destination: skip the rest of this group.
                skip_stack[-1] = True
                i += 2
            elif nxt.isalpha():
                j = i + 1
                while j < n and raw[j].isalpha():
                    j += 1
                word = raw[i + 1 : j]
                k = j
                negative = False
                if k < n and raw[k] == "-":
                    negative = True
                    k += 1
                num = ""
                while k < n and raw[k].isdigit():
                    num += raw[k]
                    k += 1
                if k < n and raw[k] == " ":
                    k += 1  # space after a control word is a delimiter, not text
                if word in ("par", "line", "row"):
                    if not skip_stack[-1]:
                        out.append("\n")
                elif word == "tab":
                    if not skip_stack[-1]:
                        out.append("\t")
                elif word == "u" and num:
                    if not skip_stack[-1]:
                        try:
                            code = -int(num) if negative else int(num)
                            out.append(chr(code % 65536))
                        except (ValueError, OverflowError):
                            pass
                    # Skip the single fallback char following \uN.
                    if k < n and raw[k] not in ("\\", "{", "}"):
                        k += 1
                elif word in skip_destinations:
                    skip_stack[-1] = True
                i = k
            else:
                i += 2  # stray backslash + char: drop both
        else:
            if not skip_stack[-1] and char not in ("\r",):
                out.append(char)
            i += 1
    text = "".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _spreadsheet_text(path: Path) -> str:
    import openpyxl

    workbook = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    lines: list[str] = []
    for sheet in workbook.worksheets:
        for row in sheet.iter_rows(values_only=True):
            lines.append(" | ".join("" if cell is None else str(cell) for cell in row))
    return "\n".join(lines)
