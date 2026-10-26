"""Bounded, session-owned document text. Never execute or extract files to disk."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import re
from html.parser import HTMLParser
from pathlib import PurePath
from threading import RLock
from time import monotonic
from typing import Literal
from uuid import UUID, uuid4
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from pydantic import Field

from gene_evidence_agent.schemas import StrictModel

MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_UPLOAD_BODY = (MAX_FILE_BYTES * 4 // 3) + 8192
MAX_TEXT_CHARS = 32_000
EXTENSIONS = {".txt", ".md", ".csv", ".tsv", ".json", ".html", ".htm", ".xml",
              ".rtf", ".pdf", ".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp"}


class AttachmentError(ValueError):
    pass


class UploadRequest(StrictModel):
    channel: Literal["education", "research", "workflow"]
    session_id: UUID
    filename: str = Field(min_length=1, max_length=180)
    content_base64: str = Field(min_length=1, max_length=MAX_UPLOAD_BODY)


class _HtmlText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hidden = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        elif tag in {"p", "br", "div", "tr", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def _xml(raw):
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise AttachmentError("unsafe_document")
    return ET.fromstring(raw)


def _office_text(raw: bytes, suffix: str) -> str:
    with ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        if (len(infos) > 3000 or sum(item.file_size for item in infos) > 32 * 1024 * 1024
                or any(item.flag_bits & 1 for item in infos)):
            raise AttachmentError("document_archive_limit")
        names = set(archive.namelist())
        if suffix == ".docx":
            root = _xml(archive.read("word/document.xml"))
            return "\n".join("".join(p.itertext()) for p in root.iter()
                             if p.tag.endswith("}p"))
        if suffix in {".odt", ".ods", ".odp"}:
            root = _xml(archive.read("content.xml"))
            return "\n".join("".join(p.itertext()) for p in root.iter()
                             if p.tag.endswith("}p"))
        if suffix == ".pptx":
            slides = sorted((name for name in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)),
                            key=lambda name: int(re.search(r"slide(\d+)\.xml", name).group(1)))
            return "\n\n".join(f"[Slide {index}]\n" + "\n".join(
                node.text or "" for node in _xml(archive.read(name)).iter() if node.tag.endswith("}t"))
                for index, name in enumerate(slides, 1))
        strings = []
        if "xl/sharedStrings.xml" in names:
            strings = ["".join(node.itertext()) for node in _xml(archive.read("xl/sharedStrings.xml"))]
        sheets = sorted(name for name in names if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name))
        output = []
        for name in sheets:
            output.append(f"[{name}]")
            for row in _xml(archive.read(name)).iter():
                if not row.tag.endswith("}row"):
                    continue
                cells = []
                for cell in row:
                    kind = cell.get("t")
                    # Formula text is not evaluated. Only the stored result is read.
                    values = [node.text or "" for node in cell.iter()
                              if node.tag.endswith("}v") or node.tag.endswith("}t")]
                    value = " ".join(values)
                    if kind == "s" and value.isdigit():
                        value = strings[int(value)]
                    cells.append(f"{cell.get('r', '')}: {value}")
                output.append("\t".join(cells))
        return "\n".join(output)


def extract_document(filename: str, raw: bytes) -> tuple[str, list[str]]:
    suffix = PurePath(filename).suffix.lower()
    if suffix not in EXTENSIONS:
        raise AttachmentError("unsupported_file_type")
    if not raw or len(raw) > MAX_FILE_BYTES:
        raise AttachmentError("file_size_limit")
    warnings = []
    try:
        if suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw), strict=True)
            if reader.is_encrypted:
                raise AttachmentError("encrypted_document")
            if len(reader.pages) > 200:
                raise AttachmentError("document_page_limit")
            parts = []
            for index, page in enumerate(reader.pages, 1):
                parts.append(f"[Page {index}]\n{page.extract_text() or ''}")
                if sum(map(len, parts)) > MAX_TEXT_CHARS:
                    warnings.append("text_truncated")
                    break
            text = "\n\n".join(parts)
            if not re.sub(r"\[Page \d+\]|\s", "", text):
                raise AttachmentError("no_readable_text")
        elif suffix in {".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp"}:
            text = _office_text(raw, suffix)
        else:
            text = None
            for encoding in ("utf-8-sig", "utf-16", "gb18030"):
                try:
                    text = raw.decode(encoding)
                    break
                except UnicodeError:
                    continue
            if text is None:
                raise AttachmentError("unsupported_text_encoding")
            if suffix in {".html", ".htm"}:
                parser = _HtmlText()
                parser.feed(text)
                text = "".join(parser.parts)
            elif suffix == ".rtf":
                text = re.sub(r"\\'([a-fA-F0-9]{2})", lambda m: bytes.fromhex(m[1]).decode("cp1252"), text)
                text = re.sub(r"\\u(-?\d+)\??", lambda m: chr(int(m[1]) % 65536), text)
                text = re.sub(r"\\(?:par|line)\b", "\n", text)
                text = re.sub(r"\\[a-zA-Z]+-?\d* ?|[{}]", "", text)
            elif suffix == ".xml":
                text = " ".join(_xml(raw).itertext())
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text).strip()
        if not text:
            raise AttachmentError("no_readable_text")
        if len(text) > MAX_TEXT_CHARS and "text_truncated" not in warnings:
            warnings.append("text_truncated")
        return text[:MAX_TEXT_CHARS], warnings
    except AttachmentError:
        raise
    except Exception:
        raise AttachmentError("document_read_failed") from None


class AttachmentStore:
    def __init__(self, clock=monotonic):
        self.clock = clock
        self.items = {}
        self.lock = RLock()

    def discard_session(self, session_id):
        with self.lock:
            self.items = {key: item for key, item in self.items.items() if item[0] != session_id}

    def add(self, session_id: UUID, filename: str, encoded: str) -> dict:
        if (len(filename) > 180 or any(char in filename for char in "/\\\x00")
                or any(ord(char) < 32 for char in filename)):
            raise AttachmentError("invalid_filename")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            raise AttachmentError("invalid_file_encoding") from None
        text, warnings = extract_document(filename, raw)
        with self.lock:
            self.items = {key: item for key, item in self.items.items() if self.clock() - item[1] < 1800}
            if len(self.items) >= 128 or sum(item[0] == session_id for item in self.items.values()) >= 4:
                raise AttachmentError("attachment_capacity")
            key = uuid4()
            document = {"document_id": str(key), "filename": filename, "text": text,
                        "sha256": hashlib.sha256(raw).hexdigest(), "warnings": warnings}
            self.items[key] = (session_id, self.clock(), document)
            return {"attachment_id": str(key), "filename": filename, "characters": len(text),
                    "preview": text[:1200], "warnings": warnings, "expires_in_seconds": 1800}

    def resolve(self, session_id: UUID, ids: list[UUID]) -> list[dict]:
        with self.lock:
            if len(ids) > 4 or len(set(ids)) != len(ids):
                raise AttachmentError("attachment_capacity")
            result = []
            for key in ids:
                item = self.items.get(key)
                if not item or item[0] != session_id or self.clock() - item[1] >= 1800:
                    raise AttachmentError("attachment_not_found")
                self.items[key] = (item[0], self.clock(), item[2])
                result.append(dict(item[2]))
            return result
