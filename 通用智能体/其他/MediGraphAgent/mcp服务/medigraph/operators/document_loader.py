"""Multi-format document ingestion operator (7+ common data formats)."""
from __future__ import annotations

import csv
import html
import json
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

from medigraph.operators.base import BaseOperator, OperatorMeta


class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._ignored = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript"}:
            self._ignored += 1
        if tag in {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._ignored:
            self._ignored -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored:
            self.parts.append(data)

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", html.unescape("".join(self.parts))).strip()


class DocumentLoaderOperator(BaseOperator):
    SUPPORTED = {".txt", ".md", ".html", ".htm", ".csv", ".json", ".jsonl", ".docx", ".pdf"}

    def __init__(self, max_file_bytes: int = 20 * 1024 * 1024):
        self.max_file_bytes = max_file_bytes
        self.meta = OperatorMeta(
            name="document_loader",
            version="1.0.0",
            description=(
                "读取 TXT/Markdown/HTML/CSV/JSON/JSONL/DOCX/PDF 等文档并统一为文本。"
                "输入 {path} 或 {paths:[...]}, 输出 {documents,text,formats,errors}。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "paths": {"type": "array"},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "documents": {"type": "array"},
                    "text": {"type": "string"},
                    "formats": {"type": "array"},
                    "errors": {"type": "array"},
                },
                "required": ["documents", "text"],
            },
        )

    def run(self, inputs: dict, **kwargs) -> dict:
        raw_paths = inputs.get("paths", [])
        if not isinstance(raw_paths, list):
            raw_paths = []
        if inputs.get("path"):
            raw_paths = [inputs["path"], *raw_paths]
        documents, errors = [], []
        for raw_path in raw_paths:
            path = Path(str(raw_path)).expanduser().resolve()
            try:
                if not path.is_file():
                    raise FileNotFoundError(path)
                if path.suffix.lower() not in self.SUPPORTED:
                    raise ValueError(f"unsupported format: {path.suffix}")
                if path.stat().st_size > self.max_file_bytes:
                    raise ValueError(f"file exceeds {self.max_file_bytes} bytes")
                text = self._read(path)
                documents.append(
                    {
                        "fileName": path.name,
                        "path": str(path),
                        "format": path.suffix.lower().lstrip("."),
                        "bytes": path.stat().st_size,
                        "text": text,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - report per-file ingestion errors
                errors.append({"path": str(path), "error": f"{exc.__class__.__name__}: {exc}"})
        return {
            "documents": documents,
            "text": "\n\n".join(document["text"] for document in documents),
            "formats": sorted({document["format"] for document in documents}),
            "errors": errors,
        }

    def _read(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return path.read_text(encoding="utf-8-sig")
        if suffix in {".html", ".htm"}:
            parser = _HTMLTextExtractor()
            parser.feed(path.read_text(encoding="utf-8-sig"))
            return parser.text()
        if suffix == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as handle:
                return "\n".join(
                    " | ".join(str(value) for value in row)
                    for row in csv.reader(handle)
                )
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            return json.dumps(data, ensure_ascii=False, indent=2)
        if suffix == ".jsonl":
            rows = []
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    rows.append(json.dumps(json.loads(line), ensure_ascii=False))
            return "\n".join(rows)
        if suffix == ".docx":
            return self._read_docx(path)
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise RuntimeError("PDF support requires pypdf; install requirements.txt") from exc
            return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
        raise ValueError(f"unsupported format: {suffix}")

    @staticmethod
    def _read_docx(path: Path) -> str:
        with zipfile.ZipFile(path) as archive:
            xml_bytes = archive.read("word/document.xml")
        root = ElementTree.fromstring(xml_bytes)
        namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        paragraphs = []
        for paragraph in root.iter(f"{namespace}p"):
            text = "".join(node.text or "" for node in paragraph.iter(f"{namespace}t"))
            if text:
                paragraphs.append(text)
        return "\n".join(paragraphs)
