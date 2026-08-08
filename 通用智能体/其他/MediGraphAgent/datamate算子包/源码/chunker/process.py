# -*- coding: utf-8 -*-
"""DataMate operator: heading-aware medical text chunking.

DataMate's Slicer writes independent files but does not feed each slice into
later operators in the same cleaning task. This operator therefore uses Mapper
semantics: it stores chunks in an internal sample field consumed by MedicalNER
and MedicalRE, while exposing a JSON ``{"chunks": [...]}`` text result when used
as the final operator.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict

from datamate.core.base_op import Mapper

_HEADING_RE = re.compile(r"^(#{1,6}\s+.+|第[一二三四五六七八九十百千\d]+[章节篇卷].*)$")


class MedicalChunker(Mapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_chars = int(kwargs.get("maxChars", 1200))
        self.overlap = int(kwargs.get("overlap", 80))
        self.max_chunks = int(kwargs.get("maxChunks", 8))
        if self.max_chars < 100:
            raise ValueError("maxChars must be at least 100")
        if self.overlap < 0 or self.overlap >= self.max_chars:
            raise ValueError("overlap must satisfy 0 <= overlap < maxChars")
        if self.max_chunks < 0:
            raise ValueError("maxChunks must be >= 0")

    def _split_long(self, text: str) -> list[str]:
        paragraphs = re.split(r"\n\s*\n", text)
        out: list[str] = []
        current = ""
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(current) + len(paragraph) + 2 <= self.max_chars:
                current = f"{current}\n\n{paragraph}" if current else paragraph
                continue
            if current:
                out.append(current)
                current = ""
            if len(paragraph) <= self.max_chars:
                current = paragraph
                continue
            step = self.max_chars - self.overlap
            out.extend(paragraph[i:i + self.max_chars] for i in range(0, len(paragraph), step))
        if current:
            out.append(current)
        return out

    def _chunk(self, text: str) -> list[str]:
        sections: list[str] = []
        buffer: list[str] = []
        for line in text.split("\n"):
            if _HEADING_RE.match(line.strip()) and buffer:
                sections.append("\n".join(buffer))
                buffer = [line]
            else:
                buffer.append(line)
        if buffer:
            sections.append("\n".join(buffer))

        chunks: list[str] = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(section) <= self.max_chars:
                chunks.append(section)
            else:
                chunks.extend(self._split_long(section))
        if self.max_chunks:
            chunks = chunks[:self.max_chunks]
        return chunks

    def execute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        self.read_file_first(sample)
        text = str(sample.get(self.text_key, "") or "").strip()
        if not text:
            sample["_medical_source_text"] = ""
            sample["_medical_chunks"] = []
            sample[self.text_key] = ""
            return sample
        chunks = self._chunk(text)
        sample["_medical_source_text"] = text
        sample["_medical_chunks"] = chunks
        sample[self.text_key] = json.dumps({"chunks": chunks}, ensure_ascii=False, indent=2)
        return sample
