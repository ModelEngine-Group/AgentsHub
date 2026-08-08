"""Heading-aware text chunking operator.

Splits long markdown/plain text into semantically coherent chunks bounded by a
target character length. Chunk boundaries prefer markdown headings, then blank
lines, so each chunk stays topical for downstream NER/RE.
"""
from __future__ import annotations

import re

from medigraph.operators.base import BaseOperator, OperatorMeta

_HEADING_RE = re.compile(r"^(#{1,6}\s+.+|第[一二三四五六七八九十百千\d]+[章节篇卷].*)$")


class ChunkerOperator(BaseOperator):
    def __init__(self, max_chars: int = 1200, overlap: int = 80):
        self.max_chars = max_chars
        self.overlap = overlap
        self.meta = OperatorMeta(
            name="chunker",
            description=(
                "把清洗后的长文本按标题层级和字数上限切分成语义连贯的文本块(chunks)，"
                "便于后续抽取。输入 {text, max_chars?}, 输出 {chunks: [str]}。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "max_chars": {"type": "integer"},
                },
                "required": ["text"],
            },
            output_schema={"type": "object", "properties": {"chunks": {"type": "array"}}},
        )

    def run(self, inputs: dict, **kwargs) -> dict:
        text = inputs.get("text", "") or ""
        max_chars = int(inputs.get("max_chars") or self.max_chars)
        if not text.strip():
            return {"chunks": []}

        # 1) Split into heading-led sections.
        sections: list[str] = []
        buf: list[str] = []
        for line in text.split("\n"):
            if _HEADING_RE.match(line.strip()) and buf:
                sections.append("\n".join(buf))
                buf = [line]
            else:
                buf.append(line)
        if buf:
            sections.append("\n".join(buf))

        # 2) Further split oversized sections by length (paragraph-aware).
        chunks: list[str] = []
        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue
            if len(sec) <= max_chars:
                chunks.append(sec)
            else:
                chunks.extend(self._split_long(sec, max_chars))
        return {"chunks": chunks}

    def _split_long(self, text: str, max_chars: int) -> list[str]:
        paras = re.split(r"\n\s*\n", text)
        out: list[str] = []
        cur = ""
        for p in paras:
            if len(cur) + len(p) + 2 <= max_chars:
                cur = f"{cur}\n\n{p}" if cur else p
            else:
                if cur:
                    out.append(cur)
                if len(p) <= max_chars:
                    cur = p
                else:
                    # hard wrap very long paragraph
                    for i in range(0, len(p), max_chars - self.overlap):
                        out.append(p[i : i + max_chars])
                    cur = ""
        if cur:
            out.append(cur)
        return out
