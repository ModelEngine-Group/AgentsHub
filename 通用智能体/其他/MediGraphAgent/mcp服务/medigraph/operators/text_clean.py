"""Rule-based medical text cleaning operator.

Removes web noise, headers/footers, links/images, LaTeX artifacts and short
fragments while protecting markdown/Chinese structural headings. Cleaning regexes
are ported from the team's previously validated `pathology_text_clean` operator.
"""
from __future__ import annotations

import re
from typing import Any

from medigraph.operators.base import BaseOperator, OperatorMeta


class TextCleanOperator(BaseOperator):
    _ZH_HEADING_RE = re.compile(
        r"^(?:第[一二三四五六七八九十百千\d]+[章节篇卷]|[一二三四五六七八九十百千]+[、])"
    )

    HEADER_FOOTER_PATTERNS = [
        r"Page\s+\d+\s+of\s+\d+",
        r"第\s*\d+\s*页\s*[，,/]\s*共\s*\d+\s*页",
        r"©\s*\d{4}.*?(?:All rights reserved|版权所有)",
        r"Downloaded from.*?(?:\n|$)",
    ]
    WEB_NOISE_PATTERNS = [
        r"Home\s*[>»›]\s*.*?[>»›]",
        r"Click here to.*?(?:\n|$)",
        r"Show Image",
        r"Read more\.{0,3}",
        r"Related (?:articles?|topics?|links?).*?(?:\n|$)",
        r"Share (?:on|this|via).*?(?:\n|$)",
        r"Advertisement",
        r"cookie(?:s)?\s*(?:policy|notice)",
    ]
    METADATA_FOOTNOTE_PATTERNS = [
        r"^数据来源[：:].+$",
        r"^参考(?:文献|书目|资料)[：:].+$",
        r"^来源[：:].+$",
        r"^出版社[：:].+$",
    ]

    def __init__(self, min_line_length: int = 8, min_paragraph_length: int = 30):
        self.min_line_length = min_line_length
        self.min_paragraph_length = min_paragraph_length
        self.meta = OperatorMeta(
            name="text_clean",
            description=(
                "清洗医疗/病理原始文本：去除网页噪声、页眉页脚、Markdown 链接与图片、"
                "LaTeX 公式残留和无意义短碎片，保护标题结构。输入 {text}, 输出 {text}。"
            ),
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            output_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        )

    # ------------------------------------------------------------------ #
    def run(self, inputs: dict, **kwargs) -> dict:
        text = inputs.get("text", "") or ""
        if not text.strip():
            return {"text": ""}
        text = self._remove_line_patterns(text, self.HEADER_FOOTER_PATTERNS)
        text = self._remove_line_patterns(text, self.WEB_NOISE_PATTERNS)
        text = self._remove_metadata_footnotes(text)
        text = self._clean_links(text)
        text = self._clean_latex(text)
        text = self._remove_short_fragments(text)
        text = self._normalize_whitespace(text)
        return {"text": text.strip()}

    # ------------------------------------------------------------------ #
    def _is_zh_heading(self, line: str) -> bool:
        return bool(self._ZH_HEADING_RE.match(line.strip()))

    def _remove_line_patterns(self, text: str, patterns: list[str]) -> str:
        out = []
        for line in text.split("\n"):
            if line.lstrip().startswith("#"):  # protect markdown headings
                out.append(line)
                continue
            for pat in patterns:
                line = re.sub(pat, "", line, flags=re.IGNORECASE)
            out.append(line)
        return "\n".join(out)

    def _remove_metadata_footnotes(self, text: str) -> str:
        out = []
        for line in text.split("\n"):
            stripped = line.strip()
            if any(re.match(p, stripped, flags=re.IGNORECASE) for p in self.METADATA_FOOTNOTE_PATTERNS):
                continue
            out.append(line)
        return "\n".join(out)

    def _clean_links(self, text: str) -> str:
        text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", "", text)        # images
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)       # links -> label
        text = re.sub(r"https?://\S+", "", text)                   # bare URLs
        text = re.sub(
            r"^(?![ \t]*#)[^\n]*\.(?:jpg|png|jpeg|gif|tif)\)?\s*$",
            "", text, flags=re.MULTILINE | re.IGNORECASE,
        )
        return text

    def _clean_latex(self, text: str) -> str:
        def simplify(m):
            inner = m.group(1).strip()
            inner = re.sub(r"\\(?:mathrm|text|mathbf|textbf)\{([^}]+)\}", r"\1", inner)
            inner = re.sub(r"[\^_]\{([^}]+)\}", r"\1", inner)
            inner = re.sub(r"\\[a-zA-Z]+", "", inner)
            return inner.strip("{}").strip()

        text = re.sub(r"\$\$[\s\S]*?\$\$", "", text)
        text = re.sub(r"\$([^$\n]+)\$", simplify, text)
        return text.replace("$", "")

    def _remove_short_fragments(self, text: str) -> str:
        out = []
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                out.append(line)
                continue
            if stripped.startswith(("#", "|", "【")) or self._is_zh_heading(stripped):
                out.append(line)
                continue
            if stripped.startswith(("-", "*", "•")) and len(stripped) > 5:
                out.append(line)
                continue
            if re.match(r"^\d+[\.、\）\)]\s", stripped):
                out.append(line)
                continue
            if len(stripped) < self.min_line_length and not stripped.endswith(("。", ".", "：", ":")):
                continue
            out.append(line)
        return "\n".join(out)

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip("\n") + "\n"
