# -*- coding: utf-8 -*-
"""DataMate operator: rule-based medical text cleaning.

This is the self-contained DataMate mirror of
``medigraph/operators/text_clean.py``. It reads the source file when used as the
first operator and keeps the cleaned text in ``sample['text']`` for downstream
chunking and extraction operators.
"""
from __future__ import annotations

import re
from typing import Any, Dict

from datamate.core.base_op import Mapper


class TextClean(Mapper):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_line_length = max(0, int(kwargs.get("minLineLength", 8)))

    @staticmethod
    def _remove_line_patterns(text: str, patterns: list[str]) -> str:
        out = []
        for line in text.split("\n"):
            if line.lstrip().startswith("#"):
                out.append(line)
                continue
            for pattern in patterns:
                line = re.sub(pattern, "", line, flags=re.IGNORECASE)
            out.append(line)
        return "\n".join(out)

    @classmethod
    def _remove_metadata_footnotes(cls, text: str) -> str:
        out = []
        for line in text.split("\n"):
            stripped = line.strip()
            if any(re.match(pattern, stripped, flags=re.IGNORECASE)
                   for pattern in cls.METADATA_FOOTNOTE_PATTERNS):
                continue
            out.append(line)
        return "\n".join(out)

    @staticmethod
    def _clean_links(text: str) -> str:
        text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"https?://\S+", "", text)
        return re.sub(
            r"^(?![ \t]*#)[^\n]*\.(?:jpg|png|jpeg|gif|tif)\)?\s*$",
            "",
            text,
            flags=re.MULTILINE | re.IGNORECASE,
        )

    @staticmethod
    def _clean_latex(text: str) -> str:
        def simplify(match: re.Match) -> str:
            inner = match.group(1).strip()
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
            if stripped.startswith(("#", "|", "【")) or self._ZH_HEADING_RE.match(stripped):
                out.append(line)
                continue
            if stripped.startswith(("-", "*", "•")) and len(stripped) > 5:
                out.append(line)
                continue
            if re.match(r"^\d+[\.、）)]\s", stripped):
                out.append(line)
                continue
            if len(stripped) < self.min_line_length and not stripped.endswith(("。", ".", "：", ":")):
                continue
            out.append(line)
        return "\n".join(out)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip("\n") + "\n" if text.strip() else ""

    def execute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        self.read_file_first(sample)
        text = str(sample.get(self.text_key, "") or "")
        if not text.strip():
            sample[self.text_key] = ""
            return sample
        text = self._remove_line_patterns(text, self.HEADER_FOOTER_PATTERNS)
        text = self._remove_line_patterns(text, self.WEB_NOISE_PATTERNS)
        text = self._remove_metadata_footnotes(text)
        text = self._clean_links(text)
        text = self._clean_latex(text)
        text = self._remove_short_fragments(text)
        sample[self.text_key] = self._normalize_whitespace(text).strip()
        return sample
