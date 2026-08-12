# -*- coding: utf-8 -*-
"""Deterministic SQLite-backed noise rule engine.

Rules are evidence-backed rows in ``noise_rules``. The engine deliberately keeps
matching deterministic and conservative:
- normalize only for detection, not for output text
- apply negative patterns before deletion
- delete by configured scope: match, line, sentence, or block
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_DB_PATH = "/opt/runtime/datamate/ops/user/llm_noise_filter/noise_kb.db"


@dataclass
class NoiseRule:
    rule_id: str
    category: str
    pattern: str
    match_type: str = "regex"
    scope: str = "line"
    confidence: float = 0.9
    medical_safe: int = 1
    status: str = "active"
    negative_patterns: str = ""


def normalize_for_match(text: str) -> str:
    trans = str.maketrans({
        "．": ".",
        "。": ".",
        "，": ",",
        "；": ";",
        "：": ":",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "　": " ",
    })
    text = text.translate(trans)
    return text


def _line_span(text: str, start: int, end: int) -> tuple[int, int]:
    left = text.rfind("\n", 0, start) + 1
    right = text.find("\n", end)
    if right < 0:
        right = len(text)
    return left, right


def _sentence_span(text: str, start: int, end: int) -> tuple[int, int]:
    left_candidates = [text.rfind(x, 0, start) for x in "。！？!?；;\n"]
    left = max(left_candidates) + 1
    right_candidates = [text.find(x, end) for x in "。！？!?；;\n"]
    right_candidates = [x for x in right_candidates if x >= 0]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    return left, right


def _block_span(text: str, start: int, end: int) -> tuple[int, int]:
    left = text.rfind("\n\n", 0, start)
    left = 0 if left < 0 else left + 2
    right = text.find("\n\n", end)
    right = len(text) if right < 0 else right
    return left, right


class NoiseRuleEngine:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.rules = self._load_rules(db_path)

    @staticmethod
    def _load_rules(db_path: str) -> list[NoiseRule]:
        if not db_path or not Path(db_path).exists():
            return []
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(noise_rules)").fetchall()}
            if not columns:
                return []
            select_cols = [
                "rule_id" if "rule_id" in columns else "id",
                "category",
                "pattern",
                "match_type",
                "scope",
                "confidence",
                "medical_safe",
                "status",
                "negative_patterns",
            ]
            query_cols = []
            for col in select_cols:
                if col in columns:
                    query_cols.append(col)
                elif col == "negative_patterns":
                    query_cols.append("'' as negative_patterns")
                elif col == "confidence":
                    query_cols.append("0.9 as confidence")
                elif col == "scope":
                    query_cols.append("'line' as scope")
                elif col == "match_type":
                    query_cols.append("'regex' as match_type")
                elif col == "medical_safe":
                    query_cols.append("1 as medical_safe")
                elif col == "status":
                    query_cols.append("'active' as status")
            rows = conn.execute(
                f"SELECT {', '.join(query_cols)} FROM noise_rules "
                "WHERE status='active' AND "
                "(medical_safe=1 OR match_type='semantic_hint')"
            ).fetchall()
            rules = []
            for row in rows:
                data = dict(row)
                rule_id = data.get("rule_id") or data.get("id") or data["pattern"][:40]
                rules.append(NoiseRule(rule_id=str(rule_id), **{k: v for k, v in data.items() if k not in {"id", "rule_id"}}))
            return rules
        finally:
            conn.close()

    def has_match(self, text: str, include_hints: bool = True) -> bool:
        return bool(self.find_matches(text, first_only=True, include_hints=include_hints))

    def find_matches(
        self,
        text: str,
        first_only: bool = False,
        include_hints: bool = True,
    ) -> list[tuple[NoiseRule, re.Match]]:
        normalized = normalize_for_match(text)
        matches = []
        for rule in self.rules:
            if rule.match_type == "semantic_hint" and not include_hints:
                continue
            pattern = (
                re.escape(normalize_for_match(rule.pattern))
                if rule.match_type == "exact"
                else rule.pattern
            )
            try:
                regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.UNICODE)
            except re.error:
                continue
            for match in regex.finditer(normalized):
                if self._is_negative(rule, normalized, match.start(), match.end()):
                    continue
                matches.append((rule, match))
                if first_only:
                    return matches
        return matches

    @staticmethod
    def _is_negative(rule: NoiseRule, normalized_text: str, start: int, end: int) -> bool:
        if not rule.negative_patterns:
            return False
        window_left, window_right = _line_span(normalized_text, start, end)
        window = normalized_text[window_left:window_right]
        for pattern in str(rule.negative_patterns).split("||"):
            pattern = pattern.strip()
            if pattern and re.search(pattern, window, re.IGNORECASE | re.UNICODE):
                return True
        return False

    def clean(self, text: str) -> tuple[str, list[str]]:
        if not text or not self.rules:
            return text, []
        matches = self.find_matches(text, include_hints=False)
        if not matches:
            return text, []

        spans: list[tuple[int, int]] = []
        for rule, match in matches:
            start, end = match.span()
            if rule.scope == "match":
                span = (start, end)
            elif rule.scope == "sentence":
                span = _sentence_span(text, start, end)
            elif rule.scope == "block":
                span = _block_span(text, start, end)
            else:
                span = _line_span(text, start, end)
            spans.append(span)

        merged = self._merge_spans(spans)
        removed = [text[start:end] for start, end in merged if text[start:end].strip()]
        cleaned_parts = []
        cursor = 0
        for start, end in merged:
            cleaned_parts.append(text[cursor:start])
            cursor = end
        cleaned_parts.append(text[cursor:])
        cleaned = "".join(cleaned_parts)
        cleaned = re.sub(r"[ \t]+(\n|$)", r"\1", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned, removed

    @staticmethod
    def _merge_spans(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
        ordered = sorted((s, e) for s, e in spans if e > s)
        merged: list[tuple[int, int]] = []
        for start, end in ordered:
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return merged
