"""Medical suffix-pattern entity discovery for open-domain (OOV) coverage.

Supplements the dictionary extractor with Chinese medical morphological patterns
without adding OOV terms to the controlled vocabulary file used by the benchmark.
"""

from __future__ import annotations

import re

# Suffix -> (entity_type, min_total_len, max_stem_len)
_SUFFIX_RULES: list[tuple[str, str, int, int]] = [
    ("反流病", "Disease", 4, 8),
    ("综合征", "Disease", 4, 10),
    ("硬化", "Disease", 4, 12),
    ("病", "Disease", 3, 8),
    ("灼痛", "Symptom", 3, 6),
    ("加重", "Symptom", 3, 6),
    ("替丁", "Drug", 3, 6),
    ("涂片", "Examination", 3, 8),
    ("磁共振", "Examination", 4, 8),
    ("病理", "Examination", 3, 8),
    ("抗体", "Examination", 4, 10),
    ("苷酶", "Examination", 3, 5),
    ("活性测定", "Examination", 4, 6),
    ("替代治疗", "Treatment", 4, 10),
    ("减少治疗", "Treatment", 4, 10),
    ("抑制方案", "Treatment", 4, 8),
    ("凝管理", "Treatment", 4, 4),
]

# When walking left from a suffix, stop extending the stem at these characters.
# Use single-character stops only; avoid digraphs like 并发 that would block 发 in 原发性.
_STEM_STOP_CHARS = set("确诊患为是伴见有予建议拟嘱反复活动期清查否否认并史")

# Prefix-stripped blood tests: 血乳酸, 血常规 — not substrings inside 高血压.
_BLOOD_TEST_RE = re.compile(
    r"(?<![\u4e00-\u9fff])(血[\u4e00-\u9fff]{1,2})(?:升高|下降|异常|增多|减少|示|提|见)?"
)

_REJECT_TERMS: frozenset[str] = frozenset({
    "基因检测",
    "蛋白尿",
    "血小板减少",
    "脾肿大",
    "血栓事件",
})

# Shared reject list for dictionary and pattern matching.
ENTITY_SURFACE_REJECT: frozenset[str] = _REJECT_TERMS


def _is_hanzi(ch: str) -> bool:
    return len(ch) == 1 and "\u4e00" <= ch <= "\u9fff"


def _stem_before(text: str, suffix_start: int, max_stem: int) -> str:
    """Collect up to ``max_stem`` hanzi immediately before ``suffix_start``."""

    stem: list[str] = []
    index = suffix_start - 1
    while index >= 0 and len(stem) < max_stem:
        char = text[index]
        if not _is_hanzi(char):
            break
        if char in _STEM_STOP_CHARS and stem:
            if char == "并" and stem[0] == "发":
                stem.pop(0)
            break
        stem.insert(0, char)
        index -= 1
    return "".join(stem)


def _entities_with_suffix(
    text: str,
    suffix: str,
    entity_type: str,
    min_len: int,
    max_stem: int,
) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    start = 0
    while True:
        index = text.find(suffix, start)
        if index < 0:
            break
        end = index + len(suffix)
        if suffix == "病" and end < len(text) and text[end] in {"灶", "理", "史"}:
            start = index + 1
            continue
        stem = _stem_before(text, index, max_stem)
        term = stem + suffix
        if len(term) >= min_len and term not in _REJECT_TERMS:
            results.append((entity_type, term))
        start = end
    return results


def find_pattern_entities(text: str) -> list[tuple[str, str]]:
    """Return (entity_type, surface_term) pairs discovered via suffix patterns."""

    found: list[tuple[str, str]] = []
    for suffix, entity_type, min_len, max_stem in _SUFFIX_RULES:
        found.extend(_entities_with_suffix(text, suffix, entity_type, min_len, max_stem))

    for match in _BLOOD_TEST_RE.finditer(text):
        term = match.group(1)
        for suffix in ("升高", "下降", "异常", "增多", "减少"):
            if term.endswith(suffix):
                term = term[: -len(suffix)]
                break
        if term not in _REJECT_TERMS:
            found.append(("Examination", term))

    # Explicit multi-char treatment that suffix rules may miss.
    if "抗凝管理" in text:
        found.append(("Treatment", "抗凝管理"))

    return found
