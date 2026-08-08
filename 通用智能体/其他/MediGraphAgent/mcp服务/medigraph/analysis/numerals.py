"""Chinese-numeral normalization for medical NL2SQL.

Why this exists
---------------
The deterministic NL2SQL router builds filters with regexes that only recognise
ASCII digits (``(\\d+)\\s*岁``, ``(?:前|top\\s*)(\\d+)`` ...). A question phrased
with Chinese numerals therefore lost its predicate *silently* -- no error, just a
wrong answer:

    六十岁以上高血压患者的平均住院费用是多少？
      -> SELECT AVG(cost) ... WHERE disease='高血压'            # age>60 dropped
    60岁以上高血压患者的平均住院费用是多少？
      -> SELECT AVG(cost) ... WHERE disease='高血压' AND age>60  # correct

Rewriting every Chinese numeral is *not* safe, because medical terminology is
full of them. Measured against the 7,465 distinct disease/department/drug/test/
entity values in the analytics DB, a naive rewrite corrupts 124+ of them
(``十二指肠白点综合征``, ``二十五味松石丸``, ``血液生化六项检查``,
``一秒用力呼出量``, ``百日咳`` ...).

So normalization is gated twice:

1. **Unit gate** -- a numeral run is only rewritten when it is immediately
   followed by a counter/unit that marks a quantity (``岁``/``个``/``种`` ...),
   or immediately preceded by a ranking prefix (``前``). ``十二指肠`` is left
   alone because ``指`` is not a counter.
2. **Vocabulary mask** -- callers may pass the known value vocabulary as
   ``protected``; any numeral run inside one of those terms is skipped. This is
   what makes otherwise-ambiguous counters such as ``天``/``年``/``项``/``味``
   safe to enable (``二天油``, ``复方万年青胶囊``, ``血液生化六项检查``,
   ``二十五味松石丸`` are all in the vocabulary).

Vague quantifiers (``十几岁``, ``几十个``) are deliberately **not** rewritten:
mapping ``十几`` to ``10`` would reintroduce exactly the class of silent wrong
answer this module removes.
"""
from __future__ import annotations

import re
from typing import Iterable

_DIGITS: dict[str, int] = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "俩": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_SMALL_UNITS: dict[str, int] = {"十": 10, "百": 100, "千": 1000}
_MYRIAD = "万"

_NUMERAL_CHARS = "".join(_DIGITS) + "".join(_SMALL_UNITS) + _MYRIAD
_NUMERAL_RUN = re.compile(f"[{_NUMERAL_CHARS}]+")

#: Counters/units that mark a *quantity* rather than part of a term name. Every
#: entry was verified to produce zero collisions against the analytics DB
#: vocabulary, except the four marked below which rely on the ``protected`` mask.
SAFE_UNITS: tuple[str, ...] = (
    "岁", "个", "种", "型", "名", "位", "次", "周", "月",
    "条", "例", "人", "份", "度", "级", "期", "分",
    # vocabulary-mask dependent (1-38 collisions each, all in the value lists)
    "天", "年", "项", "味",
)

#: Prefixes after which a bare numeral denotes a rank ("前三" -> "前3").
RANK_PREFIXES: tuple[str, ...] = ("前", "top", "TOP", "Top")

#: Characters that make a numeral vague; never rewrite when adjacent.
_VAGUE = "几多"


def chinese_to_int(text: str) -> int | None:
    """Parse a pure Chinese numeral run into an int, or None if unparseable.

    Handles 十五 / 二十 / 二十五 / 一百零八 / 三千五百 / 一万 / 两 / 十.
    """
    if not text:
        return None
    total = 0
    section = 0
    number = 0
    seen_digit = False
    for char in text:
        if char in _DIGITS:
            number = _DIGITS[char]
            seen_digit = True
        elif char in _SMALL_UNITS:
            unit = _SMALL_UNITS[char]
            # "十五" -> the leading 十 carries an implicit 1.
            section += (number if number else 1) * unit
            number = 0
            seen_digit = True
        elif char == _MYRIAD:
            section += number
            total += (section if section else 1) * 10_000
            section = 0
            number = 0
            seen_digit = True
        else:
            return None
    if not seen_digit:
        return None
    return total + section + number


def _protected_spans(text: str, protected: Iterable[str] | None) -> list[tuple[int, int]]:
    """Spans of `text` covered by a protected vocabulary term."""
    if not protected:
        return []
    spans: list[tuple[int, int]] = []
    for term in protected:
        if not term or len(term) < 2:
            continue
        start = text.find(term)
        while start >= 0:
            spans.append((start, start + len(term)))
            start = text.find(term, start + 1)
    return spans


def _gate(text: str, start: int, end: int) -> str | None:
    """Return the reason this numeral run is a quantity, or None to skip it."""
    after = text[end:]
    for unit in SAFE_UNITS:
        if after.startswith(unit):
            return f"unit:{unit}"
    before = text[:start]
    for prefix in RANK_PREFIXES:
        if before.endswith(prefix):
            return f"rank:{prefix}"
    return None


def normalize_numerals(
    text: str,
    protected: Iterable[str] | None = None,
) -> tuple[str, list[dict]]:
    """Rewrite quantity-bearing Chinese numerals to ASCII digits.

    Returns ``(normalized_text, rewrites)`` where each rewrite records the
    original surface form, its replacement and why it was considered a quantity.
    The rewrite log is surfaced in the schema-linking result so a reviewer can
    see exactly what the router did to the question.
    """
    if not text:
        return text, []
    blocked = _protected_spans(text, protected)
    pieces: list[str] = []
    rewrites: list[dict] = []
    cursor = 0
    for match in _NUMERAL_RUN.finditer(text):
        start, end = match.span()
        run = match.group()
        # Adjacent vague quantifier ("十几岁", "几十个") -> leave untouched.
        neighbours = text[max(0, start - 1):start] + text[end:end + 1]
        if any(char in _VAGUE for char in neighbours):
            continue
        if any(start < b_end and b_start < end for b_start, b_end in blocked):
            continue
        reason = _gate(text, start, end)
        if reason is None:
            continue
        value = chinese_to_int(run)
        if value is None:
            continue
        pieces.append(text[cursor:start])
        pieces.append(str(value))
        cursor = end
        rewrites.append({"surface": run, "value": value, "reason": reason})
    if not rewrites:
        return text, []
    pieces.append(text[cursor:])
    return "".join(pieces), rewrites


def normalize(text: str, protected: Iterable[str] | None = None) -> str:
    """`normalize_numerals` without the rewrite log."""
    return normalize_numerals(text, protected)[0]


def numeral_bearing(terms: Iterable[str]) -> list[str]:
    """Filter a vocabulary down to the terms a rewrite could damage.

    Only terms containing a Chinese numeral can be corrupted, so callers keep the
    protected set small (~200 of 7,465 values) instead of scanning everything.
    """
    return [term for term in terms if term and _NUMERAL_RUN.search(term)]
