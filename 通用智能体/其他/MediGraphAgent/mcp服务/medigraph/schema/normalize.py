"""Entity normalization & alignment helpers.

Improves graph quality by:
  1. canonical_key(): a case/space/punct-insensitive key so "Pheochromocytoma",
     "pheochromocytoma" and "Pheochromocytoma " collapse to one node.
  2. ALIASES: a small medical synonym map for cross-surface alignment
     (e.g. 阿斯匹林/阿司匹林 -> 阿司匹林; CgA -> chromogranin).
  3. is_structural_noise(): rejects document-structure phrases that pathology web
     pages leak (section titles like "Radiology images", "Gross description").
  4. is_valid_entity_name(): rejects placeholders / table markers such as "|"
     or a standalone Roman numeral "Ⅰ" before they enter demos.

This is intentionally lightweight (no external KB); it removes the bulk of the
observed noise. Full ontology linking (e.g. CMeKG) is a documented next step.
"""
from __future__ import annotations

import re

# Known surface-form aliases -> canonical display name (extend as needed).
ALIASES: dict[str, str] = {
    "阿斯匹林": "阿司匹林",
    "乙酰水杨酸": "阿司匹林",
    "cga": "chromogranin",
    "chromogranin a": "chromogranin",
    "syn": "synaptophysin",
    "二甲双胍片": "二甲双胍",
}

# Document-structure / navigation phrases that are not medical entities. These are
# section headings common in pathologyoutlines-style pages. Matching is on the
# canonical (lowercased) form.
_NOISE_PHRASES = {
    "definition", "general", "essential features", "terminology", "icd coding",
    "epidemiology", "sites", "pathophysiology", "etiology", "clinical features",
    "diagnosis", "laboratory", "radiology", "radiology description",
    "radiology images", "prognostic factors", "case reports", "treatment",
    "clinical images", "gross description", "gross images",
    "frozen section description", "frozen section images",
    "microscopic histologic description", "microscopic histologic images",
    "virtual slides", "cytology images", "cytology", "positive stains",
    "negative stains", "electron microscopy", "electron microscopy description",
    "electron microscopy images", "molecular cytogenetics",
    "molecular cytogenetics description", "sample pathology report",
    "differential diagnosis", "additional references", "board review",
    "images hosted on other servers", "contributed by", "table",
}


def _strip_punct(text: str) -> str:
    return re.sub(r"[^\w一-鿿]+", " ", text).strip()


def canonical_key(name: str) -> str:
    """Identity key for dedup: lowercase, punctuation->space, spaces removed."""
    base = _strip_punct(name).lower()
    return base.replace(" ", "")


def canonical_name(name: str) -> str:
    """Normalized display name: trim, collapse spaces, apply alias map."""
    cleaned = re.sub(r"\s+", " ", name).strip()
    alias = ALIASES.get(cleaned.lower())
    return alias if alias else cleaned


def is_structural_noise(name: str) -> bool:
    """True if `name` looks like a section heading / navigation, not an entity."""
    canon = _strip_punct(name).lower()
    if not canon:
        return True
    if canon in _NOISE_PHRASES:
        return True
    # very short all-caps section-ish tokens or pure numbers
    if canon.isdigit():
        return True
    return False


_STANDALONE_ROMAN = {
    "Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ", "Ⅴ", "Ⅵ", "Ⅶ", "Ⅷ", "Ⅸ", "Ⅹ", "Ⅺ", "Ⅻ",
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
}
_PLACEHOLDER_TOKENS = {
    "|", "/", "\\", "-", "—", "–", "_", "+", "*", "?", "？", ".", "。",
    "无", "未知", "其他", "其它", "NA", "N/A", "null", "None",
}


def is_valid_entity_name(name: str) -> bool:
    """False for tokens that should never be treated as medical entities.

    CM3KG occasionally contains table markers such as a standalone Roman numeral
    "Ⅰ".  Real concepts like "硝苯地平缓释片Ⅰ" or "ApoA-Ⅰ" are kept because the
    marker is part of a longer medical term.
    """
    cleaned = canonical_name(str(name or ""))
    if not cleaned:
        return False
    if is_structural_noise(cleaned):
        return False
    if cleaned in _PLACEHOLDER_TOKENS or cleaned.upper() in _PLACEHOLDER_TOKENS:
        return False
    if cleaned in _STANDALONE_ROMAN or cleaned.upper() in _STANDALONE_ROMAN:
        return False
    if re.fullmatch(r"[\W_]+", cleaned, flags=re.UNICODE):
        return False
    if len(cleaned) == 1 and not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", cleaned):
        return False
    return True
