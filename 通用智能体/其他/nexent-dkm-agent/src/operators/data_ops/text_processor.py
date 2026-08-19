"""Text processing operators for task 1."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Medical entity dictionaries (rule-based)
_DISEASE_KEYWORDS = [
    "高血压", "糖尿病", "哮喘", "冠心病", "肺炎", "肝炎", "胃炎",
    "骨折", "贫血", "甲亢", "甲状腺功能减退", "心力衰竭", "脑卒中",
    "hypertension", "diabetes", "asthma", "pneumonia", "fracture",
]

_DRUG_KEYWORDS = [
    "阿司匹林", "布洛芬", "二甲双胍", "氨氯地平", "辛伐他汀",
    "奥美拉唑", "头孢", "青霉素", "甲硝唑", "阿莫西林",
    "aspirin", "ibuprofin", "metformin", "amoxicillin",
]

_EXAM_KEYWORDS = [
    "血常规", "尿常规", "CT", "MRI", "X光", "B超", "心电图",
    "肝功能", "肾功能", "血糖", "血脂", "血沉",
    "blood test", "urinalysis", "ecg",
]

_PII_PATTERNS = [
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[PHONE]"),
    (re.compile(r"(?<!\d)\d{6}(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"), "[ID_CARD]"),
]

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_FULLWIDTH_DIGIT_RE = re.compile(r"[\uff10-\uff19]")
_FULLWIDTH_ALPHA_RE = re.compile(r"[\uff21-\uff3a\uff41-\uff5a]")
_SPECIAL_SPACE_RE = re.compile(r"[\u00a0\u2000-\u200b\u3000\ufeff]+")


def process_text(
    input_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Clean unstructured text: remove HTML, normalize Unicode, redact PII, segment."""

    text_path = Path(input_path)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    raw_text = text_path.read_text(encoding="utf-8-sig")
    records = _split_records(raw_text)

    cleaned_records = []
    total_pii_redacted = 0
    total_html_tags_removed = 0

    for record in records:
        cleaned, html_count = _remove_html_tags(record)
        total_html_tags_removed += html_count

        cleaned = _normalize_unicode(cleaned)
        cleaned, pii_count = _redact_pii(cleaned)
        total_pii_redacted += pii_count

        cleaned = _normalize_whitespace(cleaned)
        cleaned_records.append(cleaned)

    output_path = target_dir / f"{text_path.stem}_cleaned.txt"
    output_path.write_text(
        "\n---\n".join(cleaned_records) + "\n",
        encoding="utf-8",
    )

    return {
        "status": "completed",
        "output_path": str(output_path),
        "input_records": len(records),
        "output_records": len(cleaned_records),
        "html_tags_removed": total_html_tags_removed,
        "pii_redacted": total_pii_redacted,
    }


def extract_medical_entities(text: str) -> dict[str, list[str]]:
    """Extract medical entities from text using rule-based keyword matching."""

    diseases = [kw for kw in _DISEASE_KEYWORDS if kw.lower() in text.lower()]
    drugs = [kw for kw in _DRUG_KEYWORDS if kw.lower() in text.lower()]
    exams = [kw for kw in _EXAM_KEYWORDS if kw.lower() in text.lower()]

    return {
        "diseases": diseases,
        "drugs": drugs,
        "examinations": exams,
    }


def _split_records(text: str) -> list[str]:
    """Split text into individual records separated by --- or double newlines."""
    if "---" in text:
        return [r.strip() for r in text.split("---") if r.strip()]
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _remove_html_tags(text: str) -> tuple[str, int]:
    """Remove HTML tags and count removals."""
    count = len(_HTML_TAG_RE.findall(text))
    cleaned = _HTML_TAG_RE.sub("", text)
    return cleaned, count


def _normalize_unicode(text: str) -> str:
    """Convert fullwidth digits and letters to their ASCII equivalents."""
    result = []
    for ch in text:
        if "\uff10" <= ch <= "\uff19":
            result.append(str(ord(ch) - 0xff10))
        elif "\uff21" <= ch <= "\uff3a":
            result.append(chr(ord(ch) - 0xff21 + ord("A")))
        elif "\uff41" <= ch <= "\uff5a":
            result.append(chr(ord(ch) - 0xff41 + ord("a")))
        else:
            result.append(ch)
    return "".join(result)


def _redact_pii(text: str) -> tuple[str, int]:
    """Replace phone numbers and ID card numbers with placeholders."""
    count = 0
    for pattern, replacement in _PII_PATTERNS:
        matches = pattern.findall(text)
        count += len(matches) if isinstance(matches, list) else len(pattern.findall(text))
        text = pattern.sub(replacement, text)
    return text, count


def _normalize_whitespace(text: str) -> str:
    """Normalize various Unicode whitespace characters to regular spaces."""
    text = _SPECIAL_SPACE_RE.sub(" ", text)
    text = re.sub(r"  +", " ", text)
    return text.strip()
