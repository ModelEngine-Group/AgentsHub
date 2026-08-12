"""
任务一保留源格式交付清理模块。
"""

from __future__ import annotations

import csv
import json
import re
from io import StringIO
from pathlib import Path
from typing import Any


DROP_FIELDS = {"clean_reference", "noise_labels", "noise_injected", "noise", "output_format_hint"}
CONTROLLED_NOISE_RE = re.compile(
    r"controlled_noise_\d{8}\??|"
    r"HIS\s+system\s+auto\s+export|"
    r"EMR\s+system\s+auto\s+export|"
    r"\?{1,}\d{4}-\d{1,2}-\d{1,2}\s*\?*",
    re.IGNORECASE,
)
QUESTION_MOJIBAKE_RE = re.compile(r"(?<![\u4e00-\u9fffA-Za-z0-9])\?{2,}(?![\u4e00-\u9fffA-Za-z0-9])")
MARKUP_RUN_RE = re.compile(r"(?:@@@|###|[-=]{4,})")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def clean_text_value(value: str) -> str:
    text = CONTROLLED_NOISE_RE.sub(" ", value)
    text = QUESTION_MOJIBAKE_RE.sub(" ", text)
    text = MARKUP_RUN_RE.sub(" ", text)
    cjk_end = r"(?=[\u4e00-\u9fff\s\"'，,。；;：:\)）\]】]|$)"
    text = re.sub(r"(?<=[\u4e00-\u9fff])\?" + cjk_end, "？", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])!" + cjk_end, "！", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def _clean_json(value: Any, parent_key: str = "") -> Any:
    if isinstance(value, dict):
        return {key: _clean_json(child, str(key)) for key, child in value.items() if str(key) not in DROP_FIELDS}
    if isinstance(value, list):
        return [_clean_json(child, parent_key) for child in value]
    if isinstance(value, str):
        return clean_text_value(value)
    return value


def cleanup_preserved_file(path: Path) -> dict:
    suffix = path.suffix.lower()
    before = path.read_text(encoding="utf-8", errors="replace")
    after = before
    dropped_fields: list[str] = []

    if suffix == ".csv":
        reader = csv.DictReader(StringIO(before))
        rows = list(reader)
        fieldnames = [name for name in (reader.fieldnames or []) if name not in DROP_FIELDS]
        dropped_fields = [name for name in (reader.fieldnames or []) if name in DROP_FIELDS]
        out = StringIO()
        writer = csv.DictWriter(out, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: clean_text_value(row.get(name, "")) for name in fieldnames})
        after = out.getvalue()
    elif suffix == ".json":
        obj = json.loads(before)
        after = json.dumps(_clean_json(obj), ensure_ascii=False, indent=2)
    elif suffix == ".jsonl":
        lines = []
        for line in before.splitlines():
            if not line.strip():
                continue
            lines.append(json.dumps(_clean_json(json.loads(line)), ensure_ascii=False))
        after = "\n".join(lines) + ("\n" if lines else "")
    elif suffix == ".txt":
        after = clean_text_value(before)

    if after != before:
        path.write_text(after, encoding="utf-8")

    return {
        "file": path.name,
        "changed": after != before,
        "dropped_fields": dropped_fields,
        "char_delta": len(after) - len(before),
    }


def cleanup_preserved_files(paths: list[Path]) -> list[dict]:
    return [cleanup_preserved_file(path) for path in paths]
