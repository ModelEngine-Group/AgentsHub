# -*- coding: utf-8 -*-
"""把重复出现且通过安全检查的清洗差异沉淀为确定性噪声规则。"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


SAFE_CATEGORIES = {
    "chitchat",
    "colloquial",
    "mention",
    "system_export",
    "form_metadata",
    "work_instruction",
}

_MEDICAL_CONTENT_RE = re.compile(
    r"患者|主诉|现病史|既往史|诊断|症状|疼痛|发热|咳嗽|胸闷|胸痛|"
    r"否认|用药|口服|静滴|检查|检验|血压|血糖|心率|体温|"
    r"mmHg|mmol/L|mg|mL|U/L|μmol/L|%|\d",
    re.IGNORECASE,
)
_SPACE_RE = re.compile(r"[ \t]+")


def _normalize_segment(value: str) -> str:
    lines = [_SPACE_RE.sub(" ", line).strip() for line in str(value).splitlines()]
    return "\n".join(line for line in lines if line)


def _safe_segment(segment: str) -> tuple[bool, str]:
    if len(segment) < 6:
        return False, "too_short"
    if len(segment) > 120:
        return False, "too_long"
    if _MEDICAL_CONTENT_RE.search(segment):
        return False, "medical_content"
    return True, ""


def _ensure_rule_columns(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(noise_rules)")}
    for name, declaration in (
        ("source_type", "TEXT DEFAULT 'manual_controlled'"),
        ("source_ref", "TEXT DEFAULT ''"),
        ("evidence", "TEXT DEFAULT ''"),
    ):
        if name not in columns:
            conn.execute(f"ALTER TABLE noise_rules ADD COLUMN {name} {declaration}")


def distill_repeated_segments(
    audit_db_path: str | Path,
    rule_db_path: str | Path,
    *,
    min_occurrences: int = 3,
) -> dict[str, Any]:
    """从审计日志中提取重复差异，写入可审计的精确匹配规则。"""

    audit_path = Path(audit_db_path)
    rule_path = Path(rule_db_path)
    if min_occurrences < 2:
        raise ValueError("min_occurrences must be at least 2")
    if not audit_path.exists() or not rule_path.exists():
        raise FileNotFoundError("audit or rule database does not exist")

    with sqlite3.connect(audit_path) as audit_conn:
        rows = audit_conn.execute(
            "SELECT removed_text, category FROM noise_diffs "
            "WHERE removed_text IS NOT NULL AND trim(removed_text) <> ''"
        ).fetchall()

    counts: Counter[tuple[str, str]] = Counter()
    for removed_text, category in rows:
        segment = _normalize_segment(removed_text)
        counts[(segment, str(category or "unknown"))] += 1

    promoted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    with sqlite3.connect(rule_path) as rule_conn:
        _ensure_rule_columns(rule_conn)
        existing = {
            row[0]: {"rule_id": row[1], "match_type": row[2]}
            for row in rule_conn.execute(
                "SELECT pattern, rule_id, match_type FROM noise_rules"
            ).fetchall()
        }
        for (segment, category), occurrences in sorted(counts.items()):
            if occurrences < min_occurrences:
                continue
            if category not in SAFE_CATEGORIES:
                skipped.append({"segment": segment, "reason": "unsafe_category"})
                continue
            safe, reason = _safe_segment(segment)
            if not safe:
                skipped.append({"segment": segment, "reason": reason})
                continue
            digest = hashlib.sha1(segment.encode("utf-8")).hexdigest()[:12]
            evidence = json.dumps(
                {"occurrences": occurrences, "category": category},
                ensure_ascii=False,
                sort_keys=True,
            )
            existing_rule = existing.get(segment)
            if existing_rule and existing_rule["match_type"] != "semantic_hint":
                skipped.append({"segment": segment, "reason": "already_exists"})
                continue
            rule_id = existing_rule["rule_id"] if existing_rule else f"kd_{digest}"
            if existing_rule:
                rule_conn.execute(
                    "UPDATE noise_rules SET category=?, match_type='exact', scope='match', "
                    "confidence=0.96, medical_safe=1, status='active', "
                    "source_type='distilled_from_audit', source_ref=?, evidence=? "
                    "WHERE pattern=?",
                    (category, f"noise_diffs:{occurrences}", evidence, segment),
                )
            else:
                rule_conn.execute(
                    "INSERT INTO noise_rules "
                    "(rule_id, category, pattern, match_type, scope, confidence, "
                    "medical_safe, status, negative_patterns, source_type, source_ref, evidence) "
                    "VALUES (?, ?, ?, 'exact', 'match', 0.96, 1, 'active', '', "
                    "'distilled_from_audit', ?, ?)",
                    (rule_id, category, segment, f"noise_diffs:{occurrences}", evidence),
                )
            existing[segment] = {"rule_id": rule_id, "match_type": "exact"}
            promoted.append(
                {
                    "rule_id": rule_id,
                    "category": category,
                    "pattern": segment,
                    "occurrences": occurrences,
                }
            )
        rule_conn.commit()

    return {
        "observed_diffs": len(rows),
        "min_occurrences": min_occurrences,
        "promoted_count": len(promoted),
        "promoted": promoted,
        "skipped": skipped,
    }
