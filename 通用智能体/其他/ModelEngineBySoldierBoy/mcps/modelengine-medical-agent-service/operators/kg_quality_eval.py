"""Gold-label quality evaluation for medical KG triples.

The evaluator is intentionally fact-level: it compares normalized
subject-predicate-object triples and reports precision/recall/F1 by relation.
It is used in the task-2 agent loop to make extraction quality measurable on
official CBLUE-derived labels instead of relying only on graph size.
"""

from __future__ import annotations

from collections import Counter
import json
import re
from pathlib import Path
from typing import Any

from operators.kg_normalization import canonicalize_entity


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(payload)
    return rows


def _clean_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", "", text)
    return text.strip("，。；;、:：,.()（）[]【】")


def _fact_key(triple: dict[str, Any]) -> tuple[str, str, str, str, str]:
    subject_type = str(triple.get("subject_type", "")).strip()
    object_type = str(triple.get("object_type", "")).strip()
    subject, _ = canonicalize_entity(str(triple.get("subject", "")), subject_type)
    object_name, _ = canonicalize_entity(str(triple.get("object", "")), object_type)
    return (
        _clean_name(subject),
        subject_type,
        str(triple.get("predicate", "")).strip(),
        _clean_name(object_name),
        object_type,
    )


def _fact_relation(key: tuple[str, str, str, str, str]) -> str:
    return key[2]


def _fact_to_dict(key: tuple[str, str, str, str, str]) -> dict[str, str]:
    return {
        "subject": key[0],
        "subject_type": key[1],
        "predicate": key[2],
        "object": key[3],
        "object_type": key[4],
    }


def _metrics(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate_triple_facts(
    predicted_triples: list[dict[str, Any]],
    gold_triples: list[dict[str, Any]],
    *,
    sample_limit: int = 20,
) -> dict[str, Any]:
    """Compare predicted triples against gold triples at normalized fact level."""
    predicted = {_fact_key(item) for item in predicted_triples if str(item.get("predicate", "")).strip()}
    gold = {_fact_key(item) for item in gold_triples if str(item.get("predicate", "")).strip()}

    true_positive = predicted & gold
    false_positive = predicted - gold
    false_negative = gold - predicted
    relation_names = sorted({_fact_relation(key) for key in predicted | gold if _fact_relation(key)})

    by_relation: dict[str, Any] = {}
    for relation in relation_names:
        pred_rel = {key for key in predicted if _fact_relation(key) == relation}
        gold_rel = {key for key in gold if _fact_relation(key) == relation}
        tp = len(pred_rel & gold_rel)
        fp = len(pred_rel - gold_rel)
        fn = len(gold_rel - pred_rel)
        by_relation[relation] = {
            **_metrics(tp, fp, fn),
            "predicted": len(pred_rel),
            "gold": len(gold_rel),
        }

    predicted_counts = Counter(_fact_relation(key) for key in predicted)
    gold_counts = Counter(_fact_relation(key) for key in gold)
    return {
        "predicted_fact_count": len(predicted),
        "gold_fact_count": len(gold),
        "matched_fact_count": len(true_positive),
        "overall": _metrics(len(true_positive), len(false_positive), len(false_negative)),
        "by_relation": by_relation,
        "relation_coverage": {
            "predicted_relation_count": len([key for key, value in predicted_counts.items() if key and value]),
            "gold_relation_count": len([key for key, value in gold_counts.items() if key and value]),
            "covered_gold_relation_count": len(
                [relation for relation in gold_counts if relation and predicted_counts.get(relation, 0) > 0]
            ),
            "predicted_relation_counts": dict(sorted(predicted_counts.items())),
            "gold_relation_counts": dict(sorted(gold_counts.items())),
        },
        "samples": {
            "false_positive": [_fact_to_dict(key) for key in sorted(false_positive)[:sample_limit]],
            "false_negative": [_fact_to_dict(key) for key in sorted(false_negative)[:sample_limit]],
            "true_positive": [_fact_to_dict(key) for key in sorted(true_positive)[:sample_limit]],
        },
    }


def evaluate_triples_against_gold_file(
    predicted_triples: list[dict[str, Any]],
    gold_path: Path | None,
    *,
    sample_limit: int = 20,
) -> dict[str, Any]:
    if gold_path is None:
        return {
            "enabled": False,
            "reason": "gold_path_missing",
        }
    gold_rows = _read_jsonl(gold_path)
    if not gold_rows:
        return {
            "enabled": False,
            "reason": "gold_file_empty_or_missing",
            "gold_path": str(gold_path),
        }
    result = evaluate_triple_facts(predicted_triples, gold_rows, sample_limit=sample_limit)
    result.update(
        {
            "enabled": True,
            "gold_path": str(gold_path),
            "evaluation_mode": "normalized_subject_predicate_object",
        }
    )
    return result


def build_quality_eval_report(evaluation: dict[str, Any]) -> str:
    lines = ["# KG 标注级质量评估", ""]
    if not evaluation.get("enabled"):
        lines.append(f"- 状态：未启用（{evaluation.get('reason', 'unknown')}）")
        return "\n".join(lines).strip() + "\n"

    overall = evaluation.get("overall", {})
    coverage = evaluation.get("relation_coverage", {})
    lines.extend(
        [
            f"- Gold 文件：`{evaluation.get('gold_path')}`",
            f"- 评估模式：{evaluation.get('evaluation_mode')}",
            f"- 预测事实数：{evaluation.get('predicted_fact_count', 0)}",
            f"- Gold 事实数：{evaluation.get('gold_fact_count', 0)}",
            f"- 命中事实数：{evaluation.get('matched_fact_count', 0)}",
            f"- Precision：{overall.get('precision', 0):.2%}",
            f"- Recall：{overall.get('recall', 0):.2%}",
            f"- F1：{overall.get('f1', 0):.2%}",
            f"- Gold 关系覆盖：{coverage.get('covered_gold_relation_count', 0)}/{coverage.get('gold_relation_count', 0)}",
            "",
            "## 关系维度",
            "",
            "| 关系 | 预测 | Gold | P | R | F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for relation, item in sorted(evaluation.get("by_relation", {}).items()):
        lines.append(
            f"| `{relation}` | {item.get('predicted', 0)} | {item.get('gold', 0)} | "
            f"{item.get('precision', 0):.2%} | {item.get('recall', 0):.2%} | {item.get('f1', 0):.2%} |"
        )

    samples = evaluation.get("samples", {})
    lines.extend(["", "## 漏召样例", ""])
    false_negative = samples.get("false_negative", [])
    if false_negative:
        lines.extend(["| 主语 | 关系 | 宾语 |", "| --- | --- | --- |"])
        for item in false_negative[:10]:
            lines.append(f"| {item['subject']} | `{item['predicate']}` | {item['object']} |")
    else:
        lines.append("无。")

    lines.extend(["", "## 误召样例", ""])
    false_positive = samples.get("false_positive", [])
    if false_positive:
        lines.extend(["| 主语 | 关系 | 宾语 |", "| --- | --- | --- |"])
        for item in false_positive[:10]:
            lines.append(f"| {item['subject']} | `{item['predicate']}` | {item['object']} |")
    else:
        lines.append("无。")

    return "\n".join(lines).strip() + "\n"
