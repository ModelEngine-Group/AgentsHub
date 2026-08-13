"""实体级 NER 评测：预测实体 vs 金标实体（P/R/F1，按类型细分）。

口径与 dki.kg.eval 一致（微平均 + 按类别细分），但比较的是实体 (name, type)
集合而非三元组。用于：
- scripts/train_ner.py 的 compute_metrics（eval 集真实泛化 F1）；
- Notebook 的三后端对比（dict / 微调 bert / 未微调 bert）；
- scripts/eval_om_ner.py 的 OM 量化前后精度对比。

金标 entities 的 name 是规范名（span=null），故按 (name, type) 集合比较——
与 dict/bert 后端输出的 name（已归一到规范名）口径一致。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from dki.paths import GOLD_DATA_DIR

EntityKey = Tuple[str, str]  # (name, type)


def _norm_entity(e: Dict[str, Any]) -> EntityKey:
    return (str(e["name"]).strip(), str(e["type"]).strip())


def load_gold_entities(gold_dir: Path = None) -> Dict[str, Set[EntityKey]]:
    """加载金标实体：{doc_id: {(name, type)}}。"""
    gold_dir = Path(gold_dir or (GOLD_DATA_DIR / "entities"))
    gold: Dict[str, Set[EntityKey]] = {}
    for f in sorted(gold_dir.glob("*.json")):
        rows = json.loads(f.read_text(encoding="utf-8"))
        gold[f.stem] = {_norm_entity(e) for e in rows}
    return gold


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


def evaluate_entities(
    predicted_by_doc: Dict[str, List[Dict[str, Any]]],
    gold_dir: Path = None,
    doc_ids: Sequence[str] = None,
) -> Dict[str, Any]:
    """微平均实体级 P/R/F1 + 按实体类型细分。

    predicted_by_doc: {doc_id: [entity dict with name/type]}，doc_id 对齐金标 stem。
    doc_ids: 若给定，只在该文档子集上评测（如 eval 集 40 doc，避免训练集金标算作漏报）。
    """
    gold = load_gold_entities(gold_dir)
    if doc_ids is not None:
        wanted = set(doc_ids)
        gold = {d: s for d, s in gold.items() if d in wanted}
    tp = fp = fn = 0
    by_type: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    docs_evaluated = 0
    for doc_id, gold_set in gold.items():
        pred_set = {_norm_entity(e) for e in predicted_by_doc.get(doc_id, [])}
        docs_evaluated += 1
        for k in pred_set:
            if k in gold_set:
                tp += 1
                by_type[k[1]]["tp"] += 1
            else:
                fp += 1
                by_type[k[1]]["fp"] += 1
        for k in gold_set - pred_set:
            fn += 1
            by_type[k[1]]["fn"] += 1

    p, r, f1 = _prf(tp, fp, fn)
    detail = {}
    for etype, c in sorted(by_type.items()):
        ep, er, ef1 = _prf(c["tp"], c["fp"], c["fn"])
        detail[etype] = {"precision": ep, "recall": er, "f1": ef1, **c}
    return {
        "docs": docs_evaluated,
        "precision": p,
        "recall": r,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "by_type": detail,
    }


def prf_from_pairs(
    pred_pairs: Iterable[EntityKey], gold_pairs: Iterable[EntityKey]
) -> Dict[str, float]:
    """toy/单元测试用：直接对两组 (name,type) 集合算 P/R/F1。"""
    pred, gold = set(pred_pairs), set(gold_pairs)
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p, r, f1 = _prf(tp, fp, fn)
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def format_entity_eval(res: Dict[str, Any]) -> str:
    """人类可读评测报告。"""
    lines = [
        f"文档数: {res['docs']}  实体级微平均  "
        f"P={res['precision']:.4f}  R={res['recall']:.4f}  F1={res['f1']:.4f}  "
        f"(tp={res['tp']} fp={res['fp']} fn={res['fn']})",
        "",
        "| 实体类型 | P | R | F1 | tp | fp | fn |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for etype, c in res.get("by_type", {}).items():
        lines.append(
            f"| {etype} | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1']:.4f} "
            f"| {c['tp']} | {c['fp']} | {c['fn']} |"
        )
    return "\n".join(lines)
