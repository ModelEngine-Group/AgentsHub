"""抽取质量评测：流水线产出三元组 vs 金标准（P/R/F1，按关系类型细分）。"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

from dki.paths import GOLD_DATA_DIR

TripleKey = Tuple[str, str, str]


def _norm_key(t: Dict[str, Any]) -> TripleKey:
    return (str(t["head"]).strip(), str(t["relation"]).strip(), str(t["tail"]).strip())


def load_gold_triples(gold_dir: Path = None) -> Dict[str, Set[TripleKey]]:
    """加载金标准：{doc_id: {(head, relation, tail)}}"""
    gold_dir = Path(gold_dir or (GOLD_DATA_DIR / "triples"))
    gold: Dict[str, Set[TripleKey]] = {}
    for f in sorted(gold_dir.glob("*.json")):
        rows = json.loads(f.read_text(encoding="utf-8"))
        gold[f.stem] = {_norm_key(t) for t in rows}
    return gold


def evaluate_triples(
    predicted_by_doc: Dict[str, List[Dict[str, Any]]],
    gold_dir: Path = None,
) -> Dict[str, Any]:
    """微平均 P/R/F1 + 按关系类型细分。

    predicted_by_doc: {doc_id: [triple dict]}（doc_id 与 gold 文件名 stem 对齐，如 emr_0001）
    """
    gold = load_gold_triples(gold_dir)
    tp = fp = fn = 0
    by_rel: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    docs_evaluated = 0
    for doc_id, gold_set in gold.items():
        pred_list = predicted_by_doc.get(doc_id, [])
        pred_set = {_norm_key(t) for t in pred_list}
        docs_evaluated += 1
        for k in pred_set:
            if k in gold_set:
                tp += 1
                by_rel[k[1]]["tp"] += 1
            else:
                fp += 1
                by_rel[k[1]]["fp"] += 1
        for k in gold_set - pred_set:
            fn += 1
            by_rel[k[1]]["fn"] += 1

    def _prf(tp_, fp_, fn_):
        p = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        r = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return round(p, 4), round(r, 4), round(f1, 4)

    p, r, f1 = _prf(tp, fp, fn)
    detail = {}
    for rel, c in sorted(by_rel.items()):
        rp, rr, rf1 = _prf(c["tp"], c["fp"], c["fn"])
        detail[rel] = {
            "precision": rp, "recall": rr, "f1": rf1,
            "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
        }
    return {
        "docs": docs_evaluated,
        "precision": p,
        "recall": r,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "by_relation": detail,
    }


def predictions_from_samples(samples: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """把流水线输出样本整理成 {doc_id: [triples]}（doc_id 取 fileId）。"""
    from dki.kg.build import extract_triples_from_sample

    out: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        doc_id = str(s.get("fileId", ""))
        out[doc_id] = extract_triples_from_sample(s)
    return out


def format_eval_report(result: Dict[str, Any]) -> str:
    lines = [
        f"评测文档数: {result['docs']}",
        f"微平均  P={result['precision']:.2%}  R={result['recall']:.2%}  F1={result['f1']:.2%}"
        f"  (TP={result['tp']} FP={result['fp']} FN={result['fn']})",
        "按关系类型：",
    ]
    from dki.kg.schema import relation_zh

    for rel, d in result["by_relation"].items():
        lines.append(
            f"  {rel}({relation_zh(rel)}): P={d['precision']:.2%} R={d['recall']:.2%} "
            f"F1={d['f1']:.2%} (tp={d['tp']} fp={d['fp']} fn={d['fn']})"
        )
    return "\n".join(lines)
