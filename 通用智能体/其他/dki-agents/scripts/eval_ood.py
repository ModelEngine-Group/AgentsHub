"""OOD 鲁棒性评测：分布外病历上，各后端实体级 F1（span-overlap 匹配）。

回答"过拟合"问题：train/eval 同源于合成生成器 + 词典，held-out F1 测的是同分布泛化。
本脚本在**外部 LLM 生成、口语化、独立标注**的分布外病历上评测真实鲁棒性。

匹配口径（关键）：分布外文本里同一实体有大量非规范表述（"没睡好"=失眠、"高压"=高血压），
LLM 金标的规范 name 常不逐字出现在原文，故**不能用 (规范名,类型) 精确匹配**——那会把
"抽对了但写法不同"误判为漏报，严重低估召回。改用 **span-overlap 匹配**：金标 mention 是
原文逐字子串（gen_ood_eval 已保证），预测实体的字符跨度与某条同类型金标 mention 跨度
**有重叠**即算命中（标准 NER 宽松匹配）。同时给出 type-only 上界（同类型实体计数召回）
作参照。

用法（服务器，有 torch）：
    python3 scripts/eval_ood.py --ood data/ood/ood_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.nlp.evaluation import evaluate_both, evaluate_spans, explicit_spans

Span = Tuple[int, int, str]  # (start, end, type)


def _gold_spans(text: str, entities: List[dict]) -> List[Span]:
    """金标 mention 逐字定位到字符跨度（取第一个未占用出现位置）。"""
    spans: List[Span] = []
    used: List[Tuple[int, int]] = []
    for e in entities:
        m = e.get("mention") or e.get("name")
        if not m:
            continue
        start = 0
        while True:
            idx = text.find(m, start)
            if idx < 0:
                break
            seg = (idx, idx + len(m))
            if not any(not (seg[1] <= u0 or seg[0] >= u1) for u0, u1 in used):
                spans.append((seg[0], seg[1], e["type"]))
                used.append(seg)
                break
            start = idx + 1
    return spans


def _pred_spans(text: str, ents: List[dict]) -> List[Span]:
    """预测实体的字符跨度：优先用 span 字段，否则按 mention/name 在原文定位。"""
    spans: List[Span] = []
    for e in ents:
        if e.get("span") and isinstance(e["span"], (list, tuple)) and len(e["span"]) == 2:
            spans.append((int(e["span"][0]), int(e["span"][1]), e["type"]))
            continue
        m = e.get("mention") or e.get("name")
        if not m:
            continue
        idx = text.find(m)
        if idx >= 0:
            spans.append((idx, idx + len(m), e["type"]))
    return spans


def _overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or a[0] >= b[1])


def _prf_span(pred_by_doc: Dict[str, List[Span]], gold_by_doc: Dict[str, List[Span]]) -> dict:
    """span-overlap + 同类型 匹配的微平均 P/R/F1（贪心一对一）。"""
    tp = fp = fn = 0
    for did, gold in gold_by_doc.items():
        pred = list(pred_by_doc.get(did, []))
        matched_gold = [False] * len(gold)
        for ps, pe, pt in pred:
            hit = False
            for gi, (gs, ge, gt) in enumerate(gold):
                if not matched_gold[gi] and pt == gt and _overlap((ps, pe), (gs, ge)):
                    matched_gold[gi] = True
                    hit = True
                    break
            tp += 1 if hit else 0
            fp += 0 if hit else 1
        fn += matched_gold.count(False)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


def main() -> None:
    ap = argparse.ArgumentParser(description="OOD 鲁棒性评测（exact 主指标；overlap 诊断）")
    ap.add_argument("--ood", default=str(ROOT / "data" / "ood" / "ood_eval.jsonl"))
    ap.add_argument("--models", nargs="*",
                    default=["models/medical_ner_bert", "models/medical_ner_rbt6"])
    ap.add_argument("--output", default=str(ROOT / "outputs" / "ner_finetune" / "ood_eval.json"))
    ap.add_argument(
        "--strict",
        action="store_true",
        help="要求金标显式 span，并以 exact span+type 为主指标；默认保持历史 overlap 输出",
    )
    args = ap.parse_args()

    from dki.operators.medical_entity_extractor.process import _resolve_spans, load_vocab

    vocab = load_vocab()
    rows = [json.loads(l) for l in open(args.ood, encoding="utf-8") if l.strip()]
    doc_ids = [r["doc_id"] for r in rows]
    texts = {r["doc_id"]: r["text"] for r in rows}

    if args.strict:
        gold_by_doc = {
            r["doc_id"]: explicit_spans(r["text"], r["entities"])
            for r in rows
        }
    else:
        gold_by_doc = {r["doc_id"]: _gold_spans(r["text"], r["entities"]) for r in rows}
    n_gold = sum(len(g) for g in gold_by_doc.values())
    results = {
        "n_docs": len(rows),
        "n_gold_entities": n_gold,
        "match": "exact_span+type" if args.strict else "span-overlap+type",
        "protocol": "strict-explicit-spans" if args.strict else "historical-relaxed-overlap",
    }

    def score(prediction):
        return evaluate_both(prediction, gold_by_doc) if args.strict else _prf_span(prediction, gold_by_doc)

    # dict 后端（_resolve_spans 已带 span）
    dict_pred = {}
    for did in doc_ids:
        t = texts[did]
        hits = [w for w in vocab if w in t]
        ents = _resolve_spans(t, hits, vocab)
        dict_pred[did] = [(int(e["span"][0]), int(e["span"][1]), e["type"]) for e in ents]
    results["dict"] = score(dict_pred)
    print(f"dict           {results['dict']}")

    from dki.bench.npu.bert_ner_infer import BertNerBackend
    for mdl in args.models:
        if not Path(mdl).exists():
            print(f"[skip] {mdl} 不存在")
            continue
        backend = BertNerBackend(model_name=mdl, device="cpu")
        pred = {}
        for did in doc_ids:
            ents = backend.extract([texts[did]])[0]
            pred[did] = _pred_spans(texts[did], ents)
        tag = Path(mdl).name.replace("medical_ner_", "")
        results[tag] = score(pred)
        print(f"{tag:14s} {results[tag]}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] → {out}")


if __name__ == "__main__":
    main()
