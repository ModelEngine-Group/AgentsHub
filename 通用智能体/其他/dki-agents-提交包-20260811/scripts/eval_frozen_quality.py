"""一次性比较冻结质量模型，不对评测集做任何适配或选模。

输出 dict、原 BERT、单 quality 和可选双模型的同口径 span-overlap 指标以及逐类型误差。
该脚本只评测给定 checkpoint；checkpoint 选择必须在 train/dev/internal-holdout 完成。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_ood import _gold_spans, _overlap, _pred_spans, _prf_span

Span = Tuple[int, int, str]


def _errors(pred_by_doc: Dict[str, List[Span]],
            gold_by_doc: Dict[str, List[Span]]) -> dict:
    tp = Counter()
    fp = Counter()
    fn = Counter()
    for doc_id, gold in gold_by_doc.items():
        matched = [False] * len(gold)
        for ps, pe, pred_type in pred_by_doc.get(doc_id, []):
            hit = None
            for index, (gs, ge, gold_type) in enumerate(gold):
                if not matched[index] and pred_type == gold_type and _overlap((ps, pe), (gs, ge)):
                    hit = index
                    break
            if hit is None:
                fp[pred_type] += 1
            else:
                matched[hit] = True
                tp[pred_type] += 1
        for is_matched, (_, _, gold_type) in zip(matched, gold):
            if not is_matched:
                fn[gold_type] += 1
    by_type = {}
    all_types = sorted(set(tp) | set(fp) | set(fn))
    for entity_type in all_types:
        t, f_p, f_n = tp[entity_type], fp[entity_type], fn[entity_type]
        precision = t / (t + f_p) if t + f_p else 0.0
        recall = t / (t + f_n) if t + f_n else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        by_type[entity_type] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": t,
            "fp": f_p,
            "fn": f_n,
        }
    return by_type


def _score(pred: Dict[str, List[Span]], gold: Dict[str, List[Span]]) -> dict:
    return {**_prf_span(pred, gold), "by_type": _errors(pred, gold)}


def main() -> None:
    parser = argparse.ArgumentParser(description="一次性评测冻结 quality checkpoint")
    parser.add_argument("--ood", required=True)
    parser.add_argument("--quality-model", required=True)
    parser.add_argument("--baseline-model", default="models/medical_ner_bert")
    parser.add_argument("--output", required=True)
    parser.add_argument("--with-recall-mode", action="store_true")
    args = parser.parse_args()

    from dki.bench.npu.bert_ner_infer import BertNerBackend
    from dki.operators.medical_entity_extractor.process import _resolve_spans, load_vocab

    rows = [json.loads(line) for line in Path(args.ood).open(encoding="utf-8") if line.strip()]
    texts = [row["text"] for row in rows]
    gold = {row["doc_id"]: _gold_spans(row["text"], row["entities"]) for row in rows}
    result = {
        "n_docs": len(rows),
        "n_gold_entities": sum(len(spans) for spans in gold.values()),
        "match": "span-overlap+type",
        "protocol": "checkpoint/rules frozen before this one-shot evaluation",
        "quality_model": args.quality_model,
    }

    vocab = load_vocab()
    dict_pred = {}
    for row in rows:
        hits = [word for word in vocab if word in row["text"]]
        entities = _resolve_spans(row["text"], hits, vocab)
        dict_pred[row["doc_id"]] = _pred_spans(row["text"], entities)
    result["dict"] = _score(dict_pred, gold)

    for name, model_path, recall_mode in [
        ("baseline", args.baseline_model, False),
        ("quality", args.quality_model, False),
        ("quality_recall", args.quality_model, True),
    ]:
        if recall_mode and not args.with_recall_mode:
            continue
        backend = BertNerBackend(
            model_name=model_path,
            device="cpu",
            quality_recall=recall_mode,
        )
        predictions = backend.extract(texts)
        spans = {
            row["doc_id"]: _pred_spans(row["text"], entities)
            for row, entities in zip(rows, predictions)
        }
        result[name] = _score(spans, gold)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
