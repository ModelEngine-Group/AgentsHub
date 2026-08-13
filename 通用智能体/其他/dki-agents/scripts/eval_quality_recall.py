"""评测可选质量高召回双模型模式。

该模式故意不并入 scripts/eval_ood.py 的通用单模型 CLI，避免影响历史结果：
quality 主模型 + 冻结原 BERT 互补召回 + 纯 Python 质量规则。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_ood import _gold_spans, _pred_spans, _prf_span


def main() -> None:
    parser = argparse.ArgumentParser(description="评测 quality 高召回双模型模式")
    parser.add_argument("--ood", required=True)
    parser.add_argument("--quality-model", default="models/medical_ner_quality")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from dki.bench.npu.bert_ner_infer import BertNerBackend

    rows = [json.loads(line) for line in Path(args.ood).open(encoding="utf-8") if line.strip()]
    backend = BertNerBackend(
        model_name=args.quality_model,
        device="cpu",
        quality_recall=True,
    )
    texts = [row["text"] for row in rows]
    predictions = backend.extract(texts)
    gold = {row["doc_id"]: _gold_spans(row["text"], row["entities"]) for row in rows}
    pred = {
        row["doc_id"]: _pred_spans(row["text"], entities)
        for row, entities in zip(rows, predictions)
    }
    result = {
        "n_docs": len(rows),
        "n_gold_entities": sum(len(value) for value in gold.values()),
        "match": "span-overlap+type",
        "mode": "quality+baseline-rescue+medical-rules",
        "quality_model": str(args.quality_model),
        "quality_recall": _prf_span(pred, gold),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
