"""在语料上运行 NER/RE 流水线并对照金标准计算 P/R/F1。"""

from __future__ import annotations

import json
from pathlib import Path

from ..operators.entity_extract import EntityExtractOperator
from ..operators.relation_extract import RelationExtractOperator
from ..operators.triple_builder import TripleBuildOperator
from .metrics import evaluate_ner, evaluate_re, load_gold


def run_eval(corpus_path: str | Path, gold_path: str | Path | None = None) -> dict:
    gold = load_gold(gold_path) if gold_path else load_gold()
    ee = EntityExtractOperator()
    re_op = RelationExtractOperator()
    tb = TripleBuildOperator()

    ner_pred: dict[str, list[tuple[str, str]]] = {}
    re_pred: dict[str, list[tuple[str, str, str]]] = {}
    for line in Path(corpus_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        did = d.get("id", "")
        if did not in gold:
            continue
        ents = ee.process([d["text"]])[0]
        triples = tb.process(re_op.process([(d["text"], ents)])[0])
        ner_pred[did] = sorted({(e.canonical, e.etype) for e in ents})
        re_pred[did] = sorted({(t.head, t.relation, t.tail) for t in triples})

    ner = evaluate_ner(ner_pred, gold)
    rel = evaluate_re(re_pred, gold)
    return {
        "docs": len(gold),
        "ner": ner.as_dict(),
        "re": rel.as_dict(),
        "_ner_pred": ner_pred,
        "_re_pred": re_pred,
    }
