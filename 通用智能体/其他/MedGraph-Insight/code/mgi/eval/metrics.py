"""量化评测指标：精确率 / 召回率 / F1。

- NER 采用「实体级」评估：以 (规范名, 实体类型) 为单位比对，忽略偏移，
  贴近医疗信息抽取实践（同一实体多次提及只计一次）。
- 关系抽取采用「三元组级」评估：以 (头, 关系, 尾) 规范三元组比对。

金标准文件：data/eval/kg_gold.jsonl，每行
  {"id", "entities": [[text, etype], ...], "relations": [[head, rel, tail], ...]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

GOLD_PATH = Path(__file__).resolve().parents[2] / "data" / "eval" / "kg_gold.jsonl"


@dataclass
class PRF1:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn,
                "precision": round(self.precision, 4),
                "recall": round(self.recall, 4), "f1": round(self.f1, 4)}


def prf1(pred: Iterable[tuple], gold: Iterable[tuple]) -> PRF1:
    p, g = set(pred), set(gold)
    return PRF1(tp=len(p & g), fp=len(p - g), fn=len(g - p))


def load_gold(path: Path | str = GOLD_PATH) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def evaluate_ner(pred_by_doc: dict[str, list[tuple[str, str]]],
                 gold: dict[str, dict] | None = None) -> PRF1:
    gold = gold or load_gold()
    agg = PRF1()
    for did, rec in gold.items():
        gset = {tuple(x) for x in rec.get("entities", [])}
        pset = set(pred_by_doc.get(did, []))
        r = prf1(pset, gset)
        agg.tp += r.tp
        agg.fp += r.fp
        agg.fn += r.fn
    return agg


def evaluate_re(pred_by_doc: dict[str, list[tuple[str, str, str]]],
                gold: dict[str, dict] | None = None) -> PRF1:
    gold = gold or load_gold()
    agg = PRF1()
    for did, rec in gold.items():
        gset = {tuple(x) for x in rec.get("relations", [])}
        pset = set(pred_by_doc.get(did, []))
        r = prf1(pset, gset)
        agg.tp += r.tp
        agg.fp += r.fp
        agg.fn += r.fn
    return agg
