"""Measure task-2 relation extraction quality (precision / recall / F1).

Predicted relation triples are evaluated against a hand-annotated relation gold
corpus (``benchmarks/data/kg_relation_gold.json``) that is annotated by medical
correctness, not by mirroring the extractor. The selected ``--backend``
(``rule`` / ``cpu`` / ``npu``) lets the same metric confirm that the tensorized
NPU relation scoring produces identical relation-level P/R/F1 as the rule path.

Usage:
    python benchmarks/task2_relation_quality_benchmark.py
    python benchmarks/task2_relation_quality_benchmark.py --backend npu \\
        --report benchmarks/reports/task2_relation_quality_ascend_910b2c.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.operators.kg_ops import evaluate_relation_quality

DEFAULT_GOLD = ROOT / "benchmarks" / "data" / "kg_relation_gold.json"
# The co-occurrence extractor over-pairs in multi-disease records, so a
# perfect precision is not expected; recall should stay high.
PRECISION_THRESHOLD = 0.90
RECALL_THRESHOLD = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark task-2 relation extraction quality.")
    parser.add_argument("--gold", default=str(DEFAULT_GOLD), help="Relation gold corpus JSON path.")
    parser.add_argument(
        "--backend",
        default="rule",
        choices=["rule", "cpu", "npu"],
        help="Relation backend: rule (default), cpu (tensorized), npu (Ascend tensorized).",
    )
    parser.add_argument("--report", default=None, help="Optional JSON report output path.")
    return parser.parse_args()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def main() -> int:
    args = parse_args()
    gold_path = Path(args.gold)
    data = json.loads(gold_path.read_text(encoding="utf-8"))
    gold_records = data["records"]

    report = evaluate_relation_quality(gold_records, backend=args.backend)
    report["benchmark"] = _relative(gold_path)
    report["thresholds"] = {"precision": PRECISION_THRESHOLD, "recall": RECALL_THRESHOLD}
    overall = report["overall"]
    report["passed"] = (
        overall["precision"] >= PRECISION_THRESHOLD and overall["recall"] >= RECALL_THRESHOLD
    )

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
