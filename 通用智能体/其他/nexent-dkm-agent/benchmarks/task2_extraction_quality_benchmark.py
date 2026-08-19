"""Measure task-2 extraction quality (precision / recall / F1) on held-out data.

The rule-based extractor is evaluated against a hand-annotated corpus
(``benchmarks/data/kg_extraction_gold.json``) that is independent from the
bundled sample and the fine-tuning data. This turns the qualitative
"extraction is reasonable" claim into a reproducible quantitative metric.

Usage:
    python benchmarks/task2_extraction_quality_benchmark.py
    python benchmarks/task2_extraction_quality_benchmark.py --report benchmarks/reports/task2_kg_extraction_quality.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.operators.kg_ops import evaluate_extraction_quality

DEFAULT_GOLD = ROOT / "benchmarks" / "data" / "kg_extraction_gold.json"
# After expanding the medical dictionary (symptoms / examinations / treatments),
# the held-out recall reaches 1.0; the threshold is raised to 0.95 to lock in
# the improvement while leaving headroom for future out-of-vocabulary entries.
RECALL_THRESHOLD = 0.95
PRECISION_THRESHOLD = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark task-2 extraction quality.")
    parser.add_argument("--gold", default=str(DEFAULT_GOLD), help="Gold corpus JSON path.")
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

    report = evaluate_extraction_quality(gold_records)
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
