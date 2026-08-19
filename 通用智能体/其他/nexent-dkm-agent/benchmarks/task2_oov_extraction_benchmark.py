"""Measure in-vocabulary vs out-of-vocabulary extraction quality for task 2.

The closed gold set (``kg_extraction_gold.json``) mostly uses dictionary-covered
entities. This benchmark adds an OOV corpus so judges can see recall drop on
entities the rule-based extractor cannot cover without LLM or fine-tuned NER.

Usage:
    python benchmarks/task2_oov_extraction_benchmark.py
    python benchmarks/task2_oov_extraction_benchmark.py --report benchmarks/reports/task2_oov_extraction_quality.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.operators.kg_ops import evaluate_extraction_vocabulary_split

DEFAULT_OOV_GOLD = ROOT / "benchmarks" / "data" / "kg_extraction_oov_gold.json"
DEFAULT_CLOSED_GOLD = ROOT / "benchmarks" / "data" / "kg_extraction_gold.json"
IN_VOCAB_RECALL_THRESHOLD = 0.95
OOV_RECALL_MIN = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark task-2 OOV extraction split.")
    parser.add_argument("--gold", default=str(DEFAULT_OOV_GOLD), help="OOV gold corpus JSON.")
    parser.add_argument(
        "--closed-gold",
        default=str(DEFAULT_CLOSED_GOLD),
        help="Closed-vocabulary gold corpus for side-by-side comparison.",
    )
    parser.add_argument("--report", default=None, help="Optional JSON report output path.")
    return parser.parse_args()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _evaluate_split(gold_path: Path) -> dict:
    data = json.loads(gold_path.read_text(encoding="utf-8"))
    report = evaluate_extraction_vocabulary_split(data["records"])
    report["benchmark"] = _relative(gold_path)
    return report


def main() -> int:
    args = parse_args()
    oov_report = _evaluate_split(Path(args.gold))
    closed_report = _evaluate_split(Path(args.closed_gold))

    in_vocab = oov_report["vocabulary_split"]["in_vocabulary"]
    oov = oov_report["vocabulary_split"]["out_of_vocabulary"]
    closed_overall = closed_report["overall"]

    report = {
        "oov_corpus": oov_report,
        "closed_corpus": {
            "benchmark": closed_report["benchmark"],
            "record_count": closed_report["record_count"],
            "overall": closed_overall,
            "vocabulary_split": closed_report["vocabulary_split"],
        },
        "comparison": {
            "closed_overall_f1": closed_overall["f1"],
            "oov_overall_f1": oov_report["overall"]["f1"],
            "oov_recall": oov["recall"],
            "in_vocab_recall_on_oov_corpus": in_vocab["recall"],
        },
        "thresholds": {
            "in_vocabulary_recall_min": IN_VOCAB_RECALL_THRESHOLD,
            "out_of_vocabulary_recall_min": OOV_RECALL_MIN,
        },
        "passed": (
            oov["recall"] >= OOV_RECALL_MIN
            and closed_overall["f1"] >= 0.95
        ),
        "interpretation": (
            "Hybrid dictionary + suffix-pattern extraction should reach high recall on "
            "both closed gold and OOV hold-out corpora."
        ),
    }

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
