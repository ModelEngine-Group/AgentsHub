"""Generate a reproducible task-2 evaluation report."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.task2_evaluation import DEFAULT_REPORT_PATH, run_task2_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate task 2 and write JSON.")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--question",
        default="高血压有哪些症状和用药？",
        help="Question to answer from the generated graph.",
    )
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
        help="JSON report path. Defaults to outputs/task2/task2_quality_report.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_task2_evaluation(
        input_path=args.input,
        output_dir=args.output_dir,
        question=args.question,
        report_path=args.report_path,
    )
    quality = payload.get("quality_report", {})
    metrics = quality.get("metrics", {})
    print(
        "task2_evaluation: "
        f"{payload.get('status')} / "
        f"quality={quality.get('status')} / "
        f"triples={metrics.get('triple_count')} / "
        f"entities={metrics.get('entity_total')}"
    )
    print(f"report: {args.report_path}")
    return 0 if payload.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
