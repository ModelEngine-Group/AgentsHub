"""Generate a reproducible task-1 evaluation report."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.task1_evaluation import DEFAULT_REPORT_PATH, run_task1_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate task 1 and write JSON.")
    parser.add_argument("--task", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--datamate-url",
        default="http://localhost:18000",
        help="DataMate Python backend base URL. Use 'none' to disable.",
    )
    parser.add_argument(
        "--datamate-mode",
        choices=["dry_run", "submit"],
        default="dry_run",
    )
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
        help="JSON report path. Defaults to outputs/task1/task1_quality_report.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datamate_url = None if args.datamate_url.lower() == "none" else args.datamate_url
    payload = run_task1_evaluation(
        task_request=args.task,
        input_path=args.input,
        output_dir=args.output_dir,
        datamate_base_url=datamate_url,
        datamate_mode=args.datamate_mode,
        report_path=args.report_path,
    )
    quality = payload.get("quality_report", {})
    metrics = quality.get("metrics", {})
    print(
        "task1_evaluation: "
        f"{payload.get('status')} / "
        f"quality={quality.get('status')} / "
        f"operators={metrics.get('planned_operator_count')} / "
        f"datamate_ops={metrics.get('datamate_operator_count')}"
    )
    print(f"report: {args.report_path}")
    return 0 if payload.get("status") in {"completed", "completed_with_warnings"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
