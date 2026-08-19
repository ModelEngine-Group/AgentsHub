"""Generate a reproducible task-3 evaluation report."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.task3_evaluation import run_task3_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate task-3 graph analysis agent.")
    parser.add_argument("--graph-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--question", default="哪些疾病关联最多症状？")
    parser.add_argument("--task-request", default=None)
    parser.add_argument("--report", default=None, help="Optional compact JSON report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_task3_evaluation(
        graph_file=args.graph_file,
        output_dir=args.output_dir,
        question=args.question,
        task_request=args.task_request,
        report_path=args.report,
    )
    print(
        "task3_evaluation: "
        f"{payload['status']} / quality={payload['quality_report']['status']} / "
        f"charts={len(payload['visualizations']['chart_names'])} / "
        f"sql_rows={payload['nl2sql']['row_count']}"
    )
    if args.report:
        print(f"report: {args.report}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
