"""Emit task-1 local + DataMate hybrid execution plan evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.data_processing_agent.planner import plan_data_task
from src.operators.data_ops.csv_profile import profile_csv
from src.operators.data_ops.datamate_client import summarize_hybrid_execution_plan

DEFAULT_CSV = ROOT / "data" / "samples" / "task1_patients.csv"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "reports" / "task1_datamate_hybrid_evidence.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect task-1 DataMate hybrid plan evidence.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--request", default="清洗CSV、去重并填补缺失值")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = profile_csv(args.csv)
    plan = plan_data_task(args.request, data_profile=profile)
    hybrid = summarize_hybrid_execution_plan(plan.operators)
    report = {
        "status": "completed",
        "input": {"path": args.csv, "file_name": profile.get("file_name")},
        "task_request": args.request,
        "planned_operators": plan.operators,
        "hybrid_execution": hybrid,
        "interpretation": (
            "fill_missing_values runs locally; dedup and text normalization map "
            "to DataMate cleaning operators when the catalog is reachable."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
