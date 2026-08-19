"""Generate defense-ready SVG figures without running the full evidence collector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.figure_export import export_all_defense_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate defense SVG figures.")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "defense_figures"))
    parser.add_argument(
        "--task1-report",
        default=str(ROOT / "benchmarks" / "reports" / "task1_data_quality.json"),
    )
    parser.add_argument(
        "--kg-graph",
        default=str(ROOT / "outputs" / "task2" / "medical_kg.json"),
    )
    parser.add_argument(
        "--task3-report",
        default=str(ROOT / "outputs" / "task3" / "task3_analysis_report.json"),
    )
    parser.add_argument(
        "--oov-report",
        default=str(ROOT / "benchmarks" / "reports" / "task2_oov_extraction_quality.json"),
    )
    parser.add_argument(
        "--nl2sql-report",
        default=str(ROOT / "benchmarks" / "reports" / "task3_nl2sql_report.json"),
    )
    parser.add_argument(
        "--planner-llm-report",
        default=str(ROOT / "benchmarks" / "reports" / "planner_llm_evidence.json"),
    )
    parser.add_argument(
        "--pipeline-latency-report",
        default=str(ROOT / "benchmarks" / "reports" / "task2_pipeline_latency.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = export_all_defense_figures(
        output_dir=args.output_dir,
        task1_quality_report=args.task1_report,
        kg_graph_file=args.kg_graph,
        task3_report_file=args.task3_report,
        oov_extraction_report=args.oov_report,
        nl2sql_report=args.nl2sql_report,
        planner_llm_report=args.planner_llm_report,
        pipeline_latency_report=args.pipeline_latency_report,
    )
    print(json.dumps({"status": "completed", "figures": manifest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
