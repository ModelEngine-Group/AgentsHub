"""Refresh all defense-package image assets from current evidence JSON and demos."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "competition_submission" / "defense-package-final" / "evidence"
FIGURES = EVIDENCE / "figures"
BENCHMARKS = EVIDENCE / "benchmarks"


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def _regenerate_npu_figures() -> list[dict[str, str]]:
    from src.common.figure_export import export_npu_mode_speedup_figure, export_npu_utilization_figure

    return [
        export_npu_mode_speedup_figure(
            BENCHMARKS / "task2_relation_tensor_ascend_910b2c_xlarge.json",
            FIGURES / "npu_task2_mode_speedup.svg",
            name="npu_task2_mode_speedup",
            title="Task 2 relation tensor: NPU mode speedup vs CPU",
        ),
        export_npu_mode_speedup_figure(
            BENCHMARKS / "task3_graph_tensor_ascend_910b2c_large.json",
            FIGURES / "npu_task3_mode_speedup.svg",
            name="npu_task3_mode_speedup",
            title="Task 3 graph tensor: NPU mode speedup vs CPU",
        ),
        export_npu_utilization_figure(
            {
                "task2_xlarge": BENCHMARKS / "task2_relation_tensor_ascend_910b2c_xlarge.json",
                "task3_50k": BENCHMARKS / "task3_graph_tensor_ascend_910b2c_large.json",
            },
            FIGURES / "npu_utilization.svg",
        ),
    ]


def refresh_all(task3_source: Path, captured_on: str, sync_html: bool) -> dict[str, object]:
    task3_source.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "demos/task3_demo.py",
            "--graph-file",
            str(EVIDENCE / "artifacts" / "medical_kg.json"),
            "--output-dir",
            str(task3_source),
        ]
    )
    _run(
        [
            sys.executable,
            "demos/generate_defense_figures.py",
            "--output-dir",
            str(FIGURES),
            "--kg-graph",
            str(EVIDENCE / "artifacts" / "medical_kg.json"),
            "--task1-report",
            str(ROOT / "benchmarks" / "reports" / "task1_data_quality.json"),
            "--oov-report",
            str(BENCHMARKS / "task2_oov_extraction_quality.json"),
            "--nl2sql-report",
            str(BENCHMARKS / "task3_nl2sql_report.json"),
            "--planner-llm-report",
            str(BENCHMARKS / "planner_llm_evidence.json"),
            "--pipeline-latency-report",
            str(BENCHMARKS / "task2_pipeline_latency.json"),
            "--task3-report",
            str(task3_source / "task3_analysis_report.json"),
        ]
    )
    npu = _regenerate_npu_figures()
    _run(
        [
            sys.executable,
            "scripts/render_neo4j_screenshots.py",
            "--captured-on",
            captured_on,
        ]
    )
    task3_cmd = [sys.executable, "scripts/render_task3_screenshots.py"]
    if sync_html:
        task3_cmd.extend(["--sync-from", str(task3_source)])
    _run(task3_cmd)
    if sync_html:
        _run(
            [
                sys.executable,
                "demos/export_defense_pdf.py",
                "--source",
                "competition_submission/defense-package-final",
                "--sync-from",
                "docs/competition_defense_document.md",
                "--output",
                "competition_submission/defense-package-final/competition_defense_document.html",
            ]
        )
    return {
        "status": "completed",
        "figures_dir": str(FIGURES),
        "task3_source": str(task3_source),
        "npu_figures": npu,
        "captured_on": captured_on,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh defense-package PNG/SVG assets.")
    parser.add_argument("--task3-source", type=Path, default=ROOT / "outputs" / "task3_evidence")
    parser.add_argument("--captured-on", default="2026-07-03")
    parser.add_argument("--no-sync-html", action="store_true")
    return parser.parse_args()


def main() -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    args = parse_args()
    summary = refresh_all(args.task3_source, args.captured_on, sync_html=not args.no_sync_html)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
