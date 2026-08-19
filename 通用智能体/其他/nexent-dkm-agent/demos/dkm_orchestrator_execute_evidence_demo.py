"""Collect DKM orchestrator full-execution evidence for competition defense.

Runs ``DKMOrchestrator.run()`` (plan + execute) on sample medical text and
writes a structured JSON report with stage statuses, artifact paths, and graph
statistics. Complements ``dkm_nexent_toolchain_demo`` (Nexent tool simulation)
and ``end_to_end_demo`` (pipeline layer).

Usage:
    python demos/dkm_orchestrator_execute_evidence_demo.py
    python demos/dkm_orchestrator_execute_evidence_demo.py --output outputs/judge/dkm_orchestrator_evidence.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.dkm_orchestrator import DKMOrchestrator, plan_dkm_workflow

DEFAULT_INPUT = ROOT / "data" / "samples" / "task1_medical_notes.txt"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "reports" / "dkm_orchestrator_execute_evidence.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect DKM orchestrator execution evidence.")
    parser.add_argument(
        "--request",
        default="请清洗医疗文本，构建知识图谱并生成分析洞察",
        help="Natural-language DKM workflow request.",
    )
    parser.add_argument("--question", default="哪些疾病关联最多症状？")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Task-1 text input path.")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "dkm_orchestrator"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Evidence JSON path.")
    parser.add_argument("--datamate-url", default="none", help="DataMate URL or 'none'.")
    parser.add_argument(
        "--datamate-mode",
        choices=["dry_run", "submit"],
        default="dry_run",
    )
    return parser.parse_args()


def _graph_summary(graph_path: Path) -> dict[str, Any]:
    if not graph_path.is_file():
        return {"status": "missing", "path": str(graph_path)}
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    stats = graph.get("statistics", {})
    return {
        "status": "available",
        "path": str(graph_path),
        "node_count": stats.get("node_count", len(graph.get("nodes", []))),
        "edge_count": stats.get("edge_count", len(graph.get("edges", []))),
        "triple_count": stats.get("triple_count"),
    }


def collect_orchestrator_execution_evidence(
    *,
    request: str,
    question: str,
    text_input: str | Path,
    output_dir: str | Path,
    datamate_url: str | None = "none",
    datamate_mode: str = "dry_run",
) -> dict[str, Any]:
    """Run the orchestrator end-to-end and return a judge-facing evidence dict."""

    root = Path(output_dir)
    datamate_base = None if (datamate_url or "none").lower() == "none" else datamate_url
    plan = plan_dkm_workflow(request, question=question)
    orchestrator = DKMOrchestrator(
        datamate_base_url=datamate_base,
        datamate_mode=datamate_mode,
    )
    result = orchestrator.run(
        request=request,
        output_root=root,
        question=question,
        text_input=text_input,
    )

    graph_path = root / "task2" / "medical_kg.json"
    task3_report = root / "task3" / "task3_analysis_report.json"
    evidence: dict[str, Any] = {
        "status": result.get("status"),
        "execution_mode": "dkm_orchestrator_run",
        "request": request,
        "question": question,
        "input": {"path": str(text_input)},
        "planner_preview": plan,
        "stages": result.get("stages", []),
        "output_root": str(root),
        "artifacts": {
            "task1_cleaned": str(root / "task1"),
            "task2_graph": _graph_summary(graph_path),
            "task3_report": str(task3_report) if task3_report.is_file() else None,
        },
        "stage_order": [stage["task"] for stage in result.get("stages", [])],
        "interpretation": (
            "DKMOrchestrator plans stages from NL request keywords, then executes "
            "task1 -> task2 -> task3 agents in order with artifact handoff."
        ),
    }
    if result.get("status") == "failed":
        evidence["failed_stage"] = result.get("failed_stage")
        evidence["message"] = result.get("message")
    return evidence


def main() -> int:
    args = parse_args()
    evidence = collect_orchestrator_execution_evidence(
        request=args.request,
        question=args.question,
        text_input=args.input,
        output_dir=args.output_dir,
        datamate_url=args.datamate_url,
        datamate_mode=args.datamate_mode,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0 if evidence.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
