"""Offline simulation of Nexent calling the three DKM task tools in sequence.

This demo mirrors how Nexent would orchestrate ``task1_data_processing``,
``task2_medical_kg``, and ``task3_graph_analysis`` without requiring a live
Nexent runtime. It writes a structured evidence JSON for judges and defense
packages.

Usage:
    python demos/dkm_nexent_toolchain_demo.py
    python demos/dkm_nexent_toolchain_demo.py --output-dir outputs/nexent_toolchain
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

from src.agents.analysis_agent.nexent_adapter import GraphAnalysisAgentTool
from src.agents.data_processing_agent.nexent_adapter import DataProcessingAgentTool
from src.agents.dkm_nexent_suite import build_dkm_nexent_suite_spec
from src.agents.kg_agent.nexent_adapter import MedicalKGAgentTool
from src.agents.planner_comparison import compare_dkm_orchestrator_planners


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate Nexent DKM three-tool toolchain.")
    parser.add_argument(
        "--text-input",
        default=str(ROOT / "data" / "samples" / "task1_medical_notes.txt"),
        help="Task-1 text input for the toolchain.",
    )
    parser.add_argument(
        "--question",
        default="哪些疾病关联最多症状？",
        help="Task-2/3 analysis question.",
    )
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "nexent_toolchain"))
    parser.add_argument(
        "--datamate-url",
        default="none",
        help="DataMate URL for task-1 tool. Use 'none' for offline demo.",
    )
    parser.add_argument(
        "--request",
        default="请清洗医疗文本，构建知识图谱并生成分析洞察",
        help="Natural-language DKM request for planner comparison.",
    )
    return parser.parse_args()


def _step_record(
    tool_name: str,
    payload: dict[str, Any],
    *,
    input_summary: dict[str, Any],
) -> dict[str, Any]:
    artifacts = payload.get("artifacts", {})
    return {
        "tool": tool_name,
        "status": payload.get("status"),
        "message": payload.get("message"),
        "input": input_summary,
        "artifacts": {
            key: value.get("output_path") or value.get("dashboard_path") or value
            for key, value in artifacts.items()
            if isinstance(value, dict)
        },
    }


def run_nexent_toolchain(
    *,
    text_input: str | Path,
    question: str,
    output_dir: str | Path,
    datamate_url: str | None = None,
    request: str,
) -> dict[str, Any]:
    """Execute task1 -> task2 -> task3 Nexent tools and return evidence."""

    root = Path(output_dir)
    task1_dir = root / "task1"
    task2_dir = root / "task2"
    task3_dir = root / "task3"

    planner = compare_dkm_orchestrator_planners(request, question=question)
    suite_spec = build_dkm_nexent_suite_spec(
        datamate_base_url=datamate_url,
        output_root=str(root),
    )

    task1_tool = DataProcessingAgentTool(
        datamate_base_url=datamate_url,
        output_dir=str(task1_dir),
    )
    task1_payload = json.loads(
        task1_tool.forward(
            task_request="清洗医疗文本并导出",
            input_path=str(text_input),
            datamate_mode="dry_run",
        )
    )
    if task1_payload.get("status") not in {"completed", "completed_with_warnings"}:
        return {
            "status": "failed",
            "failed_tool": "task1_data_processing",
            "planner": planner,
            "nexent_suite": suite_spec,
            "steps": [_step_record("task1_data_processing", task1_payload, input_summary={"path": str(text_input)})],
        }

    cleaned_path = (
        task1_payload.get("artifacts", {}).get("processing", {}).get("output_path")
        or task1_payload.get("artifacts", {}).get("cleaning", {}).get("output_path")
        or str(text_input)
    )

    task2_tool = MedicalKGAgentTool(output_dir=str(task2_dir))
    task2_payload = json.loads(
        task2_tool.forward(
            input_path=cleaned_path,
            question=question,
        )
    )
    if task2_payload.get("status") != "completed":
        return {
            "status": "failed",
            "failed_tool": "task2_medical_kg",
            "planner": planner,
            "nexent_suite": suite_spec,
            "steps": [
                _step_record("task1_data_processing", task1_payload, input_summary={"path": str(text_input)}),
                _step_record("task2_medical_kg", task2_payload, input_summary={"path": cleaned_path}),
            ],
        }

    graph_file = task2_payload["artifacts"]["graph"]["output_path"]
    task3_tool = GraphAnalysisAgentTool(output_dir=str(task3_dir))
    task3_payload = json.loads(
        task3_tool.forward(
            graph_file=graph_file,
            question=question,
            task_request=request,
        )
    )

    steps = [
        _step_record("task1_data_processing", task1_payload, input_summary={"path": str(text_input)}),
        _step_record("task2_medical_kg", task2_payload, input_summary={"path": cleaned_path, "question": question}),
        _step_record(
            "task3_graph_analysis",
            task3_payload,
            input_summary={"graph_file": graph_file, "question": question},
        ),
    ]
    return {
        "status": "completed" if task3_payload.get("status") == "completed" else "failed",
        "simulation": "nexent_toolchain_offline",
        "tool_count": len(suite_spec.get("tools", [])),
        "planner": planner,
        "nexent_suite": {
            "name": suite_spec.get("name"),
            "tool_names": [tool.get("name") for tool in suite_spec.get("tools", [])],
            "instructions": suite_spec.get("instructions"),
        },
        "steps": steps,
        "output_root": str(root),
    }


def main() -> int:
    args = parse_args()
    datamate_url = None if args.datamate_url.lower() == "none" else args.datamate_url
    result = run_nexent_toolchain(
        text_input=args.text_input,
        question=args.question,
        output_dir=args.output_dir,
        datamate_url=datamate_url,
        request=args.request,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "nexent_toolchain_evidence.json"
    evidence_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
