from __future__ import annotations

from pathlib import Path
from typing import Any

from medgraph_agent.core.pipeline import PipelineRunner
from medgraph_agent.core.quality import audit_graph
from medgraph_agent.core.storage import load_graph_json, write_json
from medgraph_agent.core.models import to_dict


def run_datamate_operator(payload: dict[str, Any]) -> dict[str, Any]:
    """DataMate-style callable entrypoint for custom operator execution."""

    source = payload["source"]
    output_dir = payload.get("output_dir", "outputs/datamate")
    task = payload.get("task", "DataMate custom operator: 医疗文本清洗、知识抽取和图谱生成")
    run = PipelineRunner(output_dir).run(task, source)
    graph_path = Path(output_dir) / "graph.json"
    graph = load_graph_json(graph_path) if graph_path.exists() else None
    quality = audit_graph(graph) if graph else {}
    if quality:
        write_json(Path(output_dir) / "quality_report.json", quality)
    return {
        "status": run.status,
        "run": to_dict(run),
        "graph_stats": graph.stats() if graph else {},
        "quality": quality,
        "artifacts": {
            "run": str(Path(output_dir) / "run.json"),
            "graph": str(graph_path),
            "sqlite": str(Path(output_dir) / "medgraph.db"),
            "quality": str(Path(output_dir) / "quality_report.json"),
        },
    }
