from __future__ import annotations

from typing import Any, Callable, Dict

from tool_server.analysis_tools import (
    analysis_query,
    graph_driven_analysis,
    open_analysis_query,
)
from tool_server.kg_tools import kg_query, kg_subgraph_render, kg_summary
from tool_server.open_sql_tools import open_sql_examples_tool
from tool_server.pipeline_tools import (
    artifacts_status,
    datamate_pipeline_latest,
    datamate_pipeline_report_by_run,
    datamate_pipeline_run_cli_hint,
    datamate_pipeline_status_by_run,
    datamate_pipelines,
    pipeline_reports,
    run_datamate_pipeline,
)
from tool_server.report_tools import charts_list, reports_summary

TOOL_MAP: Dict[str, Callable[..., Dict[str, Any]]] = {
    "artifacts.status": artifacts_status,
    "pipeline.reports": pipeline_reports,
    "datamate.pipelines": datamate_pipelines,
    "datamate.pipeline_run": run_datamate_pipeline,
    "datamate.pipeline_run_cli_hint": datamate_pipeline_run_cli_hint,
    "datamate.pipeline_latest": datamate_pipeline_latest,
    "datamate.pipeline_status": datamate_pipeline_status_by_run,
    "datamate.pipeline_report": datamate_pipeline_report_by_run,
    "kg.summary": kg_summary,
    "kg.query": kg_query,
    "kg.subgraph_render": kg_subgraph_render,
    "analysis.open_sql_examples": open_sql_examples_tool,
    "analysis.query": analysis_query,
    "analysis.open_query": open_analysis_query,
    "analysis.graph_driven": graph_driven_analysis,
    "reports.summary": reports_summary,
    "charts.list": charts_list,
}


def route_tool(tool_name: str, **kwargs: Any) -> Dict[str, Any]:
    if tool_name not in TOOL_MAP:
        return {"status": "failed", "errors": [f"Unsupported tool: {tool_name}"]}
    return TOOL_MAP[tool_name](**kwargs)
