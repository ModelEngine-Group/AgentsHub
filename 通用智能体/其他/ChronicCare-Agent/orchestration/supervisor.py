from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List

from orchestration.execution_trace import write_trace
from orchestration.planner import build_plan
from orchestration.tool_router import route_tool
from runtime_common.common import relative_to_project, resolve_path
from tool_server.utils import load_server_config, safety_note

TOOL_ARTIFACT_MAP = {
    "artifacts.status": [
        "data/sqlite/chroniccare.db",
        "data/graph/graph.json",
        "outputs/charts/chart_index.html",
        "outputs/reports/analysis_report.html",
    ],
    "analysis.query": [
        "outputs/reports/indicator_results.json",
    ],
    "kg.query": [
        "data/graph/graph.json",
        "data/graph/graph_summary.json",
    ],
    "kg.subgraph_render": [
        "outputs/subgraphs",
    ],
    "analysis.open_query": [
        "outputs/subgraphs",
        "outputs/graph_driven_analysis",
    ],
    "reports.summary": [
        "outputs/charts/chart_index.html",
        "outputs/reports/analysis_report.html",
        "data/graph/graph.html",
    ],
    "datamate.pipelines": [
        "outputs/release/datamate_full_pipeline_report.json",
        "outputs/release/datamate_full_pipeline_check_report.json",
    ],
    "datamate.pipeline_run": [
        "outputs/release/datamate_full_pipeline_report.json",
        "outputs/release/datamate_full_pipeline_check_report.json",
        "outputs/release/datamate_sync_report.json",
    ],
    "datamate.pipeline_latest": [
        "outputs/release/datamate_full_pipeline_report.json",
        "outputs/release/datamate_full_pipeline_check_report.json",
        "outputs/release/datamate_sync_report.json",
    ],
    "datamate.pipeline_status": [
        "outputs/release/datamate_full_pipeline_report.json",
        "outputs/release/datamate_full_pipeline_check_report.json",
    ],
    "datamate.pipeline_report": [
        "outputs/release/datamate_full_pipeline_report.json",
        "outputs/release/datamate_full_pipeline_check_report.json",
        "outputs/release/datamate_sync_report.json",
    ],
}


def supervisor_plan(user_goal: str) -> Dict[str, Any]:
    return build_plan(user_goal)


def _infer_tool_input(tool_name: str, user_goal: str, plan_item: Dict[str, Any]) -> Dict[str, Any]:
    input_hint = dict(plan_item.get("input_hint", {}))
    if input_hint:
        return input_hint
    if tool_name == "analysis.query":
        return {"question": user_goal}
    if tool_name == "datamate.pipeline_run":
        return {"task_id": "supervisor_datamate_run_001", "force": True, "safe_run": True}
    if tool_name in {"datamate.pipeline_status", "datamate.pipeline_report"}:
        return {"run_id": "latest"}
    if tool_name == "analysis.open_query":
        return {"question": user_goal}
    if tool_name == "analysis.graph_driven":
        return {"question": user_goal}
    if tool_name == "kg.subgraph_render":
        return {"query": user_goal, "max_nodes": 96}
    if tool_name == "kg.query":
        return {"query_type": "disease_profile", "entity_id": "Disease::hypertension"}
    return {}


def _summarize_tool_output(output: Dict[str, Any]) -> str:
    if output.get("status") != "success":
        return "；".join(output.get("errors", ["tool failed"]))
    if "summary_text" in output:
        return str(output["summary_text"])
    if "answer" in output:
        return str(output["answer"])
    if "insight" in output:
        return str(output["insight"])
    if "artifacts" in output:
        existing = [name for name, item in output["artifacts"].items() if item.get("exists")]
        return f"已确认可用产物：{', '.join(existing)}"
    if output.get("pipeline_name") == "chroniccare_datamate_full_pipeline":
        return str(output.get("summary") or "已整理 DataMate pipeline 状态与报告入口")
    if "analysis_report_html" in output:
        return "已整理可访问的分析报告入口"
    if "html_url" in output:
        return "已生成可访问的问题驱动子图入口"
    if "graph_url" in output:
        return "已生成可访问的图谱子图入口"
    if "questions" in output:
        return f"已读取分析问题集，共 {len(output['questions'])} 个问题。"
    return "工具执行成功"


def _collect_evidence_paths(tool_results: List[Dict[str, Any]]) -> List[str]:
    paths: List[str] = []
    for result in tool_results:
        output = result["raw_output"]
        for key in [
            "html_url",
            "service_html_url",
            "graph_html",
            "graph_url",
            "analysis_report_html",
            "chart_index",
            "analysis_report_md",
            "report_url",
            "chart_url",
            "pipeline_browser_url",
            "report_browser_url",
            "check_report_browser_url",
            "sync_report_browser_url",
        ]:
            value = output.get(key)
            if isinstance(value, str):
                paths.append(value)
        if "artifacts" in output:
            for artifact in output["artifacts"].values():
                path = artifact.get("path")
                if isinstance(path, str) and (path.startswith("http://") or path.startswith("https://") or path.startswith("/artifacts/")):
                    paths.append(path)
    unique: List[str] = []
    seen = set()
    for path in paths:
        if not isinstance(path, str):
            continue
        if not (path.startswith("http://") or path.startswith("https://") or path.startswith("/artifacts/")):
            continue
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _collect_chart_markdown(tool_results: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    seen = set()
    for result in tool_results:
        charts = result.get("raw_output", {}).get("charts") or []
        for chart in charts[:2]:
            name = str(chart.get("name") or "图表")
            url = str(chart.get("url") or chart.get("png_alias_url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            lines.append(f"![{name}]({url})")
    return lines


def _expand_pipeline_run_steps(base_step: Dict[str, Any], output: Dict[str, Any]) -> List[Dict[str, Any]]:
    pipeline_steps = output.get("steps") or output.get("pipeline_steps") or []
    expanded: List[Dict[str, Any]] = []
    for index, item in enumerate(pipeline_steps, start=1):
        operator_name = str(item.get("operator") or f"operator_{index}")
        summary = item.get("summary") or {}
        if operator_name == "chronic_nl2sql_analyze" and isinstance(summary, dict) and summary:
            question_count = summary.get("question_count")
            success_count = summary.get("success_count")
            if question_count is not None and success_count == question_count:
                summary_text = f"question_count={question_count}"
            else:
                summary_text = "、".join(f"{key}={value}" for key, value in summary.items())
        elif operator_name == "chronic_triple_validate" and isinstance(summary, dict) and summary:
            triples_clean = summary.get("triples_clean")
            triples_rejected = summary.get("triples_rejected")
            if triples_clean is not None and triples_rejected is not None:
                summary_text = f"triples_clean={triples_clean}、剔除异常三元组={triples_rejected}"
            else:
                summary_text = "、".join(f"{key}={value}" for key, value in summary.items())
        else:
            summary_text = "、".join(f"{key}={value}" for key, value in summary.items()) if isinstance(summary, dict) and summary else "已完成"
        expanded.append(
            {
                "step": f"{base_step['step']}.{index}",
                "agent": "DataProcessingAgent",
                "tool": operator_name,
                "description": "DataMate 算子执行明细",
                "input": {},
                "expected_output": "单算子执行摘要",
                "output_summary": f"{operator_name}：{summary_text}",
                "status": item.get("status", "unknown"),
                "time_cost_sec": None,
                "raw_output": {
                    "status": item.get("status", "unknown"),
                    "summary": summary,
                    "artifact_keys": item.get("artifact_keys", []),
                },
            }
        )
    return expanded


def _find_first_url(tool_results: List[Dict[str, Any]], keys: List[str]) -> str | None:
    for result in tool_results:
        output = result["raw_output"]
        for key in keys:
            value = output.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _infer_graph_label(user_goal: str) -> str:
    if "运行慢病原始数据" in user_goal or "数据处理流程" in user_goal or "清洗摘要" in user_goal:
        return "查看当前知识图谱总览"
    if "高血压合并糖尿病" in user_goal:
        return "查看高血压合并糖尿病群体图谱子图"
    if "糖尿病" in user_goal and "高血压" not in user_goal:
        return "查看糖尿病患者群体图谱子图"
    if "高血压" in user_goal and "糖尿病" not in user_goal:
        return "查看高血压患者群体图谱子图"
    if "高风险" in user_goal:
        return "查看高风险患者群体图谱子图"
    return "查看本次分析对应图谱子图"


def _markdown_link(label: str, url: str) -> str:
    return f"[{label}]({url})"


def _is_pure_subgraph_goal(user_goal: str) -> bool:
    return any(token in user_goal for token in ["图谱子图", "知识图谱子图", "知识图谱", "关系图"]) and any(
        token in user_goal for token in ["高血压", "糖尿病", "高风险", "中风险", "低风险", "群体", "患者"]
    )


def _compose_final_answer(user_goal: str, tool_results: List[Dict[str, Any]], note: str) -> str:
    if _is_pure_subgraph_goal(user_goal):
        subgraph_result = next((item for item in tool_results if item.get("tool") == "kg.subgraph_render"), None)
        if subgraph_result is not None:
            output = subgraph_result.get("raw_output", {})
            graph_url = str(
                output.get("html_url") or output.get("graph_url") or output.get("service_html_url") or ""
            ).strip()
            lines = [
                "我已根据你的目标完成了以下步骤：",
                f"1. {subgraph_result['agent']} 使用 {subgraph_result['tool']}：{subgraph_result['output_summary']}",
                "关键结论：",
                f"- {subgraph_result['output_summary']}",
            ]
            if graph_url:
                lines.append("图谱子图可视化入口：")
                lines.append(f"- {_markdown_link(_infer_graph_label(user_goal), graph_url)}")
            lines.append(f"医疗安全说明：{note}")
            return "\n".join(lines).replace("{user_goal}", user_goal)

    evidence_paths = _collect_evidence_paths(tool_results)
    graph_url = _find_first_url(tool_results, ["graph_url", "html_url", "graph_html"])
    report_url = _find_first_url(tool_results, ["report_url", "report_browser_url", "analysis_report_html"])
    chart_url = _find_first_url(tool_results, ["chart_url", "chart_index", "chart_index_url"])
    if report_url and graph_url and report_url == graph_url:
        report_url = None
    if report_url and "/artifacts/subgraphs/" in report_url:
        report_url = None
    lines = [
        "我已根据你的目标完成了以下步骤：",
    ]
    for index, result in enumerate(tool_results, start=1):
        lines.append(
            f"{index}. {result['agent']} 使用 {result['tool']}：{result['output_summary']}"
        )
    lines.append("关键结论：")
    for result in tool_results:
        if result["status"] == "success":
            lines.append(f"- {result['output_summary']}")
    datamate_run = next((item for item in tool_results if item.get("tool") == "datamate.pipeline_run"), None)
    if datamate_run and datamate_run.get("time_cost_sec"):
        lines.append(f"- 本次 DataMate 全流程执行耗时约 {float(datamate_run['time_cost_sec']):.1f} 秒。")
    if evidence_paths:
        lines.append("证据路径：")
        for path in evidence_paths[:6]:
            lines.append(f"- {path}")
    if graph_url:
        lines.append("图谱子图可视化入口：")
        lines.append(f"- {_markdown_link(_infer_graph_label(user_goal), graph_url)}")
    chart_images = _collect_chart_markdown(tool_results)
    if chart_images:
        lines.append("图表预览：")
        lines.extend(chart_images)
    if chart_url:
        lines.append("图表入口：")
        lines.append(f"- {_markdown_link('查看图表总览', chart_url)}")
    if report_url:
        lines.append("报告入口：")
        lines.append(f"- {_markdown_link('查看分析报告页面', report_url)}")
    lines.append(f"医疗安全说明：{note}")
    return "\n".join(lines).replace("{user_goal}", user_goal)


def run_supervisor(user_goal: str) -> Dict[str, Any]:
    cfg = load_server_config()
    note = safety_note(cfg)
    plan_payload = build_plan(user_goal)
    plan = plan_payload["plan"]
    steps: List[Dict[str, Any]] = []
    artifacts_used: List[str] = []
    agents_used: List[str] = []
    for item in plan:
        start = time.perf_counter()
        tool_name = str(item["tool"])
        tool_input = _infer_tool_input(tool_name, user_goal, item)
        output = route_tool(tool_name, **tool_input)
        agents_used.append(str(item["agent"]))
        artifacts_used.extend(TOOL_ARTIFACT_MAP.get(tool_name, []))
        steps.append(
            {
                "step": item["step"],
                "agent": item["agent"],
                "tool": tool_name,
                "description": item.get("description", ""),
                "input": tool_input,
                "expected_output": item.get("expected_output", ""),
                "output_summary": _summarize_tool_output(output),
                "status": output.get("status", "failed"),
                "time_cost_sec": round(time.perf_counter() - start, 4),
                "raw_output": output,
            }
        )
        if tool_name == "datamate.pipeline_run":
            steps.extend(_expand_pipeline_run_steps(steps[-1], output))
    final_answer = _compose_final_answer(user_goal, steps, note)
    run_id = f"agent_run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    trace_dir = resolve_path(cfg["paths"]["agent_runs_dir"])
    trace_path = trace_dir / f"{run_id}.json"
    unique_agents = list(dict.fromkeys(agents_used))
    unique_artifacts = list(dict.fromkeys(artifacts_used))
    write_trace(trace_path, run_id, user_goal, plan, steps, final_answer, note, unique_agents, unique_artifacts)
    public_steps = []
    for step in steps:
        public_step = dict(step)
        public_step.pop("raw_output", None)
        public_steps.append(public_step)
    return {
        "status": "success",
        "run_id": run_id,
        "user_goal": user_goal,
        "plan": plan,
        "tool_results": public_steps,
        "tool_call_count": len(public_steps),
        "agents_used": unique_agents,
        "artifacts_used": unique_artifacts,
        "final_answer": final_answer,
        "trace_path": relative_to_project(trace_path),
        "safety_note": note,
    }
