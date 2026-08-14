"""Minimal HTTP API for the medical demo."""

from __future__ import annotations

import argparse
import html
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agents.analysis_agent import run_analysis_agent, run_configured_analysis, run_interactive_analysis, run_nl2sql_eval
from agents.datamate_processing_agent import run_datamate_processing_agent
from agents.data_processing_agent import run_data_processing_agent
from agents.integration_agent import run_platform_integration_agent
from agents.medical_kg_agent import ask_medical_question, build_medical_demo
from agents.orchestrator import run_orchestrator
from demo.task1_openapi import build_task1_openapi_spec
from demo.task2_openapi import build_task2_openapi_spec
from demo.task3_openapi import build_task3_openapi_spec
from operators.llm_client import is_configured as is_llm_configured


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(value: str | None, base: Path) -> Path:
    if not value:
        return base
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return base / candidate


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    raw = handler.rfile.read(length) if length > 0 else b""
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _json_response(status: int, payload: dict[str, Any]) -> tuple[int, str, bytes]:
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    return status, "application/json; charset=utf-8", body


def _text_response(status: int, payload: str, content_type: str = "text/plain; charset=utf-8") -> tuple[int, str, bytes]:
    return status, content_type, payload.encode("utf-8")


def _read_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _read_jsonl_file(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _kg_artifact_paths(output_root: Path) -> dict[str, str]:
    return {
        "entities": str(output_root / "graph" / "entities.jsonl"),
        "triples": str(output_root / "graph" / "triples.jsonl"),
        "graph_snapshot": str(output_root / "graph" / "graph_snapshot.json"),
        "kg_validation_report": str(output_root / "graph" / "kg_validation_report.json"),
        "kg_quality_eval": str(output_root / "graph" / "kg_quality_eval.json"),
        "kg_backend_probe": str(output_root / "graph" / "kg_backend_probe.json"),
        "kg_operator_profile": str(output_root / "graph" / "kg_operator_profile.json"),
        "kg_build_trace": str(output_root / "graph" / "kg_build_trace.jsonl"),
        "qa_trace": str(output_root / "demo" / "qa_trace.jsonl"),
        "hybrid_qa_trace": str(output_root / "demo" / "hybrid_qa_trace.jsonl"),
        "kg_quality_report": str(output_root / "demo" / "kg_quality_report.md"),
    }


def _task3_artifact_paths(output_root: Path) -> dict[str, str]:
    return {
        "analysis_report": str(output_root / "demo" / "analysis_report.md"),
        "analysis_summary": str(output_root / "demo" / "analysis_summary.json"),
        "analysis_planning": str(output_root / "demo" / "analysis_planning.json"),
        "analysis_trace": str(output_root / "demo" / "analysis_trace.jsonl"),
        "analysis_queries": str(output_root / "demo" / "analysis_queries.jsonl"),
        "nl2sql_eval_report": str(output_root / "demo" / "nl2sql_eval_report.json"),
        "task3_lineage": str(output_root / "demo" / "task3_lineage.json"),
        "graph_understanding": str(output_root / "demo" / "graph_understanding.json"),
        "visualization_manifest": str(output_root / "demo" / "visualization_manifest.json"),
        "task3_closed_loop_status": str(output_root / "demo" / "task3_closed_loop_status.json"),
        "graph_insights": str(output_root / "demo" / "graph_insights.json"),
        "charts_dir": str(output_root / "demo" / "charts"),
        "subgraphs": str(output_root / "demo" / "subgraphs" / "disease_subgraphs.jsonl"),
        "interactive_analysis_latest": str(output_root / "demo" / "interactive_analysis_latest.json"),
        "interactive_analysis_trace": str(output_root / "demo" / "interactive_analysis_trace.jsonl"),
    }


def _build_medical_kg_response(result: dict[str, Any], output_root: Path) -> dict[str, Any]:
    return {
        "status": "completed",
        "agent": "MedicalKGAgent",
        "pipeline": [
            "data_processing",
            "entity_extract",
            "relation_extract",
            "triple_normalize",
            "triple_validate_arbitrate",
            "entity_link",
            "graph_build",
            "kg_qa",
        ],
        "documents": len(result["documents"]),
        "entities": len(result["entities"]),
        "triples": len(result["triples"]),
        "qa_count": len(result.get("qa_traces", [])),
        "kg_validation_report": result.get("kg_validation_report", {}),
        "kg_quality_eval": result.get("kg_quality_eval", {}),
        "kg_backend_probe": result.get("kg_backend_probe", {}),
        "kg_operator_profile": result.get("kg_operator_profile", {}),
        "external_summary": result.get("external_summary", {}),
        "subgraph_count": len(result.get("subgraphs", [])),
        "analysis_summary": result.get("analysis_summary", {}),
        "llm_requested": result.get("llm_requested", False),
        "llm_configured": result.get("llm_configured", False),
        "llm_active": result.get("llm_active", False),
        "kg_backend": result.get("kg_backend", {}),
        "hybrid_qa": result.get("hybrid_qa", {}),
        "artifact_paths": _kg_artifact_paths(output_root),
        "trace": result.get("kg_trace", []),
        "output_root": str(output_root),
    }


def _run_medical_kg_build(payload: dict[str, Any], project_root: Path) -> tuple[int, str, bytes]:
    input_path = _resolve_path(payload.get("input_path"), project_root)
    output_root = _resolve_path(payload.get("output_root"), project_root)
    questions = payload.get("questions")
    if questions is not None and not isinstance(questions, list):
        return _json_response(HTTPStatus.BAD_REQUEST, {"error": "questions_must_be_list"})
    use_llm = payload.get("use_llm")
    if use_llm is not None:
        use_llm = bool(use_llm)
    kg_backend = payload.get("kg_backend")
    if kg_backend is not None and kg_backend not in {"rule", "tensor", "llm"}:
        return _json_response(HTTPStatus.BAD_REQUEST, {"error": "kg_backend_must_be_rule_tensor_or_llm"})
    external_documents = payload.get("external_documents")
    external_triples = payload.get("external_triples")
    external_entities = payload.get("external_entities")
    for field_name, value in {
        "external_documents": external_documents,
        "external_triples": external_triples,
        "external_entities": external_entities,
    }.items():
        if value is not None and not isinstance(value, str):
            return _json_response(HTTPStatus.BAD_REQUEST, {"error": f"{field_name}_must_be_string"})
    try:
        result = build_medical_demo(
            input_path=input_path,
            output_root=output_root,
            questions=questions,
            use_llm=use_llm,
            kg_backend=kg_backend,
            external_documents_path=_resolve_path(external_documents, project_root) if external_documents else None,
            external_triples_path=_resolve_path(external_triples, project_root) if external_triples else None,
            external_entities_path=_resolve_path(external_entities, project_root) if external_entities else None,
        )
    except FileNotFoundError as exc:
        return _json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
    return _json_response(HTTPStatus.OK, _build_medical_kg_response(result, output_root))


def _run_medical_kg_ask(payload: dict[str, Any], project_root: Path) -> tuple[int, str, bytes]:
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        return _json_response(HTTPStatus.BAD_REQUEST, {"error": "question_required"})
    output_root = _resolve_path(payload.get("output_root"), project_root)
    use_llm = payload.get("use_llm")
    if use_llm is not None:
        use_llm = bool(use_llm)
    use_hybrid = payload.get("use_hybrid", True)
    if use_hybrid is not None:
        use_hybrid = bool(use_hybrid)
    knowledge_base_id = payload.get("knowledge_base_id")
    if knowledge_base_id is not None and not isinstance(knowledge_base_id, str):
        return _json_response(HTTPStatus.BAD_REQUEST, {"error": "knowledge_base_id_must_be_string"})
    enable_live_datamate = payload.get("enable_live_datamate", True)
    if enable_live_datamate is not None:
        enable_live_datamate = bool(enable_live_datamate)
    datamate_timeout_seconds = payload.get("datamate_timeout_seconds", 1.5)
    try:
        datamate_timeout_seconds = float(datamate_timeout_seconds)
    except (TypeError, ValueError):
        return _json_response(HTTPStatus.BAD_REQUEST, {"error": "datamate_timeout_seconds_must_be_number"})
    try:
        trace = ask_medical_question(
            output_root,
            question.strip(),
            use_llm=use_llm,
            use_hybrid=use_hybrid,
            knowledge_base_id=knowledge_base_id,
            enable_live_datamate=enable_live_datamate,
            datamate_timeout_seconds=datamate_timeout_seconds,
        )
    except FileNotFoundError as exc:
        return _json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
    return _json_response(HTTPStatus.OK, trace)


def _render_bar_rows(counts: dict[str, int], color: str, empty_label: str) -> str:
    if not counts:
        return f'<div class="empty">{html.escape(empty_label)}</div>'
    max_value = max(counts.values()) or 1
    rows: list[str] = []
    for label, value in sorted(counts.items(), key=lambda item: item[0]):
        width = max(8, int((value / max_value) * 100))
        rows.append(
            "<div class=\"bar-row\">"
            f"<div class=\"bar-label\">{html.escape(str(label))}</div>"
            "<div class=\"bar-track\">"
            f"<div class=\"bar-fill\" style=\"background:{color};width:{width}%;\"></div>"
            "</div>"
            f"<div class=\"bar-value\">{value}</div>"
            "</div>"
        )
    return "".join(rows)


def _render_table_rows(rows: list[list[str]]) -> str:
    return "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )


def _render_dashboard_html(project_root: Path) -> str:
    graph = _read_json_file(project_root / "graph" / "graph_snapshot.json", {})
    analysis_summary = _read_json_file(project_root / "demo" / "analysis_summary.json", {})
    nl2sql_eval = _read_json_file(project_root / "demo" / "nl2sql_eval_report.json", {})
    analysis_queries = _read_jsonl_file(project_root / "demo" / "analysis_queries.jsonl")
    qa_traces = _read_jsonl_file(project_root / "demo" / "qa_trace.jsonl")
    integration_evidence = _read_json_file(project_root / "demo" / "integration" / "platform_evidence.json", {})

    statistics = graph.get("statistics", {}) if isinstance(graph, dict) else {}
    entity_counts = statistics.get("entity_type_counts", {}) if isinstance(statistics, dict) else {}
    relation_counts = statistics.get("relation_type_counts", {}) if isinstance(statistics, dict) else {}
    nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
    service_probes = integration_evidence.get("service_probes", []) if isinstance(integration_evidence, dict) else []
    datamate_api_calls = integration_evidence.get("datamate_api_calls", []) if isinstance(integration_evidence, dict) else []
    datamate_bootstrap_calls = integration_evidence.get("datamate_bootstrap_calls", []) if isinstance(integration_evidence, dict) else []
    nexent_api_calls = integration_evidence.get("nexent_api_calls", []) if isinstance(integration_evidence, dict) else []
    datamate_search_trace = integration_evidence.get("datamate_search_trace", []) if isinstance(integration_evidence, dict) else []
    integration_summary = integration_evidence.get("summary", {}) if isinstance(integration_evidence, dict) else {}
    combined_datamate_api_calls = datamate_bootstrap_calls + datamate_api_calls

    entity_chart = _render_bar_rows(entity_counts, "linear-gradient(90deg, #2457d6, #5f8cff)", "暂无实体统计")
    relation_chart = _render_bar_rows(relation_counts, "linear-gradient(90deg, #0f766e, #37a39a)", "暂无关系统计")

    graph_rows = _render_table_rows(
        [
            [str(node.get("name", "")), str(node.get("type", "")), ", ".join(node.get("source_doc_ids", [])[:3])]
            for node in nodes[:12]
        ]
    )
    qa_rows = _render_table_rows(
        [
            [
                str(item.get("question", "")),
                str(item.get("answer", "")),
                str(len(item.get("supporting_triples", []))),
            ]
            for item in qa_traces[-3:]
        ]
    )
    nl2sql_rows = _render_table_rows(
        [
            [
                str(item.get("question", "")),
                str(item.get("row_count", 0)),
                str(item.get("chart", "")),
            ]
            for item in analysis_queries
        ]
    )
    nl2sql_accuracy = nl2sql_eval.get("accuracy", analysis_summary.get("nl2sql_accuracy"))
    nl2sql_eval_summary = (
        f"{float(nl2sql_accuracy):.2%}"
        if isinstance(nl2sql_accuracy, int | float)
        else "暂无"
    )
    service_rows = _render_table_rows(
        [
            [
                str(item.get("name", "")),
                str(item.get("url", "")),
                str(item.get("status_code", "")),
                "是" if item.get("reachable") else "否",
                str(item.get("error", "") or item.get("reason", "")),
            ]
            for item in service_probes
        ]
    )
    datamate_rows = _render_table_rows(
        [
            [
                str(item.get("name", "")),
                str(item.get("method", "")),
                str(item.get("status_code", "")),
                "是" if item.get("ok") else "否",
            ]
            for item in combined_datamate_api_calls
        ]
    )
    nexent_rows = _render_table_rows(
        [
            [
                str(item.get("name", "")),
                str(item.get("method", "")),
                str(item.get("status_code", "")),
                "是" if item.get("ok") else "否",
            ]
            for item in nexent_api_calls
        ]
    )
    search_rows = _render_table_rows(
        [
            [
                str(item.get("status", "")),
                str(item.get("query", "")),
                ", ".join(item.get("knowledge_base_ids", [])),
                str(item.get("result_count", 0)),
                str(item.get("error", "") or item.get("reason", "")),
            ]
            for item in datamate_search_trace
        ]
    )

    dashboard_summary = {
        "documents": analysis_summary.get("document_count", 0),
        "entities": analysis_summary.get("entity_count", 0),
        "triples": analysis_summary.get("triple_count", 0),
        "queries": analysis_summary.get("query_count", 0),
        "subgraphs": analysis_summary.get("subgraph_count", 0),
    }
    status_badges = [
        f"<span class=\"pill\">DataMate 可达：{'是' if integration_summary.get('datamate_reachable') else '否'}</span>",
        f"<span class=\"pill\">Nexent 可达：{'是' if integration_summary.get('nexent_reachable') else '否'}</span>",
        f"<span class=\"pill\">检索成功：{'是' if integration_summary.get('datamate_search_success') else '否'}</span>",
        f"<span class=\"pill\">NL2SQL 准确率：{nl2sql_eval_summary}</span>",
    ]

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>医疗 Demo 仪表盘</title>
  <style>
    :root {{
      --bg: #f6f8fc;
      --panel: #ffffff;
      --line: #d9e0ec;
      --text: #172033;
      --muted: #5e6a82;
      --blue: #2457d6;
      --teal: #0f766e;
      --shadow: 0 8px 24px rgba(23, 32, 51, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    a {{ color: var(--blue); text-decoration: none; }}
    .wrap {{ max-width: 1440px; margin: 0 auto; padding: 18px; }}
    .hero, .band {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    .hero {{
      padding: 18px 20px;
      margin-bottom: 16px;
    }}
    .hero h1 {{
      margin: 0 0 6px;
      font-size: 22px;
      letter-spacing: 0;
    }}
    .hero p {{
      margin: 0 0 12px;
      color: var(--muted);
    }}
    .pill-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      padding: 5px 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #fbfcff;
      font-size: 12px;
      color: var(--text);
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .stat {{
      padding: 14px 16px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    .stat .k {{ color: var(--muted); font-size: 12px; }}
    .stat .v {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 16px;
    }}
    .band {{
      padding: 16px;
      margin-bottom: 16px;
    }}
    .band h2 {{
      margin: 0 0 12px;
      font-size: 15px;
      letter-spacing: 0;
    }}
    .chart {{
      display: grid;
      gap: 8px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 160px minmax(0, 1fr) 54px;
      gap: 10px;
      align-items: center;
    }}
    .bar-label, .bar-value {{ font-size: 12px; }}
    .bar-track {{
      height: 12px;
      background: #edf1f7;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar-fill {{ height: 100%; border-radius: 999px; }}
    .empty {{ color: var(--muted); font-size: 12px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      padding: 8px 6px;
      border-bottom: 1px solid #edf0f5;
      text-align: left;
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; word-break: break-word; }}
    .footer {{
      color: var(--muted);
      font-size: 12px;
      padding: 10px 2px 18px;
    }}
    @media (max-width: 1100px) {{
      .stats, .grid {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 120px minmax(0, 1fr) 40px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>医疗 Demo 仪表盘</h1>
      <p>任务一到任务三的统一展示页，直接读取本仓库生成的图谱、分析、问答和平台联调产物。</p>
      <div class="pill-list">
        <span class="pill">API：<code>/api/build</code></span>
        <span class="pill">问答：<code>/api/ask</code></span>
        <span class="pill">分析：<code>/api/analyze</code></span>
        <span class="pill">图谱：<code>/api/graph</code></span>
        <span class="pill">报告：<code>/api/report</code></span>
      </div>
      <div class="pill-list" style="margin-top:10px;">
        {''.join(status_badges)}
      </div>
    </section>

    <section class="stats">
      <div class="stat"><div class="k">文档数</div><div class="v">{dashboard_summary["documents"]}</div></div>
      <div class="stat"><div class="k">实体数</div><div class="v">{dashboard_summary["entities"]}</div></div>
      <div class="stat"><div class="k">三元组数</div><div class="v">{dashboard_summary["triples"]}</div></div>
      <div class="stat"><div class="k">分析查询数</div><div class="v">{dashboard_summary["queries"]}</div></div>
      <div class="stat"><div class="k">子图数</div><div class="v">{dashboard_summary["subgraphs"]}</div></div>
    </section>

    <section class="grid">
      <div class="band">
        <h2>实体类型分布</h2>
        <div class="chart">{entity_chart}</div>
      </div>
      <div class="band">
        <h2>关系类型分布</h2>
        <div class="chart">{relation_chart}</div>
      </div>
    </section>

    <section class="band">
      <h2>图谱预览</h2>
      <table>
        <thead><tr><th>名称</th><th>类型</th><th>来源文档</th></tr></thead>
        <tbody>{graph_rows}</tbody>
      </table>
    </section>

    <section class="band">
      <h2>最近问答</h2>
      <table>
        <thead><tr><th>问题</th><th>答案</th><th>引用三元组数</th></tr></thead>
        <tbody>{qa_rows}</tbody>
      </table>
    </section>

    <section class="band">
      <h2>NL2SQL 结果</h2>
      <p class="empty">评测准确率：{nl2sql_eval_summary}；阈值：85%。</p>
      <table>
        <thead><tr><th>问题</th><th>结果行数</th><th>图表名</th></tr></thead>
        <tbody>{nl2sql_rows}</tbody>
      </table>
    </section>

    <section class="grid">
      <div class="band">
        <h2>DataMate 联调</h2>
        <table>
          <thead><tr><th>调用</th><th>方法</th><th>状态码</th><th>成功</th></tr></thead>
          <tbody>{datamate_rows}</tbody>
        </table>
      </div>
      <div class="band">
        <h2>Nexent 联调</h2>
        <table>
          <thead><tr><th>调用</th><th>方法</th><th>状态码</th><th>成功</th></tr></thead>
          <tbody>{nexent_rows}</tbody>
        </table>
      </div>
    </section>

    <section class="band">
      <h2>服务探测</h2>
      <table>
        <thead><tr><th>服务</th><th>地址</th><th>状态码</th><th>可达</th><th>错误</th></tr></thead>
        <tbody>{service_rows}</tbody>
      </table>
    </section>

    <section class="band">
      <h2>datamate_search 证据</h2>
      <table>
        <thead><tr><th>状态</th><th>查询</th><th>知识库 ID</th><th>结果数</th><th>错误</th></tr></thead>
        <tbody>{search_rows}</tbody>
      </table>
    </section>

    <div class="footer">
      图谱与分析数据来自 `graph/*.jsonl`、`demo/analysis_*.jsonl`、`demo/charts/*.json` 和 `demo/integration/*.json`。
    </div>
  </div>
</body>
</html>"""


def export_dashboard_html(project_root: Path, output_path: Path | None = None) -> Path:
    """Write the dashboard as a standalone HTML artifact."""
    target = output_path or (project_root / "demo" / "dashboard.html")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_render_dashboard_html(project_root), encoding="utf-8")
    return target


def handle_request(method: str, route: str, body: dict[str, Any] | None, project_root: Path) -> tuple[int, str, bytes]:
    route = urlparse(route).path
    payload = body or {}

    if method == "GET":
        if route == "/":
            return _json_response(
                HTTPStatus.OK,
                {
                    "service": "medical-demo",
                    "status": "ok",
                    "endpoints": [
                        "/health",
                        "/openapi.json",
                        "/api/openapi.json",
                        "/dashboard",
                        "/api/process",
                        "/api/datamate-process",
                        "/api/task1/openapi.json",
                        "/api/task2/openapi.json",
                        "/api/task3/openapi.json",
                        "/api/orchestrate",
                        "/api/analyze",
                        "/api/nl2sql-eval",
                        "/api/build",
                        "/api/medical-kg/build",
                        "/api/ask",
                        "/api/medical-kg/ask",
                        "/api/integrate",
                        "/api/graph",
                        "/api/report",
                        "/api/qa-trace",
                    ],
                },
            )

        if route == "/dashboard":
            return _text_response(HTTPStatus.OK, _render_dashboard_html(project_root), "text/html; charset=utf-8")

        if route == "/health":
            graph_path = project_root / "graph" / "graph_snapshot.json"
            return _json_response(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "llm_configured": is_llm_configured(),
                    "graph_ready": graph_path.exists(),
                },
            )

        if route in {"/openapi.json", "/api/openapi.json"}:
            server_url = "http://127.0.0.1:8088"
            return _json_response(HTTPStatus.OK, build_task1_openapi_spec(server_url))

        if route == "/api/task1/openapi.json":
            server_url = "http://127.0.0.1:8088"
            return _json_response(HTTPStatus.OK, build_task1_openapi_spec(server_url))

        if route == "/api/task2/openapi.json":
            server_url = "http://127.0.0.1:8088"
            return _json_response(HTTPStatus.OK, build_task2_openapi_spec(server_url))

        if route == "/api/task3/openapi.json":
            server_url = "http://127.0.0.1:8088"
            return _json_response(HTTPStatus.OK, build_task3_openapi_spec(server_url))

        if route == "/api/graph":
            graph_path = project_root / "graph" / "graph_snapshot.json"
            if not graph_path.exists():
                return _json_response(HTTPStatus.NOT_FOUND, {"error": "graph_snapshot_not_found"})
            return _json_response(HTTPStatus.OK, json.loads(graph_path.read_text(encoding="utf-8")))

        if route == "/api/report":
            report_path = project_root / "demo" / "analysis_report.md"
            if not report_path.exists():
                return _json_response(HTTPStatus.NOT_FOUND, {"error": "analysis_report_not_found"})
            return _text_response(HTTPStatus.OK, report_path.read_text(encoding="utf-8"), "text/markdown; charset=utf-8")

        if route == "/api/qa-trace":
            trace_path = project_root / "demo" / "qa_trace.jsonl"
            if not trace_path.exists():
                return _json_response(HTTPStatus.NOT_FOUND, {"error": "qa_trace_not_found"})
            rows = []
            for line in trace_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
            return _json_response(HTTPStatus.OK, {"items": rows})

        return _json_response(HTTPStatus.NOT_FOUND, {"error": "not_found", "path": route})

    if method == "POST":
        if route == "/api/process":
            input_path = _resolve_path(payload.get("input_path"), project_root)
            output_root = _resolve_path(payload.get("output_root"), project_root)
            task = payload.get("task")
            if task is not None and not isinstance(task, str):
                return _json_response(HTTPStatus.BAD_REQUEST, {"error": "task_must_be_string"})
            try:
                result = run_data_processing_agent(input_path, output_root, task=task)
            except FileNotFoundError as exc:
                return _json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except ValueError as exc:
                return _json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return _json_response(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "documents": len(result["documents"]),
                    "goal": result["goal"],
                    "plan": result["plan"],
                    "planning_decision": result.get("planning_decision", {}),
                    "state": result["state"],
                    "errors": result["errors"],
                    "recovery_actions": result["recovery_actions"],
                    "quality_report": result["quality_report"],
                    "output_paths": result["output_paths"],
                    "output_root": str(output_root),
                },
            )

        if route == "/api/datamate-process":
            raw_input_paths = payload.get("input_paths") or []
            if raw_input_paths and not isinstance(raw_input_paths, list):
                return _json_response(HTTPStatus.BAD_REQUEST, {"error": "input_paths_must_be_array"})
            input_paths = [
                _resolve_path(str(item), project_root)
                for item in raw_input_paths
                if isinstance(item, str) and item.strip()
            ]
            input_path_value = payload.get("input_path")
            if not input_path_value and input_paths:
                input_path = input_paths[0]
            else:
                input_path = _resolve_path(input_path_value, project_root)
            output_root = _resolve_path(payload.get("output_root"), project_root)
            task = payload.get("task") or "基于DataMate完成医疗数据接入、清洗、转换和质量处理"
            datamate_api_base = payload.get("datamate_api_base") or "http://localhost:8080"
            wait_seconds = float(payload.get("wait_seconds", 60))
            if not isinstance(task, str):
                return _json_response(HTTPStatus.BAD_REQUEST, {"error": "task_must_be_string"})
            if not isinstance(datamate_api_base, str):
                return _json_response(HTTPStatus.BAD_REQUEST, {"error": "datamate_api_base_must_be_string"})
            try:
                result = run_datamate_processing_agent(
                    input_path,
                    output_root,
                    input_paths=input_paths or None,
                    task=task,
                    datamate_api_base=datamate_api_base,
                    wait_seconds=wait_seconds,
                )
            except FileNotFoundError as exc:
                return _json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            http_status = HTTPStatus.OK if result["status"] == "completed" else HTTPStatus.BAD_GATEWAY
            return _json_response(http_status, result)

        if route == "/api/orchestrate":
            input_path = _resolve_path(payload.get("input_path"), project_root)
            output_root = _resolve_path(payload.get("output_root"), project_root)
            goal = payload.get("goal", "形成数据-知识-洞察端到端医疗智能体链路")
            if not isinstance(goal, str) or not goal.strip():
                return _json_response(HTTPStatus.BAD_REQUEST, {"error": "goal_must_be_string"})
            questions = payload.get("questions")
            if questions is not None and not isinstance(questions, list):
                return _json_response(HTTPStatus.BAD_REQUEST, {"error": "questions_must_be_list"})
            use_llm = payload.get("use_llm")
            if use_llm is not None:
                use_llm = bool(use_llm)
            kg_backend = payload.get("kg_backend")
            if kg_backend is not None and kg_backend not in {"rule", "tensor", "llm"}:
                return _json_response(HTTPStatus.BAD_REQUEST, {"error": "kg_backend_must_be_rule_tensor_or_llm"})
            external_documents = payload.get("external_documents")
            external_triples = payload.get("external_triples")
            external_entities = payload.get("external_entities")
            for field_name, value in {
                "external_documents": external_documents,
                "external_triples": external_triples,
                "external_entities": external_entities,
            }.items():
                if value is not None and not isinstance(value, str):
                    return _json_response(HTTPStatus.BAD_REQUEST, {"error": f"{field_name}_must_be_string"})
            try:
                result = run_orchestrator(
                    input_path=input_path,
                    output_root=output_root,
                    goal=goal.strip(),
                    questions=questions,
                    use_llm=use_llm,
                    kg_backend=kg_backend,
                    external_documents_path=_resolve_path(external_documents, project_root) if external_documents else None,
                    external_triples_path=_resolve_path(external_triples, project_root) if external_triples else None,
                    external_entities_path=_resolve_path(external_entities, project_root) if external_entities else None,
                )
            except FileNotFoundError as exc:
                return _json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return _json_response(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "pipeline": result["pipeline"],
                    "contest_agents": result["contest_agents"],
                    "metrics": result["metrics"],
                    "artifact_paths": result["artifact_paths"],
                    "output_root": str(output_root),
                },
            )

        if route == "/api/analyze":
            output_root = _resolve_path(payload.get("output_root"), project_root)
            try:
                result = run_analysis_agent(output_root)
            except FileNotFoundError as exc:
                return _json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            task3_lineage = result.get("task3_lineage", {})
            graph_understanding = result.get("graph_understanding", {})
            visualization_manifest = result.get("visualization_manifest", {})
            closed_loop = result.get("task3_closed_loop_status", {})
            return _json_response(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "agent": "AnalysisAgent",
                    "analysis_summary": result["analysis_summary"],
                    "analysis_planning": result.get("analysis_planning", {}),
                    "query_count": len(result["analysis_queries"]),
                    "subgraph_count": len(result["subgraphs"]),
                    "nl2sql_eval": {
                        "accuracy": result["nl2sql_eval"]["accuracy"],
                        "passed": result["nl2sql_eval"]["passed"],
                        "total": result["nl2sql_eval"]["total"],
                        "meets_threshold": result["nl2sql_eval"]["meets_threshold"],
                    },
                    "task3_lineage_ready": bool(task3_lineage.get("ready")),
                    "task3_closed_loop_ready": bool(closed_loop.get("ready")),
                    "graph_understanding": graph_understanding,
                    "visualization_manifest": visualization_manifest,
                    "task3_closed_loop_status": closed_loop,
                    "artifact_paths": _task3_artifact_paths(output_root),
                    "output_root": str(output_root),
                },
            )

        if route == "/api/insight":
            question = payload.get("question")
            if not isinstance(question, str) or not question.strip():
                return _json_response(HTTPStatus.BAD_REQUEST, {"error": "question_required"})
            output_root = _resolve_path(payload.get("output_root"), project_root)
            try:
                result = run_interactive_analysis(output_root, question.strip())
            except FileNotFoundError as exc:
                return _json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return _json_response(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "agent": "AnalysisAgent",
                    "question": result["question"],
                    "query": result["query"],
                    "rows": result["rows"],
                    "row_count": result["row_count"],
                    "chart": result["chart"],
                    "subgraph": result["subgraph"],
                    "graph_context": result["graph_context"],
                    "nl2sql_plan": result["nl2sql_plan"],
                    "insight_plan": result["insight_plan"],
                    "insights": result["insights"],
                    "recommended_followups": result["recommended_followups"],
                    "explanation": result["explanation"],
                    "trace": result["trace"],
                    "artifact_paths": _task3_artifact_paths(output_root),
                    "output_root": str(output_root),
                },
            )

        if route == "/api/analyze-config":
            output_root = _resolve_path(payload.get("output_root"), project_root)
            config = payload.get("config")
            config_path = payload.get("config_path")
            if config is None and config_path is not None:
                if not isinstance(config_path, str):
                    return _json_response(HTTPStatus.BAD_REQUEST, {"error": "config_path_must_be_string"})
                try:
                    config = json.loads(_resolve_path(config_path, project_root).read_text(encoding="utf-8"))
                except FileNotFoundError as exc:
                    return _json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            if not isinstance(config, dict):
                return _json_response(HTTPStatus.BAD_REQUEST, {"error": "config_required"})
            try:
                result = run_configured_analysis(output_root, config)
            except (FileNotFoundError, ValueError) as exc:
                return _json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return _json_response(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "analysis_id": result["analysis_id"],
                    "summary": result["summary"],
                    "queries": result["queries"],
                    "charts": result["charts"],
                    "graph_tool_plan": result["graph_tool_plan"],
                    "graph_tool_result": result["graph_tool_result"],
                    "focus_disease": result["focus_disease"],
                    "subgraph": result["subgraph"],
                    "trace": result["trace"],
                    "artifact_paths": result["artifact_paths"],
                    "output_root": str(output_root),
                },
            )

        if route == "/api/nl2sql-eval":
            output_root = _resolve_path(payload.get("output_root"), project_root)
            eval_path = payload.get("eval_path")
            if eval_path is not None and not isinstance(eval_path, str):
                return _json_response(HTTPStatus.BAD_REQUEST, {"error": "eval_path_must_be_string"})
            threshold = payload.get("threshold", 0.85)
            try:
                threshold = float(threshold)
            except (TypeError, ValueError):
                return _json_response(HTTPStatus.BAD_REQUEST, {"error": "threshold_must_be_number"})
            try:
                result = run_nl2sql_eval(
                    output_root,
                    eval_path=Path(eval_path) if eval_path else None,
                    threshold=threshold,
                )
            except FileNotFoundError as exc:
                return _json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return _json_response(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "accuracy": result["accuracy"],
                    "passed": result["passed"],
                    "total": result["total"],
                    "meets_threshold": result["meets_threshold"],
                    "output_paths": result["output_paths"],
                },
            )

        if route in {"/api/build", "/api/medical-kg/build"}:
            response = _run_medical_kg_build(payload, project_root)
            if route == "/api/build" and response[0] == HTTPStatus.OK:
                legacy_payload = json.loads(response[2].decode("utf-8"))
                legacy_payload["status"] = "ok"
                return _json_response(HTTPStatus.OK, legacy_payload)
            return response

        if route in {"/api/ask", "/api/medical-kg/ask"}:
            return _run_medical_kg_ask(payload, project_root)

        if route == "/api/integrate":
            output_root = _resolve_path(payload.get("output_root"), project_root)
            query = payload.get("query", "高血压有哪些并发症？")
            if not isinstance(query, str) or not query.strip():
                return _json_response(HTTPStatus.BAD_REQUEST, {"error": "query_required"})
            knowledge_base_id = payload.get("knowledge_base_id")
            if knowledge_base_id is not None and not isinstance(knowledge_base_id, str):
                return _json_response(HTTPStatus.BAD_REQUEST, {"error": "knowledge_base_id_must_be_string"})
            timeout_seconds = payload.get("timeout_seconds", 1.5)
            try:
                timeout_seconds = float(timeout_seconds)
            except (TypeError, ValueError):
                return _json_response(HTTPStatus.BAD_REQUEST, {"error": "timeout_seconds_must_be_number"})
            try:
                result = run_platform_integration_agent(
                    output_root,
                    query=query.strip(),
                    knowledge_base_id=knowledge_base_id,
                    timeout_seconds=timeout_seconds,
                )
            except FileNotFoundError as exc:
                return _json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return _json_response(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "summary": result["summary"],
                    "artifact_paths": result["artifact_paths"],
                    "knowledge_base_ids": result["knowledge_base_ids"],
                    "output_root": str(output_root),
                },
            )

        return _json_response(HTTPStatus.NOT_FOUND, {"error": "not_found", "path": route})

    if method == "OPTIONS":
        return HTTPStatus.NO_CONTENT, "text/plain; charset=utf-8", b""

    return _json_response(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "method_not_allowed"})


class MedicalDemoHandler(BaseHTTPRequestHandler):
    server_version = "MedicalDemoHTTP/1.0"
    protocol_version = "HTTP/1.1"
    project_root = PROJECT_ROOT

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: A003
        return

    def _send(self, status: int, content_type: str, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def do_OPTIONS(self) -> None:  # noqa: N802
        status, content_type, payload = handle_request("OPTIONS", self.path, None, self.project_root)
        self.send_response(status)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        status, content_type, payload = handle_request("GET", self.path, None, self.project_root)
        self._send(status, content_type, payload)

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = _read_json_body(self)
        except json.JSONDecodeError:
            status, content_type, payload = _json_response(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
            self._send(status, content_type, payload)
            return
        status, content_type, payload = handle_request("POST", self.path, body, self.project_root)
        self._send(status, content_type, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the medical demo HTTP API.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8088, help="Bind port.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    server = ThreadingHTTPServer((args.host, args.port), MedicalDemoHandler)
    print(f"Medical demo API listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
