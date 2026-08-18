from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse

from kg.graph_io import build_multidigraph, load_graph_json
from kg.graph_visualize import build_fallback_html, render_graph_html
from runtime_common.cohort_context import bind_conversation_context, reset_conversation_context
from runtime_common.common import read_json, resolve_path
from tool_server.agent_tools import agent_plan, agent_run
from tool_server.analysis_tools import (
    analysis_query,
    cohort_disease_distribution_query,
    data_summary,
    disease_combination_distribution_query,
    disease_distribution_query,
    followup_high_risk_query,
    graph_driven_analysis,
    metric_query,
    open_analysis_query,
    risk_level_distribution_query,
    trend_query,
)
from tool_server.kg_tools import (
    kg_entity_query,
    kg_patient_path_query,
    kg_query,
    kg_relation_query,
    kg_subgraph_query,
    kg_subgraph_render,
    kg_summary,
)
from tool_server.npu_tools import (
    npu_benchmark_report,
    npu_readiness,
    npu_supported_operators,
    run_npu_enhanced_pipeline,
    run_npu_operator_benchmark,
)
from tool_server.open_sql_tools import (
    open_sql_eval_tool,
    open_sql_examples_tool,
    open_sql_query_tool,
    open_sql_recent_traces_tool,
    open_sql_schema_tool,
)
from tool_server.pipeline_tools import (
    artifacts_status,
    datamate_dag_cancel,
    datamate_dag_graph,
    datamate_dag_plan,
    datamate_dag_run,
    datamate_dag_status,
    datamate_pipeline_latest,
    datamate_pipeline_report,
    datamate_pipeline_report_by_run,
    datamate_pipeline_run_cli_hint,
    datamate_pipeline_status,
    datamate_pipeline_status_by_run,
    datamate_pipelines,
    run_datamate_pipeline,
)
from tool_server.report_tools import (
    chart_gallery_html,
    charts_list,
    report_overview_html,
    report_overview_markdown,
    reports_summary,
)
from tool_server.schemas import (
    AgentGoalRequest,
    AnalysisQueryRequest,
    DataMateDagPlanRequest,
    DataMateDagRunRequest,
    DataMatePipelineRunRequest,
    KGQueryRequest,
    KGTextQueryRequest,
    NPUBenchmarkRequest,
    OpenAnalysisQueryRequest,
    OpenSQLQueryRequest,
    PatientPathQueryRequest,
)
from tool_server.utils import load_server_config, project_identity, safety_note

app = FastAPI(
    title="ChronicCare-Agent Tool Server",
    version="0.4.0",
    description="Nexent-compatible local tool server for chronic disease follow-up data processing, graph QA, open NL2SQL, graph-driven analytics, and reporting.",
)


@app.middleware("http")
async def bind_chroniccare_conversation(request: Request, call_next):
    token = bind_conversation_context(request.headers.get("X-ChronicCare-Conversation-ID"))
    try:
        return await call_next(request)
    finally:
        reset_conversation_context(token)


CHART_ALIAS_MAP = {
    "line_followup_trend_10d.png": "line_followup_trend_10d.svg",
    "pie_risk_distribution_10d.png": "pie_risk_distribution_10d.svg",
    "followup_trend_line_10d.png": "line_followup_trend_10d.svg",
    "disease_inventory_distribution.png": "disease_inventory_distribution.svg",
    "risk_level_distribution.png": "risk_level_distribution.svg",
    "fasting_glucose_distribution.png": "fasting_glucose_distribution.svg",
    "disease_combination_distribution.png": "disease_combination_distribution.svg",
    "disease_combination_distribution.svg": "disease_combination_distribution.svg",
    "cohort_disease_distribution.png": "cohort_disease_distribution_30d.svg",
    "cohort_disease_distribution.svg": "cohort_disease_distribution_30d.svg",
    "hba1c_trend.png": "analysis_trend_hba1c_abnormal_6m.svg",
    "hba1c_trend.svg": "analysis_trend_hba1c_abnormal_6m.svg",
    "hba1c_abnormal_trend_6m.png": "analysis_trend_hba1c_abnormal_6m.svg",
    "hba1c_abnormal_trend_6m.svg": "analysis_trend_hba1c_abnormal_6m.svg",
    "followup_high_risk_45d.png": "line_followup_trend_high_risk_45d.svg",
    "followup_high_risk_45d.svg": "line_followup_trend_high_risk_45d.svg",
}

GRAPH_DRIVEN_ALIAS_MAP = {
    "analysis_disease_distribution": "analysis_disease_inventory",
    "analysis_disease_distribution_chart": "analysis_disease_inventory_chart",
    "analysis_disease_distribution_30d_followup": "analysis_disease_inventory",
    "analysis_disease_distribution_30d_followup_chart": "analysis_disease_inventory_chart",
    "analysis_patient_disease_distribution": "analysis_disease_inventory",
    "analysis_patient_disease_distribution_chart": "analysis_disease_inventory_chart",
    "analysis_metric_query": "analysis_metric_diabetes_avg_fpg",
    "analysis_metric_query_chart": "analysis_metric_diabetes_avg_fpg_chart",
    "analysis_cohort_disease": "analysis_future_30d_high_risk_followup_disease_distribution",
    "analysis_cohort_disease_chart": "analysis_future_30d_high_risk_followup_disease_distribution_chart",
    "analysis_datamate_pipeline": "analysis_disease_inventory",
    "analysis_datamate_pipeline_chart": "analysis_disease_inventory_chart",
}

GRAPH_DRIVEN_SUBGRAPH_ALIAS_MAP = {
    "kg_hypertension_subgraph_interactive": "subgraph_cohort_subgraph_hypertension",
    "kg_hypertension_subgraph.png": "subgraph_cohort_subgraph_hypertension",
    "kg_hypertension_graph.png": "subgraph_cohort_subgraph_hypertension",
    "subgraph_hypertension": "subgraph_cohort_subgraph_hypertension",
    "subgraph_hypertension_preview": "subgraph_cohort_subgraph_hypertension",
    "subgraph_hypertension_preview.png": "subgraph_cohort_subgraph_hypertension",
    "subgraph_hypertension_preview.svg": "subgraph_cohort_subgraph_hypertension",
    "hypertension_subgraph": "subgraph_cohort_subgraph_hypertension",
    "hypertension_subgraph_preview": "subgraph_cohort_subgraph_hypertension",
    "hypertension_subgraph_preview.png": "subgraph_cohort_subgraph_hypertension",
    "hypertension_subgraph_preview.svg": "subgraph_cohort_subgraph_hypertension",
    "htn_subgraph.png": "subgraph_cohort_subgraph_hypertension",
    "subgraph_diabetes": "subgraph_cohort_subgraph_diabetes",
    "subgraph_diabetes_preview": "subgraph_cohort_subgraph_diabetes",
    "subgraph_diabetes_preview.png": "subgraph_cohort_subgraph_diabetes",
    "diabetes_subgraph.png": "subgraph_cohort_subgraph_diabetes",
    "subgraph_stroke": "subgraph_cohort_subgraph_stroke_post",
    "subgraph_stroke_preview": "subgraph_cohort_subgraph_stroke_post",
    "subgraph_stroke_preview.png": "subgraph_cohort_subgraph_stroke_post",
    "high_risk_subgraph.png": "subgraph_cohort_subgraph_high_risk",
    "kg_subgraph_hypertension": "subgraph_cohort_subgraph_hypertension",
    "kg_subgraph_diabetes": "subgraph_cohort_subgraph_diabetes",
    "kg_subgraph_stroke": "subgraph_cohort_subgraph_stroke_post",
    "kg_subgraph_stroke.html": "subgraph_cohort_subgraph_stroke_post",
    "kg_subgraph_stroke.svg": "subgraph_cohort_subgraph_stroke_post",
    "kg_subgraph_stroke_post": "subgraph_cohort_subgraph_stroke_post",
    "kg_subgraph_stroke_post.html": "subgraph_cohort_subgraph_stroke_post",
    "kg_subgraph_stroke_post.svg": "subgraph_cohort_subgraph_stroke_post",
    "kg_subgraph_中风": "subgraph_cohort_subgraph_stroke_post",
    "kg_subgraph_中风.html": "subgraph_cohort_subgraph_stroke_post",
    "kg_subgraph_中风.svg": "subgraph_cohort_subgraph_stroke_post",
    "kg_subgraph_脑卒中": "subgraph_cohort_subgraph_stroke_post",
    "kg_subgraph_脑卒中.html": "subgraph_cohort_subgraph_stroke_post",
    "kg_subgraph_脑卒中.svg": "subgraph_cohort_subgraph_stroke_post",
}

SUBGRAPH_GENERATION_MAP = {
    "subgraph_cohort_subgraph_diabetes": "请生成糖尿病患者群体的图谱子图",
    "subgraph_cohort_subgraph_hypertension": "请生成高血压患者群体的图谱子图",
    "subgraph_cohort_subgraph_diabetes_hypertension": "请生成高血压合并糖尿病群体的图谱子图",
    "subgraph_cohort_subgraph_high_risk": "请生成高风险患者群体的图谱子图",
    "subgraph_cohort_subgraph_medium_risk": "请生成中风险患者群体的图谱子图",
    "subgraph_cohort_subgraph_low_risk": "请生成低风险患者群体的图谱子图",
    "subgraph_cohort_subgraph_stroke_post": "请生成中风（脑卒中）患者群体的图谱子图",
    "subgraph_graph_query_high_salt_bp_abnormal_relation": "画出高盐饮食和血压异常之间的关系。",
}


def _resolve_subgraph_alias_id(subgraph_id: str) -> str:
    normalized = str(subgraph_id or "").strip()
    if not normalized:
        return normalized
    normalized = _normalize_subgraph_id(normalized)
    legacy_map = {
        "subgraph_high_salt_hypertension": "画出高盐饮食和血压异常之间的关系。",
    }
    if normalized in GRAPH_DRIVEN_SUBGRAPH_ALIAS_MAP:
        return GRAPH_DRIVEN_SUBGRAPH_ALIAS_MAP[normalized]
    legacy_query = _legacy_subgraph_query_from_id(normalized)
    if legacy_query:
        payload = kg_subgraph_render(legacy_query, max_nodes=96)
        resolved = str(payload.get("subgraph_id") or "").strip()
        if resolved:
            return resolved
    if normalized.startswith("subgraph_cohort_subgraph_") and normalized.endswith("_post"):
        # Only stroke_post is a real disease key. Other "<disease>_post" ids are
        # legacy/agent suffix mistakes and should fall back to the disease id.
        without_post = normalized[:-5]
        if without_post != "subgraph_cohort_subgraph_stroke":
            return without_post
    if normalized in legacy_map:
        payload = kg_subgraph_render(legacy_map[normalized], max_nodes=96)
        resolved = str(payload.get("subgraph_id") or "").strip()
        if resolved:
            return resolved
    return normalized


def _request_payload_value(payload: Any, *keys: str) -> str:
    if payload is None:
        return ""
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""
    for key in keys:
        value = getattr(payload, key, None)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _request_payload_int(payload: Any, *keys: str) -> int | None:
    raw = _request_payload_value(payload, *keys)
    if not raw:
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _coerce_followup_question(payload: Any) -> str:
    question = _request_payload_value(payload, "question", "query")
    if question:
        return question
    days = _request_payload_int(payload, "days", "day", "time_window", "window_days")
    risk_level = _request_payload_value(payload, "risk_level", "risk", "cohort_name")
    if days is not None:
        normalized_risk = risk_level or "高风险"
        return f"未来 {days} 天需要随访的{normalized_risk}患者有多少？"
    return ""


def _coerce_metric_question(payload: Any) -> str:
    question = _request_payload_value(payload, "question", "query")
    if question:
        return question
    disease = _request_payload_value(payload, "disease", "disease_name", "cohort_name")
    metric = _request_payload_value(payload, "metric", "indicator", "indicator_name")
    if disease and metric:
        if any(token in metric for token in ("平均", "均值", "mean")):
            return f"{disease}患者的{metric}是多少？"
        return f"{disease}患者的{metric}是多少？"
    return ""


def _coerce_trend_question(payload: Any) -> str:
    question = _request_payload_value(payload, "question", "query")
    if question:
        return question
    disease = _request_payload_value(payload, "disease", "disease_name", "cohort_name")
    metric = _request_payload_value(payload, "metric", "indicator", "indicator_name")
    window = _request_payload_value(payload, "window", "time_window", "period")
    if disease and metric:
        prefix = f"{window}" if window else "最近一段时间"
        return f"{prefix}{disease}患者的{metric}趋势如何？"
    return ""


def _coerce_subgraph_query(payload: Any) -> tuple[str, int]:
    query = _request_payload_value(payload, "query", "question")
    max_nodes = _request_payload_int(payload, "max_nodes") or 80
    if query:
        return query, max_nodes
    disease = _request_payload_value(payload, "disease", "disease_name", "cohort_name")
    if disease:
        return f"{disease}的知识图谱子图", max_nodes
    relation = _request_payload_value(payload, "relation_query", "topic")
    if relation:
        return relation, max_nodes
    return "", max_nodes


def _artifact_path(path_str: str) -> Path:
    path = resolve_path(path_str)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {path_str}")
    return path


def _ensure_default_graph_artifact(cfg: dict) -> Path:
    target = resolve_path(cfg["paths"]["graph_html"])
    graph_json_path = resolve_path(cfg["paths"]["graph_json"])
    graph_summary_path = resolve_path(cfg["paths"]["graph_summary"])
    if not graph_json_path.exists() or not graph_summary_path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {cfg['paths']['graph_html']}")
    graph_doc = load_graph_json(graph_json_path)
    summary_doc = read_json(graph_summary_path)
    note = "当前未找到预渲染图谱页，已根据最新同步的图谱 JSON 自动生成默认概览页面。"
    try:
        graph = build_multidigraph(graph_doc.get("nodes", []), graph_doc.get("edges", []))
        rendered, display_node_count, display_edge_count = render_graph_html(
            graph,
            target,
            total_node_count=int(summary_doc.get("node_count", 0) or 0),
            total_edge_count=int(summary_doc.get("edge_count", 0) or 0),
        )
        if not rendered:
            build_fallback_html(
                target,
                summary=summary_doc,
                entity_type_count=summary_doc.get("entity_type_count", {}) or {},
                relation_type_count=summary_doc.get("relation_type_count", {}) or {},
                top_degree_nodes=[],
                note=(f"{note} 当前展示为摘要页。展示节点数 {display_node_count}，展示边数 {display_edge_count}。"),
            )
    except Exception:
        build_fallback_html(
            target,
            summary=summary_doc,
            entity_type_count=summary_doc.get("entity_type_count", {}) or {},
            relation_type_count=summary_doc.get("relation_type_count", {}) or {},
            top_degree_nodes=[],
            note=f"{note} 当前展示为摘要页。",
        )
    if target.exists():
        return target
    raise HTTPException(status_code=404, detail=f"Artifact not found: {cfg['paths']['graph_html']}")


def _normalize_subgraph_id(subgraph_id: str) -> str:
    normalized = str(subgraph_id or "").strip()
    normalized = unquote(normalized)
    if normalized.endswith(".html"):
        normalized = normalized[:-5]
    if normalized.endswith(".svg"):
        normalized = normalized[:-4]
    if normalized.endswith(".png"):
        normalized = normalized[:-4]
    normalized = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", normalized)
    return normalized.strip("_-")


def _legacy_subgraph_query_from_id(subgraph_id: str) -> str | None:
    normalized = _normalize_subgraph_id(subgraph_id)
    topic = normalized
    for prefix in ("subgraph_", "kg_subgraph_", "analysis_kg_subgraph_"):
        if topic.startswith(prefix):
            topic = topic[len(prefix) :]
            break
    topic = topic.strip("_- ")
    if not topic:
        return None
    alias_map = {
        "hypertension": "高血压的知识图谱子图",
        "高血压": "高血压的知识图谱子图",
        "高血压知识图谱子图": "高血压的知识图谱子图",
        "高血压的知识图谱子图": "高血压的知识图谱子图",
        "diabetes": "糖尿病的知识图谱子图",
        "糖尿病": "糖尿病的知识图谱子图",
        "糖尿病知识图谱子图": "糖尿病的知识图谱子图",
        "stroke": "中风的知识图谱子图",
        "stroke_post": "中风的知识图谱子图",
        "中风": "中风的知识图谱子图",
        "脑卒中": "中风的知识图谱子图",
        "high_risk": "请生成高风险患者群体的图谱子图",
        "高风险": "请生成高风险患者群体的图谱子图",
        "高风险患者": "请生成高风险患者群体的图谱子图",
        "medium_risk": "请生成中风险患者群体的图谱子图",
        "low_risk": "请生成低风险患者群体的图谱子图",
        "high_salt_hypertension": "画出高盐饮食和血压异常之间的关系。",
        "高盐饮食血压异常": "画出高盐饮食和血压异常之间的关系。",
    }
    compact = re.sub(r"[\s_的]+", "", topic)
    for key, query in alias_map.items():
        if compact == re.sub(r"[\s_的]+", "", key):
            return query
    if any(token in topic for token in ("高盐饮食", "血压异常")):
        return "画出高盐饮食和血压异常之间的关系。"
    if re.search(r"[\u4e00-\u9fff]", topic):
        return topic if any(token in topic for token in ("图谱", "子图", "关系图")) else f"{topic}的知识图谱子图"
    return None


def _ensure_subgraph_artifact(subgraph_id: str) -> Path:
    normalized_subgraph_id = _resolve_subgraph_alias_id(subgraph_id)
    for candidate_str in [
        f"outputs/runtime_generated/subgraphs/{normalized_subgraph_id}.html",
        f"outputs/local_runtime/subgraphs/{normalized_subgraph_id}.html",
        f"outputs/subgraphs/{normalized_subgraph_id}.html",
    ]:
        path = resolve_path(candidate_str)
        if path.exists():
            return path
    query = SUBGRAPH_GENERATION_MAP.get(normalized_subgraph_id) or _legacy_subgraph_query_from_id(
        normalized_subgraph_id
    )
    if query:
        payload = kg_subgraph_render(query, max_nodes=96)
        generated_path = payload.get("html_path")
        if generated_path:
            candidate = resolve_path(str(generated_path))
            if candidate.exists():
                return candidate
    raise HTTPException(
        status_code=404,
        detail=f"Artifact not found: outputs/runtime_generated/subgraphs/{normalized_subgraph_id}.html",
    )


def _guess_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".html":
        return "text/html"
    if suffix == ".svg":
        return "image/svg+xml"
    if suffix == ".png":
        return "image/png"
    if suffix == ".json":
        return "application/json"
    if suffix == ".md":
        return "text/markdown"
    if suffix == ".csv":
        return "text/csv"
    return "application/octet-stream"


def _no_cache_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }


def _subgraph_preview_svg(subgraph_id: str) -> str:
    normalized_subgraph_id = _normalize_subgraph_id(subgraph_id)
    meta_path = None
    for candidate_str in [
        f"outputs/runtime_generated/subgraphs/{normalized_subgraph_id}.json",
        f"outputs/local_runtime/subgraphs/{normalized_subgraph_id}.json",
        f"outputs/subgraphs/{normalized_subgraph_id}.json",
    ]:
        candidate = resolve_path(candidate_str)
        if candidate.exists():
            meta_path = candidate
            break
    if meta_path is not None and meta_path.exists():
        payload = read_json(meta_path)
    else:
        payload = kg_subgraph_render(
            SUBGRAPH_GENERATION_MAP.get(normalized_subgraph_id, normalized_subgraph_id),
            max_nodes=96,
        )
    seed_labels = [str(item).strip() for item in (payload.get("seed_labels") or []) if str(item).strip()]
    subject = "、".join(seed_labels) if seed_labels else normalized_subgraph_id
    cohort_count = int(payload.get("cohort_patient_count") or 0)
    display_count = int(payload.get("display_patient_node_count") or 0)
    semantic_count = int(payload.get("semantic_node_count") or 0)
    node_count = int(payload.get("node_count") or 0)
    edge_count = int(payload.get("edge_count") or 0)
    has_drugs = bool(payload.get("top_drugs"))
    scope = str(payload.get("graph_scope_explanation") or "已按当前问题实时生成局部子图。").strip()
    drug_node_svg = (
        """
  <circle cx="1080" cy="470" r="24" fill="#7a5ab5"/>
  <text x="1050" y="525" font-size="20" fill="#102a43">常用药物</text>
  <line x1="537" y1="470" x2="1052" y2="470" stroke="#c7d0d9" stroke-width="4"/>
"""
        if has_drugs
        else ""
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
  <defs>
    <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="#f7fbff"/>
      <stop offset="100%" stop-color="#edf5ff"/>
    </linearGradient>
  </defs>
  <rect width="1280" height="720" rx="28" fill="url(#bg)"/>
  <rect x="34" y="34" width="1212" height="652" rx="24" fill="#ffffff" stroke="#d9e2ec"/>
  <text x="70" y="105" font-size="42" font-weight="700" fill="#102a43">{subject}知识图谱子图</text>
  <text x="70" y="150" font-size="22" fill="#486581">此图为当前问题实时生成的局部知识图谱预览，不是全局图谱总览。</text>
  <rect x="70" y="195" width="250" height="120" rx="18" fill="#f8fbff" stroke="#d9e2ec"/>
  <text x="95" y="235" font-size="20" fill="#486581">群体患者总数</text>
  <text x="95" y="285" font-size="44" font-weight="700" fill="#102a43">{cohort_count}</text>
  <rect x="345" y="195" width="250" height="120" rx="18" fill="#f8fbff" stroke="#d9e2ec"/>
  <text x="370" y="235" font-size="20" fill="#486581">示例患者节点</text>
  <text x="370" y="285" font-size="44" font-weight="700" fill="#102a43">{display_count}</text>
  <rect x="620" y="195" width="250" height="120" rx="18" fill="#f8fbff" stroke="#d9e2ec"/>
  <text x="645" y="235" font-size="20" fill="#486581">当前语义节点</text>
  <text x="645" y="285" font-size="44" font-weight="700" fill="#102a43">{semantic_count}</text>
  <rect x="895" y="195" width="300" height="120" rx="18" fill="#f8fbff" stroke="#d9e2ec"/>
  <text x="920" y="235" font-size="20" fill="#486581">本次子图规模</text>
  <text x="920" y="285" font-size="34" font-weight="700" fill="#102a43">节点 {node_count} / 关系 {edge_count}</text>
  <circle cx="175" cy="470" r="28" fill="#2457a5"/>
  <text x="145" y="525" font-size="20" fill="#102a43">示例患者</text>
  <circle cx="505" cy="470" r="32" fill="#d64550"/>
  <text x="445" y="525" font-size="20" fill="#102a43">核心疾病/群体</text>
  <circle cx="815" cy="430" r="24" fill="#2f8f3a"/>
  <text x="780" y="480" font-size="20" fill="#102a43">检查指标</text>
  <circle cx="815" cy="535" r="24" fill="#e07a1f"/>
  <text x="780" y="585" font-size="20" fill="#102a43">风险事件</text>
  <line x1="205" y1="470" x2="470" y2="470" stroke="#c7d0d9" stroke-width="4"/>
  <line x1="537" y1="455" x2="791" y2="430" stroke="#c7d0d9" stroke-width="4"/>
  <line x1="537" y1="485" x2="791" y2="535" stroke="#c7d0d9" stroke-width="4"/>
  {drug_node_svg}
  <foreignObject x="70" y="620" width="1140" height="52">
    <div xmlns="http://www.w3.org/1999/xhtml" style="font-size:18px;color:#486581;line-height:1.5;">
      {scope}
    </div>
  </foreignObject>
</svg>"""


@app.get("/", summary="Root")
def root() -> dict:
    cfg = load_server_config()
    identity = project_identity(cfg)
    return {
        "project": identity["project"],
        "stage": "capability_completion",
        "message": "Welcome to ChronicCare-Agent Tool Server",
        "docs": "/docs",
        "health": "/health",
        "tools": "/tools",
        "openapi": "/openapi.json",
        "service_base_url": identity.get("service_base_url"),
        "safety_note": safety_note(cfg),
    }


@app.get("/favicon.ico", summary="Favicon", status_code=204)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/health", summary="Health")
def health() -> dict:
    cfg = load_server_config()
    identity = project_identity(cfg)
    return {
        "status": "ok",
        "project": identity["project"],
        "stage": "capability_completion",
        "message": "tool server is running",
        "base_url": identity["base_url"],
        "service_base_url": identity.get("service_base_url"),
        "safety_note": safety_note(cfg),
    }


@app.get("/tools", summary="Tools")
def tools() -> dict:
    cfg = load_server_config()
    tool_rows = [
        {"name": "health", "method": "GET", "path": "/health", "summary": "Health"},
        {"name": "tools", "method": "GET", "path": "/tools", "summary": "Tools"},
        {"name": "artifacts_status", "method": "GET", "path": "/artifacts/status", "summary": "Artifacts Status"},
        {"name": "artifacts_summary", "method": "GET", "path": "/artifacts/summary", "summary": "Artifacts Summary"},
        {
            "name": "system_data_summary",
            "method": "GET",
            "path": "/system/data-summary",
            "summary": "System Data Summary",
        },
        {
            "name": "disease_distribution",
            "method": "POST",
            "path": "/analysis/disease-distribution",
            "summary": "Disease Distribution",
        },
        {
            "name": "disease_combination_distribution",
            "method": "POST",
            "path": "/analysis/disease-combination-distribution",
            "summary": "Disease Combination Distribution",
        },
        {
            "name": "risk_level_distribution",
            "method": "POST",
            "path": "/analysis/risk-level-distribution",
            "summary": "Risk Level Distribution",
        },
        {
            "name": "followup_high_risk",
            "method": "POST",
            "path": "/analysis/followup/high-risk",
            "summary": "Future N Days High Risk Followup",
        },
        {
            "name": "cohort_disease_distribution",
            "method": "POST",
            "path": "/analysis/followup/cohort-disease-distribution",
            "summary": "Cohort Disease Distribution",
        },
        {"name": "metric_query", "method": "POST", "path": "/analysis/metric-query", "summary": "Metric Query"},
        {"name": "trend_query", "method": "POST", "path": "/analysis/trend-query", "summary": "Trend Query"},
        {
            "name": "datamate_pipeline_run",
            "method": "POST",
            "path": "/datamate/pipeline/run",
            "summary": "DataMate Pipeline Run",
        },
        {
            "name": "datamate_pipeline_run_npu",
            "method": "POST",
            "path": "/datamate/pipeline/run-npu",
            "summary": "DataMate Pipeline Run With NPU Enhancement",
        },
        {
            "name": "datamate_pipeline_status",
            "method": "GET",
            "path": "/datamate/pipeline/status",
            "summary": "DataMate Pipeline Status",
        },
        {
            "name": "datamate_pipeline_report",
            "method": "GET",
            "path": "/datamate/pipeline/report",
            "summary": "DataMate Pipeline Report",
        },
        {
            "name": "datamate_pipelines",
            "method": "GET",
            "path": "/datamate/pipelines",
            "summary": "DataMate Pipelines Overview",
        },
        {
            "name": "datamate_pipeline_latest",
            "method": "GET",
            "path": "/datamate/pipelines/latest",
            "summary": "DataMate Pipeline Latest",
        },
        {
            "name": "datamate_pipeline_status_by_run",
            "method": "GET",
            "path": "/datamate/pipelines/status/{run_id}",
            "summary": "DataMate Pipeline Status By Run",
        },
        {
            "name": "datamate_pipeline_report_by_run",
            "method": "GET",
            "path": "/datamate/pipelines/report/{run_id}",
            "summary": "DataMate Pipeline Report By Run",
        },
        {
            "name": "datamate_pipeline_cli_hint",
            "method": "GET",
            "path": "/datamate/pipelines/cli-fallback",
            "summary": "DataMate Pipeline CLI Fallback",
        },
        {"name": "npu_readiness", "method": "GET", "path": "/npu/readiness", "summary": "NPU Runtime Readiness"},
        {
            "name": "npu_supported_operators",
            "method": "GET",
            "path": "/npu/supported-operators",
            "summary": "NPU Supported Operators",
        },
        {
            "name": "npu_operator_benchmark",
            "method": "POST",
            "path": "/npu/benchmark",
            "summary": "NPU Operator Benchmark",
        },
        {
            "name": "npu_benchmark_report",
            "method": "GET",
            "path": "/npu/benchmark/report",
            "summary": "Latest NPU Benchmark Report",
        },
        {"name": "kg_summary", "method": "GET", "path": "/kg/summary", "summary": "KG Summary"},
        {"name": "kg_entity_query", "method": "POST", "path": "/kg/entity/query", "summary": "KG Entity Query"},
        {"name": "kg_relation_query", "method": "POST", "path": "/kg/relation/query", "summary": "KG Relation Query"},
        {"name": "kg_patient_path", "method": "POST", "path": "/kg/patient/path", "summary": "KG Patient Path Query"},
        {"name": "kg_subgraph_query", "method": "POST", "path": "/kg/subgraph/query", "summary": "KG Subgraph Query"},
        {
            "name": "kg_subgraph_render",
            "method": "POST",
            "path": "/kg/subgraph/render",
            "summary": "KG Subgraph Render",
        },
        {"name": "analysis_query", "method": "POST", "path": "/analysis/query", "summary": "Analysis Query"},
        {
            "name": "analysis_open_query",
            "method": "POST",
            "path": "/analysis/open-query",
            "summary": "Open Analysis Query",
        },
        {
            "name": "analysis_open_sql_eval",
            "method": "GET",
            "path": "/analysis/open-sql/eval",
            "summary": "Open SQL Eval",
        },
        {
            "name": "analysis_graph_driven",
            "method": "POST",
            "path": "/analysis/graph-driven",
            "summary": "Graph-driven Analysis",
        },
        {"name": "reports_summary", "method": "GET", "path": "/reports/summary", "summary": "Reports Summary"},
        {"name": "charts_list", "method": "GET", "path": "/charts/list", "summary": "Charts List"},
        {"name": "agent_plan", "method": "POST", "path": "/agent/plan", "summary": "Agent Plan"},
        {"name": "agent_run", "method": "POST", "path": "/agent/run", "summary": "Agent Run"},
    ]
    return {
        "status": "success",
        "tool_count": len(tool_rows),
        "tools": tool_rows,
        "safety_note": safety_note(cfg),
    }


@app.get("/artifacts/status", summary="Artifacts Status")
def api_artifacts_status() -> dict:
    return artifacts_status()


@app.get("/artifacts/summary", summary="Artifacts Summary")
def api_artifacts_summary() -> dict:
    return reports_summary()


@app.get("/system/data-summary", summary="System Data Summary")
def api_system_data_summary() -> dict:
    payload = data_summary()
    return {
        "status": payload.get("status"),
        "data_version": payload.get("data_version"),
        "patient_count": payload.get("patient_count"),
        "visit_count": payload.get("visit_count"),
        "lab_result_count": payload.get("lab_result_count"),
        "medication_record_count": payload.get("medication_record_count"),
        "node_count": payload.get("node_count"),
        "edge_count": payload.get("edge_count"),
        "table": payload.get("table"),
        "summary_text": payload.get("summary_text"),
        "safety_note": payload.get("safety_note"),
    }


@app.post("/analysis/disease-distribution", summary="Disease Distribution")
def api_disease_distribution(request: AnalysisQueryRequest) -> dict:
    return disease_distribution_query(request.question)


@app.post("/analysis/disease-combination-distribution", summary="Disease Combination Distribution")
def api_disease_combination_distribution(request: AnalysisQueryRequest) -> dict:
    return disease_combination_distribution_query(request.question)


@app.post("/analysis/risk-level-distribution", summary="Risk Level Distribution")
def api_risk_level_distribution(request: AnalysisQueryRequest) -> dict:
    return risk_level_distribution_query(request.question)


@app.post("/analysis/followup/high-risk", summary="Future N Days High Risk Followup")
def api_followup_high_risk(request: dict[str, Any] | AnalysisQueryRequest) -> dict:
    return followup_high_risk_query(_coerce_followup_question(request))


@app.post("/analysis/followup/cohort-disease-distribution", summary="Cohort Disease Distribution")
def api_cohort_disease_distribution(request: AnalysisQueryRequest) -> dict:
    return cohort_disease_distribution_query(request.question)


@app.post("/analysis/metric-query", summary="Metric Query")
def api_metric_query(request: dict[str, Any] | AnalysisQueryRequest) -> dict:
    return metric_query(_coerce_metric_question(request))


@app.post("/analysis/trend-query", summary="Trend Query")
def api_trend_query(request: dict[str, Any] | AnalysisQueryRequest) -> dict:
    return trend_query(_coerce_trend_question(request))


@app.get("/artifacts/graph", summary="Graph Artifact")
def api_graph_artifact() -> RedirectResponse:
    return RedirectResponse(url="/artifacts/graph.html")


def _latest_subgraph_path() -> Path | None:
    html_paths = []
    for base_str in ["outputs/runtime_generated/subgraphs", "outputs/local_runtime/subgraphs", "outputs/subgraphs"]:
        subgraph_dir = resolve_path(base_str)
        html_paths.extend(subgraph_dir.glob("*.html"))
    html_paths = sorted(html_paths, key=lambda item: item.stat().st_mtime, reverse=True)
    return html_paths[0] if html_paths else None


def _latest_future_followup_bundle_path(chart_page: bool = False) -> Path | None:
    suffix = "_chart.html" if chart_page else ".html"
    candidates = []
    for base_str in ["outputs/runtime_generated/graph_driven_analysis", "outputs/graph_driven_analysis"]:
        analysis_dir = resolve_path(base_str)
        for path in analysis_dir.glob(f"analysis_future_followup_chart_bundle_*d{suffix}"):
            is_chart = path.stem.endswith("_chart")
            if chart_page and not is_chart:
                continue
            if not chart_page and is_chart:
                continue
            candidates.append(path)
    candidates = sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _dynamic_chart_alias(filename: str) -> str | None:
    def latest_dynamic_target(prefix: str) -> str | None:
        matches = []
        for base_str in ["outputs/runtime_generated/charts", "outputs/charts"]:
            chart_dir = resolve_path(base_str)
            matches.extend(chart_dir.glob(f"{prefix}_*d.svg"))
        matches = sorted(matches, key=lambda item: item.stat().st_mtime, reverse=True)
        return matches[0].name if matches else None

    generic_map = {
        "line_followup_trend.png": latest_dynamic_target("line_followup_trend"),
        "pie_risk_distribution.png": latest_dynamic_target("pie_risk_distribution"),
        "followup_trend_line.png": latest_dynamic_target("line_followup_trend"),
        "risk_distribution_pie.png": latest_dynamic_target("pie_risk_distribution"),
    }
    if filename in generic_map and generic_map[filename]:
        return generic_map[filename]
    followup_match = re.match(r"^(?:line_followup_trend|followup_trend_line)_(\d+)d\.(png|svg)$", filename)
    if followup_match:
        days, ext = followup_match.group(1), followup_match.group(2)
        preferred = f"line_followup_trend_high_risk_{days}d.{ext}"
        fallback = f"line_followup_trend_high_risk_{days}d.png"
        for target in [preferred, fallback, f"line_followup_trend_high_risk_{days}d.svg"]:
            for base_str in ["outputs/runtime_generated/charts", "outputs/local_runtime/charts", "outputs/charts"]:
                if resolve_path(f"{base_str}/{target}").exists():
                    return target
        followup_high_risk_query(f"未来 {days} 天需要随访的高风险患者有多少？")
        for target in [preferred, fallback, f"line_followup_trend_high_risk_{days}d.svg"]:
            for base_str in ["outputs/runtime_generated/charts", "outputs/local_runtime/charts", "outputs/charts"]:
                if resolve_path(f"{base_str}/{target}").exists():
                    return target
        return f"line_followup_trend_high_risk_{days}d.svg"
    risk_pie_match = re.match(r"^(?:pie_risk_distribution|risk_distribution_pie)_(\d+)d\.(png|svg)$", filename)
    if risk_pie_match:
        days, ext = risk_pie_match.group(1), risk_pie_match.group(2)
        preferred = f"pie_risk_distribution_high_risk_{days}d.{ext}"
        fallback = f"pie_risk_distribution_high_risk_{days}d.png"
        for target in [preferred, fallback, f"pie_risk_distribution_high_risk_{days}d.svg"]:
            for base_str in ["outputs/runtime_generated/charts", "outputs/local_runtime/charts", "outputs/charts"]:
                if resolve_path(f"{base_str}/{target}").exists():
                    return target
        followup_high_risk_query(f"未来 {days} 天需要随访的高风险患者有多少？")
        for target in [preferred, fallback, f"pie_risk_distribution_high_risk_{days}d.svg"]:
            for base_str in ["outputs/runtime_generated/charts", "outputs/local_runtime/charts", "outputs/charts"]:
                if resolve_path(f"{base_str}/{target}").exists():
                    return target
        return f"pie_risk_distribution_high_risk_{days}d.svg"
    patterns = [
        (r"^followup_high_risk_(\d+)d\.(?:png|svg)$", "line_followup_trend_high_risk_{days}d.svg"),
        (r"^cohort_disease_distribution_(\d+)d\.(?:png|svg)$", "cohort_disease_distribution_{days}d.svg"),
        (r"^hba1c_abnormal_trend_(\d+)m\.(?:png|svg)$", "analysis_trend_hba1c_abnormal_{days}m.svg"),
        (r"^hba1c_trend_(\d+)m\.(?:png|svg)$", "analysis_trend_hba1c_abnormal_{days}m.svg"),
    ]
    for pattern, target in patterns:
        match = re.match(pattern, filename)
        if match:
            return target.format(days=match.group(1))
    if filename.endswith(".png"):
        for base_str in ["outputs/runtime_generated/charts", "outputs/local_runtime/charts", "outputs/charts"]:
            png_candidate = resolve_path(f"{base_str}/{filename}")
            if png_candidate.exists():
                return filename
        svg_candidate = filename[:-4] + ".svg"
        for base_str in ["outputs/runtime_generated/charts", "outputs/charts"]:
            candidate = resolve_path(f"{base_str}/{svg_candidate}")
            if candidate.exists():
                return svg_candidate
    return None


def _legacy_followup_chart_path(filename: str) -> Path | None:
    normalized = str(filename or "").strip()
    if not normalized:
        return None
    match = re.match(r"^followup_high_risk_(\d+)_days(?:\.html)?$", normalized)
    if match:
        days = match.group(1)
        for base_str in ["outputs/runtime_generated/graph_driven_analysis", "outputs/graph_driven_analysis"]:
            candidate = resolve_path(f"{base_str}/analysis_future_followup_chart_bundle_high_risk_{days}d_chart.html")
            if candidate.exists():
                return candidate
    compact_match = re.match(r"^analysis_followup_high_risk_(\d+)d(?:_chart)?$", normalized)
    if compact_match:
        days = compact_match.group(1)
        suffix = "_chart" if normalized.endswith("_chart") else ""
        for base_str in ["outputs/runtime_generated/graph_driven_analysis", "outputs/graph_driven_analysis"]:
            candidate = resolve_path(f"{base_str}/analysis_future_followup_chart_bundle_high_risk_{days}d{suffix}.html")
            if candidate.exists():
                return candidate
    return None


def _legacy_subgraph_html_path(analysis_id: str) -> Path | None:
    normalized = str(analysis_id or "").strip()
    if not normalized:
        return None
    if normalized.endswith(".html"):
        normalized = normalized[:-5]
    decoded = unquote(normalized)
    if not decoded.startswith("kg_subgraph_"):
        return None
    topic = decoded[len("kg_subgraph_") :].strip()
    if not topic:
        return None
    slug_alias_map = {
        "hypertension": "高血压的知识图谱子图",
        "diabetes": "糖尿病的知识图谱子图",
        "stroke": "中风的知识图谱子图",
        "stroke_post": "中风的知识图谱子图",
        "中风": "中风的知识图谱子图",
        "脑卒中": "中风的知识图谱子图",
        "high_risk": "请生成高风险患者群体的图谱子图",
        "medium_risk": "请生成中风险患者群体的图谱子图",
        "low_risk": "请生成低风险患者群体的图谱子图",
        "high_salt_hypertension": "画出高盐饮食和血压异常之间的关系。",
    }
    if topic in slug_alias_map:
        query = slug_alias_map[topic]
    elif any(token in topic for token in ("高盐饮食", "血压异常")) and "关系" not in topic:
        query = f"画出{topic}之间的关系。"
    elif any(token in topic for token in ("图谱", "子图", "关系图")):
        query = topic
    else:
        query = f"{topic}的知识图谱子图"
    payload = kg_subgraph_render(query, max_nodes=96)
    html_path = payload.get("html_path")
    if not html_path:
        return None
    candidate = resolve_path(str(html_path))
    return candidate if candidate.exists() else None


def _legacy_subgraph_preview_svg(analysis_id: str) -> str | None:
    normalized = str(analysis_id or "").strip()
    if not normalized:
        return None
    if normalized.endswith(".html"):
        normalized = normalized[:-5]
    if normalized.endswith(".svg"):
        normalized = normalized[:-4]
    if normalized.endswith(".png"):
        normalized = normalized[:-4]
    if normalized.startswith("analysis_kg_subgraph_"):
        normalized = normalized[len("analysis_kg_subgraph_") :]
    elif normalized.startswith("kg_subgraph_"):
        normalized = normalized[len("kg_subgraph_") :]
    else:
        return None
    if normalized.endswith("_preview"):
        normalized = normalized[: -len("_preview")]
    topic = unquote(normalized).strip()
    if not topic:
        return None
    slug_alias_map = {
        "hypertension": "高血压的知识图谱子图",
        "diabetes": "糖尿病的知识图谱子图",
        "stroke": "中风的知识图谱子图",
        "stroke_post": "中风的知识图谱子图",
        "中风": "中风的知识图谱子图",
        "脑卒中": "中风的知识图谱子图",
        "high_risk": "请生成高风险患者群体的图谱子图",
        "medium_risk": "请生成中风险患者群体的图谱子图",
        "low_risk": "请生成低风险患者群体的图谱子图",
        "high_salt_hypertension": "画出高盐饮食和血压异常之间的关系。",
    }
    if topic in slug_alias_map:
        payload = kg_subgraph_render(slug_alias_map[topic], max_nodes=96)
        subgraph_id = str(payload.get("subgraph_id") or "").strip()
        if not subgraph_id:
            return None
    elif any(token in topic for token in ("高盐饮食", "血压异常")) and "关系" not in topic:
        payload = kg_subgraph_render(f"画出{topic}之间的关系。", max_nodes=96)
        subgraph_id = str(payload.get("subgraph_id") or "").strip()
        if not subgraph_id:
            return None
    elif any(token in topic for token in ("图谱", "子图", "关系图")):
        payload = kg_subgraph_render(topic, max_nodes=96)
        subgraph_id = str(payload.get("subgraph_id") or "").strip()
        if not subgraph_id:
            return None
    elif re.search(r"[\u4e00-\u9fff]", topic):
        payload = kg_subgraph_render(f"{topic}的知识图谱子图", max_nodes=96)
        subgraph_id = str(payload.get("subgraph_id") or "").strip()
        if not subgraph_id:
            return None
    else:
        return None
    _ensure_subgraph_artifact(subgraph_id)
    return _subgraph_preview_svg(subgraph_id)


def _legacy_subgraph_chart_response(filename: str):
    normalized = str(filename or "").strip()
    if not normalized.startswith("kg_subgraph_"):
        return None
    if normalized.endswith(".html"):
        legacy_subgraph = _legacy_subgraph_html_path(normalized)
        if legacy_subgraph is None:
            return None
        return FileResponse(legacy_subgraph, media_type="text/html", headers=_no_cache_headers())
    if normalized.endswith(".svg"):
        legacy_svg = _legacy_subgraph_preview_svg(normalized)
        if legacy_svg is None:
            return None
        return Response(content=legacy_svg, media_type="image/svg+xml", headers=_no_cache_headers())
    if normalized.endswith(".png"):
        legacy_svg = _legacy_subgraph_preview_svg(normalized)
        if legacy_svg is None:
            return None
        return Response(content=legacy_svg, media_type="image/svg+xml", headers=_no_cache_headers())
    return None


@app.api_route("/artifacts/graph.html", methods=["GET", "HEAD"], summary="Graph HTML")
def api_graph_html() -> FileResponse:
    cfg = load_server_config()
    return FileResponse(_ensure_default_graph_artifact(cfg), media_type="text/html", headers=_no_cache_headers())


@app.api_route("/artifacts/graph-overview.html", methods=["GET", "HEAD"], summary="Graph Overview HTML")
def api_graph_overview_html() -> HTMLResponse:
    payload = kg_summary()
    entity_rows = "".join(
        f"<tr><td>{name}</td><td>{int(count):,}</td></tr>"
        for name, count in sorted(
            (payload.get("entity_type_count") or {}).items(), key=lambda item: item[1], reverse=True
        )
    )
    relation_rows = "".join(
        f"<tr><td>{name}</td><td>{int(count):,}</td></tr>"
        for name, count in sorted(
            (payload.get("relation_type_count") or {}).items(), key=lambda item: item[1], reverse=True
        )
    )
    page = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ChronicCare 知识图谱概览</title><style>
*{{box-sizing:border-box}}body{{margin:0;padding:28px;font-family:Arial,'Microsoft YaHei',sans-serif;background:#f4f7fb;color:#102a43}}
main{{max-width:1320px;margin:auto}}section{{background:#fff;border-radius:16px;padding:24px;margin-bottom:18px;box-shadow:0 10px 28px rgba(16,42,67,.07)}}
.cards{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}}.card{{border:1px solid #dce7f2;border-radius:14px;padding:18px;background:#f7faff}}
.label,.note{{color:#486581}}.value{{margin-top:8px;font-size:28px;font-weight:700}}.split{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:10px 12px;border-bottom:1px solid #e6edf3;text-align:left}}th{{background:#f7faff}}
@media(max-width:900px){{.cards,.split{{grid-template-columns:1fr}}}}</style></head><body><main>
<section><h1>ChronicCare 知识图谱概览</h1><p class="note">本页仅展示知识图谱规模及类型分布，不加载全部节点和边。</p></section>
<section class="cards"><div class="card"><div class="label">节点总数</div><div class="value">{int(payload.get("node_count") or 0):,}</div></div>
<div class="card"><div class="label">边总数</div><div class="value">{int(payload.get("edge_count") or 0):,}</div></div>
<div class="card"><div class="label">实体类型</div><div class="value">{int(payload.get("entity_type_total_count") or 0):,}</div></div>
<div class="card"><div class="label">关系类型</div><div class="value">{int(payload.get("relation_type_total_count") or 0):,}</div></div></section>
<div class="split"><section><h2>实体类型分布</h2><table><thead><tr><th>实体类型</th><th>数量</th></tr></thead><tbody>{entity_rows}</tbody></table></section>
<section><h2>关系类型分布</h2><table><thead><tr><th>关系类型</th><th>数量</th></tr></thead><tbody>{relation_rows}</tbody></table></section></div>
</main></body></html>"""
    return HTMLResponse(page, headers=_no_cache_headers())


@app.api_route("/artifacts/charts", methods=["GET", "HEAD"], summary="Chart Index")
def api_chart_index() -> HTMLResponse:
    return HTMLResponse(chart_gallery_html(), headers=_no_cache_headers())


@app.api_route("/artifacts/charts/{filename:path}", methods=["GET", "HEAD"], summary="Chart Artifact File")
def api_chart_artifact(filename: str):
    if filename == "chart_index.html":
        return HTMLResponse(chart_gallery_html(), headers=_no_cache_headers())
    legacy_subgraph_response = _legacy_subgraph_chart_response(filename)
    if legacy_subgraph_response is not None:
        return legacy_subgraph_response
    legacy_chart = _legacy_followup_chart_path(filename)
    if legacy_chart is not None:
        return FileResponse(legacy_chart, media_type="text/html", headers=_no_cache_headers())
    actual_name = _dynamic_chart_alias(filename) or CHART_ALIAS_MAP.get(filename) or filename
    actual_names = [actual_name]
    if actual_name.endswith(".png"):
        actual_names.append(f"{actual_name[:-4]}.svg")
    for target_name in actual_names:
        candidates = [
            f"outputs/runtime_generated/charts/{target_name}",
            f"outputs/local_runtime/charts/{target_name}",
            f"outputs/charts/{target_name}",
            f"outputs/runtime_generated/graph_driven_analysis/{target_name}",
            f"outputs/local_runtime/graph_driven_analysis/{target_name}",
            f"outputs/graph_driven_analysis/{target_name}",
        ]
        for candidate in candidates:
            path = resolve_path(candidate)
            if path.exists():
                return FileResponse(path, media_type=_guess_media_type(path), headers=_no_cache_headers())
    raise HTTPException(status_code=404, detail="Chart artifact not found")


@app.api_route("/artifacts/report", methods=["GET", "HEAD"], summary="Analysis Report")
def api_report_html() -> HTMLResponse:
    return HTMLResponse(report_overview_html(), headers=_no_cache_headers())


@app.api_route("/artifacts/report.md", methods=["GET", "HEAD"], summary="Analysis Report Markdown")
def api_report_markdown() -> PlainTextResponse:
    return PlainTextResponse(report_overview_markdown(), media_type="text/markdown", headers=_no_cache_headers())


@app.api_route("/artifacts/subgraphs/{subgraph_id}", methods=["GET", "HEAD"], summary="Subgraph HTML")
def api_subgraph_html(subgraph_id: str):
    if subgraph_id.endswith(".png"):
        real_id = _resolve_subgraph_alias_id(subgraph_id[:-4])
        _ensure_subgraph_artifact(real_id)
        for candidate_str in [
            f"outputs/runtime_generated/subgraphs/{_normalize_subgraph_id(real_id)}.png",
            f"outputs/local_runtime/subgraphs/{_normalize_subgraph_id(real_id)}.png",
            f"outputs/subgraphs/{_normalize_subgraph_id(real_id)}.png",
        ]:
            png_path = resolve_path(candidate_str)
            if png_path.exists():
                return FileResponse(png_path, media_type="image/png", headers=_no_cache_headers())
        return Response(content=_subgraph_preview_svg(real_id), media_type="image/svg+xml", headers=_no_cache_headers())
    if subgraph_id.endswith(".svg"):
        real_id = _resolve_subgraph_alias_id(subgraph_id[:-4])
        _ensure_subgraph_artifact(real_id)
        return Response(content=_subgraph_preview_svg(real_id), media_type="image/svg+xml", headers=_no_cache_headers())
    if subgraph_id.endswith(".html"):
        real_id = _resolve_subgraph_alias_id(subgraph_id[:-5])
        return FileResponse(_ensure_subgraph_artifact(real_id), media_type="text/html", headers=_no_cache_headers())
    real_id = _resolve_subgraph_alias_id(subgraph_id)
    return FileResponse(_ensure_subgraph_artifact(real_id), media_type="text/html", headers=_no_cache_headers())


@app.api_route("/artifacts/subgraphs/{subgraph_id}.html", methods=["GET", "HEAD"], summary="Subgraph HTML Alias")
def api_subgraph_html_alias(subgraph_id: str) -> FileResponse:
    real_id = _resolve_subgraph_alias_id(subgraph_id)
    return FileResponse(_ensure_subgraph_artifact(real_id), media_type="text/html", headers=_no_cache_headers())


@app.api_route("/artifacts/subgraphs/{subgraph_id}.svg", methods=["GET", "HEAD"], summary="Subgraph Preview SVG")
def api_subgraph_preview_svg(subgraph_id: str) -> Response:
    real_id = _resolve_subgraph_alias_id(subgraph_id)
    _ensure_subgraph_artifact(real_id)
    return Response(content=_subgraph_preview_svg(real_id), media_type="image/svg+xml", headers=_no_cache_headers())


@app.api_route("/artifacts/subgraphs/{subgraph_id}.png", methods=["GET", "HEAD"], summary="Subgraph Preview PNG")
def api_subgraph_preview_png(subgraph_id: str):
    real_id = _resolve_subgraph_alias_id(subgraph_id)
    _ensure_subgraph_artifact(real_id)
    for candidate_str in [
        f"outputs/runtime_generated/subgraphs/{_normalize_subgraph_id(real_id)}.png",
        f"outputs/local_runtime/subgraphs/{_normalize_subgraph_id(real_id)}.png",
        f"outputs/subgraphs/{_normalize_subgraph_id(real_id)}.png",
    ]:
        png_path = resolve_path(candidate_str)
        if png_path.exists():
            return FileResponse(png_path, media_type="image/png", headers=_no_cache_headers())
    return Response(content=_subgraph_preview_svg(real_id), media_type="image/svg+xml", headers=_no_cache_headers())


@app.api_route("/artifacts/kg_subgraph/{filename:path}", methods=["GET", "HEAD"], summary="Legacy KG Subgraph Artifact")
def api_legacy_kg_subgraph_artifact(filename: str):
    normalized = unquote(str(filename or "").strip())
    if not normalized:
        raise HTTPException(status_code=404, detail="KG subgraph artifact not found")
    topic = normalized
    for suffix in [".html", ".png", ".svg", ".json"]:
        if topic.endswith(suffix):
            topic = topic[: -len(suffix)]
            break
    topic = topic.replace("_preview", "")
    if topic.endswith("_subgraph"):
        topic = topic[: -len("_subgraph")]
    if topic in {"hypertension", "high_blood_pressure"}:
        query = "高血压的知识图谱子图"
    elif topic in {"diabetes"}:
        query = "糖尿病的知识图谱子图"
    elif topic in {"stroke", "stroke_post"}:
        query = "中风的知识图谱子图"
    elif "高血压" in topic:
        query = "高血压的知识图谱子图"
    elif "糖尿病" in topic:
        query = "糖尿病的知识图谱子图"
    elif "高盐" in topic or "血压异常" in topic:
        query = "画出高盐饮食和血压异常之间的关系。"
    else:
        query = f"{topic}的知识图谱子图"
    payload = kg_subgraph_render(query, max_nodes=96)
    if normalized.endswith(".html"):
        return FileResponse(
            resolve_path(str(payload.get("html_path"))), media_type="text/html", headers=_no_cache_headers()
        )
    if normalized.endswith(".json"):
        return FileResponse(
            resolve_path(str(payload.get("json_path"))), media_type="application/json", headers=_no_cache_headers()
        )
    if normalized.endswith(".svg"):
        return FileResponse(
            resolve_path(str(payload.get("preview_path"))), media_type="image/svg+xml", headers=_no_cache_headers()
        )
    png_path = resolve_path(str(payload.get("preview_png_path")))
    if png_path.exists():
        return FileResponse(png_path, media_type="image/png", headers=_no_cache_headers())
    return FileResponse(
        resolve_path(str(payload.get("preview_path"))), media_type="image/svg+xml", headers=_no_cache_headers()
    )


def _request_prefers_image(request: Request) -> bool:
    accept = str(request.headers.get("accept") or "").lower()
    sec_fetch_dest = str(request.headers.get("sec-fetch-dest") or "").lower()
    return sec_fetch_dest == "image" or ("image/" in accept and "text/html" not in accept)


@app.api_route("/artifacts/graph-driven/{analysis_id}", methods=["GET", "HEAD"], summary="Graph-driven Analysis HTML")
def api_graph_driven_html(analysis_id: str, request: Request):
    legacy_subgraph_svg = _legacy_subgraph_preview_svg(analysis_id)
    if legacy_subgraph_svg is not None and (analysis_id.endswith("_chart") or analysis_id.endswith(".svg")):
        return Response(content=legacy_subgraph_svg, media_type="image/svg+xml", headers=_no_cache_headers())
    legacy_subgraph = _legacy_subgraph_html_path(analysis_id)
    if legacy_subgraph is not None:
        if _request_prefers_image(request):
            legacy_subgraph_svg = _legacy_subgraph_preview_svg(analysis_id)
            if legacy_subgraph_svg is not None:
                return Response(content=legacy_subgraph_svg, media_type="image/svg+xml", headers=_no_cache_headers())
        return FileResponse(legacy_subgraph, media_type="text/html", headers=_no_cache_headers())
    legacy_chart = _legacy_followup_chart_path(analysis_id)
    if legacy_chart is not None:
        return FileResponse(legacy_chart, media_type="text/html", headers=_no_cache_headers())
    subgraph_alias = GRAPH_DRIVEN_SUBGRAPH_ALIAS_MAP.get(analysis_id)
    if subgraph_alias:
        if analysis_id.endswith("_interactive"):
            return FileResponse(
                _ensure_subgraph_artifact(subgraph_alias), media_type="text/html", headers=_no_cache_headers()
            )
        return Response(
            content=_subgraph_preview_svg(subgraph_alias), media_type="image/svg+xml", headers=_no_cache_headers()
        )
    alias_id = GRAPH_DRIVEN_ALIAS_MAP.get(analysis_id)
    if alias_id:
        analysis_id = alias_id
    if analysis_id == "analysis_future_followup_chart_bundle":
        latest = _latest_future_followup_bundle_path(chart_page=False)
        if latest is not None:
            return FileResponse(latest, media_type="text/html", headers=_no_cache_headers())
    if analysis_id == "analysis_future_followup_chart_bundle_chart":
        latest = _latest_future_followup_bundle_path(chart_page=True)
        if latest is not None:
            return FileResponse(latest, media_type="text/html", headers=_no_cache_headers())
    for base_str in [
        "outputs/runtime_generated/graph_driven_analysis",
        "outputs/local_runtime/graph_driven_analysis",
        "outputs/graph_driven_analysis",
    ]:
        base_dir = resolve_path(base_str)
        direct_path = base_dir / analysis_id
        if direct_path.exists():
            return FileResponse(direct_path, media_type=_guess_media_type(direct_path), headers=_no_cache_headers())
        for suffix in [".html", ".svg", ".png", ".json", ".csv"]:
            path = base_dir / f"{analysis_id}{suffix}"
            if path.exists():
                return FileResponse(path, media_type=_guess_media_type(path), headers=_no_cache_headers())
    raise HTTPException(status_code=404, detail="Graph-driven artifact not found")


@app.get("/artifacts/open-sql/eval", summary="Open SQL Eval Artifact")
def api_open_eval_artifact() -> HTMLResponse:
    payload = open_sql_eval_tool()
    body = f"<html><body><pre>{payload}</pre></body></html>"
    return HTMLResponse(body, headers=_no_cache_headers())


@app.api_route("/artifacts/open-nl2sql/{filename:path}", methods=["GET", "HEAD"], summary="Open NL2SQL Artifact")
def api_open_nl2sql_artifact(filename: str):
    path = resolve_path(f"outputs/open_nl2sql/{filename}")
    if path.exists() and path.is_file():
        return FileResponse(path, media_type=_guess_media_type(path), headers=_no_cache_headers())
    raise HTTPException(status_code=404, detail="Open NL2SQL artifact not found")


@app.api_route("/artifacts/charts/open_sql/{filename:path}", methods=["GET", "HEAD"], summary="Open SQL Chart Artifact")
def api_open_sql_chart_artifact(filename: str):
    path = resolve_path(f"outputs/charts/open_sql/{filename}")
    if path.exists() and path.is_file():
        return FileResponse(path, media_type=_guess_media_type(path), headers=_no_cache_headers())
    raise HTTPException(status_code=404, detail="Open SQL chart artifact not found")


@app.api_route("/artifacts/{filename:path}", methods=["GET", "HEAD"], summary="Generic Artifact Fallback")
def api_generic_artifact(filename: str):
    if filename in {"report.html", "analysis_report.html"}:
        return HTMLResponse(report_overview_html(), headers=_no_cache_headers())
    if filename in {"report.md", "analysis_report.md"}:
        return PlainTextResponse(report_overview_markdown(), media_type="text/markdown", headers=_no_cache_headers())
    if filename in {"chart_index.html", "charts.html"}:
        return HTMLResponse(chart_gallery_html(), headers=_no_cache_headers())
    legacy_preview_map = {
        "kg_subgraph_hypertension_preview.svg": "subgraph_cohort_subgraph_hypertension",
        "kg_subgraph_high_salt_hypertension_preview.svg": "subgraph_graph_query_a4148cf41a",
    }
    subgraph_alias = legacy_preview_map.get(filename)
    if subgraph_alias:
        _ensure_subgraph_artifact(subgraph_alias)
        return Response(
            content=_subgraph_preview_svg(subgraph_alias), media_type="image/svg+xml", headers=_no_cache_headers()
        )
    legacy_subgraph_response = _legacy_subgraph_chart_response(filename)
    if legacy_subgraph_response is not None:
        return legacy_subgraph_response
    for candidate in [
        f"outputs/runtime_generated/charts/{filename}",
        f"outputs/local_runtime/charts/{filename}",
        f"outputs/charts/{filename}",
        f"outputs/reports/{filename}",
        f"outputs/runtime_generated/graph_driven_analysis/{filename}",
        f"outputs/local_runtime/graph_driven_analysis/{filename}",
        f"outputs/graph_driven_analysis/{filename}",
        f"outputs/open_nl2sql/{filename}",
        f"outputs/charts/open_sql/{filename}",
        f"outputs/open_sql/{filename}",
        f"outputs/evaluation/{filename}",
        f"outputs/runtime_generated/subgraphs/{filename}",
        f"outputs/local_runtime/subgraphs/{filename}",
        f"outputs/subgraphs/{filename}",
        f"data/graph/{filename}",
    ]:
        path = resolve_path(candidate)
        if path.exists() and path.is_file():
            return FileResponse(path, media_type=_guess_media_type(path), headers=_no_cache_headers())
    raise HTTPException(status_code=404, detail="Artifact not found")


@app.post("/datamate/plan", summary="Plan a dynamic DataMate DAG")
def api_datamate_dag_plan(request: DataMateDagPlanRequest) -> dict:
    return datamate_dag_plan(request.goal, request.input_path, request.use_npu)


@app.post("/datamate/run", summary="Run or dry-run a dynamic DataMate DAG")
def api_datamate_dag_run(request: DataMateDagRunRequest) -> dict:
    return datamate_dag_run(
        request.goal, request.input_path, request.use_npu, request.dry_run, request.resume_run_id, request.resume_from
    )


@app.post("/datamate/resume", summary="Resume a dynamic DataMate DAG")
def api_datamate_dag_resume(request: DataMateDagRunRequest) -> dict:
    if not request.resume_run_id:
        raise HTTPException(status_code=400, detail="resume_run_id is required")
    return datamate_dag_run(
        request.goal, request.input_path, request.use_npu, False, request.resume_run_id, request.resume_from
    )


@app.get("/datamate/runs/{run_id}", summary="Get dynamic DAG run status")
def api_datamate_dag_status(run_id: str) -> dict:
    return datamate_dag_status(run_id)


@app.get("/datamate/runs/{run_id}/dag", summary="Get a run DAG")
def api_datamate_dag_graph(run_id: str) -> dict:
    return datamate_dag_graph(run_id)


@app.post("/datamate/runs/{run_id}/cancel", summary="Cancel a dynamic DAG run")
def api_datamate_dag_cancel(run_id: str) -> dict:
    return datamate_dag_cancel(run_id)


@app.post("/datamate/pipeline/run", summary="DataMate Pipeline Run")
def api_datamate_pipeline_run(request: DataMatePipelineRunRequest) -> dict:
    return run_datamate_pipeline(
        request.task_id,
        request.force,
        request.safe_run,
        request.use_npu,
        request.npu_targets,
        request.fallback,
    )


@app.post("/datamate/pipelines/run", summary="DataMate Pipelines Run")
def api_datamate_pipelines_run(request: DataMatePipelineRunRequest) -> dict:
    return run_datamate_pipeline(
        request.task_id,
        request.force,
        request.safe_run,
        request.use_npu,
        request.npu_targets,
        request.fallback,
    )


@app.post("/datamate/pipeline/run-npu", summary="DataMate Pipeline Run With NPU Enhancement")
def api_datamate_pipeline_run_npu(request: DataMatePipelineRunRequest) -> dict:
    return run_npu_enhanced_pipeline(
        task_id=request.task_id,
        use_npu=True,
        npu_targets=request.npu_targets,
        fallback=request.fallback,
        force=request.force,
        safe_run=request.safe_run,
    )


@app.get("/npu/readiness", summary="NPU Runtime Readiness")
def api_npu_readiness() -> dict:
    return npu_readiness()


@app.get("/npu/supported-operators", summary="NPU Supported Operators")
def api_npu_supported_operators() -> dict:
    return npu_supported_operators()


@app.post("/npu/benchmark", summary="NPU Operator Benchmark")
def api_npu_operator_benchmark(request: NPUBenchmarkRequest) -> dict:
    return run_npu_operator_benchmark(use_npu=request.use_npu, fallback=request.fallback)


@app.get("/npu/benchmark/report", summary="Latest NPU Benchmark Report")
def api_npu_benchmark_report() -> dict:
    return npu_benchmark_report()


@app.get("/datamate/pipeline/status", summary="DataMate Pipeline Status")
def api_datamate_pipeline_status() -> dict:
    return datamate_pipeline_status()


@app.get("/datamate/pipeline/report", summary="DataMate Pipeline Report")
def api_datamate_pipeline_report() -> dict:
    return datamate_pipeline_report()


@app.get("/datamate/pipelines", summary="DataMate Pipelines Overview")
def api_datamate_pipelines() -> dict:
    return datamate_pipelines()


@app.get("/datamate/pipelines/latest", summary="DataMate Pipelines Latest")
def api_datamate_pipelines_latest() -> dict:
    return datamate_pipeline_latest()


@app.get("/datamate/pipelines/status/{run_id}", summary="DataMate Pipeline Status By Run")
def api_datamate_pipelines_status(run_id: str) -> dict:
    return datamate_pipeline_status_by_run(run_id)


@app.get("/datamate/pipelines/report/{run_id}", summary="DataMate Pipeline Report By Run")
def api_datamate_pipelines_report(run_id: str) -> dict:
    return datamate_pipeline_report_by_run(run_id)


@app.get("/datamate/pipelines/cli-fallback", summary="DataMate Pipeline CLI Fallback")
def api_datamate_pipelines_cli_fallback() -> dict:
    return datamate_pipeline_run_cli_hint()


@app.post("/kg/query", summary="Legacy KG Query")
def api_kg_query(request: KGQueryRequest) -> dict:
    return kg_query(request.query_type, request.entity_id)


@app.get("/kg/summary", summary="KG Summary")
def api_kg_summary() -> dict:
    return kg_summary()


@app.post("/kg/entity/query", summary="KG Entity Query")
def api_kg_entity_query(request: KGTextQueryRequest) -> dict:
    return kg_entity_query(request.query)


@app.post("/kg/relation/query", summary="KG Relation Query")
def api_kg_relation_query(request: KGTextQueryRequest) -> dict:
    return kg_relation_query(request.query)


@app.post("/kg/patient/path", summary="KG Patient Path Query")
def api_kg_patient_path(request: PatientPathQueryRequest) -> dict:
    return kg_patient_path_query(request.patient_id, request.max_hops)


@app.post("/kg/subgraph/query", summary="KG Subgraph Query")
def api_kg_subgraph_query(request: dict[str, Any] | KGTextQueryRequest) -> dict:
    query, max_nodes = _coerce_subgraph_query(request)
    return kg_subgraph_query(query, max_nodes)


@app.post("/kg/subgraph/render", summary="KG Subgraph Render")
def api_kg_subgraph_render(request: dict[str, Any] | KGTextQueryRequest) -> dict:
    query, max_nodes = _coerce_subgraph_query(request)
    return kg_subgraph_render(query, max_nodes)


@app.post("/analysis/query", summary="Analysis Query")
def api_analysis_query(request: AnalysisQueryRequest) -> dict:
    return analysis_query(request.question)


@app.post("/analysis/open-query", summary="Open Analysis Query")
def api_open_analysis_query(request: OpenAnalysisQueryRequest) -> dict:
    return open_analysis_query(request.question)


@app.post("/analysis/open-sql/query", summary="Open SQL Query")
def api_open_sql_query(request: OpenSQLQueryRequest) -> dict:
    return open_sql_query_tool(
        request.question,
        prefer_llm=request.prefer_llm,
        allow_chart=request.allow_chart,
        force_llm=request.force_llm,
        as_of_date=request.as_of_date,
    )


@app.get("/analysis/open-sql/schema", summary="Open SQL Schema Catalog")
def api_open_sql_schema() -> dict:
    return open_sql_schema_tool()


@app.get("/analysis/open-sql/eval", summary="Open SQL Eval")
def api_open_sql_eval() -> dict:
    return open_sql_eval_tool()


@app.get("/analysis/open-sql/examples", summary="Open SQL Examples")
def api_open_sql_examples() -> dict:
    return open_sql_examples_tool()


@app.get("/analysis/open-sql/traces/recent", summary="Recent Open SQL Traces")
def api_open_sql_recent_traces(limit: int = 10) -> dict:
    return open_sql_recent_traces_tool(limit=limit)


@app.post("/analysis/graph-driven", summary="Graph-driven Analysis")
def api_graph_driven(request: OpenAnalysisQueryRequest) -> dict:
    return graph_driven_analysis(request.question)


@app.get("/reports/summary", summary="Reports Summary")
def api_reports_summary() -> dict:
    return reports_summary()


@app.get("/charts/list", summary="Charts List")
def api_charts_list() -> dict:
    return charts_list()


@app.post("/agent/plan", summary="Agent Plan")
def api_agent_plan(request: AgentGoalRequest) -> dict:
    return agent_plan(request.user_goal)


@app.post("/agent/run", summary="Agent Run")
def api_agent_run(request: AgentGoalRequest) -> dict:
    return agent_run(request.user_goal)
