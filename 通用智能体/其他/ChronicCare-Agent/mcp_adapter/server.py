from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from mcp_adapter.chroniccare_http import ChronicCareClient, ChronicCareHTTPError
from mcp_adapter.schemas import MCPError, MCPRequest, MCPResponse, ToolCallRequest
from mcp_adapter.tool_descriptions import TOOL_DEFINITIONS, get_tool_map
from mcp_adapter.trace_logger import append_trace, load_recent_traces, summarize_traces

DEFAULT_TOOL_SERVER_URL = "http://127.0.0.1:18088"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 18188
DEFAULT_TRANSPORT = "streamable-http"
LONG_RUNNING_TIMEOUT_SECONDS = 900
OPEN_SQL_TIMEOUT_SECONDS = 60


def _looks_like_subgraph_question(question: str) -> bool:
    text = str(question or "").strip().lower()
    if not text:
        return False
    if any(token in text for token in ("画出", "绘制", "画一张", "生成")) and any(
        token in text for token in ("关系", "关联", "路径", "图")
    ):
        return True
    if not any(token in text for token in ("图谱", "子图", "关系图")):
        return False
    return any(token in text for token in ("高血压", "糖尿病", "高风险", "中风险", "低风险", "患者", "群体", "hypertension", "diabetes"))


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def _looks_like_open_sql_question(question: str) -> bool:
    text = str(question or "").strip().lower()
    if not text:
        return False
    if any(token in text for token in ("datamate", "pipeline", "npu", "图谱", "知识图谱", "子图")):
        return False
    metric_tokens = (
        "bmi",
        "体重指数",
        "hba1c",
        "糖化",
        "空腹血糖",
        "血糖",
        "ldl",
        "ldl-c",
        "低密度",
        "血压",
        "异常率",
        "异常比例",
        "达标率",
        "控制",
        "均值",
        "平均",
        "趋势",
        "分布",
        "随访",
    )
    cohort_tokens = (
        "高血压",
        "糖尿病",
        "高脂血症",
        "高血脂",
        "冠心病",
        "慢性肾病",
        "脂肪肝",
        "慢阻肺",
        "哮喘",
        "骨关节炎",
        "痛风",
        "慢性心力衰竭",
        "糖尿病肾病",
        "睡眠呼吸暂停",
        "脑血管病",
        "房颤",
        "慢性肝炎",
        "甲减",
        "三高",
        "共病",
        "降糖药",
        "降压药",
        "高盐",
        "运动不足",
        "风险等级",
        "高风险",
        "中风险",
        "低风险",
    )
    return any(token in text for token in metric_tokens) and any(token in text for token in cohort_tokens)


def _looks_like_disease_distribution_question(question: str) -> bool:
    text = str(question or "").strip().lower().replace(" ", "")
    if not text:
        return False
    if _looks_like_disease_combination_question(text):
        return False
    if any(token in text for token in ("图谱", "子图", "关系图", "datamate", "pipeline", "npu", "随访", "风险等级")):
        return False
    disease_tokens = (
        "高血压",
        "糖尿病",
        "高脂血症",
        "高血脂",
        "冠心病",
        "肥胖",
        "脂肪肝",
        "慢性肾病",
        "高尿酸",
        "慢阻肺",
        "copd",
        "骨质疏松",
        "哮喘",
        "骨关节炎",
        "痛风",
        "慢性心力衰竭",
        "糖尿病肾病",
        "睡眠呼吸暂停",
        "脑血管病",
        "房颤",
        "慢性肝炎",
        "甲减",
        "hypertension",
        "diabetes",
        "hyperlipidemia",
    )
    distribution_tokens = (
        "疾病分布",
        "常见病",
        "有哪些疾病",
        "有哪些病",
        "疾病类型",
        "慢病类型",
        "患者人数分布",
        "有多少",
        "多少人",
        "占比",
    )
    return any(token in text for token in distribution_tokens) and (
        any(token in text for token in disease_tokens) or any(token in text for token in ("疾病", "病", "慢病"))
    )


def _looks_like_disease_combination_question(question: str) -> bool:
    text = str(question or "").strip().lower().replace(" ", "")
    if not text:
        return False
    if any(token in text for token in ("图谱", "子图", "关系图", "datamate", "pipeline", "npu", "随访")):
        return False
    return any(token in text for token in ("疾病组合", "共病组合", "多病组合", "不同疾病组合", "多病共病", "共患组合"))


def _looks_like_datamate_pipeline_run_question(question: str) -> bool:
    text = str(question or "").strip().lower().replace(" ", "")
    if not text:
        return False
    run_tokens = ("重新处理", "重新运行", "重新跑", "重跑", "运行", "执行", "处理")
    pipeline_tokens = ("datamate", "pipeline", "数据处理流程", "算子链路", "算子流程", "原始数据", "清洗摘要")
    return any(token in text for token in run_tokens) and any(token in text for token in pipeline_tokens)


def _looks_like_npu_pipeline_run_question(question: str) -> bool:
    text = str(question or "").strip().lower().replace(" ", "")
    if not text or "npu" not in text:
        return False
    run_tokens = ("重新处理", "重新运行", "重新跑", "重跑", "运行", "执行", "启用", "跑")
    pipeline_tokens = ("datamate", "pipeline", "全流程", "数据处理流程", "算子链路", "原始数据")
    return any(token in text for token in run_tokens) and any(token in text for token in pipeline_tokens)


def _looks_like_contextual_cohort_question(question: str) -> bool:
    text = str(question or "").strip().lower().replace(" ", "")
    if not text:
        return False
    pronouns = ("他们", "这些患者", "该群体", "这个群体", "这群患者", "这部分患者")
    topics = ("疾病类型", "慢病", "疾病", "病种", "风险等级", "风险分布")
    return any(token in text for token in pronouns) and any(token in text for token in topics)


def _looks_like_report_summary_question(question: str) -> bool:
    text = str(question or "").strip().lower().replace(" ", "")
    if not text:
        return False
    return any(token in text for token in ("图表", "报告", "入口", "打开")) and any(
        token in text for token in ("有哪些", "哪里", "入口", "总览", "列表")
    )


def _looks_like_capability_examples_question(question: str) -> bool:
    text = str(question or "").strip().lower().replace(" ", "")
    if not text:
        return False
    return any(token in text for token in ("分析问题", "支持哪些问题", "可以问哪些", "能问哪些", "分析能力"))


def _looks_like_followup_high_risk_question(question: str) -> bool:
    text = str(question or "").strip().lower().replace(" ", "")
    return "未来" in text and "随访" in text


def _looks_like_kg_summary_question(question: str) -> bool:
    text = str(question or "").strip().lower().replace(" ", "")
    if not text:
        return False
    graph_tokens = ("知识图谱", "图谱", "kg", "knowledgegraph")
    summary_tokens = (
        "节点和边",
        "节点边",
        "节点数",
        "边数",
        "多少节点",
        "多少边",
        "节点",
        "边",
        "规模",
        "质量评分",
        "实体类型",
        "关系类型",
    )
    return any(token in text for token in graph_tokens) and any(token in text for token in summary_tokens)


def _coerce_subgraph_query(args: Dict[str, Any]) -> str:
    query = str(args.get("query") or "").strip()
    if query:
        return query
    subgraph_type = str(args.get("subgraph_type") or args.get("subgraphType") or "").strip().lower()
    disease = str(args.get("disease") or args.get("entity") or "").strip()
    if disease:
        if any(token in disease for token in ("图谱", "子图", "关系图")):
            return disease
        if subgraph_type == "risk" and "风险" not in disease:
            return f"{disease}风险知识图谱子图"
        return f"{disease}的知识图谱子图"
    cohort_name = str(args.get("cohort_name") or args.get("cohort") or args.get("group_name") or "").strip()
    if cohort_name:
        if any(token in cohort_name for token in ("图谱", "子图", "关系图")):
            return cohort_name
        if subgraph_type == "risk" and "风险" not in cohort_name:
            return f"给我{cohort_name}风险知识图谱子图"
        return f"给我{cohort_name}知识图谱子图"
    return ""


def _coerce_question(args: Dict[str, Any]) -> str:
    return str(args.get("question") or args.get("query") or "").strip()


def _coerce_followup_question(args: Dict[str, Any]) -> str:
    question = _coerce_question(args)
    if question:
        return question
    raw_days = args.get("days", args.get("day", args.get("time_window", args.get("window_days"))))
    try:
        days = int(float(raw_days))
    except (TypeError, ValueError):
        days = None
    risk_level = str(args.get("risk_level") or args.get("risk") or "").strip() or "高风险"
    if days is not None:
        return f"未来 {days} 天需要随访的{risk_level}患者有多少？"
    return ""


def _coerce_metric_question(args: Dict[str, Any]) -> str:
    question = _coerce_question(args)
    if question:
        return question
    disease = str(args.get("disease") or args.get("disease_name") or args.get("cohort_name") or "").strip()
    metric = str(args.get("metric") or args.get("indicator") or args.get("indicator_name") or "").strip()
    if disease and metric:
        return f"{disease}患者的{metric}是多少？"
    return ""


def _coerce_trend_question(args: Dict[str, Any]) -> str:
    question = _coerce_question(args)
    if question:
        return question
    disease = str(args.get("disease") or args.get("disease_name") or args.get("cohort_name") or "").strip()
    metric = str(args.get("metric") or args.get("indicator") or args.get("indicator_name") or "").strip()
    period = str(args.get("window") or args.get("time_window") or args.get("period") or "").strip()
    if disease and metric:
        prefix = period or "最近一段时间"
        return f"{prefix}{disease}患者的{metric}趋势如何？"
    return ""


def get_settings() -> Dict[str, Any]:
    return {
        "tool_server_url": os.getenv("CHRONICCARE_TOOL_SERVER_URL", DEFAULT_TOOL_SERVER_URL),
        "host": os.getenv("CHRONICCARE_MCP_HOST", DEFAULT_HOST),
        "port": int(os.getenv("CHRONICCARE_MCP_PORT", str(DEFAULT_PORT))),
        "transport": os.getenv("CHRONICCARE_MCP_TRANSPORT", DEFAULT_TRANSPORT),
        "sdk_available": _detect_mcp_sdk(),
    }


def _detect_mcp_sdk() -> bool:
    try:
        from mcp.server.fastmcp import FastMCP  # noqa: F401

        return True
    except Exception:
        return False


def build_tool_payload(definition) -> Dict[str, Any]:
    input_schema = definition.input_schema.model_dump()
    properties = dict(input_schema.get("properties") or {})
    properties["conversation_id"] = {
        "type": "string",
        "description": "由 Nexent 运行时自动注入的真实会话 ID；模型不得编造。",
    }
    input_schema["properties"] = properties
    return {
        "name": definition.name,
        "title": definition.title,
        "description": definition.description,
        "inputSchema": input_schema,
    }


def create_client(base_url: str | None = None, conversation_id: str | None = None) -> ChronicCareClient:
    settings = get_settings()
    return ChronicCareClient(base_url or settings["tool_server_url"], conversation_id=conversation_id)


def summarize_health(payload: Dict[str, Any], base_url: str) -> str:
    stage = payload.get("stage")
    stage_text = f"发布阶段：{stage}；" if stage else ""
    return (
        f"ChronicCare Tool Server 状态：{payload.get('status', 'unknown')}；"
        f"项目：{payload.get('project', 'unknown')}；"
        f"{stage_text}"
        f"消息：{payload.get('message', '')}。"
    )


def summarize_kg(payload: Dict[str, Any]) -> str:
    top_entity_types = payload.get("top_entity_types") or []
    top_relation_types = payload.get("top_relation_types") or []
    entity_text = "，".join([f"{name}:{count}" for name, count in top_entity_types[:5]]) if top_entity_types else "N/A"
    relation_text = "，".join([f"{name}:{count}" for name, count in top_relation_types[:5]]) if top_relation_types else "N/A"
    data_scale = []
    if payload.get("patient_count") is not None:
        data_scale.append(f"患者 {payload.get('patient_count')} 人")
    if payload.get("visit_count") is not None:
        data_scale.append(f"随访记录 {payload.get('visit_count')} 条")
    if payload.get("lab_result_count") is not None:
        data_scale.append(f"检验记录 {payload.get('lab_result_count')} 条")
    if payload.get("medication_record_count") is not None:
        data_scale.append(f"用药记录 {payload.get('medication_record_count')} 条")
    return (
        f"当前系统数据规模：{'；'.join(data_scale) if data_scale else 'N/A'}；"
        f"当前慢病知识图谱共有 {payload.get('node_count', '未知')} 个节点、"
        f"{payload.get('edge_count', '未知')} 条边；"
        f"实体类型总数 {payload.get('entity_type_total_count', '未知')}，关系类型总数 {payload.get('relation_type_total_count', '未知')}；"
        f"主要实体类型分布：{entity_text}；"
        f"主要关系类型分布：{relation_text}；"
        f"图谱概览页：{payload.get('graph_url', payload.get('graph_html_path', 'N/A'))}。"
        " 若用户当前在问某疾病/某群体的知识图谱子图，请不要继续使用本工具，改用 chroniccare_kg_subgraph_render。"
    )




def summarize_data_summary(payload: Dict[str, Any]) -> str:
    rows = []
    table = payload.get("table") if isinstance(payload.get("table"), dict) else {}
    for item in (table.get("rows") or [])[:8]:
        if isinstance(item, dict):
            rows.append(f"{item.get('指标', '指标')}：{item.get('数值', 'N/A')}")
    summary = [
        f"当前数据版本：{payload.get('data_version', 'N/A')}。",
        f"当前系统数据规模：患者 {payload.get('patient_count', 'N/A')} 人，"
        f"随访记录 {payload.get('visit_count', 'N/A')} 条，"
        f"检验记录 {payload.get('lab_result_count', 'N/A')} 条，"
        f"用药记录 {payload.get('medication_record_count', 'N/A')} 条。",
        f"图谱节点 {payload.get('node_count', 'N/A')} 个，边 {payload.get('edge_count', 'N/A')} 条。",
    ]
    if rows:
        summary.append("；".join(rows))
    return "\n".join(summary)


def summarize_datamate(payload: Dict[str, Any]) -> str:
    steps = payload.get("steps", [])
    step_text = "，".join([f"{item.get('operator')}={item.get('status')}" for item in steps[:11]])
    metrics = payload.get("metrics", {})
    table = payload.get("table") if isinstance(payload.get("table"), dict) else {}
    detail_rows = []
    raw_detail_rows = table.get("detail_rows")
    if isinstance(raw_detail_rows, list):
        for row in raw_detail_rows[:11]:
            if isinstance(row, dict):
                detail_rows.append(dict(row))
    if not detail_rows:
        for item in steps[:11]:
            execution_seconds = item.get("execution_seconds")
            try:
                seconds_text = f"{float(execution_seconds):.4f}"
            except (TypeError, ValueError):
                seconds_text = "N/A"
            detail_rows.append(
                {
                    "算子名称": item.get("operator"),
                    "耗时（秒）": seconds_text,
                    "状态": item.get("status"),
                    "是否参考值": "是" if item.get("execution_seconds_is_reference") else "否",
                }
            )
    table_text = None
    if detail_rows:
        headers = list(detail_rows[0].keys())
        table_lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in detail_rows:
            table_lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
        table_text = "\n".join(table_lines)
    lines = [
        f"DataMate full pipeline 状态：{payload.get('status')}；"
        f"算子状态：{step_text}；"
        f"指标：节点 {metrics.get('node_count')}、边 {metrics.get('edge_count')}、问题数 {metrics.get('question_count')}；"
        f"报告：{payload.get('report_path')}。",
    ]
    timing = payload.get("timing") or {}
    lines.append(
        "耗时汇总："
        f"11 个算子纯执行耗时 {timing.get('pure_execution_seconds', 'N/A')} 秒；"
        f"容器内 pipeline 耗时 {timing.get('pipeline_execution_seconds', 'N/A')} 秒；"
        f"外层流程耗时 {timing.get('outer_flow_seconds', 'N/A')} 秒。"
    )
    if table_text:
        lines.append("各算子执行耗时明细：")
        lines.append(table_text)
    metric_definition = payload.get("metric_definition")
    if metric_definition:
        lines.append(f"统计口径：{metric_definition}")
    return "\n".join(lines)


def summarize_datamate_overview(payload: Dict[str, Any]) -> str:
    pipelines = payload.get("pipelines", [])
    latest = payload.get("latest_run", {}) or {}
    pipeline_text = "；".join(
        f"{item.get('pipeline_name')} -> {','.join(item.get('operators', []))}"
        for item in pipelines
    )
    return (
        f"ChronicCare 当前包含 {len(pipelines)} 条 DataMate CPU/通用主线 pipeline、"
        f"{payload.get('operator_count', 0)} 个主线算子；"
        f"最新运行状态：{latest.get('status', 'unknown')}；"
        f"推荐调用方式：{payload.get('invocation_mode', 'unknown')}；"
        f"主线拆分：{pipeline_text}。"
        "注意：这 11 个是 DataMate 主线 CPU/通用算子，不是 NPU 算子；"
        "当前 NPU 增强算子只有 2 个：chronic_entity_extract_model_npu、chronic_relation_extract_model_npu。"
        "如果用户询问系统分析能力或示例问题，应调用 chroniccare_open_sql_examples 获取当前能力边界，不使用固定题目总数。"
    )


def summarize_npu(payload: Dict[str, Any]) -> str:
    def comparison_row(item: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(item)
        cpu_records = row.get("cpu_benchmark_records")
        npu_records = row.get("npu_record_count")
        cpu_sample_seconds = row.get("cpu_bge_sample_seconds") or row.get("cpu_bge_seconds")
        npu_sample_seconds = row.get("npu_bge_sample_seconds") or row.get("npu_same_sample_seconds")
        npu_full_seconds = row.get("npu_bge_full_seconds") or row.get("npu_bge_seconds")
        estimated_cpu_full_seconds = row.get("estimated_cpu_bge_full_seconds") or row.get("estimated_cpu_full_seconds")

        def div(numerator: Any, denominator: Any, digits: int) -> float | None:
            try:
                denominator_f = float(denominator)
                if denominator_f == 0:
                    return None
                return round(float(numerator) / denominator_f, digits)
            except (TypeError, ValueError, ZeroDivisionError):
                return None

        row.setdefault("cpu_sample_throughput_records_per_second", div(cpu_records, cpu_sample_seconds, 2))
        row.setdefault("cpu_avg_latency_ms_per_record", div(float(cpu_sample_seconds) * 1000.0 if cpu_sample_seconds else None, cpu_records, 4))
        row.setdefault("cpu_estimated_full_throughput_records_per_second", div(npu_records, estimated_cpu_full_seconds, 2))
        row.setdefault("npu_sample_throughput_records_per_second", div(cpu_records, npu_sample_seconds, 2))
        row.setdefault("npu_sample_avg_latency_ms_per_record", div(float(npu_sample_seconds) * 1000.0 if npu_sample_seconds else None, cpu_records, 4))
        row.setdefault("npu_full_throughput_records_per_second", div(npu_records, npu_full_seconds, 2))
        row.setdefault("npu_full_avg_latency_ms_per_record", div(float(npu_full_seconds) * 1000.0 if npu_full_seconds else None, npu_records, 4))
        row.setdefault("npu_avg_latency_ms_per_record", row.get("npu_full_avg_latency_ms_per_record"))
        row.setdefault("cpu_resource_metrics_status", "not_collected")
        row.setdefault("resource_metrics_status", "not_collected")
        cpu_util = row.get("cpu_resource_utilization_percent")
        row.setdefault(
            "cpu_effective_cores",
            round(float(cpu_util) / 100.0, 4) if isinstance(cpu_util, (int, float)) else None,
        )
        return row

    runtime = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}
    comparison_scope_note = (
        "口径说明：CPU 与 NPU 正式测量前各预热一次；CPU（2048 条）、NPU（2048 条）和 NPU（全量）分别真实计时；"
        "NPU 抽样和全量均使用 batch 1024，并分别启停独立 npu-smi 采样器；"
        "每组吞吐量、平均延迟、资源、功耗与能耗只使用本组测量值。"
        "加速比只比较同一批 2048 条样本上的 CPU BGE 与 NPU BGE，不能拿 NPU 全量直接比较。"
    )
    operators = payload.get("supported_operators") or []
    operator_names = []
    for item in operators:
        if isinstance(item, dict):
            operator_names.append(str(item.get("operator") or "").strip())
    if payload.get("operator_results"):
        result_rows = []
        comparison_items = payload.get("npu_comparison_rows") or []
        for item in comparison_items:
            item = comparison_row(item)
            result_rows.append(
                f"{item.get('operator')}={item.get('status')}, backend={item.get('backend')}, "
                f"NPU处理={item.get('npu_record_count')}条, CPU抽样={item.get('cpu_benchmark_records')}条, "
                f"CPU规则耗时={item.get('cpu_rule_seconds')}秒, "
                f"CPU_BGE抽样耗时={item.get('cpu_bge_sample_seconds', item.get('cpu_bge_seconds'))}秒, "
                f"NPU_BGE抽样耗时={item.get('npu_bge_sample_seconds', item.get('npu_same_sample_seconds'))}秒, "
                f"抽样加速比={item.get('sample_speedup', item.get('same_sample_speedup'))}, "
                f"CPU_BGE全量估算耗时={item.get('estimated_cpu_bge_full_seconds', item.get('estimated_cpu_full_seconds'))}秒, "
                f"NPU_BGE全量实测耗时={item.get('npu_bge_full_seconds', item.get('npu_bge_seconds'))}秒, "
                f"CPU吞吐量={item.get('cpu_sample_throughput_records_per_second')}条/秒, "
                f"NPU_2048吞吐量={item.get('npu_sample_throughput_records_per_second')}条/秒, "
                f"NPU全量吞吐量={item.get('npu_full_throughput_records_per_second')}条/秒, "
                f"CPU平均单条延迟={item.get('cpu_avg_latency_ms_per_record')}ms, "
                f"NPU_2048平均单条延迟={item.get('npu_sample_avg_latency_ms_per_record')}ms, "
                f"NPU全量平均单条延迟={item.get('npu_full_avg_latency_ms_per_record')}ms, "
                f"CPU资源≈{item.get('cpu_effective_cores')}核等效, "
                f"NPU_2048_AICore={item.get('npu_sample_resource_utilization_percent')}%, "
                f"NPU_2048平均功耗={item.get('npu_sample_average_power_watt')}W, NPU_2048估算能耗={item.get('npu_sample_estimated_energy_wh')}Wh, "
                f"NPU全量_AICore={item.get('resource_utilization_percent')}%, "
                f"NPU全量平均功耗={item.get('average_power_watt')}W, "
                f"NPU全量估算能耗={item.get('estimated_energy_wh')}Wh"
            )
        return (
            f"NPU 算子 benchmark 状态：{payload.get('status')}；"
            f"runtime={runtime.get('backend')}，npu_available={runtime.get('npu_available')}，"
            f"fallback_used={payload.get('fallback_used')}。"
            f"算子结果：{'；'.join(result_rows)}。"
            f"报告：{payload.get('report_path')}。"
            f"{comparison_scope_note}"
            "若 fallback_used=true，本次没有真实 NPU 加速比。"
        )
    if payload.get("npu_benchmark"):
        benchmark = payload.get("npu_benchmark") or {}
        comparison_items = payload.get("npu_comparison_rows") or benchmark.get("npu_comparison_rows") or []
        comparison_text = "；".join(
            f"{item.get('operator')}: NPU处理={item.get('npu_record_count')}条, CPU抽样={item.get('cpu_benchmark_records')}条, "
            f"CPU规则耗时={item.get('cpu_rule_seconds')}秒, "
            f"CPU_BGE抽样耗时={item.get('cpu_bge_sample_seconds', item.get('cpu_bge_seconds'))}秒, "
            f"NPU_BGE抽样耗时={item.get('npu_bge_sample_seconds', item.get('npu_same_sample_seconds'))}秒, "
            f"抽样加速比={item.get('sample_speedup', item.get('same_sample_speedup'))}, "
            f"CPU_BGE全量估算耗时={item.get('estimated_cpu_bge_full_seconds', item.get('estimated_cpu_full_seconds'))}秒, "
            f"NPU_BGE全量实测耗时={item.get('npu_bge_full_seconds', item.get('npu_bge_seconds'))}秒, "
            f"CPU吞吐量={item.get('cpu_sample_throughput_records_per_second')}条/秒, "
            f"NPU_2048吞吐量={item.get('npu_sample_throughput_records_per_second')}条/秒, "
            f"NPU全量吞吐量={item.get('npu_full_throughput_records_per_second')}条/秒, "
            f"CPU平均单条延迟={item.get('cpu_avg_latency_ms_per_record')}ms, "
            f"NPU_2048平均单条延迟={item.get('npu_sample_avg_latency_ms_per_record')}ms, "
            f"NPU全量平均单条延迟={item.get('npu_full_avg_latency_ms_per_record')}ms, "
            f"CPU资源≈{item.get('cpu_effective_cores')}核等效, "
            f"NPU_2048_AICore={item.get('npu_sample_resource_utilization_percent')}%, "
            f"NPU_2048平均功耗={item.get('npu_sample_average_power_watt')}W, NPU_2048估算能耗={item.get('npu_sample_estimated_energy_wh')}Wh, "
            f"NPU全量_AICore={item.get('resource_utilization_percent')}%, "
            f"NPU全量平均功耗={item.get('average_power_watt')}W, "
            f"NPU全量估算能耗={item.get('estimated_energy_wh')}Wh"
            for item in (comparison_row(raw_item) for raw_item in comparison_items if isinstance(raw_item, dict))
        )
        return (
            f"NPU 增强 DataMate pipeline 状态：{payload.get('status')}；"
            f"base_pipeline={payload.get('base_pipeline', {}).get('status')}；"
            f"fallback_used={benchmark.get('fallback_used')}；"
            f"报告：{payload.get('report_path')}。"
            f"{comparison_scope_note}"
            "NPU 增强范围覆盖实体候选 BGE 标准化、关系候选 BGE 重排/过滤；NL2SQL 仍使用 CPU/通用主线算子。"
            f"{'算子对比：' + comparison_text + '。' if comparison_text else ''}"
            "最终回答硬性口径：每个算子用四列表格，列为：指标、CPU（2048条）、NPU（2048条）、NPU（全量）；其中后三列是三个独立实测结果列；"
            "三列必须分别展示各自实测处理量、BGE耗时、吞吐量和平均单条延迟；NPU 两列 batch 均为 1024；"
            "CPU 与 NPU 正式测量前各预热一次且预热不计时；NPU 2048 条与全量各自独立启停采样器，资源、功耗和能耗禁止复用综合值；"
            "同样本加速比只放在 CPU/NPU 2048 条两列，NPU 全量列写不适用；"
            "结论禁止用 ~ 表示范围；必须分别引用本轮 Observation 的实体与关系抽样加速比，不得写固定示例值；"
            "CPU 核等效和 NPU AICore 是不同资源口径，禁止写“显著降低 CPU 等效核数消耗”。"
            "默认表格不要展示 sidecar 路径；用户明确索要文件路径时再引用工具原始字段。"
        )
    if operator_names:
        return (
            f"当前 NPU 增强只支持 {len(operator_names)} 个算子：{', '.join(operator_names)}。"
            "覆盖范围包括实体候选 BGE 标准化、关系候选 BGE 重排/过滤。"
            "DataMate 主线 11 个 CPU/通用算子不是 NPU 算子，不能说成 11 个 NPU 算子。"
        )
    return (
        f"NPU readiness 状态：{payload.get('status')}；"
        f"backend={runtime.get('backend')}；npu_available={runtime.get('npu_available')}；"
        f"fallback_required={runtime.get('fallback_required')}；"
        f"说明：{payload.get('readiness_summary') or runtime.get('message')}；"
        f"报告：{payload.get('report_path')}。"
    )


def compact_npu_pipeline_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    benchmark = payload.get("npu_benchmark") if isinstance(payload.get("npu_benchmark"), dict) else {}
    datamate_npu_pipeline = payload.get("datamate_npu_pipeline") if isinstance(payload.get("datamate_npu_pipeline"), dict) else {}
    return {
        "status": payload.get("status"),
        "timestamp": payload.get("timestamp"),
        "task_id": payload.get("task_id"),
        "duration_seconds": payload.get("duration_seconds"),
        "use_npu": payload.get("use_npu"),
        "fallback_used": benchmark.get("fallback_used"),
        "runtime": benchmark.get("runtime"),
        "npu_comparison_rows": payload.get("npu_comparison_rows") or benchmark.get("npu_comparison_rows") or [],
        "npu_operator_count": len(datamate_npu_pipeline.get("operator_steps") or benchmark.get("operator_results") or []),
        "datamate_npu_timing": datamate_npu_pipeline.get("timing"),
        "report_path": payload.get("report_path"),
        "markdown_report_path": payload.get("markdown_report_path"),
        "benchmark_report_path": benchmark.get("report_path"),
        "output_root": datamate_npu_pipeline.get("output_root") or benchmark.get("output_root"),
        "errors": payload.get("errors") or benchmark.get("errors") or [],
    }


def summarize_kg_detail(payload: Dict[str, Any]) -> str:
    if payload.get("subgraph_id") and (payload.get("html_url") or payload.get("graph_url")):
        graph_url = payload.get("html_url") or payload.get("graph_url")
        graph_service_url = payload.get("service_html_url") or payload.get("graph_service_url")
        preview_url = payload.get("preview_png_url") or payload.get("preview_url")
        preview_service_url = payload.get("preview_png_service_url") or payload.get("preview_service_url")
        graph_route_path = payload.get("html_route_path") or payload.get("graph_route_path")
        preview_route_path = payload.get("preview_route_path")
        graph_link = _format_artifact_access_link("图谱子图入口", graph_url, graph_service_url, graph_route_path)
        preview_link = _format_artifact_access_link(
            "子图预览",
            preview_url,
            preview_service_url,
            preview_route_path,
        )
        lines = [
            f"问题：{payload.get('query', payload.get('patient_id', 'N/A'))}",
            "结果：已实时生成当前问题对应的知识图谱子图。",
            "输出要求：先展示下方 SVG/PNG 预览图，再给出完整 HTML 图谱页面链接；禁止输出 outputs/... 内部路径。当前 HTML 页面用于浏览结构图，不承诺节点拖拽。",
        ]
        if payload.get("graph_scope_explanation"):
            lines.append(f"展示说明：{truncate_text(payload.get('graph_scope_explanation'), 180)}")
        if payload.get("cohort_patient_count") is not None:
            lines.append(f"群体患者数：{payload.get('cohort_patient_count')} 人")
        preview_image_url = preview_url or preview_service_url
        if isinstance(preview_image_url, str) and preview_image_url.lower().endswith((".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp")):
            lines.append("子图预览：")
            lines.append(f"![子图预览]({preview_image_url})")
            lines.append(f"子图预览 URL：{preview_image_url}")
            alternate_preview_url = _alternate_local_artifact_url(preview_image_url)
            if alternate_preview_url:
                lines.append(f"子图预览备用 URL：{alternate_preview_url}")
        elif payload.get("preview_svg_data_uri"):
            lines.append("子图预览：已生成；任务详情中不输出 base64 内联图，避免前端出现长串乱码。")
        if graph_route_path or preview_route_path:
            lines.append("备用路径：")
            if graph_route_path:
                lines.append(f"- HTML 路由：{graph_route_path}")
            if preview_route_path:
                lines.append(f"- 预览路由：{preview_route_path}")
        if graph_link:
            lines.append("完整 HTML 图谱页面：")
            lines.append(graph_link)
        if graph_url:
            lines.append(f"完整 HTML 图谱页面 URL：{graph_url}")
        if preview_link:
            lines.append(preview_link)
        lines.append(f"安全声明：{payload.get('safety_note', '')}")
        return "\n".join(lines)

    links: List[str] = []
    image_blocks: List[str] = []
    graph_url = payload.get("html_url") or payload.get("graph_url")
    graph_service_url = payload.get("service_html_url") or payload.get("graph_service_url")
    preview_url = payload.get("preview_png_url") or payload.get("preview_url")
    report_url = payload.get("report_url")
    report_service_url = payload.get("report_service_url")
    chart_url = payload.get("chart_url")
    chart_service_url = payload.get("chart_service_url")
    for label, public_url, service_url in (
        ("图谱入口", graph_url, graph_service_url),
        ("分析报告", report_url, report_service_url),
        ("图表入口", chart_url, chart_service_url),
    ):
        public_text = str(public_url or "").strip()
        if public_text:
            links.append(f"{label}：[打开入口]({public_text})")
    lines = [
        f"问题：{payload.get('query', payload.get('patient_id', 'N/A'))}",
        f"结果状态：{payload.get('status')}",
        f"摘要：{payload.get('text', payload.get('insight', payload.get('explanation', '已返回结构化结果')))}",
    ]
    if payload.get("answer_guardrail"):
        lines.append(f"口径要求：{payload.get('answer_guardrail')}")
    table_rendered = False
    strict_rows_only = False
    table_preview = payload.get("table") or {}
    if isinstance(table_preview, dict):
        rows = table_preview.get("rows") or table_preview.get("detail_rows") or []
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            headers = list(rows[0].keys())[:6]
            head = "| " + " | ".join(headers) + " |"
            split = "| " + " | ".join(["---"] * len(headers)) + " |"
            strict_rows_only = bool(table_preview.get("strict_rows_only"))
            row_limit = len(rows) if strict_rows_only else (len(rows) if len(rows) <= 12 else 12)
            body = ["| " + " | ".join(str(item.get(header, ""))[:40] for header in headers) + " |" for item in rows[:row_limit]]
            lines.append("必须使用以下工具真实表格作为最终答案，禁止使用上一轮上下文或模型常识改写数字：")
            lines.append(f"最终答案锁定：本表共有 {len(rows)} 行；只允许复述这些行，禁止补充表格外的指标、药物或风险事件。")
            allowed_names = table_preview.get("allowed_names")
            if isinstance(allowed_names, list) and allowed_names:
                lines.append("允许出现的名称：" + "、".join(str(item) for item in allowed_names if str(item).strip()))
            lines.extend([head, split, *body])
            if strict_rows_only:
                lines.append("最终回答模板（必须原样复述，不得增删改任何行、名称或数字）：")
                lines.extend([head, split, *body])
            table_rendered = True
    if payload.get("final_answer_lock"):
        lines.append(f"最终答案锁定：{payload.get('final_answer_lock')}")
    if strict_rows_only and table_rendered:
        if payload.get("cohort_patient_count") is not None:
            lines.append(f"群体患者数：{payload.get('cohort_patient_count')} 人")
        lines.append(f"安全声明：{payload.get('safety_note', '')}")
        return "\n".join(lines)
    if payload.get("graph_scope_explanation"):
        lines.append(f"展示说明：{payload.get('graph_scope_explanation')}")
    if payload.get("node_count") is not None or payload.get("edge_count") is not None:
        lines.append(
            f"子图规模：节点 {payload.get('node_count', 'N/A')} 个，关系 {payload.get('edge_count', 'N/A')} 条"
        )
    seed_labels = payload.get("seed_labels") or []
    if isinstance(seed_labels, list) and seed_labels:
        lines.append(f"核心群体：{'、'.join(str(item) for item in seed_labels if str(item).strip())}")
    if payload.get("cohort_patient_count") is not None:
        lines.append(f"群体患者数：{payload.get('cohort_patient_count')} 人")
    if payload.get("display_patient_node_count") is not None:
        lines.append(f"当前展示示例患者节点：{payload.get('display_patient_node_count')} 个")
    if payload.get("semantic_node_count") is not None:
        lines.append(f"完整语义节点数：{payload.get('semantic_node_count')} 个")
    top_indicators = payload.get("top_indicators") or []
    if isinstance(top_indicators, list) and top_indicators:
        lines.append(
            "核心指标："
            + "、".join(str(item.get("indicator") or item.get("display_name") or "") for item in top_indicators[:8] if str(item.get("indicator") or item.get("display_name") or "").strip())
        )
    associated_indicators = payload.get("associated_indicators") or payload.get("shared_indicators") or []
    if isinstance(associated_indicators, list) and associated_indicators:
        lines.append(
            "关联检查指标："
            + "、".join(str(item.get("indicator") or item.get("target_label") or "") for item in associated_indicators[:10] if str(item.get("indicator") or item.get("target_label") or "").strip())
        )
    top_risk_events = payload.get("top_risk_events") or []
    if isinstance(top_risk_events, list) and top_risk_events:
        lines.append(
            "常见风险事件："
            + "、".join(str(item.get("event_type") or item.get("display_name") or "") for item in top_risk_events[:8] if str(item.get("event_type") or item.get("display_name") or "").strip())
        )
    associated_risk_events = payload.get("associated_risk_events") or payload.get("shared_risk_events") or []
    if isinstance(associated_risk_events, list) and associated_risk_events:
        lines.append(
            "关联风险事件："
            + "、".join(str(item.get("event_type") or item.get("target_label") or "") for item in associated_risk_events[:10] if str(item.get("event_type") or item.get("target_label") or "").strip())
        )
    top_drugs = payload.get("top_drugs") or []
    if isinstance(top_drugs, list) and top_drugs:
        lines.append(
            "常见药物："
            + "、".join(str(item.get("drug_name") or item.get("display_name") or "") for item in top_drugs[:8] if str(item.get("drug_name") or item.get("display_name") or "").strip())
        )
    associated_drugs = payload.get("associated_drugs") or []
    if isinstance(associated_drugs, list) and associated_drugs:
        lines.append(
            "关联药物："
            + "、".join(str(item.get("drug_name") or item.get("target_label") or "") for item in associated_drugs[:10] if str(item.get("drug_name") or item.get("target_label") or "").strip())
        )
    if isinstance(table_preview, dict) and not table_rendered:
        rows = table_preview.get("rows") or table_preview.get("detail_rows") or []
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            headers = list(rows[0].keys())[:6]
            head = "| " + " | ".join(headers) + " |"
            split = "| " + " | ".join(["---"] * len(headers)) + " |"
            body = ["| " + " | ".join(str(item.get(header, ""))[:40] for header in headers) + " |" for item in rows[:6]]
            lines.append("结果预览：")
            lines.extend([head, split, *body])
    if links:
        lines.append("仅以下入口为工具真实返回的当前可用产物：")
        lines.extend(links)
    if isinstance(preview_url, str) and preview_url.strip():
        image_blocks.append(f"![图谱子图预览]({preview_url})")
    elif isinstance(payload.get("preview_service_url"), str) and payload.get("preview_service_url").strip():
        image_blocks.append(f"![图谱子图预览]({payload.get('preview_service_url')})")
    if image_blocks:
        lines.append("图像预览：")
        lines.extend(image_blocks[:1])
    lines.append(f"安全声明：{payload.get('safety_note', '')}")
    return "\n".join(lines)


def _format_access_link(label: str, public_url: Any, service_url: Any = None) -> str | None:
    public_text = str(public_url or "").strip()
    if public_text:
        return f"{label}：[打开入口]({public_text})"
    return None


def _format_artifact_access_link(label: str, public_url: Any, service_url: Any = None, route_path: Any = None) -> str | None:
    public_text = str(public_url or "").strip()
    if public_text:
        alternate_text = _alternate_local_artifact_url(public_text)
        if alternate_text and alternate_text != public_text:
            return f"{label}：[打开入口]({public_text}) / [备用入口]({alternate_text})"
        return f"{label}：[打开入口]({public_text})"
    return None


def _format_dual_access_link(label: str, public_url: Any, service_url: Any = None) -> str | None:
    public_text = str(public_url or "").strip()
    if public_text:
        return f"{label}：[打开入口]({public_text})"
    return None


def _alternate_local_artifact_url(url: Any) -> str | None:
    text = str(url or "").strip()
    if "127.0.0.1:28088" in text:
        return text.replace("127.0.0.1:28088", "127.0.0.1:18089")
    if "localhost:28088" in text:
        return text.replace("localhost:28088", "127.0.0.1:18089")
    if "127.0.0.1:18089" in text:
        return text.replace("127.0.0.1:18089", "127.0.0.1:28088")
    if "localhost:18089" in text:
        return text.replace("localhost:18089", "127.0.0.1:28088")
    return None


def summarize_analysis(payload: Dict[str, Any]) -> str:
    def markdown_table(rows: Any, limit: int = 20) -> str | None:
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            return None
        headers = list(rows[0].keys())[:6]
        head = "| " + " | ".join(headers) + " |"
        split = "| " + " | ".join(["---"] * len(headers)) + " |"
        body = [
            "| " + " | ".join(str(item.get(header, ""))[:40] for header in headers) + " |"
            for item in rows[:limit]
        ]
        return "\n".join([head, split, *body])

    def disease_combo_table(rows: Any, limit: int = 20) -> str | None:
        if not isinstance(rows, list) or not rows:
            return None
        headers = ["疾病组合", "患者人数"]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for item in rows[:limit]:
            if isinstance(item, dict):
                lines.append(f"| {item.get('疾病组合', '')} | {item.get('患者人数', '')} |")
        return "\n".join(lines)

    links = []
    image_blocks = []
    for candidate in [
        _format_artifact_access_link("图表总览", payload.get("chart_url"), payload.get("chart_service_url"), payload.get("chart_route_path")),
        _format_artifact_access_link("分析报告", payload.get("report_url"), payload.get("report_service_url"), payload.get("report_route_path")),
        _format_artifact_access_link("图谱入口", payload.get("graph_url"), payload.get("graph_service_url"), payload.get("graph_route_path")),
    ]:
        if candidate:
            links.append(candidate)
    for item in payload.get("charts", [])[:4]:
        label = item.get("name") or item.get("type") or "图表"
        route_url = item.get("png_route_path") or item.get("route_path")
        url = item.get("png_url") or item.get("url") or item.get("png_alias_url")
        service_url = item.get("png_service_url") or item.get("service_url")
        link_text = _format_artifact_access_link(label, url, service_url, route_url)
        if link_text:
            links.append(link_text)
        image_url = (
            item.get("png_url")
            or item.get("png_alias_url")
            or item.get("png_service_url")
            or item.get("png_route_path")
            or url
            or route_url
        )
        if isinstance(image_url, str) and image_url.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
            image_blocks.append(f"![{label}]({image_url})")
    lines = [
        f"分析问题：{payload.get('question', 'N/A')}",
        f"命中问题 ID：{payload.get('matched_id', 'N/A')}",
        f"分析意图：{payload.get('intent', 'N/A')}",
        f"图表类型：{payload.get('chart_type', 'N/A')}",
        f"指标：{truncate_text(payload.get('metric', 'N/A'), 220)}",
        f"结果表：{truncate_text(payload.get('table'), 260)}",
        f"解释：{payload.get('insight', 'N/A')}",
    ]
    if payload.get("planner"):
        lines.append(f"路由规划：{truncate_text(payload.get('planner'), 240)}")
    matched_id = str(payload.get("matched_id") or "")
    table_payload = payload.get("table") if isinstance(payload.get("table"), dict) else {}
    intent = str(payload.get("intent") or payload.get("canonical_id") or "")
    if intent == "future_n_days_high_risk_followup":
        metric = payload.get("metric") if isinstance(payload.get("metric"), dict) else {}
        count_value = metric.get("value")
        unit = metric.get("unit") or "人"
        window_days = payload.get("window_days") or payload.get("exact_window_days") or ""
        window = payload.get("window") if isinstance(payload.get("window"), dict) else {}
        concise_lines = [
            f"最终答案锁定：未来 {window_days} 天需要随访的高风险患者数 = {count_value} {unit}。",
            f"统计窗口：{window.get('start_date', '')} 至 {window.get('end_date', '')}。",
            "最终回答必须使用上面这个总数，禁止把 7 天、30 天或历史会话里的结果带入本次问题。",
        ]
        metric_table = markdown_table(table_payload.get("rows"), limit=10)
        if metric_table:
            concise_lines.append("随访队列统计表：")
            concise_lines.append(metric_table)
        trend_rows = table_payload.get("trend_rows") if isinstance(table_payload.get("trend_rows"), list) else []
        if trend_rows:
            trend_total = sum(int(item.get("patient_count", 0) or 0) for item in trend_rows if isinstance(item, dict))
            concise_lines.append(f"逐日明细校验：本次返回 {len(trend_rows)} 天，逐日合计 {trend_total} {unit}；必须与总数一致。")
            trend_table = markdown_table(trend_rows, limit=140)
            if trend_table:
                concise_lines.append("每日随访趋势明细（最终回答必须复用这些日期和人数，不要补 0）：")
                concise_lines.append(trend_table)
        if links:
            concise_lines.append("当前真实入口：")
            concise_lines.extend(links)
        if image_blocks:
            concise_lines.append("图像预览：")
            concise_lines.extend(image_blocks[:2])
        concise_lines.append(f"安全声明：{payload.get('safety_note', '')}")
        return "\n".join(concise_lines)
    if matched_id in {"kg_disease_inventory", "disease_distribution"}:
        labels = [str(item).strip() for item in (payload.get("disease_labels") or []) if str(item).strip()]
        if labels:
            lines.append(f"完整疾病类型列表（{len(labels)} 种）：{'、'.join(labels)}")
        disease_table = markdown_table(table_payload.get("detail_rows") or table_payload.get("rows"), limit=32)
        if disease_table:
            lines.append("完整疾病类型统计：")
            lines.append(disease_table)
        metric = payload.get("metric") if isinstance(payload.get("metric"), dict) else {}
        rows = table_payload.get("detail_rows") or table_payload.get("rows") or []
        if metric.get("name") == "matched_disease_patient_count" and rows and isinstance(rows[0], dict):
            first = rows[0]
            lines.append(
                f"最终答案锁定：{first.get('疾病名称')}患者数 = {first.get('患者人数')} 人；"
                f"占比 = {first.get('占比')}；总患者基数 = {payload.get('patient_count')} 人。"
                "禁止改写为知识图谱患者数、节点数或 DataMate 汇总患者数。"
            )
        if payload.get("final_answer_lock"):
            lines.append(f"最终答案锁定：{payload.get('final_answer_lock')}")
    if matched_id == "disease_combination_distribution" or intent == "disease_combination_distribution":
        combo_rows = table_payload.get("detail_rows") or table_payload.get("rows") or payload.get("rows")
        combo_table = disease_combo_table(combo_rows, limit=20)
        if combo_table:
            lines.append("精确多病组合 Top 明细（仅包含疾病标签数 >= 2 的患者，单病患者已排除；禁止用省略号替代患者人数）：")
            lines.append(combo_table)
            lines.append(
                "口径锁定：此表是 patient_profile.disease_tags 的精确多病组合 Top 预览，不是全量清单，"
                "表格行人数不要求加总为多病共病总人数；禁止混入单病患者，禁止改写为单病分布。"
            )
        metric = payload.get("metric") if isinstance(payload.get("metric"), dict) else {}
        if metric.get("value") is not None:
            lines.append(
                f"多病共病患者数锁定：{metric.get('value')} {metric.get('unit') or '人'}；"
                "最终回答必须以工具表格为准，不要只展示前三行。"
            )
    if links:
        lines.append("仅以下入口为工具真实返回的当前可用产物，请不要扩写额外图表链接：")
        lines.append(f"可访问链接：{'；'.join(links)}")
    if image_blocks:
        lines.append("图像预览：")
        lines.extend(image_blocks[:2])
    lines.append(f"安全声明：{payload.get('safety_note', '')}")
    return "\n".join(lines)


def summarize_open_sql_schema(payload: Dict[str, Any]) -> str:
    tables = payload.get("tables") or {}
    rows = []
    if isinstance(tables, dict):
        for table_name, table_info in tables.items():
            columns = (table_info.get("columns") or table_info.get("fields") or []) if isinstance(table_info, dict) else []
            column_names = []
            if isinstance(columns, list):
                for column in columns:
                    column_names.append(str(column.get("name") if isinstance(column, dict) else column))
            rows.append({"表名": table_name, "字段": ", ".join([item for item in column_names if item])})
    lines = [
        f"Open SQL Schema 状态：{payload.get('status', 'success')}",
        f"白名单表数量：{len(rows)}；白名单 Join 数量：{len(payload.get('joins') or [])}",
    ]
    if rows:
        headers = ["表名", "字段"]
        lines.extend(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
                *["| " + " | ".join(str(row.get(header, "")) for header in headers) + " |" for row in rows],
            ]
        )
    lines.append(f"安全声明：{payload.get('safety_note', '')}")
    return "\n".join(lines)


def summarize_open_sql_eval(payload: Dict[str, Any]) -> str:
    fields = [
        ("评估状态", payload.get("status")),
        ("问题数", payload.get("total_questions")),
        ("意图准确率", payload.get("intent_accuracy")),
        ("Schema Link 成功率", payload.get("schema_link_success_rate")),
        ("SQL 生成成功率", payload.get("sql_generation_success_rate")),
        ("SQL Guard 通过率", payload.get("sql_guard_pass_rate")),
        ("SQL 可执行率", payload.get("sql_executable_rate")),
        ("结果成功率", payload.get("result_success_rate")),
        ("模板阶段成功率", payload.get("template_stage_success_rate")),
        ("LLM 状态", payload.get("llm_status")),
        ("不支持问题数", payload.get("unsupported_count")),
    ]
    lines = [
        "最近一次 Open SQL 评估结果如下：",
        "| 评估维度 | 结果 |",
        "| --- | --- |",
        *[f"| {label} | {value} |" for label, value in fields],
    ]
    lines.append(f"评估报告：{payload.get('report_path') or 'outputs/evaluation/open_sql_eval_report.json'}")
    lines.append(f"安全声明：{payload.get('safety_note', '')}")
    return "\n".join(lines)


def summarize_open_sql_examples(payload: Dict[str, Any]) -> str:
    examples = payload.get("examples") or []
    supported = payload.get("supported_intents") or []
    lines = [
        f"Open SQL 示例问题数量：{payload.get('example_count', len(examples))}",
        f"支持意图：{', '.join(str(item) for item in supported)}",
        f"LLM 状态：{payload.get('llm_status')}",
        "示例问题：",
    ]
    for index, question in enumerate(examples[:20], start=1):
        lines.append(f"{index}. {question}")
    lines.append(f"安全声明：{payload.get('safety_note', '')}")
    return "\n".join(lines)


def summarize_open_analysis(payload: Dict[str, Any]) -> str:
    if payload.get("answer_markdown"):
        rows = []
        table = payload.get("table") if isinstance(payload.get("table"), dict) else {}
        table_rows = table.get("rows") if isinstance(table, dict) else None
        result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        result_rows = result.get("rows") if isinstance(result, dict) else None
        if isinstance(table_rows, list) and table_rows:
            rows = table_rows
        elif isinstance(result_rows, list) and result_rows:
            rows = result_rows

        locked_rows_text = ""
        if rows and isinstance(rows[0], dict):
            first = rows[0]
            cells = [f"{key}={value}" for key, value in first.items()]
            if isinstance(first.get("abnormal_rate"), (int, float)):
                cells.append(f"abnormal_rate_percent={float(first['abnormal_rate']) * 100:.2f}%")
            if isinstance(first.get("control_rate"), (int, float)):
                cells.append(f"control_rate_percent={float(first['control_rate']) * 100:.2f}%")
            locked_rows_text = "首行锁定结果：" + "；".join(cells)

        schema_link = payload.get("schema_link") if isinstance(payload.get("schema_link"), dict) else {}
        source_tables = ", ".join(str(item) for item in schema_link.get("tables", []) if str(item).strip()) or "N/A"
        lines = [
            str(payload.get("answer_markdown")),
            "",
            locked_rows_text,
            "最终答案锁定：必须逐字使用上方表格和首行锁定结果；禁止引用历史答案、禁止重新计算、禁止把上一轮 Observation 的患者数/分母/分子复用到本轮。若需要百分比，只能用本轮 rate 字段乘以 100。",
            f"生成 SQL：{payload.get('sql') or 'N/A'}",
            f"来源表：{source_tables}",
            f"Open SQL 阶段：{payload.get('stage', 'N/A')}；SQL Guard：{'通过' if payload.get('sql_safe') else '未通过'}；trace_id：{payload.get('trace_id', 'N/A')}",
        ]
        lines = [line for line in lines if str(line).strip()]
        image_url = payload.get("image_url")
        image_service_url = payload.get("image_service_url")
        if isinstance(image_url, str) and image_url.lower().endswith((".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp")):
            lines.append(f"![Open SQL 图表]({image_url})")
        if image_url or image_service_url:
            image_link = _format_access_link("图表图片", image_url, image_service_url)
            if image_link:
                lines.append(image_link)
        trend_rows = payload.get("trend_rows")
        if isinstance(trend_rows, list) and trend_rows:
            if isinstance(trend_rows[0], dict):
                headers = list(trend_rows[0].keys())[:6]
                trend_table = "\n".join(
                    [
                        "| " + " | ".join(headers) + " |",
                        "| " + " | ".join(["---"] * len(headers)) + " |",
                        *[
                            "| " + " | ".join(str(item.get(header, ""))[:40] for header in headers) + " |"
                            for item in trend_rows[:20]
                        ],
                    ]
                )
                lines.append("每日趋势明细：")
                lines.append(trend_table)
        if payload.get("chart_url"):
            lines.append(f"图表入口：{payload.get('chart_url')}")
        lines.append(f"安全声明：{payload.get('safety_note', '')}")
        return "\n".join(lines)

    def markdown_table(rows: Any, limit: int = 6) -> str | None:
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            return None
        headers = list(rows[0].keys())[:6]
        head = "| " + " | ".join(headers) + " |"
        split = "| " + " | ".join(["---"] * len(headers)) + " |"
        body = [
            "| " + " | ".join(str(item.get(header, ""))[:40] for header in headers) + " |"
            for item in rows[:limit]
        ]
        return "\n".join([head, split, *body])

    def disease_combo_table(rows: Any, limit: int = 20) -> str | None:
        if not isinstance(rows, list) or not rows:
            return None
        headers = ["疾病组合", "患者人数"]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for item in rows[:limit]:
            if isinstance(item, dict):
                lines.append(f"| {item.get('疾病组合', '')} | {item.get('患者人数', '')} |")
        return "\n".join(lines)

    def risk_level_label(value: Any) -> str:
        mapping = {"high": "高风险", "medium": "中风险", "low": "低风险"}
        text = str(value or "").strip().lower()
        return mapping.get(text, str(value or "").strip() or "未标注")

    links = []
    image_blocks = []
    if payload.get("canonical_id") == "dynamic_subgraph_render":
        seed_labels = [str(item).strip() for item in (payload.get("seed_labels") or []) if str(item).strip()]
        graph_url = payload.get("graph_url")
        graph_service_url = payload.get("graph_service_url")
        preview_url = payload.get("preview_png_url") or payload.get("preview_url")
        preview_service_url = payload.get("preview_png_service_url") or payload.get("preview_service_url")
        graph_route_path = payload.get("html_route_path") or payload.get("graph_route_path")
        preview_route_path = payload.get("preview_route_path")
        lines = [
            f"原始问题：{payload.get('original_question', payload.get('question', 'N/A'))}",
            f"重写问题：{payload.get('rewritten_question', payload.get('question', 'N/A'))}",
            "路由结果：dynamic_subgraph_render",
            f"结果摘要：{truncate_text(payload.get('summary_text') or payload.get('summary') or '已实时生成图谱子图。', 180)}",
            "输出要求：先展示下方 SVG/PNG 预览图，再给出完整 HTML 图谱页面链接；禁止输出 outputs/... 内部路径。当前 HTML 页面用于浏览结构图，不承诺节点拖拽。",
        ]
        if seed_labels:
            lines.append(f"图谱主题：{'、'.join(seed_labels)}")
        if payload.get("graph_scope"):
            lines.append(f"展示说明：{truncate_text(payload.get('graph_scope'), 180)}")
        if payload.get("cohort_patient_count") is not None:
            lines.append(f"群体患者数：{payload.get('cohort_patient_count')} 人")
        preview_image_url = preview_url or preview_route_path or preview_service_url
        if isinstance(preview_image_url, str) and preview_image_url.lower().endswith((".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp")):
            lines.append("子图预览：")
            lines.append(f"![子图预览]({preview_image_url})")
            lines.append(f"子图预览 URL：{preview_image_url}")
            alternate_preview_url = _alternate_local_artifact_url(preview_image_url)
            if alternate_preview_url:
                lines.append(f"子图预览备用 URL：{alternate_preview_url}")
        elif payload.get("preview_svg_data_uri"):
            lines.append("子图预览：已生成；任务详情中不输出 base64 内联图，避免前端出现长串乱码。")
        graph_link = _format_artifact_access_link("图谱子图入口", graph_url, graph_service_url, graph_route_path)
        preview_link = _format_artifact_access_link("子图预览", preview_url, preview_service_url, preview_route_path)
        if graph_link:
            lines.append("完整 HTML 图谱页面：")
            lines.append(graph_link)
        if graph_url:
            lines.append(f"完整 HTML 图谱页面 URL：{graph_url}")
        if preview_link:
            lines.append(preview_link)
        if graph_route_path or preview_route_path:
            lines.append("同源备用入口：")
            if preview_route_path:
                lines.append(f"- 子图预览：{preview_route_path}")
            if graph_route_path:
                lines.append(f"- 完整 HTML 图谱页面：{graph_route_path}")
        lines.append(f"安全声明：{payload.get('safety_note', '')}")
        return "\n".join(lines)

    for candidate in [
        _format_artifact_access_link("分析报告", payload.get("report_url"), payload.get("report_service_url"), payload.get("report_route_path")),
        _format_artifact_access_link("图表总览", payload.get("chart_url"), payload.get("chart_service_url"), payload.get("chart_route_path")),
        _format_artifact_access_link("图谱入口", payload.get("graph_url"), payload.get("graph_service_url"), payload.get("graph_route_path")),
        _format_access_link("患者明细表", payload.get("cohort_table_url"), payload.get("cohort_table_service_url")),
        _format_access_link("结果表入口", payload.get("result_table_url")),
    ]:
        if candidate:
            links.append(candidate)
    for item in payload.get("charts", [])[:4]:
        label = item.get("name") or item.get("type") or "图表"
        route_url = item.get("png_route_path") or item.get("route_path")
        url = item.get("png_url") or item.get("url") or item.get("png_alias_url")
        service_url = item.get("png_service_url") or item.get("service_url")
        link_text = _format_artifact_access_link(label, url, service_url, route_url)
        if link_text:
            links.append(link_text)
        image_url = (
            item.get("png_url")
            or item.get("png_alias_url")
            or item.get("png_service_url")
            or item.get("png_route_path")
            or url
            or route_url
        )
        if isinstance(image_url, str) and image_url.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
            image_blocks.append(f"![{label}]({image_url})")
    preview_url = payload.get("preview_url")
    preview_service_url = payload.get("preview_service_url")
    if preview_url or preview_service_url:
        preview_label = payload.get("preview_label") or "图谱子图预览"
        preview_link = _format_access_link(preview_label, preview_url, preview_service_url)
        if preview_link:
            links.append(preview_link)
        preview_image_url = preview_url or preview_service_url
        if isinstance(preview_image_url, str) and preview_image_url.lower().endswith((".png", ".svg", ".jpg", ".jpeg", ".gif", ".webp")):
            image_blocks.append(f"![{preview_label}]({preview_image_url})")
    link_text = "；".join(links) if links else "N/A"
    table_payload = payload.get("table") if isinstance(payload.get("table"), dict) else {}
    preview_table = (
        markdown_table(table_payload.get("detail_rows"), limit=20)
        or markdown_table(table_payload.get("rows"), limit=20)
        or markdown_table(table_payload.get("trend_rows"), limit=20)
        or markdown_table(table_payload.get("risk_distribution_rows"), limit=20)
    )
    lines = [
        f"原始问题：{payload.get('original_question', payload.get('question', 'N/A'))}",
        f"重写问题：{payload.get('rewritten_question', payload.get('question', 'N/A'))}",
        f"路由结果：{payload.get('canonical_id', 'N/A')}",
        f"结果摘要：{truncate_text(payload.get('summary_text') or payload.get('summary') or payload.get('insight') or payload.get('metric') or payload.get('table'), 260)}",
        "仅以下入口为工具真实返回的当前可用产物，请不要扩写额外图表链接：",
        f"可访问链接：{link_text}",
    ]
    if payload.get("canonical_id") == "kg_disease_inventory" and isinstance(payload.get("disease_labels"), list):
        labels = [str(item).strip() for item in payload.get("disease_labels", []) if str(item).strip()]
        if labels:
            lines.append(f"完整疾病类型列表（{len(labels)} 种）：{'、'.join(labels)}")
        disease_rows = table_payload.get("rows") or table_payload.get("detail_rows") or []
        disease_table = markdown_table(disease_rows, limit=32)
        if disease_table:
            lines.append("完整疾病类型统计：")
            lines.append(disease_table)
    if (payload.get("canonical_id") or payload.get("matched_id")) == "disease_combination_distribution":
        combo_rows = table_payload.get("detail_rows") or table_payload.get("rows") or payload.get("rows")
        combo_table = disease_combo_table(combo_rows, limit=20)
        if combo_table:
            lines.append("疾病组合明细（必须完整展示这些真实行，禁止用省略号替代患者人数）：")
            lines.append(combo_table)
            lines.append(
                "口径锁定：疾病组合必须使用上述 patient_profile.disease_tags 的详细标签组合；"
                "禁止改写为“高血压/糖尿病/高脂血症”单病或二病简化组合，禁止自行重算。"
            )
        pairwise_rows = table_payload.get("pairwise_rows") or payload.get("pairwise_rows")
        pairwise_table = markdown_table(pairwise_rows, limit=20)
        if pairwise_table:
            lines.append("两两疾病共现表（回答“为什么没有两种病组合”时必须展示此表）：")
            lines.append(pairwise_table)
            lines.append(
                "口径说明：精确多病组合表展示患者完整标签集合的 Top 组合；两两共现表展示任意两种疾病同时出现的患者数。"
                "如果用户问“两种病组合”，优先展示两两共现表；如果用户问“不同疾病组合”，同时说明两个口径。"
            )
        combo_length_rows = table_payload.get("combo_length_rows") or payload.get("combo_length_rows")
        combo_length_table = markdown_table(combo_length_rows, limit=20)
        if combo_length_table:
            lines.append("按疾病标签数量统计：")
            lines.append(combo_length_table)
        metric = payload.get("metric") if isinstance(payload.get("metric"), dict) else {}
        if metric.get("value") is not None:
            lines.append(
                f"多病共病患者数锁定：{metric.get('value')} {metric.get('unit') or '人'}；"
                "这是所有疾病标签数 >= 2 的患者总数，不是精确多病组合 Top 表的行数加总。"
            )
    if payload.get("canonical_id") == "future_n_days_high_risk_followup":
        metric = payload.get("metric") if isinstance(payload.get("metric"), dict) else {}
        count_value = metric.get("value")
        unit = metric.get("unit") or "人"
        window_label = ""
        metric_label = str(metric.get("label") or "")
        label_match = re.search(r"未来\s*(\d+)\s*天", metric_label)
        if label_match:
            window_label = f"未来 {label_match.group(1)} 天"
        if count_value is not None:
            future_pool_patient_count = payload.get("future_pool_patient_count")
            pool_note = (
                f"；从当前日期起所有未来高风险待随访池为 {future_pool_patient_count} {unit}，这是补充口径，不可替代 N 天窗口结果"
                if future_pool_patient_count is not None and future_pool_patient_count != count_value
                else ""
            )
            lines.append(
                f"最终答案锁定：{window_label or '该时间窗口'}需要随访的高风险患者是 {count_value} {unit}。"
                "禁止改写成全体患者数；全体患者数不是本随访队列人数；"
                f"最终回答中的统计表“患者人数”也必须填写 {count_value} {unit}{pool_note}。"
            )
        if payload.get("final_answer_lock"):
            lines.append(f"最终答案锁定：{payload.get('final_answer_lock')}")
        row_table = markdown_table(table_payload.get("rows"), limit=12)
        if row_table:
            lines.append("随访队列统计表：")
            lines.append(row_table)
        related_chart_bundle = payload.get("related_chart_bundle") if isinstance(payload.get("related_chart_bundle"), dict) else {}
        related_table = related_chart_bundle.get("table") if isinstance(related_chart_bundle.get("table"), dict) else {}
        trend_rows = table_payload.get("trend_rows") if isinstance(table_payload.get("trend_rows"), list) else []
        if not trend_rows:
            trend_rows = related_table.get("trend_rows") if isinstance(related_table.get("trend_rows"), list) else []
        trend_table = markdown_table(trend_rows, limit=140)
        trend_total = sum(int(item.get("patient_count", 0) or 0) for item in trend_rows if isinstance(item, dict))
        if trend_rows:
            nonzero_days = [
                f"{item.get('followup_date')}={int(item.get('patient_count', 0) or 0)}人"
                for item in trend_rows
                if isinstance(item, dict) and int(item.get("patient_count", 0) or 0) > 0
            ]
            lines.append(
                f"图表校验口径：折线图逐日值来自原始 followup_plan；"
                f"本轮后端真实去重窗口总人数 metric.value={count_value} {unit}。最终答案、统计表、合计行必须全部写 {count_value} {unit}。"
            )
            if nonzero_days:
                lines.append(
                    "最终回答必须写出非零日期明细："
                    + "；".join(nonzero_days)
                    + "。禁止把非首日日期改成 0。"
                )
            if count_value is not None and trend_total != int(count_value):
                lines.append(
                    f"一致性警告：metric.value={count_value}，每日累计={trend_total}。"
                    "最终答案优先使用 metric.value，并保留每日表供人工复核。"
                )
        if trend_table:
            lines.append(
                "每日随访趋势明细（必须与折线图一致；最终回答若展示每日表，只能复用此表，禁止自行改写成每天 1 人）："
            )
            lines.append(trend_table)
    risk_rows = table_payload.get("risk_distribution_rows")
    if payload.get("canonical_id") == "future_followup_chart_bundle" and isinstance(risk_rows, list) and risk_rows:
        normalized_rows = []
        total = 0
        for item in risk_rows:
            patient_count = int(item.get("patient_count", 0) or 0)
            total += patient_count
            normalized_rows.append(
                {
                    "风险等级": item.get("risk_level_label") or risk_level_label(item.get("risk_level")),
                    "患者人数": patient_count,
                    "占比": item.get("ratio_text") or "",
                }
            )
        lines.append(
            "风险分布明细："
            + "；".join(
                f"{row['风险等级']} {row['患者人数']} 人{('（' + row['占比'] + '）') if row['占比'] else ''}"
                for row in normalized_rows
            )
            + f"；合计 {total} 人。"
        )
        risk_table = markdown_table(normalized_rows, limit=12)
        if risk_table:
            lines.append("风险等级统计表：")
            lines.append(risk_table)
    if payload.get("canonical_id") == "dynamic_subgraph_render":
        seed_labels = [str(item).strip() for item in payload.get("seed_labels", []) if str(item).strip()]
        if seed_labels:
            lines.append(f"图谱主题：{'、'.join(seed_labels)}")
        if payload.get("cohort_patient_count") is not None:
            lines.append(f"群体患者数：{payload.get('cohort_patient_count')} 人")
        if payload.get("display_patient_node_count") is not None:
            lines.append(f"当前展示示例患者节点：{payload.get('display_patient_node_count')} 个")
        if payload.get("semantic_node_count") is not None:
            lines.append(f"完整语义节点数：{payload.get('semantic_node_count')} 个")
    if payload.get("metric_definition"):
        lines.append(f"统计口径：{truncate_text(payload.get('metric_definition'), 220)}")
    if payload.get("planner"):
        lines.append(f"路由规划：{truncate_text(payload.get('planner'), 240)}")
    if image_blocks:
        lines.append("图像预览：")
        lines.extend(image_blocks[:4])
    if preview_table:
        lines.append("表格预览：")
        lines.append(preview_table)
    lines.append(f"安全声明：{payload.get('safety_note', '')}")
    return "\n".join(lines)


def summarize_agent(payload: Dict[str, Any]) -> str:
    return (
        f"用户目标：{payload.get('user_goal', 'N/A')}\n"
        f"计划：{jsonish(payload.get('plan'))}\n"
        f"轨迹：{jsonish(payload.get('tool_results'))}\n"
        f"最终回答：{payload.get('final_answer', 'N/A')}\n"
        f"使用产物：{jsonish(payload.get('artifacts_used'))}\n"
        f"安全声明：{payload.get('safety_note', '')}"
    )


def summarize_report(report_payload: Dict[str, Any], charts_payload: Dict[str, Any]) -> str:
    charts = charts_payload.get("charts", [])
    latest = report_payload.get("latest_graph_driven_analysis") or {}
    image_blocks = []
    links = [
        _format_dual_access_link("分析报告 HTML", report_payload.get("report_url", report_payload.get("analysis_report_html")), report_payload.get("report_service_url")),
        _format_access_link("分析报告 Markdown", report_payload.get("analysis_report_md")),
        _format_dual_access_link("图表画廊", report_payload.get("chart_index_url", report_payload.get("chart_index")), report_payload.get("chart_index_service_url")),
        _format_access_link("最新分析图谱入口", report_payload.get("graph_url", report_payload.get("graph_html")), report_payload.get("graph_service_url")),
        _format_dual_access_link("全局图谱概览", report_payload.get("global_graph_url", report_payload.get("graph_html")), report_payload.get("global_graph_service_url")),
    ]
    links = [item for item in links if item]
    for candidate in [
        _format_access_link("最新图谱驱动分析", latest.get("report_url"), latest.get("report_service_url")),
        _format_access_link("最新分析专属子图", latest.get("graph_url"), latest.get("graph_service_url")),
        _format_access_link("最新分析全量患者列表", latest.get("cohort_table_url")),
    ]:
        if candidate:
            links.append(candidate)
    if charts:
        for item in charts[:3]:
            label = item.get("title", item.get("name", "图表"))
            link_text = _format_access_link(label, item.get("url"), item.get("service_url"))
            if link_text:
                links.append(link_text)
            preview_url = item.get("url")
            if isinstance(preview_url, str) and preview_url.lower().endswith((".png", ".svg", ".jpg", ".jpeg", ".gif", ".webp")):
                image_blocks.append(f"![{label}]({preview_url})")
    lines = [
        "当前已生成可公开访问的图表与报告入口如下：",
        f"摘要：{report_payload.get('summary_text', '已整理稳定入口。')}",
        "注意：如果用户当前在问某疾病/某群体的知识图谱子图，这里不是正确工具，请改用 chroniccare_kg_subgraph_render，不要继续展示全局入口。",
        *[f"- {line}" for line in links],
        f"- 当前公开图表数量：{len(charts)}",
    ]
    if image_blocks:
        lines.append("图像预览：")
        lines.extend(image_blocks[:2])
    return "\n".join(lines)


def jsonish(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    return JSONResponse(content=value).body.decode("utf-8")


def truncate_text(value: Any, limit: int = 160) -> str:
    text = jsonish(value)
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def build_trace_id() -> str:
    return f"trace_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def summarize_trace_input(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {key: truncate_text(value, 240) for key, value in arguments.items()}


def summarize_trace_output(tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name == "chroniccare_health_check":
        return {
            "status": payload.get("status"),
            "project": payload.get("project"),
            "stage": payload.get("stage"),
        }
    if tool_name == "chroniccare_kg_summary":
        return {
            "node_count": payload.get("node_count"),
            "edge_count": payload.get("edge_count"),
        }
    if tool_name in {"chroniccare_datamate_pipeline_run", "chroniccare_datamate_pipeline_status", "chroniccare_datamate_pipeline_run_npu"}:
        metrics = payload.get("metrics", {})
        datamate_npu_pipeline = payload.get("datamate_npu_pipeline") if isinstance(payload.get("datamate_npu_pipeline"), dict) else {}
        operator_steps = payload.get("steps") or datamate_npu_pipeline.get("operator_steps") or []
        return {
            "status": payload.get("status"),
            "step_count": len(operator_steps),
            "node_count": metrics.get("node_count"),
            "edge_count": metrics.get("edge_count"),
            "fallback_used": (payload.get("npu_benchmark") or {}).get("fallback_used") if isinstance(payload.get("npu_benchmark"), dict) else None,
        }
    if tool_name in {"chroniccare_npu_readiness", "chroniccare_npu_supported_operators", "chroniccare_npu_operator_benchmark"}:
        runtime = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}
        return {
            "status": payload.get("status"),
            "backend": runtime.get("backend"),
            "npu_available": runtime.get("npu_available"),
            "fallback_used": payload.get("fallback_used"),
            "report_path": payload.get("report_path"),
        }
    if tool_name == "chroniccare_datamate_pipelines":
        return {
            "status": payload.get("status"),
            "pipeline_count": len(payload.get("pipelines", [])),
            "operator_count": payload.get("operator_count"),
            "latest_run": payload.get("latest_run"),
        }
    if tool_name == "chroniccare_datamate_pipeline_latest":
        return {
            "status": payload.get("status"),
            "run_id": payload.get("run_id"),
            "pipeline_name": payload.get("pipeline_name"),
            "summary": payload.get("summary"),
        }
    if tool_name == "chroniccare_datamate_pipeline_report":
        return {
            "status": payload.get("status"),
            "report_path": payload.get("report_path"),
            "check_report_path": payload.get("check_report_path"),
        }
    if tool_name in {
        "chroniccare_kg_entity_query",
        "chroniccare_kg_relation_query",
        "chroniccare_kg_patient_path_query",
        "chroniccare_kg_subgraph_query",
        "chroniccare_kg_subgraph_render",
    }:
        return {
            "status": payload.get("status"),
            "node_count": payload.get("node_count"),
            "edge_count": payload.get("edge_count"),
            "subgraph_id": payload.get("subgraph_id"),
        }
    if tool_name == "chroniccare_data_summary":
        return {
            "patient_count": payload.get("patient_count"),
            "visit_count": payload.get("visit_count"),
            "lab_result_count": payload.get("lab_result_count"),
            "medication_record_count": payload.get("medication_record_count"),
            "node_count": payload.get("node_count"),
            "edge_count": payload.get("edge_count"),
        }
    if tool_name == "chroniccare_analysis_query":
        metric = payload.get("metric")
        value = metric.get("value") if isinstance(metric, dict) else None
        unit = metric.get("unit") if isinstance(metric, dict) else None
        metric_name = metric.get("name") if isinstance(metric, dict) else metric
        if value is None:
            table = payload.get("table") or {}
            rows = table.get("rows") if isinstance(table, dict) else None
            first_row = rows[0] if isinstance(rows, list) and rows else {}
            if isinstance(first_row, dict):
                for key, row_value in first_row.items():
                    lowered = str(key).lower()
                    if value is None and any(token in lowered for token in ["value", "result", "avg", "mean", "rate", "ratio", "hba1c", "ldl", "bmi"]):
                        value = row_value
        return {
            "question": payload.get("question"),
            "metric_name": metric_name,
            "value": truncate_text(value, 80) if value is not None else None,
            "unit": unit,
        }
    if tool_name in {
        "chroniccare_disease_distribution",
        "chroniccare_disease_combination_distribution",
        "chroniccare_risk_level_distribution",
        "chroniccare_followup_high_risk",
        "chroniccare_cohort_disease_distribution",
        "chroniccare_metric_query",
        "chroniccare_trend_query",
    }:
        return {
            "question": payload.get("question") or payload.get("original_question"),
            "canonical_id": payload.get("canonical_id") or payload.get("matched_id"),
            "summary_text": truncate_text(
                payload.get("summary_text") or payload.get("insight") or payload.get("metric") or payload.get("table"),
                120,
            ),
        }
    if tool_name == "chroniccare_agent_run":
        return {
            "run_id": payload.get("run_id"),
            "tool_call_count": payload.get("tool_call_count"),
            "artifacts_used": payload.get("artifacts_used", [])[:6],
            "trace_path": payload.get("trace_path"),
        }
    if tool_name == "chroniccare_report_summary":
        return {
            "analysis_report_html": payload.get("analysis_report_html"),
            "analysis_report_md": payload.get("analysis_report_md"),
            "chart_index": payload.get("chart_index"),
            "graph_html": payload.get("graph_html"),
        }
    if tool_name in {
        "chroniccare_graph_driven_analysis",
        "chroniccare_open_analysis_query",
        "chroniccare_open_sql_query",
        "chroniccare_open_sql_schema",
        "chroniccare_open_sql_eval",
        "chroniccare_open_sql_examples",
    }:
        return {
            "status": payload.get("status"),
            "analysis_id": payload.get("analysis_id"),
            "variant_count": payload.get("variant_count"),
            "total_questions": payload.get("total_questions"),
            "sql_executable_rate": payload.get("sql_executable_rate"),
            "result_success_rate": payload.get("result_success_rate"),
            "stage": payload.get("stage"),
            "intent": payload.get("intent"),
            "trace_id": payload.get("trace_id"),
        }
    if tool_name == "chroniccare_trace_summary":
        return {
            "total_calls": payload.get("total_calls"),
            "success_rate": payload.get("success_rate"),
            "tool_counts": payload.get("tool_counts"),
        }
    return {"status": payload.get("status")}


def record_tool_trace(
    *,
    trace_id: str,
    tool_name: str,
    input_payload: Dict[str, Any],
    target_endpoint: str,
    tool_server_url: str,
    status: str,
    latency_ms: float,
    output_summary: Dict[str, Any],
    error: str | None,
) -> None:
    append_trace(
        {
            "trace_id": trace_id,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source": "nexent_mcp",
            "tool_name": tool_name,
            "input": summarize_trace_input(input_payload),
            "target_endpoint": target_endpoint,
            "tool_server_url": tool_server_url,
            "status": status,
            "latency_ms": round(latency_ms, 2),
            "output_summary": output_summary,
            "error": truncate_text(error, 240) if error else None,
        }
    )


def execute_http_tool(
    *,
    client: ChronicCareClient,
    tool_name: str,
    endpoint_label: str,
    input_payload: Dict[str, Any],
    invoke,
    text_builder,
    data_builder=None,
    request_timeout: int | None = None,
) -> Dict[str, Any]:
    start = time.perf_counter()
    trace_id = build_trace_id()
    record_tool_trace(
        trace_id=trace_id,
        tool_name=tool_name,
        input_payload=input_payload,
        target_endpoint=endpoint_label,
        tool_server_url=client.base_url,
        status="running",
        latency_ms=0.0,
        output_summary={"message": "tool call started"},
        error=None,
    )
    try:
        payload = invoke(request_timeout)
        latency_ms = (time.perf_counter() - start) * 1000
        data = data_builder(payload) if data_builder else payload
        record_tool_trace(
            trace_id=trace_id,
            tool_name=tool_name,
            input_payload=input_payload,
            target_endpoint=endpoint_label,
            tool_server_url=client.base_url,
            status="success",
            latency_ms=latency_ms,
            output_summary=summarize_trace_output(tool_name, data if isinstance(data, dict) else payload),
            error=None,
        )
        return {"tool": tool_name, "text": text_builder(payload), "data": data}
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        record_tool_trace(
            trace_id=trace_id,
            tool_name=tool_name,
            input_payload=input_payload,
            target_endpoint=endpoint_label,
            tool_server_url=client.base_url,
            status="error",
            latency_ms=latency_ms,
            output_summary={},
            error=str(exc),
        )
        raise


def _execute_report_summary(client: ChronicCareClient, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    def invoke_report() -> Dict[str, Any]:
        report_payload = client.get("/reports/summary")
        charts_payload = client.get("/charts/list")
        return {"report": report_payload, "charts": charts_payload}

    def build_data(payload: Dict[str, Any]) -> Dict[str, Any]:
        report_payload = payload["report"]
        charts_payload = payload["charts"]
        return {
            "status": report_payload.get("status"),
            "analysis_report_html": report_payload.get("analysis_report_html"),
            "analysis_report_url": report_payload.get("analysis_report_url"),
            "chart_count": charts_payload.get("chart_count"),
            "chart_index_url": charts_payload.get("chart_index_url"),
            "charts": charts_payload.get("charts"),
            "report": report_payload,
            "charts_payload": charts_payload,
            "rerouted_from": tool_name if tool_name != "chroniccare_report_summary" else None,
            "rerouted_to": "chroniccare_report_summary",
        }

    return execute_http_tool(
        client=client,
        tool_name=tool_name,
        endpoint_label="GET /reports/summary + GET /charts/list",
        input_payload=args,
        invoke=lambda timeout=None: invoke_report(),
        data_builder=build_data,
        text_builder=lambda payload: summarize_report(payload["report"], payload["charts"]),
    )


def _execute_disease_combination_distribution(client: ChronicCareClient, tool_name: str, question: str) -> Dict[str, Any]:
    request_payload = {"question": question}
    return execute_http_tool(
        client=client,
        tool_name=tool_name,
        endpoint_label="POST /analysis/disease-combination-distribution (rerouted)",
        input_payload=request_payload,
        invoke=lambda timeout=None: client.post("/analysis/disease-combination-distribution", request_payload, timeout=timeout),
        text_builder=lambda payload: (
            "检测到当前问题是疾病组合/共病组合分布，已自动改用 chroniccare_disease_combination_distribution 返回真实精确组合。\n"
            + summarize_analysis(payload)
        ),
        data_builder=lambda payload: {
            **payload,
            "rerouted_from": tool_name,
            "rerouted_to": "chroniccare_disease_combination_distribution",
        },
    )


def _execute_datamate_pipeline_run(client: ChronicCareClient, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    request_payload = {
        "task_id": args.get("task_id", "mcp_datamate_rerouted_run"),
        "force": bool(args.get("force", True)),
        "safe_run": bool(args.get("safe_run", False)),
    }
    return execute_http_tool(
        client=client,
        tool_name=tool_name,
        endpoint_label="POST /datamate/pipeline/run (rerouted)",
        input_payload=request_payload,
        invoke=lambda timeout=None: client.post("/datamate/pipeline/run", request_payload, timeout=timeout),
        text_builder=lambda payload: (
            "检测到当前问题明确要求重新执行 DataMate 数据处理流程，已自动改用 chroniccare_datamate_pipeline_run 真实运行算子。\n"
            + summarize_datamate(payload)
        ),
        data_builder=lambda payload: {
            **payload,
            "rerouted_from": tool_name,
            "rerouted_to": "chroniccare_datamate_pipeline_run",
        },
        request_timeout=LONG_RUNNING_TIMEOUT_SECONDS,
    )


def _execute_datamate_pipeline_run_npu(client: ChronicCareClient, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    request_payload = {
        "task_id": args.get("task_id", "mcp_datamate_npu_rerouted_run"),
        "force": bool(args.get("force", True)),
        "safe_run": bool(args.get("safe_run", True)),
        "use_npu": True,
        "npu_targets": args.get("npu_targets") or [
            "chronic_entity_extract_model_npu",
            "chronic_relation_extract_model_npu",
        ],
        "fallback": bool(args.get("fallback", True)),
    }
    return execute_http_tool(
        client=client,
        tool_name=tool_name,
        endpoint_label="POST /datamate/pipeline/run-npu (rerouted)",
        input_payload=request_payload,
        invoke=lambda timeout=None: client.post("/datamate/pipeline/run-npu", request_payload, timeout=timeout),
        text_builder=lambda payload: (
            (
                "已读取最近一次真实完成的 NPU 全量报告；本轮未重新计算。如需再次耗时运行，请明确说强制重新计算。\n"
                if payload.get("skipped")
                else "已启用 NPU 真实执行 DataMate 全流程。\n"
            )
            + summarize_npu(payload)
        ),
        data_builder=lambda payload: {
            **compact_npu_pipeline_payload(payload),
            "rerouted_from": tool_name,
            "rerouted_to": "chroniccare_datamate_pipeline_run_npu",
        },
        request_timeout=LONG_RUNNING_TIMEOUT_SECONDS,
    )


def _execute_kg_summary(client: ChronicCareClient, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return execute_http_tool(
        client=client,
        tool_name=tool_name,
        endpoint_label="GET /kg/summary (rerouted)",
        input_payload=args,
        invoke=lambda timeout=None: client.get("/kg/summary", timeout=timeout),
        text_builder=lambda payload: (
            "检测到当前问题是在询问知识图谱全局规模，已自动改用 chroniccare_kg_summary 返回节点、边和类型分布。\n"
            + summarize_kg(payload)
        ),
        data_builder=lambda payload: {
            **payload,
            "rerouted_from": tool_name,
            "rerouted_to": "chroniccare_kg_summary",
        },
    )


def _execute_capability_examples(client: ChronicCareClient, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return execute_http_tool(
        client=client,
        tool_name=tool_name,
        endpoint_label="GET /analysis/open-sql/examples (rerouted)",
        input_payload=args,
        invoke=lambda timeout=None: client.get("/analysis/open-sql/examples", timeout=timeout),
        text_builder=lambda payload: (
            "检测到当前问题是在询问系统分析能力，已自动改用 chroniccare_open_sql_examples 返回能力边界和示例。\n"
            + summarize_open_sql_examples(payload)
        ),
        data_builder=lambda payload: {
            **payload,
            "rerouted_from": tool_name,
            "rerouted_to": "chroniccare_open_sql_examples",
        },
    )


def execute_tool(name: str, arguments: Dict[str, Any] | None = None, base_url: str | None = None) -> Dict[str, Any]:
    args = arguments or {}
    client = create_client(base_url, conversation_id=args.get("conversation_id"))
    question_for_reroute = _coerce_question(args) or _coerce_subgraph_query(args)
    if name != "chroniccare_open_sql_examples" and _looks_like_capability_examples_question(question_for_reroute):
        return _execute_capability_examples(client, name, args)
    if name != "chroniccare_kg_summary" and _looks_like_kg_summary_question(question_for_reroute):
        return _execute_kg_summary(client, name, args)
    if name == "chroniccare_datamate_dag_plan":
        payload = {"goal": args.get("goal", "full"), "input_path": args.get("input_path"), "use_npu": bool(args.get("use_npu", False))}
        return execute_http_tool(client=client, tool_name=name, endpoint_label="POST /datamate/plan", input_payload=payload, invoke=lambda timeout=None: client.post("/datamate/plan", payload, timeout=timeout), text_builder=lambda x: json.dumps(x, ensure_ascii=False))
    if name == "chroniccare_datamate_dag_run":
        payload = {"goal": args.get("goal", "full"), "input_path": args.get("input_path"), "use_npu": bool(args.get("use_npu", False)), "dry_run": bool(args.get("dry_run", False))}
        return execute_http_tool(client=client, tool_name=name, endpoint_label="POST /datamate/run", input_payload=payload, invoke=lambda timeout=None: client.post("/datamate/run", payload, timeout=timeout), text_builder=lambda x: json.dumps(x, ensure_ascii=False))
    if name == "chroniccare_datamate_dag_resume":
        payload = {"goal": args.get("goal", "full"), "input_path": args.get("input_path"), "use_npu": bool(args.get("use_npu", False)), "dry_run": False, "resume_run_id": args.get("resume_run_id"), "resume_from": args.get("resume_from")}
        return execute_http_tool(client=client, tool_name=name, endpoint_label="POST /datamate/resume", input_payload=payload, invoke=lambda timeout=None: client.post("/datamate/resume", payload, timeout=timeout), text_builder=lambda x: json.dumps(x, ensure_ascii=False))
    if name == "chroniccare_datamate_dag_status":
        run_id = str(args.get("run_id") or ""); suffix = "/dag" if args.get("include_dag") else ""
        return execute_http_tool(client=client, tool_name=name, endpoint_label=f"GET /datamate/runs/{run_id}{suffix}", input_payload=args, invoke=lambda timeout=None: client.get(f"/datamate/runs/{run_id}{suffix}", timeout=timeout), text_builder=lambda x: json.dumps(x, ensure_ascii=False))
    if name == "chroniccare_health_check":
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="GET /health",
            input_payload=args,
            invoke=lambda timeout=None: client.get("/health", timeout=timeout),
            text_builder=lambda payload: summarize_health(payload, client.base_url),
        )
    if name == "chroniccare_datamate_pipeline_run":
        request_payload = {
            "task_id": args.get("task_id", "mcp_datamate_run_001"),
            "force": bool(args.get("force", True)),
            "safe_run": bool(args.get("safe_run", False)),
        }
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="POST /datamate/pipeline/run",
            input_payload=request_payload,
            invoke=lambda timeout=None: client.post("/datamate/pipeline/run", request_payload, timeout=timeout),
            text_builder=summarize_datamate,
            request_timeout=LONG_RUNNING_TIMEOUT_SECONDS,
        )
    if name == "chroniccare_npu_readiness":
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="GET /npu/readiness",
            input_payload=args,
            invoke=lambda timeout=None: client.get("/npu/readiness", timeout=timeout),
            text_builder=summarize_npu,
        )
    if name == "chroniccare_npu_supported_operators":
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="GET /npu/supported-operators",
            input_payload=args,
            invoke=lambda timeout=None: client.get("/npu/supported-operators", timeout=timeout),
            text_builder=summarize_npu,
        )
    if name == "chroniccare_npu_operator_benchmark":
        question = _coerce_question(args)
        if _looks_like_npu_pipeline_run_question(question):
            return _execute_datamate_pipeline_run_npu(client, name, args)
        force_run = bool(args.get("force_run", args.get("rerun", False)))
        request_payload = {
            "use_npu": bool(args.get("use_npu", True)),
            "fallback": bool(args.get("fallback", True)),
        }
        if not force_run:
            return execute_http_tool(
                client=client,
                tool_name=name,
                endpoint_label="GET /npu/benchmark/report",
                input_payload={**request_payload, "force_run": False},
                invoke=lambda timeout=None: client.get("/npu/benchmark/report", timeout=timeout),
                text_builder=summarize_npu,
            )
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="POST /npu/benchmark",
            input_payload={**request_payload, "force_run": True},
            invoke=lambda timeout=None: client.post("/npu/benchmark", request_payload, timeout=timeout),
            text_builder=summarize_npu,
            request_timeout=LONG_RUNNING_TIMEOUT_SECONDS,
        )
    if name == "chroniccare_datamate_pipeline_run_npu":
        request_payload = {
            "task_id": args.get("task_id", "mcp_datamate_npu_run_001"),
            "force": bool(args.get("force", True)),
            "safe_run": bool(args.get("safe_run", True)),
            "use_npu": True,
            "npu_targets": args.get("npu_targets") or [
                "chronic_entity_extract_model_npu",
                "chronic_relation_extract_model_npu",
            ],
            "fallback": bool(args.get("fallback", True)),
        }
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="POST /datamate/pipeline/run-npu",
            input_payload=request_payload,
            invoke=lambda timeout=None: client.post("/datamate/pipeline/run-npu", request_payload, timeout=timeout),
            text_builder=summarize_npu,
            data_builder=compact_npu_pipeline_payload,
            request_timeout=LONG_RUNNING_TIMEOUT_SECONDS,
        )
    if name == "chroniccare_datamate_pipelines":
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="GET /datamate/pipelines",
            input_payload=args,
            invoke=lambda timeout=None: client.get("/datamate/pipelines", timeout=timeout),
            text_builder=summarize_datamate_overview,
        )
    if name == "chroniccare_datamate_pipeline_status":
        question = _coerce_question(args)
        if _looks_like_npu_pipeline_run_question(question):
            return _execute_datamate_pipeline_run_npu(client, name, args)
        if _looks_like_datamate_pipeline_run_question(question):
            return _execute_datamate_pipeline_run(client, name, args)
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="GET /datamate/pipeline/status",
            input_payload=args,
            invoke=lambda timeout=None: client.get("/datamate/pipeline/status", timeout=timeout),
            text_builder=summarize_datamate,
        )
    if name == "chroniccare_datamate_pipeline_latest":
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="GET /datamate/pipelines/latest",
            input_payload=args,
            invoke=lambda timeout=None: client.get("/datamate/pipelines/latest", timeout=timeout),
            text_builder=summarize_datamate,
        )
    if name == "chroniccare_datamate_pipeline_report":
        question = _coerce_question(args)
        if _looks_like_npu_pipeline_run_question(question):
            return _execute_datamate_pipeline_run_npu(client, name, args)
        if _looks_like_datamate_pipeline_run_question(question):
            return _execute_datamate_pipeline_run(client, name, args)
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="GET /datamate/pipeline/report",
            input_payload=args,
            invoke=lambda timeout=None: client.get("/datamate/pipeline/report", timeout=timeout),
            text_builder=summarize_datamate,
        )
    if name == "chroniccare_kg_summary":
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="GET /kg/summary",
            input_payload=args,
            invoke=lambda timeout=None: client.get("/kg/summary", timeout=timeout),
            text_builder=summarize_kg,
        )
    if name == "chroniccare_kg_entity_query":
        request_payload = {"query": _coerce_question(args), "max_nodes": int(args.get("max_nodes", 80))}
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="POST /kg/entity/query",
            input_payload=request_payload,
            invoke=lambda timeout=None: client.post("/kg/entity/query", request_payload, timeout=timeout),
            text_builder=summarize_kg_detail,
        )
    if name == "chroniccare_kg_relation_query":
        request_payload = {"query": _coerce_question(args), "max_nodes": int(args.get("max_nodes", 80))}
        relation_text = str(request_payload["query"] or "").replace(" ", "")
        if "高盐" in relation_text and "血压异常" in relation_text:
            def invoke_relation_with_subgraph(timeout=None) -> Dict[str, Any]:
                relation_payload = client.post("/kg/relation/query", request_payload, timeout=timeout)
                subgraph_payload = client.post("/kg/subgraph/render", request_payload, timeout=timeout)
                return {"relation": relation_payload, "subgraph": subgraph_payload}

            return execute_http_tool(
                client=client,
                tool_name=name,
                endpoint_label="POST /kg/relation/query + POST /kg/subgraph/render",
                input_payload=request_payload,
                invoke=invoke_relation_with_subgraph,
                text_builder=lambda payload: (
                    summarize_kg_detail(payload["relation"])
                    + "\n\n关系子图入口：\n"
                    + summarize_kg_detail(payload["subgraph"])
                ),
                data_builder=lambda payload: {
                    **payload["relation"],
                    "subgraph": payload["subgraph"],
                    "rerouted_to_subgraph_render": True,
                },
            )
        if _looks_like_subgraph_question(request_payload["query"]):
            return execute_http_tool(
                client=client,
                tool_name=name,
                endpoint_label="POST /kg/subgraph/render (rerouted from relation-query)",
                input_payload=request_payload,
                invoke=lambda timeout=None: client.post("/kg/subgraph/render", request_payload, timeout=timeout),
                text_builder=lambda payload: (
                    "检测到当前问题是在请求绘制关系子图，已自动改用 chroniccare_kg_subgraph_render 返回实时子图结果。\n"
                    + summarize_kg_detail(payload)
                ),
                data_builder=lambda payload: {
                    **payload,
                    "rerouted_from": "chroniccare_kg_relation_query",
                    "rerouted_to": "chroniccare_kg_subgraph_render",
                },
            )
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="POST /kg/relation/query",
            input_payload=request_payload,
            invoke=lambda timeout=None: client.post("/kg/relation/query", request_payload, timeout=timeout),
            text_builder=summarize_kg_detail,
        )
    if name == "chroniccare_kg_patient_path_query":
        patient_id = str(args.get("patient_id", "") or "").strip()
        if not patient_id and "某个患者" in str(args.get("query", "") or ""):
            patient_id = "P0001"
        request_payload = {"patient_id": patient_id, "max_hops": int(args.get("max_hops", 3))}
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="POST /kg/patient/path",
            input_payload=request_payload,
            invoke=lambda timeout=None: client.post("/kg/patient/path", request_payload, timeout=timeout),
            text_builder=summarize_kg_detail,
        )
    if name == "chroniccare_kg_subgraph_query":
        request_payload = {"query": _coerce_subgraph_query(args), "max_nodes": int(args.get("max_nodes", 80))}
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="POST /kg/subgraph/query",
            input_payload=request_payload,
            invoke=lambda timeout=None: client.post("/kg/subgraph/query", request_payload, timeout=timeout),
            text_builder=summarize_kg_detail,
        )
    if name == "chroniccare_kg_subgraph_render":
        request_payload = {"query": _coerce_subgraph_query(args), "max_nodes": int(args.get("max_nodes", 80))}
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="POST /kg/subgraph/render",
            input_payload=request_payload,
            invoke=lambda timeout=None: client.post("/kg/subgraph/render", request_payload, timeout=timeout),
            text_builder=summarize_kg_detail,
        )
    if name == "chroniccare_data_summary":
        question = _coerce_question(args)
        if _looks_like_npu_pipeline_run_question(question):
            return _execute_datamate_pipeline_run_npu(client, name, args)
        if _looks_like_datamate_pipeline_run_question(question):
            return _execute_datamate_pipeline_run(client, name, args)
        if _looks_like_disease_combination_question(question):
            return _execute_disease_combination_distribution(client, name, question)
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="GET /system/data-summary",
            input_payload=args,
            invoke=lambda timeout=None: client.get("/system/data-summary", timeout=timeout),
            text_builder=summarize_data_summary,
        )
    if name == "chroniccare_analysis_query":
        request_payload = {"question": _coerce_question(args)}
        if _looks_like_npu_pipeline_run_question(request_payload["question"]):
            return _execute_datamate_pipeline_run_npu(client, name, args)
        if _looks_like_datamate_pipeline_run_question(request_payload["question"]):
            return _execute_datamate_pipeline_run(client, name, args)
        if _looks_like_disease_combination_question(request_payload["question"]):
            return _execute_disease_combination_distribution(client, name, request_payload["question"])
        if _looks_like_report_summary_question(request_payload.get("question", "")):
            return _execute_report_summary(client, name, args)
        if _looks_like_subgraph_question(request_payload.get("question", "")):
            reroute_payload = {
                "query": request_payload.get("question", ""),
                "max_nodes": int(args.get("max_nodes", 80)),
            }
            return execute_http_tool(
                client=client,
                tool_name=name,
                endpoint_label="POST /kg/subgraph/render (rerouted from analysis/query)",
                input_payload=reroute_payload,
                invoke=lambda timeout=None: client.post("/kg/subgraph/render", reroute_payload, timeout=timeout),
                text_builder=lambda payload: (
                    "检测到当前问题实际是在请求知识图谱子图，已自动改用 chroniccare_kg_subgraph_render 返回实时子图结果。\n"
                    + summarize_kg_detail(payload)
                ),
                data_builder=lambda payload: {
                    **payload,
                    "rerouted_from": "chroniccare_analysis_query",
                    "rerouted_to": "chroniccare_kg_subgraph_render",
                },
            )
        if _looks_like_open_sql_question(request_payload.get("question", "")):
            open_sql_payload = {
                "question": request_payload.get("question", ""),
                "prefer_llm": True,
                "force_llm": False,
                "allow_chart": True,
            }
            return execute_http_tool(
                client=client,
                tool_name=name,
                endpoint_label="POST /analysis/open-sql/query (rerouted from analysis/query)",
                input_payload=open_sql_payload,
                invoke=lambda timeout=None: client.post("/analysis/open-sql/query", open_sql_payload, timeout=timeout),
                text_builder=lambda payload: (
                    "检测到当前问题是慢病指标/开放 SQL 统计请求，已自动改用 chroniccare_open_sql_query 返回真实 SQL 结果。\n"
                    + summarize_open_analysis(payload)
                ),
                data_builder=lambda payload: {
                    **payload,
                    "rerouted_from": "chroniccare_analysis_query",
                    "rerouted_to": "chroniccare_open_sql_query",
                },
                request_timeout=OPEN_SQL_TIMEOUT_SECONDS,
            )
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="POST /analysis/query",
            input_payload=request_payload,
            invoke=lambda timeout=None: client.post("/analysis/query", request_payload, timeout=timeout),
            text_builder=summarize_analysis,
        )
    if name == "chroniccare_disease_distribution":
        request_payload = {"question": _coerce_question(args)}
        if _looks_like_disease_combination_question(request_payload["question"]):
            return _execute_disease_combination_distribution(client, name, request_payload["question"])
        if _looks_like_contextual_cohort_question(request_payload["question"]):
            return execute_http_tool(
                client=client,
                tool_name=name,
                endpoint_label="POST /analysis/followup/cohort-disease-distribution (rerouted from disease_distribution)",
                input_payload=request_payload,
                invoke=lambda timeout=None: client.post(
                    "/analysis/followup/cohort-disease-distribution", request_payload, timeout=timeout
                ),
                text_builder=summarize_open_analysis,
            )
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="POST /analysis/disease-distribution",
            input_payload=request_payload,
            invoke=lambda timeout=None: client.post("/analysis/disease-distribution", request_payload, timeout=timeout),
            text_builder=summarize_analysis,
        )
    if name == "chroniccare_disease_combination_distribution":
        request_payload = {"question": _coerce_question(args)}
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="POST /analysis/disease-combination-distribution",
            input_payload=request_payload,
            invoke=lambda timeout=None: client.post("/analysis/disease-combination-distribution", request_payload, timeout=timeout),
            text_builder=summarize_analysis,
        )
    if name == "chroniccare_risk_level_distribution":
        request_payload = {"question": _coerce_question(args)}
        if _looks_like_disease_combination_question(request_payload["question"]):
            return _execute_disease_combination_distribution(client, name, request_payload["question"])
        if _looks_like_contextual_cohort_question(request_payload["question"]):
            return execute_http_tool(
                client=client,
                tool_name=name,
                endpoint_label="POST /analysis/followup/cohort-disease-distribution (rerouted from risk_level_distribution)",
                input_payload=request_payload,
                invoke=lambda timeout=None: client.post(
                    "/analysis/followup/cohort-disease-distribution", request_payload, timeout=timeout
                ),
                text_builder=summarize_open_analysis,
            )
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="POST /analysis/risk-level-distribution",
            input_payload=request_payload,
            invoke=lambda timeout=None: client.post("/analysis/risk-level-distribution", request_payload, timeout=timeout),
            text_builder=summarize_analysis,
        )
    if name == "chroniccare_followup_high_risk":
        request_payload = {"question": _coerce_followup_question(args)}
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="POST /analysis/followup/high-risk",
            input_payload=request_payload,
            invoke=lambda timeout=None: client.post("/analysis/followup/high-risk", request_payload, timeout=timeout),
            text_builder=summarize_analysis,
        )
    if name == "chroniccare_cohort_disease_distribution":
        request_payload = {"question": _coerce_question(args)}
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="POST /analysis/followup/cohort-disease-distribution",
            input_payload=request_payload,
            invoke=lambda timeout=None: client.post("/analysis/followup/cohort-disease-distribution", request_payload, timeout=timeout),
            text_builder=summarize_open_analysis,
        )
    if name == "chroniccare_metric_query":
        question = _coerce_metric_question(args)
        if _looks_like_disease_combination_question(question):
            return _execute_disease_combination_distribution(client, name, question)
        if _looks_like_disease_distribution_question(question):
            request_payload = {"question": question}
            return execute_http_tool(
                client=client,
                tool_name=name,
                endpoint_label="POST /analysis/disease-distribution (rerouted from metric)",
                input_payload=request_payload,
                invoke=lambda timeout=None: client.post("/analysis/disease-distribution", request_payload, timeout=timeout),
                text_builder=lambda payload: (
                    "检测到当前问题是疾病分布/单病患者数，已自动改用 chroniccare_disease_distribution 返回真实人数。\n"
                    + summarize_analysis(payload)
                ),
                data_builder=lambda payload: {
                    **payload,
                    "rerouted_from": "chroniccare_metric_query",
                    "rerouted_to": "chroniccare_disease_distribution",
                },
            )
        if _looks_like_followup_high_risk_question(question):
            request_payload = {"question": question}
            return execute_http_tool(
                client=client,
                tool_name=name,
                endpoint_label="POST /analysis/followup/high-risk (rerouted high-risk followup from metric)",
                input_payload=request_payload,
                invoke=lambda timeout=None: client.post("/analysis/followup/high-risk", request_payload, timeout=timeout),
                text_builder=lambda payload: (
                    "检测到当前问题是未来 N 天高风险随访人数，已自动改用 chroniccare_followup_high_risk 返回真实窗口人数。\n"
                    + summarize_analysis(payload)
                ),
            )
        request_payload = {"question": question, "prefer_llm": True, "allow_chart": True}
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="POST /analysis/open-sql/query (metric compatibility)",
            input_payload=request_payload,
            invoke=lambda timeout=None: client.post("/analysis/open-sql/query", request_payload, timeout=timeout),
            text_builder=summarize_open_analysis,
            request_timeout=OPEN_SQL_TIMEOUT_SECONDS,
        )
    if name == "chroniccare_trend_query":
        question = _coerce_trend_question(args)
        if _looks_like_followup_high_risk_question(question):
            followup_payload = {"question": question}
            return execute_http_tool(
                client=client,
                tool_name=name,
                endpoint_label="POST /analysis/followup/high-risk (rerouted from trend)",
                input_payload=followup_payload,
                invoke=lambda timeout=None: client.post("/analysis/followup/high-risk", followup_payload, timeout=timeout),
                text_builder=summarize_analysis,
            )
        request_payload = {"question": question, "prefer_llm": True, "allow_chart": True}
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="POST /analysis/open-sql/query (trend compatibility)",
            input_payload=request_payload,
            invoke=lambda timeout=None: client.post("/analysis/open-sql/query", request_payload, timeout=timeout),
            text_builder=summarize_open_analysis,
            request_timeout=OPEN_SQL_TIMEOUT_SECONDS,
        )
    if name == "chroniccare_graph_driven_analysis":
        request_payload = {"question": _coerce_question(args)}
        if _looks_like_npu_pipeline_run_question(request_payload["question"]):
            return _execute_datamate_pipeline_run_npu(client, name, args)
        if _looks_like_datamate_pipeline_run_question(request_payload["question"]):
            return _execute_datamate_pipeline_run(client, name, args)
        if _looks_like_disease_combination_question(request_payload["question"]):
            return _execute_disease_combination_distribution(client, name, request_payload["question"])
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="POST /analysis/graph-driven",
            input_payload=request_payload,
            invoke=lambda timeout=None: client.post("/analysis/graph-driven", request_payload, timeout=timeout),
            text_builder=summarize_open_analysis,
        )
    if name == "chroniccare_open_analysis_query":
        request_payload = {"question": _coerce_question(args)}
        if _looks_like_npu_pipeline_run_question(request_payload["question"]):
            return _execute_datamate_pipeline_run_npu(client, name, args)
        if _looks_like_datamate_pipeline_run_question(request_payload["question"]):
            return _execute_datamate_pipeline_run(client, name, args)
        if _looks_like_disease_combination_question(request_payload["question"]):
            return _execute_disease_combination_distribution(client, name, request_payload["question"])
        if _looks_like_disease_distribution_question(request_payload["question"]):
            return execute_http_tool(
                client=client,
                tool_name=name,
                endpoint_label="POST /analysis/disease-distribution (rerouted from open-analysis)",
                input_payload=request_payload,
                invoke=lambda timeout=None: client.post("/analysis/disease-distribution", request_payload, timeout=timeout),
                text_builder=lambda payload: (
                    "检测到当前问题是疾病分布/单病患者数，已自动改用 chroniccare_disease_distribution 返回真实人数。\n"
                    + summarize_analysis(payload)
                ),
                data_builder=lambda payload: {
                    **payload,
                    "rerouted_from": "chroniccare_open_analysis_query",
                    "rerouted_to": "chroniccare_disease_distribution",
                },
            )
        if _looks_like_followup_high_risk_question(request_payload["question"]):
            return execute_http_tool(
                client=client,
                tool_name=name,
                endpoint_label="POST /analysis/followup/high-risk (rerouted high-risk followup from open-analysis)",
                input_payload=request_payload,
                invoke=lambda timeout=None: client.post("/analysis/followup/high-risk", request_payload, timeout=timeout),
                text_builder=lambda payload: (
                    "检测到当前问题是未来 N 天高风险随访人数，已自动改用 chroniccare_followup_high_risk 返回真实窗口人数。\n"
                    + summarize_analysis(payload)
                ),
                data_builder=lambda payload: {
                    **payload,
                    "rerouted_from": "chroniccare_open_analysis_query",
                    "rerouted_to": "chroniccare_followup_high_risk",
                },
            )
        lowered_question = str(request_payload.get("question") or "").lower()
        timeout = LONG_RUNNING_TIMEOUT_SECONDS if any(
            token in lowered_question for token in ("datamate", "pipeline", "重跑", "重新处理", "数据处理流程", "算子")
        ) else OPEN_SQL_TIMEOUT_SECONDS
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="POST /analysis/open-query",
            input_payload=request_payload,
            invoke=lambda timeout=None: client.post("/analysis/open-query", request_payload, timeout=timeout),
            text_builder=summarize_open_analysis,
            request_timeout=timeout,
        )
    if name == "chroniccare_open_sql_query":
        force_llm = _to_bool(args.get("force_llm"), False)
        request_payload = {
            "question": _coerce_question(args),
            "prefer_llm": force_llm,
            "force_llm": force_llm,
            "allow_chart": _to_bool(args.get("allow_chart"), True),
        }
        if _looks_like_npu_pipeline_run_question(request_payload["question"]):
            return _execute_datamate_pipeline_run_npu(client, name, args)
        if _looks_like_datamate_pipeline_run_question(request_payload["question"]):
            return _execute_datamate_pipeline_run(client, name, args)
        if _looks_like_disease_combination_question(request_payload["question"]):
            return _execute_disease_combination_distribution(client, name, request_payload["question"])
        if _looks_like_report_summary_question(request_payload["question"]):
            return _execute_report_summary(client, name, args)
        if _looks_like_disease_distribution_question(request_payload["question"]):
            disease_payload = {"question": request_payload["question"]}
            return execute_http_tool(
                client=client,
                tool_name=name,
                endpoint_label="POST /analysis/disease-distribution (rerouted from open-sql)",
                input_payload=disease_payload,
                invoke=lambda timeout=None: client.post("/analysis/disease-distribution", disease_payload, timeout=timeout),
                text_builder=lambda payload: (
                    "检测到当前问题是疾病分布/单病患者数，已自动改用 chroniccare_disease_distribution 返回真实人数。\n"
                    + summarize_analysis(payload)
                ),
                data_builder=lambda payload: {
                    **payload,
                    "rerouted_from": "chroniccare_open_sql_query",
                    "rerouted_to": "chroniccare_disease_distribution",
                },
            )
        if _looks_like_followup_high_risk_question(request_payload["question"]):
            followup_payload = {"question": request_payload["question"]}
            return execute_http_tool(
                client=client,
                tool_name=name,
                endpoint_label="POST /analysis/followup/high-risk (rerouted high-risk followup from open-sql)",
                input_payload=followup_payload,
                invoke=lambda timeout=None: client.post("/analysis/followup/high-risk", followup_payload, timeout=timeout),
                text_builder=lambda payload: (
                    "检测到当前问题是未来 N 天高风险随访人数，已自动改用 chroniccare_followup_high_risk 返回真实窗口人数。\n"
                    + summarize_analysis(payload)
                ),
                data_builder=lambda payload: {
                    **payload,
                    "rerouted_from": "chroniccare_open_sql_query",
                    "rerouted_to": "chroniccare_followup_high_risk",
                },
            )
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="POST /analysis/open-sql/query",
            input_payload=request_payload,
            invoke=lambda timeout=None: client.post("/analysis/open-sql/query", request_payload, timeout=timeout),
            text_builder=summarize_open_analysis,
            request_timeout=OPEN_SQL_TIMEOUT_SECONDS,
        )
    if name == "chroniccare_open_sql_schema":
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="GET /analysis/open-sql/schema",
            input_payload=args,
            invoke=lambda timeout=None: client.get("/analysis/open-sql/schema", timeout=timeout),
            text_builder=summarize_open_sql_schema,
        )
    if name == "chroniccare_open_sql_eval":
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="GET /analysis/open-sql/eval",
            input_payload=args,
            invoke=lambda timeout=None: client.get("/analysis/open-sql/eval", timeout=timeout),
            text_builder=summarize_open_sql_eval,
        )
    if name == "chroniccare_open_sql_examples":
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="GET /analysis/open-sql/examples",
            input_payload=args,
            invoke=lambda timeout=None: client.get("/analysis/open-sql/examples", timeout=timeout),
            text_builder=summarize_open_sql_examples,
        )
    if name == "chroniccare_agent_run":
        request_payload = {"user_goal": args.get("user_goal", "")}
        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="POST /agent/run",
            input_payload=request_payload,
            invoke=lambda timeout=None: client.post("/agent/run", request_payload, timeout=timeout),
            text_builder=summarize_agent,
            request_timeout=LONG_RUNNING_TIMEOUT_SECONDS,
            data_builder=lambda payload: {
                "status": payload.get("status"),
                "run_id": payload.get("run_id"),
                "user_goal": payload.get("user_goal"),
                "plan": payload.get("plan"),
                "tool_results": payload.get("tool_results"),
                "tool_call_count": payload.get("tool_call_count"),
                "trace": payload.get("tool_results"),
                "trace_path": payload.get("trace_path"),
                "final_answer": payload.get("final_answer"),
                "artifacts_used": payload.get("artifacts_used"),
                "safety_note": payload.get("safety_note"),
                "raw": payload,
            },
        )
    if name == "chroniccare_report_summary":
        question = _coerce_question(args)
        if _looks_like_npu_pipeline_run_question(question):
            return _execute_datamate_pipeline_run_npu(client, name, args)
        if _looks_like_datamate_pipeline_run_question(question):
            return _execute_datamate_pipeline_run(client, name, args)
        if _looks_like_disease_combination_question(question):
            return _execute_disease_combination_distribution(client, name, question)
        def invoke_report() -> Dict[str, Any]:
            report_payload = client.get("/reports/summary")
            charts_payload = client.get("/charts/list")
            return {"report": report_payload, "charts": charts_payload}

        def build_data(payload: Dict[str, Any]) -> Dict[str, Any]:
            report_payload = payload["report"]
            charts_payload = payload["charts"]
            return {
                "status": report_payload.get("status"),
                "analysis_report_html": report_payload.get("analysis_report_html"),
                "analysis_report_md": report_payload.get("analysis_report_md"),
                "chart_index": report_payload.get("chart_index"),
                "graph_html": report_payload.get("graph_html"),
                "graph_url": report_payload.get("graph_url"),
                "global_graph_url": report_payload.get("global_graph_url"),
                "chart_index_url": report_payload.get("chart_index_url"),
                "report_url": report_payload.get("report_url"),
                "summary_text": report_payload.get("summary_text"),
                "entry_guide": report_payload.get("entry_guide"),
                "latest_graph_driven_analysis": report_payload.get("latest_graph_driven_analysis"),
                "charts": charts_payload.get("charts", []),
                "safety_note": report_payload.get("safety_note"),
            }

        return execute_http_tool(
            client=client,
            tool_name=name,
            endpoint_label="GET /reports/summary",
            input_payload=args,
            invoke=lambda timeout=None: invoke_report(),
            text_builder=lambda payload: summarize_report(payload["report"], payload["charts"]),
            data_builder=build_data,
        )
    if name == "chroniccare_trace_summary":
        summary = summarize_traces()
        return {
            "tool": name,
            "text": (
                f"最近共记录 {summary.get('total_calls', 0)} 次 MCP 工具调用，"
                f"成功率 {summary.get('success_rate', 0)}，"
                f"平均耗时 {summary.get('avg_latency_ms', 0)} ms。"
            ),
            "data": summary,
        }
    raise KeyError(f"Unsupported tool: {name}")


def create_app() -> FastAPI:
    settings = get_settings()
    tool_map = get_tool_map()
    app = FastAPI(
        title="ChronicCare MCP Adapter",
        version="0.1.0",
        description="Thin MCP-style adapter that forwards ChronicCare tool calls to the deployed FastAPI Tool Server.",
    )

    @app.get("/")
    def root() -> Dict[str, Any]:
        return {
            "project": "ChronicCare MCP Adapter",
            "status": "ok",
            "transport": settings["transport"],
            "tool_server_url": settings["tool_server_url"],
            "mcp_endpoint": "/mcp",
            "sse_endpoint": "/sse",
            "tools_endpoint": "/tools",
            "sdk_available": settings["sdk_available"],
            "tool_count": len(TOOL_DEFINITIONS),
        }

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "project": "ChronicCare MCP Adapter",
            "tool_server_url": settings["tool_server_url"],
            "transport": settings["transport"],
            "sdk_available": settings["sdk_available"],
        }

    @app.get("/tools")
    def tools() -> Dict[str, Any]:
        return {"status": "success", "tools": [build_tool_payload(item) for item in TOOL_DEFINITIONS]}

    @app.get("/trace/recent")
    def trace_recent(limit: int = 20) -> Dict[str, Any]:
        traces = load_recent_traces(limit=max(1, min(limit, 200)))
        slimmed = [
            {
                "trace_id": item.get("trace_id"),
                "timestamp": item.get("timestamp"),
                "tool_name": item.get("tool_name"),
                "status": item.get("status"),
                "latency_ms": item.get("latency_ms"),
                "input": item.get("input"),
                "output_summary": item.get("output_summary"),
                "error": item.get("error"),
            }
            for item in traces
        ]
        return {"status": "success", "count": len(slimmed), "traces": slimmed}

    @app.get("/trace/summary")
    def trace_summary() -> Dict[str, Any]:
        return summarize_traces()

    @app.post("/invoke")
    def invoke(request: ToolCallRequest) -> Dict[str, Any]:
        if request.name not in tool_map:
            raise HTTPException(status_code=404, detail=f"Unknown tool: {request.name}")
        try:
            return {"status": "success", **execute_tool(request.name, request.arguments)}
        except ChronicCareHTTPError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/mcp")
    def mcp(request: MCPRequest) -> Dict[str, Any]:
        try:
            if request.method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "chroniccare-mcp-adapter", "version": "0.1.0"},
                    "capabilities": {"tools": {"listChanged": False}},
                }
            elif request.method == "tools/list":
                result = {"tools": [build_tool_payload(item) for item in TOOL_DEFINITIONS]}
            elif request.method == "tools/call":
                tool_name = request.params.get("name", "")
                if tool_name not in tool_map:
                    raise KeyError(f"Unknown tool: {tool_name}")
                tool_result = execute_tool(tool_name, request.params.get("arguments", {}))
                result = {
                    "content": [{"type": "text", "text": tool_result["text"]}],
                    "structuredContent": tool_result["data"],
                    "isError": False,
                }
            elif request.method == "ping":
                result = {"pong": True}
            else:
                raise KeyError(f"Unsupported MCP method: {request.method}")
            return MCPResponse(id=request.id, result=result).model_dump(exclude_none=True)
        except ChronicCareHTTPError as exc:
            return MCPResponse(id=request.id, error=MCPError(code=-32002, message=str(exc))).model_dump(exclude_none=True)
        except KeyError as exc:
            return MCPResponse(id=request.id, error=MCPError(code=-32601, message=str(exc))).model_dump(exclude_none=True)
        except Exception as exc:
            return MCPResponse(id=request.id, error=MCPError(code=-32000, message=str(exc))).model_dump(exclude_none=True)

    @app.get("/mcp")
    def mcp_info() -> Dict[str, Any]:
        return {
            "status": "success",
            "message": "Use POST /mcp with JSON-RPC methods initialize, tools/list, tools/call, ping.",
            "tools": [item.name for item in TOOL_DEFINITIONS],
        }

    @app.get("/sse")
    def sse() -> StreamingResponse:
        async def event_stream():
            yield "event: message\n"
            yield 'data: {"status":"ok","message":"Use POST /mcp for tool calls; this SSE endpoint is informational."}\n\n'

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/sse/info")
    def sse_info() -> PlainTextResponse:
        return PlainTextResponse("SSE endpoint is informational. Preferred endpoint: /mcp")

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Run ChronicCare MCP Adapter.")
    settings = get_settings()
    parser.add_argument("--host", default=settings["host"])
    parser.add_argument("--port", default=settings["port"], type=int)
    args = parser.parse_args()
    uvicorn.run("mcp_adapter.server:create_app", host=args.host, port=args.port, factory=True)


if __name__ == "__main__":
    main()
