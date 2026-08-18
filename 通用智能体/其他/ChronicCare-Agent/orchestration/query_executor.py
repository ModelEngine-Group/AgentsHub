from __future__ import annotations

from typing import Any, Dict

from tool_server.utils import load_server_config, safety_note


def execute_query_plan(plan: Dict[str, Any]) -> Dict[str, Any] | None:
    intent = str(plan.get("intent") or "")
    query = str(plan.get("query") or "")
    if plan.get("executor") != "direct_tool":
        return None

    if intent == "unsupported_negation_query":
        cfg = load_server_config()
        return {
            "status": "success",
            "intent": intent,
            "question": query,
            "summary_text": "当前否定条件查询暂未开放，请改问受支持的问题，例如疾病分布、风险分布、随访统计或知识图谱子图。",
            "supported_followups": [
                "当前常见病有哪些？",
                "不同风险等级患者人数分布是多少？",
                "未来 30 天需要随访的高风险患者有多少？",
                "给我高血压患者知识图谱子图",
            ],
            "warnings": ["检测到否定条件语义，本阶段仅记录并安全回退，不执行复杂否定查询。"],
            "safety_note": safety_note(cfg),
        }

    if intent == "system_status":
        cfg = load_server_config()
        return {
            "status": "success",
            "intent": intent,
            "question": query,
            "summary_text": "系统当前正常运行，Tool Server 可提供健康检查、图谱、分析和报告入口。",
            "base_url": f"{cfg['server'].get('browser_base_url', '').rstrip('/')}",
            "safety_note": safety_note(cfg),
        }

    if intent == "report_summary":
        from tool_server.report_tools import reports_summary

        return reports_summary()
    if intent == "data_summary":
        from tool_server.analysis_tools import data_summary

        return data_summary()
    if intent == "kg_summary":
        from tool_server.kg_tools import kg_summary

        return kg_summary()
    if intent == "capability_examples":
        from tool_server.open_sql_tools import open_sql_examples_tool

        return open_sql_examples_tool()
    if intent == "kg_relation_query":
        from tool_server.kg_tools import kg_relation_query

        return kg_relation_query(query)
    if intent == "kg_entity_query":
        from tool_server.kg_tools import kg_entity_query

        return kg_entity_query(query)
    if intent == "kg_subgraph_render":
        from tool_server.kg_tools import kg_subgraph_render

        return kg_subgraph_render(query, max_nodes=96)
    if intent == "kg_patient_path_query":
        from tool_server.kg_tools import kg_patient_path_query

        patient_id = ""
        for token in query.replace("，", " ").split():
            if token.upper().startswith("P") and len(token) >= 5:
                patient_id = token.upper()
                break
        if not patient_id and "某个患者" in query:
            patient_id = "P0001"
        if not patient_id:
            cfg = load_server_config()
            return {
                "status": "failed",
                "intent": intent,
                "question": query,
                "errors": ["请提供明确患者编号，例如 P0001。"],
                "safety_note": safety_note(cfg),
            }
        return kg_patient_path_query(patient_id)
    if intent == "datamate_pipeline_run":
        from tool_server.pipeline_tools import run_datamate_pipeline

        return run_datamate_pipeline(task_id="rule_pipeline_run", force=True, safe_run=True)
    if intent == "datamate_pipeline_run_npu":
        from tool_server.npu_tools import run_npu_enhanced_pipeline

        return run_npu_enhanced_pipeline(task_id="rule_pipeline_run_npu", use_npu=True, fallback=True, force=True, safe_run=True)
    if intent == "datamate_pipelines":
        from tool_server.pipeline_tools import datamate_pipelines_overview

        return datamate_pipelines_overview()
    if intent == "datamate_pipeline_status":
        from tool_server.pipeline_tools import datamate_pipeline_status_by_run

        return datamate_pipeline_status_by_run("latest")
    if intent == "npu_readiness_query":
        from tool_server.npu_tools import npu_readiness

        return npu_readiness()
    if intent == "npu_supported_operators":
        from tool_server.npu_tools import npu_supported_operators

        return npu_supported_operators()
    if intent == "npu_operator_benchmark":
        from tool_server.npu_tools import run_npu_operator_benchmark

        return run_npu_operator_benchmark(use_npu=True, fallback=True)
    return None
