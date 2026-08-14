from __future__ import annotations

from typing import Any, Dict


def format_answer(plan: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    if not payload or payload.get("status") != "success":
        return payload

    intent = str(plan.get("intent") or payload.get("intent") or "")
    result = dict(payload)
    if result.get("answer"):
        return result

    if intent == "kg_summary":
        result["answer"] = (
            f"当前主线数据规模为：患者 {result.get('patient_count', 0)} 人，"
            f"随访记录 {result.get('visit_count', 0)} 条，检验记录 {result.get('lab_result_count', 0)} 条，"
            f"用药记录 {result.get('medication_record_count', 0)} 条；知识图谱节点 {result.get('node_count', 0)} 个，"
            f"边 {result.get('edge_count', 0)} 条。"
        )
    elif intent == "report_summary":
        result["answer"] = "已整理当前可直接打开的图表、报告和图谱入口，请优先使用返回的 HTTP URL。"
    elif intent == "capability_examples":
        result["answer"] = "已返回当前 Open SQL 能力边界和示例问题；系统能力不以固定题目总数限定。"
    elif intent == "kg_subgraph_render":
        result["answer"] = f"已生成问题驱动知识图谱子图，可直接打开：{result.get('html_url', '')}"
        if result.get("html_url") and not result.get("graph_url"):
            result["graph_url"] = result["html_url"]
    elif intent == "kg_relation_query":
        result["answer"] = str(result.get("text") or result.get("insight") or "")
    elif intent == "kg_entity_query":
        result["answer"] = str(result.get("text") or "")
    elif intent == "system_status":
        result["answer"] = str(result.get("summary_text") or "系统当前正常运行。")
    elif intent == "unsupported_negation_query":
        result["answer"] = str(result.get("summary_text") or "")
    elif intent == "datamate_pipeline_status":
        result["answer"] = str(result.get("summary") or result.get("summary_text") or "已返回最近一次 DataMate pipeline 状态。")
    elif intent == "datamate_pipeline_run":
        result["answer"] = str(result.get("summary") or result.get("summary_text") or "已返回 DataMate pipeline 执行摘要。")
    else:
        result["answer"] = str(result.get("summary_text") or result.get("text") or "")
    if result.get("answer") and not result.get("summary_text"):
        result["summary_text"] = result["answer"]
    return result
