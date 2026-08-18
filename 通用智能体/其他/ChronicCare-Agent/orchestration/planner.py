from __future__ import annotations

from typing import Any, Dict, List

from tool_server.utils import load_server_config, safety_note

DATA_PROCESSING_TOOLS = {
    "datamate.pipelines",
    "datamate.pipeline_run",
    "datamate.pipeline_latest",
    "datamate.pipeline_status",
    "datamate.pipeline_report",
    "artifacts.status",
}

KNOWLEDGE_GRAPH_TOOLS = {
    "kg.summary",
    "kg.query",
    "kg.subgraph_render",
    "analysis.open_query",
}

DATA_ANALYSIS_TOOLS = {
    "analysis.open_sql_examples",
    "analysis.query",
    "analysis.open_query",
    "analysis.graph_driven",
    "reports.summary",
    "charts.list",
}


def _contains_any(text: str, keywords: List[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _is_subgraph_request(user_goal: str) -> bool:
    return _contains_any(user_goal, ["子图", "图谱子图", "关系图", "知识图谱"]) and _contains_any(
        user_goal,
        ["糖尿病", "高血压", "高风险", "中风险", "低风险", "群体", "患者"],
    )


def _is_data_processing_request(user_goal: str) -> bool:
    return _contains_any(
        user_goal,
        [
            "运行数据处理流程",
            "执行数据处理流程",
            "重新清洗",
            "重建主线产物",
            "pipeline 状态",
            "pipeline",
            "算子",
            "datamate",
            "全流程",
            "数据接入",
            "清洗摘要",
            "最新一次 datamate",
            "原始数据",
        ],
    )


def _is_kg_request(user_goal: str) -> bool:
    return _contains_any(
        user_goal,
        [
            "图谱",
            "子图",
            "知识图谱",
            "节点数",
            "边数",
            "实体类型",
            "关系类型",
            "图谱质量",
            "药物关系",
            "风险事件关系",
            "重建知识图谱",
            "几种病",
            "多少种病",
            "疾病总数",
            "疾病名称",
            "疾病类型总数",
            "有哪些病",
            "有什么病",
            "有啥病",
            "病种",
        ],
    )


def _is_analysis_request(user_goal: str) -> bool:
    return _contains_any(
        user_goal,
        [
            "统计",
            "趋势",
            "图表",
            "nl2sql",
            "分析",
            "报告",
            "分布",
            "top",
            "压力",
            "未来",
            "随访",
            "csv",
        ],
    )


def _is_future_followup_visual_request(user_goal: str) -> bool:
    if not _contains_any(user_goal, ["未来", "随访"]):
        return False
    return _contains_any(
        user_goal,
        ["图", "图表", "可视化", "趋势", "折线图", "饼图", "画图", "画出来", "展示", "人数", "数量", "多少"],
    )


def _infer_analysis_question(user_goal: str) -> str:
    if _contains_any(user_goal, ["未来", "随访"]):
        return user_goal
    if _contains_any(user_goal, ["hba1c", "糖尿病"]):
        return "高血压合并糖尿病患者的平均 HbA1c 是多少？"
    if _contains_any(user_goal, ["ldl", "血脂"]):
        return "不同疾病组合的 LDL-C 异常比例是多少？"
    if _contains_any(user_goal, ["空腹血糖", "趋势"]):
        return "不同月份的空腹血糖异常人数趋势如何？"
    if _contains_any(user_goal, ["bmi"]):
        return "BMI 偏高患者中血压异常比例是多少？"
    return "高血压合并糖尿病患者的平均 HbA1c 是多少？"


def _infer_kg_entity(user_goal: str) -> Dict[str, str]:
    if _contains_any(user_goal, ["糖尿病", "diabetes"]):
        return {"query_type": "disease_profile", "entity_id": "Disease::diabetes"}
    if _contains_any(user_goal, ["高血压", "hypertension"]):
        return {"query_type": "disease_profile", "entity_id": "Disease::hypertension"}
    if _contains_any(user_goal, ["空腹血糖", "fasting_glucose"]):
        return {"query_type": "indicator_profile", "entity_id": "Indicator::fasting_glucose"}
    if _contains_any(user_goal, ["药物", "metformin"]):
        return {"query_type": "drug_profile", "entity_id": "Drug::metformin"}
    return {"query_type": "disease_profile", "entity_id": "Disease::hypertension"}


def _is_disease_inventory_request(user_goal: str) -> bool:
    return _contains_any(
        user_goal,
        [
            "几种病",
            "多少种病",
            "疾病总数",
            "疾病名称",
            "疾病类型总数",
            "有哪些病",
            "有什么病",
            "有什么疾病",
            "患者有什么病",
            "患者有什么疾病",
            "患者得了什么病",
            "患者得了什么疾病",
            "患者患有什么病",
            "患者患有什么疾病",
            "有啥病",
            "都有什么病",
            "病种",
            "疾病分布",
            "常见病",
            "常见疾病",
        ],
    )


def _plan_step(
    step: int,
    agent: str,
    tool_group: str,
    tool: str,
    description: str,
    input_hint: Dict[str, Any],
    expected_output: str,
) -> Dict[str, Any]:
    return {
        "step": step,
        "agent": agent,
        "tool_group": tool_group,
        "tool": tool,
        "description": description,
        "input_hint": input_hint,
        "expected_output": expected_output,
    }


def _tool_group_for_tool(tool: str) -> str:
    if tool in DATA_PROCESSING_TOOLS:
        return "data_processing_tools"
    if tool in KNOWLEDGE_GRAPH_TOOLS:
        return "knowledge_graph_tools"
    if tool in DATA_ANALYSIS_TOOLS:
        return "data_analysis_tools"
    return "shared_tools"


def build_plan(user_goal: str) -> Dict[str, Any]:
    plan: List[Dict[str, Any]] = []
    step = 1
    if _is_disease_inventory_request(user_goal):
        plan.append(
            _plan_step(
                step,
                "AnalysisAgent",
                "data_analysis_tools",
                "analysis.open_query",
                "按当前问句实时统计疾病类型与患者覆盖人数，并生成疾病分布图与分析页面。",
                {"question": user_goal},
                "disease_type_count、table、chart_url、report_url、graph_url",
            )
        )
        cfg = load_server_config()
        return {
            "status": "success",
            "user_goal": user_goal,
            "primary_tool_group": "data_analysis_tools",
            "plan": plan,
            "safety_note": safety_note(cfg),
        }

    if _is_subgraph_request(user_goal):
        plan.append(
            _plan_step(
                step,
                "KGSubgraphAgent",
                "knowledge_graph_tools",
                "kg.subgraph_render",
                "按用户当前问题实时生成群体图谱子图，并返回本次生成的稳定图谱链接；禁止退回旧报告或历史图表索引。",
                {"query": user_goal, "max_nodes": 96},
                "html_url、subgraph_id、node_count、edge_count、cohort_patient_count",
            )
        )
        cfg = load_server_config()
        return {
            "status": "success",
            "user_goal": user_goal,
            "plan": plan,
            "safety_note": safety_note(cfg),
        }

    if _is_data_processing_request(user_goal):
        should_run = _contains_any(user_goal, ["运行", "执行", "重跑", "重新", "重建"])
        if should_run:
            plan.append(
                _plan_step(
                    step,
                    "DataProcessingAgent",
                    "data_processing_tools",
                    "datamate.pipeline_run",
                    "从原始数据重新触发 DataMate 11 个算子全流程，并返回执行摘要；若当前环境无法直接触发，则返回诚实的 CLI fallback。",
                    {"task_id": "planner_datamate_run_001", "force": True, "safe_run": True},
                    "status、run_id、summary、artifact_paths、warnings、errors",
                )
            )
            step += 1
        else:
            plan.append(
                _plan_step(
                    step,
                    "DataProcessingAgent",
                    "data_processing_tools",
                    "datamate.pipelines",
                    "说明三条 DataMate pipeline 与 11 个算子映射、当前可调用方式和最新状态。",
                    {},
                    "pipelines、operators、invocation_mode、latest_run",
                )
            )
            step += 1
        plan.append(
            _plan_step(
                step,
                "DataProcessingAgent",
                "data_processing_tools",
                "datamate.pipeline_status",
                "读取最近一次 DataMate pipeline 的 11 个算子执行状态与当前主线指标。",
                {"run_id": "latest"},
                "run_id、pipeline_name、steps、summary、metrics",
            )
        )
        step += 1
        plan.append(
            _plan_step(
                step,
                "DataProcessingAgent",
                "data_processing_tools",
                "datamate.pipeline_report",
                "返回最近一次 DataMate pipeline 的运行报告、检查报告和同步摘要，便于查看清洗结果和产物。",
                {},
                "run_id、pipeline_name、summary、artifact_paths、report_path",
            )
        )
        cfg = load_server_config()
        return {
            "status": "success",
            "user_goal": user_goal,
            "primary_tool_group": "data_processing_tools",
            "plan": plan,
            "safety_note": safety_note(cfg),
        }

    needs_analysis = _is_analysis_request(user_goal)
    needs_kg = _is_kg_request(user_goal)
    needs_report = _contains_any(user_goal, ["图表", "报告", "展示", "入口", "总结", "汇总", "可视化", "csv"])
    prefer_open_analysis = _is_future_followup_visual_request(user_goal)

    if needs_analysis:
        question = _infer_analysis_question(user_goal)
        plan.append(
            _plan_step(
                step,
                "AnalysisAgent",
                "data_analysis_tools",
                "analysis.open_query" if prefer_open_analysis else "analysis.query",
                "匹配慢病分析问题并返回指标结果、表格和洞察。未来随访可视化问题优先走开放式图表路由，并严格按用户请求的 N 天实时生成。",
                {"question": question},
                "指标结果、表格、insight、graph_url、chart_url、report_url",
            )
        )
        step += 1

    if needs_kg:
        if _is_disease_inventory_request(user_goal):
            plan.append(
                _plan_step(
                    step,
                    "KGSummaryAgent",
                    "knowledge_graph_tools",
                    "kg.summary",
                    "读取真实知识图谱中的疾病类型总数、疾病名称、节点数和边数摘要。",
                    {},
                    "disease_type_count、disease_labels、node_count、edge_count",
                )
            )
        else:
            plan.append(
                _plan_step(
                    step,
                    "KGQAAgent",
                    "knowledge_graph_tools",
                    "kg.query",
                    "查询相关疾病、指标或药物的图谱关系，用于解释分析结果。",
                    _infer_kg_entity(user_goal),
                    "answer、facts、evidence_paths",
                )
            )
        step += 1

    if needs_report and not prefer_open_analysis:
        if not plan:
            plan.append(
                _plan_step(
                    step,
                    "DataProcessingAgent",
                    "data_processing_tools",
                    "artifacts.status",
                    "检查 SQLite、知识图谱、图表和报告等关键产物是否已存在。",
                    {},
                    "产物存在性、相对路径、可用状态",
                )
            )
            step += 1
        plan.append(
            _plan_step(
                step,
                "VisualizationAgent",
                "data_analysis_tools",
                "reports.summary",
                "返回图表索引、图谱页面和分析报告入口，便于演示展示。",
                {},
                "chart_index、analysis_report_html、graph_html",
            )
        )
    elif not plan:
        plan.append(
            _plan_step(
                step,
                "DataProcessingAgent",
                "data_processing_tools",
                "artifacts.status",
                "检查 SQLite、知识图谱、图表和报告等关键产物是否已存在。",
                {},
                "产物存在性、相对路径、可用状态",
            )
        )
        step += 1
        plan.append(
            _plan_step(
                step,
                "VisualizationAgent",
                "data_analysis_tools",
                "reports.summary",
                "返回图表索引、图谱页面和分析报告入口，便于演示展示。",
                {},
                "chart_index、analysis_report_html、graph_html",
            )
        )

    cfg = load_server_config()
    return {
        "status": "success",
        "user_goal": user_goal,
        "primary_tool_group": (
            "knowledge_graph_tools"
            if needs_kg and not needs_analysis
            else "data_analysis_tools"
            if needs_analysis
            else "shared_tools"
        ),
        "plan": plan,
        "safety_note": safety_note(cfg),
    }
