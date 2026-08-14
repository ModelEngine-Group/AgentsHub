from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List
from urllib.request import ProxyHandler, Request, build_opener

import pandas as pd
import plotly.express as px
import streamlit as st

from tool_server.utils import load_server_config, public_artifact_url, service_artifact_url

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EVALUATION_DIR = OUTPUT_DIR / "evaluation"
CHART_DIR = OUTPUT_DIR / "charts"
SUBGRAPH_DIR = OUTPUT_DIR / "subgraphs"
GRAPH_ANALYSIS_DIR = OUTPUT_DIR / "graph_driven_analysis"
SQLITE_PATH = PROJECT_ROOT / "data/sqlite/chroniccare.db"
SAFETY_NOTE = "医疗安全说明：本项目仅用于慢病随访数据处理、知识组织和辅助分析，不提供临床诊断，不替代医生决策。"
RISK_COLORS = {"high": "#d1495b", "medium": "#e9a23b", "low": "#2e7d32", "unknown": "#8a94a6"}


@st.cache_data(show_spinner=False)
def load_json_if_exists(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_urls() -> Dict[str, str]:
    cfg = load_server_config()
    return {
        "graph_url": public_artifact_url(cfg, "/artifacts/graph.html"),
        "graph_service_url": service_artifact_url(cfg, "/artifacts/graph.html"),
        "chart_index_url": public_artifact_url(cfg, "/artifacts/charts"),
        "report_url": public_artifact_url(cfg, "/artifacts/report"),
    }


def _query_rows(sql: str, params: List[Any] | None = None) -> List[Dict[str, Any]]:
    with sqlite3.connect(f"file:{SQLITE_PATH}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        return [dict(row) for row in conn.execute(sql, params or []).fetchall()]


def _filtered_patients(diseases: List[str], risk: str, gender: str, age_range: tuple[int, int]) -> List[Dict[str, Any]]:
    sql = """
    WITH latest_risk AS (
      SELECT patient_id, risk_level, risk_score,
             ROW_NUMBER() OVER (PARTITION BY patient_id ORDER BY datetime(created_at) DESC, visit_id DESC) AS rn
      FROM patient_risk_score
    )
    SELECT p.patient_id, p.gender, CAST(p.age AS INTEGER) AS age,
           ROUND(CAST(p.bmi AS REAL), 2) AS bmi, p.disease_tags,
           COALESCE(r.risk_level, 'unknown') AS risk_level,
           CAST(r.risk_score AS REAL) AS risk_score
    FROM patient_profile p
    LEFT JOIN latest_risk r ON p.patient_id = r.patient_id AND r.rn = 1
    WHERE CAST(p.age AS INTEGER) BETWEEN ? AND ?
    """
    params: List[Any] = [age_range[0], age_range[1]]
    if gender != "全部":
        sql += " AND p.gender = ?"
        params.append(gender)
    for disease in diseases:
        sql += " AND lower(p.disease_tags) LIKE ?"
        params.append(f"%{disease.lower()}%")
    if risk != "全部":
        sql += " AND COALESCE(r.risk_level, 'unknown') = ?"
        params.append(risk)
    sql += " ORDER BY p.patient_id"
    return _query_rows(sql, params)


def _monthly_visit_rows(patient_ids: List[str]) -> List[Dict[str, Any]]:
    if not patient_ids:
        return []
    placeholders = ",".join("?" for _ in patient_ids)
    return _query_rows(
        f"""
        SELECT substr(visit_date, 1, 7) AS month, COUNT(*) AS visit_count,
               COUNT(DISTINCT patient_id) AS patient_count
        FROM visit_record
        WHERE patient_id IN ({placeholders}) AND visit_date IS NOT NULL
        GROUP BY substr(visit_date, 1, 7)
        ORDER BY month
        """,
        patient_ids,
    )


def _service_probe(url: str) -> Dict[str, Any]:
    try:
        response = build_opener(ProxyHandler({})).open(url, timeout=3)
        return {"status": "online", "http": response.status}
    except Exception as exc:
        return {"status": "offline", "error": type(exc).__name__}


def _post_json(url: str, body: Dict[str, Any]) -> Dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with build_opener(ProxyHandler({})).open(request, timeout=120) as response:
        return json.load(response)


def render_metric_grid(metrics: Dict[str, Any]) -> None:
    labels = [
        ("患者数", metrics.get("patient_count"), "当前筛选条件下去重 patient_id 数"),
        ("随访记录数", metrics.get("visit_count"), "visit_record 记录数"),
        ("检验记录数", metrics.get("lab_result_count"), "lab_result 记录数"),
        ("用药记录数", metrics.get("medication_record_count"), "medication_record 记录数"),
        ("图谱节点数", metrics.get("node_count"), "最终事件级 provenance 图谱节点数"),
        ("图谱边数", metrics.get("edge_count"), "事件边与语义关系边总数"),
        ("NL2SQL 盲测题数", metrics.get("nl2sql_question_count") or metrics.get("question_count"), "正式 blind 数据集题数"),
    ]
    cols = st.columns(4)
    for index, (label, value, help_text) in enumerate(labels):
        cols[index % 4].metric(label, value if value is not None else "N/A", help=help_text, border=True)


def _render_paginated_table(rows: List[Dict[str, Any]], key: str) -> None:
    page_size = st.selectbox("每页行数", [10, 20, 50, 100], index=1, key=f"{key}_size")
    page_count = max(1, math.ceil(len(rows) / page_size))
    page = st.number_input("页码", min_value=1, max_value=page_count, value=1, key=f"{key}_page")
    start = (int(page) - 1) * page_size
    st.dataframe(rows[start : start + page_size], width="stretch", hide_index=True)
    st.caption(f"第 {int(page)}/{page_count} 页 · 共 {len(rows)} 行")


def _render_plot(fig, source: str) -> None:
    fig.update_layout(margin=dict(l=20, r=20, t=55, b=20), legend_title_text="")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"数据来源：{source}")


def _recent_trace_rows(limit: int = 20) -> List[Dict[str, Any]]:
    path = OUTPUT_DIR / "mcp_traces/mcp_tool_calls.jsonl"
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[-limit:]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append({k: row.get(k) for k in ("trace_id", "tool_name", "status", "started_at", "duration_ms", "error")})
    return rows


def main() -> None:
    st.set_page_config(page_title="ChronicCare 决赛驾驶舱", page_icon="🩺", layout="wide")
    st.title("ChronicCare-Agent · 数据—知识—洞察驾驶舱")
    st.caption("DataMate × Nexent × 医疗知识图谱 × 受控 NL2SQL × 昇腾 NPU")
    st.warning(SAFETY_NOTE)

    metrics = load_json_if_exists(PROJECT_ROOT / "configs/current_metrics.json") or {}
    graph_summary = load_json_if_exists(PROJECT_ROOT / "data/graph/graph_summary.json") or {}
    consistency = load_json_if_exists(EVALUATION_DIR / "current_data_consistency_report.json") or {}
    context = consistency
    version = context.get("data_version") or metrics.get("data_version") or "unknown"
    as_of = (context.get("generated_at") or "实时").split("T", 1)[0]

    st.sidebar.header("全局筛选")
    diseases = st.sidebar.multiselect(
        "疾病/风险标签",
        ["hypertension", "diabetes", "hyperlipidemia", "obesity", "ckd_risk", "coronary_risk"],
    )
    risk = st.sidebar.selectbox("风险等级", ["全部", "low", "medium", "high"])
    gender = st.sidebar.selectbox("性别", ["全部", "male", "female"])
    age = st.sidebar.slider("年龄范围", 18, 100, (18, 100))
    rows = _filtered_patients(diseases, risk, gender, age)
    patient_ids = [str(row["patient_id"]) for row in rows]
    st.sidebar.caption(f"数据版本：{version}\n\n分析日期：{as_of}\n\n筛选患者：{len(rows)}")

    overview, dag_tab, kg_tab, insight_tab, sql_tab, npu_tab = st.tabs(
        ["系统状态", "DataMate 数据处理智能体", "知识图谱", "数据分析", "NL2SQL 安全与评测", "NPU 对比"]
    )

    with overview:
        st.subheader("闭环运行状态")
        merged = dict(metrics)
        merged.update(
            {
                "patient_count": len(rows),
                "node_count": graph_summary.get("node_count", metrics.get("node_count")),
                "edge_count": graph_summary.get("edge_count", metrics.get("edge_count")),
            }
        )
        render_metric_grid(merged)
        services = {
            "Tool Server": _service_probe("http://127.0.0.1:18088/health"),
            "MCP Adapter": _service_probe("http://127.0.0.1:18188/tools"),
        }
        c1, c2, c3 = st.columns(3)
        c1.metric("数据版本", version, border=True)
        c2.metric("分析日期", as_of, border=True)
        c3.metric("跨工具一致性", f"{consistency.get('consistency_rate', 0):.0%}", border=True)
        st.dataframe([{"服务": name, **state} for name, state in services.items()], width="stretch", hide_index=True)
        pipeline = load_json_if_exists(OUTPUT_DIR / "release/datamate_full_pipeline_report.json") or {}
        st.caption(f"最近 DataMate：{pipeline.get('status', 'unknown')} · {pipeline.get('pure_execution_seconds', 'N/A')} 秒")
        st.markdown("#### 筛选患者明细")
        _render_paginated_table(rows, "overview_patients")
        traces = _recent_trace_rows()
        with st.expander("最近 MCP 调用 Trace"):
            if traces:
                st.dataframe(traces, width="stretch", hide_index=True)
            else:
                st.info("暂无 MCP trace；在 Nexent 中执行一次工具调用后会显示。")

    with dag_tab:
        st.subheader("动态 DAG、状态与 Lineage")
        dag = load_json_if_exists(EVALUATION_DIR / "dynamic_dag_acceptance_report.json") or {}
        case_rows = [{"验收场景": name, "通过": passed} for name, passed in (dag.get("cases") or {}).items()]
        st.dataframe(case_rows, width="stretch", hide_index=True)
        plan_names = list((dag.get("plans") or {}).keys())
        selected_plan = st.selectbox("DAG 目标", plan_names or ["无计划"])
        plan = (dag.get("plans") or {}).get(selected_plan) or {}
        st.json({k: plan.get(k) for k in ("status", "goal", "dag_hash", "dry_run", "skipped", "risks", "estimated_resources")})
        st.dataframe(plan.get("nodes") or [], width="stretch", hide_index=True)
        run_dirs = sorted((OUTPUT_DIR / "dag_runs").glob("dag_*"), reverse=True)
        if run_dirs:
            selected_run = st.selectbox("运行记录", [path.name for path in run_dirs])
            run_doc = load_json_if_exists(OUTPUT_DIR / "dag_runs" / selected_run / "run.json") or {}
            st.json({k: run_doc.get(k) for k in ("run_id", "state", "started_at", "ended_at", "resume_from", "degraded")})
            node_rows = list((run_doc.get("nodes") or {}).values()) if isinstance(run_doc.get("nodes"), dict) else run_doc.get("nodes") or []
            st.dataframe(node_rows, width="stretch", hide_index=True)
        st.info("plan/dry-run 零写入；run 才执行；节点详情保留输入输出 hash、重试、resume 与异常。")

    with kg_tab:
        st.subheader("知识图谱与证据钻取")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("节点", graph_summary.get("node_count"), help="事件级 provenance 图谱节点", border=True)
        c2.metric("边", graph_summary.get("edge_count"), help="事件边和语义边总量", border=True)
        c3.metric("事件边", graph_summary.get("event_edge_count"), help="带唯一事件身份的纵向边", border=True)
        c4.metric("精确重复边", graph_summary.get("exact_duplicate_edge_count"), border=True)
        graph_path = PROJECT_ROOT / "data/graph/graph.html"
        if graph_path.exists():
            st.iframe("http://127.0.0.1:18088/artifacts/graph.html", height=600)
        else:
            st.error("图谱 HTML 不存在。")
        graph = load_json_if_exists(PROJECT_ROOT / "data/graph/graph.json") or {}
        edges = graph.get("edges") or []
        disease_ids = {f"Disease::{item}" for item in diseases}
        linked_edges = [edge for edge in edges if not disease_ids or edge.get("source") in disease_ids or edge.get("target") in disease_ids]
        candidates = linked_edges[:200] or edges[:200]
        if candidates:
            labels = [f"{edge.get('edge_id', index)} · {edge.get('source')} —{edge.get('relation')}→ {edge.get('target')}" for index, edge in enumerate(candidates)]
            selected = st.selectbox("选择关系查看 provenance", range(len(labels)), format_func=lambda index: labels[index])
            evidence = candidates[int(selected)]
            fields = ("edge_id", "source", "relation", "target", "source_type", "source_table", "source_record_id", "source_chunk_id", "source_span", "visit_id", "observed_at", "generated_at", "extractor", "extractor_version", "rule_id", "model_name", "confidence", "data_version")
            st.json({key: evidence.get(key) for key in fields})
        else:
            st.info("当前筛选没有可钻取关系。")
        semantic = load_json_if_exists(EVALUATION_DIR / "kg_semantic_quality_report.json") or {}
        qa = load_json_if_exists(EVALUATION_DIR / "kgqa_quality_report.json") or {}
        st.json({"entity_strict": semantic.get("entity_strict"), "relation": semantic.get("relation"), "kgqa_exact": qa.get("exact_accuracy"), "annotation_type": semantic.get("annotation_type")})

    with insight_tab:
        st.subheader("筛选联动洞察")
        disease_counts = Counter(tag for row in rows for tag in str(row.get("disease_tags") or "").split(";") if tag and tag != "nan")
        if disease_counts:
            disease_df = pd.DataFrame([{"疾病": name, "患者数": count} for name, count in disease_counts.most_common(15)])
            _render_plot(px.bar(disease_df, x="疾病", y="患者数", title="当前筛选队列疾病分布", labels={"患者数": "去重患者数（人）"}), "patient_profile.disease_tags；当前侧栏筛选队列")
        else:
            st.info("当前筛选无疾病数据。")
        risk_counts = Counter(str(row.get("risk_level") or "unknown") for row in rows)
        risk_df = pd.DataFrame([{"风险等级": level, "患者数": count} for level, count in risk_counts.items()])
        if not risk_df.empty:
            fig = px.bar(risk_df, x="风险等级", y="患者数", color="风险等级", color_discrete_map=RISK_COLORS, title="当前筛选队列最新风险分层", labels={"患者数": "去重患者数（人）"})
            _render_plot(fig, "patient_risk_score；按 created_at/visit_id 取每名患者最新记录")
        trend = _monthly_visit_rows(patient_ids)
        if trend:
            trend_df = pd.DataFrame(trend)
            _render_plot(px.line(trend_df, x="month", y="visit_count", markers=True, title="当前筛选队列月度随访趋势", labels={"month": "月份", "visit_count": "随访记录数（条）"}), "visit_record.visit_date；当前侧栏筛选队列")
        labels = sorted(disease_counts)[:10]
        matrix = [[sum(left in str(row.get("disease_tags")) and right in str(row.get("disease_tags")) for row in rows) for right in labels] for left in labels]
        if labels:
            _render_plot(px.imshow(matrix, x=labels, y=labels, color_continuous_scale="Blues", text_auto=True, title="共病矩阵（患者交集）", labels={"x": "疾病", "y": "疾病", "color": "患者数"}), "patient_profile.disease_tags；去重患者交集")
        st.caption("指标口径：患者数均为去重 patient_id；风险等级采用最新记录策略；日期窗口包含起始日。")

        st.markdown("#### 自然语言分析入口")
        question = st.text_input("分析问题", value="糖尿病患者的空腹血糖平均值是多少？")
        if st.button("执行受控分析", type="primary"):
            with st.spinner("正在执行工具规划、SQL Guard 和只读查询……"):
                try:
                    result = _post_json("http://127.0.0.1:18088/analysis/open-query", {"question": question})
                    st.success(result.get("summary_text") or result.get("answer_markdown") or result.get("status"))
                    st.code(result.get("sql") or "未生成 SQL", language="sql")
                    st.json({k: result.get(k) for k in ("stage", "trace_id", "data_version", "as_of_date", "sql_guard", "table", "charts")})
                except Exception as exc:
                    st.error(f"分析失败：{type(exc).__name__}")

    with sql_tab:
        st.subheader("受控 NL2SQL")
        blind = load_json_if_exists(EVALUATION_DIR / "nl2sql_blind_eval_report.json") or {}
        security = load_json_if_exists(EVALUATION_DIR / "open_sql_security_eval_report.json") or {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("盲测题数", blind.get("total"), border=True)
        c2.metric("总执行准确率", f"{blind.get('execution_accuracy', 0):.2%}", border=True)
        c3.metric("真实 LLM 调用", (blind.get("llm") or {}).get("actual_calls"), border=True)
        c4.metric("危险阻断率", f"{security.get('dangerous_block_rate', 0):.0%}", border=True)
        subset_rows = [{"子集": name, **value} for name, value in (blind.get("subset_metrics") or {}).items()]
        if subset_rows:
            st.dataframe(subset_rows, width="stretch", hide_index=True)
        st.json({"failure_categories": blind.get("failure_categories"), "latency_ms": blind.get("latency_ms"), "llm": blind.get("llm"), "rejection_codes": security.get("rejection_code_counts")})
        failures = load_json_if_exists(EVALUATION_DIR / "nl2sql_blind_failures.json") or {}
        failure_rows = failures.get("failures") or failures.get("rows") or []
        with st.expander("失败案例与分类"):
            if failure_rows:
                _render_paginated_table(failure_rows, "nl2sql_failures")
            else:
                st.info("当前失败案例文件为空。")
        st.info("模型只有候选 SQL 生成权；AST Guard、只读 SQLite authorizer、限时/限步执行器是独立安全边界。")

    with npu_tab:
        st.subheader("CPU / NPU 质量与性能")
        benchmark = load_json_if_exists(EVALUATION_DIR / "npu_operator_benchmark_report.json") or {}
        operator_rows = []
        benchmark_rows = []
        for item in benchmark.get("operator_results") or []:
            summary = item.get("summary") or {}
            inference = summary.get("model_inference") or {}
            if not str(item.get("operator", "")).endswith("_npu"):
                continue
            operator_rows.extend([
                {"算子": item.get("operator"), "设备": "CPU", "耗时（秒）": inference.get("cpu_total_model_seconds")},
                {"算子": item.get("operator"), "设备": "NPU", "耗时（秒）": inference.get("npu_same_sample_total_model_seconds")},
            ])
            benchmark_rows.append({
                "算子": item.get("operator"),
                "CPU 2,048条（秒）": inference.get("cpu_total_model_seconds"),
                "NPU 2,048条（秒）": inference.get("npu_same_sample_total_model_seconds"),
                "同样本加速比": inference.get("same_sample_speedup"),
                "NPU全量条数": inference.get("record_count"),
                "NPU全量（秒）": inference.get("npu_total_model_seconds"),
                "等价性门": (inference.get("quality_gate") or {}).get("passed"),
            })
        if operator_rows:
            operator_df = pd.DataFrame(operator_rows)
            _render_plot(px.bar(operator_df, x="算子", y="耗时（秒）", color="设备", barmode="group", title="2,048条同样本CPU/NPU独立实测"), "npu_operator_benchmark_report.json；当前为一轮独立实测")
        if benchmark_rows:
            st.dataframe(benchmark_rows, width="stretch", hide_index=True)
        st.json({"status": benchmark.get("status"), "fallback_used": benchmark.get("fallback_used"), "runtime": benchmark.get("runtime")})
        st.info("正式表格采用当前机器可读报告的一轮独立实测。不同前端运行受共享服务器负载、预热和测量时点影响可能波动，不跨轮次混算。")
        st.warning("NPU质量门验证同样本embedding、候选Top-1和保留决策等价性；端到端耗时与BGE encode加速分开报告。")

    st.divider()
    urls = artifact_urls()
    st.caption(f"更新时间：{as_of} · 数据版本：{version} · 核心数字可回溯到 SQL、MCP trace、图谱 edge_id 或正式评测 JSON。")
    st.markdown(f"[图谱入口]({urls['graph_url']}) · [图表入口]({urls['chart_index_url']}) · [报告入口]({urls['report_url']})")


if __name__ == "__main__":
    main()
