from __future__ import annotations

from html import escape
from typing import Any

from visualization.html_utils import html_page, render_table


def build_markdown_report(
    insights: dict[str, Any],
    indicator_doc: dict[str, Any],
    nl2sql_eval: dict[str, Any],
    graph_summary: dict[str, Any],
    kg_quality: dict[str, Any],
    chart_index_path: str,
    sqlite_report: dict[str, Any],
) -> str:
    lines = [
        "# ChronicCare-Agent分析报告",
        "",
        "## 1. 项目简介",
        "ChronicCare-Agent 面向三高慢病随访场景，当前已完成数据处理、知识图谱、规则型 NL2SQL 分析和图表化展示的本地闭环。",
        "",
        "## 2. 数据处理闭环",
        f"- SQLite patient_profile: {sqlite_report['tables']['patient_profile']}",
        f"- SQLite visit_record: {sqlite_report['tables']['visit_record']}",
        f"- SQLite lab_result: {sqlite_report['tables']['lab_result']}",
        f"- SQLite medication_record: {sqlite_report['tables']['medication_record']}",
        "",
        "## 3. 知识图谱概览",
        f"- 图谱节点数: {graph_summary['node_count']}",
        f"- 图谱边数: {graph_summary['edge_count']}",
        f"- 已知 rejected triples: {graph_summary.get('rejected_triples_count', 0)}",
        "",
        "## 4. NL2SQL 分析能力",
        f"- 分析问题数: {len(indicator_doc['items'])}",
        f"- SQL 生成成功率: {nl2sql_eval['sql_generation_success_rate']}",
        f"- SQL 可执行率: {nl2sql_eval['sql_executable_rate']}",
        f"- 结果成功率: {nl2sql_eval['result_success_rate']}",
        "- 当前采用模板与LLM候选相结合的受控Open SQL，所有候选SQL均经过SQL Guard。",
        "",
        "## 5. 核心指标与图表",
    ]
    for item in insights["analysis_items"]:
        lines.extend(
            [
                f"### {item['id']} {item['title']}",
                f"- 问题: {item['question']}",
                f"- 图表类型: {item['chart_type']}",
                f"- 图表路径: {item['chart_path']}",
                f"- 洞察: {item['insight']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 6. 工程边界",
            "当前系统采用规则抽取与BGE模型增强相结合的实体关系处理，并通过受控NL2SQL提供只读分析。",
            "不符合schema的数据保留在rejected triples中，不进入正式知识图谱。",
            "",
            "## 7. 医疗安全说明",
            insights["safety_note"],
            "",
            f"图表索引: {chart_index_path}",
        ]
    )
    return "\n".join(lines)


def markdown_to_html(markdown_text: str, title: str) -> str:
    blocks: list[str] = []
    for line in markdown_text.splitlines():
        if line.startswith("# "):
            blocks.append(f"<h1>{escape(line[2:])}</h1>")
        elif line.startswith("## "):
            blocks.append(f"<h2>{escape(line[3:])}</h2>")
        elif line.startswith("### "):
            blocks.append(f"<h3>{escape(line[4:])}</h3>")
        elif line.startswith("- "):
            blocks.append(f"<p>{escape(line)}</p>")
        elif line.strip():
            blocks.append(f"<p>{escape(line)}</p>")
    return html_page(title, "".join(f'<div class="card">{block}</div>' if block.startswith("<h") else block for block in blocks))


def build_index_html(entries: list[dict[str, Any]], safety_note: str) -> str:
    rows = [
        {
            "title": entry["title"],
            "question": entry["question"],
            "chart_type": entry["chart_type"],
            "path": entry["path"],
        }
        for entry in entries
    ]
    body = f"""
    <div class="card">
      <h1>ChronicCare-Agent图表索引</h1>
      <p>以下内容汇总当前指标图、图谱统计图和分析图表。</p>
      {render_table(["title", "question", "chart_type", "path"], rows)}
      <div class="safety">{escape(safety_note)}</div>
    </div>
    """
    return html_page("Chart Index", body)
