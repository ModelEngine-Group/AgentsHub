"""BI 报告生成：ECharts 单文件 HTML（图文结合）。"""

import json
import html
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment
from markupsafe import Markup

from dki.paths import OUTPUT_DIR

_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "report.html.j2"
_ECHARTS_PATH = Path(__file__).resolve().parent / "static" / "echarts.min.js"


def _md_to_html(md: str) -> str:
    """极简 Markdown 转 HTML（标题/无序列表/粗体/段落，够报告使用，零依赖）。"""
    lines = md.split("\n")
    out: List[str] = []
    in_list = False
    for line in lines:
        s = line.rstrip()
        if s.startswith("### "):
            if in_list:
                out.append("</ul>"); in_list = False
            out.append(f"<h3>{html.escape(s[4:])}</h3>")
        elif s.startswith("## "):
            if in_list:
                out.append("</ul>"); in_list = False
            out.append(f"<h2>{html.escape(s[3:])}</h2>")
        elif s.startswith("# "):
            if in_list:
                out.append("</ul>"); in_list = False
            out.append(f"<h2>{html.escape(s[2:])}</h2>")
        elif s.startswith(("- ", "* ")):
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append(f"<li>{html.escape(s[2:])}</li>")
        elif not s:
            if in_list:
                out.append("</ul>"); in_list = False
        else:
            if in_list:
                out.append("</ul>"); in_list = False
            out.append(f"<p>{html.escape(s)}</p>")
    if in_list:
        out.append("</ul>")
    rendered = "\n".join(out)
    rendered = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", rendered)
    return rendered


def _json_for_script(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


class ReportBuilder:
    """累积图表与文字，最终渲染单文件 HTML 报告。"""

    def __init__(self):
        self.charts: List[Dict[str, Any]] = []
        self.markdown_parts: List[str] = []
        self.overview: Dict[str, Any] = {}

    def set_overview(self, stats: Dict[str, Any]):
        self.overview = stats

    def add_chart(self, chart_type: str, title: str, data: Dict[str, Any], description: str = "",
                  provenance: str = ""):
        assert chart_type in ("bar", "line", "pie", "graph", "heatmap", "sankey"), \
            f"未知图表类型 {chart_type}"
        self.charts.append({
            "type": chart_type, "title": title, "data": data, "description": description,
            "provenance": provenance,
        })

    def add_markdown(self, md: str):
        self.markdown_parts.append(md)

    def render(self, title: str, out_path: Optional[Path] = None) -> Path:
        out_path = Path(out_path or (OUTPUT_DIR / "reports" / "insight_report.html"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        template = Environment(autoescape=True).from_string(
            _TEMPLATE_PATH.read_text(encoding="utf-8")
        )
        html = template.render(
            title=title,
            generated_at=time.strftime("%Y-%m-%d %H:%M"),
            overview=self.overview,
            charts=self.charts,
            charts_json=Markup(_json_for_script(self.charts)),
            echarts_js=Markup(
                _ECHARTS_PATH.read_text(encoding="utf-8")
                if _ECHARTS_PATH.is_file()
                else ""
            ),
            summary_html=Markup(
                _md_to_html("\n\n".join(self.markdown_parts))
                if self.markdown_parts
                else ""
            ),
        )
        out_path.write_text(html, encoding="utf-8")
        return out_path


def _rows_to_chart_data(columns: List[str], rows: List[List[Any]], value_name: str = "数量") -> Dict[str, Any]:
    return {
        "categories": [str(r[0]) for r in rows],
        "series": [{"name": value_name, "values": [r[1] for r in rows]}],
    }


def build_default_report(db_path: Path, kg=None, out_path: Optional[Path] = None) -> Path:
    """无 LLM 的完整示例报告：关键洞察 + 标准指标 + 热力/桑基/图谱关系图 + 图谱驱动关联。"""
    from dki.analysis.metrics import (
        age_disease_heatmap, disease_department_sankey, graph_associations,
        key_insights, overview_stats, standard_metrics,
    )

    builder = ReportBuilder()
    overview = overview_stats(db_path)
    builder.set_overview(overview)
    metrics = standard_metrics(db_path)

    # 置顶「关键洞察 Top N」——数据驱动自动萃取
    insights = key_insights(db_path, metrics, overview)
    if insights:
        builder.add_markdown("## 关键洞察\n" + "\n".join(f"- {s}" for s in insights))

    descriptions = {
        "dept_visits": "{top}以 {topv} 次就诊居各科室之首，占总就诊量的 {pct}%。科室工作量分布可为医疗资源调配提供依据。",
        "top_diseases": "就诊量最高的疾病为{top}（{topv} 次）。高频疾病是慢病管理与健康宣教的重点对象。",
        "monthly_trend": "近 24 个月共 {total} 次就诊，月均约 {avg} 次。趋势曲线可用于监测季节性波动与容量规划。",
        "dept_avg_cost": "{top}的平均费用最高（{topv} 元）。这是描述性差异，不足以推断费用差异的原因。",
        "age_group": "就诊人群以{top}为主（{topv} 次，占 {pct}%），提示该年龄段是主要服务对象。",
        "gender_dist": "各性别就诊量为：{pairs}。",
        "top_drugs": "处方频次最高的药物为{top}（{topv} 次），与高频疾病谱相互印证。",
        "kg_core_entities": "知识图谱中提及最多的实体是{top}（{topv} 次），是图谱中的枢纽节点。",
    }

    total_visits = builder.overview.get("就诊记录", 0) or 1
    for m in metrics:
        rows = m["rows"]
        if not rows:
            continue
        top, topv = rows[0][0], rows[0][1]
        tpl = descriptions.get(m["key"], "")
        desc = ""
        if tpl:
            try:
                desc = tpl.format(
                    top=top, topv=topv,
                    pct=round(100 * (topv / total_visits), 1) if isinstance(topv, (int, float)) else "-",
                    total=sum(r[1] for r in rows if isinstance(r[1], (int, float))),
                    avg=round(sum(r[1] for r in rows) / max(1, len(rows)), 1),
                    pairs="、".join(f"{r[0]} {r[1]} 次" for r in rows),
                )
            except Exception:
                desc = ""
        builder.add_chart(
            m["chart_type"], m["title"], _rows_to_chart_data(m["columns"], rows), desc,
            provenance=(
                f"来源：Task 1 manifest → outputs/insight.db；SQL：{m['sql']}；"
                f"结果行数：{len(rows)}；相关不等于因果，不构成临床建议。"
            ),
        )

    # 热力图：年龄段 × Top 疾病 就诊分布（二维交叉分析）
    heat = age_disease_heatmap(db_path)
    if heat and heat.get("data"):
        peak = max(heat["data"], key=lambda d: d[2]) if heat["data"] else None
        peak_desc = ""
        if peak:
            peak_desc = (f"就诊最集中的组合是「{heat['ycats'][peak[1]]}岁 × "
                         f"{heat['xcats'][peak[0]]}」（{peak[2]} 次）。")
        builder.add_chart("heatmap", "年龄段 × 疾病 就诊热力分布", heat,
                          "颜色越深表示该年龄段—疾病组合的就诊越集中。" + peak_desc,
                          provenance=(
                              "来源：Task 1 manifest → outputs/insight.db；"
                              f"SQL：{heat['sql']}；结果行数：{heat['result_rows']}；"
                              "关联为描述性统计。"
                          ))

    # 桑基图：疾病 → 科室 就诊流向
    sankey = disease_department_sankey(db_path)
    if sankey and sankey.get("links"):
        builder.add_chart("sankey", "疾病 → 科室 就诊流向", sankey,
                          "连线宽度表示该疾病流向该科室的就诊次数，可见跨科室诊疗的分布格局。",
                          provenance=(
                              "来源：Task 1 manifest → outputs/insight.db；"
                              f"SQL：{sankey['sql']}；结果行数：{sankey['result_rows']}；"
                              "关联不代表因果或临床推荐。"
                          ))

    if kg is not None:
        stats = kg.stats()
        builder.add_chart(
            "graph", "医疗知识图谱（核心子图）", kg.to_echarts_graph(max_nodes=90),
            f"图谱共 {stats['n_entities']} 个实体、{stats['n_triples']} 条三元组，"
            f"覆盖 {len(stats['triples_by_relation'])} 类关系，平均置信度 {stats['avg_confidence']}。"
            f"可拖拽节点查看关联，箭头方向为关系方向。",
            provenance="来源：Task 2 knowledge graph；边仅代表抽取证据与置信度，不构成医学因果结论。"
        )

    # 图谱驱动的关联洞察（跨表 JOIN 图谱三元组）
    assoc = graph_associations(db_path)
    if assoc:
        lines = ["## 图谱驱动的关联洞察"]
        for block in assoc:
            lines.append(f"### {block['title']}")
            for row in block["rows"][:6]:
                lines.append(f"- **{row[0]}**：{row[1]}")
            lines.append(
                f"- 证据：SQL `{block['sql']}`；结果行数 {block['result_rows']}；"
                "来源 Task 2 knowledge graph。"
            )
        builder.add_markdown("\n".join(lines))

    builder.add_markdown(
        "## 分析说明\n"
        "- 本报告由数据分析智能体基于任务一（结构化就诊数据）与任务二（医疗知识图谱）的产出自动生成；\n"
        "- 图表数据均来自 SQLite 分析库（outputs/insight.db），可用 NL2SQL 自然语言复查任一数字；\n"
        "- 每张图都给出来源和 SQL；除非另有实验设计，图中关联均不构成因果解释或临床建议；\n"
        "- 知识图谱关系图支持交互探索（缩放/拖拽/邻接高亮）；\n"
        "- **数据为确定性种子合成语料，仅用于演示「数据→知识→洞察」链路能力。**"
    )
    return builder.render(title="医疗数据洞察报告", out_path=out_path)
