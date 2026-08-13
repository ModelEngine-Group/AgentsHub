"""任务三工具集：NL2SQL 查询 / 图表构建 / 报告生成。"""

import json
import threading
from typing import Optional

from smolagents.tools import Tool

from dki.agents.tools.data_tools import _err, _notify, _ok
from dki.paths import OUTPUT_DIR

DB_PATH = OUTPUT_DIR / "insight.db"
_BUILDER_LOCK = threading.Lock()
_CURRENT_BUILDER = None  # 会话级报告构建器（make_chart 累积，save_report 落盘）


def _ensure_db():
    """数据库不存在时只从任务一 manifest + 任务二图谱构建。"""
    from dki.analysis.database import build_database
    from dki.analysis.database import database_matches_manifest
    from dki.kg.store import KnowledgeGraph
    from dki.pipeline.artifacts import (
        load_task1_manifest,
        manifest_path,
        validate_task2_artifact,
    )

    kg_path = OUTPUT_DIR / "kg" / "medical_kg.json"
    load_task1_manifest()
    validate_task2_artifact(kg_path)
    if database_matches_manifest(DB_PATH, manifest_path(), kg_path):
        return
    kg = KnowledgeGraph.load(kg_path)
    build_database(
        db_path=DB_PATH,
        manifest_path=manifest_path(),
        kg=kg,
        kg_path=kg_path,
    )


def _get_builder():
    global _CURRENT_BUILDER
    with _BUILDER_LOCK:
        if _CURRENT_BUILDER is None:
            from dki.analysis.bi import ReportBuilder

            _CURRENT_BUILDER = ReportBuilder()
        return _CURRENT_BUILDER


def _reset_builder():
    global _CURRENT_BUILDER
    with _BUILDER_LOCK:
        _CURRENT_BUILDER = None


class Nl2SqlTool(Tool):
    name = "nl2sql_query"
    description = (
        "自然语言查询就诊分析数据库（NL2SQL）。数据表：patients(患者)、visits(就诊记录：日期/科室/"
        "疾病/药物/检查/费用/住院天数)、kg_entities/kg_triples(知识图谱)。"
        "支持计数/分组/TopK/均值/趋势/图谱关系等问法。返回 JSON（sql/columns/rows）。"
    )
    inputs = {
        "question": {"type": "string", "description": "自然语言分析问题，如：按科室统计就诊量"},
    }
    output_type = "string"

    def __init__(self, observer=None):
        super().__init__()
        self.observer = observer

    def forward(self, question: str) -> str:
        try:
            _ensure_db()
            from dki.analysis.nl2sql_profile_factory import build_engine

            _notify(self.observer, f"NL2SQL: {question}")
            engine = build_engine(DB_PATH)
            res = engine.query(question)
            if res.get("error"):
                if res.get("terminal"):
                    # 引擎已判定：在保留全部限定条件的前提下无法作答。改写问法或
                    # 放宽限定只会得到一个悄悄丢条件的错误答案，因此不给重试提示。
                    return _err(
                        res["error"],
                        "这是终止性结论：请如实说明该限定条件组合无法被可靠查询，"
                        "不要删减、放宽限定条件或改写问法后重试。",
                        terminal=True,
                        disposition=res.get("disposition", "blocked"),
                    )
                return _err(res["error"], "换一种更具体的问法，如『按科室统计就诊量』『高血压的就诊次数』")
            return _ok(**res)
        except Exception as e:
            return _err(str(e), "确认已生成数据（gen-data）；或换一种问法")


class ChartTool(Tool):
    name = "make_chart"
    description = (
        "向当前分析报告添加一张 ECharts 图表。chart_type: bar/line/pie/graph。"
        'bar/line/pie 的 data 为 JSON：{"categories": [...], "series": [{"name":"...","values":[...]}]}；'
        "graph（图谱关系图）不传 data 时自动使用当前知识图谱。返回 JSON。"
    )
    inputs = {
        "chart_type": {"type": "string", "description": "bar / line / pie / graph"},
        "title": {"type": "string", "description": "图表标题（中文）"},
        "data": {"type": "string", "description": "图表数据 JSON 字符串（graph 可省略）", "nullable": True},
        "description": {"type": "string", "description": "图表下方的中文解读（2-3句，引用具体数字）", "nullable": True},
        "source_sql": {"type": "string", "description": "生成图表数据的 NL2SQL SQL；graph 可省略", "nullable": True},
        "sample_size": {"type": "integer", "description": "图表实际使用的结果行数；graph 可省略", "nullable": True},
        "data_source": {"type": "string", "description": "数据来源说明，默认任务一 manifest + 任务二图谱", "nullable": True},
    }
    output_type = "string"

    def __init__(self, observer=None):
        super().__init__()
        self.observer = observer

    def forward(
        self,
        chart_type: str,
        title: str,
        data: Optional[str] = None,
        description: Optional[str] = None,
        source_sql: Optional[str] = None,
        sample_size: Optional[int] = None,
        data_source: Optional[str] = None,
    ) -> str:
        try:
            if chart_type not in ("bar", "line", "pie", "graph"):
                return _err(f"不支持的 chart_type: {chart_type}", "可用 bar/line/pie/graph")
            builder = _get_builder()
            if chart_type == "graph" and not data:
                from dki.kg.store import KnowledgeGraph
                from dki.pipeline.artifacts import validate_task2_artifact

                kg_path = OUTPUT_DIR / "kg" / "medical_kg.json"
                validate_task2_artifact(kg_path)
                chart_data = KnowledgeGraph.load(kg_path).to_echarts_graph()
                provenance = (
                    f"来源：{data_source or 'Task 2 knowledge graph'}；"
                    f"节点数：{len(chart_data.get('nodes', []))}；"
                    f"边数：{len(chart_data.get('links', []))}；"
                    "图谱关联不构成医学因果或临床建议。"
                )
            else:
                if not data:
                    return _err("bar/line/pie 必须提供 data", "格式见工具描述")
                if not source_sql or sample_size is None:
                    return _err(
                        "分析图表必须提供 source_sql 与 sample_size",
                        "直接使用 nl2sql_query 返回的 sql 和 rows 行数",
                    )
                _ensure_db()
                from dki.analysis.database import run_sql

                _, verified_rows = run_sql(source_sql, DB_PATH)
                if int(sample_size) != len(verified_rows):
                    return _err(
                        f"sample_size={sample_size} 与 SQL 实际结果行数 "
                        f"{len(verified_rows)} 不一致",
                        "使用 nl2sql_query 返回的 rows 长度，勿手工估算",
                    )
                chart_data = json.loads(data)
                provenance = (
                    f"来源：{data_source or 'Task 1 manifest → outputs/insight.db'}；"
                    f"SQL：{source_sql}；结果行数：{int(sample_size)}；"
                    "相关不等于因果，不构成临床建议。"
                )
            builder.add_chart(
                chart_type,
                title,
                chart_data,
                description or "",
                provenance=provenance,
            )
            _notify(self.observer, f"添加图表: {title}")
            return _ok(n_charts=len(builder.charts), title=title)
        except json.JSONDecodeError as e:
            return _err(f"data 不是合法 JSON: {e}", "检查引号与括号")
        except Exception as e:
            return _err(str(e))


class SaveReportTool(Tool):
    name = "save_report"
    description = (
        "渲染并保存当前分析报告（HTML，含已添加的全部图表 + 你的分析总结）。"
        "summary_markdown 支持 Markdown（标题/列表/粗体）。返回 JSON（报告路径）。"
    )
    inputs = {
        "title": {"type": "string", "description": "报告标题"},
        "summary_markdown": {"type": "string", "description": "图文报告的文字总结（Markdown）"},
    }
    output_type = "string"

    def __init__(self, observer=None):
        super().__init__()
        self.observer = observer

    def forward(self, title: str, summary_markdown: str) -> str:
        try:
            builder = _get_builder()
            if not builder.charts:
                return _err("当前报告没有图表", "先用 make_chart 添加带 SQL 证据的图表")
            builder.add_markdown(
                summary_markdown
                + "\n\n## 分析边界\n"
                "- 报告中的关系与差异均为描述性结果，相关不等于因果，不构成临床建议。"
            )
            out = builder.render(title=title,
                                 out_path=OUTPUT_DIR / "reports" / "agent_insight_report.html")
            _reset_builder()
            _notify(self.observer, f"报告已保存: {out}")
            return _ok(path=str(out))
        except Exception as e:
            return _err(str(e), "先用 make_chart 添加至少一张图表")
