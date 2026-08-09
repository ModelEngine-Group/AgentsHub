"""任务三：数据分析智能体与图谱驱动可视化。

复用任务一/二产出：
- 任务二生成的知识图谱（JSON）→ 导出 SQLite → NL2SQL 分析；
- 图谱结构感知：自动读取实体/关系分布，规划统计与关联分析；
- 输出：BI 图表（柱状/饼图）+ 图谱子图 + 图文结合 HTML 报告。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .base import BaseAgent, Step
from ..analysis.nl2sql import NL2SQL
from ..analysis import viz
from ..kg import MedicalKG, RELATION_TYPES
from ..kg.schema import ENTITY_TYPES


class AnalysisAgent(BaseAgent):
    """数据分析智能体：图谱感知 → NL2SQL/统计分析 → BI 可视化报告。"""

    name = "analysis_agent"

    def __init__(self, kg_path: str | Path, db_path: str | Path) -> None:
        super().__init__()
        self.kg = MedicalKG.load(kg_path)
        self.db_path = Path(db_path)
        self.kg.to_sqlite(self.db_path)
        self.nl2sql = NL2SQL(self.db_path)

        self.register_tool("graph_overview", self._graph_overview, "图谱结构感知：实体/关系分布统计")
        self.register_tool("nl_query", self._nl_query, "自然语言分析查询（NL2SQL）")
        self.register_tool("relation_ranking", self._relation_ranking, "关联分析：关系数量 TopN 实体排行")
        self.register_tool("subgraph", self._subgraph, "以指定实体为中心绘制知识子图")
        self.register_tool("coverage_heatmap", self._coverage_heatmap,
                           "覆盖分析：实体类型 × 关系类型 热力图")
        self.register_tool("complication_network", self._complication_network,
                           "并发症关系网络（多节点关系图）")
        self.register_tool("render_report", self._render_report, "生成图文结合的 HTML 分析报告")

    # ---- 工具 -----------------------------------------------------------

    def _graph_overview(self, state: dict | None, **params: Any) -> tuple[dict, str]:
        state = state or {"sections": []}
        stats = self.kg.stats()
        et = stats["entity_types"]
        rt = stats["relation_types"]
        et_zh = {ENTITY_TYPES.get(k, k): v for k, v in et.items()}
        rt_zh = {RELATION_TYPES.get(k, ("", "", k))[2]: v for k, v in rt.items()}
        state["stats"] = stats
        state["sections"].append({
            "heading": "图谱总览",
            "text": (f"知识图谱共 <b>{stats['num_entities']}</b> 个实体、"
                     f"<b>{stats['num_relations']}</b> 条关系；"
                     f"实体类型 {len(et)} 种，关系类型 {len(rt)} 种。"),
            "image_b64": viz.bar_chart(list(et_zh), list(et_zh.values()), "实体类型分布"),
        })
        state["sections"].append({
            "heading": "关系类型分布",
            "text": "各类语义关系在图谱中的数量分布，反映语料对不同医疗知识维度的覆盖。",
            "image_b64": viz.pie_chart(list(rt_zh), list(rt_zh.values()), "关系类型占比"),
        })
        return state, f"图谱感知: {stats['num_entities']} 实体 / {stats['num_relations']} 关系"

    def _nl_query(self, state: dict | None, question: str = "", **params: Any) -> tuple[dict, str]:
        state = state or {"sections": []}
        res = self.nl2sql.query(question)
        state["sections"].append({
            "heading": f"分析查询：{question}",
            "text": f"NL2SQL 生成语句：<code>{res['sql']}</code>，返回 {len(res['rows'])} 行。",
            "table": {"columns": res["columns"], "rows": res["rows"][:20]},
        })
        return state, f"NL2SQL: {question} -> {len(res['rows'])} 行"

    def _relation_ranking(self, state: dict | None, relation: str = "HAS_SYMPTOM",
                          top_n: int = 8, **params: Any) -> tuple[dict, str]:
        state = state or {"sections": []}
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT head, COUNT(*) c FROM relations WHERE relation=? GROUP BY head ORDER BY c DESC LIMIT ?",
            (relation, top_n)).fetchall()
        conn.close()
        rel_zh = RELATION_TYPES[relation][2]
        state["sections"].append({
            "heading": f"关联分析：{rel_zh} TopN",
            "text": f"按「{rel_zh}」关系数量排序的实体排行，可用于识别语料中的重点疾病。",
            "image_b64": viz.bar_chart([r[0] for r in rows], [r[1] for r in rows],
                                       f"{rel_zh} 关系数量 Top{top_n}"),
        })
        return state, f"关联分析 {relation}: top{top_n}"

    def _subgraph(self, state: dict | None, entity: str = "", **params: Any) -> tuple[dict, str]:
        state = state or {"sections": []}
        state["sections"].append({
            "heading": f"知识子图：{entity}",
            "text": f"以「{entity}」为中心的一跳知识子图，直观展示其症状、用药、检查与并发关系。",
            "image_b64": viz.subgraph_chart(self.kg, entity),
        })
        return state, f"子图: {entity}"

    def _coverage_heatmap(self, state: dict | None, **params: Any) -> tuple[dict, str]:
        state = state or {"sections": []}
        mat, rows, cols = viz.coverage_matrix(self.kg)
        state["sections"].append({
            "heading": "覆盖分析：实体类型 × 关系类型",
            "text": "热力图展示各类头实体在不同语义关系上的边数分布，"
                    "颜色越深覆盖越密，可评估图谱知识维度的完整性与偏斜。",
            "image_b64": viz.heatmap_chart(mat, rows, cols, "实体类型 × 关系类型 覆盖热力图"),
        })
        return state, f"覆盖热力图: {len(rows)}×{len(cols)}"

    def _complication_network(self, state: dict | None, relation: str = "COMPLICATION_OF",
                              **params: Any) -> tuple[dict, str]:
        state = state or {"sections": []}
        rel_zh = RELATION_TYPES.get(relation, ("", "", relation))[2]
        state["sections"].append({
            "heading": f"关系网络：{rel_zh}",
            "text": f"由「{rel_zh}」关系构成的多节点网络，节点大小反映其关联度（度中心性），"
                    "可用于识别疾病并发的枢纽节点。",
            "image_b64": viz.network_chart(self.kg, relation),
        })
        return state, f"关系网络: {relation}"

    def _render_report(self, state: dict | None, out_path: str = "output/analysis_report.html",
                       title: str = "医疗知识图谱分析报告", **params: Any) -> tuple[dict, str]:
        state = state or {"sections": []}
        path = viz.render_report(title, state["sections"], out_path)
        state["report_path"] = str(path)
        return state, f"报告已生成 -> {path}"

    # ---- 规划 -----------------------------------------------------------

    def rule_plan(self, task: str) -> list[Step]:
        plan = [Step(tool="graph_overview", reason="先感知图谱结构")]
        entity = next((e for e in self.kg.entities() if e in task), None)
        plan.append(Step(tool="coverage_heatmap", reason="评估图谱知识维度覆盖"))
        plan.append(Step(tool="relation_ranking", params={"relation": "HAS_SYMPTOM"},
                         reason="疾病-症状关联分析"))
        plan.append(Step(tool="relation_ranking", params={"relation": "TREATED_BY_DRUG"},
                         reason="疾病-用药关联分析"))
        plan.append(Step(tool="complication_network", reason="并发症关系网络分析"))
        if entity:
            plan.append(Step(tool="subgraph", params={"entity": entity}, reason="任务提及该实体"))
            plan.append(Step(tool="nl_query", params={"question": f"{entity}有哪些症状"},
                             reason="实体明细查询"))
        plan.append(Step(tool="render_report", reason="输出图文报告"))
        return plan

    def analyze(self, task: str, out_path: str = "output/analysis_report.html"):
        plan, planner = self.make_plan(task)
        if not any(s.tool == "render_report" for s in plan):
            plan.append(Step(tool="render_report", reason="输出图文报告"))
        for s in plan:
            if s.tool == "render_report":
                s.params.setdefault("out_path", out_path)
        from .base import AgentRun

        run = AgentRun(task=task, plan=plan, planner=planner)
        state: Any = None
        for step in plan:
            record = self._execute_step(step, state)
            run.records.append(record)
            if record.status in ("ok", "retried"):
                state = record.summary_state
            else:
                break
        run.result = state
        return run
