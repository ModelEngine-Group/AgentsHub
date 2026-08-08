"""Data Analysis Agent (Task 3): NL question -> data -> BI chart + insight.

Reuses Task-2 outputs: the knowledge graph (associations) and the relational DB
derived from it (statistics/trends). The agent:
  1. understands the graph schema (profile injected into the LLM),
  2. plans a route -- GRAPH (association/relationship questions) vs SQL
     (statistics / ranking / trend over records),
  3. executes (graph traversal or NL2SQL), 4. composes a natural-language insight,
  5. emits a BI/graph visualization (ECharts HTML).

This is the "数据->知识->洞察" closing loop.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from medigraph.analysis.graph_profile import build_profile, load_graph, profile_prompt_block
from medigraph.analysis.nl2sql import NL2SQL
from medigraph.analysis import viz
from medigraph.agents.qa_agent import _relation_label, infer_intent_relations, rank_evidence, select_evidence

_ROUTE_SYSTEM = "你是分析任务规划器，判断问题应走关系数据库(SQL)还是知识图谱(GRAPH)，只输出 JSON。"

_ROUTE_PROMPT = """已知两类数据源：
- SQL：就诊记录关系库（patient_visits/prescriptions/lab_tests），适合统计、排名、趋势、计数、平均等问题。
- GRAPH：医疗知识图谱（疾病-症状/药物/检查/并发/标志物等关系），适合“某疾病有哪些并发症/用药/相关概念”等关联问题。

{profile}

请判断下面的问题应使用哪个数据源，并给出理由。只输出 JSON：{{"route":"SQL"或"GRAPH","reason":"..."}}
问题：{question}"""

_INSIGHT_SYSTEM = "你是医疗数据分析师，根据数据给出简洁、专业、可读的中文洞察，不要编造数据。"

_INSIGHT_PROMPT = """问题：{question}

分析得到的数据（{source}）：
{data}

请用 2-4 句话给出洞察解读，指出关键发现/最大值/趋势/异常，便于非技术读者理解。"""

_SQL_HINTS = (
    "统计", "多少", "人次", "数量", "计数", "平均", "总计", "总共", "排名",
    "排序", "最多", "最少", "最高", "最低", "前", "趋势", "每月", "每年",
    "比例", "占比", "分布", "费用", "异常率", "top",
)
_GRAPH_HINTS = (
    "关系", "关联", "症状", "并发症", "推荐药", "用药", "检查", "标志物",
    "基因", "形态", "位于", "亚型", "知识图谱", "邻居",
)

# Graph-analytics intents: global structure (who's central, what clusters
# together, how two entities relate) rather than "what is X directly linked
# to" -- the existing GRAPH route only ever does 1-hop neighbor lookup around
# a named anchor entity, which cannot answer any of these.
_CENTRALITY_HINTS = ("中心性", "核心节点", "最核心", "最重要的", "pagerank", "枢纽", "hub")
_COMMUNITY_HINTS = ("社区", "聚类", "簇", "社群", "分组", "community")
_PATH_HINTS = ("最短路径", "怎么关联", "如何关联", "路径是什么", "shortest path")


class AnalysisAgent:
    def __init__(self, db_path: str, graph_json: str | None = None, llm: Any | None = None):
        if llm is None:
            from medigraph.llm.client import LLMClient
            llm = LLMClient()
        self.llm = llm
        self.store, self.used_example_graph = load_graph(graph_json)
        self.profile = build_profile(self.store)
        self.nl2sql = NL2SQL(db_path, llm=llm)

    # ------------------------------------------------------------------ #
    def plan_route(self, question: str) -> dict:
        """Return an explainable route plan; use deterministic intents first."""
        q = question.lower()
        algo_hits = [
            hint
            for hint in (*_CENTRALITY_HINTS, *_COMMUNITY_HINTS, *_PATH_HINTS)
            if hint.lower() in q
        ]
        if algo_hits:
            return {
                "route": "GRAPH_ALGO",
                "reason": f"问题包含图算法意图：{', '.join(algo_hits[:4])}",
                "planner": "deterministic_intent_router",
            }
        sql_hits = [hint for hint in _SQL_HINTS if hint.lower() in q]
        graph_hits = [hint for hint in _GRAPH_HINTS if hint.lower() in q]
        # Aggregations over the KG mirror tables are SQL tasks even when the
        # wording contains "知识图谱/关联/关系".  The previous rule sent
        # "统计关联症状最多的 Top 10 疾病" to entity-centric graph traversal,
        # which has no anchor entity and therefore returned zero rows.
        if sql_hits:
            return {
                "route": "SQL",
                "reason": f"问题包含统计/趋势意图：{', '.join(sql_hits[:4])}",
                "planner": "deterministic_intent_router",
            }
        if graph_hits:
            return {
                "route": "GRAPH",
                "reason": f"问题包含图谱关联意图：{', '.join(graph_hits[:4])}",
                "planner": "deterministic_intent_router",
            }
        data = self.llm.chat_json(
            _ROUTE_PROMPT.format(profile=profile_prompt_block(self.profile), question=question),
            system=_ROUTE_SYSTEM, default={"route": "SQL"},
        )
        route = (data.get("route") if isinstance(data, dict) else "SQL") or "SQL"
        normalized = "GRAPH" if str(route).upper().startswith("G") else "SQL"
        return {
            "route": normalized,
            "reason": str(data.get("reason", "LLM 根据数据源能力选择")) if isinstance(data, dict) else "LLM 路由",
            "planner": "llm_router",
        }

    def route(self, question: str) -> str:
        return self.plan_route(question)["route"]

    def _graph_answer(self, question: str) -> dict:
        """Association analysis via graph traversal around question entities."""
        triples: list[dict] = []
        intent_relations = infer_intent_relations(question)
        compact_question = question.lower().replace("—", "-").replace("－", "-")
        generic_relation_network = (
            "疾病" in compact_question
            and "症状" in compact_question
            and any(term in compact_question for term in ("关系图", "关联图", "网络图", "关系网络", "疾病-症状"))
        )

        if generic_relation_network and hasattr(self.store, "g"):
            # A relation-network request has no named anchor entity.  Traverse a
            # representative slice of the requested relation instead of asking
            # NER to turn the generic words "疾病/症状" into fake anchors.
            grouped: dict[str, list[dict]] = {}
            for head, tail, data in self.store.g.edges(data=True):
                if data.get("relation") != "has_symptom":
                    continue
                if self.store.g.nodes[head].get("type") != "Disease":
                    continue
                if self.store.g.nodes[tail].get("type") != "Symptom":
                    continue
                grouped.setdefault(head, []).append(self.store._edge_dict(head, tail, data))
            ranked_heads = sorted(grouped, key=lambda name: (-len(grouped[name]), name))[:8]
            anchors = ranked_heads
            for head in ranked_heads:
                triples.extend(
                    sorted(grouped[head], key=lambda item: str(item.get("tail", "")))[:3]
                )
            evidence_total = sum(len(items) for items in grouped.values())
            triples = triples[:24]
        else:
            from medigraph.operators.base import get_operator, load_default_operators
            load_default_operators(llm=self.llm)
            ents = get_operator("medical_ner").run({"text": question}).get("entities", [])
            anchors = self.store.find_entities([e["name"] for e in ents])
            if not anchors:
                anchors = self.store.find_entities([question])
            seen = set()
            for a in anchors:
                for t in self.store.neighbors(a, hops=1):
                    k = (t["head"], t["relation"], t["tail"])
                    if k not in seen:
                        seen.add(k)
                        triples.append(t)
            evidence_total = len(triples)
            triples = select_evidence(
                rank_evidence(triples, intent_relations, anchors=anchors),
                intent_relations,
                anchors=anchors,
            )
            triples = [
                triple for triple in triples
                if str(triple.get("head", "")) != str(triple.get("tail", ""))
            ]
        columns = ["head", "relation", "tail"]
        rows = [(t["head"], _relation_label(t), t["tail"]) for t in triples]
        citations = [
            {
                "id": index,
                "triple": f"{t['head']} --[{_relation_label(t)}]--> {t['tail']}",
                "relation": t["relation"],
                "confidence": t.get("confidence"),
                "source": t.get("source", ""),
            }
            for index, t in enumerate(triples[:24], start=1)
        ]
        return {"source": "GRAPH", "columns": columns, "rows": rows, "triples": triples,
                "anchors": anchors, "sql": "", "attempts": [],
                "intent_relations": intent_relations, "evidence_total": evidence_total,
                "evidence_used": len(triples), "citations": citations}

    def _graph_algorithm_answer(self, question: str) -> dict:
        """Global graph structure: centrality/PageRank ranking, community
        detection, or shortest path -- as opposed to `_graph_answer`'s 1-hop
        neighbor lookup around a named anchor."""
        from medigraph.analysis.graph_algorithms import (
            degree_centrality_ranking,
            detect_communities,
            pagerank_ranking,
            shortest_path,
        )

        q = question.lower()
        entity_type = None
        for label, type_name in (
            ("疾病", "Disease"), ("症状", "Symptom"), ("药物", "Drug"),
            ("检查", "Examination"), ("部位", "Body"),
        ):
            if label in question:
                entity_type = type_name
                break

        if any(hint.lower() in q for hint in _PATH_HINTS):
            from medigraph.operators.base import get_operator, load_default_operators

            load_default_operators(llm=self.llm)
            ents = get_operator("medical_ner").run({"text": question}).get("entities", [])
            anchors = self.store.find_entities([e["name"] for e in ents])
            if len(anchors) >= 2:
                path = shortest_path(self.store, anchors[0], anchors[1])
            else:
                path = None
            if not path:
                return {
                    "source": "GRAPH_ALGO", "algorithm": "shortest_path", "columns": ["head", "relation", "tail"],
                    "rows": [], "triples": [], "anchors": anchors, "sql": "", "attempts": [],
                    "intent_relations": [], "evidence_total": 0, "evidence_used": 0, "citations": [],
                }
            triples = [
                {"head": step["head"], "relation": step["relation"], "tail": step["tail"],
                 "confidence": step.get("confidence")}
                for step in path["steps"]
            ]
            rows = [(t["head"], _relation_label(t), t["tail"]) for t in triples]
            return {
                "source": "GRAPH_ALGO", "algorithm": "shortest_path", "columns": ["head", "relation", "tail"],
                "rows": rows, "triples": triples, "anchors": path["path"], "sql": "", "attempts": [],
                "intent_relations": [], "evidence_total": len(triples), "evidence_used": len(triples),
                "citations": [{"id": i, "triple": f"{t['head']} --[{_relation_label(t)}]--> {t['tail']}",
                               "relation": t["relation"], "confidence": t.get("confidence")}
                              for i, t in enumerate(triples, start=1)],
            }

        if any(hint.lower() in q for hint in _COMMUNITY_HINTS):
            communities = detect_communities(self.store, top_n=8)
            columns = ["community_id", "size", "representative_entities"]
            rows = [(c["community_id"], c["size"], "、".join(c["representative_entities"])) for c in communities]
            return {
                "source": "GRAPH_ALGO", "algorithm": "community", "columns": columns, "rows": rows,
                "triples": None, "anchors": [], "sql": "", "attempts": [], "intent_relations": [],
                "evidence_total": len(communities), "evidence_used": len(communities), "citations": [],
                "communities": communities,
            }

        # Centrality/PageRank (default for this route when no community/path
        # keyword matched -- _CENTRALITY_HINTS is what triggered GRAPH_ALGO).
        use_pagerank = "pagerank" in q or "枢纽" in q or "hub" in q
        ranking = (
            pagerank_ranking(self.store, top_n=10, entity_type=entity_type)
            if use_pagerank
            else degree_centrality_ranking(self.store, top_n=10, entity_type=entity_type)
        )
        score_key = "pagerank" if use_pagerank else "degree_centrality"
        columns = ["name", "type", score_key]
        rows = [(item["name"], item["type"], item[score_key]) for item in ranking]
        return {
            "source": "GRAPH_ALGO", "algorithm": score_key, "columns": columns, "rows": rows,
            "triples": None, "anchors": [], "sql": "", "attempts": [], "intent_relations": [],
            "evidence_total": len(ranking), "evidence_used": len(ranking), "citations": [],
        }

    def _sql_answer(self, question: str) -> dict:
        res = self.nl2sql.query(question)
        return {"source": "SQL", "columns": res["columns"], "rows": res["rows"],
                "sql": res["sql"], "error": res["error"], "triples": None,
                "attempts": res["attempts"], "anchors": [], "intent_relations": [],
                "evidence_total": 0, "evidence_used": 0, "citations": []}

    def _insight(self, question: str, source: str, columns: list[str], rows: list) -> str:
        if source == "GRAPH" and "并发" in question:
            complications = list(dict.fromkeys(
                str(row[2]) for row in rows
                if len(row) > 2 and str(row[1]) in {"并发", "complication"}
            ))
            drugs = list(dict.fromkeys(
                str(row[2]) for row in rows
                if len(row) > 2 and str(row[1]) in {"推荐药物", "recommend_drug"}
            ))
            findings = []
            if complications:
                findings.append(
                    f"当前图谱检索到 {len(complications)} 个直接并发症："
                    f"{'、'.join(complications)}"
                )
            if drugs:
                findings.append(
                    f"推荐药物共 {len(drugs)} 种：{'、'.join(drugs)}"
                )
            if findings:
                return (
                    "；".join(findings)
                    + "。并发症结论仅采用从问题疾病出发的有向关系，"
                    "未把指向该疾病的入边计作其并发症。"
                )
        preview: dict[str, Any] = {
            "columns": columns,
            "row_count": len(rows),
            "rows_preview": [list(r) for r in rows[:30]],
        }
        if len(columns) >= 2:
            numeric_rows = [
                (r[0], float(r[1]))
                for r in rows
                if len(r) >= 2 and isinstance(r[1], (int, float))
            ]
            if numeric_rows:
                ranked = sorted(numeric_rows, key=lambda item: item[1], reverse=True)
                preview["second_column_statistics"] = {
                    "column": columns[1],
                    "sum": sum(value for _, value in numeric_rows),
                    "max": list(ranked[0]),
                    "min": list(ranked[-1]),
                    "top5": [list(item) for item in ranked[:5]],
                }
        try:
            return self.llm.chat(
                _INSIGHT_PROMPT.format(question=question, source=source, data=preview),
                system=_INSIGHT_SYSTEM, temperature=0.2,
            )
        except Exception:
            # Reporting must not fail merely because an optional narrative LLM
            # is unavailable or out of quota.  The queried rows are already the
            # source of truth, so produce a compact deterministic explanation.
            numeric = preview.get("second_column_statistics")
            if numeric:
                maximum = numeric["max"]
                minimum = numeric["min"]
                return (
                    f"共得到 {len(rows)} 条结果。最高项为“{maximum[0]}”"
                    f"（{maximum[1]:g}），最低项为“{minimum[0]}”"
                    f"（{minimum[1]:g}）；图表与明细均来自本次只读查询。"
                )
            unique_heads = len({str(row[0]) for row in rows if row})
            unique_tails = len({str(row[2]) for row in rows if len(row) > 2})
            return (
                f"共展示 {len(rows)} 条知识图谱关系，涉及 {unique_heads} 个疾病"
                f"和 {unique_tails} 个症状；所有连边均来自当前图谱证据。"
            )

    # ------------------------------------------------------------------ #
    def analyze(self, question: str, out_html: str | Path | None = None, verbose: bool = True) -> dict:
        plan = self.plan_route(question)
        route = plan["route"]
        if verbose:
            print(f"[AnalysisAgent] route = {route}")
        if route == "GRAPH_ALGO":
            result = self._graph_algorithm_answer(question)
        elif route == "GRAPH":
            result = self._graph_answer(question)
        else:
            result = self._sql_answer(question)

        if not result["rows"]:
            insight = "未查询到相关数据，无法分析。" + (f"（SQL 错误：{result.get('error')}）" if result.get("error") else "")
            chart_type = "table"
        else:
            insight = self._insight(question, result["source"], result["columns"], result["rows"])
            if route == "GRAPH_ALGO":
                chart_type = "graph" if result.get("algorithm") == "shortest_path" else "bar"
            else:
                chart_type = "graph" if route == "GRAPH" else viz.pick_chart_type(question, result["columns"], result["rows"])

        html_path = ""
        if out_html:
            html_path = viz.render_report(
                out_html, question, chart_type, result["columns"], result["rows"],
                insight, sql=result.get("sql", ""), triples=result.get("triples"),
            )

        return {
            "question": question, "route": route, "chart_type": chart_type,
            "route_reason": plan["reason"], "planner": plan["planner"],
            "columns": result["columns"], "rows": result["rows"],
            "sql": result.get("sql", ""), "insight": insight, "html": html_path,
            "attempts": result.get("attempts", []),
            "anchors": result.get("anchors", []),
            "intent_relations": result.get("intent_relations", []),
            "evidence_total": result.get("evidence_total", 0),
            "evidence_used": result.get("evidence_used", 0),
            "citations": result.get("citations", []),
            "graph_profile": self.profile,
            "used_example_graph": self.used_example_graph,
        }
