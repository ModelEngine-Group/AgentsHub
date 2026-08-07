from __future__ import annotations

from pathlib import Path
from typing import Any

from medgraph_agent.core.models import AnalysisResult, RELATION_LABELS
from medgraph_agent.core.storage import GraphStore


class GraphAnalyzer:
    def __init__(self, db_path: str | Path) -> None:
        self.store = GraphStore(db_path)

    def analyze(self, question: str) -> AnalysisResult:
        question = question.strip()
        intent, sql, chart_type, title = self._plan_sql(question)
        rows = self.store.query(sql)
        narrative = self._narrative(intent, rows)
        return AnalysisResult(
            question=question,
            intent=intent,
            sql=sql,
            rows=rows,
            chart={"type": chart_type, "title": title, "x": "name", "y": "value"},
            narrative=narrative,
            confidence=0.9 if rows else 0.55,
        )

    def _plan_sql(self, question: str) -> tuple[str, str, str, str]:
        q = question.lower()
        if any(token in question for token in ["实体类型", "各类实体", "实体分布"]):
            return (
                "entity_distribution",
                "SELECT label AS name, COUNT(*) AS value FROM entities GROUP BY label ORDER BY value DESC",
                "bar",
                "实体类型分布",
            )
        if any(token in question for token in ["检查", "诊断"]):
            return (
                "diagnosis_distribution",
                """
                SELECT subject_name || ' -> ' || object_name AS name, COUNT(*) AS value
                FROM relations
                WHERE predicate = 'diagnosed_by'
                GROUP BY subject_name, object_name
                ORDER BY value DESC, name ASC
                LIMIT 12
                """,
                "bar",
                "诊断检查关联",
            )
        if "禁忌" in question or "慎用" in question:
            return (
                "contraindication_links",
                """
                SELECT subject_name || ' -> ' || object_name AS name, COUNT(*) AS value
                FROM relations
                WHERE predicate = 'contraindicated_with'
                GROUP BY subject_name, object_name
                ORDER BY value DESC, name ASC
                """,
                "bar",
                "禁忌关系",
            )
        if any(token in question for token in ["关系类型", "关系分布", "三元组类型"]):
            return (
                "relation_distribution",
                "SELECT predicate_label AS name, COUNT(*) AS value FROM relations GROUP BY predicate_label ORDER BY value DESC",
                "bar",
                "关系类型分布",
            )
        if any(token in question for token in ["高频症状", "常见症状", "最多症状", "症状排行"]):
            return (
                "top_symptoms",
                """
                SELECT object_name AS name, COUNT(*) AS value
                FROM relations
                WHERE predicate = 'has_symptom'
                GROUP BY object_name
                ORDER BY value DESC, name ASC
                LIMIT 10
                """,
                "bar",
                "高频症状",
            )
        if any(token in question for token in ["症状最多", "每个疾病症状", "疾病症状数"]):
            return (
                "disease_symptom_count",
                """
                SELECT subject_name AS name, COUNT(DISTINCT object_name) AS value
                FROM relations
                WHERE predicate = 'has_symptom'
                GROUP BY subject_name
                ORDER BY value DESC, name ASC
                """,
                "bar",
                "疾病关联症状数",
            )
        if any(token in question for token in ["治疗", "用药", "药物"]):
            return (
                "treatment_distribution",
                """
                SELECT subject_name || ' -> ' || object_name AS name, COUNT(*) AS value
                FROM relations
                WHERE predicate = 'treated_by'
                GROUP BY subject_name, object_name
                ORDER BY value DESC, name ASC
                LIMIT 12
                """,
                "bar",
                "治疗/用药关联",
            )
        if "科室" in question:
            return (
                "department_distribution",
                """
                SELECT object_name AS name, COUNT(DISTINCT subject_name) AS value
                FROM relations
                WHERE predicate = 'belongs_to_department'
                GROUP BY object_name
                ORDER BY value DESC, name ASC
                """,
                "bar",
                "科室覆盖疾病数",
            )
        if "风险" in question or "危险因素" in question:
            return (
                "risk_factor_distribution",
                """
                SELECT object_name AS name, COUNT(DISTINCT subject_name) AS value
                FROM relations
                WHERE predicate = 'has_risk_factor'
                GROUP BY object_name
                ORDER BY value DESC, name ASC
                """,
                "bar",
                "风险因素覆盖疾病数",
            )
        if "并发" in question or "合并" in question:
            return (
                "complication_links",
                """
                SELECT subject_name || ' -> ' || object_name AS name, COUNT(*) AS value
                FROM relations
                WHERE predicate = 'complicates'
                GROUP BY subject_name, object_name
                ORDER BY value DESC, name ASC
                """,
                "bar",
                "并发关系",
            )
        if "疾病" in question and any(token in question for token in ["数量", "多少", "统计", "总数"]):
            return (
                "disease_count",
                "SELECT name, COUNT(*) AS value FROM entities WHERE type = 'disease' GROUP BY name ORDER BY name ASC",
                "table",
                "疾病清单",
            )
        if "sql" in q:
            return (
                "sql_overview",
                "SELECT predicate_label AS name, COUNT(*) AS value FROM relations GROUP BY predicate_label ORDER BY value DESC",
                "bar",
                "SQL 关系概览",
            )
        return (
            "overview",
            """
            SELECT predicate_label AS name, COUNT(*) AS value
            FROM relations
            GROUP BY predicate_label
            ORDER BY value DESC
            """,
            "bar",
            "图谱关系概览",
        )

    def _narrative(self, intent: str, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "当前图谱没有可用于该分析的问题数据。"
        top = rows[0]
        relation_hint = RELATION_LABELS.get(str(top.get("name")), str(top.get("name")))
        return (
            f"分析意图为 {intent}。结果共 {len(rows)} 行，最大项是"
            f" {relation_hint}，数值为 {top.get('value')}。"
            "该结果来自任务二生成的同一份知识图谱，保证数据处理、知识生成与洞察展示闭环一致。"
        )


def evaluate_nl2sql(db_path: str | Path, questions: list[dict[str, str]]) -> dict[str, Any]:
    analyzer = GraphAnalyzer(db_path)
    correct = 0
    details = []
    for item in questions:
        result = analyzer.analyze(item["question"])
        passed = result.intent == item["intent"]
        correct += int(passed)
        details.append({"question": item["question"], "expected": item["intent"], "actual": result.intent, "passed": passed})
    accuracy = correct / len(questions) if questions else 0.0
    return {"total": len(questions), "correct": correct, "accuracy": round(accuracy, 4), "details": details}
