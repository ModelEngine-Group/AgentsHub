"""Task-3 analysis agent for the medical demo."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import hashlib
from collections import Counter, OrderedDict
from pathlib import Path
from time import perf_counter
from typing import Any

from operators.medical_pipeline import build_charts, detect_question_disease


DEFAULT_ANALYSIS_QUESTIONS = [
    "统计每种疾病关联的药物数量",
    "统计高频症状",
    "统计每种疾病的并发症数量",
    "找出症状共现最强的疾病对",
]

DEFAULT_NL2SQL_EVAL_PATH = Path("data/eval/nl2sql_questions.jsonl")

ANALYSIS_INTENT_HINTS = {
    "statistical": ["统计", "数量", "分布", "排行", "高频", "最多"],
    "association": ["共现", "关联", "相似", "共享", "相关"],
    "risk": ["风险", "危险因素", "并发", "原因", "病因"],
    "trend": ["趋势", "变化", "随访", "阶段", "分期"],
    "graph": ["图谱", "核心", "中心", "路径", "连接", "节点", "关系"],
    "visualization": ["可视化", "图表", "展示", "BI", "bi"],
}

GRAPH_INSIGHTS_CACHE_MAX_SIZE = 4
GRAPH_INSIGHTS_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_artifacts(output_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    required_paths = [
        output_root / "data" / "processed" / "documents.jsonl",
        output_root / "graph" / "entities.jsonl",
        output_root / "graph" / "triples.jsonl",
        output_root / "graph" / "graph_snapshot.json",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing analysis inputs: {', '.join(missing)}")

    processed_documents = _read_jsonl(output_root / "data" / "processed" / "documents.jsonl")
    entities = _read_jsonl(output_root / "graph" / "entities.jsonl")
    triples = _read_jsonl(output_root / "graph" / "triples.jsonl")
    graph_snapshot_path = output_root / "graph" / "graph_snapshot.json"
    graph_snapshot = json.loads(graph_snapshot_path.read_text(encoding="utf-8")) if graph_snapshot_path.exists() else {}
    qa_traces = _read_jsonl(output_root / "demo" / "qa_trace.jsonl")
    return processed_documents, entities, triples, graph_snapshot, qa_traces


def _detect_analysis_intents(questions: list[str]) -> dict[str, Any]:
    joined = " ".join(questions)
    intent_flags = {
        name: any(keyword in joined for keyword in keywords)
        for name, keywords in ANALYSIS_INTENT_HINTS.items()
    }
    if not any(intent_flags.values()):
        intent_flags.update({"statistical": True, "association": True, "graph": True, "visualization": True})
    return {
        "input_questions": questions,
        "intent_flags": intent_flags,
        "priority_order": [name for name, enabled in intent_flags.items() if enabled],
    }


def _graph_schema_summary(entities: list[dict[str, Any]], triples: list[dict[str, Any]]) -> dict[str, Any]:
    entity_counts = Counter(str(item.get("type", "")) for item in entities)
    relation_counts = Counter(str(item.get("predicate", "")) for item in triples)
    disease_count = sum(1 for item in entities if item.get("type") == "Disease")
    relation_diversity = len([key for key, value in relation_counts.items() if key and value])
    return {
        "entity_count": len(entities),
        "triple_count": len(triples),
        "disease_count": disease_count,
        "entity_type_counts": dict(sorted(entity_counts.items())),
        "relation_type_counts": dict(sorted(relation_counts.items())),
        "relation_diversity": relation_diversity,
        "supports_risk_analysis": any(relation_counts[predicate] for predicate in ("has_risk_factor", "has_complication", "causes")),
        "supports_treatment_analysis": relation_counts.get("treated_by", 0) > 0,
        "supports_similarity_analysis": relation_counts.get("has_symptom", 0) > 0,
    }


def _artifact_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "sha256": "", "size_bytes": 0}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "exists": True,
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _build_task3_lineage(
    output_root: Path,
    processed_documents: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    triples: list[dict[str, Any]],
    graph_snapshot: dict[str, Any],
) -> dict[str, Any]:
    task1_paths = [
        output_root / "data" / "processed" / "documents.jsonl",
        output_root / "data" / "processed" / "data_processing_trace.jsonl",
        output_root / "data" / "processed" / "data_quality_report.json",
        output_root / "data" / "processed" / "datamate_processing_summary.json",
        output_root / "data" / "processed" / "datamate_processing_trace.jsonl",
        output_root / "data" / "processed" / "datamate_structured_etl_contract.json",
    ]
    task2_paths = [
        output_root / "graph" / "entities.jsonl",
        output_root / "graph" / "triples.jsonl",
        output_root / "graph" / "graph_snapshot.json",
        output_root / "graph" / "kg_validation_report.json",
        output_root / "graph" / "kg_operator_profile.json",
        output_root / "graph" / "kg_build_trace.jsonl",
    ]
    task1_fingerprints = [_artifact_fingerprint(path) for path in task1_paths]
    task2_fingerprints = [_artifact_fingerprint(path) for path in task2_paths]
    graph_nodes = graph_snapshot.get("nodes", []) if isinstance(graph_snapshot, dict) else []
    graph_edges = graph_snapshot.get("edges", []) if isinstance(graph_snapshot, dict) else []
    required_task1 = [
        output_root / "data" / "processed" / "documents.jsonl",
        output_root / "data" / "processed" / "data_quality_report.json",
    ]
    required_task2 = [
        output_root / "graph" / "entities.jsonl",
        output_root / "graph" / "triples.jsonl",
        output_root / "graph" / "graph_snapshot.json",
    ]
    blockers = [
        str(path)
        for path in required_task1 + required_task2
        if not path.exists()
    ]
    if not processed_documents:
        blockers.append("task1_documents_empty")
    if not entities:
        blockers.append("task2_entities_empty")
    if not triples:
        blockers.append("task2_triples_empty")
    if not graph_nodes and not entities:
        blockers.append("graph_nodes_empty")
    if not graph_edges and not triples:
        blockers.append("graph_edges_empty")
    return {
        "lineage_version": "task3_lineage_v1",
        "goal": "reuse_task1_processed_data_and_task2_knowledge_graph_for_task3_analysis",
        "task1": {
            "artifact_count": len(task1_fingerprints),
            "available_artifact_count": sum(1 for item in task1_fingerprints if item["exists"]),
            "documents": len(processed_documents),
            "artifacts": task1_fingerprints,
        },
        "task2": {
            "artifact_count": len(task2_fingerprints),
            "available_artifact_count": sum(1 for item in task2_fingerprints if item["exists"]),
            "entities": len(entities),
            "triples": len(triples),
            "graph_nodes": len(graph_nodes) if graph_nodes else len(entities),
            "graph_edges": len(graph_edges) if graph_edges else len(triples),
            "artifacts": task2_fingerprints,
        },
        "ready": not blockers,
        "blockers": blockers,
        "no_synthetic_task3_input_policy": "Task-3 reads Task-1 and Task-2 artifacts only; missing upstream artifacts block analysis.",
    }


def _build_graph_understanding(
    entities: list[dict[str, Any]],
    triples: list[dict[str, Any]],
    graph_snapshot: dict[str, Any],
) -> dict[str, Any]:
    schema_summary = _graph_schema_summary(entities, triples)
    entity_properties: dict[str, set[str]] = {}
    relation_properties: dict[str, set[str]] = {}
    for entity in entities:
        entity_type = str(entity.get("type") or "Unknown")
        entity_properties.setdefault(entity_type, set()).update(
            key for key, value in entity.items() if value not in (None, "", [], {})
        )
    for triple in triples:
        predicate = str(triple.get("predicate") or "Unknown")
        relation_properties.setdefault(predicate, set()).update(
            key for key, value in triple.items() if value not in (None, "", [], {})
        )
    adjacency_by_type: dict[str, Counter[str]] = {}
    for triple in triples:
        source_type = str(triple.get("subject_type") or "Unknown")
        target_type = str(triple.get("object_type") or "Unknown")
        predicate = str(triple.get("predicate") or "Unknown")
        adjacency_by_type.setdefault(source_type, Counter())[f"{predicate}->{target_type}"] += 1
    node_count = len(graph_snapshot.get("nodes", [])) if isinstance(graph_snapshot, dict) and graph_snapshot.get("nodes") else len(entities)
    edge_count = len(graph_snapshot.get("edges", [])) if isinstance(graph_snapshot, dict) and graph_snapshot.get("edges") else len(triples)
    query_capabilities = [
        {"capability": "统计分析", "supported": bool(triples), "tables": ["triples", "entities", "documents"]},
        {"capability": "关联分析", "supported": schema_summary["supports_similarity_analysis"], "relations": ["has_symptom"]},
        {"capability": "趋势/结构代理分析", "supported": bool(triples), "relations": ["has_stage", "source_doc_id", "relation_family"]},
        {"capability": "风险与因果路径", "supported": schema_summary["supports_risk_analysis"], "relations": ["causes", "has_risk_factor", "has_complication"]},
        {"capability": "BI图表", "supported": bool(triples), "outputs": ["bar", "ranked_bar", "network", "relation_table"]},
        {"capability": "图结构可视化", "supported": bool(node_count and edge_count), "outputs": ["subgraph", "network"]},
    ]
    return {
        "understanding_version": "task3_graph_understanding_v1",
        "graph_schema_summary": schema_summary,
        "node_count": node_count,
        "edge_count": edge_count,
        "entity_properties": {key: sorted(values) for key, values in sorted(entity_properties.items())},
        "relation_properties": {key: sorted(values) for key, values in sorted(relation_properties.items())},
        "adjacency_by_entity_type": {
            key: dict(counter.most_common())
            for key, counter in sorted(adjacency_by_type.items())
        },
        "query_capabilities": query_capabilities,
        "supported_analysis_modes": [
            item["capability"] for item in query_capabilities if item["supported"]
        ],
    }


def _analysis_tool_plan(
    intent: dict[str, Any],
    schema_summary: dict[str, Any],
    query_catalog: list[dict[str, str]],
) -> list[dict[str, Any]]:
    flags = intent["intent_flags"]
    plan: list[dict[str, Any]] = []

    def add(step: str, tool: str, purpose: str, outputs: list[str], priority: int) -> None:
        plan.append(
            {
                "order": len(plan) + 1,
                "step": step,
                "tool": tool,
                "purpose": purpose,
                "outputs": outputs,
                "priority": priority,
            }
        )

    add("load_graph_artifacts", "artifact_loader", "读取任务一和任务二产物，建立分析上下文。", ["documents", "entities", "triples", "graph_snapshot"], 100)
    add("profile_graph_schema", "graph_schema_profiler", "理解实体类型、关系类型和可分析能力。", ["graph_schema_summary"], 95)
    if flags.get("statistical"):
        add("run_nl2sql_statistics", "nl2sql_sqlite_engine", "执行统计类自然语言分析查询。", [item["chart"] for item in query_catalog], 90)
    if flags.get("association") and schema_summary.get("supports_similarity_analysis"):
        add("run_association_analysis", "symptom_and_multi_relation_similarity", "执行症状共现和多关系相似度分析。", ["disease_similarity", "multi_relation_similarity"], 86)
    if flags.get("risk") and schema_summary.get("supports_risk_analysis"):
        add("run_risk_path_analysis", "causal_path_and_risk_counter", "分析危险因素、并发症和因果路径。", ["disease_risk_factor_counts", "causal_paths"], 84)
    if flags.get("graph"):
        add("run_graph_algorithms", "pagerank_degree_community", "执行中心性、社区和子图排序。", ["graph_pagerank", "graph_degree_centrality", "subgraph_ranking"], 82)
    if flags.get("trend"):
        add("run_trend_proxy_analysis", "stage_and_document_distribution", "基于阶段、来源和图谱结构生成趋势代理分析。", ["relation_families", "graph_quality_diagnostics"], 70)
    add("render_bi_charts", "chart_renderer", "输出 BI 图表和图谱结构可视化数据。", ["demo/charts/*.json"], 78)
    add("render_explanatory_report", "report_writer", "生成图文结合的分析说明。", ["demo/analysis_report.md"], 76)
    return sorted(plan, key=lambda item: (-item["priority"], item["order"]))


def _build_analysis_planning(
    *,
    questions: list[str],
    entities: list[dict[str, Any]],
    triples: list[dict[str, Any]],
    query_catalog: list[dict[str, str]],
) -> dict[str, Any]:
    intent = _detect_analysis_intents(questions)
    schema_summary = _graph_schema_summary(entities, triples)
    tool_plan = _analysis_tool_plan(intent, schema_summary, query_catalog)
    chart_recommendations = []
    for query in query_catalog:
        chart = query.get("chart", "")
        chart_type = "bar"
        if "cooccurrence" in chart or "similarity" in chart:
            chart_type = "network"
        elif "centrality" in chart or "pagerank" in chart:
            chart_type = "ranked_bar"
        chart_recommendations.append(
            {
                "chart": chart,
                "chart_type": chart_type,
                "source_query_id": query.get("query_id", ""),
                "reason": "匹配统计/关联分析输出，适合 BI 展示。",
            }
        )
    return {
        "goal": "从数据到知识再到洞察的图谱驱动分析",
        "intent": intent,
        "graph_schema_summary": schema_summary,
        "tool_plan": tool_plan,
        "chart_recommendations": chart_recommendations,
        "selected_query_count": len(query_catalog),
        "selected_algorithm_count": sum(1 for item in tool_plan if item["tool"] in {"pagerank_degree_community", "symptom_and_multi_relation_similarity", "causal_path_and_risk_counter"}),
    }


def _build_sqlite(triples: list[dict[str, Any]], entities: list[dict[str, Any]], processed_documents: list[dict[str, Any]]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE triples (
            subject TEXT,
            subject_type TEXT,
            predicate TEXT,
            object TEXT,
            object_type TEXT,
            source_doc_id TEXT,
            confidence REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE entities (
            entity_id TEXT,
            name TEXT,
            type TEXT,
            confidence REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE documents (
            doc_id TEXT,
            title TEXT,
            source TEXT,
            length INTEGER,
            sentence_count INTEGER
        )
        """
    )

    conn.executemany(
        "INSERT INTO triples VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                item.get("subject", ""),
                item.get("subject_type", ""),
                item.get("predicate", ""),
                item.get("object", ""),
                item.get("object_type", ""),
                item.get("source_doc_id", ""),
                float(item.get("confidence", 0)),
            )
            for item in triples
        ],
    )
    conn.executemany(
        "INSERT INTO entities VALUES (?, ?, ?, ?)",
        [
            (
                item.get("entity_id", ""),
                item.get("name", ""),
                item.get("type", ""),
                float(item.get("confidence", 0)),
            )
            for item in entities
        ],
    )
    conn.executemany(
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
        [
            (
                item.get("doc_id", ""),
                item.get("title", ""),
                item.get("source", ""),
                int(item.get("quality", {}).get("length", 0)),
                int(item.get("quality", {}).get("sentence_count", 0)),
            )
            for item in processed_documents
        ],
    )
    conn.commit()
    return conn


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _run_query(conn: sqlite3.Connection, sql: str) -> list[dict[str, Any]]:
    cursor = conn.execute(sql)
    return _rows_to_dicts(cursor.fetchall())


def _analysis_query_catalog() -> list[dict[str, str]]:
    return [
        {
            "query_id": "disease_drug_counts",
            "question": "统计每种疾病关联的药物数量",
            "sql": (
                "SELECT subject AS disease, COUNT(*) AS drug_count "
                "FROM triples "
                "WHERE predicate = 'treated_by' AND object_type = 'Drug' "
                "GROUP BY subject "
                "ORDER BY drug_count DESC, disease ASC "
                "LIMIT 10"
            ),
            "chart": "disease_drug_counts",
        },
        {
            "query_id": "symptom_frequency",
            "question": "统计高频症状",
            "sql": (
                "SELECT object AS symptom, COUNT(DISTINCT subject) AS disease_count "
                "FROM triples "
                "WHERE predicate = 'has_symptom' "
                "GROUP BY object "
                "ORDER BY disease_count DESC, symptom ASC "
                "LIMIT 10"
            ),
            "chart": "symptom_frequency",
        },
        {
            "query_id": "disease_complication_counts",
            "question": "统计每种疾病的并发症数量",
            "sql": (
                "SELECT subject AS disease, COUNT(*) AS complication_count "
                "FROM triples "
                "WHERE predicate = 'has_complication' "
                "GROUP BY subject "
                "ORDER BY complication_count DESC, disease ASC "
                "LIMIT 10"
            ),
            "chart": "disease_complication_counts",
        },
        {
            "query_id": "symptom_cooccurrence",
            "question": "找出症状共现最强的疾病对",
            "sql": (
                "SELECT a.subject AS disease_a, b.subject AS disease_b, COUNT(DISTINCT a.object) AS shared_symptom_count "
                "FROM triples a "
                "JOIN triples b ON a.object = b.object "
                "WHERE a.predicate = 'has_symptom' "
                "AND b.predicate = 'has_symptom' "
                "AND a.subject < b.subject "
                "GROUP BY a.subject, b.subject "
                "ORDER BY shared_symptom_count DESC, disease_a ASC, disease_b ASC "
                "LIMIT 10"
            ),
            "chart": "symptom_cooccurrence",
        },
        {
            "query_id": "department_distribution",
            "question": "统计疾病所属科室分布",
            "sql": (
                "SELECT object AS department, COUNT(DISTINCT subject) AS disease_count "
                "FROM triples "
                "WHERE predicate = 'belongs_to_department' "
                "GROUP BY object "
                "ORDER BY disease_count DESC, department ASC "
                "LIMIT 10"
            ),
            "chart": "department_distribution",
        },
        {
            "query_id": "drug_contraindications",
            "question": "统计存在禁忌的药物数量",
            "sql": (
                "SELECT subject AS drug, COUNT(*) AS contraindication_count "
                "FROM triples "
                "WHERE predicate = 'contraindicated_for' "
                "GROUP BY subject "
                "ORDER BY contraindication_count DESC, drug ASC "
                "LIMIT 10"
            ),
            "chart": "drug_contraindications",
        },
        {
            "query_id": "disease_risk_factor_counts",
            "question": "统计每种疾病的危险因素数量",
            "sql": (
                "SELECT subject AS disease, COUNT(*) AS risk_factor_count "
                "FROM triples "
                "WHERE predicate = 'has_risk_factor' "
                "GROUP BY subject "
                "ORDER BY risk_factor_count DESC, disease ASC "
                "LIMIT 10"
            ),
            "chart": "disease_risk_factor_counts",
        },
        {
            "query_id": "disease_prevention_counts",
            "question": "统计每种疾病的预防管理措施数量",
            "sql": (
                "SELECT subject AS disease, COUNT(*) AS prevention_count "
                "FROM triples "
                "WHERE predicate = 'prevented_by' "
                "GROUP BY subject "
                "ORDER BY prevention_count DESC, disease ASC "
                "LIMIT 10"
            ),
            "chart": "disease_prevention_counts",
        },
        {
            "query_id": "entity_centrality",
            "question": "找出图谱中连接最多的核心实体",
            "sql": (
                "SELECT name AS entity, type AS entity_type, COUNT(*) AS degree "
                "FROM ("
                "SELECT subject AS name, subject_type AS type FROM triples "
                "UNION ALL "
                "SELECT object AS name, object_type AS type FROM triples"
                ") "
                "GROUP BY name, type "
                "ORDER BY degree DESC, entity ASC "
                "LIMIT 10"
            ),
            "chart": "entity_centrality",
        },
        {
            "query_id": "entity_type_distribution",
            "question": "统计图谱实体类型分布",
            "sql": (
                "SELECT type AS entity_type, COUNT(*) AS entity_count "
                "FROM entities "
                "GROUP BY type "
                "ORDER BY entity_count DESC, entity_type ASC "
                "LIMIT 10"
            ),
            "chart": "entity_type_distribution",
        },
        {
            "query_id": "relation_family_distribution",
            "question": "统计关系家族分布",
            "sql": (
                "SELECT "
                "CASE "
                "WHEN predicate IN ('has_symptom', 'has_complication', 'has_risk_factor') THEN 'clinical_profile' "
                "WHEN predicate IN ('diagnosed_by', 'belongs_to_department') THEN 'diagnosis' "
                "WHEN predicate IN ('treated_by', 'prevented_by', 'contraindicated_for') THEN 'treatment_management' "
                "WHEN predicate IN ('causes') THEN 'causal' "
                "ELSE 'other' END AS relation_family, "
                "COUNT(*) AS triple_count "
                "FROM triples "
                "GROUP BY relation_family "
                "ORDER BY triple_count DESC, relation_family ASC"
            ),
            "chart": "relation_family_distribution",
        },
        {
            "query_id": "source_document_contribution",
            "question": "统计来源文档贡献的三元组数量",
            "sql": (
                "SELECT t.source_doc_id AS doc_id, COALESCE(d.title, '') AS title, "
                "COUNT(*) AS triple_count, ROUND(AVG(t.confidence), 4) AS avg_confidence "
                "FROM triples t "
                "LEFT JOIN documents d ON t.source_doc_id = d.doc_id "
                "GROUP BY t.source_doc_id, d.title "
                "ORDER BY triple_count DESC, doc_id ASC "
                "LIMIT 10"
            ),
            "chart": "source_document_contribution",
        },
        {
            "query_id": "low_confidence_relations",
            "question": "列出低置信度关系用于质量审查",
            "sql": (
                "SELECT subject, predicate, object, confidence "
                "FROM triples "
                "WHERE confidence < 0.85 "
                "ORDER BY confidence ASC, subject ASC, predicate ASC, object ASC "
                "LIMIT 10"
            ),
            "chart": "low_confidence_relations",
        },
        {
            "query_id": "high_confidence_relation_counts",
            "question": "统计高置信度关系类型数量",
            "sql": (
                "SELECT predicate AS relation, COUNT(*) AS fact_count, ROUND(AVG(confidence), 4) AS avg_confidence "
                "FROM triples "
                "WHERE confidence >= 0.9 "
                "GROUP BY predicate "
                "ORDER BY fact_count DESC, relation ASC "
                "LIMIT 10"
            ),
            "chart": "high_confidence_relation_counts",
        },
        {
            "query_id": "risk_factor_pairs",
            "question": "列出疾病和危险因素关系",
            "sql": (
                "SELECT subject AS disease, object AS risk_factor, confidence "
                "FROM triples "
                "WHERE predicate = 'has_risk_factor' "
                "ORDER BY confidence DESC, disease ASC, risk_factor ASC "
                "LIMIT 10"
            ),
            "chart": "risk_factor_pairs",
        },
    ]


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _query_payload(query_id: str, question: str, sql: str, chart: str = "") -> dict[str, str]:
    return {
        "query_id": query_id,
        "question": question,
        "sql": sql,
        "chart": chart,
    }


def _disease_query_payload(question: str, disease: str, relation: str) -> dict[str, str]:
    disease_sql = _sql_literal(disease)
    if relation == "has_symptom":
        return _query_payload(
            "disease_symptoms",
            question,
            (
                "SELECT object AS symptom "
                "FROM triples "
                f"WHERE subject = {disease_sql} AND predicate = 'has_symptom' "
                "ORDER BY symptom ASC "
                "LIMIT 10"
            ),
            "disease_symptoms",
        )
    if relation == "treated_by":
        return _query_payload(
            "disease_treatments",
            question,
            (
                "SELECT object AS treatment, object_type "
                "FROM triples "
                f"WHERE subject = {disease_sql} AND predicate = 'treated_by' "
                "ORDER BY object_type ASC, treatment ASC "
                "LIMIT 10"
            ),
            "disease_treatments",
        )
    if relation == "diagnosed_by":
        return _query_payload(
            "disease_tests",
            question,
            (
                "SELECT object AS test "
                "FROM triples "
                f"WHERE subject = {disease_sql} AND predicate = 'diagnosed_by' "
                "ORDER BY test ASC "
                "LIMIT 10"
            ),
            "disease_tests",
        )
    if relation == "has_complication":
        return _query_payload(
            "disease_complications",
            question,
            (
                "SELECT object AS complication "
                "FROM triples "
                f"WHERE subject = {disease_sql} AND predicate = 'has_complication' "
                "ORDER BY complication ASC "
                "LIMIT 10"
            ),
            "disease_complications",
        )
    if relation == "has_risk_factor":
        return _query_payload(
            "disease_risk_factors",
            question,
            (
                "SELECT object AS risk_factor "
                "FROM triples "
                f"WHERE subject = {disease_sql} AND predicate = 'has_risk_factor' "
                "ORDER BY risk_factor ASC "
                "LIMIT 10"
            ),
            "disease_risk_factors",
        )
    if relation == "prevented_by":
        return _query_payload(
            "disease_preventions",
            question,
            (
                "SELECT object AS prevention "
                "FROM triples "
                f"WHERE subject = {disease_sql} AND predicate = 'prevented_by' "
                "ORDER BY prevention ASC "
                "LIMIT 10"
            ),
            "disease_preventions",
        )
    raise ValueError(f"Unsupported disease relation: {relation}")


def _normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


def _canonical_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {key: str(value) for key, value in sorted(row.items())}
        for row in rows
    ]


NL2SQL_SCHEMA_TERMS = {
    "triples": ["三元组", "关系", "图谱边", "事实"],
    "entities": ["实体", "节点", "图谱节点"],
    "documents": ["文档", "记录", "来源"],
    "subject": ["疾病", "病种", "subject", "主语"],
    "object": ["症状", "药物", "检查", "科室", "危险因素", "预防", "object", "宾语"],
    "predicate": ["关系", "谓词", "并发症", "治疗", "诊断", "属于科室", "禁忌"],
    "confidence": ["置信度", "可信度", "confidence"],
    "source_doc_id": ["来源", "文档", "证据", "出处"],
    "type": ["类型", "类别", "实体类型"],
}

NL2SQL_QUERY_HINTS = {
    "disease_drug_counts": ["药物", "用药", "治疗", "覆盖", "drug"],
    "symptom_frequency": ["症状", "表现", "高频", "常见", "frequency"],
    "disease_complication_counts": ["并发", "并发症", "合并症", "风险", "complication"],
    "symptom_cooccurrence": ["共现", "共同", "共享", "相似", "组合"],
    "department_distribution": ["科室", "就诊", "department"],
    "drug_contraindications": ["禁忌", "慎用", "避免", "contraindication"],
    "disease_risk_factor_counts": ["危险因素", "风险因素", "病因", "原因"],
    "disease_prevention_counts": ["预防", "管理", "干预", "随访"],
    "entity_centrality": ["中心", "核心", "连接", "度数", "重要", "节点"],
    "entity_type_distribution": ["实体类型", "实体类别", "类型分布", "节点类型", "实体分布"],
    "relation_family_distribution": ["关系家族", "关系类别", "关系类型", "临床画像", "诊疗管理", "因果", "分布"],
    "source_document_contribution": ["来源", "文档", "出处", "贡献", "证据", "数据源"],
    "low_confidence_relations": ["低置信度", "置信度低", "质量审查", "质量问题", "不可靠", "待复核"],
    "high_confidence_relation_counts": ["高置信度", "可信", "高质量", "可靠", "置信度高"],
    "risk_factor_pairs": ["危险因素", "风险因素", "病因", "原因", "列出", "明细", "关系对"],
}

NL2SQL_RELATION_LINKS = {
    "has_symptom": ["症状", "表现", "体征", "临床表现"],
    "treated_by": ["治疗", "用药", "方案", "药物", "干预"],
    "diagnosed_by": ["检查", "诊断", "检验", "辅助检查", "化验"],
    "has_complication": ["并发", "合并症", "风险"],
    "has_risk_factor": ["危险因素", "风险因素", "病因", "原因"],
    "prevented_by": ["预防", "管理", "随访"],
}


def _schema_link_question(question: str) -> dict[str, Any]:
    normalized = re.sub(r"\s+", "", question)
    linked_columns = [
        column
        for column, hints in NL2SQL_SCHEMA_TERMS.items()
        if any(hint in normalized for hint in hints)
    ]
    linked_relations = [
        relation
        for relation, hints in NL2SQL_RELATION_LINKS.items()
        if any(hint in normalized for hint in hints)
    ]
    intents = {
        "aggregation": any(keyword in normalized for keyword in ("统计", "数量", "排行", "最多", "分布", "频率")),
        "association": any(keyword in normalized for keyword in ("共现", "共同", "共享", "相似", "组合")),
        "entity_focus": detect_question_disease(question) is not None,
        "graph_centrality": any(keyword in normalized for keyword in ("中心", "核心", "连接", "度数", "重要")),
        "quality": any(keyword in normalized for keyword in ("置信度", "质量", "可信", "可靠", "复核")),
        "source": any(keyword in normalized for keyword in ("来源", "文档", "出处", "证据", "贡献")),
        "list": any(keyword in normalized for keyword in ("列出", "明细", "有哪些", "清单")),
    }
    return {
        "linked_tables": ["triples"] if linked_columns or linked_relations else [],
        "linked_columns": linked_columns,
        "linked_relations": linked_relations,
        "intents": intents,
        "focus_disease": detect_question_disease(question),
    }


def _score_query_candidate(question: str, candidate: dict[str, str], schema_links: dict[str, Any]) -> dict[str, Any]:
    normalized = re.sub(r"\s+", "", question)
    query_id = candidate["query_id"]
    hints = NL2SQL_QUERY_HINTS.get(query_id, [])
    hint_hits = sum(1 for hint in hints if hint in normalized)
    question_tokens = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", candidate.get("question", "")))
    input_tokens = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", question))
    token_overlap = len(question_tokens & input_tokens) / max(1, len(question_tokens | input_tokens))
    relation_bonus = 0.0
    sql = candidate.get("sql", "")
    for relation in schema_links.get("linked_relations", []):
        if relation in sql:
            relation_bonus += 0.18
    aggregation_bonus = 0.12 if schema_links["intents"].get("aggregation") and "COUNT(" in sql.upper() else 0.0
    association_bonus = 0.16 if schema_links["intents"].get("association") and "JOIN" in sql.upper() else 0.0
    centrality_bonus = 0.2 if schema_links["intents"].get("graph_centrality") and query_id == "entity_centrality" else 0.0
    quality_bonus = 0.2 if schema_links["intents"].get("quality") and "confidence" in sql.lower() else 0.0
    source_bonus = 0.22 if schema_links["intents"].get("source") and "source_doc_id" in sql.lower() else 0.0
    list_bonus = 0.1 if schema_links["intents"].get("list") and "COUNT(" not in sql.upper() else 0.0
    entity_type_bonus = 0.2 if "实体类型" in normalized and query_id == "entity_type_distribution" else 0.0
    relation_family_bonus = 0.18 if any(keyword in normalized for keyword in ("关系家族", "关系类别", "关系类型")) and query_id == "relation_family_distribution" else 0.0
    score = min(
        1.0,
        hint_hits * 0.16
        + token_overlap * 0.22
        + relation_bonus
        + aggregation_bonus
        + association_bonus
        + centrality_bonus
        + quality_bonus
        + source_bonus
        + list_bonus
        + entity_type_bonus
        + relation_family_bonus,
    )
    return {
        "query_id": query_id,
        "score": round(score, 4),
        "hint_hits": hint_hits,
        "token_overlap": round(token_overlap, 4),
        "relation_bonus": round(relation_bonus, 4),
        "aggregation_bonus": aggregation_bonus,
        "association_bonus": association_bonus,
        "centrality_bonus": centrality_bonus,
        "quality_bonus": quality_bonus,
        "source_bonus": source_bonus,
        "list_bonus": list_bonus,
        "entity_type_bonus": entity_type_bonus,
        "relation_family_bonus": relation_family_bonus,
    }


def _validate_sql_payload(query: dict[str, str] | None) -> dict[str, Any]:
    if query is None:
        return {"valid": False, "repaired": False, "reason": "no_query"}
    sql = query.get("sql", "").strip()
    upper_sql = sql.upper()
    required_tokens = ["SELECT", "FROM"]
    missing = [token for token in required_tokens if token not in upper_sql]
    unsafe = any(token in upper_sql for token in ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER"))
    repaired = False
    if sql and not sql.rstrip().endswith(";"):
        repaired = True
    return {
        "valid": bool(sql and not missing and not unsafe),
        "repaired": repaired,
        "missing_tokens": missing,
        "unsafe": unsafe,
        "allowed_tables": ["triples", "entities", "documents"],
    }


def _legacy_rule_generate_sql(question: str) -> dict[str, str] | None:
    normalized = re.sub(r"\s+", "", question)
    catalog = _analysis_query_catalog()
    direct = {item["question"]: item for item in catalog}
    if question in direct:
        return dict(direct[question])

    disease = detect_question_disease(question)
    if disease:
        if re.search(r"危险因素|风险因素|病因|原因|相关因素", normalized):
            return _disease_query_payload(question, disease, "has_risk_factor")
        if re.search(r"治疗|用药|方案|怎么治|药物", normalized):
            return _disease_query_payload(question, disease, "treated_by")
        if re.search(r"预防|管理|干预|如何防|怎么防|随访", normalized):
            return _disease_query_payload(question, disease, "prevented_by")
        if re.search(r"并发|合并症|风险", normalized):
            return _disease_query_payload(question, disease, "has_complication")
        if re.search(r"症状|表现|体征|主诉|临床表现", normalized):
            return _disease_query_payload(question, disease, "has_symptom")
        if re.search(r"检查|诊断|检验|需要做|辅助检查|化验", normalized):
            return _disease_query_payload(question, disease, "diagnosed_by")

    rules: list[tuple[str, str]] = [
        (r"(低置信度|置信度低|低可信|质量审查|待复核|质量问题)", "low_confidence_relations"),
        (r"(高置信度|置信度高|高质量|可靠|可信).*(关系|三元组|类型|统计|数量)", "high_confidence_relation_counts"),
        (r"(来源|文档|出处|证据).*(贡献|数量|最多|排行|统计)|贡献.*(来源|文档|证据)", "source_document_contribution"),
        (r"(实体类型|实体类别|节点类型|实体分布).*(统计|分布|数量)|统计.*(实体类型|实体类别|节点类型)", "entity_type_distribution"),
        (r"(关系家族|关系类别|关系类型).*(分布|统计|数量)|临床画像.*诊疗管理|因果.*关系.*分布", "relation_family_distribution"),
        (r"(列出|明细|清单).*(危险因素|风险因素|病因|原因).*(关系|疾病|三元组)|危险因素.*(明细|清单|关系对)", "risk_factor_pairs"),
        (r"(禁忌|慎用|避免).*(药物|用药)|药物.*(禁忌|慎用)", "drug_contraindications"),
        (r"(症状).*(共现|共同|共享|相似).*(疾病|病种|组合|疾病对)|疾病对.*(症状|共享|相似)|共同症状.*(病种|疾病).*组合", "symptom_cooccurrence"),
        (r"(疾病|病种).*(药物|用药).*(数量|统计|排行|最多)|药物.*(数量|统计|排行).*(疾病|病种)|用药.*(疾病|病种)", "disease_drug_counts"),
        (r"(高频|常见).*(症状)|症状.*(频率|统计|排行|疾病|关联|最多)|哪些症状", "symptom_frequency"),
        (r"(疾病|病种).*(并发症|并发|合并症|风险).*(数量|统计)|并发症.*(数量|统计|排行)", "disease_complication_counts"),
        (r"(科室).*(分布|统计|数量|疾病)|疾病.*科室|就诊科室.*(分布|排行)", "department_distribution"),
        (r"(危险因素|风险因素|病因|原因).*(数量|统计|排行)|疾病.*(危险因素|病因)", "disease_risk_factor_counts"),
        (r"(预防|管理|干预).*(数量|统计|排行)|疾病.*(预防|管理|干预)", "disease_prevention_counts"),
        (r"(核心|中心|连接最多|度数|最重要|高连接).*(实体|节点)|实体.*(中心性|连接|度数)", "entity_centrality"),
    ]
    by_id = {item["query_id"]: item for item in catalog}
    for pattern, query_id in rules:
        if re.search(pattern, normalized):
            return dict(by_id[query_id])
    return None


def plan_nl2sql_query(question: str) -> dict[str, Any]:
    """Plan NL2SQL with schema linking, candidate ranking, and SQL validation."""
    catalog = _analysis_query_catalog()
    schema_links = _schema_link_question(question)
    legacy_query = _legacy_rule_generate_sql(question)
    candidate_scores = [
        _score_query_candidate(question, candidate, schema_links)
        for candidate in catalog
    ]
    candidate_scores.sort(key=lambda item: (-float(item["score"]), item["query_id"]))
    selected_query = legacy_query
    selection_mode = "direct_or_rule"
    if selected_query is None and candidate_scores and candidate_scores[0]["score"] >= 0.28:
        by_id = {item["query_id"]: item for item in catalog}
        selected_query = dict(by_id[candidate_scores[0]["query_id"]])
        selection_mode = "schema_linking_candidate_rank"
    validation = _validate_sql_payload(selected_query)
    trace = [
        {
            "step": "schema_linking",
            "status": "success",
            **schema_links,
        },
        {
            "step": "candidate_recall",
            "status": "success",
            "candidate_count": len(candidate_scores),
            "top_candidates": candidate_scores[:5],
        },
        {
            "step": "query_select",
            "status": "success" if selected_query else "empty",
            "selection_mode": selection_mode,
            "selected_query_id": selected_query.get("query_id") if selected_query else None,
        },
        {
            "step": "sql_validate_repair",
            "status": "success" if validation["valid"] else "failed",
            **validation,
        },
    ]
    return {
        "question": question,
        "query": selected_query if validation["valid"] else None,
        "schema_links": schema_links,
        "candidate_scores": candidate_scores,
        "selection_mode": selection_mode,
        "sql_validation": validation,
        "trace": trace,
    }


def generate_sql_for_question(question: str) -> dict[str, str] | None:
    """Map supported analysis questions to deterministic SQL templates."""
    return plan_nl2sql_query(question)["query"]


def _default_eval_items() -> list[dict[str, str]]:
    catalog = _analysis_query_catalog()
    by_id = {item["query_id"]: item for item in catalog}
    questions_by_id = {
        "disease_drug_counts": [
            "统计每种疾病关联的药物数量",
            "哪些疾病关联的药物最多？",
            "按疾病统计用药数量排行",
        ],
        "symptom_frequency": [
            "统计高频症状",
            "哪些症状关联的疾病最多？",
            "给出常见症状排行榜",
        ],
        "disease_complication_counts": [
            "统计每种疾病的并发症数量",
            "哪些疾病并发症数量最多？",
            "按疾病统计并发症数量",
        ],
        "symptom_cooccurrence": [
            "找出症状共现最强的疾病对",
            "哪些疾病对共享症状最多？",
            "基于症状相似度找疾病对",
        ],
        "disease_risk_factor_counts": [
            "统计每种疾病的危险因素数量",
            "按疾病统计风险因素数量",
            "哪些疾病病因因素最多？",
        ],
        "disease_prevention_counts": [
            "统计每种疾病的预防管理措施数量",
            "按疾病统计管理干预数量",
            "哪些疾病预防措施最多？",
        ],
        "entity_centrality": [
            "找出图谱中连接最多的核心实体",
            "哪些实体节点度数最高？",
            "图谱里最重要的高连接节点有哪些？",
        ],
        "entity_type_distribution": [
            "统计图谱实体类型分布",
            "实体类别数量怎么分布？",
            "按节点类型统计实体数量",
        ],
        "relation_family_distribution": [
            "统计关系家族分布",
            "关系类别数量怎么分布？",
            "临床画像、诊疗管理和因果关系占比如何？",
        ],
        "source_document_contribution": [
            "统计来源文档贡献的三元组数量",
            "哪些证据文档贡献关系最多？",
            "按出处统计事实数量和平均置信度",
        ],
        "low_confidence_relations": [
            "列出低置信度关系用于质量审查",
            "哪些三元组置信度低需要复核？",
            "给我质量问题关系明细",
        ],
        "high_confidence_relation_counts": [
            "统计高置信度关系类型数量",
            "哪些关系类型最可靠？",
            "高质量三元组按谓词统计",
        ],
        "risk_factor_pairs": [
            "列出疾病和危险因素关系",
            "给出危险因素关系明细",
            "风险因素和疾病的关系对清单",
        ],
    }
    rows: list[dict[str, str]] = []
    for query_id, questions in questions_by_id.items():
        expected = by_id[query_id]
        for question in questions:
            rows.append(
                {
                    "question": question,
                    "expected_query_id": query_id,
                    "expected_sql": expected["sql"],
                }
            )
    return rows


def _load_eval_items(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return _default_eval_items()
    rows = _read_jsonl(path)
    items: list[dict[str, str]] = []
    for row in rows:
        question = str(row.get("question", "")).strip()
        expected_query_id = str(row.get("expected_query_id", "")).strip()
        expected_sql = str(row.get("expected_sql", "")).strip()
        if question and expected_query_id and expected_sql:
            items.append(
                {
                    "question": question,
                    "expected_query_id": expected_query_id,
                    "expected_sql": expected_sql,
                }
            )
    return items or _default_eval_items()


def _build_nl2sql_eval_report(eval_result: dict[str, Any]) -> str:
    lines = [
        "# NL2SQL 评测报告",
        "",
        f"- 评测样本数：{eval_result['total']}",
        f"- 通过样本数：{eval_result['passed']}",
        f"- 准确率：{eval_result['accuracy']:.2%}",
        f"- 目标阈值：{eval_result['threshold']:.2%}",
        f"- 是否达标：{'是' if eval_result['meets_threshold'] else '否'}",
        "",
        "## 明细",
        "",
        "| 问题 | 期望 query_id | 预测 query_id | SQL 匹配 | 结果匹配 | 通过 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in eval_result["items"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["question"]),
                    str(item["expected_query_id"]),
                    str(item.get("predicted_query_id") or ""),
                    "是" if item["sql_match"] else "否",
                    "是" if item["result_match"] else "否",
                    "是" if item["passed"] else "否",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def run_nl2sql_eval(
    output_root: Path,
    *,
    eval_path: Path | None = None,
    threshold: float = 0.85,
) -> dict[str, Any]:
    loaded_documents, loaded_entities, loaded_triples, _, _ = _load_artifacts(output_root)
    eval_file = eval_path or DEFAULT_NL2SQL_EVAL_PATH
    if not eval_file.is_absolute():
        eval_file = output_root / eval_file
    eval_items = _load_eval_items(eval_file)

    conn = _build_sqlite(loaded_triples, loaded_entities, loaded_documents)
    try:
        detailed: list[dict[str, Any]] = []
        for item in eval_items:
            expected_rows = _run_query(conn, item["expected_sql"])
            prediction_plan = plan_nl2sql_query(item["question"])
            predicted = prediction_plan["query"]
            predicted_rows = _run_query(conn, predicted["sql"]) if predicted else []
            sql_match = bool(predicted) and _normalize_sql(predicted["sql"]) == _normalize_sql(item["expected_sql"])
            result_match = _canonical_rows(predicted_rows) == _canonical_rows(expected_rows)
            passed = sql_match and result_match
            detailed.append(
                {
                    "question": item["question"],
                    "expected_query_id": item["expected_query_id"],
                    "predicted_query_id": predicted.get("query_id") if predicted else "",
                    "expected_sql": item["expected_sql"],
                    "predicted_sql": predicted.get("sql") if predicted else "",
                    "expected_row_count": len(expected_rows),
                    "predicted_row_count": len(predicted_rows),
                    "sql_match": sql_match,
                    "result_match": result_match,
                    "passed": passed,
                    "nl2sql_plan": {
                        "selection_mode": prediction_plan["selection_mode"],
                        "schema_links": prediction_plan["schema_links"],
                        "sql_validation": prediction_plan["sql_validation"],
                        "top_candidates": prediction_plan["candidate_scores"][:3],
                    },
                }
            )

        passed_count = sum(1 for item in detailed if item["passed"])
        total = len(detailed)
        accuracy = passed_count / total if total else 0.0
        result: dict[str, Any] = {
            "task": "task3_nl2sql_eval",
            "eval_path": str(eval_file),
            "total": total,
            "passed": passed_count,
            "failed": total - passed_count,
            "accuracy": accuracy,
            "threshold": threshold,
            "meets_threshold": accuracy >= threshold,
            "items": detailed,
            "output_paths": {
                "json": str(output_root / "demo" / "nl2sql_eval_report.json"),
                "markdown": str(output_root / "demo" / "nl2sql_eval_report.md"),
            },
        }
        _write_json(Path(result["output_paths"]["json"]), result)
        _write_text(Path(result["output_paths"]["markdown"]), _build_nl2sql_eval_report(result))
        return result
    finally:
        conn.close()


def _chart_from_rows(rows: list[dict[str, Any]], x_key: str, y_key: str, chart_type: str = "bar") -> dict[str, Any]:
    return {
        "chart_type": chart_type,
        "labels": [str(row.get(x_key, "")) for row in rows],
        "values": [row.get(y_key, 0) for row in rows],
        "rows": rows,
    }


def _chart_for_query_result(query: dict[str, str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    chart = query.get("chart", "")
    if not rows:
        return {"chart": chart, "chart_type": "empty", "rows": []}
    first = rows[0]
    keys = list(first.keys())
    numeric_keys = [
        key for key in keys
        if isinstance(first.get(key), int | float) or str(first.get(key, "")).replace(".", "", 1).isdigit()
    ]
    if "cooccurrence" in chart or {"disease_a", "disease_b"}.issubset(set(keys)):
        return {"chart": chart, "chart_type": "network", "pairs": rows, "rows": rows}
    if numeric_keys and len(keys) >= 2:
        y_key = numeric_keys[0]
        x_key = next((key for key in keys if key != y_key), keys[0])
        payload = _chart_from_rows(rows, x_key, y_key)
        payload["chart"] = chart
        payload["chart_type"] = "ranked_bar" if "centrality" in chart else "bar"
        return payload
    return {"chart": chart, "chart_type": "table", "rows": rows}


def _explain_interactive_result(question: str, query: dict[str, str] | None, rows: list[dict[str, Any]], graph_context: dict[str, Any]) -> str:
    if query is None:
        return "当前问题未匹配到可执行的 NL2SQL 模板，已返回图谱结构摘要供继续分析。"
    if not rows:
        return f"问题“{question}”已执行 `{query['query_id']}`，但当前图谱中没有命中结果。"
    first = rows[0]
    key_values = "，".join(f"{key}={value}" for key, value in list(first.items())[:3])
    return (
        f"问题“{question}”被规划为 `{query['query_id']}` 分析，返回 {len(rows)} 行结果。"
        f"最高项为：{key_values}。图谱上下文包含 {graph_context.get('node_count', 0)} 个节点、"
        f"{graph_context.get('edge_count', 0)} 条边。"
    )


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _detect_row_dimensions(rows: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    if not rows:
        return None, None
    keys = list(rows[0].keys())
    numeric_key = next((key for key in keys if _numeric_value(rows[0].get(key)) is not None), None)
    label_key = next((key for key in keys if key != numeric_key), keys[0] if keys else None)
    return label_key, numeric_key


def _select_focus_disease_from_rows(rows: list[dict[str, Any]], explicit_disease: str | None) -> str | None:
    if explicit_disease:
        return explicit_disease
    label_key, numeric_key = _detect_row_dimensions(rows)
    if not rows or not label_key:
        return None
    if "disease" not in label_key:
        return None
    ranked_rows = rows
    if numeric_key:
        ranked_rows = sorted(rows, key=lambda row: _numeric_value(row.get(numeric_key)) or 0.0, reverse=True)
    candidate = str(ranked_rows[0].get(label_key, "")).strip()
    return candidate or None


INTERACTIVE_GRAPH_TOOL_CATALOG = [
    {
        "tool": "pagerank_degree_centrality",
        "output_key": "pagerank",
        "purpose": "识别图谱核心实体、关键节点和高连接对象。",
        "requires": ["triples"],
        "keywords": ["核心", "中心", "重要", "连接", "度数", "pagerank", "PageRank", "关键节点"],
        "supported_intents": ["graph_centrality"],
        "base_priority": 72,
    },
    {
        "tool": "multi_relation_similarity",
        "output_key": "multi_relation_similarity",
        "purpose": "融合症状、检查、治疗、并发、风险因素等多关系特征做疾病相似度。",
        "requires": ["triples"],
        "keywords": ["相似", "共现", "共享", "共同", "相近", "疾病对"],
        "supported_intents": ["association"],
        "base_priority": 70,
    },
    {
        "tool": "disease_community_detection",
        "output_key": "disease_communities",
        "purpose": "基于多关系共享特征发现疾病社区、簇内共性和桥接疾病。",
        "requires": ["triples"],
        "keywords": ["社区", "社群", "聚类", "分群", "模块", "疾病簇", "群落", "结构"],
        "supported_intents": ["association"],
        "base_priority": 69,
    },
    {
        "tool": "symptom_jaccard_similarity",
        "output_key": "disease_similarity",
        "purpose": "基于症状集合重叠计算疾病 Jaccard 相似度。",
        "requires": ["has_symptom"],
        "keywords": ["症状", "表现", "共现", "共享", "共同"],
        "supported_intents": ["association"],
        "base_priority": 64,
    },
    {
        "tool": "causal_path_search",
        "output_key": "causal_paths",
        "purpose": "搜索危险因素、病因、并发和症状之间的因果路径。",
        "requires": ["causes"],
        "keywords": ["因果", "导致", "原因", "病因", "路径", "传播", "风险链", "链路"],
        "supported_intents": [],
        "base_priority": 68,
    },
    {
        "tool": "relation_family_counter",
        "output_key": "relation_families",
        "purpose": "按临床画像、诊断、治疗管理、因果等关系家族汇总图谱结构。",
        "requires": ["triples"],
        "keywords": ["关系家族", "关系类别", "关系类型", "临床画像", "诊疗管理", "因果", "占比"],
        "supported_intents": [],
        "base_priority": 60,
    },
    {
        "tool": "graph_quality_diagnostics",
        "output_key": "quality_diagnostics",
        "purpose": "诊断低置信、孤立节点、证据缺失和稀疏关系。",
        "requires": ["entities", "triples"],
        "keywords": ["质量", "低置信", "孤立", "稀疏", "证据缺失", "可信", "可靠", "复核"],
        "supported_intents": ["quality"],
        "base_priority": 66,
    },
    {
        "tool": "subgraph_context",
        "output_key": "focus_subgraph",
        "purpose": "围绕问题焦点实体构建局部图谱上下文。",
        "requires": ["triples"],
        "keywords": ["上下文", "局部", "子图", "附近", "相关"],
        "supported_intents": ["entity_focus"],
        "base_priority": 46,
    },
]

CONFIGURED_ANALYSIS_ALLOWED_TABLES = {"triples", "entities", "documents"}
CONFIGURED_ANALYSIS_CHART_TYPES = {
    "bar",
    "ranked_bar",
    "network",
    "table",
    "quality_table",
    "relation_table",
    "schema_summary",
    "empty",
}


def _sanitize_artifact_id(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z_\-]+", "_", value.strip())
    normalized = normalized.strip("_")
    return normalized[:80] or fallback


def _validate_configured_sql(sql: str) -> dict[str, Any]:
    normalized = sql.strip()
    upper_sql = normalized.upper()
    unsafe_tokens = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "REPLACE",
        "TRUNCATE",
        "ATTACH",
        "DETACH",
        "PRAGMA",
        "VACUUM",
        "EXPLAIN",
    ]
    unsafe = [token for token in unsafe_tokens if re.search(rf"\b{token}\b", upper_sql)]
    semicolon_count = normalized.count(";")
    has_multiple_statements = semicolon_count > 1 or (semicolon_count == 1 and not normalized.endswith(";"))
    referenced_tables = sorted(
        {
            match.group(1).lower()
            for match in re.finditer(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", normalized, flags=re.IGNORECASE)
        }
    )
    unknown_tables = [table for table in referenced_tables if table not in CONFIGURED_ANALYSIS_ALLOWED_TABLES]
    starts_with_select = bool(re.match(r"^\s*SELECT\b", normalized, flags=re.IGNORECASE))
    return {
        "valid": bool(
            normalized
            and starts_with_select
            and not unsafe
            and not has_multiple_statements
            and referenced_tables
            and not unknown_tables
        ),
        "starts_with_select": starts_with_select,
        "unsafe_tokens": unsafe,
        "multiple_statements": has_multiple_statements,
        "referenced_tables": referenced_tables,
        "unknown_tables": unknown_tables,
        "allowed_tables": sorted(CONFIGURED_ANALYSIS_ALLOWED_TABLES),
    }


def _resolve_configured_queries(config: dict[str, Any]) -> list[dict[str, Any]]:
    raw_queries = config.get("queries")
    if raw_queries is None:
        raw_queries = [{"query_id": item["query_id"]} for item in _analysis_query_catalog()[:4]]
    if not isinstance(raw_queries, list) or not raw_queries:
        raise ValueError("configured_analysis_requires_non_empty_queries")

    catalog = {item["query_id"]: item for item in _analysis_query_catalog()}
    chart_overrides = config.get("chart_overrides", {})
    if chart_overrides is None:
        chart_overrides = {}
    if not isinstance(chart_overrides, dict):
        raise ValueError("chart_overrides_must_be_object")

    queries: list[dict[str, Any]] = []
    for index, raw_query in enumerate(raw_queries, start=1):
        if not isinstance(raw_query, dict):
            raise ValueError("configured_query_must_be_object")
        query_id = str(raw_query.get("query_id", "")).strip()
        if query_id:
            if query_id not in catalog:
                raise ValueError(f"unknown_configured_query_id:{query_id}")
            query = dict(catalog[query_id])
            query["source"] = "catalog"
        else:
            sql = str(raw_query.get("sql", "")).strip()
            name = str(raw_query.get("name") or raw_query.get("id") or f"custom_query_{index}")
            query = {
                "query_id": _sanitize_artifact_id(name, f"custom_query_{index}"),
                "question": str(raw_query.get("question") or raw_query.get("description") or name),
                "sql": sql,
                "chart": _sanitize_artifact_id(str(raw_query.get("chart") or name), f"custom_chart_{index}"),
                "source": "custom_sql",
            }
        if raw_query.get("question"):
            query["question"] = str(raw_query["question"])
        if raw_query.get("chart"):
            query["chart"] = _sanitize_artifact_id(str(raw_query["chart"]), f"configured_chart_{index}")

        override = chart_overrides.get(query["query_id"], chart_overrides.get(query.get("chart", "")))
        if isinstance(override, str) and override:
            if override in CONFIGURED_ANALYSIS_CHART_TYPES:
                query["chart_type_override"] = override
            else:
                query["chart"] = _sanitize_artifact_id(override, query.get("chart") or f"configured_chart_{index}")

        validation = _validate_configured_sql(str(query.get("sql", "")))
        query["sql_validation"] = validation
        if not validation["valid"]:
            raise ValueError(f"invalid_configured_sql:{query['query_id']}:{validation}")
        queries.append(query)
    return queries


def _relation_counts_from_triples(triples: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(item.get("predicate", "")) for item in triples if item.get("predicate"))


def _candidate_graph_tool_plan(
    question: str,
    schema_links: dict[str, Any],
    query: dict[str, str] | None,
    *,
    triples: list[dict[str, Any]] | None = None,
    entities: list[dict[str, Any]] | None = None,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized = re.sub(r"\s+", "", question)
    relation_counts = _relation_counts_from_triples(triples or [])
    schema_intents = schema_links.get("intents", {})
    rows = rows or []
    diagnostics = {
        "sql_missing": query is None,
        "sql_empty_result": query is not None and not rows,
        "has_focus_entity": bool(schema_intents.get("entity_focus")),
        "relation_counts": dict(relation_counts),
        "entity_count": len(entities or []),
        "triple_count": len(triples or []),
    }

    candidates: list[dict[str, Any]] = []
    for spec in INTERACTIVE_GRAPH_TOOL_CATALOG:
        reasons: list[str] = []
        score = float(spec["base_priority"])
        keyword_hits = [keyword for keyword in spec["keywords"] if keyword in normalized]
        if keyword_hits:
            score += min(24.0, 8.0 * len(keyword_hits))
            reasons.append("question_keyword_match:" + ",".join(keyword_hits[:4]))
        explicit_intent_hit = bool(keyword_hits)
        for intent in spec["supported_intents"]:
            if schema_intents.get(intent):
                score += 12.0
                reasons.append(f"schema_intent:{intent}")
        requirements = spec.get("requires", [])
        unavailable = [
            requirement
            for requirement in requirements
            if requirement not in {"triples", "entities"} and relation_counts.get(requirement, 0) == 0
        ]
        if unavailable:
            score -= 12.0 if explicit_intent_hit else 35.0
            reasons.append("missing_relation:" + ",".join(unavailable))
        if "triples" in requirements and triples is not None and not triples:
            score -= 40.0
            reasons.append("missing_triples")
        if "entities" in requirements and entities is not None and not entities:
            score -= 20.0
            reasons.append("missing_entities")
        if query:
            query_id = query.get("query_id")
            if query_id == "entity_centrality" and spec["tool"] == "pagerank_degree_centrality":
                score += 18.0
                reasons.append("nl2sql_query_supports_centrality")
            elif query_id == "symptom_cooccurrence" and spec["tool"] in {"multi_relation_similarity", "symptom_jaccard_similarity"}:
                score += 16.0
                reasons.append("nl2sql_query_supports_similarity")
            elif query_id == "symptom_cooccurrence" and spec["tool"] == "disease_community_detection":
                score += 12.0
                reasons.append("nl2sql_query_supports_community_detection")
            elif query_id in {"low_confidence_relations", "high_confidence_relation_counts"} and spec["tool"] == "graph_quality_diagnostics":
                score += 14.0
                reasons.append("nl2sql_query_supports_quality_review")
            elif query_id == "relation_family_distribution" and spec["tool"] == "relation_family_counter":
                score += 14.0
                reasons.append("nl2sql_query_supports_relation_family")
            elif query_id in {"disease_risk_factor_counts", "risk_factor_pairs"} and spec["tool"] == "causal_path_search":
                score += 10.0
                reasons.append("nl2sql_query_supports_risk_path")
        if diagnostics["sql_missing"]:
            if spec["tool"] in {"pagerank_degree_centrality", "multi_relation_similarity", "disease_community_detection", "relation_family_counter", "graph_quality_diagnostics"}:
                score += 12.0
                reasons.append("sql_missing_graph_fallback")
        if diagnostics["sql_empty_result"]:
            if spec["tool"] in {"subgraph_context", "relation_family_counter", "graph_quality_diagnostics"}:
                score += 14.0
                reasons.append("empty_sql_result_repair")
        if spec["tool"] == "subgraph_context" and diagnostics["has_focus_entity"]:
            score += 18.0
            reasons.append("focus_entity_available")
        if not reasons:
            reasons.append("background_candidate")
        candidates.append(
            {
                "tool": spec["tool"],
                "output_key": spec["output_key"],
                "purpose": spec["purpose"],
                "depends_on": spec["requires"],
                "score": round(max(0.0, score), 4),
                "reason": ";".join(reasons),
                "priority": int(round(max(0.0, score))),
            }
        )

    candidates.sort(key=lambda item: (-float(item["score"]), item["tool"]))
    selected: list[dict[str, Any]] = []
    selected_tools: set[str] = set()
    for candidate in candidates:
        if candidate["score"] < 58 and selected:
            continue
        if candidate["tool"] in selected_tools:
            continue
        selected.append(candidate)
        selected_tools.add(candidate["tool"])
        if len(selected) >= 4:
            break
    if not selected:
        selected = [
            {
                "tool": "subgraph_context",
                "output_key": "focus_subgraph",
                "purpose": "默认图谱上下文回退。",
                "depends_on": ["triples"],
                "score": 50.0,
                "reason": "default_graph_context_for_interactive_insight",
                "priority": 50,
            }
        ]

    repair_actions: list[dict[str, Any]] = []
    if diagnostics["sql_missing"]:
        repair_actions.append(
            {
                "issue": "nl2sql_template_missing",
                "action": "switch_to_graph_algorithm_plan",
                "selected_tools": [item["tool"] for item in selected],
            }
        )
    if diagnostics["sql_empty_result"]:
        repair_actions.append(
            {
                "issue": "nl2sql_empty_result",
                "action": "augment_with_graph_context_and_quality_tools",
                "selected_tools": [item["tool"] for item in selected],
            }
        )
    return {
        "planner": "score_based_graph_tool_planner",
        "diagnostics": diagnostics,
        "candidate_tools": candidates,
        "selected_tools": selected,
        "repair_actions": repair_actions,
        "selection_policy": "score, schema support, SQL diagnostics, and graph relation availability",
    }


def _select_interactive_graph_tools(
    question: str,
    schema_links: dict[str, Any],
    query: dict[str, str] | None,
    *,
    triples: list[dict[str, Any]] | None = None,
    entities: list[dict[str, Any]] | None = None,
    rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    return _candidate_graph_tool_plan(
        question,
        schema_links,
        query,
        triples=triples,
        entities=entities,
        rows=rows,
    )["selected_tools"]


def _select_interactive_graph_tools_legacy(question: str, schema_links: dict[str, Any], query: dict[str, str] | None) -> list[dict[str, Any]]:
    normalized = re.sub(r"\s+", "", question)
    selected: list[dict[str, Any]] = []

    def add(tool: str, reason: str, output_key: str, priority: int) -> None:
        if tool not in {item["tool"] for item in selected}:
            selected.append({"tool": tool, "reason": reason, "output_key": output_key, "priority": priority})

    if any(keyword in normalized for keyword in ("核心", "中心", "重要", "连接", "度数", "pagerank", "PageRank")):
        add("pagerank_degree_centrality", "question_requests_core_graph_entities", "pagerank", 96)
    if any(keyword in normalized for keyword in ("相似", "共现", "共享", "共同", "相近")):
        add("multi_relation_similarity", "question_requests_association_or_similarity", "multi_relation_similarity", 92)
        add("symptom_jaccard_similarity", "question_requests_symptom_overlap", "disease_similarity", 86)
    if any(keyword in normalized for keyword in ("社区", "社群", "聚类", "分群", "模块", "疾病簇")):
        add("disease_community_detection", "question_requests_disease_community_detection", "disease_communities", 91)
    if any(keyword in normalized for keyword in ("因果", "导致", "原因", "病因", "路径", "传播", "风险链")):
        add("causal_path_search", "question_requests_causal_or_path_reasoning", "causal_paths", 94)
    if any(keyword in normalized for keyword in ("关系家族", "关系类别", "关系类型", "临床画像", "诊疗管理")):
        add("relation_family_counter", "question_requests_relation_family_mix", "relation_families", 88)
    if any(keyword in normalized for keyword in ("质量", "低置信", "孤立", "稀疏", "证据缺失", "可信")):
        add("graph_quality_diagnostics", "question_requests_quality_diagnostics", "quality_diagnostics", 90)
    if query and query.get("query_id") in {"entity_centrality"}:
        add("pagerank_degree_centrality", "nl2sql_query_is_centrality", "pagerank", 89)
    if query and query.get("query_id") in {"symptom_cooccurrence"}:
        add("multi_relation_similarity", "nl2sql_query_is_cooccurrence", "multi_relation_similarity", 87)
    if schema_links.get("intents", {}).get("graph_centrality"):
        add("pagerank_degree_centrality", "schema_linked_graph_centrality_intent", "pagerank", 84)

    if not selected:
        add("subgraph_context", "default_graph_context_for_interactive_insight", "focus_subgraph", 50)
    return sorted(selected, key=lambda item: (-item["priority"], item["tool"]))


def _graph_insights_cache_key(triples: list[dict[str, Any]], entities: list[dict[str, Any]]) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"triples:{len(triples)}|entities:{len(entities)}".encode("utf-8"))
    for triple in sorted(
        triples,
        key=lambda item: (
            str(item.get("subject", "")),
            str(item.get("subject_type", "")),
            str(item.get("predicate", "")),
            str(item.get("object", "")),
            str(item.get("object_type", "")),
            str(item.get("source_doc_id", "")),
        ),
    ):
        payload = {
            "s": triple.get("subject", ""),
            "st": triple.get("subject_type", ""),
            "p": triple.get("predicate", ""),
            "o": triple.get("object", ""),
            "ot": triple.get("object_type", ""),
            "c": round(float(triple.get("confidence", 0) or 0), 4),
            "d": triple.get("source_doc_id", ""),
        }
        hasher.update(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    for entity in sorted(
        entities,
        key=lambda item: (str(item.get("name", "")), str(item.get("type", ""))),
    ):
        payload = {
            "n": entity.get("name", ""),
            "t": entity.get("type", ""),
            "c": round(float(entity.get("confidence", 0) or 0), 4),
        }
        hasher.update(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return hasher.hexdigest()


def _get_cached_graph_insights(
    triples: list[dict[str, Any]],
    entities: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    cache_key = _graph_insights_cache_key(triples, entities)
    cached = GRAPH_INSIGHTS_CACHE.get(cache_key)
    if cached is not None:
        GRAPH_INSIGHTS_CACHE.move_to_end(cache_key)
        return cached, {
            "enabled": True,
            "hit": True,
            "cache_key": cache_key[:16],
            "cache_size": len(GRAPH_INSIGHTS_CACHE),
        }
    graph_insights = _build_graph_insights(triples)
    graph_insights["quality_diagnostics"] = _diagnose_graph_quality(entities, triples, graph_insights)
    GRAPH_INSIGHTS_CACHE[cache_key] = graph_insights
    GRAPH_INSIGHTS_CACHE.move_to_end(cache_key)
    while len(GRAPH_INSIGHTS_CACHE) > GRAPH_INSIGHTS_CACHE_MAX_SIZE:
        GRAPH_INSIGHTS_CACHE.popitem(last=False)
    return graph_insights, {
        "enabled": True,
        "hit": False,
        "cache_key": cache_key[:16],
        "cache_size": len(GRAPH_INSIGHTS_CACHE),
    }


def _execute_interactive_graph_tools(
    *,
    question: str,
    triples: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    graph_tools: list[dict[str, Any]],
    focus_disease: str | None,
    subgraph: dict[str, Any],
) -> dict[str, Any]:
    graph_insights, cache_status = _get_cached_graph_insights(triples, entities)
    outputs: dict[str, Any] = {}
    for tool in graph_tools:
        output_key = tool["output_key"]
        if output_key == "pagerank":
            outputs[output_key] = graph_insights.get("pagerank", [])[:10]
            outputs["top_degree"] = graph_insights.get("top_degree", [])[:10]
        elif output_key == "multi_relation_similarity":
            outputs[output_key] = graph_insights.get("multi_relation_similarity", [])[:10]
        elif output_key == "disease_communities":
            outputs[output_key] = graph_insights.get("disease_community_detection", [])[:10]
        elif output_key == "disease_similarity":
            outputs[output_key] = graph_insights.get("disease_similarity", [])[:10]
        elif output_key == "causal_paths":
            causal_paths = graph_insights.get("causal_paths", [])
            if focus_disease:
                focused = [
                    item for item in causal_paths
                    if focus_disease in item.get("nodes", []) or item.get("start") == focus_disease or item.get("end") == focus_disease
                ]
                outputs[output_key] = focused[:10] or causal_paths[:10]
            else:
                outputs[output_key] = causal_paths[:10]
        elif output_key == "relation_families":
            outputs[output_key] = graph_insights.get("relation_families", {})
        elif output_key == "quality_diagnostics":
            outputs[output_key] = graph_insights.get("quality_diagnostics", {})
        elif output_key == "focus_subgraph":
            outputs[output_key] = subgraph
    return {
        "question": question,
        "selected_tools": graph_tools,
        "outputs": outputs,
        "metrics": {
            "tool_count": len(graph_tools),
            "output_keys": sorted(outputs),
            "node_count": graph_insights.get("node_count", 0),
            "edge_count": graph_insights.get("edge_count", 0),
            "causal_path_count": len(graph_insights.get("causal_paths", [])),
            "multi_relation_similarity_count": len(graph_insights.get("multi_relation_similarity", [])),
            "disease_community_count": len(graph_insights.get("disease_community_detection", [])),
            "graph_insights_cache": cache_status,
        },
    }


def _summarize_graph_tool_outputs(graph_tool_result: dict[str, Any]) -> str:
    outputs = graph_tool_result.get("outputs", {})
    parts: list[str] = []
    pagerank = outputs.get("pagerank")
    if isinstance(pagerank, list) and pagerank:
        first = pagerank[0]
        parts.append(f"图算法识别的核心实体是 {first.get('entity')}（PageRank {first.get('score')}）。")
    similarity = outputs.get("multi_relation_similarity")
    if isinstance(similarity, list) and similarity:
        first = similarity[0]
        parts.append(
            f"多关系相似度最高的疾病对是 {first.get('disease_a')} 与 {first.get('disease_b')}，"
            f"得分 {first.get('weighted_similarity')}。"
        )
    disease_similarity = outputs.get("disease_similarity")
    if isinstance(disease_similarity, list) and disease_similarity:
        first = disease_similarity[0]
        parts.append(
            f"症状重叠最高的疾病对是 {first.get('disease_a')} 与 {first.get('disease_b')}，"
            f"共享症状 {first.get('shared_symptom_count')} 个。"
        )
    disease_communities = outputs.get("disease_communities")
    if isinstance(disease_communities, list) and disease_communities:
        first = disease_communities[0]
        parts.append(
            f"社区发现识别到 {first.get('member_count')} 个疾病组成的疾病簇，"
            f"簇内凝聚度 {first.get('cohesion')}。"
        )
    causal_paths = outputs.get("causal_paths")
    if isinstance(causal_paths, list) and causal_paths:
        first = causal_paths[0]
        parts.append(f"已找到因果路径，例如 {' -> '.join(first.get('nodes', []))}。")
    relation_families = outputs.get("relation_families")
    if isinstance(relation_families, dict) and relation_families:
        top_family, top_count = max(relation_families.items(), key=lambda item: item[1])
        parts.append(f"关系家族中 `{top_family}` 占比最高，包含 {top_count} 条关系。")
    quality = outputs.get("quality_diagnostics")
    if isinstance(quality, dict) and quality:
        parts.append(f"图谱质量分为 {quality.get('quality_score')}，孤立实体 {quality.get('isolated_entity_count')} 个。")
    return "".join(parts)


def _build_interactive_insights(
    *,
    question: str,
    query: dict[str, str] | None,
    rows: list[dict[str, Any]],
    graph_context: dict[str, Any],
    subgraph: dict[str, Any],
    graph_tool_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    label_key, numeric_key = _detect_row_dimensions(rows)
    ranked_rows = rows
    values: list[float] = []
    if numeric_key:
        ranked_rows = sorted(rows, key=lambda row: _numeric_value(row.get(numeric_key)) or 0.0, reverse=True)
        values = [_numeric_value(row.get(numeric_key)) or 0.0 for row in ranked_rows]
    top_items = ranked_rows[:3]
    outliers: list[dict[str, Any]] = []
    if values and len(values) >= 3:
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        std_value = variance ** 0.5
        threshold = mean_value + std_value
        for row in ranked_rows:
            value = _numeric_value(row.get(numeric_key)) if numeric_key else None
            if value is not None and value >= threshold:
                outliers.append(
                    {
                        "row": row,
                        "value": value,
                        "reason": f"{numeric_key} 高于均值+1倍标准差",
                    }
                )
    subgraph_edges = subgraph.get("edges", []) if isinstance(subgraph, dict) else []
    relation_mix = Counter(str(edge.get("predicate", "")) for edge in subgraph_edges if edge.get("predicate"))
    summary_parts = []
    if query is None:
        summary_parts.append("未匹配到结构化查询模板，当前以图谱结构摘要作为回退洞察。")
    elif top_items:
        label = label_key or "item"
        value = numeric_key or "value"
        first = top_items[0]
        summary_parts.append(f"主查询 `{query['query_id']}` 返回 {len(rows)} 行，最高项是 {first.get(label)}={first.get(value, '')}。")
    else:
        summary_parts.append("主查询已执行，但当前图谱没有返回可排名结果。")
    if outliers:
        summary_parts.append(f"检测到 {len(outliers)} 个高值重点项，适合进一步做风险或资源配置分析。")
    if subgraph_edges:
        summary_parts.append(f"已构建焦点子图，包含 {len(subgraph.get('nodes', []))} 个节点、{len(subgraph_edges)} 条边。")
    if graph_tool_result:
        graph_summary = _summarize_graph_tool_outputs(graph_tool_result)
        if graph_summary:
            summary_parts.append(graph_summary)
    summary_parts.append(f"全局图谱上下文为 {graph_context.get('node_count', 0)} 个节点、{graph_context.get('edge_count', 0)} 条边。")
    return {
        "summary": "".join(summary_parts),
        "top_items": top_items,
        "outliers": outliers[:5],
        "subgraph_relation_mix": dict(relation_mix.most_common()),
        "metrics": {
            "row_count": len(rows),
            "top_item_count": len(top_items),
            "outlier_count": len(outliers),
            "subgraph_edge_count": len(subgraph_edges),
            "graph_tool_count": graph_tool_result.get("metrics", {}).get("tool_count", 0) if graph_tool_result else 0,
        },
    }


def _chart_type_from_payload(name: str, payload: dict[str, Any]) -> str:
    if payload.get("chart_type"):
        return str(payload["chart_type"])
    if "pairs" in payload or "paths" in payload or "communities" in payload:
        return "network"
    if "rows" in payload and "values" not in payload:
        return "table"
    if "values" in payload:
        return "ranked_bar" if any(token in name for token in ("centrality", "pagerank", "ranking")) else "bar"
    return "object"


def _build_visualization_manifest(
    output_root: Path,
    chart_payloads: dict[str, Any],
    analysis_queries: list[dict[str, Any]],
    subgraphs: list[dict[str, Any]],
    graph_insights: dict[str, Any],
) -> dict[str, Any]:
    query_by_chart = {str(item.get("chart")): item for item in analysis_queries}
    chart_items: list[dict[str, Any]] = []
    for name, raw_payload in sorted(chart_payloads.items()):
        payload = raw_payload if isinstance(raw_payload, dict) else {"value": raw_payload}
        chart_type = _chart_type_from_payload(name, payload)
        query = query_by_chart.get(name, {})
        chart_items.append(
            {
                "name": name,
                "chart_type": chart_type,
                "path": str(output_root / "demo" / "charts" / f"{name}.json"),
                "source_query_id": query.get("query_id"),
                "source_question": query.get("question"),
                "row_count": len(payload.get("rows", []) or payload.get("pairs", []) or payload.get("paths", []) or payload.get("communities", []) or []),
                "supports_bi": chart_type in {"bar", "ranked_bar", "network", "table", "quality_table", "relation_table"},
            }
        )
    graph_visuals = [
        {
            "name": "disease_subgraphs",
            "chart_type": "network",
            "path": str(output_root / "demo" / "subgraphs" / "disease_subgraphs.jsonl"),
            "subgraph_count": len(subgraphs),
            "source": "task2_knowledge_graph",
        },
        {
            "name": "graph_pagerank",
            "chart_type": "ranked_bar",
            "path": str(output_root / "demo" / "charts" / "graph_pagerank.json"),
            "row_count": len(graph_insights.get("pagerank", [])),
            "source": "graph_algorithm_pagerank",
        },
        {
            "name": "multi_relation_similarity",
            "chart_type": "network",
            "path": str(output_root / "demo" / "charts" / "multi_relation_similarity.json"),
            "row_count": len(graph_insights.get("multi_relation_similarity", [])),
            "source": "graph_algorithm_similarity",
        },
        {
            "name": "causal_paths",
            "chart_type": "network",
            "path": str(output_root / "demo" / "charts" / "causal_paths.json"),
            "row_count": len(graph_insights.get("causal_paths", [])),
            "source": "graph_algorithm_path_search",
        },
    ]
    return {
        "manifest_version": "task3_bi_visualization_manifest_v1",
        "chart_count": len(chart_items),
        "bi_chart_count": sum(1 for item in chart_items if item["supports_bi"]),
        "graph_visual_count": len(graph_visuals),
        "chart_types": dict(Counter(item["chart_type"] for item in chart_items)),
        "charts": chart_items,
        "graph_visualizations": graph_visuals,
        "narrative_outputs": [
            str(output_root / "demo" / "analysis_report.md"),
            str(output_root / "demo" / "interactive_analysis_latest.json"),
        ],
    }


def _build_task3_closed_loop_status(
    lineage: dict[str, Any],
    graph_understanding: dict[str, Any],
    analysis_queries: list[dict[str, Any]],
    nl2sql_eval: dict[str, Any],
    visualization_manifest: dict[str, Any],
    graph_insights: dict[str, Any],
) -> dict[str, Any]:
    non_empty_queries = sum(1 for item in analysis_queries if item.get("row_count", 0) > 0)
    blockers: list[str] = []
    if not lineage.get("ready"):
        blockers.extend(str(item) for item in lineage.get("blockers", []))
    if not graph_understanding.get("edge_count"):
        blockers.append("graph_edges_empty")
    if not analysis_queries:
        blockers.append("analysis_queries_empty")
    if not nl2sql_eval.get("meets_threshold"):
        blockers.append("nl2sql_below_85_percent_threshold")
    if visualization_manifest.get("bi_chart_count", 0) < 3:
        blockers.append("insufficient_bi_charts")
    if not graph_insights.get("pagerank"):
        blockers.append("graph_algorithm_pagerank_empty")
    if not graph_insights.get("multi_relation_similarity") and not graph_insights.get("disease_similarity"):
        blockers.append("graph_association_analysis_empty")
    return {
        "closed_loop_version": "task3_data_knowledge_insight_loop_v1",
        "ready": not blockers,
        "blockers": blockers,
        "stages": [
            {"stage": "data", "source": "task1_processed_documents", "status": "ready" if lineage.get("task1", {}).get("documents", 0) else "blocked"},
            {"stage": "knowledge", "source": "task2_knowledge_graph", "status": "ready" if lineage.get("task2", {}).get("triples", 0) else "blocked"},
            {"stage": "analysis", "source": "nl2sql_and_graph_algorithms", "status": "ready" if non_empty_queries else "warning"},
            {"stage": "insight", "source": "bi_charts_graph_visuals_narrative", "status": "ready" if visualization_manifest.get("bi_chart_count", 0) else "blocked"},
        ],
        "metrics": {
            "nl2sql_accuracy": nl2sql_eval.get("accuracy", 0.0),
            "nl2sql_threshold": nl2sql_eval.get("threshold", 0.85),
            "analysis_query_count": len(analysis_queries),
            "non_empty_query_count": non_empty_queries,
            "bi_chart_count": visualization_manifest.get("bi_chart_count", 0),
            "graph_visual_count": visualization_manifest.get("graph_visual_count", 0),
            "graph_node_count": graph_understanding.get("node_count", 0),
            "graph_edge_count": graph_understanding.get("edge_count", 0),
        },
    }


def _build_interactive_plan(
    *,
    question: str,
    query: dict[str, str] | None,
    rows: list[dict[str, Any]],
    disease: str | None,
    focus_disease: str | None,
    chart: dict[str, Any],
    subgraph: dict[str, Any],
    graph_tool_result: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    plan = [
        {
            "order": 1,
            "step": "understand_question",
            "tool": "intent_and_entity_detector",
            "status": "success",
            "outputs": {"detected_disease": disease, "focus_disease": focus_disease},
        },
        {
            "order": 2,
            "step": "profile_graph_schema",
            "tool": "graph_schema_profiler",
            "status": "success",
            "outputs": ["entity_count", "triple_count", "relation_types"],
        },
        {
            "order": 3,
            "step": "run_primary_nl2sql",
            "tool": "nl2sql_sqlite_engine",
            "status": "success" if query is not None else "fallback",
            "outputs": {"query_id": query.get("query_id") if query else None, "row_count": len(rows)},
        },
        {
            "order": 4,
            "step": "detect_key_findings",
            "tool": "rank_outlier_detector",
            "status": "success" if rows else "empty",
            "outputs": ["top_items", "outliers"],
        },
        {
            "order": 5,
            "step": "build_graph_focus",
            "tool": "disease_subgraph_builder",
            "status": "success" if subgraph.get("edges") else "empty",
            "outputs": {"focus_disease": focus_disease, "edge_count": len(subgraph.get("edges", []))},
        },
        {
            "order": 6,
            "step": "run_graph_algorithms",
            "tool": "graph_algorithm_selector",
            "status": "success" if graph_tool_result else "skipped",
            "outputs": graph_tool_result.get("metrics", {}) if graph_tool_result else {},
        },
        {
            "order": 7,
            "step": "render_bi_and_narrative",
            "tool": "chart_and_explanation_renderer",
            "status": "success",
            "outputs": {"chart_type": chart.get("chart_type")},
        },
    ]
    return plan


def _recommended_followups(query: dict[str, str] | None, focus_disease: str | None) -> list[str]:
    followups: list[str] = []
    if focus_disease:
        followups.extend(
            [
                f"{focus_disease}有哪些症状？",
                f"{focus_disease}如何治疗？",
                f"{focus_disease}有哪些危险因素？",
            ]
        )
    if query and query.get("query_id") != "entity_centrality":
        followups.append("图谱里最重要的高连接节点有哪些？")
    followups.append("找出症状共现最强的疾病对")
    return followups[:4]


def run_interactive_analysis(output_root: Path, question: str) -> dict[str, Any]:
    processed_documents, entities, triples, graph_snapshot, qa_traces = _load_artifacts(output_root)
    nl2sql_plan = plan_nl2sql_query(question)
    query = nl2sql_plan["query"]
    conn = _build_sqlite(triples, entities, processed_documents)
    rows: list[dict[str, Any]] = []
    try:
        if query is not None:
            rows = _run_query(conn, query["sql"])
    finally:
        conn.close()

    graph_context = {
        "node_count": len(graph_snapshot.get("nodes", [])) if isinstance(graph_snapshot, dict) else len(entities),
        "edge_count": len(graph_snapshot.get("edges", [])) if isinstance(graph_snapshot, dict) else len(triples),
        "entity_count": len(entities),
        "triple_count": len(triples),
    }
    disease = detect_question_disease(question)
    focus_disease = _select_focus_disease_from_rows(rows, disease)
    subgraph = _build_disease_subgraph(focus_disease, triples) if focus_disease else {}
    chart = _chart_for_query_result(query, rows) if query is not None else {"chart_type": "schema_summary", "rows": []}
    graph_tool_planning = _candidate_graph_tool_plan(
        question,
        nl2sql_plan["schema_links"],
        query,
        triples=triples,
        entities=entities,
        rows=rows,
    )
    graph_tools = graph_tool_planning["selected_tools"]
    graph_tool_result = _execute_interactive_graph_tools(
        question=question,
        triples=triples,
        entities=entities,
        graph_tools=graph_tools,
        focus_disease=focus_disease,
        subgraph=subgraph,
    )
    insights = _build_interactive_insights(
        question=question,
        query=query,
        rows=rows,
        graph_context=graph_context,
        subgraph=subgraph,
        graph_tool_result=graph_tool_result,
    )
    insight_plan = _build_interactive_plan(
        question=question,
        query=query,
        rows=rows,
        disease=disease,
        focus_disease=focus_disease,
        chart=chart,
        subgraph=subgraph,
        graph_tool_result=graph_tool_result,
    )
    insight = {
        "question": question,
        "query": query,
        "rows": rows,
        "row_count": len(rows),
        "chart": chart,
        "subgraph": subgraph,
        "graph_tool_planning": graph_tool_planning,
        "graph_tool_plan": graph_tools,
        "graph_tool_result": graph_tool_result,
        "graph_context": graph_context,
        "nl2sql_plan": {
            "selection_mode": nl2sql_plan["selection_mode"],
            "schema_links": nl2sql_plan["schema_links"],
            "sql_validation": nl2sql_plan["sql_validation"],
            "top_candidates": nl2sql_plan["candidate_scores"][:5],
            "trace": nl2sql_plan["trace"],
        },
        "insight_plan": insight_plan,
        "insights": insights,
        "recommended_followups": _recommended_followups(query, focus_disease),
        "explanation": insights["summary"] + " " + _explain_interactive_result(question, query, rows, graph_context),
        "trace": [
            {
                "step": "understand_analysis_question",
                "status": "success",
                "question": question,
                "detected_disease": disease,
                "focus_disease": focus_disease,
                "schema_links": nl2sql_plan["schema_links"],
            },
            {
                "step": "nl2sql_schema_linking",
                "status": "success",
                "selection_mode": nl2sql_plan["selection_mode"],
                "selected_query_id": query.get("query_id") if query else None,
                "sql_valid": nl2sql_plan["sql_validation"]["valid"],
                "top_candidates": nl2sql_plan["candidate_scores"][:3],
            },
            {
                "step": "plan_tool_calls",
                "status": "success" if query is not None else "fallback",
                "query_id": query.get("query_id") if query else None,
                "tools": [item["tool"] for item in insight_plan],
                "graph_tools": [item["tool"] for item in graph_tools],
                "graph_tool_planner": graph_tool_planning["planner"],
                "repair_actions": graph_tool_planning["repair_actions"],
            },
            {
                "step": "execute_analysis",
                "status": (
                    "success" if query is not None and rows else
                    "empty_result_repaired" if query is not None else
                    "no_sql_template_repaired"
                ),
                "row_count": len(rows),
            },
            {
                "step": "execute_graph_algorithms",
                "status": "success" if graph_tool_result.get("metrics", {}).get("tool_count", 0) else "skipped",
                "selected_tools": [item["tool"] for item in graph_tools],
                "output_keys": graph_tool_result.get("metrics", {}).get("output_keys", []),
                "graph_insights_cache": graph_tool_result.get("metrics", {}).get("graph_insights_cache", {}),
            },
            {
                "step": "detect_key_findings",
                "status": "success" if rows else "empty",
                "top_item_count": insights["metrics"]["top_item_count"],
                "outlier_count": insights["metrics"]["outlier_count"],
            },
            {
                "step": "render_interactive_insight",
                "status": "success",
                "chart_type": chart.get("chart_type"),
                "subgraph_edge_count": insights["metrics"]["subgraph_edge_count"],
            },
        ],
    }
    trace_path = output_root / "demo" / "interactive_analysis_trace.jsonl"
    existing = _read_jsonl(trace_path) if trace_path.exists() else []
    existing.append(
        {
            "agent": "analysis_agent",
            "goal": "交互式自然语言洞察分析",
            "question": question,
            "query_id": query.get("query_id") if query else None,
            "row_count": len(rows),
            "chart_type": chart.get("chart_type"),
            "focus_disease": focus_disease,
            "insight_metrics": insights["metrics"],
            "graph_tool_metrics": graph_tool_result.get("metrics", {}),
            "graph_tool_planner": graph_tool_planning,
            "nl2sql_selection_mode": nl2sql_plan["selection_mode"],
            "trace": insight["trace"],
        }
    )
    _write_jsonl(trace_path, existing[-200:])
    _write_json(output_root / "demo" / "interactive_analysis_latest.json", insight)
    return insight


def _resolve_configured_graph_tools(
    *,
    config: dict[str, Any],
    question_context: str,
    query: dict[str, Any] | None,
    rows: list[dict[str, Any]],
    triples: list[dict[str, Any]],
    entities: list[dict[str, Any]],
) -> dict[str, Any]:
    requested_tools = config.get("graph_tools")
    schema_links = _schema_link_question(question_context)
    if requested_tools is None:
        return _candidate_graph_tool_plan(
            question_context,
            schema_links,
            query,
            triples=triples,
            entities=entities,
            rows=rows,
        )
    if not isinstance(requested_tools, list):
        raise ValueError("graph_tools_must_be_list")

    catalog = {item["tool"]: item for item in INTERACTIVE_GRAPH_TOOL_CATALOG}
    selected: list[dict[str, Any]] = []
    for index, raw_tool in enumerate(requested_tools, start=1):
        if isinstance(raw_tool, str):
            tool_name = raw_tool
            reason = "configured_graph_tool"
            priority = 100 - index
        elif isinstance(raw_tool, dict):
            tool_name = str(raw_tool.get("tool", ""))
            reason = str(raw_tool.get("reason", "configured_graph_tool"))
            priority = int(raw_tool.get("priority", 100 - index))
        else:
            raise ValueError("configured_graph_tool_must_be_string_or_object")
        if tool_name not in catalog:
            raise ValueError(f"unknown_configured_graph_tool:{tool_name}")
        spec = catalog[tool_name]
        if tool_name in {item["tool"] for item in selected}:
            continue
        selected.append(
            {
                "tool": tool_name,
                "output_key": spec["output_key"],
                "purpose": spec["purpose"],
                "depends_on": spec["requires"],
                "score": float(priority),
                "reason": reason,
                "priority": priority,
            }
        )

    return {
        "planner": "configuration_graph_tool_planner",
        "diagnostics": {
            "configured": True,
            "entity_count": len(entities),
            "triple_count": len(triples),
            "relation_counts": dict(_relation_counts_from_triples(triples)),
        },
        "candidate_tools": selected,
        "selected_tools": selected,
        "repair_actions": [],
        "selection_policy": "explicit configuration with tool catalog validation",
    }


def run_configured_analysis(output_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("configured_analysis_config_must_be_object")

    processed_documents, entities, triples, graph_snapshot, _qa_traces = _load_artifacts(output_root)
    analysis_id = _sanitize_artifact_id(str(config.get("analysis_id") or config.get("name") or "configured_analysis"), "configured_analysis")
    questions = [
        str(item.get("question") or item.get("query_id") or item.get("name") or "")
        for item in config.get("queries", [])
        if isinstance(item, dict)
    ]
    if config.get("goal"):
        questions.insert(0, str(config["goal"]))
    if config.get("focus_disease"):
        questions.append(str(config["focus_disease"]))
    question_context = " ".join(item for item in questions if item).strip() or "配置化图谱分析"

    trace: list[dict[str, Any]] = [
        {
            "agent": "analysis_agent",
            "goal": "配置化图谱驱动分析",
            "step": "load_config",
            "status": "success",
            "analysis_id": analysis_id,
            "configured_query_count": len(config.get("queries", []) or []),
            "configured_graph_tool_count": len(config.get("graph_tools", []) or []) if isinstance(config.get("graph_tools"), list) else 0,
        }
    ]
    configured_queries = _resolve_configured_queries(config)
    trace.append(
        {
            "agent": "analysis_agent",
            "goal": "配置化图谱驱动分析",
            "step": "validate_config_queries",
            "status": "success",
            "query_count": len(configured_queries),
            "allowed_tables": sorted(CONFIGURED_ANALYSIS_ALLOWED_TABLES),
            "query_ids": [item["query_id"] for item in configured_queries],
        }
    )

    conn = _build_sqlite(triples, entities, processed_documents)
    query_results: list[dict[str, Any]] = []
    charts: list[dict[str, Any]] = []
    try:
        for query in configured_queries:
            rows = _run_query(conn, str(query["sql"]))
            chart = _chart_for_query_result(query, rows)
            if query.get("chart_type_override"):
                chart["chart_type"] = query["chart_type_override"]
            query_results.append(
                {
                    "query_id": query["query_id"],
                    "question": query["question"],
                    "source": query["source"],
                    "sql": query["sql"],
                    "sql_validation": query["sql_validation"],
                    "rows": rows,
                    "row_count": len(rows),
                    "chart": query["chart"],
                }
            )
            chart_path = output_root / "demo" / "charts" / f"configured_{analysis_id}_{query['chart']}.json"
            _write_json(chart_path, chart)
            charts.append(
                {
                    "query_id": query["query_id"],
                    "chart": query["chart"],
                    "chart_type": chart.get("chart_type"),
                    "path": str(chart_path),
                    "payload": chart,
                }
            )
    finally:
        conn.close()

    trace.append(
        {
            "agent": "analysis_agent",
            "goal": "配置化图谱驱动分析",
            "step": "execute_configured_queries",
            "status": "success",
            "query_count": len(query_results),
            "row_count": sum(item["row_count"] for item in query_results),
            "non_empty_query_count": sum(1 for item in query_results if item["row_count"] > 0),
        }
    )

    explicit_focus = str(config.get("focus_disease", "")).strip() or None
    detected_focus = detect_question_disease(question_context)
    first_rows = next((item["rows"] for item in query_results if item["rows"]), [])
    focus_disease = _select_focus_disease_from_rows(first_rows, explicit_focus or detected_focus)
    subgraph = _build_disease_subgraph(focus_disease, triples) if focus_disease else {}
    representative_query = configured_queries[0] if configured_queries else None
    representative_rows = query_results[0]["rows"] if query_results else []
    graph_tool_planning = _resolve_configured_graph_tools(
        config=config,
        question_context=question_context,
        query=representative_query,
        rows=representative_rows,
        triples=triples,
        entities=entities,
    )
    graph_tool_result = _execute_interactive_graph_tools(
        question=question_context,
        triples=triples,
        entities=entities,
        graph_tools=graph_tool_planning["selected_tools"],
        focus_disease=focus_disease,
        subgraph=subgraph,
    )
    trace.append(
        {
            "agent": "analysis_agent",
            "goal": "配置化图谱驱动分析",
            "step": "execute_configured_graph_tools",
            "status": "success" if graph_tool_result.get("metrics", {}).get("tool_count", 0) else "skipped",
            "selected_tools": [item["tool"] for item in graph_tool_planning["selected_tools"]],
            "output_keys": graph_tool_result.get("metrics", {}).get("output_keys", []),
            "graph_insights_cache": graph_tool_result.get("metrics", {}).get("graph_insights_cache", {}),
        }
    )

    graph_context = {
        "node_count": len(graph_snapshot.get("nodes", [])) if isinstance(graph_snapshot, dict) else len(entities),
        "edge_count": len(graph_snapshot.get("edges", [])) if isinstance(graph_snapshot, dict) else len(triples),
        "entity_count": len(entities),
        "triple_count": len(triples),
        "schema_summary": _graph_schema_summary(entities, triples),
    }
    top_query = next((item for item in query_results if item["row_count"] > 0), query_results[0] if query_results else {})
    top_rows = top_query.get("rows", []) if isinstance(top_query, dict) else []
    label_key, numeric_key = _detect_row_dimensions(top_rows)
    top_findings = []
    if label_key and top_rows:
        ranked_rows = top_rows
        if numeric_key:
            ranked_rows = sorted(top_rows, key=lambda row: _numeric_value(row.get(numeric_key)) or 0.0, reverse=True)
        top_findings = ranked_rows[:5]

    summary = {
        "analysis_id": analysis_id,
        "query_count": len(query_results),
        "non_empty_query_count": sum(1 for item in query_results if item["row_count"] > 0),
        "total_row_count": sum(item["row_count"] for item in query_results),
        "chart_count": len(charts),
        "graph_tool_count": graph_tool_result.get("metrics", {}).get("tool_count", 0),
        "graph_output_keys": graph_tool_result.get("metrics", {}).get("output_keys", []),
        "focus_disease": focus_disease,
        "top_findings": top_findings,
        "explanation": (
            f"配置 `{analysis_id}` 执行 {len(query_results)} 个分析查询，"
            f"生成 {len(charts)} 个图表，并调度 {graph_tool_result.get('metrics', {}).get('tool_count', 0)} 个图谱算法工具。"
            + (_summarize_graph_tool_outputs(graph_tool_result) or "")
        ),
    }
    result = {
        "analysis_id": analysis_id,
        "mode": "configured",
        "config": config,
        "queries": query_results,
        "charts": charts,
        "graph_context": graph_context,
        "graph_tool_planning": graph_tool_planning,
        "graph_tool_plan": graph_tool_planning["selected_tools"],
        "graph_tool_result": graph_tool_result,
        "focus_disease": focus_disease,
        "subgraph": subgraph,
        "summary": summary,
        "trace": trace + [
            {
                "agent": "analysis_agent",
                "goal": "配置化图谱驱动分析",
                "step": "render_configured_analysis",
                "status": "success",
                "chart_count": len(charts),
            }
        ],
        "artifact_paths": {
            "latest": str(output_root / "demo" / "configured_analysis_latest.json"),
            "trace": str(output_root / "demo" / "configured_analysis_trace.jsonl"),
            "charts": [item["path"] for item in charts],
        },
    }
    _write_json(output_root / "demo" / "configured_analysis_latest.json", result)
    trace_path = output_root / "demo" / "configured_analysis_trace.jsonl"
    existing = _read_jsonl(trace_path) if trace_path.exists() else []
    existing.append(
        {
            "agent": "analysis_agent",
            "goal": "配置化图谱驱动分析",
            "analysis_id": analysis_id,
            "summary": summary,
            "graph_tool_planner": graph_tool_planning,
            "trace": result["trace"],
        }
    )
    _write_jsonl(trace_path, existing[-200:])
    return result


def _build_disease_subgraph(disease: str, triples: list[dict[str, Any]], limit: int = 12) -> dict[str, Any]:
    selected = [item for item in triples if item.get("subject") == disease][:limit]
    nodes: dict[tuple[str, str], dict[str, Any]] = {
        (disease, "Disease"): {"name": disease, "type": "Disease"}
    }
    for triple in selected:
        nodes[(triple["object"], triple["object_type"])] = {
            "name": triple["object"],
            "type": triple["object_type"],
        }
    ordered_nodes = [
        {"id": f"N{index:03d}", "name": node["name"], "type": node["type"]}
        for index, node in enumerate(sorted(nodes.values(), key=lambda item: (item["type"], item["name"])), start=1)
    ]
    node_lookup = {(node["name"], node["type"]): node["id"] for node in ordered_nodes}
    edges = [
        {
            "id": triple["triple_id"],
            "source": node_lookup[(triple["subject"], triple["subject_type"])],
            "target": node_lookup[(triple["object"], triple["object_type"])],
            "predicate": triple["predicate"],
            "source_doc_id": triple["source_doc_id"],
            "confidence": triple["confidence"],
            "evidence": triple.get("evidence", ""),
        }
        for triple in selected
        if (triple["subject"], triple["subject_type"]) in node_lookup
        and (triple["object"], triple["object_type"]) in node_lookup
    ]
    return {
        "focus_disease": disease,
        "nodes": ordered_nodes,
        "edges": edges,
        "source_triples": selected,
    }


def _node_key(name: str, entity_type: str) -> str:
    return f"{entity_type}::{name}"


def _split_node_key(key: str) -> tuple[str, str]:
    entity_type, _, name = key.partition("::")
    return name, entity_type


SIMILARITY_RELATION_WEIGHTS = {
    "has_symptom": 0.32,
    "diagnosed_by": 0.16,
    "treated_by": 0.18,
    "has_complication": 0.14,
    "has_risk_factor": 0.12,
    "causes": 0.08,
}


def _build_causal_paths(triples: list[dict[str, Any]], *, max_depth: int = 3, limit: int = 12) -> list[dict[str, Any]]:
    adjacency: dict[str, list[dict[str, Any]]] = {}
    for triple in triples:
        if triple.get("predicate") != "causes":
            continue
        subject = str(triple.get("subject", "")).strip()
        obj = str(triple.get("object", "")).strip()
        if not subject or not obj or subject == obj:
            continue
        adjacency.setdefault(subject, []).append(triple)

    paths: list[dict[str, Any]] = []
    for start in sorted(adjacency):
        queue: list[tuple[str, list[dict[str, Any]], set[str]]] = [(start, [], {start})]
        while queue and len(paths) < limit * 4:
            current, path, visited = queue.pop(0)
            if len(path) >= max_depth:
                continue
            for edge in adjacency.get(current, []):
                target = str(edge.get("object", "")).strip()
                if not target or target in visited:
                    continue
                next_path = path + [edge]
                if len(next_path) >= 2 or edge.get("object_type") in {"Disease", "Symptom"}:
                    confidence_values = [float(item.get("confidence", 0)) for item in next_path]
                    paths.append(
                        {
                            "start": start,
                            "end": target,
                            "length": len(next_path),
                            "score": round(sum(confidence_values) / len(confidence_values), 4),
                            "nodes": [start] + [str(item.get("object", "")) for item in next_path],
                            "edges": [
                                {
                                    "subject": item.get("subject"),
                                    "predicate": item.get("predicate"),
                                    "object": item.get("object"),
                                    "confidence": item.get("confidence"),
                                    "evidence": item.get("evidence", ""),
                                }
                                for item in next_path
                            ],
                        }
                    )
                queue.append((target, next_path, visited | {target}))
    paths.sort(key=lambda item: (-item["score"], -item["length"], item["start"], item["end"]))
    return paths[:limit]


def _build_multi_relation_similarity(triples: list[dict[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    rows, _ = _build_multi_relation_similarity_with_metadata(triples, limit=limit)
    return rows


def _task3_tensor_config() -> tuple[str, bool]:
    requested_device = os.getenv("TASK3_ANALYSIS_TENSOR_DEVICE", os.getenv("MEDICAL_KG_TENSOR_DEVICE", "auto")).strip().lower() or "auto"
    enabled = os.getenv("TASK3_ANALYSIS_ENABLE_TENSOR", "true").strip().lower() not in {"0", "false", "no", "off"}
    return requested_device, enabled


def _build_multi_relation_similarity_with_metadata(
    triples: list[dict[str, Any]],
    *,
    limit: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    disease_features: dict[str, dict[str, set[str]]] = {}
    for triple in triples:
        if triple.get("subject_type") != "Disease":
            continue
        predicate = str(triple.get("predicate", ""))
        if predicate not in SIMILARITY_RELATION_WEIGHTS:
            continue
        disease = str(triple.get("subject", "")).strip()
        obj = str(triple.get("object", "")).strip()
        if not disease or not obj:
            continue
        disease_features.setdefault(disease, {}).setdefault(predicate, set()).add(obj)

    diseases = sorted(disease_features)
    requested_device, tensor_enabled = _task3_tensor_config()
    metadata: dict[str, Any] = {
        "operator": "task3_multi_relation_similarity_tensor",
        "enabled": tensor_enabled,
        "requested_device": requested_device,
        "selected_device": "disabled",
        "disease_count": len(diseases),
        "relation_count": len(SIMILARITY_RELATION_WEIGHTS),
        "feature_count": sum(
            len({feature for features in disease_features.values() for feature in features.get(predicate, set())})
            for predicate in SIMILARITY_RELATION_WEIGHTS
        ),
        "pair_count": len(diseases) * (len(diseases) - 1) // 2,
        "fallback": None,
    }
    if tensor_enabled and len(diseases) >= 2:
        try:
            rows = _build_multi_relation_similarity_tensor(
                disease_features,
                diseases=diseases,
                requested_device=requested_device,
                limit=limit,
                metadata=metadata,
            )
            return rows, metadata
        except Exception as exc:  # noqa: BLE001
            if requested_device == "npu":
                raise
            metadata["enabled"] = False
            metadata["selected_device"] = "cpu_rule"
            metadata["fallback"] = f"tensor_failed:{exc.__class__.__name__}"

    rows = _build_multi_relation_similarity_rule(disease_features, diseases=diseases, limit=limit)
    metadata["selected_device"] = "cpu_rule"
    return rows, metadata


def _build_multi_relation_similarity_tensor(
    disease_features: dict[str, dict[str, set[str]]],
    *,
    diseases: list[str],
    requested_device: str,
    limit: int,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    from operators.npu_tensor_kg_ops import make_tensor, select_tensor_device, sync_tensor_device

    selected_device = select_tensor_device(requested_device)
    metadata["selected_device"] = selected_device
    started = perf_counter()
    relation_score_matrices: dict[str, list[list[float]]] = {}
    for predicate, weight in SIMILARITY_RELATION_WEIGHTS.items():
        features = sorted({feature for values in disease_features.values() for feature in values.get(predicate, set())})
        if not features:
            continue
        feature_index = {feature: index for index, feature in enumerate(features)}
        rows: list[list[float]] = []
        for disease in diseases:
            vector = [0.0] * len(features)
            for feature in disease_features[disease].get(predicate, set()):
                vector[feature_index[feature]] = 1.0
            rows.append(vector)
        feature_tensor = make_tensor(rows, selected_device)
        sync_tensor_device(selected_device)
        intersection = feature_tensor @ feature_tensor.T
        counts = feature_tensor.sum(dim=1, keepdim=True)
        union = counts + counts.T - intersection
        jaccard = intersection / union.clamp_min(1.0)
        weighted = jaccard * float(weight)
        sync_tensor_device(selected_device)
        relation_score_matrices[predicate] = weighted.detach().to("cpu").tolist()

    output: list[dict[str, Any]] = []
    for index, disease_a in enumerate(diseases):
        features_a = disease_features[disease_a]
        for next_index in range(index + 1, len(diseases)):
            disease_b = diseases[next_index]
            features_b = disease_features[disease_b]
            score = 0.0
            shared_features: dict[str, list[str]] = {}
            for predicate in SIMILARITY_RELATION_WEIGHTS:
                matrix = relation_score_matrices.get(predicate)
                if matrix:
                    score += float(matrix[index][next_index])
                shared = features_a.get(predicate, set()) & features_b.get(predicate, set())
                if shared:
                    shared_features[predicate] = sorted(shared)[:8]
            if score <= 0:
                continue
            output.append(
                {
                    "disease_a": disease_a,
                    "disease_b": disease_b,
                    "weighted_similarity": round(score, 6),
                    "shared_relation_count": len(shared_features),
                    "shared_features": shared_features,
                }
            )
    output.sort(key=lambda item: (-item["weighted_similarity"], -item["shared_relation_count"], item["disease_a"], item["disease_b"]))
    metadata["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
    metadata["output_count"] = len(output)
    metadata["tensor_relation_matrices"] = len(relation_score_matrices)
    return output[:limit]


def _build_multi_relation_similarity_rule(
    disease_features: dict[str, dict[str, set[str]]],
    *,
    diseases: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, disease_a in enumerate(diseases):
        features_a = disease_features[disease_a]
        for disease_b in diseases[index + 1:]:
            features_b = disease_features[disease_b]
            score = 0.0
            shared_features: dict[str, list[str]] = {}
            for predicate, weight in SIMILARITY_RELATION_WEIGHTS.items():
                values_a = features_a.get(predicate, set())
                values_b = features_b.get(predicate, set())
                union = values_a | values_b
                if not union:
                    continue
                shared = values_a & values_b
                if shared:
                    shared_features[predicate] = sorted(shared)[:8]
                score += weight * (len(shared) / len(union))
            if score <= 0:
                continue
            rows.append(
                {
                    "disease_a": disease_a,
                    "disease_b": disease_b,
                    "weighted_similarity": round(score, 6),
                    "shared_relation_count": len(shared_features),
                    "shared_features": shared_features,
                }
            )
    rows.sort(key=lambda item: (-item["weighted_similarity"], -item["shared_relation_count"], item["disease_a"], item["disease_b"]))
    return rows[:limit]


def _build_disease_communities(triples: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    disease_features: dict[str, dict[str, set[str]]] = {}
    feature_to_diseases: dict[tuple[str, str], set[str]] = {}
    for triple in triples:
        if triple.get("subject_type") != "Disease":
            continue
        predicate = str(triple.get("predicate", ""))
        if predicate not in SIMILARITY_RELATION_WEIGHTS:
            continue
        disease = str(triple.get("subject", "")).strip()
        obj = str(triple.get("object", "")).strip()
        if not disease or not obj:
            continue
        disease_features.setdefault(disease, {}).setdefault(predicate, set()).add(obj)
        feature_to_diseases.setdefault((predicate, obj), set()).add(disease)

    diseases = sorted(disease_features)
    if not diseases:
        return []

    weighted_edges: dict[tuple[str, str], dict[str, Any]] = {}
    for index, disease_a in enumerate(diseases):
        for disease_b in diseases[index + 1:]:
            shared_features: dict[str, list[str]] = {}
            weight = 0.0
            for predicate, relation_weight in SIMILARITY_RELATION_WEIGHTS.items():
                values_a = disease_features[disease_a].get(predicate, set())
                values_b = disease_features[disease_b].get(predicate, set())
                shared = values_a & values_b
                if not shared:
                    continue
                shared_features[predicate] = sorted(shared)
                weight += relation_weight * len(shared)
            if weight <= 0:
                continue
            key = (disease_a, disease_b)
            weighted_edges[key] = {
                "disease_a": disease_a,
                "disease_b": disease_b,
                "weight": round(weight, 6),
                "shared_features": shared_features,
            }

    labels = {disease: disease for disease in diseases}
    adjacency: dict[str, list[tuple[str, float]]] = {disease: [] for disease in diseases}
    for (disease_a, disease_b), edge in weighted_edges.items():
        weight = float(edge["weight"])
        adjacency[disease_a].append((disease_b, weight))
        adjacency[disease_b].append((disease_a, weight))

    for _ in range(12):
        changed = False
        for disease in diseases:
            label_scores: Counter[str] = Counter()
            for neighbor, weight in adjacency.get(disease, []):
                label_scores[labels[neighbor]] += weight
            if not label_scores:
                continue
            best_score = max(label_scores.values())
            best_labels = sorted(label for label, score in label_scores.items() if score == best_score)
            next_label = best_labels[0]
            if next_label != labels[disease]:
                labels[disease] = next_label
                changed = True
        if not changed:
            break

    grouped: dict[str, list[str]] = {}
    for disease, label in labels.items():
        grouped.setdefault(label, []).append(disease)

    communities: list[dict[str, Any]] = []
    for label, members in grouped.items():
        members = sorted(members)
        if len(members) < 2:
            continue
        member_set = set(members)
        internal_edges = [
            edge
            for (disease_a, disease_b), edge in weighted_edges.items()
            if disease_a in member_set and disease_b in member_set
        ]
        external_edges = [
            edge
            for (disease_a, disease_b), edge in weighted_edges.items()
            if (disease_a in member_set) ^ (disease_b in member_set)
        ]
        shared_feature_counter: Counter[str] = Counter()
        relation_mix: Counter[str] = Counter()
        for edge in internal_edges:
            for predicate, values in edge.get("shared_features", {}).items():
                relation_mix[predicate] += len(values)
                for value in values:
                    shared_feature_counter[f"{predicate}:{value}"] += 1
        bridge_scores: Counter[str] = Counter()
        for edge in internal_edges + external_edges:
            bridge_scores[str(edge["disease_a"])] += float(edge["weight"])
            bridge_scores[str(edge["disease_b"])] += float(edge["weight"])
        bridge_diseases = [
            {"disease": disease, "bridge_score": round(score, 6)}
            for disease, score in bridge_scores.most_common()
            if disease in member_set
        ][:5]
        cohesion = sum(float(edge["weight"]) for edge in internal_edges) / max(1, len(internal_edges))
        communities.append(
            {
                "community_id": f"C{len(communities) + 1:03d}",
                "label": label,
                "members": members,
                "member_count": len(members),
                "internal_edge_count": len(internal_edges),
                "external_edge_count": len(external_edges),
                "cohesion": round(cohesion, 6),
                "relation_mix": dict(relation_mix.most_common()),
                "top_shared_features": [
                    {"feature": feature, "count": count}
                    for feature, count in shared_feature_counter.most_common(10)
                ],
                "bridge_diseases": bridge_diseases,
            }
        )
    communities.sort(key=lambda item: (-item["member_count"], -item["cohesion"], item["label"]))
    return communities[:limit]


def _rank_subgraphs(subgraphs: list[dict[str, Any]], graph_insights: dict[str, Any]) -> list[dict[str, Any]]:
    pagerank = {
        str(item.get("entity")): float(item.get("score", 0))
        for item in graph_insights.get("pagerank", [])
    }
    ranked: list[dict[str, Any]] = []
    for subgraph in subgraphs:
        focus = str(subgraph.get("focus_disease", ""))
        edges = subgraph.get("edges", [])
        relation_diversity = len({edge.get("predicate") for edge in edges})
        evidence_edges = sum(1 for edge in edges if str(edge.get("evidence", "")).strip())
        score = (
            pagerank.get(focus, 0.0) * 100
            + len(edges) * 0.25
            + relation_diversity * 0.8
            + evidence_edges * 0.1
        )
        ranked.append(
            {
                "focus_disease": focus,
                "score": round(score, 6),
                "edge_count": len(edges),
                "node_count": len(subgraph.get("nodes", [])),
                "relation_diversity": relation_diversity,
                "evidence_edges": evidence_edges,
            }
        )
    ranked.sort(key=lambda item: (-item["score"], item["focus_disease"]))
    return ranked


def _diagnose_graph_quality(
    entities: list[dict[str, Any]],
    triples: list[dict[str, Any]],
    graph_insights: dict[str, Any],
) -> dict[str, Any]:
    entity_keys = {(item.get("name"), item.get("type")) for item in entities}
    connected_keys = {
        (triple.get("subject"), triple.get("subject_type"))
        for triple in triples
    } | {
        (triple.get("object"), triple.get("object_type"))
        for triple in triples
    }
    isolated = sorted(
        [
            {"entity": name, "entity_type": entity_type}
            for name, entity_type in entity_keys - connected_keys
            if name and entity_type
        ],
        key=lambda item: (item["entity_type"], item["entity"]),
    )
    low_confidence = [
        triple for triple in triples
        if float(triple.get("confidence", 0)) < 0.8
    ]
    relation_counts = Counter(str(item.get("predicate", "")) for item in triples)
    sparse_relations = [
        {"predicate": predicate, "count": count}
        for predicate, count in sorted(relation_counts.items())
        if count <= 2
    ]
    evidence_missing = [
        triple for triple in triples
        if not str(triple.get("evidence", "")).strip()
    ]
    total_entities = max(1, len(entities))
    total_triples = max(1, len(triples))
    quality_score = 100.0
    quality_score -= min(25.0, len(isolated) / total_entities * 100)
    quality_score -= min(25.0, len(low_confidence) / total_triples * 100)
    quality_score -= min(25.0, len(evidence_missing) / total_triples * 100)
    quality_score -= min(10.0, len(sparse_relations))
    return {
        "quality_score": round(max(0.0, quality_score), 2),
        "isolated_entity_count": len(isolated),
        "isolated_entity_samples": isolated[:20],
        "low_confidence_triple_count": len(low_confidence),
        "missing_evidence_triple_count": len(evidence_missing),
        "sparse_relations": sparse_relations,
        "causal_path_count": len(graph_insights.get("causal_paths", [])),
        "multi_relation_similarity_count": len(graph_insights.get("multi_relation_similarity", [])),
    }


def _build_graph_insights(triples: list[dict[str, Any]], *, top_n: int = 12) -> dict[str, Any]:
    adjacency: dict[str, Counter[str]] = {}
    relation_counts: Counter[str] = Counter()
    disease_symptoms: dict[str, set[str]] = {}
    disease_neighbors: dict[str, set[str]] = {}

    for triple in triples:
        subject = str(triple.get("subject", "")).strip()
        obj = str(triple.get("object", "")).strip()
        subject_type = str(triple.get("subject_type", "")).strip() or "Unknown"
        object_type = str(triple.get("object_type", "")).strip() or "Unknown"
        predicate = str(triple.get("predicate", "")).strip()
        if not subject or not obj or not predicate:
            continue

        source_key = _node_key(subject, subject_type)
        target_key = _node_key(obj, object_type)
        adjacency.setdefault(source_key, Counter())[target_key] += 1
        adjacency.setdefault(target_key, Counter())[source_key] += 1
        relation_counts[predicate] += 1

        if subject_type == "Disease":
            disease_neighbors.setdefault(subject, set()).add(obj)
            if predicate == "has_symptom":
                disease_symptoms.setdefault(subject, set()).add(obj)

    node_count = len(adjacency)
    if not node_count:
        return {
            "node_count": 0,
            "edge_count": 0,
            "top_degree": [],
            "pagerank": [],
            "disease_similarity": [],
            "multi_relation_similarity": [],
            "causal_paths": [],
            "disease_communities": [],
            "relation_families": {},
        }

    degree_scores = {
        key: sum(neighbors.values())
        for key, neighbors in adjacency.items()
    }
    pagerank = {key: 1.0 / node_count for key in adjacency}
    damping = 0.85
    for _ in range(20):
        next_rank = {key: (1.0 - damping) / node_count for key in adjacency}
        for key, neighbors in adjacency.items():
            weight_sum = sum(neighbors.values())
            if not weight_sum:
                continue
            share = damping * pagerank[key] / weight_sum
            for neighbor, weight in neighbors.items():
                next_rank[neighbor] += share * weight
        pagerank = next_rank

    def node_record(key: str, score: float, *, degree: int | None = None) -> dict[str, Any]:
        name, entity_type = _split_node_key(key)
        record: dict[str, Any] = {
            "entity": name,
            "entity_type": entity_type,
            "score": round(float(score), 6),
        }
        if degree is not None:
            record["degree"] = degree
        return record

    top_degree = [
        node_record(key, score, degree=score)
        for key, score in sorted(degree_scores.items(), key=lambda item: (-item[1], item[0]))[:top_n]
    ]
    top_pagerank = [
        node_record(key, score, degree=degree_scores.get(key, 0))
        for key, score in sorted(pagerank.items(), key=lambda item: (-item[1], item[0]))[:top_n]
    ]

    similarities: list[dict[str, Any]] = []
    diseases = sorted(disease_symptoms)
    for index, disease_a in enumerate(diseases):
        symptoms_a = disease_symptoms[disease_a]
        for disease_b in diseases[index + 1:]:
            symptoms_b = disease_symptoms[disease_b]
            union = symptoms_a | symptoms_b
            if not union:
                continue
            shared = symptoms_a & symptoms_b
            if not shared:
                continue
            similarities.append(
                {
                    "disease_a": disease_a,
                    "disease_b": disease_b,
                    "shared_symptoms": sorted(shared),
                    "shared_symptom_count": len(shared),
                    "jaccard": round(len(shared) / len(union), 6),
                }
            )
    similarities.sort(key=lambda item: (-item["jaccard"], -item["shared_symptom_count"], item["disease_a"], item["disease_b"]))

    communities: list[dict[str, Any]] = []
    for disease, neighbors in sorted(disease_neighbors.items(), key=lambda item: (-len(item[1]), item[0]))[:top_n]:
        predicate_counts = Counter(
            str(triple.get("predicate", ""))
            for triple in triples
            if triple.get("subject") == disease
        )
        communities.append(
            {
                "disease": disease,
                "neighbor_count": len(neighbors),
                "top_neighbors": sorted(neighbors)[:10],
                "relation_mix": dict(predicate_counts.most_common()),
            }
        )

    causal_paths = _build_causal_paths(triples, limit=top_n)
    multi_relation_similarity, tensor_acceleration = _build_multi_relation_similarity_with_metadata(triples, limit=top_n)
    disease_communities = _build_disease_communities(triples, limit=top_n)

    relation_families = {
        "clinical_profile": sum(relation_counts[predicate] for predicate in ("has_symptom", "has_complication", "has_risk_factor")),
        "diagnosis": sum(relation_counts[predicate] for predicate in ("diagnosed_by", "belongs_to_department")),
        "treatment_management": sum(relation_counts[predicate] for predicate in ("treated_by", "prevented_by", "contraindicated_for")),
        "causal": sum(relation_counts[predicate] for predicate in ("causes",)),
        "other": 0,
    }
    relation_families["other"] = sum(relation_counts.values()) - sum(relation_families.values())

    return {
        "node_count": node_count,
        "edge_count": sum(relation_counts.values()),
        "top_degree": top_degree,
        "pagerank": top_pagerank,
        "disease_similarity": similarities[:top_n],
        "multi_relation_similarity": multi_relation_similarity,
        "causal_paths": causal_paths,
        "disease_communities": communities,
        "disease_community_detection": disease_communities,
        "relation_families": relation_families,
        "tensor_acceleration": {
            "enabled": bool(tensor_acceleration.get("enabled")),
            "operators": [tensor_acceleration],
        },
    }


def _build_analysis_report(
    processed_documents: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    triples: list[dict[str, Any]],
    analysis_queries: list[dict[str, Any]],
    subgraphs: list[dict[str, Any]],
    graph_insights: dict[str, Any] | None = None,
    nl2sql_eval: dict[str, Any] | None = None,
    analysis_planning: dict[str, Any] | None = None,
) -> str:
    entity_type_counts = Counter(item["type"] for item in entities)
    relation_type_counts = Counter(item["predicate"] for item in triples)
    top_entities = Counter(item["name"] for item in entities).most_common(8)

    lines: list[str] = []
    lines.append("# 医疗分析报告")
    lines.append("")
    lines.append(f"- 文档数：{len(processed_documents)}")
    lines.append(f"- 实体数：{len(entities)}")
    lines.append(f"- 三元组数：{len(triples)}")
    lines.append(f"- 分析查询数：{len(analysis_queries)}")
    lines.append(f"- 子图数：{len(subgraphs)}")
    lines.append("")
    if analysis_planning:
        lines.append("## 分析任务规划")
        lines.append("")
        lines.append(f"- 目标：{analysis_planning.get('goal', '')}")
        lines.append(f"- 图谱实体数：{analysis_planning.get('graph_schema_summary', {}).get('entity_count', 0)}")
        lines.append(f"- 图谱三元组数：{analysis_planning.get('graph_schema_summary', {}).get('triple_count', 0)}")
        lines.append(f"- 规划工具数：{len(analysis_planning.get('tool_plan', []))}")
        lines.append("")
        lines.append("| 顺序 | 工具 | 目的 | 输出 |")
        lines.append("| ---: | --- | --- | --- |")
        for item in analysis_planning.get("tool_plan", []):
            lines.append(
                f"| {item['order']} | `{item['tool']}` | {item['purpose']} | "
                f"`{', '.join(item.get('outputs', []))}` |"
            )
        lines.append("")
    lines.append("## 实体类型分布")
    lines.append("| 类型 | 数量 |")
    lines.append("| --- | ---: |")
    for name, value in sorted(entity_type_counts.items()):
        lines.append(f"| {name} | {value} |")
    lines.append("")
    lines.append("## 关系类型分布")
    lines.append("| 关系 | 数量 |")
    lines.append("| --- | ---: |")
    for name, value in sorted(relation_type_counts.items()):
        lines.append(f"| {name} | {value} |")
    lines.append("")
    lines.append("## 高频实体")
    lines.append("| 实体 | 次数 |")
    lines.append("| --- | ---: |")
    for name, value in top_entities:
        lines.append(f"| {name} | {value} |")
    lines.append("")
    if graph_insights:
        lines.append("## 图谱算法洞察")
        lines.append("")
        lines.append(f"- 图节点数：{graph_insights['node_count']}")
        lines.append(f"- 图边数：{graph_insights['edge_count']}")
        tensor_acceleration = graph_insights.get("tensor_acceleration", {})
        tensor_operators = tensor_acceleration.get("operators", []) if isinstance(tensor_acceleration, dict) else []
        if tensor_operators:
            first_operator = tensor_operators[0]
            lines.append(
                "- Tensor 算子："
                f"{first_operator.get('operator')}，"
                f"device={first_operator.get('selected_device')}，"
                f"pair_count={first_operator.get('pair_count')}。"
            )
        lines.append("")
        lines.append("### PageRank 核心实体")
        lines.append("| 实体 | 类型 | PageRank | 度数 |")
        lines.append("| --- | --- | ---: | ---: |")
        for item in graph_insights.get("pagerank", [])[:8]:
            lines.append(f"| {item['entity']} | {item['entity_type']} | {item['score']} | {item['degree']} |")
        lines.append("")
        lines.append("### 症状相似疾病对")
        lines.append("| 疾病A | 疾病B | Jaccard | 共享症状 |")
        lines.append("| --- | --- | ---: | --- |")
        for item in graph_insights.get("disease_similarity", [])[:8]:
            lines.append(
                f"| {item['disease_a']} | {item['disease_b']} | {item['jaccard']} | "
                f"{'、'.join(item['shared_symptoms'])} |"
            )
        lines.append("")
        lines.append("### 多关系疾病相似度")
        lines.append("| 疾病A | 疾病B | 加权相似度 | 共享关系数 |")
        lines.append("| --- | --- | ---: | ---: |")
        for item in graph_insights.get("multi_relation_similarity", [])[:8]:
            lines.append(
                f"| {item['disease_a']} | {item['disease_b']} | {item['weighted_similarity']} | "
                f"{item['shared_relation_count']} |"
            )
        lines.append("")
        communities = graph_insights.get("disease_community_detection", [])
        if communities:
            lines.append("### 疾病社区发现")
            lines.append("| 社区 | 疾病数 | 凝聚度 | 代表疾病 | 共享特征 |")
            lines.append("| --- | ---: | ---: | --- | --- |")
            for item in communities[:8]:
                shared = [
                    str(feature.get("feature", ""))
                    for feature in item.get("top_shared_features", [])[:4]
                ]
                lines.append(
                    f"| {item.get('community_id')} | {item.get('member_count')} | {item.get('cohesion')} | "
                    f"{'、'.join(item.get('members', [])[:5])} | {'、'.join(shared)} |"
                )
            lines.append("")
        lines.append("### 因果路径")
        lines.append("| 起点 | 终点 | 长度 | 分数 | 路径 |")
        lines.append("| --- | --- | ---: | ---: | --- |")
        for item in graph_insights.get("causal_paths", [])[:8]:
            lines.append(
                f"| {item['start']} | {item['end']} | {item['length']} | {item['score']} | "
                f"{' -> '.join(item['nodes'])} |"
            )
        lines.append("")
        quality = graph_insights.get("quality_diagnostics", {})
        if quality:
            lines.append("### 图谱质量诊断")
            lines.append("")
            lines.append(f"- 质量分数：{quality.get('quality_score', 0)}")
            lines.append(f"- 孤立实体数：{quality.get('isolated_entity_count', 0)}")
            lines.append(f"- 低置信三元组数：{quality.get('low_confidence_triple_count', 0)}")
            lines.append(f"- 缺失证据三元组数：{quality.get('missing_evidence_triple_count', 0)}")
            lines.append("")
    lines.append("## NL2SQL 分析")
    for item in analysis_queries:
        lines.append(f"### {item['question']}")
        lines.append("")
        lines.append("```sql")
        lines.append(item["sql"])
        lines.append("```")
        lines.append("")
        if item["rows"]:
            headers = list(item["rows"][0].keys())
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for row in item["rows"][:10]:
                lines.append("| " + " | ".join(str(row.get(key, "")) for key in headers) + " |")
            lines.append("")
    if nl2sql_eval:
        lines.append("## NL2SQL 评测")
        lines.append("")
        lines.append(f"- 样本数：{nl2sql_eval['total']}")
        lines.append(f"- 通过数：{nl2sql_eval['passed']}")
        lines.append(f"- 准确率：{nl2sql_eval['accuracy']:.2%}")
        lines.append(f"- 阈值：{nl2sql_eval['threshold']:.2%}")
        lines.append(f"- 是否达标：{'是' if nl2sql_eval['meets_threshold'] else '否'}")
        lines.append("")
    lines.append("## 子图示例")
    for item in subgraphs[:3]:
        lines.append(f"### {item['focus_disease']}")
        lines.append(f"- 节点数：{len(item['nodes'])}")
        lines.append(f"- 边数：{len(item['edges'])}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def run_analysis_agent(
    output_root: Path,
    *,
    processed_documents: list[dict[str, Any]] | None = None,
    entities: list[dict[str, Any]] | None = None,
    triples: list[dict[str, Any]] | None = None,
    graph_snapshot: dict[str, Any] | None = None,
    qa_traces: list[dict[str, Any]] | None = None,
    questions: list[str] | None = None,
    nl2sql_eval_path: Path | None = None,
    nl2sql_threshold: float = 0.85,
) -> dict[str, Any]:
    if processed_documents is None or entities is None or triples is None or graph_snapshot is None or qa_traces is None:
        loaded_documents, loaded_entities, loaded_triples, loaded_graph_snapshot, loaded_qa_traces = _load_artifacts(output_root)
        processed_documents = processed_documents or loaded_documents
        entities = entities or loaded_entities
        triples = triples or loaded_triples
        graph_snapshot = graph_snapshot or loaded_graph_snapshot
        qa_traces = qa_traces or loaded_qa_traces

    assert processed_documents is not None
    assert entities is not None
    assert triples is not None
    assert graph_snapshot is not None
    assert qa_traces is not None

    conn = _build_sqlite(triples, entities, processed_documents)
    try:
        query_catalog = _analysis_query_catalog()
        analysis_questions = questions or DEFAULT_ANALYSIS_QUESTIONS
        task3_lineage = _build_task3_lineage(output_root, processed_documents, entities, triples, graph_snapshot)
        if not task3_lineage["ready"]:
            raise RuntimeError(f"Task-3 upstream artifacts are incomplete: {task3_lineage['blockers']}")
        graph_understanding = _build_graph_understanding(entities, triples, graph_snapshot)
        analysis_planning = _build_analysis_planning(
            questions=analysis_questions,
            entities=entities,
            triples=triples,
            query_catalog=query_catalog,
        )
        analysis_queries: list[dict[str, Any]] = []
        analysis_trace: list[dict[str, Any]] = [
            {
                "agent": "analysis_agent",
                "goal": "基于知识图谱生成统计分析和可视化",
                "step": "plan_analysis_task",
                "status": "success",
                "intent": analysis_planning["intent"],
                "tool_plan_count": len(analysis_planning["tool_plan"]),
                "selected_query_count": analysis_planning["selected_query_count"],
                "graph_schema_summary": analysis_planning["graph_schema_summary"],
                "lineage_ready": task3_lineage["ready"],
            },
            {
                "agent": "analysis_agent",
                "goal": "自动接入并理解任务二生成的数据/知识图谱",
                "step": "understand_graph_structure",
                "status": "success",
                "node_count": graph_understanding["node_count"],
                "edge_count": graph_understanding["edge_count"],
                "entity_type_count": len(graph_understanding["entity_properties"]),
                "relation_type_count": len(graph_understanding["relation_properties"]),
                "supported_analysis_modes": graph_understanding["supported_analysis_modes"],
            }
        ]

        for query in query_catalog:
            rows = _run_query(conn, query["sql"])
            analysis_queries.append(
                {
                    "query_id": query["query_id"],
                    "question": query["question"],
                    "sql": query["sql"],
                    "rows": rows,
                    "row_count": len(rows),
                    "chart": query["chart"],
                }
            )
            analysis_trace.append(
                {
                    "agent": "analysis_agent",
                    "goal": "基于知识图谱生成统计分析和可视化",
                    "step": "nl2sql",
                    "query_id": query["query_id"],
                    "question": query["question"],
                    "status": "success",
                    "row_count": len(rows),
                    "sql": query["sql"],
                }
            )

        chart_payloads = dict(build_charts(entities, triples))
        for item in analysis_queries:
            if item["query_id"] == "disease_drug_counts":
                chart_payloads[item["chart"]] = _chart_from_rows(item["rows"], "disease", "drug_count")
            elif item["query_id"] == "symptom_frequency":
                chart_payloads[item["chart"]] = _chart_from_rows(item["rows"], "symptom", "disease_count")
            elif item["query_id"] == "disease_complication_counts":
                chart_payloads[item["chart"]] = _chart_from_rows(item["rows"], "disease", "complication_count")
            elif item["query_id"] == "symptom_cooccurrence":
                chart_payloads[item["chart"]] = {
                    "pairs": item["rows"],
                    "nodes": sorted(
                        {
                            disease
                            for row in item["rows"]
                            for disease in (row.get("disease_a"), row.get("disease_b"))
                        }
                    ),
                }
            elif item["query_id"] == "department_distribution":
                chart_payloads[item["chart"]] = _chart_from_rows(item["rows"], "department", "disease_count")
            elif item["query_id"] == "drug_contraindications":
                chart_payloads[item["chart"]] = _chart_from_rows(item["rows"], "drug", "contraindication_count")
            elif item["query_id"] == "disease_risk_factor_counts":
                chart_payloads[item["chart"]] = _chart_from_rows(item["rows"], "disease", "risk_factor_count")
            elif item["query_id"] == "disease_prevention_counts":
                chart_payloads[item["chart"]] = _chart_from_rows(item["rows"], "disease", "prevention_count")
            elif item["query_id"] == "entity_centrality":
                chart_payloads[item["chart"]] = _chart_from_rows(item["rows"], "entity", "degree")
            elif item["query_id"] == "entity_type_distribution":
                chart_payloads[item["chart"]] = _chart_from_rows(item["rows"], "entity_type", "entity_count")
            elif item["query_id"] == "relation_family_distribution":
                chart_payloads[item["chart"]] = _chart_from_rows(item["rows"], "relation_family", "triple_count")
            elif item["query_id"] == "source_document_contribution":
                chart_payloads[item["chart"]] = _chart_from_rows(item["rows"], "doc_id", "triple_count")
            elif item["query_id"] == "low_confidence_relations":
                chart_payloads[item["chart"]] = {
                    "rows": item["rows"],
                    "chart_type": "quality_table",
                    "metric": "confidence",
                }
            elif item["query_id"] == "high_confidence_relation_counts":
                chart_payloads[item["chart"]] = _chart_from_rows(item["rows"], "relation", "fact_count")
            elif item["query_id"] == "risk_factor_pairs":
                chart_payloads[item["chart"]] = {
                    "rows": item["rows"],
                    "chart_type": "relation_table",
                    "metric": "confidence",
                }

        disease_counts = Counter(item["subject"] for item in triples if item.get("subject_type") == "Disease")
        top_diseases = [name for name, _ in disease_counts.most_common(3)]
        subgraphs = [_build_disease_subgraph(disease, triples) for disease in top_diseases]
        graph_insights = _build_graph_insights(triples)
        subgraph_ranking = _rank_subgraphs(subgraphs, graph_insights)
        graph_insights["subgraph_ranking"] = subgraph_ranking
        graph_insights["quality_diagnostics"] = _diagnose_graph_quality(entities, triples, graph_insights)
        chart_payloads["graph_pagerank"] = {
            "labels": [item["entity"] for item in graph_insights["pagerank"]],
            "values": [item["score"] for item in graph_insights["pagerank"]],
            "rows": graph_insights["pagerank"],
        }
        chart_payloads["graph_degree_centrality"] = {
            "labels": [item["entity"] for item in graph_insights["top_degree"]],
            "values": [item["degree"] for item in graph_insights["top_degree"]],
            "rows": graph_insights["top_degree"],
        }
        chart_payloads["disease_similarity"] = {
            "pairs": graph_insights["disease_similarity"],
            "metric": "symptom_jaccard",
        }
        chart_payloads["multi_relation_similarity"] = {
            "pairs": graph_insights["multi_relation_similarity"],
            "metric": "weighted_multi_relation_jaccard",
        }
        chart_payloads["disease_community_detection"] = {
            "communities": graph_insights["disease_community_detection"],
            "metric": "weighted_label_propagation",
        }
        chart_payloads["causal_paths"] = {
            "paths": graph_insights["causal_paths"],
            "metric": "mean_path_confidence",
        }
        chart_payloads["subgraph_ranking"] = {
            "labels": [item["focus_disease"] for item in subgraph_ranking],
            "values": [item["score"] for item in subgraph_ranking],
            "rows": subgraph_ranking,
        }
        chart_payloads["graph_quality_diagnostics"] = graph_insights["quality_diagnostics"]
        chart_payloads["relation_families"] = {
            "labels": list(graph_insights["relation_families"].keys()),
            "values": list(graph_insights["relation_families"].values()),
            "rows": [
                {"family": key, "count": value}
                for key, value in graph_insights["relation_families"].items()
            ],
        }
        analysis_trace.append(
            {
                "agent": "analysis_agent",
                "goal": "基于知识图谱生成统计分析和可视化",
                "step": "build_subgraphs",
                "status": "success",
                "subgraph_count": len(subgraphs),
                "top_diseases": top_diseases,
            }
        )
        analysis_trace.append(
            {
                "agent": "analysis_agent",
                "goal": "基于知识图谱生成统计分析和可视化",
                "step": "graph_algorithms",
                "status": "success",
                "node_count": graph_insights["node_count"],
                "edge_count": graph_insights["edge_count"],
                "pagerank_count": len(graph_insights["pagerank"]),
                "disease_similarity_count": len(graph_insights["disease_similarity"]),
                "multi_relation_similarity_count": len(graph_insights["multi_relation_similarity"]),
                "disease_community_detection_count": len(graph_insights["disease_community_detection"]),
                "causal_path_count": len(graph_insights["causal_paths"]),
                "graph_quality_score": graph_insights["quality_diagnostics"]["quality_score"],
                "community_count": len(graph_insights["disease_communities"]),
                "tensor_acceleration": graph_insights.get("tensor_acceleration", {}),
            }
        )

        nl2sql_eval = run_nl2sql_eval(output_root, eval_path=nl2sql_eval_path, threshold=nl2sql_threshold)
        visualization_manifest = _build_visualization_manifest(
            output_root,
            chart_payloads,
            analysis_queries,
            subgraphs,
            graph_insights,
        )
        task3_closed_loop_status = _build_task3_closed_loop_status(
            task3_lineage,
            graph_understanding,
            analysis_queries,
            nl2sql_eval,
            visualization_manifest,
            graph_insights,
        )
        report = _build_analysis_report(
            processed_documents,
            entities,
            triples,
            analysis_queries,
            subgraphs,
            graph_insights,
            nl2sql_eval,
            analysis_planning,
        )
        analysis_trace.append(
            {
                "agent": "analysis_agent",
                "goal": "基于知识图谱生成统计分析和可视化",
                "step": "nl2sql_eval",
                "status": "success",
                "accuracy": nl2sql_eval["accuracy"],
                "meets_threshold": nl2sql_eval["meets_threshold"],
                "report_path": nl2sql_eval["output_paths"]["markdown"],
            }
        )
        analysis_trace.append(
            {
                "agent": "analysis_agent",
                "goal": "基于知识图谱生成统计分析和可视化",
                "step": "render_report",
                "status": "success",
                "report_path": str(output_root / "demo" / "analysis_report.md"),
            }
        )
        analysis_trace.append(
            {
                "agent": "analysis_agent",
                "goal": "数据到知识到洞察闭环验收",
                "step": "validate_task3_closed_loop",
                "status": "success" if task3_closed_loop_status["ready"] else "blocked",
                "closed_loop_ready": task3_closed_loop_status["ready"],
                "blockers": task3_closed_loop_status["blockers"],
                "metrics": task3_closed_loop_status["metrics"],
            }
        )

        output_root.mkdir(parents=True, exist_ok=True)
        _write_text(output_root / "demo" / "analysis_report.md", report)
        _write_json(output_root / "demo" / "analysis_planning.json", analysis_planning)
        _write_json(output_root / "demo" / "task3_lineage.json", task3_lineage)
        _write_json(output_root / "demo" / "graph_understanding.json", graph_understanding)
        _write_json(output_root / "demo" / "visualization_manifest.json", visualization_manifest)
        _write_json(output_root / "demo" / "task3_closed_loop_status.json", task3_closed_loop_status)
        _write_json(output_root / "demo" / "analysis_summary.json", {
            "document_count": len(processed_documents),
            "entity_count": len(entities),
            "triple_count": len(triples),
            "query_count": len(analysis_queries),
            "subgraph_count": len(subgraphs),
            "graph_algorithm_count": 8,
            "graph_pagerank_count": len(graph_insights["pagerank"]),
            "disease_similarity_count": len(graph_insights["disease_similarity"]),
            "multi_relation_similarity_count": len(graph_insights["multi_relation_similarity"]),
            "disease_community_detection_count": len(graph_insights["disease_community_detection"]),
            "causal_path_count": len(graph_insights["causal_paths"]),
            "disease_community_count": len(graph_insights["disease_communities"]),
            "graph_quality_score": graph_insights["quality_diagnostics"]["quality_score"],
            "tensor_acceleration": graph_insights.get("tensor_acceleration", {}),
            "nl2sql_accuracy": nl2sql_eval["accuracy"],
            "nl2sql_meets_threshold": nl2sql_eval["meets_threshold"],
            "analysis_tool_plan_count": len(analysis_planning["tool_plan"]),
            "analysis_intents": analysis_planning["intent"]["priority_order"],
            "task3_lineage_ready": task3_lineage["ready"],
            "graph_understanding_version": graph_understanding["understanding_version"],
            "supported_analysis_modes": graph_understanding["supported_analysis_modes"],
            "bi_chart_count": visualization_manifest["bi_chart_count"],
            "graph_visual_count": visualization_manifest["graph_visual_count"],
            "task3_closed_loop_ready": task3_closed_loop_status["ready"],
        })
        _write_json(output_root / "demo" / "graph_insights.json", graph_insights)
        _write_jsonl(output_root / "demo" / "analysis_trace.jsonl", analysis_trace)
        _write_jsonl(output_root / "demo" / "analysis_queries.jsonl", analysis_queries)

        _write_json(output_root / "demo" / "charts" / "entity_types.json", chart_payloads["entity_types"])
        _write_json(output_root / "demo" / "charts" / "relation_types.json", chart_payloads["relation_types"])
        _write_json(output_root / "demo" / "charts" / "disease_drug_counts.json", chart_payloads["disease_drug_counts"])
        _write_json(output_root / "demo" / "charts" / "symptom_frequency.json", chart_payloads["symptom_frequency"])
        _write_json(output_root / "demo" / "charts" / "disease_complication_counts.json", chart_payloads["disease_complication_counts"])
        _write_json(output_root / "demo" / "charts" / "symptom_cooccurrence.json", chart_payloads["symptom_cooccurrence"])
        _write_json(output_root / "demo" / "charts" / "department_distribution.json", chart_payloads["department_distribution"])
        _write_json(output_root / "demo" / "charts" / "drug_contraindications.json", chart_payloads["drug_contraindications"])
        _write_json(output_root / "demo" / "charts" / "disease_risk_factor_counts.json", chart_payloads["disease_risk_factor_counts"])
        _write_json(output_root / "demo" / "charts" / "disease_prevention_counts.json", chart_payloads["disease_prevention_counts"])
        _write_json(output_root / "demo" / "charts" / "entity_centrality.json", chart_payloads["entity_centrality"])
        _write_json(output_root / "demo" / "charts" / "graph_pagerank.json", chart_payloads["graph_pagerank"])
        _write_json(output_root / "demo" / "charts" / "graph_degree_centrality.json", chart_payloads["graph_degree_centrality"])
        _write_json(output_root / "demo" / "charts" / "disease_similarity.json", chart_payloads["disease_similarity"])
        _write_json(output_root / "demo" / "charts" / "multi_relation_similarity.json", chart_payloads["multi_relation_similarity"])
        _write_json(output_root / "demo" / "charts" / "disease_community_detection.json", chart_payloads["disease_community_detection"])
        _write_json(output_root / "demo" / "charts" / "causal_paths.json", chart_payloads["causal_paths"])
        _write_json(output_root / "demo" / "charts" / "subgraph_ranking.json", chart_payloads["subgraph_ranking"])
        _write_json(output_root / "demo" / "charts" / "graph_quality_diagnostics.json", chart_payloads["graph_quality_diagnostics"])
        _write_json(output_root / "demo" / "charts" / "relation_families.json", chart_payloads["relation_families"])
        _write_jsonl(output_root / "demo" / "subgraphs" / "disease_subgraphs.jsonl", subgraphs)

        return {
            "analysis_report": report,
            "analysis_summary": {
                "document_count": len(processed_documents),
                "entity_count": len(entities),
                "triple_count": len(triples),
                "query_count": len(analysis_queries),
                "subgraph_count": len(subgraphs),
                "graph_algorithm_count": 8,
                "graph_pagerank_count": len(graph_insights["pagerank"]),
                "disease_similarity_count": len(graph_insights["disease_similarity"]),
                "multi_relation_similarity_count": len(graph_insights["multi_relation_similarity"]),
                "disease_community_detection_count": len(graph_insights["disease_community_detection"]),
                "causal_path_count": len(graph_insights["causal_paths"]),
                "disease_community_count": len(graph_insights["disease_communities"]),
                "graph_quality_score": graph_insights["quality_diagnostics"]["quality_score"],
                "tensor_acceleration": graph_insights.get("tensor_acceleration", {}),
                "nl2sql_accuracy": nl2sql_eval["accuracy"],
                "nl2sql_meets_threshold": nl2sql_eval["meets_threshold"],
                "analysis_tool_plan_count": len(analysis_planning["tool_plan"]),
                "analysis_intents": analysis_planning["intent"]["priority_order"],
                "task3_lineage_ready": task3_lineage["ready"],
                "graph_understanding_version": graph_understanding["understanding_version"],
                "supported_analysis_modes": graph_understanding["supported_analysis_modes"],
                "bi_chart_count": visualization_manifest["bi_chart_count"],
                "graph_visual_count": visualization_manifest["graph_visual_count"],
                "task3_closed_loop_ready": task3_closed_loop_status["ready"],
            },
            "analysis_trace": analysis_trace,
            "analysis_queries": analysis_queries,
            "nl2sql_eval": nl2sql_eval,
            "charts": chart_payloads,
            "subgraphs": subgraphs,
            "graph_insights": graph_insights,
            "analysis_planning": analysis_planning,
            "task3_lineage": task3_lineage,
            "graph_understanding": graph_understanding,
            "visualization_manifest": visualization_manifest,
            "task3_closed_loop_status": task3_closed_loop_status,
        }
    finally:
        conn.close()
