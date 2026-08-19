from __future__ import annotations

import html
import json
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medinsight.common.utils import ensure_dir, write_json


DEFAULT_GRAPH_PATH = Path("data/shared/knowledge_graph/task2_user_medical_kg_library.json")
DEFAULT_PENDING_GRAPH_PATH = Path("data/shared/knowledge_graph/task2_pending_medical_kg.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/task3/analysis")
DEFAULT_CONFIG_PATH = Path("data/task3/samples/analysis_config.json")
DEFAULT_TASK = "复用任务一处理报告和任务二知识图谱，输出图谱分析指标、SQL 查询结果和 BI 可视化。"
DEFAULT_TASK1_REPORT_CANDIDATES = [
    Path("artifacts/task1/task2_cblue_processed/data_processing_report.json"),
]

SYMPTOM_PREDICATES = {"临床表现", "症状", "主要症状"}
TEST_PREDICATES = {"辅助检查", "检查", "诊断检查"}
TREATMENT_PREDICATES = {"药物治疗", "辅助治疗", "治疗方法", "治疗", "用药"}
CAUSE_PREDICATES = {"病因", "病理分型", "并发症"}

ANALYZER_CATALOG = [
    {
        "name": "EntityDistributionAnalyzer",
        "purpose": "实体类型分布分析",
        "example": "当前图谱有哪些实体类型？",
    },
    {
        "name": "RelationDistributionAnalyzer",
        "purpose": "关系类型分布分析",
        "example": "当前图谱中哪类关系最多？",
    },
    {
        "name": "DiseaseSymptomAnalyzer",
        "purpose": "疾病-症状频次分析",
        "example": "哪些症状覆盖疾病最多？",
    },
    {
        "name": "DrugDiseaseAssociationAnalyzer",
        "purpose": "药物-疾病关联分析",
        "example": "哪些药物关联疾病最多？",
    },
    {
        "name": "GraphCentralityAnalyzer",
        "purpose": "中心节点分析",
        "example": "当前图谱最核心的疾病是什么？",
    },
    {
        "name": "EvidencePathAnalyzer",
        "purpose": "证据路径分析",
        "example": "某个关系来自哪条原文证据？",
    },
    {
        "name": "GapDetectionAnalyzer",
        "purpose": "知识缺口分析",
        "example": "哪些疾病节点信息不完整？",
    },
]


@dataclass
class AnalysisContext:
    graph_path: Path
    graph: dict[str, Any]
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    node_by_id: dict[str, dict[str, Any]]


def resolve_graph_path(graph_path: Path | str | None = None) -> Path:
    if graph_path:
        return Path(graph_path)
    if DEFAULT_GRAPH_PATH.exists():
        return DEFAULT_GRAPH_PATH
    return DEFAULT_PENDING_GRAPH_PATH


def load_graph(graph_path: Path | str | None = None) -> AnalysisContext:
    path = resolve_graph_path(graph_path)
    if not path.exists():
        raise FileNotFoundError(
            f"未找到任务二知识图谱：{path}。请先运行 scripts/task2/task2_cli_menu.sh 构建并保存图谱。"
        )
    graph = json.loads(path.read_text(encoding="utf-8"))
    nodes = [node for node in graph.get("nodes") or [] if isinstance(node, dict)]
    edges = [edge for edge in graph.get("edges") or [] if isinstance(edge, dict)]
    node_by_id = {str(node.get("node_id")): node for node in nodes if node.get("node_id")}
    return AnalysisContext(path, graph, nodes, edges, node_by_id)


def resolve_task1_report_path(task1_report_path: Path | str | None = None) -> Path | None:
    if task1_report_path:
        path = Path(task1_report_path)
        return path if path.exists() else None
    for path in DEFAULT_TASK1_REPORT_CANDIDATES:
        if path.exists():
            return path
    return None


def read_json_array(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def task1_related_artifact_paths(report_path: Path | None) -> dict[str, Path | None]:
    if report_path is None:
        return {"documents_path": None, "segments_path": None}
    base_dir = report_path.parent
    documents_path = base_dir / "processed_documents.json"
    segments_path = base_dir / "segments.json"
    return {
        "documents_path": documents_path if documents_path.exists() else None,
        "segments_path": segments_path if segments_path.exists() else None,
    }


def load_task1_profile(task1_report_path: Path | str | None = None) -> dict[str, Any]:
    path = resolve_task1_report_path(task1_report_path)
    if path is None:
        return {
            "status": "missing",
            "message": "未发现任务一数据处理报告；任务三仍可基于任务二图谱运行。",
        }
    artifact_paths = task1_related_artifact_paths(path)
    processed_documents = read_json_array(artifact_paths["documents_path"])
    processed_segments = read_json_array(artifact_paths["segments_path"])
    report = json.loads(path.read_text(encoding="utf-8"))
    quality = report.get("quality") or {}
    gate = report.get("quality_gate") or {}
    datamate = report.get("datamate_integration") or {}
    return {
        "status": "ok",
        "report_path": str(path),
        "input_records": quality.get("input_records"),
        "documents": quality.get("documents"),
        "segments": quality.get("segments"),
        "warning_count": quality.get("warning_count"),
        "error_count": quality.get("error_count"),
        "quality_gate_status": gate.get("status"),
        "quality_gate_passed": gate.get("passed"),
        "datamate_connected": datamate.get("connected"),
        "datamate_base_url": datamate.get("base_url"),
        "documents_path": str(artifact_paths["documents_path"]) if artifact_paths["documents_path"] else None,
        "segments_path": str(artifact_paths["segments_path"]) if artifact_paths["segments_path"] else None,
        "artifact_document_count": len(processed_documents),
        "artifact_segment_count": len(processed_segments),
    }


def node_name(node: dict[str, Any] | None, default: str = "") -> str:
    if not node:
        return default
    return str(node.get("name") or default)


def node_type(node: dict[str, Any] | None, default: str = "") -> str:
    if not node:
        return default
    return str(node.get("type") or default)


def edge_subject(edge: dict[str, Any], context: AnalysisContext) -> str:
    props = edge.get("properties") or {}
    return str(props.get("subject") or node_name(context.node_by_id.get(str(edge.get("source"))), str(edge.get("source") or "")))


def edge_object(edge: dict[str, Any], context: AnalysisContext) -> str:
    props = edge.get("properties") or {}
    return str(props.get("object") or node_name(context.node_by_id.get(str(edge.get("target"))), str(edge.get("target") or "")))


def edge_subject_type(edge: dict[str, Any], context: AnalysisContext) -> str:
    props = edge.get("properties") or {}
    return str(props.get("subject_type") or node_type(context.node_by_id.get(str(edge.get("source")))))


def edge_object_type(edge: dict[str, Any], context: AnalysisContext) -> str:
    props = edge.get("properties") or {}
    return str(props.get("object_type") or node_type(context.node_by_id.get(str(edge.get("target")))))


def first_evidence_text(edge: dict[str, Any]) -> str:
    evidence = edge.get("evidence") or []
    for item in evidence:
        if isinstance(item, dict) and item.get("text"):
            return str(item["text"])
    return ""


def infer_document_topic(document: dict[str, Any]) -> str:
    metadata = document.get("metadata") or {}
    original_fields = metadata.get("original_fields") or {}
    entity_terms = metadata.get("entity_terms") or {}
    disease_terms = entity_terms.get("Disease") if isinstance(entity_terms, dict) else None
    if original_fields.get("name"):
        return str(original_fields["name"])
    if isinstance(disease_terms, list) and disease_terms:
        return str(disease_terms[0])
    if document.get("disease_topic"):
        return str(document["disease_topic"])
    if document.get("title"):
        title = str(document["title"])
        for suffix in ("公开知识图谱字段摘要", "字段摘要"):
            title = title.replace(suffix, "")
        return title.strip()
    return ""


def infer_segment_section(segment: dict[str, Any]) -> str:
    metadata = segment.get("metadata") or {}
    if metadata.get("section"):
        return str(metadata["section"])
    text = str(segment.get("text") or "")
    if any(key in text for key in ("症状", "表现")):
        return "症状"
    if any(key in text for key in ("检查", "诊断")):
        return "检查"
    if any(key in text for key in ("治疗", "药品", "用药")):
        return "治疗"
    if any(key in text for key in ("并发", "并发症")):
        return "并发症"
    if any(key in text for key in ("科室", "就诊")):
        return "就诊科室"
    return ""


def first_evidence_doc(edge: dict[str, Any]) -> str:
    evidence = edge.get("evidence") or []
    for item in evidence:
        if isinstance(item, dict):
            doc_id = item.get("doc_id") or item.get("source_doc")
            if doc_id:
                return str(doc_id)
    return ""


def create_analysis_database(context: AnalysisContext, db_path: Path, task1_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    ensure_dir(db_path.parent)
    if db_path.exists():
        db_path.unlink()
    task1_profile = task1_profile or {}
    task1_documents = read_json_array(Path(task1_profile["documents_path"])) if task1_profile.get("documents_path") else []
    task1_segments = read_json_array(Path(task1_profile["segments_path"])) if task1_profile.get("segments_path") else []
    evidence_path_rows = 0
    started = time.perf_counter()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE task1_documents (
              doc_id TEXT PRIMARY KEY,
              title TEXT,
              source_dataset TEXT,
              disease_topic TEXT,
              text_length INTEGER,
              metadata_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE task1_segments (
              segment_id TEXT PRIMARY KEY,
              doc_id TEXT,
              section TEXT,
              order_index INTEGER,
              text TEXT,
              text_length INTEGER,
              metadata_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE nodes (
              node_id TEXT PRIMARY KEY,
              name TEXT,
              type TEXT,
              source_docs TEXT,
              properties_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE edges (
              edge_id TEXT PRIMARY KEY,
              source_id TEXT,
              target_id TEXT,
              subject TEXT,
              subject_type TEXT,
              predicate TEXT,
              object TEXT,
              object_type TEXT,
              evidence_text TEXT,
              properties_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE medical_triples (
              triple_id TEXT PRIMARY KEY,
              head TEXT,
              head_type TEXT,
              relation TEXT,
              tail TEXT,
              tail_type TEXT,
              source_doc TEXT,
              evidence TEXT,
              confidence REAL,
              source_dataset TEXT,
              properties_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE evidence_paths (
              path_id TEXT PRIMARY KEY,
              triple_id TEXT,
              head TEXT,
              relation TEXT,
              tail TEXT,
              source_doc TEXT,
              evidence_text TEXT
            )
            """
        )
        conn.execute("CREATE INDEX idx_task1_documents_topic ON task1_documents(disease_topic)")
        conn.execute("CREATE INDEX idx_task1_segments_doc ON task1_segments(doc_id)")
        conn.execute("CREATE INDEX idx_nodes_type ON nodes(type)")
        conn.execute("CREATE INDEX idx_edges_subject ON edges(subject)")
        conn.execute("CREATE INDEX idx_edges_predicate ON edges(predicate)")
        conn.execute("CREATE INDEX idx_edges_object ON edges(object)")
        conn.execute("CREATE INDEX idx_triples_head ON medical_triples(head)")
        conn.execute("CREATE INDEX idx_triples_relation ON medical_triples(relation)")
        conn.execute("CREATE INDEX idx_triples_tail ON medical_triples(tail)")
        conn.execute("CREATE INDEX idx_triples_source_doc ON medical_triples(source_doc)")
        conn.execute("CREATE INDEX idx_evidence_source_doc ON evidence_paths(source_doc)")

        for document in task1_documents:
            metadata = document.get("metadata") or {}
            conn.execute(
                "INSERT OR REPLACE INTO task1_documents VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(document.get("doc_id") or ""),
                    str(document.get("title") or ""),
                    str(metadata.get("source_dataset") or document.get("source_dataset") or ""),
                    infer_document_topic(document),
                    len(str(document.get("text") or "")),
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
        for segment in task1_segments:
            metadata = segment.get("metadata") or {}
            conn.execute(
                "INSERT OR REPLACE INTO task1_segments VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(segment.get("segment_id") or ""),
                    str(segment.get("doc_id") or ""),
                    infer_segment_section(segment),
                    int(segment.get("order") or 0),
                    str(segment.get("text") or ""),
                    len(str(segment.get("text") or "")),
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )

        for node in context.nodes:
            conn.execute(
                "INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?)",
                (
                    str(node.get("node_id") or ""),
                    str(node.get("name") or ""),
                    str(node.get("type") or ""),
                    json.dumps(node.get("source_docs") or [], ensure_ascii=False),
                    json.dumps(node.get("properties") or {}, ensure_ascii=False),
                ),
            )
        for edge in context.edges:
            conn.execute(
                "INSERT OR REPLACE INTO edges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(edge.get("edge_id") or ""),
                    str(edge.get("source") or ""),
                    str(edge.get("target") or ""),
                    edge_subject(edge, context),
                    edge_subject_type(edge, context),
                    str(edge.get("predicate") or ""),
                    edge_object(edge, context),
                    edge_object_type(edge, context),
                    first_evidence_text(edge),
                    json.dumps(edge.get("properties") or {}, ensure_ascii=False),
                ),
            )
            props = edge.get("properties") or {}
            triple_id = str(props.get("triple_id") or edge.get("edge_id") or "")
            subject = edge_subject(edge, context)
            predicate = str(edge.get("predicate") or "")
            obj = edge_object(edge, context)
            evidence_text = first_evidence_text(edge)
            source_doc = first_evidence_doc(edge)
            conn.execute(
                "INSERT OR REPLACE INTO medical_triples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    triple_id,
                    subject,
                    edge_subject_type(edge, context),
                    predicate,
                    obj,
                    edge_object_type(edge, context),
                    source_doc,
                    evidence_text,
                    float(props.get("confidence") or 0.0),
                    str(props.get("source_dataset") or ""),
                    json.dumps(props, ensure_ascii=False),
                ),
            )
            for index, evidence_item in enumerate(edge.get("evidence") or []):
                if not isinstance(evidence_item, dict):
                    continue
                source_doc_value = str(evidence_item.get("doc_id") or evidence_item.get("source_doc") or source_doc)
                evidence_value = str(evidence_item.get("text") or "")
                conn.execute(
                    "INSERT OR REPLACE INTO evidence_paths VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"{triple_id}:{index}",
                        triple_id,
                        subject,
                        predicate,
                        obj,
                        source_doc_value,
                        evidence_value,
                    ),
                )
                evidence_path_rows += 1
        conn.commit()
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    return {
        "db_path": str(db_path),
        "task1_documents": len(task1_documents),
        "task1_segments": len(task1_segments),
        "nodes": len(context.nodes),
        "edges": len(context.edges),
        "medical_triples": len(context.edges),
        "evidence_paths": evidence_path_rows,
        "task1_document_rows": len(task1_documents),
        "task1_segment_rows": len(task1_segments),
        "node_rows": len(context.nodes),
        "edge_rows": len(context.edges),
        "triple_rows": len(context.edges),
        "elapsed_ms": elapsed_ms,
    }


def query_rows(db_path: Path, sql: str, params: tuple[Any, ...] = (), limit: int = 100) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql, params)
        rows = cursor.fetchmany(limit)
    return [dict(row) for row in rows]


def scalar_count(db_path: Path, sql: str, params: tuple[Any, ...] = ()) -> int:
    rows = query_rows(db_path, sql, params, limit=1)
    return int(rows[0][0] if rows and 0 in rows[0] else next(iter(rows[0].values()), 0) if rows else 0)


def build_metrics(db_path: Path) -> dict[str, Any]:
    entity_distribution = query_rows(
        db_path,
        "SELECT type, COUNT(*) AS count FROM nodes GROUP BY type ORDER BY count DESC, type",
        limit=50,
    )
    relation_distribution = query_rows(
        db_path,
        "SELECT predicate, COUNT(*) AS count FROM edges GROUP BY predicate ORDER BY count DESC, predicate",
        limit=50,
    )
    top_diseases = query_rows(
        db_path,
        """
        SELECT subject AS disease, COUNT(*) AS relation_count
        FROM edges
        WHERE subject_type='疾病'
        GROUP BY subject
        ORDER BY relation_count DESC, disease
        LIMIT 20
        """,
        limit=20,
    )
    disease_count = query_rows(db_path, "SELECT COUNT(*) AS count FROM nodes WHERE type='疾病'", limit=1)[0]["count"]
    edge_count = query_rows(db_path, "SELECT COUNT(*) AS count FROM edges", limit=1)[0]["count"]
    node_count = query_rows(db_path, "SELECT COUNT(*) AS count FROM nodes", limit=1)[0]["count"]
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "disease_count": disease_count,
        "entity_distribution": entity_distribution,
        "relation_distribution": relation_distribution,
        "top_diseases": top_diseases,
        "common_symptoms": build_common_symptoms(db_path),
        "treatment_distribution": build_treatment_distribution(db_path),
        "drug_disease_association": build_drug_disease_association(db_path),
        "graph_centrality": build_graph_centrality(db_path),
        "disease_relation_matrix": build_disease_relation_matrix(db_path),
        "coverage": build_disease_coverage(db_path),
        "data_to_knowledge_lineage": build_data_to_knowledge_lineage(db_path),
        "document_knowledge_contribution": build_document_knowledge_contribution(db_path),
        "knowledge_density_by_topic": build_knowledge_density_by_topic(db_path),
        "evidence_path_samples": build_evidence_path_samples(db_path),
    }


def build_common_symptoms(db_path: Path) -> list[dict[str, Any]]:
    return query_rows(
        db_path,
        """
        SELECT object AS symptom,
               COUNT(DISTINCT subject) AS disease_count,
               COUNT(*) AS relation_count
        FROM edges
        WHERE subject_type='疾病'
          AND predicate IN ('临床表现','症状','主要症状')
        GROUP BY object
        ORDER BY disease_count DESC, relation_count DESC, symptom
        LIMIT 20
        """,
        limit=20,
    )


def build_treatment_distribution(db_path: Path) -> list[dict[str, Any]]:
    return query_rows(
        db_path,
        """
        SELECT object AS treatment,
               object_type AS type,
               COUNT(DISTINCT subject) AS disease_count,
               COUNT(*) AS relation_count
        FROM edges
        WHERE subject_type='疾病'
          AND predicate IN ('药物治疗','辅助治疗','治疗方法','治疗','用药')
        GROUP BY object, object_type
        ORDER BY disease_count DESC, relation_count DESC, treatment
        LIMIT 20
        """,
        limit=20,
    )


def build_drug_disease_association(db_path: Path) -> list[dict[str, Any]]:
    return query_rows(
        db_path,
        """
        SELECT tail AS drug,
               COUNT(DISTINCT head) AS disease_count,
               COUNT(*) AS relation_count,
               GROUP_CONCAT(DISTINCT head) AS diseases
        FROM medical_triples
        WHERE head_type='疾病'
          AND tail_type='药物'
          AND relation IN ('药物治疗','治疗','用药')
        GROUP BY tail
        ORDER BY disease_count DESC, relation_count DESC, drug
        LIMIT 20
        """,
        limit=20,
    )


def build_graph_centrality(db_path: Path) -> list[dict[str, Any]]:
    return query_rows(
        db_path,
        """
        SELECT name,
               type,
               SUM(degree) AS degree
        FROM (
          SELECT head AS name, head_type AS type, COUNT(*) AS degree
          FROM medical_triples
          GROUP BY head, head_type
          UNION ALL
          SELECT tail AS name, tail_type AS type, COUNT(*) AS degree
          FROM medical_triples
          GROUP BY tail, tail_type
        )
        GROUP BY name, type
        ORDER BY degree DESC, name
        LIMIT 20
        """,
        limit=20,
    )


def build_disease_relation_matrix(db_path: Path) -> list[dict[str, Any]]:
    return query_rows(
        db_path,
        """
        SELECT subject AS disease,
               SUM(CASE WHEN predicate IN ('临床表现','症状','主要症状') THEN 1 ELSE 0 END) AS symptom_edges,
               SUM(CASE WHEN predicate IN ('辅助检查','检查','诊断检查') THEN 1 ELSE 0 END) AS test_edges,
               SUM(CASE WHEN predicate IN ('药物治疗','辅助治疗','治疗方法','治疗','用药') THEN 1 ELSE 0 END) AS treatment_edges,
               SUM(CASE WHEN predicate IN ('病因','病理分型','并发症') THEN 1 ELSE 0 END) AS cause_or_type_edges,
               COUNT(*) AS total_edges
        FROM edges
        WHERE subject_type='疾病'
        GROUP BY subject
        ORDER BY total_edges DESC, disease
        LIMIT 50
        """,
        limit=50,
    )


def build_disease_coverage(db_path: Path) -> dict[str, Any]:
    rows = query_rows(
        db_path,
        """
        SELECT subject AS disease,
               SUM(CASE WHEN predicate IN ('临床表现','症状','主要症状') THEN 1 ELSE 0 END) AS symptom_edges,
               SUM(CASE WHEN predicate IN ('辅助检查','检查','诊断检查') THEN 1 ELSE 0 END) AS test_edges,
               SUM(CASE WHEN predicate IN ('药物治疗','辅助治疗','治疗方法','治疗','用药') THEN 1 ELSE 0 END) AS treatment_edges,
               COUNT(*) AS total_edges
        FROM edges
        WHERE subject_type='疾病'
        GROUP BY subject
        ORDER BY total_edges DESC, disease
        """,
        limit=500,
    )
    gaps = []
    complete = 0
    for row in rows:
        missing = []
        if not row["symptom_edges"]:
            missing.append("症状")
        if not row["test_edges"]:
            missing.append("检查")
        if not row["treatment_edges"]:
            missing.append("治疗")
        if not missing:
            complete += 1
        gaps.append({"disease": row["disease"], "missing": missing, "total_edges": row["total_edges"]})
    total = len(rows)
    return {
        "disease_total": total,
        "complete_diseases": complete,
        "complete_ratio": round(complete / total, 4) if total else 0.0,
        "gap_examples": [item for item in gaps if item["missing"]][:20],
    }


def build_data_to_knowledge_lineage(db_path: Path) -> dict[str, Any]:
    task1_docs = query_rows(db_path, "SELECT COUNT(*) AS count FROM task1_documents", limit=1)[0]["count"]
    task1_segments = query_rows(db_path, "SELECT COUNT(*) AS count FROM task1_segments", limit=1)[0]["count"]
    triples = query_rows(db_path, "SELECT COUNT(*) AS count FROM medical_triples", limit=1)[0]["count"]
    evidence_docs = query_rows(
        db_path,
        "SELECT COUNT(DISTINCT source_doc) AS count FROM medical_triples WHERE source_doc <> ''",
        limit=1,
    )[0]["count"]
    linked_docs = query_rows(
        db_path,
        """
        SELECT COUNT(DISTINCT d.doc_id) AS count
        FROM task1_documents d
        JOIN medical_triples t ON t.source_doc=d.doc_id
        """,
        limit=1,
    )[0]["count"]
    return {
        "task1_documents": task1_docs,
        "task1_segments": task1_segments,
        "medical_triples": triples,
        "evidence_source_docs": evidence_docs,
        "task1_docs_linked_to_triples": linked_docs,
        "triples_per_document": round(triples / task1_docs, 4) if task1_docs else 0.0,
        "triples_per_100_segments": round(100 * triples / task1_segments, 4) if task1_segments else 0.0,
        "task1_to_task2_doc_link_ratio": round(linked_docs / task1_docs, 4) if task1_docs else 0.0,
        "lineage_note": (
            "source_doc 与任务一 doc_id 匹配时，可计算端到端转化率；"
            "如果为 0，说明当前任务二图谱不是由这批任务一文档直接生成，任务三会如实标记链路缺口。"
        ),
    }


def build_document_knowledge_contribution(db_path: Path) -> list[dict[str, Any]]:
    return query_rows(
        db_path,
        """
        SELECT COALESCE(NULLIF(t.source_doc, ''), 'unknown') AS source_doc,
               COALESCE(d.disease_topic, '') AS disease_topic,
               COUNT(*) AS triple_count,
               COUNT(DISTINCT relation) AS relation_type_count
        FROM medical_triples t
        LEFT JOIN task1_documents d ON d.doc_id=t.source_doc
        GROUP BY source_doc, disease_topic
        ORDER BY triple_count DESC, source_doc
        LIMIT 30
        """,
        limit=30,
    )


def build_knowledge_density_by_topic(db_path: Path) -> list[dict[str, Any]]:
    return query_rows(
        db_path,
        """
        SELECT COALESCE(NULLIF(d.disease_topic, ''), COALESCE(NULLIF(t.head, ''), 'unknown')) AS topic,
               COUNT(DISTINCT d.doc_id) AS document_count,
               COUNT(t.triple_id) AS triple_count,
               CASE
                 WHEN COUNT(DISTINCT d.doc_id) > 0
                 THEN ROUND(CAST(COUNT(t.triple_id) AS REAL) / COUNT(DISTINCT d.doc_id), 4)
                 ELSE CAST(COUNT(t.triple_id) AS REAL)
               END AS triples_per_doc
        FROM medical_triples t
        LEFT JOIN task1_documents d ON d.doc_id=t.source_doc
        GROUP BY topic
        ORDER BY triple_count DESC, topic
        LIMIT 30
        """,
        limit=30,
    )


def build_evidence_path_samples(db_path: Path) -> list[dict[str, Any]]:
    return query_rows(
        db_path,
        """
        SELECT head, relation, tail, source_doc, evidence_text
        FROM evidence_paths
        WHERE evidence_text <> ''
        ORDER BY head, relation, tail
        LIMIT 20
        """,
        limit=20,
    )


def build_graph_quality(context: AnalysisContext, db_path: Path) -> dict[str, Any]:
    node_ids = {str(node.get("node_id")) for node in context.nodes if node.get("node_id")}
    connected_ids = set()
    evidence_edges = 0
    source_docs: set[str] = set()
    for edge in context.edges:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source:
            connected_ids.add(source)
        if target:
            connected_ids.add(target)
        evidence = edge.get("evidence") or []
        if evidence:
            evidence_edges += 1
        for item in evidence:
            if isinstance(item, dict):
                doc_id = item.get("doc_id") or item.get("source_doc")
                if doc_id:
                    source_docs.add(str(doc_id))
    node_count = len(node_ids)
    edge_count = len(context.edges)
    isolated_count = len(node_ids - connected_ids)
    density = edge_count / (node_count * (node_count - 1)) if node_count > 1 else 0.0
    duplicate_edges = query_rows(
        db_path,
        """
        SELECT subject, predicate, object, COUNT(*) AS count
        FROM edges
        GROUP BY subject, predicate, object
        HAVING count > 1
        ORDER BY count DESC
        LIMIT 20
        """,
        limit=20,
    )
    return {
        "evidence_edge_ratio": round(evidence_edges / edge_count, 4) if edge_count else 0.0,
        "isolated_node_count": isolated_count,
        "isolated_node_ratio": round(isolated_count / node_count, 4) if node_count else 0.0,
        "graph_density": round(density, 6),
        "average_degree": round((2 * edge_count) / node_count, 4) if node_count else 0.0,
        "source_doc_count": len(source_docs),
        "duplicate_edge_groups": duplicate_edges,
        "quality_summary": (
            f"证据覆盖率 {evidence_edges / edge_count:.1%}，"
            f"孤立节点比例 {isolated_count / node_count:.1%}，"
            f"平均度 {((2 * edge_count) / node_count) if node_count else 0.0:.2f}"
        )
        if edge_count and node_count
        else "图谱为空，无法计算质量指标。",
    }


def build_source_trend(context: AnalysisContext) -> dict[str, Any]:
    doc_counts: dict[str, int] = {}
    batch_counts: dict[str, int] = {}
    for edge in context.edges:
        properties = edge.get("properties") or {}
        import_ids = properties.get("custom_import_ids")
        if isinstance(import_ids, list):
            for import_id in import_ids:
                batch_counts[str(import_id)] = batch_counts.get(str(import_id), 0) + 1
        elif properties.get("custom_import_id"):
            import_id = str(properties["custom_import_id"])
            batch_counts[import_id] = batch_counts.get(import_id, 0) + 1
        evidence = edge.get("evidence") or []
        for item in evidence:
            if isinstance(item, dict):
                doc_id = str(item.get("doc_id") or item.get("source_doc") or "")
                if doc_id:
                    doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
    source_docs = [
        {"source_doc": key, "edge_count": value}
        for key, value in sorted(doc_counts.items(), key=lambda item: (-item[1], item[0]))[:20]
    ]
    batches = [
        {"batch": key, "edge_count": value}
        for key, value in sorted(batch_counts.items(), key=lambda item: (-item[1], item[0]))[:20]
    ]
    return {
        "trend_type": "source_or_import_batch_distribution",
        "source_docs": source_docs,
        "import_batches": batches,
        "note": "当前图谱没有标准时间字段；这里展示来源文档或导入批次的关系分布，可作为后续增量趋势分析基础。",
    }


def build_graph_schema_profile(context: AnalysisContext, db_path: Path) -> dict[str, Any]:
    entity_types = [row["type"] for row in query_rows(db_path, "SELECT DISTINCT type FROM nodes WHERE type <> '' ORDER BY type", limit=100)]
    relation_types = [
        row["predicate"]
        for row in query_rows(db_path, "SELECT DISTINCT predicate FROM edges WHERE predicate <> '' ORDER BY predicate", limit=100)
    ]
    candidate_rules = [
        ("疾病" in entity_types and bool(SYMPTOM_PREDICATES & set(relation_types)), "疾病-症状频次分析"),
        ("疾病" in entity_types and bool(TREATMENT_PREDICATES & set(relation_types)), "疾病-治疗/药物覆盖分析"),
        ("疾病" in entity_types and bool(TEST_PREDICATES & set(relation_types)), "疾病-检查项目覆盖分析"),
        (bool(entity_types), "实体类型分布分析"),
        (bool(relation_types), "关系类型分布分析"),
        (bool(context.edges), "图谱中心节点分析"),
        (bool(context.edges), "证据路径分析"),
        ("疾病" in entity_types, "知识缺口检测"),
    ]
    return {
        "entity_types": entity_types,
        "relation_types": relation_types,
        "node_type_counts": context.graph.get("metadata", {}).get("node_type_counts") or {},
        "predicate_counts": context.graph.get("metadata", {}).get("predicate_counts") or {},
        "analysis_candidates": [name for enabled, name in candidate_rules if enabled],
        "operator_catalog": ANALYZER_CATALOG,
        "triple_table": "medical_triples(head, head_type, relation, tail, tail_type, source_doc, evidence, confidence)",
        "graph_tables": ["nodes", "edges", "evidence_paths"],
        "task1_tables": ["task1_documents", "task1_segments"],
    }


def verify_sql_query(db_path: Path, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any]:
    normalized = " ".join(sql.strip().split())
    if not normalized.lower().startswith("select"):
        return {
            "status": "blocked",
            "passed": False,
            "reason": "Only SELECT statements are allowed in Task3 analysis.",
            "sql": sql,
            "repaired_sql": sql,
        }
    repaired_sql = normalized
    repairs = []
    replacements = {
        "relation_name": "relation",
        "predicate_name": "predicate",
        "source_document": "source_doc",
    }
    for old, new in replacements.items():
        if old in repaired_sql:
            repaired_sql = repaired_sql.replace(old, new)
            repairs.append({"from": old, "to": new})
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(f"EXPLAIN QUERY PLAN {repaired_sql}", params)
    except sqlite3.Error as exc:
        return {
            "status": "failed",
            "passed": False,
            "reason": str(exc),
            "sql": sql,
            "repaired_sql": repaired_sql,
            "repairs": repairs,
        }
    return {
        "status": "passed" if not repairs else "repaired",
        "passed": True,
        "reason": "SQL is executable and read-only.",
        "sql": sql,
        "repaired_sql": repaired_sql,
        "repairs": repairs,
    }


def execute_verified_query(db_path: Path, intent: dict[str, Any], limit: int = 100) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    params = tuple(intent.get("params") or ())
    verification = verify_sql_query(db_path, intent["sql"], params)
    if not verification["passed"]:
        return [], verification
    rows = query_rows(db_path, verification["repaired_sql"], params, limit=limit)
    if not rows and intent["intent"] not in {"task2_kgqa_redirect"}:
        verification = dict(verification)
        verification["empty_result"] = True
        verification["reason"] = verification["reason"] + " Result is empty; caller should surface this as an analysis gap."
    return rows, verification


def build_result_evidence_paths(db_path: Path, rows: list[dict[str, Any]], intent: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    candidates: list[dict[str, str]] = []
    if intent["intent"] in {"top_diseases", "disease_relation_profile"}:
        for row in rows:
            disease = str(row.get("disease") or "")
            if disease:
                candidates.extend(
                    query_rows(
                        db_path,
                        """
                        SELECT head, relation, tail, source_doc, evidence_text
                        FROM evidence_paths
                        WHERE head=?
                        ORDER BY relation, tail
                        LIMIT 3
                        """,
                        (disease,),
                        limit=3,
                    )
                )
    elif intent["intent"] in {"common_symptoms"}:
        for row in rows[:3]:
            symptom = str(row.get("symptom") or "")
            candidates.extend(
                query_rows(
                    db_path,
                    """
                    SELECT head, relation, tail, source_doc, evidence_text
                    FROM evidence_paths
                    WHERE tail=?
                    ORDER BY head, relation
                    LIMIT 3
                    """,
                    (symptom,),
                    limit=3,
                )
            )
    elif intent["intent"] in {"treatment_distribution", "drug_disease_association"}:
        for row in rows[:3]:
            treatment = str(row.get("treatment") or row.get("drug") or "")
            candidates.extend(
                query_rows(
                    db_path,
                    """
                    SELECT head, relation, tail, source_doc, evidence_text
                    FROM evidence_paths
                    WHERE tail=?
                    ORDER BY head, relation
                    LIMIT 3
                    """,
                    (treatment,),
                    limit=3,
                )
            )
    else:
        candidates.extend(build_evidence_path_samples(db_path)[:limit])
    seen = set()
    unique = []
    for item in candidates:
        key = (item.get("head"), item.get("relation"), item.get("tail"), item.get("source_doc"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= limit:
            break
    return unique


def build_result_subgraph(rows: list[dict[str, Any]], evidence_paths: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: dict[str, dict[str, str]] = {}
    edges: list[dict[str, str]] = []
    for item in evidence_paths:
        head = str(item.get("head") or "")
        tail = str(item.get("tail") or "")
        relation = str(item.get("relation") or "")
        if not head or not tail:
            continue
        nodes.setdefault(head, {"id": head, "label": head})
        nodes.setdefault(tail, {"id": tail, "label": tail})
        edges.append({"source": head, "target": tail, "label": relation})
    if not nodes:
        for row in rows[:10]:
            for key in ("disease", "subject", "head", "tail", "object", "symptom", "treatment", "drug"):
                value = str(row.get(key) or "")
                if value:
                    nodes.setdefault(value, {"id": value, "label": value})
    return {"nodes": list(nodes.values()), "edges": edges}


def detect_intent(question: str, db_path: Path) -> dict[str, Any]:
    text = question.strip()
    disease = detect_disease_name(text, db_path)
    if any(key in text for key in ["数据到知识", "转化率", "生成多少", "抽出多少", "任务一到任务二"]):
        return {
            "intent": "data_to_knowledge_lineage",
            "sql": "SELECT COUNT(*) AS triple_count FROM medical_triples",
            "params": [],
            "description": "分析任务一清洗数据到任务二三元组/图谱的转化情况。",
        }
    if any(key in text for key in ["文档贡献", "贡献知识", "贡献最大", "来源贡献", "哪些文档"]):
        return {
            "intent": "document_knowledge_contribution",
            "sql": "SELECT source_doc, disease_topic, triple_count, relation_type_count FROM (SELECT COALESCE(NULLIF(t.source_doc, ''), 'unknown') AS source_doc, COALESCE(d.disease_topic, '') AS disease_topic, COUNT(*) AS triple_count, COUNT(DISTINCT relation) AS relation_type_count FROM medical_triples t LEFT JOIN task1_documents d ON d.doc_id=t.source_doc GROUP BY source_doc, disease_topic) ORDER BY triple_count DESC, source_doc LIMIT 20",
            "params": [],
            "description": "统计每个来源文档贡献的三元组数量和关系类型数量。",
        }
    if any(key in text for key in ["知识密度", "主题密度", "疾病主题", "每条文本"]):
        return {
            "intent": "knowledge_density_by_topic",
            "sql": "SELECT COALESCE(NULLIF(d.disease_topic, ''), COALESCE(NULLIF(t.head, ''), 'unknown')) AS topic, COUNT(DISTINCT d.doc_id) AS document_count, COUNT(t.triple_id) AS triple_count FROM medical_triples t LEFT JOIN task1_documents d ON d.doc_id=t.source_doc GROUP BY topic ORDER BY triple_count DESC, topic LIMIT 20",
            "params": [],
            "description": "比较不同疾病主题或来源主题的知识密度。",
        }
    if any(key in text for key in ["中心节点", "核心节点", "中心性", "最核心"]):
        return {
            "intent": "graph_centrality",
            "sql": "SELECT name, type, degree FROM (SELECT name, type, SUM(degree) AS degree FROM (SELECT head AS name, head_type AS type, COUNT(*) AS degree FROM medical_triples GROUP BY head, head_type UNION ALL SELECT tail AS name, tail_type AS type, COUNT(*) AS degree FROM medical_triples GROUP BY tail, tail_type) GROUP BY name, type) ORDER BY degree DESC, name LIMIT 20",
            "params": [],
            "description": "统计图谱中度数最高的核心节点。",
        }
    if any(key in text for key in ["药物关联", "药物覆盖", "药物关联疾病", "关联药物", "药物和疾病"]):
        return {
            "intent": "drug_disease_association",
            "sql": "SELECT tail AS drug, COUNT(DISTINCT head) AS disease_count, COUNT(*) AS relation_count, GROUP_CONCAT(DISTINCT head) AS diseases FROM medical_triples WHERE head_type='疾病' AND tail_type='药物' AND relation IN ('药物治疗','治疗','用药') GROUP BY tail ORDER BY disease_count DESC, relation_count DESC, drug LIMIT 20",
            "params": [],
            "description": "分析药物关联的疾病数量和关系数量。",
        }
    entity_pair = detect_entity_pair(text, db_path)
    if entity_pair and any(key in text for key in ["证据", "路径"]):
        head, tail = entity_pair
        return {
            "intent": "evidence_path",
            "sql": "SELECT head, relation, tail, source_doc, evidence_text FROM evidence_paths WHERE (head=? AND tail=?) OR (head=? AND tail=?) ORDER BY relation, source_doc LIMIT 20",
            "params": [head, tail, tail, head],
            "description": "查询两个实体之间的图谱证据路径。",
        }
    if any(key in text for key in ["生成报告", "分析报告", "洞察报告", "总结报告"]):
        return {
            "intent": "insight_report",
            "sql": "SELECT predicate, COUNT(*) AS count FROM edges GROUP BY predicate ORDER BY count DESC, predicate",
            "params": [],
            "description": "生成整体洞察报告摘要。",
        }
    if disease and any(key in text for key in ["关系结构", "关系分布", "关系覆盖", "覆盖情况", "完整度", "缺口", "关联情况", "分析", "指标"]):
        return disease_relation_profile_intent(disease)
    if any(key in text for key in ["图谱质量", "质量怎么样", "证据覆盖", "孤立节点", "数据质量"]):
        return {
            "intent": "graph_quality",
            "sql": "SELECT COUNT(*) AS edge_count FROM edges",
            "params": [],
            "description": "查看图谱质量、证据覆盖和孤立节点等质量指标。",
        }
    if any(key in text for key in ["常见症状", "症状最多", "高频症状", "症状关联", "症状覆盖疾病"]):
        return {
            "intent": "common_symptoms",
            "sql": "SELECT object AS symptom, COUNT(DISTINCT subject) AS disease_count, COUNT(*) AS relation_count FROM edges WHERE subject_type='疾病' AND predicate IN ('临床表现','症状','主要症状') GROUP BY object ORDER BY disease_count DESC, relation_count DESC, symptom LIMIT 20",
            "params": [],
            "description": "统计连接疾病最多的症状。",
        }
    if any(key in text for key in ["治疗覆盖", "常用治疗", "常用药", "治疗分布", "药物分布", "用药覆盖"]):
        return {
            "intent": "treatment_distribution",
            "sql": "SELECT object AS treatment, object_type AS type, COUNT(DISTINCT subject) AS disease_count, COUNT(*) AS relation_count FROM edges WHERE subject_type='疾病' AND predicate IN ('药物治疗','辅助治疗','治疗方法','治疗','用药') GROUP BY object, object_type ORDER BY disease_count DESC, relation_count DESC, treatment LIMIT 20",
            "params": [],
            "description": "统计治疗或药物覆盖的疾病数量。",
        }
    if any(key in text for key in ["关联分析", "关系矩阵", "疾病关系矩阵", "疾病和关系类型"]):
        return {
            "intent": "association_matrix",
            "sql": "SELECT subject AS disease, predicate, COUNT(*) AS count FROM edges WHERE subject_type='疾病' GROUP BY subject, predicate ORDER BY subject, count DESC",
            "params": [],
            "description": "分析疾病和关系类型之间的关联分布。",
        }
    if any(key in text for key in ["趋势", "来源", "批次", "增量"]):
        return {
            "intent": "source_trend",
            "sql": "SELECT subject, predicate, object, evidence_text FROM edges ORDER BY subject LIMIT 100",
            "params": [],
            "description": "分析来源文档或导入批次的关系分布。",
        }
    if any(key in text for key in ["实体分布", "实体统计", "各类实体", "节点分布", "实体类型"]):
        return {
            "intent": "entity_distribution",
            "sql": "SELECT type, COUNT(*) AS count FROM nodes GROUP BY type ORDER BY count DESC, type",
            "params": [],
            "description": "统计各类实体节点数量。",
        }
    if any(key in text for key in ["关系分布", "关系统计", "三元组分布", "边分布", "哪类关系", "关系类型"]):
        return {
            "intent": "relation_distribution",
            "sql": "SELECT predicate, COUNT(*) AS count FROM edges GROUP BY predicate ORDER BY count DESC, predicate",
            "params": [],
            "description": "统计各类图谱关系数量。",
        }
    if any(key in text for key in ["最多关系", "关系最多", "关联最多", "中心疾病", "核心疾病", "top", "TOP"]):
        return {
            "intent": "top_diseases",
            "sql": "SELECT subject AS disease, COUNT(*) AS relation_count FROM edges WHERE subject_type='疾病' GROUP BY subject ORDER BY relation_count DESC, disease LIMIT 20",
            "params": [],
            "description": "统计关系数量最多的疾病节点。",
        }
    if any(key in text for key in ["知识缺口", "缺少", "覆盖率", "完整度", "信息不完整", "结构不完整", "不完整"]):
        return {
            "intent": "coverage_gap",
            "sql": "SELECT subject AS disease, predicate, COUNT(*) AS count FROM edges WHERE subject_type='疾病' GROUP BY subject, predicate ORDER BY disease, count DESC",
            "params": [],
            "description": "分析疾病节点的症状、检查、治疗关系覆盖情况。",
        }
    if any(key in text for key in ["疾病列表", "有哪些疾病", "可问疾病"]):
        return {
            "intent": "disease_list",
            "sql": "SELECT name AS disease FROM nodes WHERE type='疾病' ORDER BY disease LIMIT 100",
            "params": [],
            "description": "列出当前图谱中的疾病节点。",
        }
    if disease and any(key in text for key in ["症状", "表现", "检查", "诊断", "治疗", "用药", "药", "医治", "是什么"]):
        return {
            "intent": "task2_kgqa_redirect",
            "sql": "SELECT name AS disease, type FROM nodes WHERE type='疾病' AND name=? LIMIT 1",
            "params": [disease],
            "description": "识别到医疗事实问答；该类问题属于任务二知识图谱问答，不作为任务三分析问题处理。",
        }
    if disease:
        return disease_relation_profile_intent(disease)
    return {
        "intent": "overview",
        "sql": "SELECT predicate, COUNT(*) AS count FROM edges GROUP BY predicate ORDER BY count DESC, predicate",
        "params": [],
        "description": "默认返回图谱关系分布概览。",
    }


def disease_relation_profile_intent(disease: str) -> dict[str, Any]:
    return {
        "intent": "disease_relation_profile",
        "sql": (
            "SELECT subject AS disease, predicate, object_type, COUNT(*) AS relation_count "
            "FROM edges "
            "WHERE (subject=? OR subject IN (SELECT object FROM edges WHERE subject=? AND predicate IN ('病理分型','分型'))) "
            "GROUP BY subject, predicate, object_type ORDER BY disease, relation_count DESC, predicate"
        ),
        "params": [disease, disease],
        "description": "分析指定疾病及其分型节点的关系结构和覆盖情况。",
    }


def detect_disease_name(text: str, db_path: Path) -> str | None:
    diseases = query_rows(db_path, "SELECT name FROM nodes WHERE type='疾病' ORDER BY LENGTH(name) DESC", limit=1000)
    for row in diseases:
        name = str(row["name"])
        if name and name in text:
            return name
    return None


def detect_entity_pair(text: str, db_path: Path) -> tuple[str, str] | None:
    entities = query_rows(db_path, "SELECT name FROM nodes WHERE name <> '' ORDER BY LENGTH(name) DESC", limit=2000)
    matched = []
    for row in entities:
        name = str(row["name"])
        if name and name in text and name not in matched:
            matched.append(name)
        if len(matched) >= 2:
            return matched[0], matched[1]
    return None


def answer_from_rows(question: str, intent: dict[str, Any], rows: list[dict[str, Any]], metrics: dict[str, Any]) -> str:
    if intent["intent"] == "task2_kgqa_redirect":
        disease = intent.get("params", [""])[0]
        if rows:
            return (
                f"识别到问题涉及 {disease} 的医疗事实问答。"
                "这类问题属于任务二“知识图谱问答智能体”；任务三只做图谱结构、指标、覆盖率、关联和 BI 洞察分析。"
                f"可以在任务三中改问：分析 {disease} 的关系覆盖情况。"
            )
        return "识别到医疗事实问答，但当前图谱中没有匹配到对应疾病节点。"
    if not rows and intent["intent"].startswith("disease_"):
        disease = intent.get("params", [""])[0]
        return f"当前图谱中找到了 {disease}，但没有匹配到该分析问题需要的关系。"
    if intent["intent"] == "entity_distribution":
        return "实体分布：" + "；".join(f"{row['type']} {row['count']} 个" for row in rows)
    if intent["intent"] == "relation_distribution":
        return "关系分布：" + "；".join(f"{row['predicate']} {row['count']} 条" for row in rows)
    if intent["intent"] == "data_to_knowledge_lineage":
        lineage = metrics.get("data_to_knowledge_lineage") or {}
        return (
            "数据到知识转化："
            f"任务一文档 {lineage.get('task1_documents')} 条、片段 {lineage.get('task1_segments')} 个，"
            f"任务二三元组 {lineage.get('medical_triples')} 条；"
            f"每 100 个片段生成 {lineage.get('triples_per_100_segments')} 条三元组；"
            f"任务一文档与三元组来源匹配率 {float(lineage.get('task1_to_task2_doc_link_ratio') or 0):.2%}。"
        )
    if intent["intent"] == "document_knowledge_contribution":
        return "文档知识贡献：" + "；".join(
            f"{row['source_doc']} 贡献 {row['triple_count']} 条三元组" for row in rows[:10]
        )
    if intent["intent"] == "knowledge_density_by_topic":
        return "知识密度：" + "；".join(
            f"{row['topic']} {row['triple_count']} 条三元组" for row in rows[:10]
        )
    if intent["intent"] == "graph_centrality":
        return "核心节点：" + "；".join(f"{row['name']}({row['type']}) 度数 {row['degree']}" for row in rows[:10])
    if intent["intent"] == "drug_disease_association":
        return "药物-疾病关联：" + "；".join(
            f"{row['drug']} 关联 {row['disease_count']} 个疾病" for row in rows[:10]
        )
    if intent["intent"] == "evidence_path":
        if not rows:
            return "当前图谱中没有找到这两个实体之间的直接证据路径。"
        return "证据路径：" + "；".join(
            f"{row['head']} --{row['relation']}--> {row['tail']}（来源 {row['source_doc']}）" for row in rows[:8]
        )
    if intent["intent"] == "insight_report":
        return "已生成图谱洞察报告，包含图谱结构、数据到知识转化、中心节点、知识缺口、证据路径和 BI 看板。"
    if intent["intent"] == "top_diseases":
        return "关系最多的疾病：" + "；".join(f"{row['disease']} {row['relation_count']} 条" for row in rows[:10])
    if intent["intent"] == "disease_list":
        return "当前图谱包含疾病：" + "、".join(row["disease"] for row in rows[:30])
    if intent["intent"] == "coverage_gap":
        coverage = metrics["coverage"]
        examples = "；".join(f"{item['disease']} 缺少 {','.join(item['missing'])}" for item in coverage["gap_examples"][:8])
        return f"疾病关系完整度为 {coverage['complete_ratio']:.2%}。知识缺口示例：{examples or '暂无明显缺口'}。"
    if intent["intent"] == "graph_quality":
        quality = metrics.get("graph_quality") or {}
        return (
            "图谱质量摘要："
            f"{quality.get('quality_summary')}；"
            f"来源文档数 {quality.get('source_doc_count')}；"
            f"重复边组数 {len(quality.get('duplicate_edge_groups') or [])}。"
        )
    if intent["intent"] == "common_symptoms":
        return "高频症状：" + "；".join(
            f"{row['symptom']} 覆盖 {row['disease_count']} 个疾病" for row in rows[:10]
        )
    if intent["intent"] == "treatment_distribution":
        return "治疗覆盖：" + "；".join(
            f"{row['treatment']} 覆盖 {row['disease_count']} 个疾病" for row in rows[:10]
        )
    if intent["intent"] == "association_matrix":
        grouped: dict[str, list[str]] = {}
        for row in rows:
            grouped.setdefault(str(row.get("disease")), []).append(f"{row.get('predicate')} {row.get('count')} 条")
        return "疾病-关系关联：" + "；".join(
            f"{disease}: {'、'.join(items[:5])}" for disease, items in list(grouped.items())[:8]
        )
    if intent["intent"] == "source_trend":
        source_trend = metrics.get("source_trend") or {}
        docs = source_trend.get("source_docs") or []
        batches = source_trend.get("import_batches") or []
        if batches:
            return "导入批次分布：" + "；".join(f"{item['batch']} {item['edge_count']} 条关系" for item in batches[:8])
        if docs:
            return "来源文档分布：" + "；".join(f"{item['source_doc']} {item['edge_count']} 条关系" for item in docs[:8])
        return str(source_trend.get("note") or "当前图谱缺少来源或批次信息，无法形成趋势分布。")
    if intent["intent"] == "disease_relation_profile":
        disease = str(intent.get("params", [""])[0])
        grouped: dict[str, int] = {}
        subjects = set()
        for row in rows:
            subjects.add(str(row.get("disease") or disease))
            predicate = str(row.get("predicate") or "其他关系")
            grouped[predicate] = grouped.get(predicate, 0) + int(row.get("relation_count") or 0)
        relation_summary = "；".join(f"{predicate} {count} 条" for predicate, count in grouped.items())
        return (
            f"{disease} 相关节点覆盖 {len(subjects)} 个疾病/分型节点，"
            f"关系结构为：{relation_summary or '暂无关系'}。"
        )
    return f"共返回 {len(rows)} 条结果。"


def run_analysis(
    graph_path: Path | str | None = None,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    task: str = DEFAULT_TASK,
    task1_report_path: Path | str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = ensure_dir(Path(output_dir))
    context = load_graph(graph_path)
    task1_profile = load_task1_profile(task1_report_path)
    db_path = output / "medical_kg_analysis.sqlite"
    database = create_analysis_database(context, db_path, task1_profile)
    metrics = build_metrics(db_path)
    metrics["graph_quality"] = build_graph_quality(context, db_path)
    metrics["source_trend"] = build_source_trend(context)
    metrics["graph_schema_profile"] = build_graph_schema_profile(context, db_path)
    nl2sql_eval = evaluate_nl2sql(db_path)
    insights = build_insights(str(context.graph_path), task1_profile, metrics, nl2sql_eval)
    insight_report_path = output / "insight_report.md"
    write_insight_report(insight_report_path, insights, metrics, nl2sql_eval)
    visualization_path = output / "insight_dashboard.html"
    visualization = build_visualization(context, metrics, visualization_path, task1_profile, insights)
    report = {
        "status": "ok",
        "task": task,
        "graph_path": str(context.graph_path),
        "task1_reuse": task1_profile,
        "database": database,
        "metrics": metrics,
        "nl2sql_evaluation": nl2sql_eval,
        "insights": insights,
        "agent_plan": build_agent_analysis_plan(metrics),
        "insight_report_path": str(insight_report_path),
        "visualization": visualization,
        "runtime": "task3_graph_aware_analysis_agent",
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    write_json(output / "task3_analysis_report.json", report)
    return {
        "status": "ok",
        "report_path": str(output / "task3_analysis_report.json"),
        "insight_report_path": str(insight_report_path),
        "db_path": str(db_path),
        "visualization_path": str(visualization_path),
        "graph_path": str(context.graph_path),
        "task1_reuse": task1_profile,
        "database": database,
        "metrics": metrics,
        "nl2sql_evaluation": nl2sql_eval,
        "insights": insights,
        "agent_plan": build_agent_analysis_plan(metrics),
    }


def run_query(
    question: str,
    graph_path: Path | str | None = None,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    task1_report_path: Path | str | None = None,
) -> dict[str, Any]:
    analysis = run_analysis(graph_path, output_dir, task1_report_path=task1_report_path)
    result = run_query_on_analysis(question, analysis, output_dir)
    write_json(Path(output_dir) / "task3_query_result.json", result)
    return result


def run_query_on_analysis(
    question: str,
    analysis: dict[str, Any],
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    db_path = Path(analysis["db_path"])
    intent = detect_intent(question, db_path)
    rows, sql_verification = execute_verified_query(db_path, intent, limit=100)
    answer = (
        answer_from_rows(question, intent, rows, analysis["metrics"])
        if sql_verification["passed"]
        else f"SQL 校验失败：{sql_verification['reason']}"
    )
    evidence_paths = build_result_evidence_paths(db_path, rows, intent)
    subgraph = build_result_subgraph(rows, evidence_paths)
    return {
        "status": "ok",
        "execution_engine": "graph_aware_nl2sql+sqlite+kg_path",
        "question": question,
        "intent": intent["intent"],
        "description": intent["description"],
        "sql": intent["sql"],
        "verified_sql": sql_verification.get("repaired_sql"),
        "sql_verification": sql_verification,
        "params": intent["params"],
        "rows": rows,
        "answer": answer,
        "evidence_paths": evidence_paths,
        "subgraph": subgraph,
        "analysis_report_path": analysis["report_path"],
        "visualization_path": analysis["visualization_path"],
    }


def load_analysis_config(config: dict[str, Any] | str | Path | None = None) -> dict[str, Any]:
    if config is None:
        path = DEFAULT_CONFIG_PATH
        if not path.exists():
            return {
                "task": DEFAULT_TASK,
                "analysis_tasks": [
                    "统计实体分布",
                    "统计关系分布",
                    "哪些疾病关系最多",
                    "知识缺口有哪些",
                    "图谱质量怎么样",
                    "治疗覆盖怎么样",
                ],
            }
        return json.loads(path.read_text(encoding="utf-8"))
    if isinstance(config, dict):
        return config
    path = Path(config)
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_configured_questions(config: dict[str, Any]) -> list[dict[str, str]]:
    raw_tasks = config.get("analysis_tasks") or config.get("questions") or config.get("tasks") or []
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(raw_tasks, start=1):
        if isinstance(item, str):
            normalized.append({"name": f"analysis_{index}", "question": item})
        elif isinstance(item, dict):
            question = str(item.get("question") or item.get("task") or item.get("analysis") or "").strip()
            if question:
                normalized.append({"name": str(item.get("name") or f"analysis_{index}"), "question": question})
    return normalized


def run_configured_analysis(
    config: dict[str, Any] | str | Path | None = None,
    graph_path: Path | str | None = None,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    task1_report_path: Path | str | None = None,
) -> dict[str, Any]:
    output = ensure_dir(Path(output_dir))
    loaded_config = load_analysis_config(config)
    resolved_graph_path = graph_path or loaded_config.get("graph_path")
    resolved_task1_report = task1_report_path or loaded_config.get("task1_report_path")
    task = str(loaded_config.get("task") or DEFAULT_TASK)
    analysis = run_analysis(
        graph_path=resolved_graph_path,
        output_dir=output,
        task=task,
        task1_report_path=resolved_task1_report,
    )
    configured_results = []
    for item in normalize_configured_questions(loaded_config):
        query_result = run_query_on_analysis(item["question"], analysis, output)
        configured_results.append(
            {
                "name": item["name"],
                "question": item["question"],
                "intent": query_result["intent"],
                "sql": query_result["sql"],
                "params": query_result["params"],
                "row_count": len(query_result["rows"]),
                "answer": query_result["answer"],
            }
        )
    report = {
        "status": "ok",
        "trigger_mode": "config",
        "config": loaded_config,
        "analysis": analysis,
        "configured_results": configured_results,
        "report_path": analysis["report_path"],
        "insight_report_path": analysis["insight_report_path"],
        "db_path": analysis["db_path"],
        "visualization_path": analysis["visualization_path"],
    }
    write_json(output / "task3_configured_analysis_report.json", report)
    return report


def evaluate_nl2sql(db_path: Path) -> dict[str, Any]:
    cases = [
        ("统计实体分布", "entity_distribution"),
        ("当前图谱有哪些实体类型", "entity_distribution"),
        ("各类实体数量分别是多少", "entity_distribution"),
        ("节点分布情况", "entity_distribution"),
        ("统计关系分布", "relation_distribution"),
        ("当前图谱中哪类关系最多", "relation_distribution"),
        ("关系统计", "relation_distribution"),
        ("三元组分布怎么样", "relation_distribution"),
        ("哪些疾病关系最多", "top_diseases"),
        ("关系最多的疾病节点", "top_diseases"),
        ("当前核心疾病有哪些", "top_diseases"),
        ("当前有哪些疾病", "disease_list"),
        ("列出疾病列表", "disease_list"),
        ("知识缺口有哪些", "coverage_gap"),
        ("哪些疾病节点信息不完整", "coverage_gap"),
        ("疾病关系完整度怎么样", "coverage_gap"),
        ("哪些疾病缺少检查或治疗关系", "coverage_gap"),
        ("图谱质量怎么样", "graph_quality"),
        ("证据覆盖情况怎么样", "graph_quality"),
        ("数据质量怎么样", "graph_quality"),
        ("孤立节点比例是多少", "graph_quality"),
        ("常见症状有哪些", "common_symptoms"),
        ("哪些症状覆盖疾病最多", "common_symptoms"),
        ("高频症状分析", "common_symptoms"),
        ("症状关联情况", "common_symptoms"),
        ("治疗覆盖怎么样", "treatment_distribution"),
        ("常用治疗分布", "treatment_distribution"),
        ("治疗分布情况", "treatment_distribution"),
        ("用药覆盖怎么样", "treatment_distribution"),
        ("做一个疾病关系矩阵", "association_matrix"),
        ("疾病和关系类型之间有什么关联", "association_matrix"),
        ("关系矩阵分析", "association_matrix"),
        ("来源趋势怎么看", "source_trend"),
        ("来源文档分布", "source_trend"),
        ("导入批次趋势", "source_trend"),
        ("增量来源分析", "source_trend"),
        ("任务一到任务二的转化率", "data_to_knowledge_lineage"),
        ("数据到知识生成多少三元组", "data_to_knowledge_lineage"),
        ("每100个片段生成多少知识", "data_to_knowledge_lineage"),
        ("哪些文档贡献知识最多", "document_knowledge_contribution"),
        ("来源贡献分析", "document_knowledge_contribution"),
        ("文档贡献最大的是谁", "document_knowledge_contribution"),
        ("不同疾病主题的知识密度", "knowledge_density_by_topic"),
        ("主题密度对比", "knowledge_density_by_topic"),
        ("疾病主题生成三元组情况", "knowledge_density_by_topic"),
        ("当前图谱最核心的节点是什么", "graph_centrality"),
        ("中心节点分析", "graph_centrality"),
        ("图谱中心性排行", "graph_centrality"),
        ("哪些药物关联疾病最多", "drug_disease_association"),
        ("药物关联疾病分析", "drug_disease_association"),
        ("药物和疾病关联分布", "drug_disease_association"),
        ("生成一份洞察报告", "insight_report"),
        ("请生成图谱分析报告", "insight_report"),
    ]
    diseases = query_rows(db_path, "SELECT name FROM nodes WHERE type='疾病' ORDER BY name LIMIT 3", limit=3)
    for row in diseases:
        disease = row["name"]
        cases.extend(
            [
                (f"分析{disease}的关系结构", "disease_relation_profile"),
                (f"{disease}关系覆盖情况", "disease_relation_profile"),
                (f"{disease}的完整度指标", "disease_relation_profile"),
            ]
        )
    pair_rows = query_rows(db_path, "SELECT head, tail FROM evidence_paths LIMIT 1", limit=1)
    if pair_rows:
        cases.append((f"{pair_rows[0]['head']}和{pair_rows[0]['tail']}有什么证据路径", "evidence_path"))
    correct = 0
    details = []
    for question, expected in cases:
        predicted = detect_intent(question, db_path)["intent"]
        ok = predicted == expected
        correct += int(ok)
        details.append({"question": question, "expected_intent": expected, "predicted_intent": predicted, "correct": ok})
    total = len(details)
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "target_accuracy": 0.85,
        "passed": (correct / total) >= 0.85 if total else False,
        "details": details,
    }


def bar_items(rows: list[dict[str, Any]], label_key: str, value_key: str, max_items: int = 12) -> str:
    values = rows[:max_items]
    max_value = max([int(row.get(value_key) or 0) for row in values] or [1])
    parts = []
    for row in values:
        label = html.escape(str(row.get(label_key) or ""))
        value = int(row.get(value_key) or 0)
        width = 8 + int(88 * value / max_value) if max_value else 8
        parts.append(
            f'<div class="bar-row"><span class="bar-label">{label}</span>'
            f'<span class="bar-track"><span class="bar-fill" style="width:{width}%"></span></span>'
            f'<span class="bar-value">{value}</span></div>'
        )
    return "\n".join(parts)


def build_graph_svg(context: AnalysisContext, max_nodes: int = 18) -> str:
    selected_nodes = context.nodes[:max_nodes]
    ids = [str(node.get("node_id")) for node in selected_nodes]
    node_index = {node_id: index for index, node_id in enumerate(ids)}
    width, height = 780, 420
    center_x, center_y = width / 2, height / 2
    radius = 160
    import math

    positions: dict[str, tuple[float, float]] = {}
    for idx, node_id in enumerate(ids):
        angle = 2 * math.pi * idx / max(len(ids), 1)
        positions[node_id] = (center_x + radius * math.cos(angle), center_y + radius * math.sin(angle))
    lines = []
    for edge in context.edges:
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source in positions and target in positions:
            x1, y1 = positions[source]
            x2, y2 = positions[target]
            lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" />')
    circles = []
    for node in selected_nodes:
        node_id = str(node.get("node_id"))
        x, y = positions[node_id]
        label = html.escape(str(node.get("name") or "")[:8])
        node_type_value = str(node.get("type") or "")
        cls = "disease" if node_type_value == "疾病" else "entity"
        circles.append(f'<g class="{cls}"><circle cx="{x:.1f}" cy="{y:.1f}" r="18" /><text x="{x:.1f}" y="{y + 36:.1f}">{label}</text></g>')
    return f'<svg viewBox="0 0 {width} {height}" role="img">{"".join(lines)}{"".join(circles)}</svg>'


def build_visualization(
    context: AnalysisContext,
    metrics: dict[str, Any],
    path: Path,
    task1_profile: dict[str, Any] | None = None,
    insights: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    ensure_dir(path.parent)
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MedInsight Task3 图谱洞察看板</title>
  <style>
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#18212f; background:#f5f7fb; }}
    header {{ padding:28px 36px; background:#fff; border-bottom:1px solid #dfe5ee; }}
    h1 {{ margin:0 0 8px; font-size:28px; }}
    main {{ padding:24px 36px 44px; display:grid; gap:20px; }}
    .summary {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; }}
    .metric,.panel {{ background:#fff; border:1px solid #dfe5ee; border-radius:8px; padding:16px; }}
    .metric strong {{ display:block; font-size:26px; margin-top:8px; }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
    .bar-row {{ display:grid; grid-template-columns:96px 1fr 54px; gap:10px; align-items:center; margin:9px 0; }}
    .bar-label {{ font-size:13px; color:#445066; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .bar-track {{ height:10px; background:#edf1f6; border-radius:99px; overflow:hidden; }}
    .bar-fill {{ display:block; height:100%; background:#2166ac; border-radius:99px; }}
    .bar-value {{ font-variant-numeric:tabular-nums; text-align:right; }}
    .insight {{ border-bottom:1px solid #e7ebf1; padding:10px 0; }}
    .insight:last-child {{ border-bottom:0; }}
    .insight h3 {{ margin:0 0 6px; font-size:16px; }}
    .insight p {{ margin:0 0 4px; }}
    .insight small {{ color:#5d6a7c; display:block; margin-top:4px; }}
    table {{ width:100%; border-collapse:collapse; font-size:14px; }}
    th,td {{ border-bottom:1px solid #e7ebf1; padding:9px 6px; text-align:left; }}
    svg {{ width:100%; height:420px; background:#fbfcfe; border:1px solid #e4e9f0; border-radius:8px; }}
    svg line {{ stroke:#aab6c6; stroke-width:1.2; }}
    svg circle {{ fill:#e8f2ff; stroke:#2166ac; stroke-width:1.5; }}
    svg .disease circle {{ fill:#fff2d6; stroke:#c17b12; }}
    svg text {{ font-size:12px; text-anchor:middle; fill:#243044; }}
  </style>
</head>
<body>
  <header>
    <h1>MedInsight Task3 图谱洞察看板</h1>
    <div>图谱来源：{html.escape(str(context.graph_path))}</div>
  </header>
  <main>
    <section class="summary">
      <div class="metric">实体节点<strong>{metrics['node_count']}</strong></div>
      <div class="metric">图谱关系<strong>{metrics['edge_count']}</strong></div>
      <div class="metric">疾病节点<strong>{metrics['disease_count']}</strong></div>
      <div class="metric">完整关系覆盖<strong>{metrics['coverage']['complete_ratio']:.1%}</strong></div>
      <div class="metric">任务一文档<strong>{metrics['data_to_knowledge_lineage']['task1_documents']}</strong></div>
      <div class="metric">三元组<strong>{metrics['data_to_knowledge_lineage']['medical_triples']}</strong></div>
    </section>
    <section class="panel"><h2>Agent 执行计划</h2>{agent_plan_html(build_agent_analysis_plan(metrics))}</section>
    <section class="panel"><h2>任务一数据处理复用</h2>{task1_reuse_html(task1_profile or {})}</section>
    <section class="panel"><h2>数据 → 知识转化</h2>{lineage_html(metrics)}</section>
    <section class="panel"><h2>图文分析说明</h2>{insights_html(insights or [])}</section>
    <section class="panel"><h2>图谱质量摘要</h2>{graph_quality_html(metrics)}</section>
    <section class="grid">
      <div class="panel"><h2>实体分布</h2>{bar_items(metrics['entity_distribution'], 'type', 'count')}</div>
      <div class="panel"><h2>关系分布</h2>{bar_items(metrics['relation_distribution'], 'predicate', 'count')}</div>
    </section>
    <section class="grid">
      <div class="panel"><h2>关系最多的疾病</h2>{bar_items(metrics['top_diseases'], 'disease', 'relation_count')}</div>
      <div class="panel"><h2>知识缺口示例</h2><table><thead><tr><th>疾病</th><th>缺失关系</th><th>已有关系数</th></tr></thead><tbody>{gap_rows(metrics)}</tbody></table></div>
    </section>
    <section class="grid">
      <div class="panel"><h2>高频症状</h2>{bar_items(metrics['common_symptoms'], 'symptom', 'disease_count')}</div>
      <div class="panel"><h2>治疗覆盖</h2>{bar_items(metrics['treatment_distribution'], 'treatment', 'disease_count')}</div>
    </section>
    <section class="grid">
      <div class="panel"><h2>中心节点</h2>{bar_items(metrics['graph_centrality'], 'name', 'degree')}</div>
      <div class="panel"><h2>药物-疾病关联</h2>{bar_items(metrics['drug_disease_association'], 'drug', 'disease_count')}</div>
    </section>
    <section class="grid">
      <div class="panel"><h2>疾病-关系矩阵</h2><table><thead><tr><th>疾病</th><th>症状</th><th>检查</th><th>治疗</th><th>病因/分型</th></tr></thead><tbody>{relation_matrix_rows(metrics)}</tbody></table></div>
      <div class="panel"><h2>来源/批次分布</h2>{source_trend_html(metrics)}</div>
    </section>
    <section class="panel"><h2>证据路径样例</h2>{evidence_path_html(metrics)}</section>
    <section class="panel"><h2>图谱结构预览</h2>{build_graph_svg(context)}</section>
  </main>
</body>
</html>"""
    path.write_text(html_text, encoding="utf-8")
    return {
        "path": str(path),
        "format": "html",
        "sections": [
            "summary",
            "task1_reuse",
            "agent_plan",
            "data_to_knowledge_lineage",
            "insights",
            "graph_quality",
            "entity_distribution",
            "relation_distribution",
            "top_diseases",
            "knowledge_gaps",
            "common_symptoms",
            "treatment_distribution",
            "graph_centrality",
            "drug_disease_association",
            "association_matrix",
            "source_trend",
            "evidence_paths",
            "graph_preview",
        ],
    }


def build_agent_analysis_plan(metrics: dict[str, Any]) -> list[dict[str, str]]:
    schema = metrics.get("graph_schema_profile") or {}
    return [
        {"step": "问题理解", "tool": "analysis_intent_router", "output": "识别自然语言/配置化分析需求。"},
        {"step": "图谱感知", "tool": "graph_schema_parser", "output": f"实体类型 {len(schema.get('entity_types') or [])} 类，关系类型 {len(schema.get('relation_types') or [])} 类。"},
        {"step": "双引擎选择", "tool": "sql_kg_query_planner", "output": "统计类问题走 SQLite，证据类问题补图谱路径。"},
        {"step": "查询校验", "tool": "sql_verifier", "output": "检查 SQL 只读、字段合法、可执行计划。"},
        {"step": "分析算子", "tool": "analysis_operator_library", "output": f"可用算子 {len(ANALYZER_CATALOG)} 个。"},
        {"step": "可信洞察", "tool": "evidence_grounded_insight_builder", "output": "生成带数据依据、证据路径和建议的洞察卡。"},
        {"step": "图文展示", "tool": "bi_dashboard_builder", "output": "输出 BI 图表、图谱子图和 Markdown/HTML 报告。"},
    ]


def build_insights(
    graph_path: str,
    task1_profile: dict[str, Any],
    metrics: dict[str, Any],
    nl2sql_eval: dict[str, Any],
) -> list[dict[str, str]]:
    insights: list[dict[str, str]] = []
    lineage = metrics.get("data_to_knowledge_lineage") or {}
    insights.append(
        {
            "title": "数据到知识的输入链路",
            "finding": (
                f"任务三读取图谱 {graph_path}，并复用任务一报告 "
                f"{task1_profile.get('report_path', '未发现')}。"
            ),
            "evidence": (
                f"任务一输入 {task1_profile.get('input_records')} 条，"
                f"片段 {task1_profile.get('segments')} 个，"
                f"质量门控 {task1_profile.get('quality_gate_status')}。"
            ),
            "data_basis": (
                f"任务一文档 {lineage.get('task1_documents')} 条，"
                f"任务二三元组 {lineage.get('medical_triples')} 条，"
                f"doc_id 匹配率 {float(lineage.get('task1_to_task2_doc_link_ratio') or 0):.2%}。"
            ),
            "confidence": "0.82" if lineage.get("task1_docs_linked_to_triples") else "0.62",
            "recommendation": (
                "建议在任务二构图时优先使用任务一 processed_documents 的 doc_id 作为 source_doc，"
                "这样任务三可计算更完整的端到端转化率。"
            ),
        }
    )
    top_entities = metrics.get("entity_distribution") or []
    top_relations = metrics.get("relation_distribution") or []
    top_diseases = metrics.get("top_diseases") or []
    if top_entities:
        insights.append(
            {
                "title": "实体结构",
                "finding": f"当前图谱共有 {metrics.get('node_count')} 个实体节点，主要实体类型为 {top_entities[0].get('type')}。",
                "evidence": f"{top_entities[0].get('type')} 节点数量为 {top_entities[0].get('count')}。",
                "data_basis": "SQL: SELECT type, COUNT(*) FROM nodes GROUP BY type",
                "confidence": "0.90",
                "recommendation": "可结合实体类型分布判断当前数据是否偏向疾病、症状、药物或检查。",
            }
        )
    if top_relations:
        insights.append(
            {
                "title": "关系结构",
                "finding": f"当前图谱共有 {metrics.get('edge_count')} 条关系，最主要关系为 {top_relations[0].get('predicate')}。",
                "evidence": f"{top_relations[0].get('predicate')} 关系数量为 {top_relations[0].get('count')}。",
                "data_basis": "SQL: SELECT predicate, COUNT(*) FROM edges GROUP BY predicate",
                "confidence": "0.90",
                "recommendation": "若某类关系过少，可回到任务二补充对应类型医疗文本。",
            }
        )
    if top_diseases:
        insights.append(
            {
                "title": "核心疾病节点",
                "finding": f"关系最多的疾病是 {top_diseases[0].get('disease')}。",
                "evidence": f"该疾病关联 {top_diseases[0].get('relation_count')} 条图谱关系。",
                "data_basis": "SQL: SELECT subject, COUNT(*) FROM edges WHERE subject_type='疾病' GROUP BY subject",
                "confidence": "0.86",
                "recommendation": "核心疾病可优先用于演示中心子图、知识缺口和证据路径分析。",
            }
        )
    quality = metrics.get("graph_quality") or {}
    insights.append(
        {
            "title": "图谱质量",
            "finding": str(quality.get("quality_summary") or "暂无质量摘要。"),
            "evidence": f"来源文档数 {quality.get('source_doc_count')}，重复边组数 {len(quality.get('duplicate_edge_groups') or [])}。",
            "data_basis": "图谱边证据、节点连通性和重复三元组统计。",
            "confidence": "0.88",
            "recommendation": "证据覆盖率低时应回到任务二检查证据绑定算子；孤立节点高时应检查实体去重和关系抽取。",
        }
    )
    coverage = metrics.get("coverage") or {}
    gap_examples = coverage.get("gap_examples") or []
    complete_ratio = float(coverage.get("complete_ratio") or 0.0)
    insights.append(
        {
            "title": "知识缺口",
            "finding": f"疾病关系完整度为 {complete_ratio:.2%}。",
            "evidence": (
                "；".join(f"{item.get('disease')} 缺少 {','.join(item.get('missing') or [])}" for item in gap_examples[:5])
                if gap_examples
                else "暂无明显缺口。"
            ),
            "data_basis": "按疾病节点统计症状、检查、治疗三类关键关系覆盖。",
            "confidence": "0.84",
            "recommendation": "优先补充缺少检查和治疗关系的疾病文本，可提升图谱可分析性。",
        }
    )
    centrality = metrics.get("graph_centrality") or []
    if centrality:
        insights.append(
            {
                "title": "中心节点",
                "finding": f"当前图谱中心性最高的节点是 {centrality[0].get('name')}。",
                "evidence": f"该节点度数为 {centrality[0].get('degree')}，类型为 {centrality[0].get('type')}。",
                "data_basis": "基于 medical_triples 中 head/tail 出现次数计算近似 degree centrality。",
                "confidence": "0.83",
                "recommendation": "中心节点适合生成局部子图，用于解释当前图谱知识重心。",
            }
        )
    drug_assoc = metrics.get("drug_disease_association") or []
    if drug_assoc:
        insights.append(
            {
                "title": "药物关联",
                "finding": f"当前药物关联中，{drug_assoc[0].get('drug')} 覆盖疾病最多。",
                "evidence": f"关联疾病数 {drug_assoc[0].get('disease_count')}，关系数 {drug_assoc[0].get('relation_count')}。",
                "data_basis": "SQL: medical_triples WHERE tail_type='药物' AND relation IN ('药物治疗','治疗','用药')",
                "confidence": "0.81",
                "recommendation": "药物关联较少时，建议补充治疗方案或用药说明文本。",
            }
        )
    insights.append(
        {
            "title": "NL2SQL 可用性",
            "finding": f"内置分析问题集准确率为 {nl2sql_eval.get('accuracy')}，目标为 {nl2sql_eval.get('target_accuracy')}。",
            "evidence": f"正确 {nl2sql_eval.get('correct')}/{nl2sql_eval.get('total')}，通过={nl2sql_eval.get('passed')}。",
            "data_basis": "内置分析问题集覆盖分布、覆盖、中心性、证据路径和配置化分析场景。",
            "confidence": "0.87",
            "recommendation": "后续可扩展同义表达和更多疾病主题，提高复杂分析问题覆盖。",
        }
    )
    return insights


def write_insight_report(
    path: Path,
    insights: list[dict[str, str]],
    metrics: dict[str, Any],
    nl2sql_eval: dict[str, Any],
) -> None:
    lines = [
        "# MedInsight Task3 图谱洞察报告",
        "",
        "## 指标概览",
        "",
        f"- 实体节点：{metrics.get('node_count')}",
        f"- 图谱关系：{metrics.get('edge_count')}",
        f"- 疾病节点：{metrics.get('disease_count')}",
        f"- NL2SQL 准确率：{nl2sql_eval.get('accuracy')} ({nl2sql_eval.get('correct')}/{nl2sql_eval.get('total')})",
        "",
        "## 分析说明",
        "",
    ]
    for insight in insights:
        lines.extend(
            [
                f"### {insight['title']}",
                "",
                insight["finding"],
                "",
                f"依据：{insight['evidence']}",
                "",
                f"数据来源：{insight.get('data_basis', '未记录')}",
                "",
                f"置信度：{insight.get('confidence', '未计算')}",
                "",
                f"建议：{insight.get('recommendation', '暂无')}",
                "",
            ]
        )
    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8")


def insights_html(insights: list[dict[str, str]]) -> str:
    if not insights:
        return "<p>暂无分析说明。</p>"
    parts = []
    for insight in insights:
        parts.append(
            '<div class="insight">'
            f"<h3>{html.escape(insight.get('title') or '')}</h3>"
            f"<p>{html.escape(insight.get('finding') or '')}</p>"
            f"<small>依据：{html.escape(insight.get('evidence') or '')}</small>"
            f"<small>数据来源：{html.escape(insight.get('data_basis') or '')}</small>"
            f"<small>置信度：{html.escape(str(insight.get('confidence') or ''))}</small>"
            f"<small>建议：{html.escape(insight.get('recommendation') or '')}</small>"
            "</div>"
        )
    return "".join(parts)


def task1_reuse_html(profile: dict[str, Any]) -> str:
    if profile.get("status") != "ok":
        return f"<p>{html.escape(str(profile.get('message') or '未接入任务一报告'))}</p>"
    return (
        "<table><tbody>"
        f"<tr><th>任务一报告</th><td>{html.escape(str(profile.get('report_path')))}</td></tr>"
        f"<tr><th>输入记录</th><td>{html.escape(str(profile.get('input_records')))}</td></tr>"
        f"<tr><th>文档/片段</th><td>{html.escape(str(profile.get('documents')))} / {html.escape(str(profile.get('segments')))}</td></tr>"
        f"<tr><th>质量门控</th><td>{html.escape(str(profile.get('quality_gate_status')))}</td></tr>"
        f"<tr><th>DataMate</th><td>{html.escape(str(profile.get('datamate_connected')))}</td></tr>"
        "</tbody></table>"
    )


def agent_plan_html(plan: list[dict[str, str]]) -> str:
    rows = []
    for index, item in enumerate(plan, start=1):
        rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{html.escape(str(item.get('step') or ''))}</td>"
            f"<td>{html.escape(str(item.get('tool') or ''))}</td>"
            f"<td>{html.escape(str(item.get('output') or ''))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>#</th><th>阶段</th><th>工具/算子</th><th>输出</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def lineage_html(metrics: dict[str, Any]) -> str:
    lineage = metrics.get("data_to_knowledge_lineage") or {}
    contribution = metrics.get("document_knowledge_contribution") or []
    top_docs = "；".join(
        f"{item.get('source_doc')} {item.get('triple_count')} 条"
        for item in contribution[:5]
    )
    return (
        "<table><tbody>"
        f"<tr><th>任务一文档</th><td>{html.escape(str(lineage.get('task1_documents')))}</td></tr>"
        f"<tr><th>任务一片段</th><td>{html.escape(str(lineage.get('task1_segments')))}</td></tr>"
        f"<tr><th>任务二三元组</th><td>{html.escape(str(lineage.get('medical_triples')))}</td></tr>"
        f"<tr><th>每100片段三元组</th><td>{html.escape(str(lineage.get('triples_per_100_segments')))}</td></tr>"
        f"<tr><th>doc_id 匹配率</th><td>{float(lineage.get('task1_to_task2_doc_link_ratio') or 0):.2%}</td></tr>"
        f"<tr><th>贡献最高来源</th><td>{html.escape(top_docs or '暂无')}</td></tr>"
        f"<tr><th>说明</th><td>{html.escape(str(lineage.get('lineage_note') or ''))}</td></tr>"
        "</tbody></table>"
    )


def graph_quality_html(metrics: dict[str, Any]) -> str:
    quality = metrics.get("graph_quality") or {}
    duplicates = quality.get("duplicate_edge_groups") or []
    duplicate_label = f"{len(duplicates)} 组" if duplicates else "无"
    return (
        "<table><tbody>"
        f"<tr><th>质量摘要</th><td>{html.escape(str(quality.get('quality_summary')))}</td></tr>"
        f"<tr><th>证据覆盖率</th><td>{html.escape(str(quality.get('evidence_edge_ratio')))}</td></tr>"
        f"<tr><th>孤立节点比例</th><td>{html.escape(str(quality.get('isolated_node_ratio')))}</td></tr>"
        f"<tr><th>图谱密度</th><td>{html.escape(str(quality.get('graph_density')))}</td></tr>"
        f"<tr><th>来源文档数</th><td>{html.escape(str(quality.get('source_doc_count')))}</td></tr>"
        f"<tr><th>重复边</th><td>{html.escape(duplicate_label)}</td></tr>"
        "</tbody></table>"
    )


def evidence_path_html(metrics: dict[str, Any]) -> str:
    rows = []
    for item in (metrics.get("evidence_path_samples") or [])[:10]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('head')))}</td>"
            f"<td>{html.escape(str(item.get('relation')))}</td>"
            f"<td>{html.escape(str(item.get('tail')))}</td>"
            f"<td>{html.escape(str(item.get('source_doc')))}</td>"
            f"<td>{html.escape(str(item.get('evidence_text'))[:120])}</td>"
            "</tr>"
        )
    tbody = "".join(rows) or '<tr><td colspan="5">暂无证据路径</td></tr>'
    return (
        "<table><thead><tr><th>头实体</th><th>关系</th><th>尾实体</th><th>来源</th><th>证据文本</th></tr></thead>"
        f"<tbody>{tbody}</tbody></table>"
    )


def relation_matrix_rows(metrics: dict[str, Any]) -> str:
    rows = []
    for item in metrics.get("disease_relation_matrix", [])[:12]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('disease')))}</td>"
            f"<td>{int(item.get('symptom_edges') or 0)}</td>"
            f"<td>{int(item.get('test_edges') or 0)}</td>"
            f"<td>{int(item.get('treatment_edges') or 0)}</td>"
            f"<td>{int(item.get('cause_or_type_edges') or 0)}</td>"
            "</tr>"
        )
    return "".join(rows) or '<tr><td colspan="5">暂无疾病关系数据</td></tr>'


def source_trend_html(metrics: dict[str, Any]) -> str:
    trend = metrics.get("source_trend") or {}
    rows = trend.get("import_batches") or trend.get("source_docs") or []
    if not rows:
        return f"<p>{html.escape(str(trend.get('note') or '暂无来源或批次分布'))}</p>"
    key = "batch" if "batch" in rows[0] else "source_doc"
    return bar_items(rows, key, "edge_count", max_items=12) + f"<p>{html.escape(str(trend.get('note') or ''))}</p>"


def gap_rows(metrics: dict[str, Any]) -> str:
    rows = []
    for item in metrics["coverage"]["gap_examples"][:12]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item['disease']))}</td>"
            f"<td>{html.escape('、'.join(item['missing']))}</td>"
            f"<td>{int(item['total_edges'])}</td>"
            "</tr>"
        )
    return "".join(rows) or '<tr><td colspan="3">暂无明显缺口</td></tr>'
