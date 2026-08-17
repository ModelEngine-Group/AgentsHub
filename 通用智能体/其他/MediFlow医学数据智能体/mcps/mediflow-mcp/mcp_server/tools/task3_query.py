"""
任务三知识图谱和疾病分析查询 MCP 工具。
"""

from mcp_server.tools import mcp
from mcp_server.kg.persistence import ensure_kg_schema
from mcp_server.shared.frontend_status import get_validation_frontend_status_payload
from mcp_server.shared.sqlite_utils import connect_analytics, connect_kg, row_dicts
from mcp_server.tools.task3_runtime import get_task3_analysis_service


@mcp.tool
def get_validation_frontend_status() -> str:
    """返回可视化平台入口 URL。仅返回一个纯字符串 URL，禁止使用其他地址。"""
    payload = get_validation_frontend_status_payload()
    return payload.get("医学数据智能体可视化平台", "")


@mcp.tool
def query_knowledge_graph(subject: str) -> list:
    """查询与指定疾病、症状、药物或医学实体相关的知识图谱三元组。"""
    ensure_kg_schema()
    conn = connect_kg()
    c = conn.cursor()
    pattern = f"%{subject}%"
    c.execute(
        """SELECT e1.canonical_name, r.display_name, e2.canonical_name, t.confidence,
                  src.source_name, t.source_file, t.source_record_id, t.evidence,
                  t.reliability_level
           FROM kg_triples t
           JOIN kg_entities e1 ON t.subject_id = e1.entity_id
           JOIN kg_entities e2 ON t.object_id = e2.entity_id
           JOIN kg_relations r ON t.relation_code = r.relation_code
           LEFT JOIN kg_sources src ON t.source_id = src.source_id
           WHERE e1.canonical_name LIKE ? OR e2.canonical_name LIKE ?
           LIMIT 60""",
        (pattern, pattern),
    )
    results = [
        {
            "subject": r[0],
            "predicate": r[1],
            "object": r[2],
            "confidence": r[3],
            "source_dataset": r[4],
            "source_file": r[5],
            "source_record_id": r[6],
            "evidence": r[7],
            "reliability_level": r[8],
        }
        for r in c.fetchall()
    ]
    conn.close()
    return results


@mcp.tool
def get_medical_data_sources(limit: int = 20) -> dict:
    """列出任务二流水线已登记的医学知识图谱数据来源。"""
    conn = connect_kg()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM kg_sources")
    total_source_count = int(c.fetchone()[0] or 0)
    c.execute(
        """SELECT s.source_id,
                  s.source_name AS dataset_name,
                  s.source_name,
                  s.source_path,
                  s.source_type,
                  s.record_count,
                  COALESCE(t.triple_count, 0) AS triple_count,
                  s.created_at
           FROM kg_sources s
           LEFT JOIN (
               SELECT source_id, COUNT(*) AS triple_count
               FROM kg_triples
               GROUP BY source_id
           ) t ON t.source_id = s.source_id
           ORDER BY s.source_id DESC
           LIMIT ?""",
        (limit,),
    )
    sources = row_dicts(c)
    conn.close()
    return {
        "total_source_count": total_source_count,
        "returned_source_count": len(sources),
        "source_count": total_source_count,
        "limit": limit,
        "scope": "recent_kg_sources",
        "sources": sources,
        "note": "sources 为最近登记来源列表；source_count/total_source_count 为当前 KG 已登记来源总数。",
    }


def _disease_aliases(name: str) -> list[str]:
    return [
        name,
        name.replace("2型", "II型").replace("1型", "I型"),
        name.replace("II型", "2型").replace("I型", "1型"),
    ]


def _query_disease_aspect(disease: str, table: str, value_col: str, label: str) -> list[dict]:
    conn = connect_analytics()
    c = conn.cursor()
    table_exists = c.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    if not table_exists:
        conn.close()
        return []

    results = []
    for alias in _disease_aliases(disease):
        c.execute(
            f"SELECT disease, {value_col} FROM {table} WHERE disease LIKE ? LIMIT 30",
            (f"%{alias}%",),
        )
        for r in c.fetchall():
            results.append({"disease": r[0], label: r[1]})
        if results:
            break
    conn.close()
    return results


@mcp.tool
def query_disease_analytics(disease: str, aspect: str = "all") -> dict:
    """查询疾病症状、药物、并发症和科室等结构化分析数据。"""
    aspects_config = {
        "symptoms": ("disease_symptoms", "symptom", "symptom"),
        "drugs": ("disease_drugs", "drug", "drug"),
        "complications": ("disease_complications", "complication", "complication"),
        "departments": ("disease_departments", "department", "department"),
        "tests": ("disease_tests", "test", "test"),
        "procedures": ("disease_procedures", "procedure", "procedure"),
        "populations": ("disease_populations", "population", "population"),
    }
    result = {"disease": disease}
    for asp_key, (table, col, label) in aspects_config.items():
        if aspect in ("all", asp_key):
            items = _query_disease_aspect(disease, table, col, label)
            if items:
                result[asp_key] = items
    return result


@mcp.tool
def ask_medical_analytics(question: str) -> dict:
    """执行统一任务三分析链并返回同源证据、图表和追溯信息。"""
    return get_task3_analysis_service().analyze(question)
