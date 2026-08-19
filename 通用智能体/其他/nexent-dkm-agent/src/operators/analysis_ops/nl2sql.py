"""Small NL2SQL operator over an in-memory graph analytics schema."""

from __future__ import annotations

import re
import sqlite3
from typing import Any

_ALLOWED_TABLES = {"nodes", "edges"}
_DISEASE_QUERY_ALIASES = {
    "血糖偏高": "糖尿病",
    "血糖高": "糖尿病",
    "高血糖": "糖尿病",
}

# Canonical SQL for each recognized analysis intent. Every relation in the
# task-2 medical KG is disease-centric (subject is always a Disease), so the
# intent set is built around disease -> {symptom, drug, examination,
# treatment, complication} plus structural/aggregate queries.
INTENT_SQL: dict[str, str] = {
    "top_disease_symptoms": (
        "SELECT n.name AS disease, COUNT(*) AS symptom_count "
        "FROM edges e JOIN nodes n ON e.source = n.id "
        "WHERE e.predicate = 'has_symptom' AND n.type = 'Disease' "
        "GROUP BY n.name ORDER BY symptom_count DESC, disease ASC LIMIT 10"
    ),
    "top_disease_drugs": (
        "SELECT n.name AS disease, COUNT(*) AS drug_count "
        "FROM edges e JOIN nodes n ON e.source = n.id "
        "WHERE e.predicate = 'treated_by' AND n.type = 'Disease' "
        "GROUP BY n.name ORDER BY drug_count DESC, disease ASC LIMIT 10"
    ),
    "top_disease_exams": (
        "SELECT n.name AS disease, COUNT(*) AS examination_count "
        "FROM edges e JOIN nodes n ON e.source = n.id "
        "WHERE e.predicate = 'diagnosed_by' AND n.type = 'Disease' "
        "GROUP BY n.name ORDER BY examination_count DESC, disease ASC LIMIT 10"
    ),
    "top_disease_treatments": (
        "SELECT n.name AS disease, COUNT(*) AS treatment_count "
        "FROM edges e JOIN nodes n ON e.source = n.id "
        "WHERE e.predicate = 'recommended_treatment' AND n.type = 'Disease' "
        "GROUP BY n.name ORDER BY treatment_count DESC, disease ASC LIMIT 10"
    ),
    "disease_complications": (
        "SELECT s.name AS disease, t.name AS complication "
        "FROM edges e JOIN nodes s ON e.source = s.id JOIN nodes t ON e.target = t.id "
        "WHERE e.predicate = 'complication_of' "
        "ORDER BY disease ASC, complication ASC LIMIT 20"
    ),
    "relation_distribution": (
        "SELECT predicate, COUNT(*) AS edge_count FROM edges "
        "GROUP BY predicate ORDER BY edge_count DESC, predicate ASC"
    ),
    "list_diseases": (
        "SELECT name AS disease, mention_count FROM nodes "
        "WHERE type = 'Disease' ORDER BY mention_count DESC, disease ASC LIMIT 20"
    ),
    "top_mentioned_entities": (
        "SELECT name, type, mention_count FROM nodes "
        "ORDER BY mention_count DESC, name ASC LIMIT 10"
    ),
    "entity_distribution": (
        "SELECT type, COUNT(*) AS node_count FROM nodes "
        "GROUP BY type ORDER BY node_count DESC, type ASC"
    ),
    "graph_hub_nodes": (
        "SELECT n.name, n.type, "
        "(SELECT COUNT(*) FROM edges e WHERE e.source = n.id OR e.target = n.id) AS degree "
        "FROM nodes n ORDER BY degree DESC, n.name ASC LIMIT 10"
    ),
    "symptom_to_diseases": (
        "SELECT DISTINCT s.name AS disease FROM edges e "
        "JOIN nodes s ON e.source = s.id JOIN nodes t ON e.target = t.id "
        "WHERE e.predicate = 'has_symptom' AND s.type = 'Disease' AND t.type = 'Symptom' "
        "ORDER BY disease ASC LIMIT 20"
    ),
    "drug_to_diseases": (
        "SELECT DISTINCT s.name AS disease FROM edges e "
        "JOIN nodes s ON e.source = s.id JOIN nodes t ON e.target = t.id "
        "WHERE e.predicate = 'treated_by' AND s.type = 'Disease' AND t.type = 'Drug' "
        "ORDER BY disease ASC LIMIT 20"
    ),
    "treatment_to_diseases": (
        "SELECT DISTINCT s.name AS disease FROM edges e "
        "JOIN nodes s ON e.source = s.id JOIN nodes t ON e.target = t.id "
        "WHERE e.predicate = 'recommended_treatment' AND s.type = 'Disease' AND t.type = 'Treatment' "
        "ORDER BY disease ASC LIMIT 20"
    ),
    "unrecognized": (
        "SELECT 'unrecognized' AS intent, COUNT(*) AS total_nodes FROM nodes LIMIT 1"
    ),
}

# Intent keyword table, ordered by priority (earlier wins ties). Each question
# is scored by counting matched keywords per intent; the highest score wins,
# defaulting to ``entity_distribution`` when nothing matches.
_INTENT_KEYWORDS: list[tuple[str, list[str]]] = [
    ("disease_complications", [
        "并发症", "并发", "合并症", "complication", "comorbid",
    ]),
    ("top_disease_treatments", [
        "治疗方案", "推荐治疗", "治疗建议", "疗法", "治疗措施", "如何治疗",
        "怎么治", "处理办法", "干预措施", "干预手段", "饮食上的建议", "饮食建议",
        "recommended treatment", "treatment plan", "how to treat",
    ]),
    ("top_disease_exams", [
        "检查", "化验", "检验", "检测", "确诊", "诊断方式", "诊断方法", "怎么确诊", "如何确诊",
        "examination", "diagnos", "exam", "lab test",
    ]),
    ("top_disease_drugs", [
        "药", "用药", "药物", "药品", "服用", "吃什么药", "日常用药", "缓解什么病",
        "drug", "medication", "medicine", "prescri",
    ]),
    ("top_disease_symptoms", [
        "症状最多的", "症状最多", "症状", "symptom", "临床表现", "表现", "不适", "感觉", "sign", "symptom count",
    ]),
    ("relation_distribution", [
        "关系类型", "关系分布", "有哪些关系", "关系数量", "边的类型", "几种关系",
        "relation type", "relation distribution", "predicate", "每种关系", "关系类型分布",
    ]),
    ("top_mentioned_entities", [
        "提及最多", "出现次数最多", "提及次数", "高频实体", "频次最高", "出现最频繁",
        "最频繁", "最重要的实体", "被引用", "引用次数", "被提及", "提及更多",
        "most mentioned", "mention count", "frequent entit", "出现频率", "频率最高",
    ]),
    ("list_diseases", [
        "有哪些疾病", "所有疾病", "列出疾病", "列举疾病", "疾病列表", "疾病有哪些",
        "包含哪些疾病", "哪些病", "最常见", "list disease", "list all disease",
        "what disease", "which disease", "most common disease",
    ]),
    ("entity_distribution", [
        "实体分布", "类型分布", "各类型", "节点类型", "实体类型", "多少种类型",
        "实体数量", "类型统计", "节点统计", "分布", "类型", "统计",
        "各类节点", "各类实体", "节点的数量", "实体的数量",
        "entity type", "node type", "distribution", "category",
    ]),
    ("graph_hub_nodes", [
        "枢纽", "中心节点", "核心节点", "关键节点", "hub node", "most connected",
        "度最高", "连接最多", "网络中心",
    ]),
    ("symptom_to_diseases", [
        "具有", "会出现", "具有该症状", "symptom appears in", "diseases with symptom",
        "哪些病有", "哪些疾病有", "什么病会出现", "哪些疾病会出现",
        "症状的疾病", "症状的疾病有哪些", "有该症状",
    ]),
    ("drug_to_diseases", [
        "用药的疾病", "用该药的疾病", "用该药", "哪些病用此药", "什么病用此药",
        "哪些疾病用", "哪些病用", "吃这个药", "服用该药", "服用此药",
        "适用疾病", "适应症", "适应疾病",
        "diseases treated by", "which diseases use", "diseases using",
    ]),
    ("treatment_to_diseases", [
        "采用该治疗的疾病", "用此疗法的疾病", "哪些疾病采用", "哪些病采用",
        "推荐此方案的疾病", "适用该方案的疾病", "diseases with treatment",
        "diseases using treatment", "diseases treated with",
        "diseases using", "用此疗法",
    ]),
]


def classify_question_intent(question: str | None) -> str:
    """Classify a natural-language question into a known analysis intent.

    Uses keyword scoring across :data:`_INTENT_KEYWORDS`. Returns the
    ``entity_distribution`` fallback intent when generic keywords match.
    Returns ``unrecognized`` when no keyword matches at all.
    """

    normalized = (question or "").lower()
    if not normalized.strip():
        return "unrecognized"

    if "症状" in normalized and any(
        token in normalized for token in ("疾病", "哪些病", "什么病", "disease", "diseases")
    ):
        if any(
            token in normalized
            for token in ("具有", "出现", "会出现", "有该症状", "with", "having", "have")
        ):
            return "symptom_to_diseases"
    if "symptom" in normalized and any(
        token in normalized for token in ("disease", "diseases", "疾病", "哪些病", "什么病")
    ):
        if any(
            token in normalized
            for token in ("with", "having", "具有", "出现", "会出现", "有该症状")
        ):
            return "symptom_to_diseases"

    # Aggregate-ranking cues (种类最多/排名/多少种) force a top_disease_* intent
    # even when a reverse-lookup keyword like "用药" or "哪些疾病用" also matches.
    _AGGREGATE_CUES = ("种类最多", "种类最多", "多少种", "排名", "排行", "频次", "最多", "最频繁",
                       "count", "ranking", "most", "top")
    has_aggregate_cue = any(cue in normalized for cue in _AGGREGATE_CUES)
    if not has_aggregate_cue:
        # Reverse-lookup: "哪些疾病用X治疗" / "用该药的疾病" / "diseases using X"
        if any(kw in normalized for kw in ("用该药", "服用该药", "服用此药", "吃这个药",
                                           "用药的疾病", "用该药的疾病", "适应症", "适应疾病",
                                           "diseases treated by", "which diseases use")):
            return "drug_to_diseases"
        if any(kw in normalized for kw in ("采用该治疗", "用此疗法", "用此疗法的疾病",
                                           "哪些疾病采用", "哪些病采用", "推荐此方案",
                                           "适用该方案", "diseases with treatment",
                                           "diseases treated with")):
            return "treatment_to_diseases"
        if "diseases using" in normalized and "treatment" in normalized:
            return "treatment_to_diseases"

    best_intent = "unrecognized"
    best_score = 0
    skip_reverse_lookup = has_aggregate_cue
    for intent, keywords in _INTENT_KEYWORDS:
        if skip_reverse_lookup and intent in {
            "symptom_to_diseases", "drug_to_diseases", "treatment_to_diseases",
        }:
            continue
        # Weight longer phrases higher so relation-specific terms beat generic
        # tokens such as "统计" / "分布" / "类型" in compound questions.
        score = sum(len(kw) for kw in keywords if kw.lower() in normalized)
        if score > best_score:
            best_score = score
            best_intent = intent
    return best_intent
_MUTATING_SQL_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX)\b",
    re.IGNORECASE,
)
_TABLE_REF_RE = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][\w.]*)", re.IGNORECASE)
_QUOTED_TABLE_REF_RE = re.compile(r"\b(?:FROM|JOIN)\s*[\"`\[]", re.IGNORECASE)
_LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+)\b", re.IGNORECASE)
_COMMA_LIMIT_RE = re.compile(r"\bLIMIT\s+\d+\s*,", re.IGNORECASE)
_ALLOWED_SQL_FUNCTIONS = frozenset({
    "abs",
    "avg",
    "coalesce",
    "count",
    "ifnull",
    "length",
    "lower",
    "max",
    "min",
    "nullif",
    "round",
    "sum",
    "upper",
})


def build_graph_sqlite(graph: dict[str, Any]) -> sqlite3.Connection:
    """Build an in-memory SQLite database from graph nodes and edges."""

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE nodes (id TEXT PRIMARY KEY, name TEXT, type TEXT, mention_count INTEGER)"
    )
    conn.execute(
        "CREATE TABLE edges (source TEXT, target TEXT, predicate TEXT, confidence REAL)"
    )
    for node in graph.get("nodes", []):
        conn.execute(
            "INSERT INTO nodes VALUES (?, ?, ?, ?)",
            (
                node.get("id"),
                node.get("name"),
                node.get("type"),
                int(node.get("mention_count", 0)),
            ),
        )
    for edge in graph.get("edges", []):
        conn.execute(
            "INSERT INTO edges VALUES (?, ?, ?, ?)",
            (
                edge.get("source"),
                edge.get("target"),
                edge.get("predicate"),
                float(edge.get("confidence", 0.0)),
            ),
        )
    conn.commit()
    return conn


# Maps a disease-centric aggregate intent to the (predicate, result alias,
# entity-specific intent name) used when the question targets one disease.
_DISEASE_SPECIFIC_INTENTS: dict[str, tuple[str, str, str]] = {
    "top_disease_symptoms": ("has_symptom", "symptom", "disease_specific_symptoms"),
    "top_disease_drugs": ("treated_by", "drug", "disease_specific_drugs"),
    "top_disease_exams": ("diagnosed_by", "examination", "disease_specific_exams"),
    "top_disease_treatments": ("recommended_treatment", "treatment", "disease_specific_treatments"),
}

# Maps a reverse-lookup intent (target-centric) to (predicate, source type,
# target type, result alias, entity-specific intent name). When the question
# names a specific target entity (e.g. a Symptom), an entity-filtered query
# returns only diseases linked to that entity, instead of all diseases.
_REVERSE_LOOKUP_INTENTS: dict[str, tuple[str, str, str, str, str]] = {
    "symptom_to_diseases": ("has_symptom", "Disease", "Symptom", "disease", "symptom_specific_diseases"),
    "drug_to_diseases": ("treated_by", "Disease", "Drug", "disease", "drug_specific_diseases"),
    "treatment_to_diseases": ("recommended_treatment", "Disease", "Treatment", "disease", "treatment_specific_diseases"),
}


def disease_names_from_graph(graph: dict[str, Any]) -> list[str]:
    """Return the names of all Disease nodes in a graph dict."""

    return [
        node.get("name", "")
        for node in graph.get("nodes", [])
        if node.get("type") == "Disease" and node.get("name")
    ]


def disease_names_from_connection(conn: sqlite3.Connection) -> list[str]:
    """Return Disease node names from a built SQLite analytics database."""

    return _entity_names_from_connection(conn, "Disease")


def symptom_names_from_connection(conn: sqlite3.Connection) -> list[str]:
    """Return Symptom node names from a built SQLite analytics database."""

    return _entity_names_from_connection(conn, "Symptom")


def drug_names_from_connection(conn: sqlite3.Connection) -> list[str]:
    """Return Drug node names from a built SQLite analytics database."""

    return _entity_names_from_connection(conn, "Drug")


def treatment_names_from_connection(conn: sqlite3.Connection) -> list[str]:
    """Return Treatment node names from a built SQLite analytics database."""

    return _entity_names_from_connection(conn, "Treatment")


def _entity_names_from_connection(
    conn: sqlite3.Connection,
    entity_type: str,
) -> list[str]:
    """Return node names of ``entity_type`` from a built SQLite analytics database."""

    try:
        rows = conn.execute(
            "SELECT name FROM nodes WHERE type = ? AND name IS NOT NULL",
            (entity_type,),
        ).fetchall()
    except sqlite3.Error:
        return []
    return [row["name"] if isinstance(row, sqlite3.Row) else row[0] for row in rows]


def _entity_names_from_graph(graph: dict[str, Any], entity_type: str) -> list[str]:
    """Return the names of all nodes of ``entity_type`` in a graph dict."""

    return [
        node.get("name", "")
        for node in graph.get("nodes", [])
        if node.get("type") == entity_type and node.get("name")
    ]


def symptom_names_from_graph(graph: dict[str, Any]) -> list[str]:
    """Return the names of all Symptom nodes in a graph dict."""

    return _entity_names_from_graph(graph, "Symptom")


def drug_names_from_graph(graph: dict[str, Any]) -> list[str]:
    """Return the names of all Drug nodes in a graph dict."""

    return _entity_names_from_graph(graph, "Drug")


def treatment_names_from_graph(graph: dict[str, Any]) -> list[str]:
    """Return the names of all Treatment nodes in a graph dict."""

    return _entity_names_from_graph(graph, "Treatment")


def _detect_known_diseases(
    question: str | None,
    disease_names: list[str],
) -> list[str]:
    """Return known diseases named directly or through a small medical alias map."""

    text = question or ""
    positions: dict[str, int] = {}
    for name in disease_names:
        if name and name in text:
            positions[name] = text.find(name)
    for alias, canonical in _DISEASE_QUERY_ALIASES.items():
        if canonical in disease_names and alias in text:
            positions.setdefault(canonical, text.find(alias))
    return sorted(positions, key=lambda name: (positions[name], -len(name)))


def _detect_known_entities(
    question: str | None,
    entity_names: list[str],
) -> list[str]:
    """Return known entities named directly in the question.

    Used for reverse-lookup intents (symptom/drug/treatment -> diseases) where
    the question names a specific target entity and expects only the diseases
    linked to that entity, not all diseases in the graph.
    """

    text = question or ""
    positions: dict[str, int] = {}
    for name in entity_names:
        if name and name in text:
            positions[name] = text.find(name)
    return sorted(positions, key=lambda name: (positions[name], -len(name)))


def _build_reverse_lookup_sql(
    predicate: str,
    source_type: str,
    target_type: str,
    alias: str,
    target_entity: str,
) -> str:
    """Build a reverse-lookup SELECT filtered by a specific target entity.

    Returns the source (Disease) names linked to ``target_entity`` via
    ``predicate``. The entity name comes from a trusted graph node list; single
    quotes are escaped to keep the literal safe.
    """

    safe_entity = target_entity.replace("'", "''")
    return (
        f"SELECT DISTINCT s.name AS {alias} FROM edges e "
        "JOIN nodes s ON e.source = s.id JOIN nodes t ON e.target = t.id "
        f"WHERE e.predicate = '{predicate}' "
        f"AND s.type = '{source_type}' AND t.type = '{target_type}' "
        f"AND t.name = '{safe_entity}' "
        f"ORDER BY {alias} ASC LIMIT 20"
    )


def _build_disease_specific_sql(predicate: str, alias: str, disease: str) -> str:
    """Build a disease-filtered SELECT. The disease name comes from a trusted
    graph node list; single quotes are escaped to keep the literal safe."""

    safe_disease = disease.replace("'", "''")
    return (
        f"SELECT t.name AS {alias} FROM edges e "
        "JOIN nodes s ON e.source = s.id JOIN nodes t ON e.target = t.id "
        f"WHERE e.predicate = '{predicate}' AND s.name = '{safe_disease}' "
        f"ORDER BY {alias} ASC LIMIT 20"
    )


def _build_disease_mention_sql(diseases: list[str]) -> str:
    safe_names = ", ".join(
        f"'{disease.replace(chr(39), chr(39) * 2)}'"
        for disease in diseases
    )
    return (
        "SELECT name AS disease, mention_count FROM nodes "
        f"WHERE type = 'Disease' AND name IN ({safe_names}) "
        "ORDER BY mention_count DESC, disease ASC LIMIT 20"
    )


def translate_question_to_sql(
    question: str | None,
    disease_names: list[str] | None = None,
    *,
    symptom_names: list[str] | None = None,
    drug_names: list[str] | None = None,
    treatment_names: list[str] | None = None,
) -> dict[str, Any]:
    """Translate a graph-analysis question into safe, intent-mapped SQL.

    The question is classified into one of the intents in :data:`INTENT_SQL`
    via keyword scoring. Two entity-aware refinements are applied when the
    relevant node-name lists are supplied:

    * **Disease-centric**: for ``top_disease_*`` intents, if the question names
      a specific disease, an entity-filtered query is generated instead of the
      global aggregate template (so "高血压用什么药" returns 高血压's drugs,
      not per-disease counts).
    * **Reverse-lookup**: for ``symptom_to_diseases`` / ``drug_to_diseases`` /
      ``treatment_to_diseases``, if the question names a specific target
      entity, only the diseases linked to that entity are returned (so
      "哪些疾病会出现头晕症状" returns the diseases that have 头晕, not all
      diseases in the graph).
    """

    intent = classify_question_intent(question)
    if disease_names:
        diseases = _detect_known_diseases(question, disease_names)
        if intent == "top_mentioned_entities" and diseases:
            return {
                "status": "completed",
                "intent": "disease_mention_comparison",
                "sql": _build_disease_mention_sql(diseases),
                "entities": diseases,
            }
        spec = _DISEASE_SPECIFIC_INTENTS.get(intent)
        if diseases and spec:
            disease = diseases[0]
            predicate, alias, specific_intent = spec
            return {
                "status": "completed",
                "intent": specific_intent,
                "sql": _build_disease_specific_sql(predicate, alias, disease),
                "entity": disease,
            }

    reverse_spec = _REVERSE_LOOKUP_INTENTS.get(intent)
    if reverse_spec:
        predicate, source_type, target_type, alias, specific_intent = reverse_spec
        target_names_map = {
            "Symptom": symptom_names,
            "Drug": drug_names,
            "Treatment": treatment_names,
        }
        candidate_names = target_names_map.get(target_type)
        if candidate_names:
            entities = _detect_known_entities(question, candidate_names)
            if entities:
                target_entity = entities[0]
                return {
                    "status": "completed",
                    "intent": specific_intent,
                    "sql": _build_reverse_lookup_sql(
                        predicate, source_type, target_type, alias, target_entity
                    ),
                    "entity": target_entity,
                }

    return {
        "status": "completed",
        "intent": intent,
        "sql": INTENT_SQL[intent],
    }


def evaluate_nl2sql_accuracy(
    benchmark: list[dict[str, Any]],
    translator=classify_question_intent,
) -> dict[str, Any]:
    """Measure intent-classification accuracy of an NL2SQL translator.

    ``benchmark`` is a list of ``{"question": str, "expected_intent": str}``
    records. Each canonical intent maps to exactly one SQL template, so intent
    accuracy is equivalent to SQL-correctness on this fixed schema.

    Returns a report with overall accuracy, per-intent breakdown, and the list
    of misclassified cases for transparency.
    """

    total = len(benchmark)
    correct = 0
    per_intent: dict[str, dict[str, int]] = {}
    mistakes: list[dict[str, str]] = []

    for case in benchmark:
        question = case.get("question", "")
        expected = case.get("expected_intent", "")
        predicted = translator(question)
        bucket = per_intent.setdefault(expected, {"total": 0, "correct": 0})
        bucket["total"] += 1
        if predicted == expected:
            correct += 1
            bucket["correct"] += 1
        else:
            mistakes.append({
                "question": question,
                "expected": expected,
                "predicted": predicted,
            })

    accuracy = round(correct / total, 4) if total else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "per_intent": per_intent,
        "mistakes": mistakes,
    }


def _rows_signature(rows: list[dict[str, Any]] | None) -> Any:
    """Order-independent signature of a result set for comparison."""

    if rows is None:
        return None
    return sorted(tuple(sorted(row.items())) for row in rows)


def evaluate_nl2sql_execution_accuracy(
    benchmark: list[dict[str, Any]],
    graph: dict[str, Any],
    max_limit: int = 20,
    translator: Any = None,
) -> dict[str, Any]:
    """Measure end-to-end NL2SQL correctness by executing generated SQL.

    For each case ``{"question": str, "gold_sql": str}`` a translator generates
    SQL, which is executed against an in-memory SQLite built from ``graph``. The
    result rows are compared (order-independent) against the gold SQL's rows.
    This is stricter than intent-classification accuracy: it verifies the
    generated query returns the same data as a hand-written reference query.

    ``translator`` lets the caller benchmark a specific NL2SQL path. It is a
    callable ``translator(question, conn, disease_names) -> {"sql": str,
    "translator"?: str}``. When ``None`` the entity-aware template translator is
    used (label ``"template"``). The returned report includes a
    ``per_translator`` breakdown keyed on the path that produced each answer
    (e.g. ``local_model`` / ``llm`` / ``template``), so the LLM and
    local-model paths can be measured separately from the rule template.
    """

    conn = build_graph_sqlite(graph)
    disease_names = disease_names_from_graph(graph)
    symptom_names = symptom_names_from_graph(graph)
    drug_names = drug_names_from_graph(graph)
    treatment_names = treatment_names_from_graph(graph)

    total = len(benchmark)
    correct = 0
    mistakes: list[dict[str, Any]] = []
    per_translator: dict[str, dict[str, int]] = {}

    for case in benchmark:
        question = case.get("question", "")
        gold_sql = case.get("gold_sql", "")
        if translator is None:
            translation = translate_question_to_sql(
                question,
                disease_names=disease_names,
                symptom_names=symptom_names,
                drug_names=drug_names,
                treatment_names=treatment_names,
            )
            translator_label = "template"
        else:
            translation = translator(question, conn, disease_names)
            translator_label = translation.get("translator", "custom")

        try:
            predicted_rows = execute_read_only_sql(conn, translation["sql"], max_limit=max_limit)["rows"]
        except (sqlite3.Error, ValueError):
            predicted_rows = None
        try:
            gold_rows = execute_read_only_sql(conn, gold_sql, max_limit=max_limit)["rows"]
        except (sqlite3.Error, ValueError) as exc:
            raise ValueError(f"Invalid gold_sql for question {question!r}: {exc}") from exc

        bucket = per_translator.setdefault(translator_label, {"total": 0, "correct": 0})
        bucket["total"] += 1
        if _rows_signature(predicted_rows) == _rows_signature(gold_rows):
            correct += 1
            bucket["correct"] += 1
        else:
            mistakes.append({
                "question": question,
                "intent": translation.get("intent"),
                "translator": translator_label,
                "predicted_sql": translation["sql"],
                "gold_sql": gold_sql,
            })

    for bucket in per_translator.values():
        bucket["accuracy"] = round(bucket["correct"] / bucket["total"], 4) if bucket["total"] else 0.0

    accuracy = round(correct / total, 4) if total else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "metric": "execution_row_match",
        "per_translator": per_translator,
        "mistakes": mistakes,
    }


def execute_sql(conn: sqlite3.Connection, sql: str) -> list[dict[str, Any]]:
    """Execute generated read-only SQL against the graph database."""

    return execute_read_only_sql(conn, sql)["rows"]


def execute_read_only_sql(
    conn: sqlite3.Connection,
    sql: str,
    max_limit: int = 20,
) -> dict[str, Any]:
    """Validate and execute a read-only graph SQL query."""

    safe_sql = validate_read_only_sql(sql, max_limit=max_limit)
    conn.set_authorizer(_graph_query_authorizer)
    try:
        rows = [
            dict(row)
            for row in conn.execute(safe_sql).fetchmany(max_limit)
        ]
    finally:
        conn.set_authorizer(None)
    return {"sql": safe_sql, "rows": rows}


def validate_read_only_sql(sql: str, max_limit: int = 20) -> str:
    """Return a bounded SELECT query or raise ValueError for unsafe SQL."""

    statement = " ".join((sql or "").strip().split())
    if not statement:
        raise ValueError("SQL is empty.")
    if ";" in statement:
        raise ValueError("Multiple SQL statements are not allowed.")
    if "--" in statement or "/*" in statement or "*/" in statement:
        raise ValueError("SQL comments are not allowed.")
    if not statement.lower().startswith("select "):
        raise ValueError("Only SELECT statements are allowed for task-3 NL2SQL.")
    if _MUTATING_SQL_RE.search(statement):
        raise ValueError("Mutating or schema-changing SQL is not allowed.")
    if _QUOTED_TABLE_REF_RE.search(statement):
        raise ValueError("Quoted table identifiers are not allowed.")

    table_refs = {_normalize_table(match.group(1)) for match in _TABLE_REF_RE.finditer(statement)}
    if not table_refs:
        raise ValueError("SQL must read from the task-3 graph tables.")
    invalid_tables = table_refs - _ALLOWED_TABLES
    if invalid_tables:
        raise ValueError(f"Only task-3 graph tables are allowed: {sorted(invalid_tables)}")
    if _COMMA_LIMIT_RE.search(statement):
        raise ValueError("Comma-style LIMIT clauses are not allowed.")

    limit_match = _LIMIT_RE.search(statement)
    if not limit_match:
        return f"{statement} LIMIT {max_limit}"
    if int(limit_match.group(1)) > max_limit:
        return _LIMIT_RE.sub(f"LIMIT {max_limit}", statement, count=1)
    return statement


def _normalize_table(name: str) -> str:
    return name.rsplit(".", 1)[-1].strip('"`[]').lower()


def _graph_query_authorizer(
    action: int,
    arg1: str | None,
    arg2: str | None,
    database_name: str | None,
    source: str | None,
) -> int:
    """Restrict SQLite reads to graph tables and a small function whitelist."""

    del database_name, source
    if action == sqlite3.SQLITE_READ:
        table = (arg1 or "").lower()
        return sqlite3.SQLITE_OK if table in _ALLOWED_TABLES else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        function_name = (arg2 or "").lower()
        return (
            sqlite3.SQLITE_OK
            if function_name in _ALLOWED_SQL_FUNCTIONS
            else sqlite3.SQLITE_DENY
        )
    return sqlite3.SQLITE_OK
