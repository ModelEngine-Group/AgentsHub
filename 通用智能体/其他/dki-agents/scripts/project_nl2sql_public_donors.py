"""Project authoritative public NL2SQL donors onto the sealed four-table DB.

The public donor index contains only immutable row/hash/AST pointers from
selection-used CSS and EHRSQL train/validation splits.  This projector never
loads the source questions or SQL and never imports the project's NL2SQL
engine.  Instead, a deterministic typed-IR library transfers the independently
published construct mix onto the sealed four-table medical schema.

Question and SQL are rendered from the same typed IR.  Every result is then
checked by the engine-independent blind protocol, executed through a read-only
non-truncating SQLite connection, and required to return a non-empty result.
Detailed rows remain private; the public attestation is aggregate and
hash-bound only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.analysis.nl2sql_protocol import (
    execute_complete_readonly,
    normalise_question,
    validate_gold_sql,
    validate_question_sql_grounding,
)
from dki.analysis.nl2sql_sqlite import connect_readonly


CATEGORY_QUOTAS = {
    "filtering": 30,
    "aggregation_grouping": 45,
    "top_k": 24,
    "multi_table_join": 45,
    "temporal": 30,
    "knowledge_graph": 36,
    "having": 15,
    "nested_correlated": 30,
    "cte": 20,
    "set_operation": 10,
    "window": 15,
}
DIFFICULTY_BY_CATEGORY = {
    "filtering": {"medium": 25, "hard": 5},
    "aggregation_grouping": {"medium": 20, "hard": 25},
    "top_k": {"medium": 10, "hard": 14},
    "multi_table_join": {"medium": 15, "hard": 30},
    "temporal": {"medium": 10, "hard": 20},
    "knowledge_graph": {"medium": 10, "hard": 20, "extra": 6},
    "having": {"hard": 10, "extra": 5},
    "nested_correlated": {
        "hard": 15,
        "extra": 10,
        "extra-hard": 5,
    },
    "cte": {"hard": 10, "extra": 5, "extra-hard": 5},
    "set_operation": {"hard": 1, "extra": 2, "extra-hard": 7},
    "window": {"extra": 2, "extra-hard": 13},
}
EXPECTED_DIFFICULTY_QUOTAS = {
    "medium": 90,
    "hard": 150,
    "extra": 30,
    "extra-hard": 30,
}
DONOR_CONSTRUCT = {
    "filtering": "filtering",
    "aggregation_grouping": "aggregation_grouping",
    "top_k": "top_k",
    "multi_table_join": "multi_table_join",
    "temporal": "temporal",
    # Until the optional MIMIC-SPARQL source is frozen, KG intents are a
    # declared domain transfer from independently published join/filter forms.
    "knowledge_graph": "multi_table_join",
    "having": "having",
    "nested_correlated": "nested_subquery",
    # CSS/EHRSQL contain no separately labelled CTE donor in the pinned
    # revisions.  A CTE is therefore a semantics-preserving lift of a nested
    # subquery donor and is disclosed as such in every private row.
    "cte": "nested_subquery",
    "set_operation": "set_operation",
    "window": "window",
}
TRANSFORM_KIND = {
    category: (
        "nested_subquery_to_cte_semantics_preserving_lift"
        if category == "cte"
        else (
            "relational_join_to_medical_kg_domain_transfer"
            if category == "knowledge_graph"
            else "same_construct_four_table_typed_ir_projection"
        )
    )
    for category in CATEGORY_QUOTAS
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return (
        "\n".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for row in rows
        )
        + "\n"
    ).encode("utf-8")


def _write_synced(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_file_once(path: Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"projection public output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sql_string(value: str) -> str:
    return value.replace("'", "''")


def _num(value: float | int) -> str:
    return f"{float(value):g}" if isinstance(value, float) else str(value)


def _pick(values: list[Any], index: int, attempt: int, stride: int = 7) -> Any:
    if not values:
        raise ValueError("sealed database lacks a required domain")
    return values[(index * stride + attempt) % len(values)]


@dataclass(frozen=True)
class Domains:
    departments: list[str]
    diseases: list[str]
    regions: list[str]
    drugs: list[str]
    exams: list[str]
    genders: list[str]
    entity_types: list[str]
    head_types: list[str]
    relations: list[str]
    kg_heads: list[str]
    dates: list[str]


@dataclass(frozen=True)
class Rendered:
    family: str
    question: str
    sql: str
    parameters: dict[str, Any]


def _values(conn: sqlite3.Connection, sql: str) -> list[str]:
    return sorted(
        {
            str(row[0])
            for row in conn.execute(sql)
            if row and row[0] is not None and str(row[0]) != ""
        }
    )


def _load_domains(db_path: Path) -> Domains:
    conn = connect_readonly(db_path)
    try:
        dates = _values(
            conn,
            "SELECT DISTINCT visit_date FROM visits ORDER BY visit_date",
        )
        return Domains(
            departments=_values(conn, "SELECT DISTINCT department FROM visits"),
            diseases=_values(conn, "SELECT DISTINCT disease FROM visits"),
            regions=_values(conn, "SELECT DISTINCT region FROM patients"),
            drugs=_values(
                conn,
                "SELECT DISTINCT drug_prescribed FROM visits "
                "WHERE length(drug_prescribed)>0",
            ),
            exams=_values(
                conn,
                "SELECT DISTINCT exam FROM visits WHERE length(exam)>0",
            ),
            genders=_values(conn, "SELECT DISTINCT gender FROM patients"),
            entity_types=_values(conn, "SELECT DISTINCT type FROM kg_entities"),
            head_types=_values(conn, "SELECT DISTINCT head_type FROM kg_triples"),
            relations=_values(conn, "SELECT DISTINCT relation_zh FROM kg_triples"),
            kg_heads=_values(conn, "SELECT DISTINCT head FROM kg_triples"),
            dates=dates,
        )
    finally:
        conn.close()


def _date_pair(domains: Domains, index: int, attempt: int) -> tuple[str, str]:
    if len(domains.dates) < 2:
        raise ValueError("sealed visits table lacks two distinct dates")
    left_position = (index * 11 + attempt) % max(1, len(domains.dates) - 1)
    width = 20 + ((index + attempt) % 120)
    right_position = min(len(domains.dates) - 1, left_position + width)
    if right_position == left_position:
        right_position = min(len(domains.dates) - 1, left_position + 1)
    return domains.dates[left_position], domains.dates[right_position]


def _filtering(index: int, attempt: int, d: Domains) -> Rendered:
    family = index % 5
    if family == 0:
        region = _pick(d.regions, index, attempt)
        gender = _pick(d.genders, index, attempt, 3)
        age = [25, 35, 45, 55, 65][(index + attempt) % 5]
        return Rendered(
            "patient_region_gender_age",
            f"查询地区为{region}、性别为{gender}且年龄至少{age}岁的患者，"
            "返回患者编号、姓名和年龄。",
            "SELECT patient_id, name, age FROM patients "
            f"WHERE region='{_sql_string(region)}' "
            f"AND gender='{_sql_string(gender)}' AND age>={age} "
            "ORDER BY age DESC, patient_id",
            {"region": region, "gender": gender, "age": age},
        )
    if family == 1:
        department = _pick(d.departments, index, attempt)
        cost = [1000, 3000, 5000, 8000, 12000][(index + attempt) % 5]
        return Rendered(
            "visit_department_cost",
            f"查询科室为{department}且费用至少{cost}元的就诊，"
            "返回就诊编号、患者编号、诊断和费用。",
            "SELECT visit_id, patient_id, disease, cost FROM visits "
            f"WHERE department='{_sql_string(department)}' AND cost>={cost} "
            "ORDER BY cost DESC, visit_id",
            {"department": department, "cost": cost},
        )
    if family == 2:
        disease = _pick(d.diseases, index, attempt)
        los = [1, 2, 3, 5, 7][(index + attempt) % 5]
        return Rendered(
            "visit_disease_los",
            f"查询诊断为{disease}且住院天数至少{los}天的就诊，"
            "返回就诊编号、科室和住院天数。",
            "SELECT visit_id, department, los_days FROM visits "
            f"WHERE disease='{_sql_string(disease)}' AND los_days>={los} "
            "ORDER BY los_days DESC, visit_id",
            {"disease": disease, "los_days": los},
        )
    if family == 3:
        entity_type = _pick(d.entity_types, index, attempt)
        mentions = [2, 5, 10, 20, 30][(index + attempt) % 5]
        return Rendered(
            "kg_entity_type_mentions",
            f"查询知识图谱中类型为{entity_type}且提及次数至少{mentions}次的实体，"
            "返回实体名称和提及次数。",
            "SELECT name, mention_count FROM kg_entities "
            f"WHERE type='{_sql_string(entity_type)}' "
            f"AND mention_count>={mentions} "
            "ORDER BY mention_count DESC, name",
            {"entity_type": entity_type, "mentions": mentions},
        )
    head_type = _pick(d.head_types, index, attempt)
    relation = _pick(d.relations, index, attempt)
    confidence = [0.5, 0.6, 0.7, 0.8][(index + attempt) % 4]
    return Rendered(
        "kg_triple_type_relation_confidence",
        f"查询头实体类型为{head_type}、中文关系为{relation}且置信度至少"
        f"{_num(confidence)}的知识图谱三元组，返回头实体、关系和尾实体。",
        "SELECT head, relation_zh, tail FROM kg_triples "
        f"WHERE head_type='{_sql_string(head_type)}' "
        f"AND relation_zh='{_sql_string(relation)}' "
        f"AND confidence>={_num(confidence)} "
        "ORDER BY head, tail",
        {
            "head_type": head_type,
            "relation": relation,
            "confidence": confidence,
        },
    )


def _aggregation(index: int, attempt: int, d: Domains) -> Rendered:
    family = index % 5
    cost = [1000, 3000, 5000, 8000][(index + attempt) % 4]
    if family == 0:
        los = [0, 1, 3, 5][(index + attempt // 4) % 4]
        return Rendered(
            "department_visit_count",
            f"对费用至少{cost}元且住院天数至少{los}天的就诊按科室分组，"
            "返回各科室的就诊次数。",
            "SELECT department, COUNT(*) AS visit_count FROM visits "
            f"WHERE cost>={cost} AND los_days>={los} GROUP BY department "
            "ORDER BY visit_count DESC, department",
            {"cost": cost, "los_days": los},
        )
    if family == 1:
        los = [0, 1, 3, 5][(index + attempt) % 4]
        minimum_cost = [500, 1500, 3500, 6500][
            (index + attempt // 4) % 4
        ]
        return Rendered(
            "disease_average_cost",
            f"对住院天数至少{los}天且费用至少{minimum_cost}元的就诊按诊断"
            "分组，返回各诊断的就诊次数和平均费用。",
            "SELECT disease, COUNT(*) AS visit_count, AVG(cost) AS avg_cost "
            f"FROM visits WHERE los_days>={los} AND cost>={minimum_cost} "
            "GROUP BY disease "
            "ORDER BY visit_count DESC, disease",
            {"los_days": los, "cost": minimum_cost},
        )
    if family == 2:
        age = [25, 35, 45, 55, 65][(index + attempt) % 5]
        gender = _pick(d.genders, index, attempt, 3)
        return Rendered(
            "region_patient_visit_aggregation",
            f"统计年龄至少{age}岁且性别为{gender}的患者，按地区返回不同"
            "患者数和就诊总费用。",
            "SELECT p.region, COUNT(DISTINCT p.patient_id) AS patient_count, "
            "SUM(v.cost) AS total_cost FROM patients p "
            "JOIN visits v ON p.patient_id=v.patient_id "
            f"WHERE p.age>={age} AND p.gender='{_sql_string(gender)}' "
            "GROUP BY p.region "
            "ORDER BY patient_count DESC, p.region",
            {"age": age, "gender": gender},
        )
    if family == 3:
        start, end = _date_pair(d, index, attempt)
        return Rendered(
            "monthly_visit_aggregation",
            f"统计{start}至{end}之间的就诊，按月份返回就诊次数和平均费用。",
            "SELECT strftime('%Y-%m', visit_date) AS month, "
            "COUNT(*) AS visit_count, AVG(cost) AS avg_cost FROM visits "
            f"WHERE visit_date BETWEEN '{start}' AND '{end}' GROUP BY month "
            "ORDER BY month",
            {"start_date": start, "end_date": end},
        )
    department = _pick(d.departments, index, attempt)
    return Rendered(
        "drug_cost_aggregation",
        f"对科室为{department}且费用至少{cost}元的就诊按处方药物分组，"
        "返回处方次数和总费用。",
        "SELECT drug_prescribed, COUNT(*) AS prescription_count, "
        "SUM(cost) AS total_cost FROM visits "
        f"WHERE department='{_sql_string(department)}' AND cost>={cost} "
        "GROUP BY drug_prescribed "
        "ORDER BY prescription_count DESC, drug_prescribed",
        {"department": department, "cost": cost},
    )


def _top_k(index: int, attempt: int, d: Domains) -> Rendered:
    family = index % 4
    limit = [2, 3, 4, 5][(index + attempt) % 4]
    if family == 0:
        start, end = _date_pair(d, index, attempt)
        return Rendered(
            "top_departments_by_visits",
            f"在{start}至{end}之间，返回就诊次数最多的前{limit}个科室"
            "及其就诊次数。",
            "SELECT department, COUNT(*) AS visit_count FROM visits "
            f"WHERE visit_date BETWEEN '{start}' AND '{end}' "
            "GROUP BY department ORDER BY visit_count DESC, department "
            f"LIMIT {limit}",
            {"start_date": start, "end_date": end, "limit": limit},
        )
    if family == 1:
        cost = [1000, 3000, 5000, 8000][
            (index + attempt // 4) % 4
        ]
        return Rendered(
            "top_diseases_by_average_cost",
            f"对费用至少{cost}元的就诊，返回平均费用最高的前{limit}种诊断"
            "及其平均费用。",
            "SELECT disease, AVG(cost) AS avg_cost FROM visits "
            f"WHERE cost>={cost} GROUP BY disease "
            "ORDER BY avg_cost DESC, disease "
            f"LIMIT {limit}",
            {"cost": cost, "limit": limit},
        )
    if family == 2:
        department = _pick(d.departments, index, attempt)
        return Rendered(
            "top_patients_by_department_cost",
            f"在科室{department}中，返回累计费用最高的前{limit}位患者"
            "及其累计费用。",
            "SELECT patient_id, SUM(cost) AS total_cost FROM visits "
            f"WHERE department='{_sql_string(department)}' GROUP BY patient_id "
            "ORDER BY total_cost DESC, patient_id "
            f"LIMIT {limit}",
            {"department": department, "limit": limit},
        )
    relation = _pick(d.relations, index, attempt)
    confidence = [0.5, 0.6, 0.7][(index + attempt) % 3]
    return Rendered(
        "top_kg_heads_by_observations",
        f"对中文关系为{relation}且置信度至少{_num(confidence)}的三元组，"
        f"返回累计观察次数最高的前{limit}个头实体。",
        "SELECT head, SUM(count) AS observed_count FROM kg_triples "
        f"WHERE relation_zh='{_sql_string(relation)}' "
        f"AND confidence>={_num(confidence)} GROUP BY head "
        "ORDER BY observed_count DESC, head "
        f"LIMIT {limit}",
        {"relation": relation, "confidence": confidence, "limit": limit},
    )


def _join(index: int, attempt: int, d: Domains) -> Rendered:
    family = index % 5
    if family == 0:
        region = _pick(d.regions, index, attempt)
        department = _pick(d.departments, index, attempt)
        cost = [1000, 3000, 5000, 8000][(index + attempt) % 4]
        return Rendered(
            "patient_visit_projection",
            f"查询地区为{region}、就诊科室为{department}且费用至少{cost}元的"
            "患者姓名、就诊日期和诊断。",
            "SELECT p.name, v.visit_date, v.disease FROM patients p "
            "JOIN visits v ON p.patient_id=v.patient_id "
            f"WHERE p.region='{_sql_string(region)}' "
            f"AND v.department='{_sql_string(department)}' "
            f"AND v.cost>={cost} ORDER BY v.visit_date, p.patient_id",
            {"region": region, "department": department, "cost": cost},
        )
    if family == 1:
        disease = _pick(d.diseases, index, attempt)
        cost = [1000, 3000, 5000][(index + attempt) % 3]
        return Rendered(
            "patient_visit_gender_group",
            f"对诊断为{disease}且费用至少{cost}元的就诊，按患者性别返回"
            "不同患者数和平均费用。",
            "SELECT p.gender, COUNT(DISTINCT p.patient_id) AS patient_count, "
            "AVG(v.cost) AS avg_cost FROM patients p "
            "JOIN visits v ON p.patient_id=v.patient_id "
            f"WHERE v.disease='{_sql_string(disease)}' AND v.cost>={cost} "
            "GROUP BY p.gender ORDER BY p.gender",
            {"disease": disease, "cost": cost},
        )
    if family == 2:
        exam = _pick(d.exams, index, attempt)
        age = [30, 40, 50, 60][(index + attempt) % 4]
        return Rendered(
            "patient_visit_exam_age",
            f"查询年龄至少{age}岁且检查为{exam}的患者姓名、年龄、诊断和"
            "就诊日期。",
            "SELECT p.name, p.age, v.disease, v.visit_date FROM patients p "
            "JOIN visits v ON p.patient_id=v.patient_id "
            f"WHERE p.age>={age} AND v.exam='{_sql_string(exam)}' "
            "ORDER BY v.visit_date, p.patient_id",
            {"age": age, "exam": exam},
        )
    relation = _pick(d.relations, index, attempt)
    confidence = [0.5, 0.6, 0.7][(index + attempt) % 3]
    if family == 3:
        return Rendered(
            "kg_head_entity_join",
            f"查询中文关系为{relation}且置信度至少{_num(confidence)}的三元组，"
            "同时返回头实体的提及次数。",
            "SELECT t.head, t.relation_zh, t.tail, e.mention_count "
            "FROM kg_triples t JOIN kg_entities e "
            "ON t.head=e.name AND t.head_type=e.type "
            f"WHERE t.relation_zh='{_sql_string(relation)}' "
            f"AND t.confidence>={_num(confidence)} "
            "ORDER BY e.mention_count DESC, t.head, t.tail",
            {"relation": relation, "confidence": confidence},
        )
    return Rendered(
        "kg_tail_entity_join",
        f"查询中文关系为{relation}且置信度至少{_num(confidence)}的三元组，"
        "同时返回尾实体的类型和提及次数。",
        "SELECT t.head, t.tail, e.type, e.mention_count "
        "FROM kg_triples t JOIN kg_entities e "
        "ON t.tail=e.name AND t.tail_type=e.type "
        f"WHERE t.relation_zh='{_sql_string(relation)}' "
        f"AND t.confidence>={_num(confidence)} "
        "ORDER BY e.mention_count DESC, t.head, t.tail",
        {"relation": relation, "confidence": confidence},
    )


def _temporal(index: int, attempt: int, d: Domains) -> Rendered:
    family = index % 5
    start, end = _date_pair(d, index, attempt)
    if family == 0:
        return Rendered(
            "date_range_department_count",
            f"统计{start}至{end}之间各科室的就诊次数。",
            "SELECT department, COUNT(*) AS visit_count FROM visits "
            f"WHERE visit_date BETWEEN '{start}' AND '{end}' "
            "GROUP BY department ORDER BY visit_count DESC, department",
            {"start_date": start, "end_date": end},
        )
    if family == 1:
        return Rendered(
            "yearly_visit_count",
            f"统计{start}及以后每年的就诊次数和总费用。",
            "SELECT strftime('%Y', visit_date) AS year, "
            "COUNT(*) AS visit_count, SUM(cost) AS total_cost FROM visits "
            f"WHERE visit_date>='{start}' GROUP BY year ORDER BY year",
            {"start_date": start},
        )
    if family == 2:
        department = _pick(d.departments, index, attempt)
        return Rendered(
            "monthly_department_average",
            f"统计科室{department}在{start}至{end}之间每月的就诊次数和"
            "平均费用。",
            "SELECT strftime('%Y-%m', visit_date) AS month, "
            "COUNT(*) AS visit_count, AVG(cost) AS avg_cost FROM visits "
            f"WHERE department='{_sql_string(department)}' "
            f"AND visit_date BETWEEN '{start}' AND '{end}' "
            "GROUP BY month ORDER BY month",
            {"department": department, "start_date": start, "end_date": end},
        )
    if family == 3:
        cost = [1000, 3000, 5000][(index + attempt) % 3]
        return Rendered(
            "latest_visit_before_date",
            f"截至{end}，返回每位患者最近一次费用至少{cost}元的就诊日期、"
            "科室和费用。",
            "SELECT v.patient_id, v.visit_date, v.department, v.cost "
            "FROM visits v "
            f"WHERE v.visit_date<='{end}' AND v.cost>={cost} "
            "AND v.visit_date=(SELECT MAX(v2.visit_date) FROM visits v2 "
            "WHERE v2.patient_id=v.patient_id "
            f"AND v2.visit_date<='{end}' AND v2.cost>={cost}) "
            "ORDER BY v.patient_id, v.visit_id",
            {"end_date": end, "cost": cost},
        )
    disease = _pick(d.diseases, index, attempt)
    return Rendered(
        "first_disease_visit_after_date",
        f"返回{start}及以后每位患者诊断为{disease}的首次就诊日期。",
        "SELECT patient_id, MIN(visit_date) AS first_visit_date FROM visits "
        f"WHERE visit_date>='{start}' AND disease='{_sql_string(disease)}' "
        "GROUP BY patient_id ORDER BY patient_id",
        {"start_date": start, "disease": disease},
    )


def _knowledge_graph(index: int, attempt: int, d: Domains) -> Rendered:
    family = index % 6
    relation = _pick(d.relations, index, attempt)
    confidence = [0.5, 0.6, 0.7][(index + attempt) % 3]
    if family == 0:
        head_type = _pick(d.head_types, index, attempt)
        return Rendered(
            "kg_relation_filter",
            f"返回头实体类型为{head_type}、中文关系为{relation}且置信度至少"
            f"{_num(confidence)}的头实体、尾实体和观察次数。",
            "SELECT head, tail, count FROM kg_triples "
            f"WHERE head_type='{_sql_string(head_type)}' "
            f"AND relation_zh='{_sql_string(relation)}' "
            f"AND confidence>={_num(confidence)} "
            "ORDER BY count DESC, head, tail",
            {
                "head_type": head_type,
                "relation": relation,
                "confidence": confidence,
            },
        )
    if family == 1:
        head = _pick(d.kg_heads, index, attempt)
        return Rendered(
            "kg_head_neighbours",
            f"查询头实体为{head}且置信度至少{_num(confidence)}的所有关系、"
            "中文关系名和尾实体。",
            "SELECT relation, relation_zh, tail FROM kg_triples "
            f"WHERE head='{_sql_string(head)}' "
            f"AND confidence>={_num(confidence)} "
            "ORDER BY relation, tail",
            {"head": head, "confidence": confidence},
        )
    if family == 2:
        head_type = _pick(d.head_types, index, attempt)
        return Rendered(
            "kg_relation_aggregation",
            f"对头实体类型为{head_type}且置信度至少{_num(confidence)}的"
            "三元组，按中文关系统计三元组数和观察次数总和。",
            "SELECT relation_zh, COUNT(*) AS triple_count, "
            "SUM(count) AS observed_count FROM kg_triples "
            f"WHERE head_type='{_sql_string(head_type)}' "
            f"AND confidence>={_num(confidence)} GROUP BY relation_zh "
            "ORDER BY triple_count DESC, relation_zh",
            {"head_type": head_type, "confidence": confidence},
        )
    if family == 3:
        entity_type = _pick(d.entity_types, index, attempt)
        mentions = [2, 5, 10, 20][
            (index + attempt // max(1, len(d.entity_types))) % 4
        ]
        return Rendered(
            "kg_entity_outgoing_join",
            f"查询类型为{entity_type}且提及次数至少{mentions}次的实体，返回"
            "它们作为头实体时的中文关系和尾实体。",
            "SELECT e.name, t.relation_zh, t.tail FROM kg_entities e "
            "JOIN kg_triples t ON e.name=t.head AND e.type=t.head_type "
            f"WHERE e.type='{_sql_string(entity_type)}' "
            f"AND e.mention_count>={mentions} "
            "ORDER BY e.name, t.relation_zh, t.tail",
            {"entity_type": entity_type, "mentions": mentions},
        )
    if family == 4:
        entity_type = _pick(d.entity_types, index, attempt)
        mentions = [1, 3, 5, 10][
            (index + attempt // max(1, len(d.entity_types))) % 4
        ]
        return Rendered(
            "kg_entity_without_outgoing",
            f"返回类型为{entity_type}、提及次数至少{mentions}次且没有作为任何"
            "三元组头实体出现的实体名称。",
            "SELECT e.name FROM kg_entities e "
            f"WHERE e.type='{_sql_string(entity_type)}' "
            f"AND e.mention_count>={mentions} "
            "AND NOT EXISTS (SELECT 1 FROM kg_triples t "
            "WHERE t.head=e.name AND t.head_type=e.type) ORDER BY e.name",
            {"entity_type": entity_type, "mentions": mentions},
        )
    return Rendered(
        "kg_tail_observation_aggregation",
        f"对中文关系为{relation}且置信度至少{_num(confidence)}的三元组，"
        "按尾实体统计头实体数和观察次数总和。",
        "SELECT tail, COUNT(DISTINCT head) AS head_count, "
        "SUM(count) AS observed_count FROM kg_triples "
        f"WHERE relation_zh='{_sql_string(relation)}' "
        f"AND confidence>={_num(confidence)} GROUP BY tail "
        "ORDER BY observed_count DESC, tail",
        {"relation": relation, "confidence": confidence},
    )


def _having(index: int, attempt: int, d: Domains) -> Rendered:
    family = index % 3
    minimum = [1, 2, 3, 4][(index + attempt) % 4]
    if family == 0:
        drug = _pick(d.drugs, index, attempt)
        return Rendered(
            "department_prescription_having",
            f"找出开具药物{drug}至少{minimum}次的科室，并返回处方次数。",
            "SELECT department, COUNT(*) AS prescription_count FROM visits "
            f"WHERE drug_prescribed='{_sql_string(drug)}' GROUP BY department "
            f"HAVING COUNT(*)>={minimum} "
            "ORDER BY prescription_count DESC, department",
            {"drug": drug, "minimum": minimum},
        )
    if family == 1:
        cost = [3000, 5000, 8000, 10000][(index + attempt) % 4]
        visit_minimum = [1, 2, 3, 4][
            (index + attempt // 4) % 4
        ]
        return Rendered(
            "disease_average_cost_having",
            f"找出至少有{visit_minimum}次就诊且平均费用至少{cost}元的诊断，"
            "并返回就诊次数和平均费用。",
            "SELECT disease, COUNT(*) AS visit_count, AVG(cost) AS avg_cost "
            "FROM visits GROUP BY disease "
            f"HAVING COUNT(*)>={visit_minimum} AND AVG(cost)>={cost} "
            "ORDER BY avg_cost DESC, disease",
            {"cost": cost, "visit_minimum": visit_minimum},
        )
    age = [30, 40, 50, 60][(index + attempt) % 4]
    minimum = [1, 2, 3, 4][(index + attempt // 4) % 4]
    return Rendered(
        "region_patient_having",
        f"对年龄至少{age}岁的患者按地区分组，找出至少有{minimum}位不同"
        "患者的地区。",
        "SELECT p.region, COUNT(DISTINCT p.patient_id) AS patient_count "
        "FROM patients p JOIN visits v ON p.patient_id=v.patient_id "
        f"WHERE p.age>={age} GROUP BY p.region "
        f"HAVING COUNT(DISTINCT p.patient_id)>={minimum} "
        "ORDER BY patient_count DESC, p.region",
        {"age": age, "minimum": minimum},
    )


def _nested(index: int, attempt: int, d: Domains) -> Rendered:
    family = index % 5
    if family == 0:
        department = _pick(d.departments, index, attempt)
        delta = [0, 500, 1000, 2000][(index + attempt) % 4]
        return Rendered(
            "cost_above_department_average",
            f"查询科室{department}中费用至少比该科室平均费用高{delta}元的"
            "就诊编号、患者编号、诊断和费用。",
            "SELECT visit_id, patient_id, disease, cost FROM visits v "
            f"WHERE department='{_sql_string(department)}' "
            f"AND cost>=(SELECT AVG(cost)+{delta} FROM visits "
            f"WHERE department='{_sql_string(department)}') "
            "ORDER BY cost DESC, visit_id",
            {"department": department, "delta": delta},
        )
    if family == 1:
        _, end = _date_pair(d, index, attempt)
        cost = [1000, 3000, 5000][(index + attempt) % 3]
        return Rendered(
            "correlated_latest_visit",
            f"截至{end}，返回每位患者最近一次费用至少{cost}元的就诊编号和"
            "日期。",
            "SELECT v.patient_id, v.visit_id, v.visit_date FROM visits v "
            f"WHERE v.visit_date<='{end}' AND v.cost>={cost} "
            "AND v.visit_date=(SELECT MAX(v2.visit_date) FROM visits v2 "
            "WHERE v2.patient_id=v.patient_id "
            f"AND v2.visit_date<='{end}' AND v2.cost>={cost}) "
            "ORDER BY v.patient_id, v.visit_id",
            {"end_date": end, "cost": cost},
        )
    if family == 2:
        department = _pick(d.departments, index, attempt)
        cost = [1000, 3000, 5000][(index + attempt) % 3]
        return Rendered(
            "patient_exists_department_visit",
            f"返回至少有一次在科室{department}且费用至少{cost}元就诊记录的"
            "患者编号和姓名。",
            "SELECT p.patient_id, p.name FROM patients p WHERE EXISTS "
            "(SELECT 1 FROM visits v WHERE v.patient_id=p.patient_id "
            f"AND v.department='{_sql_string(department)}' "
            f"AND v.cost>={cost}) ORDER BY p.patient_id",
            {"department": department, "cost": cost},
        )
    if family == 3:
        start, end = _date_pair(d, index, attempt)
        return Rendered(
            "patient_not_exists_date_range",
            f"返回在{start}至{end}之间没有就诊记录的患者编号和姓名。",
            "SELECT p.patient_id, p.name FROM patients p WHERE NOT EXISTS "
            "(SELECT 1 FROM visits v WHERE v.patient_id=p.patient_id "
            f"AND v.visit_date BETWEEN '{start}' AND '{end}') "
            "ORDER BY p.patient_id",
            {"start_date": start, "end_date": end},
        )
    cost = [1000, 3000, 5000][(index + attempt) % 3]
    los = [0, 1, 3, 5][(index + attempt // 3) % 4]
    return Rendered(
        "maximum_cost_per_department",
        f"返回每个科室中住院天数至少{los}天、费用至少{cost}元且费用等于"
        "该科室在相同住院天数条件下最高费用的就诊编号、科室和费用。",
        "SELECT v.visit_id, v.department, v.cost FROM visits v "
        f"WHERE v.los_days>={los} AND v.cost>={cost} "
        "AND v.cost=(SELECT MAX(v2.cost) FROM visits v2 "
        "WHERE v2.department=v.department "
        f"AND v2.los_days>={los}) "
        "ORDER BY v.department, v.visit_id",
        {"cost": cost, "los_days": los},
    )


def _cte(index: int, attempt: int, d: Domains) -> Rendered:
    family = index % 4
    if family == 0:
        department = _pick(d.departments, index, attempt)
        total = [3000, 5000, 8000, 12000][(index + attempt) % 4]
        limit = [2, 3, 4, 5][(index + attempt) % 4]
        return Rendered(
            "cte_patient_cost",
            f"先汇总科室{department}中每位患者的累计费用，再返回累计费用至少"
            f"{total}元的前{limit}位患者。",
            "WITH patient_cost AS (SELECT patient_id, SUM(cost) AS total_cost "
            f"FROM visits WHERE department='{_sql_string(department)}' "
            "GROUP BY patient_id) SELECT patient_id, total_cost "
            f"FROM patient_cost WHERE total_cost>={total} "
            "ORDER BY total_cost DESC, patient_id "
            f"LIMIT {limit}",
            {"department": department, "total": total, "limit": limit},
        )
    if family == 1:
        minimum = [2, 3, 4, 5][(index + attempt) % 4]
        average = [3000, 5000, 8000][(index + attempt) % 3]
        return Rendered(
            "cte_department_stats",
            f"先计算各科室的就诊次数和平均费用，再返回就诊次数至少{minimum}次"
            f"且平均费用至少{average}元的科室。",
            "WITH department_stats AS (SELECT department, COUNT(*) AS n, "
            "AVG(cost) AS avg_cost FROM visits GROUP BY department) "
            "SELECT department, n, avg_cost FROM department_stats "
            f"WHERE n>={minimum} AND avg_cost>={average} "
            "ORDER BY n DESC, department",
            {"minimum": minimum, "average": average},
        )
    if family == 2:
        start, end = _date_pair(d, index, attempt)
        minimum = [1, 2, 3, 4][(index + attempt) % 4]
        return Rendered(
            "cte_monthly_stats",
            f"先汇总{start}至{end}之间每月的就诊次数和总费用，再返回就诊次数"
            f"至少{minimum}次的月份。",
            "WITH monthly AS (SELECT strftime('%Y-%m', visit_date) AS month, "
            "COUNT(*) AS n, SUM(cost) AS total_cost FROM visits "
            f"WHERE visit_date BETWEEN '{start}' AND '{end}' GROUP BY month) "
            "SELECT month, n, total_cost FROM monthly "
            f"WHERE n>={minimum} ORDER BY month",
            {"start_date": start, "end_date": end, "minimum": minimum},
        )
    relation = _pick(d.relations, index, attempt)
    minimum = [2, 3, 5, 8][(index + attempt) % 4]
    return Rendered(
        "cte_kg_head_stats",
        f"先汇总中文关系为{relation}的每个头实体的观察次数，再返回观察次数"
        f"至少{minimum}次的头实体。",
        "WITH head_stats AS (SELECT head, SUM(count) AS observed_count "
        "FROM kg_triples "
        f"WHERE relation_zh='{_sql_string(relation)}' GROUP BY head) "
        "SELECT head, observed_count FROM head_stats "
        f"WHERE observed_count>={minimum} "
        "ORDER BY observed_count DESC, head",
        {"relation": relation, "minimum": minimum},
    )


def _set_operation(index: int, attempt: int, d: Domains) -> Rendered:
    family = index % 3
    if family == 0:
        first = _pick(d.regions, index, attempt)
        second = _pick(
            [value for value in d.regions if value != first],
            index,
            attempt,
            5,
        )
        return Rendered(
            "union_regions",
            f"返回地区为{first}或地区为{second}的患者编号，合并并去重。",
            "SELECT patient_id FROM patients "
            f"WHERE region='{_sql_string(first)}' UNION "
            "SELECT patient_id FROM patients "
            f"WHERE region='{_sql_string(second)}' ORDER BY patient_id",
            {"first_region": first, "second_region": second},
        )
    first = _pick(d.departments, index, attempt)
    second = _pick(
        [value for value in d.departments if value != first],
        index,
        attempt,
        5,
    )
    if family == 1:
        return Rendered(
            "intersect_departments",
            f"返回既在科室{first}就诊过、又在科室{second}就诊过的患者编号。",
            "SELECT patient_id FROM visits "
            f"WHERE department='{_sql_string(first)}' INTERSECT "
            "SELECT patient_id FROM visits "
            f"WHERE department='{_sql_string(second)}' ORDER BY patient_id",
            {"first_department": first, "second_department": second},
        )
    return Rendered(
        "except_departments",
        f"返回在科室{first}就诊过但没有在科室{second}就诊过的患者编号。",
        "SELECT patient_id FROM visits "
        f"WHERE department='{_sql_string(first)}' EXCEPT "
        "SELECT patient_id FROM visits "
        f"WHERE department='{_sql_string(second)}' ORDER BY patient_id",
        {"first_department": first, "second_department": second},
    )


def _window(index: int, attempt: int, d: Domains) -> Rendered:
    family = index % 3
    limit = [2, 3, 4, 5][(index + attempt) % 4]
    if family == 0:
        department = _pick(d.departments, index, attempt)
        cost = [1000, 3000, 5000][(index + attempt) % 3]
        return Rendered(
            "row_number_department_cost",
            f"对科室{department}中费用至少{cost}元的就诊按费用降序编号，"
            f"返回前{limit}名及其名次。",
            "SELECT visit_id, patient_id, cost, ROW_NUMBER() OVER "
            "(PARTITION BY department ORDER BY cost DESC, visit_id) AS fee_rank "
            "FROM visits "
            f"WHERE department='{_sql_string(department)}' AND cost>={cost} "
            "ORDER BY fee_rank "
            f"LIMIT {limit}",
            {"department": department, "cost": cost, "limit": limit},
        )
    if family == 1:
        department = _pick(d.departments, index, attempt)
        total = [3000, 5000, 8000][(index + attempt) % 3]
        return Rendered(
            "dense_rank_patient_cost",
            f"在科室{department}中按患者累计费用排名，筛选累计费用至少{total}元"
            f"的记录并返回前{limit}名及排名。",
            "SELECT patient_id, SUM(cost) AS total_cost, "
            "DENSE_RANK() OVER (ORDER BY SUM(cost) DESC) AS total_rank "
            "FROM visits "
            f"WHERE department='{_sql_string(department)}' GROUP BY patient_id "
            f"HAVING SUM(cost)>={total} ORDER BY total_rank, patient_id "
            f"LIMIT {limit}",
            {"department": department, "total": total, "limit": limit},
        )
    start = _pick(d.dates, index, attempt)
    return Rendered(
        "running_patient_cost",
        f"对{start}及以后的就诊，按患者和日期计算累计费用，返回前{limit}条"
        "记录。",
        "SELECT patient_id, visit_date, cost, SUM(cost) OVER "
        "(PARTITION BY patient_id ORDER BY visit_date, visit_id "
        "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_cost "
        "FROM visits "
        f"WHERE visit_date>='{start}' "
        "ORDER BY patient_id, visit_date, visit_id "
        f"LIMIT {limit}",
        {"start_date": start, "limit": limit},
    )


BUILDERS: dict[str, Callable[[int, int, Domains], Rendered]] = {
    "filtering": _filtering,
    "aggregation_grouping": _aggregation,
    "top_k": _top_k,
    "multi_table_join": _join,
    "temporal": _temporal,
    "knowledge_graph": _knowledge_graph,
    "having": _having,
    "nested_correlated": _nested,
    "cte": _cte,
    "set_operation": _set_operation,
    "window": _window,
}


def _difficulty_sequence(category: str) -> list[str]:
    distribution = DIFFICULTY_BY_CATEGORY[category]
    sequence = [
        difficulty
        for difficulty in ("medium", "hard", "extra", "extra-hard")
        for _ in range(distribution.get(difficulty, 0))
    ]
    if len(sequence) != CATEGORY_QUOTAS[category]:
        raise AssertionError(f"difficulty allocation differs for {category}")
    return sequence


def _validate_category_sql(category: str, sql: str) -> str:
    try:
        from sqlglot import exp, parse_one
    except ImportError as exc:
        raise RuntimeError("sqlglot is required for public projection") from exc
    tree = parse_one(sql, read="sqlite")
    checks = {
        "filtering": bool(tree.find(exp.Where)),
        "aggregation_grouping": bool(
            tree.find(exp.Group) or tree.find(exp.AggFunc)
        ),
        "top_k": bool(tree.find(exp.Order) and tree.find(exp.Limit)),
        "multi_table_join": bool(tree.find(exp.Join)),
        "temporal": any(
            "date" in str(column.name).casefold()
            or "time" in str(column.name).casefold()
            for column in tree.find_all(exp.Column)
        )
        or "STRFTIME" in tree.sql(dialect="sqlite").upper(),
        "knowledge_graph": any(
            table.name.casefold() in {"kg_entities", "kg_triples"}
            for table in tree.find_all(exp.Table)
        ),
        "having": bool(tree.find(exp.Having)),
        "nested_correlated": bool(
            tree.find(exp.Subquery) or tree.find(exp.Exists)
        ),
        "cte": bool(tree.find(exp.CTE)),
        "set_operation": any(
            tree.find(kind) for kind in (exp.Union, exp.Intersect, exp.Except)
        ),
        "window": bool(tree.find(exp.Window)),
    }
    if not checks[category]:
        raise ValueError(f"{category} SQL lacks its declared construct")
    return validate_gold_sql(sql)


class DonorSelector:
    def __init__(
        self,
        private_index: dict[str, Any],
        public_attestation: dict[str, Any],
        private_index_sha256: str,
    ):
        if (
            private_index.get("kind")
            != "private_authoritative_nl2sql_donor_index"
            or private_index.get("contains_source_questions_or_sql") is not False
            or private_index.get("official_source_test_content_parsed") is not False
        ):
            raise ValueError("private donor index violates source isolation")
        if (
            public_attestation.get("kind")
            != "public_authoritative_nl2sql_donor_index_attestation"
            or public_attestation.get("private_index_sha256")
            != private_index_sha256
            or public_attestation.get("official_source_test_content_parsed")
            is not False
        ):
            raise ValueError("public donor attestation binding failed")
        if public_attestation.get("aggregate") != private_index.get("aggregate"):
            raise ValueError("public/private donor aggregates differ")
        self._pools: dict[str, list[dict[str, Any]]] = {}
        for construct in set(DONOR_CONSTRUCT.values()):
            values = [
                row
                for row in private_index.get("donors", [])
                if construct in row.get("constructs", [])
            ]
            values.sort(
                key=lambda row: (
                    row["question_sha256"],
                    row["sql_sha256"],
                    row["dataset"],
                    row["source_file"],
                    row["source_row_index"],
                )
            )
            if not values:
                raise ValueError(f"donor index has no {construct} donors")
            self._pools[construct] = values
        self._cursor = Counter()
        self._used: set[
            tuple[str, str, int]
        ] = set()

    def take(self, category: str) -> dict[str, Any]:
        construct = DONOR_CONSTRUCT[category]
        pool = self._pools[construct]
        start = self._cursor[construct]
        for offset in range(len(pool)):
            row = pool[(start + offset) % len(pool)]
            identity = (
                str(row["dataset"]),
                str(row["source_file"]),
                int(row["source_row_index"]),
            )
            if identity in self._used:
                continue
            self._used.add(identity)
            self._cursor[construct] = start + offset + 1
            return {
                "dataset": row["dataset"],
                "source_revision": row["source_revision"],
                "source_file": row["source_file"],
                "source_split": row["source_split"],
                "source_row_index": row["source_row_index"],
                "source_id_sha256": row["source_id_sha256"],
                "source_db_id_sha256": row["source_db_id_sha256"],
                "question_sha256": row["question_sha256"],
                "sql_sha256": row["sql_sha256"],
                "ast_skeleton_sha256": row["ast_skeleton_sha256"],
                "constructs": row["constructs"],
            }
        raise ValueError(f"not enough unique {construct} donors")


def _result_fingerprint(columns: list[str], rows: list[tuple]) -> str:
    payload = {
        "columns": [str(column) for column in columns],
        "rows": [
            [
                value
                if value is None
                or isinstance(value, (str, int, float, bool))
                else repr(value)
                for value in row
            ]
            for row in rows
        ],
    }
    return _sha256_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def build_projection(
    *,
    donor_index_path: Path,
    donor_attestation_path: Path,
    db_path: Path,
    source_tree_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", source_tree_sha256):
        raise ValueError("source tree identity must be a full SHA-256")
    donor_index_sha256 = _sha256(donor_index_path)
    donor_index = json.loads(donor_index_path.read_text(encoding="utf-8"))
    donor_attestation = json.loads(
        donor_attestation_path.read_text(encoding="utf-8")
    )
    selector = DonorSelector(
        donor_index,
        donor_attestation,
        donor_index_sha256,
    )
    domains = _load_domains(db_path)
    rows: list[dict[str, Any]] = []
    questions: set[str] = set()
    ids: set[str] = set()
    skeletons = Counter()
    result_rows = 0
    source_counts = Counter()

    for category, quota in CATEGORY_QUOTAS.items():
        difficulties = _difficulty_sequence(category)
        builder = BUILDERS[category]
        for index in range(quota):
            donor = selector.take(category)
            accepted: Rendered | None = None
            accepted_skeleton = ""
            accepted_result: tuple[list[str], list[tuple]] | None = None
            last_error: Exception | None = None
            for attempt in range(256):
                candidate = builder(index, attempt, domains)
                try:
                    normalised = normalise_question(candidate.question)
                    if normalised in questions:
                        raise ValueError("duplicate normalized question")
                    validate_question_sql_grounding(
                        candidate.question,
                        candidate.sql,
                    )
                    skeleton = _validate_category_sql(
                        category,
                        candidate.sql,
                    )
                    columns, result = execute_complete_readonly(
                        candidate.sql,
                        db_path,
                    )
                    if not result:
                        raise ValueError("projected SQL returned no rows")
                    if not columns:
                        raise ValueError("projected SQL returned no columns")
                except Exception as exc:
                    last_error = exc
                    continue
                accepted = candidate
                accepted_skeleton = skeleton
                accepted_result = (columns, result)
                break
            if accepted is None or accepted_result is None:
                raise ValueError(
                    f"unable to render non-empty {category} item {index}: "
                    f"{last_error}"
                )

            item_id = (
                "public-projection-v1-"
                f"{category.replace('_', '-')}-{index + 1:03d}"
            )
            if item_id in ids:
                raise AssertionError("projection generated a duplicate id")
            ids.add(item_id)
            questions.add(normalise_question(accepted.question))
            skeletons[accepted_skeleton] += 1
            result_rows += len(accepted_result[1])
            source_counts[str(donor["dataset"])] += 1
            rows.append(
                {
                    "id": item_id,
                    "question": accepted.question,
                    "gold_sql": accepted.sql,
                    "difficulty": difficulties[index],
                    "category": category,
                    "projection_provenance": {
                        "typed_ir_family": accepted.family,
                        "typed_ir_parameters": accepted.parameters,
                        "donor": donor,
                        "donor_construct": DONOR_CONSTRUCT[category],
                        "transform_kind": TRANSFORM_KIND[category],
                        "result_fingerprint_sha256": _result_fingerprint(
                            *accepted_result
                        ),
                        "result_row_count": len(accepted_result[1]),
                    },
                }
            )

    category_counts = Counter(row["category"] for row in rows)
    difficulty_counts = Counter(row["difficulty"] for row in rows)
    if dict(category_counts) != CATEGORY_QUOTAS:
        raise AssertionError("projected category quotas differ")
    if dict(difficulty_counts) != EXPECTED_DIFFICULTY_QUOTAS:
        raise AssertionError("projected difficulty quotas differ")
    if len(rows) != 300 or len(questions) != 300 or len(ids) != 300:
        raise AssertionError("projection uniqueness/count contract failed")
    aggregate = {
        "n_items": len(rows),
        "category_counts": dict(sorted(category_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "n_sql_ast_skeletons": len(skeletons),
        "maximum_ast_family_size": max(skeletons.values()),
        "all_gold_sql_validated": True,
        "all_gold_sql_executed_readonly_without_truncation": True,
        "all_result_sets_nonempty": True,
        "total_result_rows": result_rows,
        "all_predicate_literals_grounded_in_question": True,
    }
    bindings = {
        "source_tree_sha256": source_tree_sha256.lower(),
        "donor_index": {
            "path": str(donor_index_path.resolve()),
            "sha256": donor_index_sha256,
        },
        "donor_attestation": {
            "path": str(donor_attestation_path.resolve()),
            "sha256": _sha256(donor_attestation_path),
        },
        "database": {
            "path": str(db_path.resolve()),
            "sha256": _sha256(db_path),
        },
        "projector_sha256": _sha256(Path(__file__)),
        "protocol_validator_sha256": _sha256(
            ROOT / "dki" / "analysis" / "nl2sql_protocol.py"
        ),
    }
    return rows, {"aggregate": aggregate, "bindings": bindings}


def publish_projection(
    *,
    rows: list[dict[str, Any]],
    evidence: dict[str, Any],
    private_output_dir: Path,
    public_output: Path,
    created_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    private_output_dir = private_output_dir.resolve()
    public_output = public_output.resolve()
    if private_output_dir.exists() or public_output.exists():
        raise FileExistsError("projection output exists; choose a new RID")
    if not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
        created_at,
    ):
        raise ValueError("projection created_at must be UTC RFC3339")
    private_output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = (
        private_output_dir.parent
        / f".{private_output_dir.name}.projection-staging"
    )
    if staging.exists():
        raise FileExistsError(
            f"stale projection staging exists; do not reuse it: {staging}"
        )
    staging.mkdir()
    try:
        corpus_path = staging / "corpus.jsonl"
        corpus_bytes = _jsonl_bytes(rows)
        _write_synced(corpus_path, corpus_bytes)
        attestation = {
            "schema_version": 1,
            "kind": "public_authoritative_four_table_projection_attestation",
            "dataset_role": "prospective_public_derived_curation_unopened",
            "selection_used": False,
            "created_at": created_at,
            "independence_basis": {
                "public_source_development_rows": (
                    "hash/AST pointers only; source text and SQL not copied"
                ),
                "official_source_test_content_parsed": False,
                "project_nl2sql_engine_imported": False,
                "project_local_evaluation_questions_used": False,
                "development_or_holdout_preassigned": False,
                "question_sql_semantics": (
                    "rendered from the same typed IR and independently checked "
                    "by AST grounding plus read-only SQLite execution"
                ),
            },
            "curator_identity": (
                "deterministic-public-donor-four-table-projector-v1"
            ),
            "semantic_reviewer_identity": (
                "nl2sql-protocol-ast-grounding-readonly-executor-v1"
            ),
            "mechanical_projection": True,
            "all_gold_sql_executed": True,
            "all_pairs_semantically_validated_by_construction": True,
            "engine_context_provided": False,
            "local_evaluation_context_provided": False,
            "project_local_question_reuse": False,
            "project_local_question_paraphrase": False,
            "development_or_holdout_preassigned": False,
            "n_items": len(rows),
            "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
            **evidence,
        }
        attestation_path = staging / "projection_attestation.json"
        _write_synced(attestation_path, _json_bytes(attestation))
        manifest = {
            "schema_version": 1,
            "kind": "private_public_authoritative_projection_manifest",
            "created_once": True,
            "submission_package_allowed": False,
            "files": {
                "corpus.jsonl": {
                    "size": corpus_path.stat().st_size,
                    "sha256": _sha256(corpus_path),
                },
                "projection_attestation.json": {
                    "size": attestation_path.stat().st_size,
                    "sha256": _sha256(attestation_path),
                },
            },
        }
        _write_synced(staging / "projection_manifest.json", _json_bytes(manifest))
        os.rename(staging, private_output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    public = {
        "schema_version": 1,
        "kind": "public_authoritative_four_table_projection_receipt",
        "dataset_role": attestation["dataset_role"],
        "selection_used": False,
        "private_projection_manifest_sha256": _sha256(
            private_output_dir / "projection_manifest.json"
        ),
        "corpus_sha256": attestation["corpus_sha256"],
        "aggregate": attestation["aggregate"],
        "bindings": {
            key: (
                {"sha256": value["sha256"]}
                if isinstance(value, dict) and "sha256" in value
                else value
            )
            for key, value in attestation["bindings"].items()
        },
        "contains_questions_or_gold_sql": False,
        "contains_source_row_pointers": False,
        "official_source_test_content_parsed": False,
        "project_nl2sql_engine_imported": False,
        "four_table_blind_gate": False,
        "freeze_required": True,
    }
    _publish_file_once(public_output, _json_bytes(public))
    return attestation, public


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor-index", type=Path, required=True)
    parser.add_argument("--donor-attestation", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--created-at", required=True)
    args = parser.parse_args()
    if "outputs/private/" not in args.private_output_dir.resolve().as_posix():
        raise SystemExit("detailed projection must remain under outputs/private/<RID>")
    if "outputs/private/" in args.public_output.resolve().as_posix():
        raise SystemExit("public projection receipt must not be private")
    try:
        rows, evidence = build_projection(
            donor_index_path=args.donor_index,
            donor_attestation_path=args.donor_attestation,
            db_path=args.db,
            source_tree_sha256=args.source_tree_sha256,
        )
        _, public = publish_projection(
            rows=rows,
            evidence=evidence,
            private_output_dir=args.private_output_dir,
            public_output=args.public_output,
            created_at=args.created_at,
        )
    except (
        FileExistsError,
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
        sqlite3.Error,
    ) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "private_output_dir": str(args.private_output_dir),
                "public_output": str(args.public_output),
                "n_items": public["aggregate"]["n_items"],
                "n_sql_ast_skeletons": public["aggregate"][
                    "n_sql_ast_skeletons"
                ],
                "all_result_sets_nonempty": True,
                "official_source_test_content_parsed": False,
                "four_table_blind_gate": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
