from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import dki.llm as llm_module
from dki.analysis.nl2sql import NL2SQL


AGE_QUESTION = (
    "筛选性别为女且年龄在28到42岁之间（含边界）的患者编号和年龄，"
    "按年龄升序排列。"
)
AGE_SQL = (
    "SELECT patient_id, age FROM patients WHERE age BETWEEN 28 AND 42 "
    "AND gender = '女' ORDER BY age ASC, patient_id ASC"
)

DEPARTMENT_QUESTION = (
    "在合成内科科室中，按疾病汇总就诊次数和总费用，返回合成内科科室下"
    "各疾病的就诊次数和保留2位小数的总费用，按就诊次数降序排列。"
)
DEPARTMENT_SQL = (
    "SELECT disease, COUNT(*) AS visit_count, ROUND(SUM(cost), 2) AS total_cost "
    "FROM visits WHERE department = '合成内科' GROUP BY disease "
    "ORDER BY visit_count DESC, disease ASC"
)

TYPED_KG_QUESTION = (
    "把就诊疾病经DIAGNOSED_BY关系连接到Exam实体，限定置信度至少0.85；"
    "返回就诊编号、尾实体名称和置信度，并按置信度降序。"
)
TYPED_KG_SQL = (
    "SELECT v.visit_id, t.tail, t.confidence FROM visits v "
    "JOIN kg_triples t ON t.head = v.disease AND t.relation = 'DIAGNOSED_BY' "
    "JOIN kg_entities e ON e.name = t.tail AND e.type = 'Exam' "
    "WHERE t.confidence >= 0.85 ORDER BY t.confidence DESC"
)


@pytest.fixture()
def synthetic_closed_contract_db(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic-closed-contract.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE patients (
            patient_id TEXT PRIMARY KEY,
            name TEXT,
            gender TEXT,
            age INTEGER,
            region TEXT
        );
        CREATE TABLE visits (
            visit_id TEXT PRIMARY KEY,
            patient_id TEXT,
            visit_date TEXT,
            department TEXT,
            disease TEXT,
            drug_prescribed TEXT,
            exam TEXT,
            cost REAL,
            los_days INTEGER
        );
        CREATE TABLE kg_entities (
            name TEXT,
            type TEXT,
            mention_count INTEGER
        );
        CREATE TABLE kg_triples (
            head TEXT,
            head_type TEXT,
            relation TEXT,
            relation_zh TEXT,
            tail TEXT,
            tail_type TEXT,
            confidence REAL,
            count INTEGER
        );

        INSERT INTO patients VALUES ('pf', '合成人甲', '女', 35, '东区');
        INSERT INTO patients VALUES ('pm', '合成人乙', '男', 35, '西区');
        INSERT INTO patients VALUES ('po', '合成人丙', '女', 63, '东区');

        INSERT INTO visits VALUES (
            'visit_a', 'pf', '2026-01-03', '合成内科', '疾病甲', '', '检查甲',
            100.0, 2
        );
        INSERT INTO visits VALUES (
            'visit_b', 'pm', '2026-02-04', '合成内科', '疾病乙', '', '检查乙',
            200.0, 3
        );
        INSERT INTO visits VALUES (
            'visit_c', 'po', '2026-03-05', '合成内科', '疾病乙', '', '检查乙',
            50.0, 4
        );
        INSERT INTO visits VALUES (
            'visit_other', 'pf', '2026-04-06', '其他科', '疾病丙', '', '检查丙',
            999.0, 5
        );

        INSERT INTO kg_entities VALUES ('检查甲', 'Exam', 8);
        INSERT INTO kg_entities VALUES ('检查甲', 'Symptom', 3);
        INSERT INTO kg_entities VALUES ('检查乙', 'Exam', 7);
        INSERT INTO kg_triples VALUES (
            '疾病甲', 'Disease', 'DIAGNOSED_BY', '诊断检查',
            '检查甲', 'Exam', 0.93, 2
        );
        """
    )
    connection.commit()
    connection.close()
    return path


@pytest.mark.parametrize(
    ("question", "expected_sql", "expected_columns", "expected_rows"),
    (
        (
            AGE_QUESTION,
            AGE_SQL,
            ["patient_id", "age"],
            [("pf", 35)],
        ),
        (
            DEPARTMENT_QUESTION,
            DEPARTMENT_SQL,
            ["disease", "visit_count", "total_cost"],
            [("疾病乙", 2, 250.0), ("疾病甲", 1, 100.0)],
        ),
        (
            TYPED_KG_QUESTION,
            TYPED_KG_SQL,
            ["visit_id", "tail", "confidence"],
            [("visit_a", "检查甲", 0.93)],
        ),
    ),
)
def test_v28_closed_contracts_compile_and_preserve_exact_result_shape(
    synthetic_closed_contract_db: Path,
    question: str,
    expected_sql: str,
    expected_columns: list[str],
    expected_rows: list[tuple[object, ...]],
) -> None:
    nl = NL2SQL(synthetic_closed_contract_db)
    normalized = nl._normalize(question)
    sql = nl._compile_output_complete_structural_sql(normalized)
    assert sql == expected_sql

    connection = sqlite3.connect(synthetic_closed_contract_db)
    cursor = connection.execute(sql)
    columns = [item[0] for item in cursor.description]
    rows = cursor.fetchall()
    connection.close()
    assert columns == expected_columns
    assert rows == expected_rows

    rule_result = nl._rule_sql_normalized_result(normalized)
    assert nl._authoritative_rule_before_model(normalized, rule_result)


@pytest.mark.parametrize(
    "question",
    (
        "筛选性别为女且年龄在42到28岁之间（含边界）的患者编号和年龄，按年龄升序排列。",
        "筛选性别为女且年龄在28到42岁之间（含边界）且地区为东区的患者编号和年龄，按年龄升序排列。",
        "在合成内科科室中，按疾病汇总就诊次数和总费用，返回其他科科室下各疾病的就诊次数和保留2位小数的总费用，按就诊次数降序排列。",
        "在不存在科室科室中，按疾病汇总就诊次数和总费用，返回不存在科室科室下各疾病的就诊次数和保留2位小数的总费用，按就诊次数降序排列。",
        "把就诊疾病经DIAGNOSED_BY关系连接到Symptom实体，限定置信度至少0.85；返回就诊编号、尾实体名称和置信度，并按置信度降序。",
        "把就诊疾病经UNKNOWN_RELATION关系连接到Exam实体，限定置信度至少0.85；返回就诊编号、尾实体名称和置信度，并按置信度降序。",
        "把就诊疾病经DIAGNOSED_BY关系连接到Exam实体，限定置信度至少1.5；返回就诊编号、尾实体名称和置信度，并按置信度降序。",
    ),
)
def test_v28_closed_contracts_reject_unbound_or_non_closed_near_misses(
    synthetic_closed_contract_db: Path,
    question: str,
) -> None:
    nl = NL2SQL(synthetic_closed_contract_db)
    assert nl._compile_output_complete_structural_sql(nl._normalize(question)) is None


def _column(table: str, column: str) -> dict[str, str]:
    return {"type": "column", "table": table, "column": column}


def _literal(value: object) -> dict[str, object]:
    return {"type": "literal", "value": value}


def _wrong_age_plan() -> dict[str, object]:
    return {
        "plan_version": 2,
        "query": {
            "type": "select",
            "from": {"type": "table", "table": "patients", "alias": "p"},
            "select": [
                {"expr": _column("p", "patient_id")},
                {"expr": _column("p", "age")},
            ],
            "where": {
                "type": "binary",
                "operator": "and",
                "left": {
                    "type": "binary",
                    "operator": "gte",
                    "left": _column("p", "age"),
                    "right": _literal(28),
                },
                "right": {
                    "type": "binary",
                    "operator": "lte",
                    "left": _column("p", "age"),
                    "right": _literal(42),
                },
            },
            "order_by": [
                {"expr": _column("p", "age"), "direction": "asc"},
                {"expr": _column("p", "patient_id"), "direction": "asc"},
            ],
        },
    }


def _wrong_department_plan() -> dict[str, object]:
    return {
        "plan_version": 2,
        "query": {
            "type": "select",
            "from": {"type": "table", "table": "visits", "alias": "v"},
            "select": [
                {
                    "expr": {
                        "type": "aggregate",
                        "function": "count",
                        "argument": None,
                    },
                    "alias": "visit_count",
                },
                {
                    "expr": {
                        "type": "function",
                        "function": "round",
                        "args": [
                            {
                                "type": "aggregate",
                                "function": "sum",
                                "argument": _column("v", "cost"),
                            },
                            _literal(2),
                        ],
                    },
                    "alias": "total_cost",
                },
            ],
            "where": {
                "type": "binary",
                "operator": "eq",
                "left": _column("v", "department"),
                "right": _literal("合成内科"),
            },
            "group_by": [_column("v", "disease")],
            "order_by": [
                {
                    "expr": {
                        "type": "aggregate",
                        "function": "count",
                        "argument": None,
                    },
                    "direction": "desc",
                },
                {"expr": _column("v", "disease"), "direction": "asc"},
            ],
        },
    }


def _wrong_typed_kg_plan() -> dict[str, object]:
    return {
        "plan_version": 2,
        "query": {
            "type": "select",
            "from": {"type": "table", "table": "visits", "alias": "v"},
            "joins": [
                {
                    "source": {
                        "type": "table",
                        "table": "kg_triples",
                        "alias": "t",
                    },
                    "on": [
                        {
                            "type": "binary",
                            "operator": "eq",
                            "left": _column("v", "disease"),
                            "right": _column("t", "head"),
                        }
                    ],
                },
                {
                    "source": {
                        "type": "table",
                        "table": "kg_entities",
                        "alias": "e",
                    },
                    "on": [
                        {
                            "type": "binary",
                            "operator": "eq",
                            "left": _column("t", "tail"),
                            "right": _column("e", "name"),
                        }
                    ],
                },
            ],
            "select": [
                {"expr": _column("v", "visit_id")},
                {"expr": _column("t", "tail")},
                {"expr": _column("t", "confidence")},
            ],
            "where": {
                "type": "binary",
                "operator": "and",
                "left": {
                    "type": "binary",
                    "operator": "eq",
                    "left": _column("t", "relation"),
                    "right": _literal("DIAGNOSED_BY"),
                },
                "right": {
                    "type": "binary",
                    "operator": "gte",
                    "left": _column("t", "confidence"),
                    "right": _literal(0.85),
                },
            },
            "order_by": [
                {"expr": _column("t", "confidence"), "direction": "desc"}
            ],
        },
    }


@pytest.mark.parametrize(
    ("question", "expected_sql", "wrong_plan"),
    (
        (AGE_QUESTION, AGE_SQL, _wrong_age_plan()),
        (DEPARTMENT_QUESTION, DEPARTMENT_SQL, _wrong_department_plan()),
        (TYPED_KG_QUESTION, TYPED_KG_SQL, _wrong_typed_kg_plan()),
    ),
)
def test_plan_first_executable_but_incomplete_plan_falls_back_to_closed_contract(
    synthetic_closed_contract_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    question: str,
    expected_sql: str,
    wrong_plan: dict[str, object],
) -> None:
    monkeypatch.setattr(llm_module, "llm_available", lambda: True)
    monkeypatch.setattr(
        llm_module,
        "chat",
        lambda *_args, **_kwargs: json.dumps(wrong_plan, ensure_ascii=False),
    )
    nl = NL2SQL(
        synthetic_closed_contract_db,
        llm_strategy="query_plan_v2",
        agent_rule_policy="plan_first",
        llm_rounds=2,
        llm_fail_closed=True,
    )

    result = nl.to_sql(question)

    assert result["engine"] == "rule"
    assert result["sql"] == expected_sql
    assert result["plan"]["route"] == "rule_after_query_plan_abstain"
    assert result["llm_meta"]["failure_stages"] == (
        "semantic_contract",
        "semantic_contract",
    )
