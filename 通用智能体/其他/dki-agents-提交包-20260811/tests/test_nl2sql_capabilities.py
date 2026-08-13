"""Tests for deterministic NL2SQL semantic capability classification."""

from dki.analysis.nl2sql_capabilities import (
    CAPABILITY_ORDER,
    classify_nl2sql_capabilities,
    classify_nl2sql_capability_flags,
    normalise_capabilities,
)


def test_question_classification_covers_cross_table_distinct_grain_and_contract():
    capabilities = classify_nl2sql_capabilities(
        "返回每个科室不同患者数，按人数降序列出。"
    )

    assert capabilities == tuple(
        name
        for name in CAPABILITY_ORDER
        if name
        in {
            "schema_linking",
            "grain",
            "joins_cardinality",
            "count_distinct",
            "aggregation_stage",
            "output_contract",
        }
    )


def test_sql_ast_adds_structural_capabilities_deterministically():
    sql = """
        WITH per_patient AS (
            SELECT p.patient_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY p.patient_id
                       ORDER BY v.visit_date DESC
                   ) AS visit_rank
            FROM patients AS p
            LEFT JOIN visits AS v ON v.patient_id = p.patient_id
        )
        SELECT patient_id
        FROM per_patient
        WHERE visit_rank = 1 OR visit_rank IS NULL
        UNION
        SELECT patient_id FROM patients WHERE patient_id IS NULL
    """

    flags = classify_nl2sql_capability_flags("返回患者编号", sql)

    assert flags["schema_linking"] is True
    assert flags["grain"] is True
    assert flags["joins_cardinality"] is True
    assert flags["preservation"] is True
    assert flags["cte"] is True
    assert flags["set_operation"] is True
    assert flags["window"] is True
    assert flags["date"] is True
    assert flags["null"] is True
    assert flags["output_contract"] is True


def test_correlated_ties_and_date_are_detected_from_question_and_ast():
    question = (
        "截至2025-03-01，返回每个科室费用等于该科室最高费用的就诊，"
        "并列全部保留。"
    )
    sql = """
        SELECT v.visit_id, v.department, v.cost
        FROM visits AS v
        WHERE v.visit_date <= '2025-03-01'
          AND v.cost = (
              SELECT MAX(v2.cost)
              FROM visits AS v2
              WHERE v2.department = v.department
          )
    """

    capabilities = set(classify_nl2sql_capabilities(question, sql))

    assert {
        "schema_linking",
        "grain",
        "aggregation_stage",
        "correlated",
        "ties",
        "date",
        "output_contract",
    } <= capabilities


def test_explicit_capability_metadata_is_normalised_without_losing_extensions():
    assert normalise_capabilities(
        {
            "output contract": True,
            "count-distinct": True,
            "custom capability": True,
            "cte": False,
        }
    ) == ("count_distinct", "output_contract", "custom_capability")
