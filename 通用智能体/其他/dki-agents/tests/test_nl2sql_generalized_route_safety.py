"""Regression coverage for fail-closed generalized NL2SQL root routing."""
from __future__ import annotations

import pytest

import dki.analysis.nl2sql as nl2sql_module
from dki.analysis.nl2sql import NL2SQL
from dki.analysis.nl2sql_compile_result import CompileDisposition
from dki.analysis.nl2sql_contract import extract_semantic_contract
from dki.analysis.nl2sql_semantic_obligations import extract_semantic_obligations


GENERALIZED_COMPILERS = (
    "compile_independent_v11_sql",
    "compile_independent_v10_sql",
    "compile_independent_v9_sql",
    "compile_independent_v8_sql",
    "compile_independent_v7_sql",
)

DOMAINS = {
    "department": [],
    "disease": [],
    "region": ["深圳"],
    "patient_name": [],
    "patient_id": [],
    "visit_id": [],
    "drug": [],
    "exam": [],
    "kg_entity": [],
    "kg_entity_type": [],
    "kg_relation": [],
    "kg_head_type": [],
    "kg_relation_zh": [],
}


def _bare_engine() -> NL2SQL:
    engine = object.__new__(NL2SQL)
    engine._domains = {key: list(values) for key, values in DOMAINS.items()}
    return engine


def _route_generalized_candidate(
    monkeypatch,
    *,
    compiler_name: str,
    question: str,
    candidate: str,
):
    """Install one matched generalized candidate and expose any fallthrough."""

    engine = _bare_engine()
    calls: list[str] = []
    downstream_calls: list[str] = []
    monkeypatch.setattr(
        nl2sql_module,
        "compile_independent_structured_sql",
        lambda _question, _domains: None,
    )
    monkeypatch.setattr(
        engine,
        "_full_width_relation_union_owner_result",
        lambda _question, **_kwargs: None,
    )

    for name in GENERALIZED_COMPILERS:
        returned = candidate if name == compiler_name else None

        def compiler(_question, _domains, *, _name=name, _returned=returned):
            calls.append(_name)
            return _returned

        monkeypatch.setattr(nl2sql_module, name, compiler)

    def downstream(_question, _domains):
        downstream_calls.append("relation")
        raise RuntimeError("rejected generalized candidate fell through")

    monkeypatch.setattr(nl2sql_module, "analyze_relation_request", downstream)
    result = engine._rule_sql_normalized_result(question)
    return result, calls, downstream_calls


def test_real_date_topk_never_drops_explicit_bounds():
    engine = _bare_engine()
    question = (
        "从2024-07-20到2024-10-16期间，"
        "返回就诊次数最多的前2个科室及其就诊次数。"
    )

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition in {
        CompileDisposition.SQL,
        CompileDisposition.BLOCKED,
    }
    if result.disposition is CompileDisposition.SQL:
        assert result.sql is not None
        assert "2024-07-20" in result.sql
        assert "2024-10-16" in result.sql


def test_real_qualified_global_age_is_reserved_before_generalized_compilers(
    monkeypatch,
):
    engine = _bare_engine()
    calls: list[str] = []
    question = (
        "找出深圳地区年龄至少40岁的女性患者，年龄高于全部患者平均年龄，"
        "返回患者ID、姓名、年龄和地区；按年龄降序、患者ID升序排列。"
    )

    for name in GENERALIZED_COMPILERS:
        monkeypatch.setattr(
            nl2sql_module,
            name,
            lambda _question, _domains, *, _name=name: calls.append(_name),
        )

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.BLOCKED
    assert calls == []


@pytest.mark.parametrize(
    "compiler_name",
    GENERALIZED_COMPILERS,
    ids=("v11", "v10", "v9", "v8", "v7"),
)
def test_each_generalized_candidate_passes_ast_contract_and_obligation_gate(
    monkeypatch,
    compiler_name,
):
    result, calls, downstream_calls = _route_generalized_candidate(
        monkeypatch,
        compiler_name=compiler_name,
        question="返回患者编号、姓名和年龄。",
        candidate="SELECT * FROM patients",
    )

    target_index = GENERALIZED_COMPILERS.index(compiler_name)
    assert result.disposition is CompileDisposition.BLOCKED
    assert calls == list(GENERALIZED_COMPILERS[: target_index + 1])
    assert downstream_calls == []


@pytest.mark.parametrize(
    ("question", "candidate"),
    (
        pytest.param(
            "从2024-07-20到2024-10-16期间，"
            "返回就诊次数最多的前2个科室及其就诊次数。",
            "SELECT department, COUNT(*) AS visit_count FROM visits "
            "WHERE visit_date >= '2024-07-20' GROUP BY department "
            "ORDER BY visit_count DESC, department LIMIT 2",
            id="missing-date",
        ),
        pytest.param(
            "返回患者编号、姓名和年龄。",
            "SELECT patient_id, name FROM patients ORDER BY patient_id",
            id="missing-projection",
        ),
    ),
)
def test_generalized_rejection_is_terminal(
    monkeypatch,
    question,
    candidate,
):
    result, calls, downstream_calls = _route_generalized_candidate(
        monkeypatch,
        compiler_name="compile_independent_v11_sql",
        question=question,
        candidate=candidate,
    )

    assert result.disposition is CompileDisposition.BLOCKED
    assert calls == ["compile_independent_v11_sql"]
    assert downstream_calls == []


def test_self_validated_owner_cannot_bypass_reliable_contract():
    engine = _bare_engine()
    question = "返回患者编号、姓名和年龄。"
    contract = extract_semantic_contract(question)
    obligations = extract_semantic_obligations(question, engine._domains)

    result = engine._owned_candidate_result(
        question,
        contract,
        "SELECT patient_id FROM patients",
        obligations=obligations,
        contract_self_validated=True,
    )

    assert result.disposition is CompileDisposition.BLOCKED


def test_real_v9_candidate_can_use_recomputed_provenance_after_generic_false_negative():
    engine = _bare_engine()
    question = (
        "在每个地区内按年龄降序给患者做连续排名，"
        "列出地区、患者编号、年龄和排名。"
    )

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.SQL
    assert result.sql is not None
    assert "DENSE_RANK()" in result.sql
    assert result.fallback_fenced


def test_v9_provenance_refuses_an_unconsumed_output_override():
    engine = _bare_engine()
    question = (
        "在每个地区内按年龄降序给患者做连续排名，"
        "列出地区、患者编号、年龄和排名；但只返回患者编号。"
    )

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.BLOCKED


def test_real_v9_three_column_global_age_owner_is_not_lost_to_legacy_shape():
    engine = _bare_engine()
    question = "列出年龄高于全部患者平均年龄的患者编号、姓名和年龄。"

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.SQL
    assert result.sql == (
        "SELECT patient_id,name,age FROM patients WHERE age>(SELECT AVG(age) "
        "FROM patients) ORDER BY age DESC"
    )
    assert result.fallback_fenced
