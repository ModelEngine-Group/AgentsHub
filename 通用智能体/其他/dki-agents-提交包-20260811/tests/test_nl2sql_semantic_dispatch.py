"""Focused tests for contract-gated semantic NL2SQL dispatch."""
from __future__ import annotations

import pytest

import dki.analysis.nl2sql as nl2sql_module
from dki.analysis.nl2sql import NL2SQL
from dki.analysis.nl2sql_age_average_intent import (
    AgeAverageDisposition,
    AgeAverageIntent,
    classify_age_average_intent,
)
from dki.analysis.nl2sql_compile_result import (
    CompileDisposition,
    CompileResult,
)
from dki.analysis.nl2sql_compositional import compile_compositional_medical_sql
from dki.analysis.nl2sql_contract import (
    ContractStatus,
    SemanticContract,
    ValidationResult,
    extract_semantic_contract,
)
from dki.analysis.nl2sql_semantic import (
    compile_semantic_medical_result,
    compile_semantic_medical_sql,
)
from dki.analysis.nl2sql_semantic_simple import compile_simple_medical_sql
from dki.analysis.nl2sql_temporal_cohort_intent import (
    TemporalCohortDisposition,
    TemporalCohortIntent,
    TemporalCohortKind,
    classify_temporal_cohort_intent,
)


def _bare_engine() -> NL2SQL:
    engine = object.__new__(NL2SQL)
    engine._domains = {}
    return engine


def _disable_existing_routes(monkeypatch, engine: NL2SQL) -> None:
    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        lambda _question, _domains: None,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_explicit_medical_sql",
        lambda _question, _domains, **_kwargs: None,
    )
    monkeypatch.setattr(engine, "_rule_structured_medical_analytics", lambda _q: None)
    monkeypatch.setattr(engine, "_rule_advanced", lambda _q: None)
    monkeypatch.setattr(engine, "_needs_llm", lambda _q: False)


def test_semantic_dispatch_continues_after_callback_rejection(monkeypatch):
    import dki.analysis.nl2sql_semantic_advanced as advanced_module
    import dki.analysis.nl2sql_semantic_kg as kg_module
    import dki.analysis.nl2sql_semantic_simple as simple_module

    advanced = "SELECT patient_id FROM patients"
    kg = "SELECT name FROM patients"
    simple = "SELECT patient_id, name, age FROM patients"
    monkeypatch.setattr(
        advanced_module,
        "compile_advanced_medical_result",
        lambda _question, _domains, **_kwargs: CompileResult(
            CompileDisposition.SQL,
            advanced,
        ),
    )
    monkeypatch.setattr(
        kg_module,
        "compile_kg_medical_sql",
        lambda _question, _domains: kg,
    )
    monkeypatch.setattr(
        simple_module,
        "compile_simple_medical_sql",
        lambda _question, _domains: simple,
    )

    seen: list[str] = []

    def accept(candidate: str) -> bool:
        seen.append(candidate)
        return candidate == simple

    assert compile_semantic_medical_sql(
        "synthetic question",
        {},
        accept_candidate=accept,
    ) == simple
    assert seen == [advanced, kg, simple]


@pytest.mark.parametrize(
    ("question", "required_sql"),
    [
        ("按性别统计就诊量", "GROUP BY gender"),
        ("按年龄段统计就诊量", "CASE WHEN p.age <= 18"),
        ("提及次数最多的疾病有哪些", "FROM kg_entities WHERE type = 'Disease'"),
    ],
)
def test_compact_public_shapes_reach_contract_compatible_lowerer(
    question,
    required_sql,
    monkeypatch,
):
    engine = _bare_engine()
    engine._domains = {
        "department": [],
        "disease": [],
        "drug": [],
        "exam": [],
        "region": [],
        "kg_entity": [],
        "kg_relation_zh": [],
    }
    # This regression exercises the root semantic-to-legacy hand-off.  Public
    # development compilers are unrelated earlier candidates and expect the
    # full database-loaded domain map, which a deliberately bare unit engine
    # does not provide.
    monkeypatch.setattr(engine, "_rule_structured_medical_analytics", lambda _q: None)
    monkeypatch.setattr(engine, "_rule_advanced", lambda _q: None)
    monkeypatch.setattr(engine, "_needs_llm", lambda _q: False)

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.SQL
    assert result.sql is not None
    assert required_sql in result.sql


def test_root_classifies_nonexclusive_intents_once_and_forwards_them(monkeypatch):
    """Root and semantic dispatch must share one immutable intent record."""

    engine = _bare_engine()
    question = "synthetic ordinary question"
    age_intent = AgeAverageIntent(AgeAverageDisposition.NOT_APPLICABLE)
    temporal_intent = TemporalCohortIntent(TemporalCohortDisposition.NOT_APPLICABLE)
    age_calls: list[str] = []
    temporal_calls: list[str] = []
    owners: list[tuple[AgeAverageIntent, TemporalCohortIntent]] = []
    seen: list[tuple[AgeAverageIntent, TemporalCohortIntent | None]] = []

    monkeypatch.setattr(
        nl2sql_module,
        "classify_age_average_intent",
        lambda question, _domains: age_calls.append(question) or age_intent,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "classify_temporal_cohort_intent",
        lambda question, _domains: temporal_calls.append(question) or temporal_intent,
    )
    _disable_existing_routes(monkeypatch, engine)

    def exclusive(
        _question,
        *,
        age_average_intent,
        temporal_cohort_intent,
        obligations,
    ):
        owners.append((age_average_intent, temporal_cohort_intent))
        assert obligations is not None
        return None

    monkeypatch.setattr(engine, "_exclusive_owner_result", exclusive)

    def semantic(_question, _domains, **kwargs):
        seen.append(
            (
                kwargs.get("age_average_intent"),
                kwargs.get("temporal_cohort_intent"),
            ),
        )
        return CompileResult(CompileDisposition.BLOCKED)

    monkeypatch.setattr(nl2sql_module, "compile_semantic_medical_result", semantic)

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.BLOCKED
    assert age_calls == [question]
    assert temporal_calls == [question]
    assert owners == [(age_intent, temporal_intent)]
    assert seen == [(age_intent, temporal_intent)]


def test_semantic_blocked_prevents_legacy_and_all_narrow_fallbacks(monkeypatch):
    engine = _bare_engine()
    question = "synthetic terminal advanced request"
    calls: list[str] = []

    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_explicit_medical_sql",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(engine, "_rule_structured_medical_analytics", lambda _q: None)

    def semantic(*_args, **_kwargs):
        calls.append("semantic")
        return CompileResult(CompileDisposition.BLOCKED)

    def unexpected(name):
        def fail(*_args, **_kwargs):
            raise AssertionError(f"{name} ran after semantic BLOCKED")

        return fail

    monkeypatch.setattr(nl2sql_module, "compile_semantic_medical_result", semantic)
    monkeypatch.setattr(engine, "_rule_advanced", unexpected("legacy advanced"))
    monkeypatch.setattr(engine, "_needs_llm", unexpected("LLM policy"))
    monkeypatch.setattr(engine, "_rule_kg", unexpected("legacy KG"))
    monkeypatch.setattr(engine, "_rule_visits", unexpected("visits"))

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.BLOCKED
    assert calls == ["semantic"]


def test_semantic_not_applicable_reaches_contract_gated_legacy(monkeypatch):
    engine = _bare_engine()
    question = "synthetic ordinary legacy request"
    expected = "SELECT patient_id FROM patients"
    calls: list[str] = []

    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_explicit_medical_sql",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(engine, "_rule_structured_medical_analytics", lambda _q: None)

    def semantic(*_args, **_kwargs):
        calls.append("semantic")
        return CompileResult(CompileDisposition.NOT_APPLICABLE)

    def legacy(_question):
        calls.append("legacy")
        return expected

    monkeypatch.setattr(nl2sql_module, "compile_semantic_medical_result", semantic)
    monkeypatch.setattr(engine, "_rule_advanced", legacy)
    monkeypatch.setattr(engine, "_needs_llm", lambda _q: False)
    monkeypatch.setattr(engine, "_rule_kg", lambda _q: None)
    monkeypatch.setattr(engine, "_rule_visits", lambda _q: None)

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.SQL
    assert result.sql == expected
    assert calls == ["semantic", "legacy"]


@pytest.mark.parametrize(
    "question",
    [
        "就诊量至少30次的诊断有哪几种",
        "平均花费不低于5000元的科室及其平均花费",
        "统计每个科室每种患者性别的记录数",
    ],
)
def test_recognized_group_shapes_block_after_candidate_rejection(question):
    result = compile_semantic_medical_result(
        question,
        {},
        accept_candidate=lambda _sql: False,
    )

    assert result.disposition is CompileDisposition.BLOCKED
    assert result.sql is None


def test_incomplete_early_qualifier_candidate_falls_through_to_complete_one(
    monkeypatch,
):
    engine = _bare_engine()
    question = "返回女性患者的患者编号"
    incomplete = "SELECT patient_id FROM patients"
    complete = "SELECT patient_id FROM patients WHERE gender='女'"
    semantic_calls: list[str] = []

    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        lambda *_args, **_kwargs: incomplete,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_explicit_medical_sql",
        lambda *_args, **_kwargs: complete,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        lambda *_args, **_kwargs: semantic_calls.append("semantic")
        or CompileResult(CompileDisposition.BLOCKED),
    )

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.SQL
    assert result.sql == complete
    assert semantic_calls == []


def test_unassigned_three_dates_are_terminal_before_any_candidate(monkeypatch):
    engine = _bare_engine()
    question = "比较2026-01-01、2026-02-01、2026-03-01的就诊"

    def unexpected(*_args, **_kwargs):
        raise AssertionError("a compiler ran after terminal date extraction")

    monkeypatch.setattr(engine, "_exclusive_owner_result", unexpected)
    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        unexpected,
    )
    monkeypatch.setattr(nl2sql_module, "compile_explicit_medical_sql", unexpected)
    monkeypatch.setattr(nl2sql_module, "compile_semantic_medical_result", unexpected)

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.BLOCKED


def test_declined_kg_relation_owner_blocks_simple_fallback(monkeypatch):
    import dki.analysis.nl2sql_semantic_advanced as advanced_module
    import dki.analysis.nl2sql_semantic_kg as kg_module
    import dki.analysis.nl2sql_semantic_simple as simple_module

    candidate = "SELECT relation_zh FROM kg_triples"
    attempted: list[str] = []
    monkeypatch.setattr(
        advanced_module,
        "compile_advanced_medical_result",
        lambda *_args, **_kwargs: CompileResult(CompileDisposition.NOT_APPLICABLE),
    )
    monkeypatch.setattr(
        kg_module,
        "compile_kg_medical_sql",
        lambda _question, _domains: candidate,
    )

    def terminal_relation_owner(_question, _domains):
        assert attempted == [candidate]
        return CompileDisposition.BLOCKED

    monkeypatch.setattr(
        kg_module,
        "relation_fallback_disposition",
        terminal_relation_owner,
    )
    monkeypatch.setattr(
        simple_module,
        "compile_simple_medical_sql",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("unexpected simple fallback"),
        ),
    )

    result = compile_semantic_medical_result(
        "synthetic structured relation request",
        {},
        accept_candidate=lambda sql: attempted.append(sql) and False,
    )

    assert result.disposition is CompileDisposition.BLOCKED


def test_relation_semantic_dispatch_runs_kg_before_advanced(monkeypatch):
    import dki.analysis.nl2sql_semantic_advanced as advanced_module
    import dki.analysis.nl2sql_semantic_kg as kg_module
    import dki.analysis.nl2sql_semantic_simple as simple_module

    question = "呈现症状关系的三元组。"
    domains = {"kg_relation_zh": ["呈现症状"]}
    kg_sql = "SELECT head FROM kg_triples WHERE relation_zh = '呈现症状'"

    monkeypatch.setattr(
        advanced_module,
        "compile_advanced_medical_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("relation reached generic advanced compiler"),
        ),
    )
    monkeypatch.setattr(
        kg_module,
        "compile_kg_medical_sql",
        lambda _question, _domains: kg_sql,
    )
    monkeypatch.setattr(
        simple_module,
        "compile_simple_medical_sql",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("accepted KG SQL reached simple compiler"),
        ),
    )

    result = compile_semantic_medical_result(
        question,
        domains,
        accept_candidate=lambda sql: sql == kg_sql,
    )

    assert result.disposition is CompileDisposition.SQL
    assert result.sql == kg_sql
    assert result.fallback_fenced


def test_relation_family_runs_after_kg_decline_before_terminal_fallback(monkeypatch):
    import dki.analysis.nl2sql_semantic_advanced as advanced_module
    import dki.analysis.nl2sql_semantic_families as families_module
    import dki.analysis.nl2sql_semantic_kg as kg_module
    import dki.analysis.nl2sql_semantic_simple as simple_module

    question = "呈现症状关系的尾实体提及次数高于同类型平均。"
    domains = {"kg_relation_zh": ["呈现症状"]}
    family_sql = "SELECT tail FROM kg_triples WHERE relation_zh = '呈现症状'"
    calls: list[str] = []
    monkeypatch.setattr(
        advanced_module,
        "compile_advanced_medical_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("relation reached generic advanced compiler"),
        ),
    )
    monkeypatch.setattr(
        kg_module,
        "compile_kg_medical_sql",
        lambda _question, _domains: calls.append("kg") or None,
    )
    monkeypatch.setattr(
        families_module,
        "compile_semantic_family_sql",
        lambda _question, _domains: calls.append("family") or family_sql,
    )
    monkeypatch.setattr(
        kg_module,
        "relation_fallback_disposition",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("accepted family candidate reached terminal fallback"),
        ),
    )
    monkeypatch.setattr(
        simple_module,
        "compile_simple_medical_sql",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("accepted family candidate reached simple compiler"),
        ),
    )

    result = compile_semantic_medical_result(question, domains)

    assert result.disposition is CompileDisposition.SQL
    assert result.sql == family_sql
    assert result.fallback_fenced
    assert calls == ["kg", "family"]


def test_relation_family_accepts_metric_label_repeat_after_kg_decline(monkeypatch):
    import dki.analysis.nl2sql_semantic_families as families_module
    import dki.analysis.nl2sql_semantic_kg as kg_module
    import dki.analysis.nl2sql_semantic_simple as simple_module

    question = "呈现症状条件计数覆盖审计，呈现症状数至少2。"
    domains = {"kg_relation_zh": ["呈现症状"]}
    family_sql = "SELECT head FROM kg_triples WHERE relation_zh = '呈现症状'"
    calls: list[str] = []
    monkeypatch.setattr(
        kg_module,
        "compile_kg_medical_sql",
        lambda *_args: calls.append("kg") or None,
    )
    monkeypatch.setattr(
        families_module,
        "compile_semantic_family_sql",
        lambda *_args: calls.append("family") or family_sql,
    )
    monkeypatch.setattr(
        kg_module,
        "relation_fallback_disposition",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("qualified repeat reached terminal fallback"),
        ),
    )
    monkeypatch.setattr(
        simple_module,
        "compile_simple_medical_sql",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("qualified repeat reached simple compiler"),
        ),
    )

    result = compile_semantic_medical_result(question, domains)

    assert result.disposition is CompileDisposition.SQL
    assert result.sql == family_sql
    assert result.fallback_fenced
    assert calls == ["kg", "family"]


def test_relation_family_accepts_multiple_metric_label_repeats_after_kg_decline(monkeypatch):
    import dki.analysis.nl2sql_semantic_families as families_module
    import dki.analysis.nl2sql_semantic_kg as kg_module
    import dki.analysis.nl2sql_semantic_simple as simple_module

    question = (
        "并发症与用药禁忌风险链，"
        "并发症关系置信度至少0.7且用药禁忌关系置信度至少0.8。"
    )
    domains = {"kg_relation_zh": ["并发症", "用药禁忌"]}
    family_sql = (
        "SELECT head FROM kg_triples WHERE relation_zh IN ('并发症', '用药禁忌')"
    )
    calls: list[str] = []
    monkeypatch.setattr(
        kg_module,
        "compile_kg_medical_sql",
        lambda *_args: calls.append("kg") or None,
    )
    monkeypatch.setattr(
        families_module,
        "compile_semantic_family_sql",
        lambda *_args: calls.append("family") or family_sql,
    )
    monkeypatch.setattr(
        kg_module,
        "relation_fallback_disposition",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("qualified multi-role repeat reached terminal fallback"),
        ),
    )
    monkeypatch.setattr(
        simple_module,
        "compile_simple_medical_sql",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("qualified multi-role repeat reached simple compiler"),
        ),
    )

    result = compile_semantic_medical_result(question, domains)

    assert result.disposition is CompileDisposition.SQL
    assert result.sql == family_sql
    assert result.fallback_fenced
    assert calls == ["kg", "family"]


def test_relation_family_rejects_unqualified_distinct_list_before_fallback(monkeypatch):
    import dki.analysis.nl2sql_semantic_families as families_module
    import dki.analysis.nl2sql_semantic_kg as kg_module
    import dki.analysis.nl2sql_semantic_simple as simple_module

    question = "呈现症状和接受治疗关系的三元组。"
    domains = {"kg_relation_zh": ["呈现症状", "接受治疗"]}
    monkeypatch.setattr(kg_module, "compile_kg_medical_sql", lambda *_args: None)
    monkeypatch.setattr(
        families_module,
        "compile_semantic_family_sql",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("bare relation list reached family lowering"),
        ),
    )
    monkeypatch.setattr(
        kg_module,
        "relation_fallback_disposition",
        lambda *_args: CompileDisposition.BLOCKED,
    )
    monkeypatch.setattr(
        simple_module,
        "compile_simple_medical_sql",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("bare relation list reached simple compiler"),
        ),
    )

    result = compile_semantic_medical_result(question, domains)

    assert result.disposition is CompileDisposition.BLOCKED


def test_relation_family_rejects_unqualified_repeat_before_fallback(monkeypatch):
    import dki.analysis.nl2sql_semantic_families as families_module
    import dki.analysis.nl2sql_semantic_kg as kg_module
    import dki.analysis.nl2sql_semantic_simple as simple_module

    question = "呈现症状和呈现症状关系的三元组。"
    domains = {"kg_relation_zh": ["呈现症状"]}
    monkeypatch.setattr(kg_module, "compile_kg_medical_sql", lambda *_args: None)
    monkeypatch.setattr(
        families_module,
        "compile_semantic_family_sql",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("unqualified repeat reached family lowering"),
        ),
    )
    monkeypatch.setattr(
        kg_module,
        "relation_fallback_disposition",
        lambda *_args: CompileDisposition.BLOCKED,
    )
    monkeypatch.setattr(
        simple_module,
        "compile_simple_medical_sql",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("unqualified repeat reached simple compiler"),
        ),
    )

    result = compile_semantic_medical_result(question, domains)

    assert result.disposition is CompileDisposition.BLOCKED


def test_rejected_relation_family_candidate_blocks_generic_fallback(monkeypatch):
    import dki.analysis.nl2sql_semantic_families as families_module
    import dki.analysis.nl2sql_semantic_kg as kg_module
    import dki.analysis.nl2sql_semantic_simple as simple_module

    question = "呈现症状关系的尾实体提及次数高于同类型平均。"
    domains = {"kg_relation_zh": ["呈现症状"]}
    family_sql = "SELECT tail FROM kg_triples WHERE relation_zh = '呈现症状'"
    monkeypatch.setattr(kg_module, "compile_kg_medical_sql", lambda *_args: None)
    monkeypatch.setattr(
        families_module,
        "compile_semantic_family_sql",
        lambda *_args: family_sql,
    )
    monkeypatch.setattr(
        kg_module,
        "relation_fallback_disposition",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("rejected family candidate reached terminal fallback"),
        ),
    )
    monkeypatch.setattr(
        simple_module,
        "compile_simple_medical_sql",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("rejected family candidate reached simple compiler"),
        ),
    )

    result = compile_semantic_medical_result(
        question,
        domains,
        accept_candidate=lambda _sql: False,
    )

    assert result.disposition is CompileDisposition.BLOCKED


def test_non_graph_relation_family_candidate_blocks_generic_fallback(monkeypatch):
    import dki.analysis.nl2sql_semantic_families as families_module
    import dki.analysis.nl2sql_semantic_kg as kg_module
    import dki.analysis.nl2sql_semantic_simple as simple_module

    question = "呈现症状关系的尾实体提及次数高于同类型平均。"
    domains = {"kg_relation_zh": ["呈现症状"]}
    monkeypatch.setattr(kg_module, "compile_kg_medical_sql", lambda *_args: None)
    monkeypatch.setattr(
        families_module,
        "compile_semantic_family_sql",
        lambda *_args: "SELECT patient_id FROM patients",
    )
    monkeypatch.setattr(
        kg_module,
        "relation_fallback_disposition",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("non-graph family candidate reached terminal fallback"),
        ),
    )
    monkeypatch.setattr(
        simple_module,
        "compile_simple_medical_sql",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("non-graph family candidate reached simple compiler"),
        ),
    )

    result = compile_semantic_medical_result(question, domains)

    assert result.disposition is CompileDisposition.BLOCKED


def test_plain_relation_semantic_decline_skips_row_compilers(monkeypatch):
    import dki.analysis.nl2sql_semantic_advanced as advanced_module
    import dki.analysis.nl2sql_semantic_kg as kg_module
    import dki.analysis.nl2sql_semantic_simple as simple_module

    question = "呈现症状关系。"
    domains = {"kg_relation_zh": ["呈现症状"]}
    monkeypatch.setattr(
        advanced_module,
        "compile_advanced_medical_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("plain relation reached generic advanced compiler"),
        ),
    )
    monkeypatch.setattr(kg_module, "compile_kg_medical_sql", lambda *_args: None)
    monkeypatch.setattr(
        kg_module,
        "relation_fallback_disposition",
        lambda _question, _domains: CompileDisposition.NOT_APPLICABLE,
    )
    monkeypatch.setattr(
        simple_module,
        "compile_simple_medical_sql",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("plain relation reached row compiler"),
        ),
    )

    result = compile_semantic_medical_result(question, domains)

    assert result.disposition is CompileDisposition.NOT_APPLICABLE


def test_existing_exact_compiler_stays_ahead_of_semantic(monkeypatch):
    engine = _bare_engine()
    expected = "SELECT patient_id FROM patients"
    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        lambda _question, _domains: expected,
    )

    def semantic_should_not_run(*_args, **_kwargs):
        raise AssertionError("semantic dispatch ran before an exact compiler")

    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        semantic_should_not_run,
    )

    assert engine._rule_sql_normalized("返回患者编号") == expected


def test_explicit_compiler_stays_ahead_of_semantic(monkeypatch):
    engine = _bare_engine()
    expected = "SELECT visit_id FROM visits"
    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        lambda _question, _domains: None,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_explicit_medical_sql",
        lambda _question, _domains, **_kwargs: expected,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("semantic dispatch ran before explicit compiler"),
        ),
    )

    assert engine._rule_sql_normalized("返回就诊编号") == expected


def test_curated_compiler_stays_ahead_of_semantic(monkeypatch):
    engine = _bare_engine()
    expected = "SELECT department FROM visits"
    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        lambda _question, _domains: None,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_explicit_medical_sql",
        lambda _question, _domains, **_kwargs: None,
    )
    monkeypatch.setattr(engine, "_rule_structured_medical_analytics", lambda _q: expected)
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("semantic dispatch ran before curated compiler"),
        ),
    )

    assert engine._rule_sql_normalized("返回科室") == expected


def test_semantic_decline_preserves_legacy_advanced_fallback(monkeypatch):
    engine = _bare_engine()
    expected = "SELECT COUNT(*) FROM visits"
    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        lambda _question, _domains: None,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_explicit_medical_sql",
        lambda _question, _domains, **_kwargs: None,
    )
    monkeypatch.setattr(engine, "_rule_structured_medical_analytics", lambda _q: None)
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        lambda *_args, **_kwargs: CompileResult(CompileDisposition.NOT_APPLICABLE),
    )
    monkeypatch.setattr(engine, "_rule_advanced", lambda _q: expected)

    assert engine._rule_sql_normalized("统计就诊总数") == expected


def test_patient_visit_count_threshold_uses_correlated_count_before_list_fallback():
    """A count qualifier cannot degrade to a DISTINCT patient list."""

    sql = compile_compositional_medical_sql(
        "列出内科就诊次数至少2次的患者，返回患者编号。",
        {"department": ["内科"]},
    )

    assert sql is not None
    assert "v.department = '内科'" in sql
    assert "(SELECT COUNT(*) AS visit_count FROM visits AS v" in sql
    assert ") >= 2" in sql
    assert "SELECT DISTINCT" not in sql


def test_semantic_candidates_are_validated_and_returned_unchanged(monkeypatch):
    engine = _bare_engine()
    _disable_existing_routes(monkeypatch, engine)
    rejected = "SELECT patient_id FROM patients"
    accepted = "SELECT patient_id, name, age FROM patients"

    def semantic(_question, _domains, *, accept_candidate, **_kwargs):
        for candidate in (rejected, accepted):
            if accept_candidate(candidate):
                return CompileResult(CompileDisposition.SQL, candidate)
        return CompileResult(CompileDisposition.NOT_APPLICABLE)

    monkeypatch.setattr(nl2sql_module, "compile_semantic_medical_result", semantic)
    monkeypatch.setattr(engine, "_rule_kg", lambda _q: None)
    monkeypatch.setattr(engine, "_rule_visits", lambda _q: None)

    assert engine._rule_sql_normalized("返回患者编号、姓名和年龄") == accepted


def test_semantic_none_contract_uses_ast_gate(monkeypatch):
    """An ordinary empty contract still validates semantic SQL structurally."""

    engine = _bare_engine()
    _disable_existing_routes(monkeypatch, engine)
    expected = "SELECT patient_id FROM patients"
    observed: list[bool] = []
    monkeypatch.setattr(
        nl2sql_module,
        "extract_semantic_contract",
        lambda _question: SemanticContract(),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "_candidate_matches_contract",
        lambda _contract, _sql, *, enforce_semantics: observed.append(
            enforce_semantics,
        )
        or True,
    )

    def semantic(_question, _domains, *, accept_candidate, **_kwargs):
        return (
            CompileResult(CompileDisposition.SQL, expected)
            if accept_candidate(expected)
            else CompileResult(CompileDisposition.NOT_APPLICABLE)
        )

    monkeypatch.setattr(nl2sql_module, "compile_semantic_medical_result", semantic)

    assert engine._rule_sql_normalized("synthetic ordinary question") == expected
    assert observed == [False]


def test_rejected_legacy_candidate_allows_later_legacy_candidate(monkeypatch):
    engine = _bare_engine()
    _disable_existing_routes(monkeypatch, engine)
    rejected = "SELECT patient_id FROM patients"
    accepted = "SELECT patient_id, name, age FROM patients"
    calls: list[str] = []
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        lambda *_args, **_kwargs: CompileResult(CompileDisposition.NOT_APPLICABLE),
    )
    monkeypatch.setattr(
        engine,
        "_rule_kg",
        lambda _q: calls.append("kg") or rejected,
    )
    monkeypatch.setattr(
        engine,
        "_rule_visits",
        lambda _q: calls.append("visits") or accepted,
    )

    assert engine._rule_sql_normalized("返回患者编号、姓名和年龄") == accepted
    assert calls == ["kg", "visits"]


def test_contract_relevant_rejections_abstain(monkeypatch):
    engine = _bare_engine()
    _disable_existing_routes(monkeypatch, engine)
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        lambda *_args, **_kwargs: CompileResult(CompileDisposition.NOT_APPLICABLE),
    )
    monkeypatch.setattr(engine, "_rule_kg", lambda _q: "SELECT patient_id FROM patients")
    monkeypatch.setattr(engine, "_rule_visits", lambda _q: "SELECT name FROM patients")

    assert engine._rule_sql_normalized("返回患者编号、姓名和年龄") is None


def test_semantic_dispatch_precedes_hard_structure_llm_deferral(monkeypatch):
    engine = _bare_engine()
    _disable_existing_routes(monkeypatch, engine)
    expected = "SELECT patient_id FROM patients"
    monkeypatch.setattr(
        nl2sql_module,
        "extract_semantic_contract",
        lambda _question: SemanticContract(
            projections=("patient_id",),
            confidence=1.0,
            status=ContractStatus.COMPLETE,
        ),
    )
    monkeypatch.setattr(engine, "_needs_llm", lambda _q: True)

    def semantic(_question, _domains, *, accept_candidate, **_kwargs):
        return (
            CompileResult(CompileDisposition.SQL, expected)
            if accept_candidate(expected)
            else CompileResult(CompileDisposition.NOT_APPLICABLE)
        )

    monkeypatch.setattr(nl2sql_module, "compile_semantic_medical_result", semantic)
    monkeypatch.setattr(engine, "_rule_kg", lambda _q: None)
    monkeypatch.setattr(engine, "_rule_visits", lambda _q: None)

    assert engine._rule_sql_normalized("返回患者编号，费用高于平均费用") == expected


def test_reliable_partial_contract_uses_strict_validation(monkeypatch):
    contract = SemanticContract(
        projections=("patient_id",),
        confidence=0.6,
        status=ContractStatus.PARTIAL,
    )
    observed: list[bool] = []

    def validate(_contract, _sql, *, strict):
        observed.append(strict)
        return ValidationResult(True)

    monkeypatch.setattr(nl2sql_module, "validate_candidate", validate)

    assert nl2sql_module._candidate_matches_contract(
        contract,
        "SELECT patient_id FROM patients",
        enforce_semantics=True,
    )
    assert observed == [True]


def test_complete_implicit_contract_runs_before_llm_deferral(monkeypatch):
    engine = _bare_engine()
    _disable_existing_routes(monkeypatch, engine)
    question = "按科室统计就诊次数；按次数降序；前3个"
    contract = extract_semantic_contract(question)
    expected = (
        "SELECT department, COUNT(*) AS visit_count FROM visits "
        "GROUP BY department ORDER BY visit_count DESC LIMIT 3"
    )

    assert contract.status is ContractStatus.COMPLETE
    assert nl2sql_module._has_reliable_contract(question, contract)
    monkeypatch.setattr(engine, "_needs_llm", lambda _q: True)
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        lambda _question, _domains, *, accept_candidate, **_kwargs: (
            CompileResult(CompileDisposition.SQL, expected)
            if accept_candidate(expected)
            else CompileResult(CompileDisposition.NOT_APPLICABLE)
        ),
    )
    monkeypatch.setattr(engine, "_rule_kg", lambda _q: None)
    monkeypatch.setattr(engine, "_rule_visits", lambda _q: None)

    assert engine._rule_sql_normalized(question) == expected


def test_implicit_contract_rejection_cannot_downgrade_to_ast_only(monkeypatch):
    engine = _bare_engine()
    _disable_existing_routes(monkeypatch, engine)
    question = "按科室统计就诊次数；按次数降序；前3个"
    contract = extract_semantic_contract(question)

    assert contract.status is ContractStatus.COMPLETE
    assert not nl2sql_module._candidate_matches_contract(
        contract,
        "SELECT patient_id FROM patients",
        enforce_semantics=False,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        lambda *_args, **_kwargs: CompileResult(CompileDisposition.NOT_APPLICABLE),
    )
    monkeypatch.setattr(engine, "_rule_kg", lambda _q: "SELECT patient_id FROM patients")
    monkeypatch.setattr(engine, "_rule_visits", lambda _q: None)

    assert engine._rule_sql_normalized(question) is None


def test_none_contract_keeps_safe_legacy_compatibility(monkeypatch):
    engine = _bare_engine()
    _disable_existing_routes(monkeypatch, engine)
    expected = "SELECT tail FROM kg_triples"
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        lambda *_args, **_kwargs: CompileResult(CompileDisposition.NOT_APPLICABLE),
    )
    monkeypatch.setattr(engine, "_rule_kg", lambda _q: expected)
    monkeypatch.setattr(
        engine,
        "_rule_visits",
        lambda _q: (_ for _ in ()).throw(AssertionError("unexpected visits fallback")),
    )

    assert engine._rule_sql_normalized("高血压有哪些症状") == expected


def test_none_contract_keeps_the_read_only_ast_gate(monkeypatch):
    engine = _bare_engine()
    _disable_existing_routes(monkeypatch, engine)
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        lambda *_args, **_kwargs: CompileResult(CompileDisposition.NOT_APPLICABLE),
    )
    monkeypatch.setattr(engine, "_rule_kg", lambda _q: "DELETE FROM kg_triples")
    monkeypatch.setattr(engine, "_rule_visits", lambda _q: None)

    assert engine._rule_sql_normalized("高血压有哪些症状") is None


def test_compile_result_requires_an_explicit_disposition_check():
    result = CompileResult(CompileDisposition.SQL, "SELECT patient_id FROM patients")

    with pytest.raises(TypeError, match="disposition"):
        bool(result)
    with pytest.raises(ValueError, match="requires nonempty SQL"):
        CompileResult(CompileDisposition.SQL)
    with pytest.raises(ValueError, match="must not include SQL"):
        CompileResult(CompileDisposition.BLOCKED, "SELECT patient_id FROM patients")
    with pytest.raises(ValueError, match="only SQL dispositions"):
        CompileResult(CompileDisposition.BLOCKED, fallback_fenced=True)


def test_semantic_result_blocks_an_exclusive_candidate_rejection(monkeypatch):
    import dki.analysis.nl2sql_semantic_advanced as advanced_module
    import dki.analysis.nl2sql_semantic_kg as kg_module
    import dki.analysis.nl2sql_semantic_simple as simple_module

    intent = AgeAverageIntent(AgeAverageDisposition.ADVANCED_PEER)
    candidate = "SELECT patient_id, age FROM patients"
    monkeypatch.setattr(
        advanced_module,
        "compile_advanced_medical_result",
        lambda _question, _domains, **_kwargs: CompileResult(
            CompileDisposition.SQL,
            candidate,
        ),
    )
    monkeypatch.setattr(
        kg_module,
        "compile_kg_medical_sql",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected KG fallback")),
    )
    monkeypatch.setattr(
        simple_module,
        "compile_simple_medical_sql",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected simple fallback")),
    )

    result = compile_semantic_medical_result(
        "synthetic peer-age request",
        {},
        age_average_intent=intent,
        accept_candidate=lambda _sql: False,
    )

    assert result.disposition is CompileDisposition.BLOCKED


def test_semantic_result_blocks_an_exclusive_owner_miss(monkeypatch):
    import dki.analysis.nl2sql_semantic_advanced as advanced_module
    import dki.analysis.nl2sql_semantic_kg as kg_module
    import dki.analysis.nl2sql_semantic_simple as simple_module

    intent = AgeAverageIntent(AgeAverageDisposition.ADVANCED_PEER)
    monkeypatch.setattr(
        advanced_module,
        "compile_advanced_medical_result",
        lambda *_args, **_kwargs: CompileResult(CompileDisposition.NOT_APPLICABLE),
    )
    monkeypatch.setattr(
        kg_module,
        "compile_kg_medical_sql",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected KG fallback")),
    )
    monkeypatch.setattr(
        simple_module,
        "compile_simple_medical_sql",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected simple fallback")),
    )

    result = compile_semantic_medical_result(
        "synthetic peer-age request",
        {},
        age_average_intent=intent,
    )

    assert result.disposition is CompileDisposition.BLOCKED


def test_global_age_owner_runs_before_compositional_and_keeps_its_record(monkeypatch):
    engine = _bare_engine()
    question = "synthetic global age request"
    intent = AgeAverageIntent(AgeAverageDisposition.EXPLICIT_GLOBAL)
    temporal_intent = TemporalCohortIntent(TemporalCohortDisposition.NOT_APPLICABLE)
    expected = "SELECT patient_id, age FROM patients"
    seen: list[AgeAverageIntent] = []
    age_calls: list[str] = []
    temporal_calls: list[str] = []
    monkeypatch.setattr(
        nl2sql_module,
        "classify_age_average_intent",
        lambda value, _domains: age_calls.append(value) or intent,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "classify_temporal_cohort_intent",
        lambda value, _domains: temporal_calls.append(value) or temporal_intent,
    )

    def explicit(_question, _domains, *, age_average_intent, **_kwargs):
        seen.append(age_average_intent)
        return expected

    monkeypatch.setattr(nl2sql_module, "compile_explicit_medical_sql", explicit)
    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected compositional fallback")),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "_candidate_matches_contract",
        lambda *_args, **_kwargs: True,
    )

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.SQL
    assert result.sql == expected
    assert seen == [intent]
    assert age_calls == [question]
    assert temporal_calls == [question]


def test_global_age_owner_decline_blocks_every_broad_route(monkeypatch):
    engine = _bare_engine()
    intent = AgeAverageIntent(AgeAverageDisposition.EXPLICIT_GLOBAL)
    monkeypatch.setattr(
        nl2sql_module,
        "classify_age_average_intent",
        lambda _question, _domains: intent,
    )
    monkeypatch.setattr(nl2sql_module, "compile_explicit_medical_sql", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected compositional fallback")),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected semantic fallback")),
    )
    monkeypatch.setattr(
        engine,
        "_rule_structured_medical_analytics",
        lambda _q: (_ for _ in ()).throw(AssertionError("unexpected curated fallback")),
    )
    monkeypatch.setattr(
        engine,
        "_rule_advanced",
        lambda _q: (_ for _ in ()).throw(AssertionError("unexpected legacy advanced fallback")),
    )
    monkeypatch.setattr(
        engine,
        "_rule_kg",
        lambda _q: (_ for _ in ()).throw(AssertionError("unexpected KG fallback")),
    )
    monkeypatch.setattr(
        engine,
        "_rule_visits",
        lambda _q: (_ for _ in ()).throw(AssertionError("unexpected visit fallback")),
    )

    result = engine._rule_sql_normalized_result("synthetic global age request")

    assert result.disposition is CompileDisposition.BLOCKED


def test_age_terminal_abstention_blocks_before_any_broad_route(monkeypatch):
    engine = _bare_engine()
    question = "synthetic terminal age request"
    intent = AgeAverageIntent(AgeAverageDisposition.TERMINAL_ABSTAIN)
    temporal_intent = TemporalCohortIntent(TemporalCohortDisposition.NOT_APPLICABLE)
    age_calls: list[str] = []
    temporal_calls: list[str] = []
    monkeypatch.setattr(
        nl2sql_module,
        "classify_age_average_intent",
        lambda value, _domains: age_calls.append(value) or intent,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "classify_temporal_cohort_intent",
        lambda value, _domains: temporal_calls.append(value) or temporal_intent,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_explicit_medical_sql",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected explicit owner")),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_advanced_medical_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected advanced owner")),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected compositional fallback")),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected semantic fallback")),
    )
    monkeypatch.setattr(
        engine,
        "_rule_visits",
        lambda _q: (_ for _ in ()).throw(AssertionError("unexpected visit fallback")),
    )

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.BLOCKED
    assert age_calls == [question]
    assert temporal_calls == [question]


def test_peer_age_owner_candidate_rejection_cannot_reopen_fallbacks(monkeypatch):
    engine = _bare_engine()
    intent = AgeAverageIntent(AgeAverageDisposition.ADVANCED_PEER)
    candidate = "SELECT patient_id, age FROM patients"
    seen: list[AgeAverageIntent] = []
    monkeypatch.setattr(
        nl2sql_module,
        "classify_age_average_intent",
        lambda _question, _domains: intent,
    )

    def advanced(_question, _domains, *, age_average_intent=None, **_kwargs):
        seen.append(age_average_intent)
        return CompileResult(CompileDisposition.SQL, candidate)

    monkeypatch.setattr(nl2sql_module, "compile_advanced_medical_result", advanced)
    monkeypatch.setattr(
        nl2sql_module,
        "_candidate_matches_contract",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected compositional fallback")),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected semantic fallback")),
    )
    monkeypatch.setattr(
        engine,
        "_rule_visits",
        lambda _q: (_ for _ in ()).throw(AssertionError("unexpected visit fallback")),
    )

    result = engine._rule_sql_normalized_result("synthetic peer-age request")

    assert result.disposition is CompileDisposition.BLOCKED
    assert seen == [intent]


def test_temporal_owner_uses_the_shared_record_before_broad_routes(monkeypatch):
    engine = _bare_engine()
    age_intent = AgeAverageIntent(AgeAverageDisposition.NOT_APPLICABLE)
    intent = TemporalCohortIntent(
        TemporalCohortDisposition.ADVANCED_DATE_ONLY,
        kind=TemporalCohortKind.FUNNEL,
    )
    expected = "SELECT cohort_month FROM visits"
    seen: list[TemporalCohortIntent] = []
    seen_age: list[AgeAverageIntent] = []
    monkeypatch.setattr(
        nl2sql_module,
        "classify_age_average_intent",
        lambda _question, _domains: age_intent,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "classify_temporal_cohort_intent",
        lambda _question, _domains: intent,
    )

    def advanced(
        _question,
        _domains,
        *,
        age_average_intent=None,
        temporal_cohort_intent=None,
        **_kwargs,
    ):
        seen_age.append(age_average_intent)
        seen.append(temporal_cohort_intent)
        return CompileResult(CompileDisposition.SQL, expected)

    monkeypatch.setattr(nl2sql_module, "compile_advanced_medical_result", advanced)
    monkeypatch.setattr(
        nl2sql_module,
        "_candidate_matches_contract",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected compositional fallback")),
    )

    result = engine._rule_sql_normalized_result("synthetic temporal cohort request")

    assert result.disposition is CompileDisposition.SQL
    assert result.sql == expected
    assert result.fallback_fenced
    assert seen_age == [age_intent]
    assert seen == [intent]


def test_temporal_owner_candidate_rejection_cannot_reopen_fallbacks(monkeypatch):
    engine = _bare_engine()
    age_intent = AgeAverageIntent(AgeAverageDisposition.NOT_APPLICABLE)
    temporal_intent = TemporalCohortIntent(
        TemporalCohortDisposition.ADVANCED_DATE_ONLY,
        kind=TemporalCohortKind.FUNNEL,
    )
    candidate = "SELECT cohort_month FROM visits"
    monkeypatch.setattr(
        nl2sql_module,
        "classify_age_average_intent",
        lambda _question, _domains: age_intent,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "classify_temporal_cohort_intent",
        lambda _question, _domains: temporal_intent,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_advanced_medical_result",
        lambda *_args, **_kwargs: CompileResult(CompileDisposition.SQL, candidate),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "_candidate_matches_contract",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected compositional fallback")),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_explicit_medical_sql",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected explicit fallback")),
    )
    monkeypatch.setattr(
        engine,
        "_rule_structured_medical_analytics",
        lambda _q: (_ for _ in ()).throw(AssertionError("unexpected curated fallback")),
    )
    monkeypatch.setattr(
        engine,
        "_rule_advanced",
        lambda _q: (_ for _ in ()).throw(AssertionError("unexpected legacy advanced fallback")),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected semantic fallback")),
    )
    monkeypatch.setattr(
        engine,
        "_rule_kg",
        lambda _q: (_ for _ in ()).throw(AssertionError("unexpected KG fallback")),
    )
    monkeypatch.setattr(
        engine,
        "_rule_visits",
        lambda _q: (_ for _ in ()).throw(AssertionError("unexpected visit fallback")),
    )

    result = engine._rule_sql_normalized_result("synthetic temporal cohort request")

    assert result.disposition is CompileDisposition.BLOCKED


def test_temporal_owner_decline_blocks_legacy_visits(monkeypatch):
    engine = _bare_engine()
    intent = TemporalCohortIntent(
        TemporalCohortDisposition.FAMILY_RETENTION,
        kind=TemporalCohortKind.RETENTION,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "classify_age_average_intent",
        lambda _question, _domains: AgeAverageIntent(
            AgeAverageDisposition.NOT_APPLICABLE,
        ),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "classify_temporal_cohort_intent",
        lambda _question, _domains: intent,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_advanced_medical_result",
        lambda *_args, **_kwargs: CompileResult(CompileDisposition.BLOCKED),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected compositional fallback")),
    )
    monkeypatch.setattr(
        engine,
        "_rule_visits",
        lambda _q: (_ for _ in ()).throw(AssertionError("unexpected visit fallback")),
    )

    result = engine._rule_sql_normalized_result("synthetic temporal cohort request")

    assert result.disposition is CompileDisposition.BLOCKED


def test_complete_bounded_retention_decline_does_not_reopen_fallbacks(monkeypatch):
    engine = _bare_engine()
    question = "按首诊月统计年龄至少40岁且不超过50岁的女性患者留存率，偏移月0至3。"
    intent = classify_temporal_cohort_intent(question, engine._domains)
    seen: list[TemporalCohortIntent] = []

    assert intent.disposition is TemporalCohortDisposition.FAMILY_RETENTION
    assert intent.has_complete_family_retention

    monkeypatch.setattr(
        nl2sql_module,
        "classify_age_average_intent",
        lambda _question, _domains: AgeAverageIntent(
            AgeAverageDisposition.NOT_APPLICABLE,
        ),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "classify_temporal_cohort_intent",
        lambda _question, _domains: intent,
    )

    def declined_owner(
        _question,
        _domains,
        *,
        temporal_cohort_intent=None,
        **_kwargs,
    ):
        seen.append(temporal_cohort_intent)
        return CompileResult(CompileDisposition.NOT_APPLICABLE)

    monkeypatch.setattr(
        nl2sql_module,
        "compile_advanced_medical_result",
        declined_owner,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("unexpected compositional fallback"),
        ),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_explicit_medical_sql",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected explicit fallback"),
        ),
    )
    monkeypatch.setattr(
        engine,
        "_rule_structured_medical_analytics",
        lambda _q: (_ for _ in ()).throw(AssertionError("unexpected curated fallback")),
    )
    monkeypatch.setattr(
        engine,
        "_rule_advanced",
        lambda _q: (_ for _ in ()).throw(AssertionError("unexpected legacy advanced fallback")),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected semantic fallback"),
        ),
    )
    monkeypatch.setattr(
        engine,
        "_rule_kg",
        lambda _q: (_ for _ in ()).throw(AssertionError("unexpected KG fallback")),
    )
    monkeypatch.setattr(
        engine,
        "_rule_visits",
        lambda _q: (_ for _ in ()).throw(AssertionError("unexpected visit fallback")),
    )

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.BLOCKED
    assert seen == [intent]


def test_terminal_rule_result_does_not_invoke_llm(monkeypatch):
    engine = _bare_engine()
    monkeypatch.setattr(engine, "_normalize", lambda question: question)
    monkeypatch.setattr(
        engine,
        "_rule_sql_normalized_result",
        lambda _question: CompileResult(CompileDisposition.BLOCKED),
    )
    monkeypatch.setattr(
        engine,
        "_llm_pipeline",
        lambda _question: (_ for _ in ()).throw(AssertionError("unexpected LLM fallback")),
    )
    monkeypatch.setattr(
        engine,
        "_llm_query",
        lambda _question: (_ for _ in ()).throw(AssertionError("unexpected LLM fallback")),
    )

    to_sql = engine.to_sql("synthetic terminal request")
    query = engine.query("synthetic terminal request")

    assert to_sql["engine"] == "none"
    assert query["engine"] == "none"


def test_exclusive_owner_execution_error_does_not_reopen_llm(monkeypatch):
    engine = _bare_engine()
    engine.db_path = "synthetic.db"
    question = "synthetic global age request"
    intent = AgeAverageIntent(AgeAverageDisposition.EXPLICIT_GLOBAL)
    expected = "SELECT patient_id, age FROM patients"
    monkeypatch.setattr(engine, "_normalize", lambda value: value)
    monkeypatch.setattr(
        nl2sql_module,
        "classify_age_average_intent",
        lambda _question, _domains: intent,
    )
    def explicit(_question, _domains, *, age_average_intent, **_kwargs):
        assert age_average_intent is intent
        return expected

    monkeypatch.setattr(nl2sql_module, "compile_explicit_medical_sql", explicit)
    monkeypatch.setattr(
        nl2sql_module,
        "_candidate_matches_contract",
        lambda *_args, **_kwargs: True,
    )

    owned_result = engine._rule_sql_normalized_result(question)

    assert owned_result.disposition is CompileDisposition.SQL
    assert owned_result.sql == expected
    assert owned_result.fallback_fenced
    monkeypatch.setattr(
        engine,
        "_rule_sql_normalized_result",
        lambda _question: owned_result,
    )

    def execution_error(sql, db_path):
        assert sql == expected
        assert db_path == "synthetic.db"
        raise RuntimeError("synthetic execution failure")

    monkeypatch.setattr(nl2sql_module, "run_sql", execution_error)
    monkeypatch.setattr(
        engine,
        "_llm_query",
        lambda _question: (_ for _ in ()).throw(
            AssertionError("unexpected LLM fallback"),
        ),
    )

    result = engine.query(question)

    assert result["engine"] == "none"
    assert result["sql"] == ""
    assert result["validated"] is False
    assert result["columns"] == []
    assert result["rows"] == []
    assert "不能交由通用回退处理" in result["error"]


def test_real_anchored_reengagement_blocks_before_contract_or_fallback(monkeypatch):
    """The shared temporal record owns a sparse first-visit cohort request."""

    engine = _bare_engine()
    question = "统计2024年首次就诊后30天内再次就诊人数，按月分组。"
    intent = classify_temporal_cohort_intent(question, engine._domains)

    assert intent.disposition is TemporalCohortDisposition.TERMINAL_ABSTAIN
    assert intent.kind is TemporalCohortKind.REENGAGEMENT

    def unexpected(name: str):
        def route(*_args, **_kwargs):
            raise AssertionError(f"unexpected {name} route")

        return route

    monkeypatch.setattr(nl2sql_module, "extract_semantic_contract", unexpected("contract"))
    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        unexpected("compositional"),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_explicit_medical_sql",
        unexpected("explicit"),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_advanced_medical_result",
        unexpected("advanced"),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        unexpected("semantic"),
    )
    monkeypatch.setattr(
        engine,
        "_rule_structured_medical_analytics",
        unexpected("curated"),
    )
    monkeypatch.setattr(engine, "_rule_advanced", unexpected("legacy advanced"))
    monkeypatch.setattr(engine, "_rule_kg", unexpected("KG"))
    monkeypatch.setattr(engine, "_rule_visits", unexpected("visits"))
    monkeypatch.setattr(engine, "_normalize", lambda value: value)
    monkeypatch.setattr(engine, "_llm_pipeline", unexpected("LLM pipeline"))
    monkeypatch.setattr(engine, "_llm_query", unexpected("LLM query"))

    result = engine._rule_sql_normalized_result(question)
    to_sql = engine.to_sql(question)
    query = engine.query(question)

    assert result.disposition is CompileDisposition.BLOCKED
    assert to_sql["engine"] == "none"
    assert query["engine"] == "none"


def test_full_width_relation_or_blocks_legacy_visits_after_kg_decline(monkeypatch):
    """A declined graph union cannot degrade to a regional visit count."""

    from dki.analysis.nl2sql_semantic_kg import relation_fallback_disposition

    engine = _bare_engine()
    engine._domains = {"kg_relation_zh": ["呈现症状", "接受治疗"]}
    question = (
        "具有呈现症状ＯＲ接受治疗关系的疾病头实体患者就诊，"
        "按地区分组，返回地区、就诊次数和不同疾病数。"
    )

    assert (
        relation_fallback_disposition(question, engine._domains)
        is CompileDisposition.BLOCKED
    )
    def unexpected(name: str):
        def route(*_args, **_kwargs):
            raise AssertionError(f"unexpected {name} route")

        return route

    # This request used to reach the compositional group compiler before the
    # post-KG fence.  A declined KG owner must stop every earlier partial route
    # as well as the legacy and LLM fallbacks, but only after KG declines.
    calls: list[str] = []
    original_fallback_disposition = nl2sql_module.relation_fallback_disposition

    def declined_kg(*_args):
        calls.append("kg")
        return None

    def fallback_after_kg(*args):
        assert calls == ["kg"]
        calls.append("fallback")
        return original_fallback_disposition(*args)

    monkeypatch.setattr(nl2sql_module, "compile_kg_medical_sql", declined_kg)
    monkeypatch.setattr(
        nl2sql_module,
        "relation_fallback_disposition",
        fallback_after_kg,
    )
    monkeypatch.setattr(nl2sql_module, "extract_semantic_contract", unexpected("contract"))
    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        unexpected("compositional"),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_explicit_medical_sql",
        unexpected("explicit"),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        unexpected("semantic"),
    )
    monkeypatch.setattr(engine, "_rule_structured_medical_analytics", unexpected("curated"))
    monkeypatch.setattr(engine, "_rule_advanced", unexpected("legacy advanced"))
    monkeypatch.setattr(engine, "_needs_llm", unexpected("LLM"))
    monkeypatch.setattr(engine, "_rule_kg", unexpected("legacy KG"))
    monkeypatch.setattr(engine, "_rule_visits", unexpected("visits"))

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.BLOCKED
    assert calls == ["kg", "fallback"]


def test_full_width_relation_kg_exception_blocks_broad_fallbacks(monkeypatch):
    """A KG compiler fault is terminal rather than a declined union."""

    engine = _bare_engine()
    engine._domains = {"kg_relation_zh": ["呈现症状", "接受治疗"]}
    question = "具有呈现症状ＯＲ接受治疗关系的疾病头实体患者就诊。"

    def unexpected(name: str):
        def route(*_args, **_kwargs):
            raise AssertionError(f"unexpected {name} route")

        return route

    def kg_error(*_args, **_kwargs):
        raise RuntimeError("synthetic KG compiler failure")

    monkeypatch.setattr(nl2sql_module, "compile_kg_medical_sql", kg_error)
    monkeypatch.setattr(
        nl2sql_module,
        "relation_fallback_disposition",
        unexpected("relation fallback"),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        unexpected("compositional"),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_explicit_medical_sql",
        unexpected("explicit"),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        unexpected("semantic"),
    )
    monkeypatch.setattr(engine, "_rule_structured_medical_analytics", unexpected("curated"))
    monkeypatch.setattr(engine, "_rule_advanced", unexpected("legacy advanced"))
    monkeypatch.setattr(engine, "_needs_llm", unexpected("LLM"))
    monkeypatch.setattr(engine, "_rule_kg", unexpected("legacy KG"))
    monkeypatch.setattr(engine, "_rule_visits", unexpected("visits"))

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.BLOCKED


def test_grounded_single_relation_reserves_visit_compilers_for_kg(monkeypatch):
    """A plain relation may use legacy KG SQL, never a relationless visit row."""

    question = "统计每个疾病的呈现症状关系数量。"
    domains = {"kg_relation_zh": ["呈现症状"]}

    assert compile_compositional_medical_sql(question, domains) is None
    assert compile_simple_medical_sql(question, domains) is None

    # The real root must terminally abstain from this unsupported graph
    # aggregation rather than reinterpret it as a visit count grouped by disease
    # or reopen public LLM fallback after both graph owners decline.
    root_domains = {
        key: []
        for key in (
            "department",
            "disease",
            "region",
            "patient_name",
            "patient_id",
            "visit_id",
            "drug",
            "exam",
            "kg_entity",
            "kg_entity_type",
            "kg_head_type",
            "kg_relation_zh",
        )
    }
    root_domains["kg_relation_zh"] = ["呈现症状"]
    real_engine = _bare_engine()
    real_engine._domains = root_domains
    actual = real_engine._rule_sql_normalized_result(question)
    assert actual.disposition is CompileDisposition.BLOCKED
    assert actual.sql is None

    engine = _bare_engine()
    engine._domains = domains
    legacy_kg_sql = (
        "SELECT head, COUNT(*) AS n FROM kg_triples "
        "WHERE relation_zh = '呈现症状' GROUP BY head"
    )

    monkeypatch.setattr(
        nl2sql_module,
        "compile_explicit_medical_sql",
        lambda _question, _domains, **_kwargs: None,
    )
    monkeypatch.setattr(engine, "_rule_structured_medical_analytics", lambda _q: None)
    monkeypatch.setattr(engine, "_rule_advanced", lambda _q: None)
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        lambda *_args, **_kwargs: CompileResult(CompileDisposition.NOT_APPLICABLE),
    )
    # A hard-shape heuristic must not preempt a compatible graph fallback.
    monkeypatch.setattr(engine, "_needs_llm", lambda _q: True)
    monkeypatch.setattr(engine, "_rule_kg", lambda _q: legacy_kg_sql)
    monkeypatch.setattr(
        engine,
        "_rule_visits",
        lambda _q: (_ for _ in ()).throw(
            AssertionError("grounded relation reached visit fallback"),
        ),
    )
    monkeypatch.setattr(nl2sql_module, "_candidate_matches_contract", lambda *_args, **_kwargs: True)

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.SQL
    assert result.sql == legacy_kg_sql
    assert result.fallback_fenced


def test_grounded_single_relation_legacy_exhaustion_blocks_llm(monkeypatch):
    """A plain edge can try legacy KG once, then must remain terminal."""

    engine = _bare_engine()
    engine._domains = {"kg_relation_zh": ["呈现症状"]}
    engine._normalization_pattern = None
    question = "统计每个疾病的呈现症状关系数量。"
    calls: list[str] = []

    _disable_existing_routes(monkeypatch, engine)
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        lambda *_args, **_kwargs: CompileResult(CompileDisposition.NOT_APPLICABLE),
    )
    monkeypatch.setattr(
        engine,
        "_rule_kg",
        lambda _q: calls.append("legacy_kg") and None,
    )
    monkeypatch.setattr(
        engine,
        "_rule_visits",
        lambda _q: (_ for _ in ()).throw(
            AssertionError("grounded relation reached visit fallback"),
        ),
    )
    monkeypatch.setattr(
        engine,
        "_llm_pipeline",
        lambda _q: (_ for _ in ()).throw(
            AssertionError("grounded relation reopened LLM fallback"),
        ),
    )

    result = engine.to_sql(question)

    assert result["engine"] == "none"
    assert calls == ["legacy_kg"]


def test_grounded_relation_semantic_fault_is_terminal(monkeypatch):
    """A graph-owner dispatcher fault cannot reopen a legacy route."""

    engine = _bare_engine()
    engine._domains = {"kg_relation_zh": ["呈现症状"]}
    question = "统计每个疾病的呈现症状关系数量。"

    _disable_existing_routes(monkeypatch, engine)
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic semantic fault"),
        ),
    )
    monkeypatch.setattr(
        engine,
        "_rule_kg",
        lambda _q: (_ for _ in ()).throw(
            AssertionError("semantic fault reached legacy KG"),
        ),
    )

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.BLOCKED


def test_grounded_relation_contract_fault_is_terminal(monkeypatch):
    """A relation contract-parser fault cannot reopen a broad route."""

    engine = _bare_engine()
    engine._domains = {"kg_relation_zh": ["呈现症状"]}
    question = "统计每个疾病的呈现症状关系数量。"

    _disable_existing_routes(monkeypatch, engine)
    monkeypatch.setattr(
        nl2sql_module,
        "extract_semantic_contract",
        lambda _q: (_ for _ in ()).throw(RuntimeError("synthetic contract fault")),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("contract fault reached semantic dispatcher"),
        ),
    )
    monkeypatch.setattr(
        engine,
        "_rule_kg",
        lambda _q: (_ for _ in ()).throw(
            AssertionError("contract fault reached legacy KG"),
        ),
    )

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.BLOCKED


def test_grounded_relation_legacy_kg_fault_is_terminal(monkeypatch):
    """A compatible legacy KG owner fault cannot reopen LLM fallback."""

    engine = _bare_engine()
    engine._domains = {"kg_relation_zh": ["呈现症状"]}
    question = "统计每个疾病的呈现症状关系数量。"

    _disable_existing_routes(monkeypatch, engine)
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        lambda *_args, **_kwargs: CompileResult(CompileDisposition.NOT_APPLICABLE),
    )
    monkeypatch.setattr(
        engine,
        "_rule_kg",
        lambda _q: (_ for _ in ()).throw(RuntimeError("synthetic legacy KG fault")),
    )
    monkeypatch.setattr(
        engine,
        "_rule_visits",
        lambda _q: (_ for _ in ()).throw(
            AssertionError("legacy KG fault reached visits fallback"),
        ),
    )

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.BLOCKED


def test_grounded_relation_unsafe_legacy_kg_candidate_is_terminal(monkeypatch):
    """A legacy rule named KG still must retain the grounded edge literal."""

    engine = _bare_engine()
    engine._domains = {"kg_relation_zh": ["呈现症状"]}
    question = "统计每个疾病的呈现症状关系数量。"

    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        lambda _question, _domains: None,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_explicit_medical_sql",
        lambda _question, _domains, **_kwargs: None,
    )
    monkeypatch.setattr(engine, "_rule_structured_medical_analytics", lambda _q: None)
    monkeypatch.setattr(engine, "_rule_advanced", lambda _q: None)
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        lambda *_args, **_kwargs: CompileResult(CompileDisposition.NOT_APPLICABLE),
    )
    monkeypatch.setattr(engine, "_needs_llm", lambda _q: False)
    monkeypatch.setattr(
        engine,
        "_rule_kg",
        lambda _q: "SELECT disease, COUNT(*) AS n FROM visits GROUP BY disease",
    )
    monkeypatch.setattr(
        engine,
        "_rule_visits",
        lambda _q: (_ for _ in ()).throw(
            AssertionError("grounded relation reached visit fallback"),
        ),
    )

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.BLOCKED


def test_grounded_relation_direct_candidates_must_preserve_graph_literal(monkeypatch):
    """Pre-semantic routes may not return visit SQL after seeing a relation."""

    engine = _bare_engine()
    engine._domains = {"kg_relation_zh": ["呈现症状"]}
    question = "统计每个疾病的呈现症状关系数量。"
    calls: list[str] = []

    def relationless_visit_route(name: str):
        def route(*_args, **_kwargs):
            calls.append(name)
            return "SELECT disease, COUNT(*) AS n FROM visits GROUP BY disease"

        return route

    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        relationless_visit_route("compositional"),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_explicit_medical_sql",
        relationless_visit_route("explicit"),
    )
    monkeypatch.setattr(
        engine,
        "_rule_structured_medical_analytics",
        relationless_visit_route("curated"),
    )
    monkeypatch.setattr(engine, "_rule_advanced", relationless_visit_route("advanced"))

    def terminal_semantic(*_args, **_kwargs):
        calls.append("semantic")
        return CompileResult(CompileDisposition.BLOCKED)

    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        terminal_semantic,
    )
    monkeypatch.setattr(
        engine,
        "_needs_llm",
        lambda _q: (_ for _ in ()).throw(AssertionError("reached LLM fallback")),
    )

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.BLOCKED
    assert calls == ["compositional", "explicit", "curated", "semantic"]


def test_grounded_relation_direct_kg_candidate_remains_eligible(monkeypatch):
    """The fence reserves relation ownership without rejecting graph SQL."""

    engine = _bare_engine()
    engine._domains = {"kg_relation_zh": ["呈现症状"]}
    question = "统计每个头实体的呈现症状关系数量。"
    graph_sql = (
        "SELECT head, COUNT(*) AS relation_count FROM kg_triples "
        "WHERE relation_zh = '呈现症状' GROUP BY head"
    )

    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        lambda _question, _domains: None,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_explicit_medical_sql",
        lambda _question, _domains, **_kwargs: graph_sql,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("graph candidate did not retain direct ownership"),
        ),
    )

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.SQL
    assert result.sql == graph_sql
    assert result.fallback_fenced


def test_structured_relation_owner_preempts_direct_templates(monkeypatch):
    """A blocked structured relation cannot be flattened by an exact template."""

    engine = _bare_engine()
    engine._domains = {"kg_relation_zh": ["呈现症状"]}
    question = "呈现症状关系的三元组置信度至少0.8，返回头实体。"
    calls: list[str] = []

    def unexpected(name: str):
        def route(*_args, **_kwargs):
            raise AssertionError(f"unexpected {name} route")

        return route

    monkeypatch.setattr(
        nl2sql_module,
        "relation_fallback_disposition",
        lambda _question, _domains: CompileDisposition.BLOCKED,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        unexpected("compositional"),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_explicit_medical_sql",
        unexpected("explicit"),
    )
    monkeypatch.setattr(engine, "_rule_structured_medical_analytics", unexpected("curated"))
    monkeypatch.setattr(engine, "_rule_advanced", unexpected("advanced"))

    def declined_semantic(*_args, **_kwargs):
        calls.append("semantic")
        return CompileResult(CompileDisposition.NOT_APPLICABLE)

    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        declined_semantic,
    )

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.BLOCKED
    assert calls == ["semantic"]


def test_structured_relation_owner_sql_is_fallback_fenced(monkeypatch):
    """A structured graph SQL execution failure cannot reopen the LLM."""

    engine = _bare_engine()
    engine._domains = {"kg_relation_zh": ["呈现症状"]}
    question = "呈现症状关系的三元组置信度至少0.8，返回头实体。"
    graph_sql = "SELECT head FROM kg_triples WHERE relation_zh = '呈现症状'"

    monkeypatch.setattr(
        nl2sql_module,
        "relation_fallback_disposition",
        lambda _question, _domains: CompileDisposition.BLOCKED,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        lambda *_args, **_kwargs: CompileResult(CompileDisposition.SQL, graph_sql),
    )

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.SQL
    assert result.sql == graph_sql
    assert result.fallback_fenced


def test_structured_relation_none_contract_uses_ast_gate(monkeypatch):
    """An empty contract admits graph SQL through AST/provenance validation."""

    engine = _bare_engine()
    engine._domains = {"kg_relation_zh": ["呈现症状"]}
    question = "呈现症状关系的三元组置信度至少0.8，返回头实体。"
    graph_sql = "SELECT head FROM kg_triples WHERE relation_zh = '呈现症状'"
    observed: list[bool] = []

    monkeypatch.setattr(
        nl2sql_module,
        "relation_fallback_disposition",
        lambda _question, _domains: CompileDisposition.BLOCKED,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "extract_semantic_contract",
        lambda _question: SemanticContract(),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "_candidate_matches_contract",
        lambda _contract, _sql, *, enforce_semantics: observed.append(
            enforce_semantics,
        )
        or True,
    )

    def semantic(_question, _domains, *, accept_candidate, **_kwargs):
        return (
            CompileResult(CompileDisposition.SQL, graph_sql)
            if accept_candidate(graph_sql)
            else CompileResult(CompileDisposition.NOT_APPLICABLE)
        )

    monkeypatch.setattr(nl2sql_module, "compile_semantic_medical_result", semantic)

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.SQL
    assert result.sql == graph_sql
    assert result.fallback_fenced
    assert observed == [False]


def test_root_none_contract_accepts_all_grounded_kg_conditional_counts(monkeypatch):
    """A real graph family may use the NONE contract's AST-only gate."""

    engine = _bare_engine()
    engine._domains = {
        "kg_relation_zh": ["临床表现", "治疗药物", "诊断检查"],
    }
    question = (
        "临床表现、治疗药物和诊断检查条件计数覆盖审计，"
        "症状数至少2且不超过4，治疗药物数至少3且不超过5，"
        "诊断检查数至少1且不超过2。"
    )
    strictness: list[bool] = []

    monkeypatch.setattr(
        nl2sql_module,
        "extract_semantic_contract",
        lambda _question: SemanticContract(),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "validate_candidate",
        lambda _contract, _sql, *, strict: strictness.append(strict)
        or ValidationResult(True),
    )

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.SQL
    assert result.sql is not None
    assert result.fallback_fenced
    assert strictness == [False]
    for predicate in (
        "'临床表现' THEN tail END) >= 2",
        "'临床表现' THEN tail END) <= 4",
        "'治疗药物' THEN tail END) >= 3",
        "'治疗药物' THEN tail END) <= 5",
        "'诊断检查' THEN tail END) >= 1",
        "'诊断检查' THEN tail END) <= 2",
    ):
        assert predicate in result.sql


def test_root_none_contract_accepts_all_grounded_observed_risk_chain(monkeypatch):
    """Root dispatch retains all multi-role risk-chain qualifiers under NONE."""

    engine = _bare_engine()
    engine._domains = {
        "region": ["北京", "深圳"],
        "kg_relation_zh": ["呈现症状", "接受治疗", "并发症", "治疗药物", "用药禁忌"],
    }
    question = (
        "并发症与用药禁忌风险链零次也保留并统计实际处方，"
        "年龄至少40岁的女性深圳患者自2025-01-01以来，"
        "并发症关系置信度至少0.7且观察次数至少2，"
        "用药禁忌关系置信度至少0.8且观察次数至少3，"
        "实际处方次数至少2且不超过4。"
    )
    strictness: list[bool] = []

    monkeypatch.setattr(
        nl2sql_module,
        "extract_semantic_contract",
        lambda _question: SemanticContract(),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "validate_candidate",
        lambda _contract, _sql, *, strict: strictness.append(strict)
        or ValidationResult(True),
    )

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.SQL
    assert result.sql is not None
    assert result.fallback_fenced
    assert strictness == [False]
    for predicate in (
        "c.relation_zh = '并发症'",
        "x.relation_zh = '用药禁忌'",
        "c.confidence >= 0.7",
        "x.confidence >= 0.8",
        "c.count >= 2",
        "x.count >= 3",
        "HAVING COUNT(v.visit_id) >= 2 AND COUNT(v.visit_id) <= 4",
    ):
        assert predicate in result.sql


def test_structured_relation_unreliable_contract_stays_terminal(monkeypatch):
    """Partial output requirements cannot be weakened to an empty contract."""

    engine = _bare_engine()
    engine._domains = {"kg_relation_zh": ["呈现症状"]}
    question = "呈现症状关系的三元组置信度至少0.8。"
    graph_sql = "SELECT head FROM kg_triples WHERE relation_zh = '呈现症状'"
    observed: list[bool] = []

    monkeypatch.setattr(
        nl2sql_module,
        "relation_fallback_disposition",
        lambda _question, _domains: CompileDisposition.BLOCKED,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "extract_semantic_contract",
        lambda _question: SemanticContract(
            projections=("head",),
            confidence=0.2,
            status=ContractStatus.PARTIAL,
        ),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "_candidate_matches_contract",
        lambda _contract, _sql, *, enforce_semantics: observed.append(
            enforce_semantics,
        )
        or True,
    )

    def semantic(_question, _domains, *, accept_candidate, **_kwargs):
        return (
            CompileResult(CompileDisposition.SQL, graph_sql)
            if accept_candidate(graph_sql)
            else CompileResult(CompileDisposition.NOT_APPLICABLE)
        )

    monkeypatch.setattr(nl2sql_module, "compile_semantic_medical_result", semantic)

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.BLOCKED
    assert observed == []


def test_multiple_grounded_relations_stay_with_semantic_owner(monkeypatch):
    """Repeated graph slots remain structured even if classification regresses."""

    engine = _bare_engine()
    engine._domains = {"kg_relation_zh": ["呈现症状", "接受治疗"]}
    question = "具有呈现症状与接受治疗关系的疾病头实体。"
    calls: list[str] = []

    def unexpected(name: str):
        def route(*_args, **_kwargs):
            raise AssertionError(f"unexpected {name} route")

        return route

    # The occurrence cardinality fence is independent of the evolving parser
    # classification so duplicated/chained values cannot become one flat slot.
    monkeypatch.setattr(
        nl2sql_module,
        "relation_fallback_disposition",
        lambda _question, _domains: CompileDisposition.NOT_APPLICABLE,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        unexpected("compositional"),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_explicit_medical_sql",
        unexpected("explicit"),
    )
    monkeypatch.setattr(engine, "_rule_structured_medical_analytics", unexpected("curated"))
    monkeypatch.setattr(engine, "_rule_advanced", unexpected("advanced"))

    def declined_semantic(*_args, **_kwargs):
        calls.append("semantic")
        return CompileResult(CompileDisposition.NOT_APPLICABLE)

    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        declined_semantic,
    )

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.BLOCKED
    assert calls == ["semantic"]


def test_structured_relation_contract_rejection_blocks_broad_fallbacks(monkeypatch):
    engine = _bare_engine()
    engine._domains = {"kg_relation_zh": ["呈现症状", "接受治疗"]}
    question = (
        "针对呈现症状ＯＲ接受治疗关系三元组边，置信度至少0.8且观察次数至少4，"
        "返回头实体、尾实体、置信度和观察次数。"
    )

    def unexpected(name: str):
        def route(*_args, **_kwargs):
            raise AssertionError(f"unexpected {name} route")

        return route

    calls: list[str] = []
    original_fallback_disposition = nl2sql_module.relation_fallback_disposition

    def rejected_kg(*_args):
        calls.append("kg")
        return "SELECT head, tail FROM kg_triples"

    def rejected_candidate(*_args, **_kwargs):
        calls.append("candidate")
        return False

    def fallback_after_rejection(*args):
        assert calls == ["kg", "candidate"]
        calls.append("fallback")
        return original_fallback_disposition(*args)

    monkeypatch.setattr(nl2sql_module, "compile_kg_medical_sql", rejected_kg)
    monkeypatch.setattr(
        nl2sql_module,
        "relation_fallback_disposition",
        fallback_after_rejection,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "extract_semantic_contract",
        lambda _question: SemanticContract(
            projections=("head", "tail"),
            confidence=1.0,
            status=ContractStatus.COMPLETE,
        ),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "_candidate_matches_contract",
        rejected_candidate,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        unexpected("compositional"),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_explicit_medical_sql",
        unexpected("explicit"),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        unexpected("semantic"),
    )
    monkeypatch.setattr(engine, "_rule_structured_medical_analytics", unexpected("curated"))
    monkeypatch.setattr(engine, "_rule_advanced", unexpected("legacy advanced"))
    monkeypatch.setattr(engine, "_needs_llm", unexpected("LLM"))
    monkeypatch.setattr(engine, "_rule_kg", unexpected("legacy KG"))
    monkeypatch.setattr(engine, "_rule_visits", unexpected("visits"))

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.BLOCKED
    assert calls == ["kg", "candidate", "fallback"]


def test_complete_full_width_relation_union_stays_with_kg_owner(monkeypatch):
    engine = _bare_engine()
    engine._domains = {"kg_relation_zh": ["呈现症状", "接受治疗"]}
    question = (
        "针对呈现症状ＯＲ接受治疗关系三元组边，置信度至少0.8且观察次数至少4，"
        "返回头实体、尾实体、置信度和观察次数。"
    )
    def unexpected(name: str):
        def route(*_args, **_kwargs):
            raise AssertionError(f"unexpected {name} route")

        return route

    monkeypatch.setattr(
        nl2sql_module,
        "relation_fallback_disposition",
        unexpected("relation fallback"),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        unexpected("compositional"),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_explicit_medical_sql",
        unexpected("explicit"),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        unexpected("semantic"),
    )
    monkeypatch.setattr(engine, "_rule_structured_medical_analytics", unexpected("curated"))
    monkeypatch.setattr(engine, "_rule_advanced", unexpected("legacy advanced"))
    monkeypatch.setattr(engine, "_needs_llm", unexpected("LLM"))
    monkeypatch.setattr(engine, "_rule_kg", unexpected("legacy KG"))
    monkeypatch.setattr(engine, "_rule_visits", unexpected("visits"))
    monkeypatch.setattr(
        nl2sql_module,
        "extract_semantic_contract",
        lambda _question: SemanticContract(
            projections=("head", "tail", "confidence", "count"),
            confidence=1.0,
            status=ContractStatus.COMPLETE,
        ),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "_candidate_matches_contract",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        engine,
        "_needs_llm",
        lambda _q: (_ for _ in ()).throw(AssertionError("unexpected LLM fallback")),
    )
    monkeypatch.setattr(
        engine,
        "_rule_kg",
        lambda _q: (_ for _ in ()).throw(AssertionError("unexpected KG fallback")),
    )
    monkeypatch.setattr(
        engine,
        "_rule_visits",
        lambda _q: (_ for _ in ()).throw(
            AssertionError("unexpected visit fallback"),
        ),
    )

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.SQL
    assert result.sql is not None
    assert result.fallback_fenced
    assert "relation_zh IN ('呈现症状', '接受治疗')" in result.sql


def test_structured_relation_execution_error_does_not_reopen_llm(monkeypatch):
    engine = _bare_engine()
    engine._domains = {"kg_relation_zh": ["呈现症状", "接受治疗"]}
    engine.db_path = "synthetic.db"
    question = (
        "针对呈现症状ＯＲ接受治疗关系三元组边，置信度至少0.8且观察次数至少4，"
        "返回头实体、尾实体、置信度和观察次数。"
    )
    candidate = "SELECT head, tail, confidence, count FROM kg_triples"

    monkeypatch.setattr(engine, "_normalize", lambda value: value)
    monkeypatch.setattr(
        nl2sql_module,
        "compile_kg_medical_sql",
        lambda *_args: candidate,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "extract_semantic_contract",
        lambda _question: SemanticContract(
            projections=("head", "tail", "confidence", "count"),
            confidence=1.0,
            status=ContractStatus.COMPLETE,
        ),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "_candidate_matches_contract",
        lambda *_args, **_kwargs: True,
    )

    def execution_error(sql, db_path):
        assert sql == candidate
        assert db_path == "synthetic.db"
        raise RuntimeError("synthetic execution failure")

    monkeypatch.setattr(nl2sql_module, "run_sql", execution_error)
    monkeypatch.setattr(
        engine,
        "_llm_query",
        lambda _question: (_ for _ in ()).throw(
            AssertionError("unexpected LLM fallback"),
        ),
    )

    result = engine.query(question)

    assert result["engine"] == "none"
    assert result["sql"] == ""
    assert result["validated"] is False
    assert result["columns"] == []
    assert result["rows"] == []
    assert "不能交由通用回退处理" in result["error"]


def test_real_qualified_global_age_blocks_before_contract_or_fallback(monkeypatch):
    """A global age template cannot discard an additional patient filter."""

    engine = _bare_engine()
    engine._domains = {"region": ["深圳"]}
    question = (
        "找出深圳地区年龄至少40岁的女性患者，年龄高于全部患者平均年龄，"
        "返回患者ID、姓名、年龄和地区；按年龄降序、患者ID升序排列。"
    )
    intent = classify_age_average_intent(question, engine._domains)

    assert intent.disposition is AgeAverageDisposition.TERMINAL_ABSTAIN

    def unexpected(name: str):
        def route(*_args, **_kwargs):
            raise AssertionError(f"unexpected {name} route")

        return route

    monkeypatch.setattr(nl2sql_module, "extract_semantic_contract", unexpected("contract"))
    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        unexpected("compositional"),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_explicit_medical_sql",
        unexpected("explicit"),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_advanced_medical_result",
        unexpected("advanced"),
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_semantic_medical_result",
        unexpected("semantic"),
    )
    monkeypatch.setattr(
        engine,
        "_rule_structured_medical_analytics",
        unexpected("curated"),
    )
    monkeypatch.setattr(engine, "_rule_advanced", unexpected("legacy advanced"))
    monkeypatch.setattr(engine, "_rule_kg", unexpected("KG"))
    monkeypatch.setattr(engine, "_rule_visits", unexpected("visits"))

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.BLOCKED


def test_legacy_ratio_preserves_iso_date_scope():
    """The direct ratio route must constrain both numerator and denominator."""

    engine = _bare_engine()
    engine._domains = {"disease": ["高血压"], "department": [], "region": []}

    sql = engine._rule_advanced(
        "计算2024-01-01至2024-01-31期间高血压就诊占全部就诊的百分比。",
    )

    assert sql is not None
    assert "CASE WHEN disease = '高血压'" in sql
    assert "FROM visits WHERE visit_date BETWEEN '2024-01-01' AND '2024-01-31'" in sql


def test_legacy_count_average_preserves_region_scope_and_requested_order():
    """Both populations in a count-vs-average comparison share patient scope."""

    engine = _bare_engine()
    engine._domains = {
        "department": [],
        "disease": [],
        "drug": [],
        "exam": [],
        "region": ["广州"],
    }

    sql = engine._rule_advanced(
        "统计广州患者各科室的就诊次数，找出就诊次数超过平均每个科室就诊次数的科室，"
        "并按就诊次数升序排列。",
    )

    assert sql is not None
    assert "FROM visits AS v JOIN patients AS p ON p.patient_id = v.patient_id" in sql
    assert "WHERE p.region = '广州' GROUP BY v.department" in sql
    assert "WHERE p2.region = '广州' GROUP BY v2.department" in sql
    assert sql.endswith("ORDER BY n ASC")


def test_legacy_department_intersection_preserves_disease_order_and_limit():
    """A compound visit-set query cannot discard its explicit result modifiers."""

    engine = _bare_engine()
    engine._domains = {
        "department": ["呼吸内科", "骨科"],
        "disease": [],
        "drug": [],
        "exam": [],
        "region": [],
    }

    sql = engine._rule_advanced(
        "列出在呼吸内科和骨科都出现过的疾病的交集，并按疾病降序排列，只返回前2种。",
    )

    assert sql is not None
    assert "department = '呼吸内科' INTERSECT SELECT disease" in sql
    assert "department = '骨科'" in sql
    assert sql.endswith("ORDER BY disease DESC LIMIT 2")


def test_terminal_region_filter_blocks_before_legacy_advanced(monkeypatch):
    """An owned abstention must terminate before legacy partial matching."""

    engine = _bare_engine()
    engine._domains = {
        "department": [],
        "disease": [],
        "drug": [],
        "exam": [],
        "region": ["广州", "深圳"],
    }
    question = (
        "统计广州或不包括深圳患者各科室的就诊次数，"
        "找出就诊次数超过平均每个科室就诊次数的科室。"
    )

    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        lambda _question, _domains: None,
    )
    monkeypatch.setattr(
        nl2sql_module,
        "compile_explicit_medical_sql",
        lambda _question, _domains, **_kwargs: None,
    )
    monkeypatch.setattr(engine, "_rule_structured_medical_analytics", lambda _q: None)
    monkeypatch.setattr(
        engine,
        "_rule_advanced",
        lambda _q: (_ for _ in ()).throw(
            AssertionError("terminal semantic request reached legacy advanced"),
        ),
    )
    monkeypatch.setattr(
        engine,
        "_rule_kg",
        lambda _q: (_ for _ in ()).throw(AssertionError("unexpected KG fallback")),
    )
    monkeypatch.setattr(
        engine,
        "_rule_visits",
        lambda _q: (_ for _ in ()).throw(
            AssertionError("unexpected visit fallback"),
        ),
    )

    result = engine._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.BLOCKED
