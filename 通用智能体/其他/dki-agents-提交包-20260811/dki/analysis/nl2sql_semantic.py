"""High-precision schema/slot NL2SQL compiler.

Specialized compilers are ordered by structural specificity.  Each one must
consume a complete semantic signature or abstain; none may fall back to a
single metric merely because a related noun occurred in the question.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from dki.analysis.nl2sql_semantic_kg import RelationRequest
    from dki.analysis.nl2sql_semantic_obligations import SemanticObligations

from dki.analysis.nl2sql_age_average_intent import (
    AgeAverageDisposition,
    AgeAverageIntent,
    classify_age_average_intent,
)
from dki.analysis.nl2sql_compile_result import CompileDisposition, CompileResult
from dki.analysis.nl2sql_semantic_common import esc
from dki.analysis.nl2sql_temporal_cohort_intent import (
    TemporalCohortDisposition,
    TemporalCohortIntent,
    classify_temporal_cohort_intent,
)

Domains = Mapping[str, Sequence[str]]
CandidateAcceptor = Callable[[str], bool]


def _family_candidate_owns_relations(sql: str, relations: Sequence[str]) -> bool:
    """Require a relation-family fallback to retain every grounded literal.

    Relation families may consume multi-role and non-flat graph shapes, so this
    deliberately does not impose the direct KG compiler's flat-predicate
    admission rule.  It does prove that a family fallback is graph-aware and
    has not reinterpreted the request as a patients/visits-only query.
    """

    normalized_sql = sql.lower()
    return (
        bool(relations)
        and "kg_triples" in normalized_sql
        and all(f"'{esc(value).lower()}'" in normalized_sql for value in relations)
    )


def _qualified_relation_reference(question: str, end: int) -> bool:
    """Recognize a repeated relation used only to label a following metric.

    A phrase such as ``并发症关系置信度`` refers back to an already declared
    edge; it does not declare a second edge.  The same holds for a category
    aggregate label such as ``治疗药物数``.  This is intentionally a local
    suffix test: a repeated literal elsewhere remains a distinct raw role.
    """

    return re.match(
        r"\s*(?:(?:关系|边)?(?:置信度|观察(?:次数|计数|量)|出现次数|边次数|count)"
        r"|(?:数|数量|计数|总数|覆盖数))",
        question[end:],
        re.I,
    ) is not None


def _qualified_relation_label_reference(question: str, start: int, end: int) -> bool:
    """Recognize a repeated relation used only as a trailing schema label."""

    prefix = question[max(0, start - 24):start]
    suffix = question[end:end + 16]
    return bool(
        re.search(r"(?:涉及的)?知识关系\s*按\s*[‘'\"“]?$", prefix)
        and re.match(r"[’'\"”]?\s*(?:标签)?识别", suffix)
    )


def _family_signature_owns_parser_conflict(
    question: str,
    request: RelationRequest,
) -> bool:
    """Admit only declared high-specificity families for known parser conflicts."""

    values = set(request.values)
    errors = set(request.errors)
    if not request.metric_complete or not errors:
        return False
    if errors - {
        "unowned relation connective",
        "partially quoted relation list",
        "unowned disjunction",
    }:
        return False
    if (
        {"并发症", "治疗药物"}.issubset(values)
        and re.search(r"两跳|两段式|路径|知识链|→", question)
    ):
        return True
    if (
        {"并发症", "用药禁忌"}.issubset(values)
        and re.search(r"风险链|零次|实际处方|匹配处方|共同出现|观察次数", question)
    ):
        return True
    if (
        {"临床表现", "治疗药物", "诊断检查"}.issubset(values)
        and re.search(r"症状|临床表现", question)
        and re.search(r"条件计数|覆盖审计|三类数量|各类目标数|覆盖量", question)
    ):
        return True
    if (
        re.search(r"统一名称列表|同一结果|合并|集合去重", question)
        and "患者" in question
        and "实体" in question
    ):
        return True
    if (
        len(values) == 2
        and re.search(r"逐一满足|完整覆盖|关系清单|两项要求均|每项要求", question)
        and re.search(r"就诊|记录|平均费用|费用均值|平均花费", question)
    ):
        return True
    return False


def _family_relation_request_is_eligible(
    question: str,
    request: RelationRequest,
) -> bool:
    """Admit a family after only a metric-label duplicate parser conflict.

    Raw occurrence preservation remains essential for ordinary graph routes.
    A high-specificity family may nevertheless own a request where declared
    relations are repeated solely as metric or category-count labels.  Admit
    exactly that shape, never a general malformed connective: each raw role's
    first occurrence must be declarative, and every later occurrence must be an
    adjacent metric-label reference.
    """

    if not request.metric_complete:
        return False
    if not request.errors:
        return True
    if _family_signature_owns_parser_conflict(question, request):
        return True
    if set(request.errors) - {
        "unowned relation connective",
        "partially quoted relation list",
    }:
        return False
    if len(request.occurrences) < 2:
        return False

    # A family can own multiple declared relation roles (for example,
    # ``并发症与用药禁忌风险链`` or a three-category coverage audit).  Preserve
    # each role's first mention as declarative and permit only later mentions of
    # that same literal when they are adjacent metric/category labels.  At least
    # one label reference is required, so a bare ambiguous relation list does
    # not become family-eligible merely because all its values differ.
    declared: set[str] = set()
    saw_reference = False
    for item in request.occurrences:
        is_reference = _qualified_relation_reference(question, item.end)
        if item.value in declared:
            is_reference = is_reference or _qualified_relation_label_reference(
                question,
                item.start,
                item.end,
            )
        if item.value not in declared:
            if is_reference:
                return False
            declared.add(item.value)
            continue
        if not is_reference:
            return False
        saw_reference = True
    return saw_reference


def compile_semantic_medical_result(
    question: str,
    domains: Domains,
    *,
    accept_candidate: Optional[CandidateAcceptor] = None,
    accept_self_validated_candidate: Optional[CandidateAcceptor] = None,
    age_average_intent: Optional[AgeAverageIntent] = None,
    temporal_cohort_intent: Optional[TemporalCohortIntent] = None,
    obligations: Optional["SemanticObligations"] = None,
) -> CompileResult:
    """Compile semantic SQL without collapsing owned abstentions into ``None``.

    Candidate rejection normally lets a less specialized compiler try to
    express the same request.  That remains useful for unrelated input.  It is
    unsafe for an exclusive age-average or temporal-cohort owner: once such a
    candidate is rejected, another compiler could only return SQL by dropping
    the owner-defining qualifier.  The result contract makes that boundary
    explicit without changing the legacy public ``Optional[str]`` API.
    """

    if not isinstance(question, str) or not question.strip() or not isinstance(domains, Mapping):
        return CompileResult(CompileDisposition.NOT_APPLICABLE)

    def accepted(sql: str, *, contract_self_validated: bool) -> bool:
        if accept_candidate is None or accept_candidate(sql):
            return True
        return bool(
            contract_self_validated
            and accept_self_validated_candidate is not None
            and accept_self_validated_candidate(sql)
        )

    # Imports are local so the package remains importable while individual
    # compiler modules are developed and tested independently.
    from dki.analysis.nl2sql_semantic_advanced import compile_advanced_medical_result
    from dki.analysis.nl2sql_semantic_kg import (
        analyze_relation_request,
        compile_kg_medical_sql,
        relation_fallback_disposition,
    )
    from dki.analysis.nl2sql_semantic_simple import compile_simple_medical_sql

    parsed_age = (
        age_average_intent
        if age_average_intent is not None
        else classify_age_average_intent(question, domains)
    )
    # Age ownership is checked first.  There is no reason to classify a second
    # family once the caller has already designated an exclusive owner.
    parsed_temporal = temporal_cohort_intent
    if (
        parsed_temporal is None
        and parsed_age.disposition is AgeAverageDisposition.NOT_APPLICABLE
    ):
        parsed_temporal = classify_temporal_cohort_intent(question, domains)
    exclusive_owner = (
        parsed_age.disposition is not AgeAverageDisposition.NOT_APPLICABLE
        or (
            parsed_temporal is not None
            and parsed_temporal.disposition
            is not TemporalCohortDisposition.NOT_APPLICABLE
        )
    )

    try:
        relation_request = analyze_relation_request(question, domains)
        relation_intent = relation_request.has_relation_intent
    except Exception:
        return CompileResult(CompileDisposition.BLOCKED)

    # Outside an already exclusive age/temporal owner, graph relations take
    # precedence over generic advanced analytics.  The latter has no graph
    # slots and could otherwise return a plausible patients/visits query while
    # silently discarding the grounded relation.
    if not relation_intent or exclusive_owner:
        advanced = compile_advanced_medical_result(
            question,
            domains,
            age_average_intent=parsed_age,
            temporal_cohort_intent=parsed_temporal,
            obligations=obligations,
        )
        if advanced.disposition is CompileDisposition.BLOCKED:
            return advanced
        if advanced.disposition is CompileDisposition.SQL:
            assert advanced.sql is not None
            if accepted(
                advanced.sql,
                contract_self_validated=advanced.contract_self_validated,
            ):
                return advanced
            if exclusive_owner or advanced.fallback_fenced:
                return CompileResult(CompileDisposition.BLOCKED)
        elif exclusive_owner:
            # A designated owner must never become a generic miss.  This also
            # preserves the terminal boundary if a future owner implementation
            # accidentally reports NOT_APPLICABLE after receiving its own record.
            return CompileResult(CompileDisposition.BLOCKED)

    try:
        kg_sql = compile_kg_medical_sql(question, domains)
    except Exception:
        if relation_intent:
            return CompileResult(CompileDisposition.BLOCKED)
        kg_sql = None
    # A rejected candidate normally leaves an unrelated request available to a
    # less specialized compiler.  A structured relation request is different:
    # after the KG owner declines or its candidate is rejected, a row compiler
    # cannot safely discard relation roles or edge metrics.
    if kg_sql and accepted(kg_sql, contract_self_validated=True):
        # A successfully lowered grounded edge remains owned through execution:
        # an execution failure must not reopen row or LLM fallback that could
        # discard the relation predicate.
        return CompileResult(
            CompileDisposition.SQL,
            kg_sql,
            fallback_fenced=relation_intent,
        )

    # Direct KG lowering has priority, but it does not own every complete
    # graph-and-clinical composition.  Give a high-specificity relation family
    # one chance before the structured fallback fence turns a KG decline into a
    # terminal abstention.  This is deliberately not the generic advanced
    # dispatcher: it may return only a graph-aware candidate that retains every
    # grounded relation literal.  Most malformed relation requests remain with
    # the shared terminal classifier below; the narrow exception is a repeated
    # declared literal used only as an adjacent metric/category-label reference,
    # which a family can prove and retain without flattening a second edge role.
    if relation_intent and _family_relation_request_is_eligible(question, relation_request):
        try:
            from dki.analysis.nl2sql_semantic_families import compile_semantic_family_sql

            family_sql = compile_semantic_family_sql(question, domains)
        except Exception:
            return CompileResult(CompileDisposition.BLOCKED)
        if family_sql:
            if not _family_candidate_owns_relations(family_sql, relation_request.values):
                return CompileResult(CompileDisposition.BLOCKED)
            if accepted(family_sql, contract_self_validated=True):
                return CompileResult(
                    CompileDisposition.SQL,
                    family_sql,
                    fallback_fenced=True,
                    contract_self_validated=True,
                )
            # A family candidate that had complete graph ownership but failed
            # validation cannot be replaced by a generic route that omits an
            # owned graph role or qualifier.
            return CompileResult(CompileDisposition.BLOCKED)
    try:
        disposition = relation_fallback_disposition(question, domains)
    except Exception:
        if relation_intent:
            return CompileResult(CompileDisposition.BLOCKED)
        disposition = CompileDisposition.NOT_APPLICABLE
    if disposition is CompileDisposition.BLOCKED:
        return CompileResult(CompileDisposition.BLOCKED)
    if relation_intent:
        # A plain relation may still reach a compatible legacy KG template at
        # the root, but the row-oriented simple compiler has no edge slot.
        return CompileResult(CompileDisposition.NOT_APPLICABLE)

    simple_sql = compile_simple_medical_sql(question, domains)
    # The callback remains outside this module so validation can inspect the
    # original SQL string without changing compiler output or coupling rules to
    # a parser.
    if simple_sql and accepted(simple_sql, contract_self_validated=True):
        return CompileResult(
            CompileDisposition.SQL,
            simple_sql,
            contract_self_validated=True,
        )
    return CompileResult(CompileDisposition.NOT_APPLICABLE)


def compile_semantic_medical_sql(
    question: str,
    domains: Domains,
    *,
    accept_candidate: Optional[CandidateAcceptor] = None,
) -> Optional[str]:
    """Compile semantic SQL through the backwards-compatible public API."""

    return compile_semantic_medical_result(
        question,
        domains,
        accept_candidate=accept_candidate,
    ).sql
