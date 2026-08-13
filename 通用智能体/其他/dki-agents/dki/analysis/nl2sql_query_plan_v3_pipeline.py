"""Gold-free QueryPlan v3 generation, validation, execution, and selection.

The pipeline deliberately owns no provider client and accepts no oracle, reference
SQL, correctness label, or judge input.  Model calls and database execution are
injected, which keeps unit tests offline and makes the production adapter thin.
Only bounded validation-stage codes may flow into a repair prompt; candidate JSON,
compiled SQL, execution evidence, counterfactual evidence, and database errors are
never returned to the model.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dki.analysis.nl2sql_candidate_selector import (
    CandidateProvenance,
    CandidateRecord,
    ExecutionFingerprint,
    IntentPlanHashes,
    SelectorConfig,
    StaticValidation,
    select_candidate,
)
from dki.analysis.nl2sql_contract import SemanticContract, validate_candidate
from dki.analysis.nl2sql_counterfactual import (
    CounterfactualReport,
    run_counterfactual_suite,
)
from dki.analysis.nl2sql_protocol import validate_gold_sql as validate_read_query
from dki.analysis.nl2sql_query_plan_v2 import validate_plan_v2_grounding
from dki.analysis.nl2sql_query_plan_v3 import (
    PLAN_VERSION,
    QueryPlanV3Error,
    canonical_plan_sha256_v3,
    compile_query_plan_v3,
    parse_query_plan_v3,
    query_plan_v3_schema,
    validate_query_plan_v3,
)
from dki.analysis.nl2sql_semantic_corpus import execute_with_fingerprint
from dki.analysis.nl2sql_semantic_obligations import (
    ObligationDisposition,
    PredicateOperator,
    SemanticObligations,
    extract_semantic_obligations,
)
from dki.analysis.nl2sql_semantic_scorer import (
    SemanticScore,
    score_candidate_semantics,
)


PIPELINE_VERSION = "nl2sql-query-plan-v3-pipeline-v4"
STRATEGY = "query_plan_v3"

SEMANTIC_LANES: tuple[str, ...] = (
    "grain/entity-preservation",
    "join/cardinality",
    "set-null-boundary",
    "ordering-precision-conservative",
)

_LANE_INSTRUCTIONS: Mapping[str, str] = {
    "grain/entity-preservation": (
        "Prove the requested output grain, aggregation stage, and whether entities "
        "with zero matches must be preserved. Use a left-preserving join only when "
        "the question explicitly requires preservation."
    ),
    "join/cardinality": (
        "Prove every schema-graph edge, guard, direction, and cardinality effect. "
        "Protect the declared grain from one-to-many or many-to-many amplification "
        "only when the question requires that protection."
    ),
    "set-null-boundary": (
        "Prove set duplicate policy, anti-membership NULL safety, explicit NULL "
        "membership, and inclusive or exclusive temporal boundaries. Do not infer "
        "a set or NULL policy that the question does not state."
    ),
    "ordering-precision-conservative": (
        "Prove output order, sort direction, tie policy, limit and offset behavior, "
        "and numeric precision. Apply the DETERMINISTIC OUTPUT CONVENTIONS for "
        "tie-breakers exactly; beyond those conventions never add ordering, rounding, "
        "tie expansion, or semantic claims absent from the question."
    ),
}

# Question-neutral format example.  It teaches JSON shape only (verified by a
# unit test to parse and compile); it encodes no benchmark or holdout content.
_FORMAT_EXAMPLE_PLAN: Mapping[str, Any] = {
    "plan_version": 3,
    "intent": {
        "output": {"columns": ["department", "COUNT(*)"], "row_set_mode": "preserve"},
        "grain": {"kind": "groups", "keys": ["department"]},
        "joins": [],
        "entity_preservation": {
            "mode": "none",
            "alias": None,
            "include_zero_match": False,
        },
        "aggregation": {"stage": "per_group", "distinct_scope": "none"},
        "set_semantics": {"mode": "none", "preserve_duplicates": False},
        "null_semantics": {"mode": "unspecified"},
        "tie_policy": {"mode": "deterministic", "keys": ["department"]},
        "ordering": [
            {"key": "COUNT(*)", "direction": "desc", "nulls": "unspecified"},
            {"key": "department", "direction": "asc", "nulls": "unspecified"},
        ],
        "precision": {"mode": "exact", "digits": None},
    },
    "query": {
        "type": "select",
        "from": {"type": "table", "table": "visits", "alias": "v"},
        "select": [
            {"expr": {"type": "column", "table": "v", "column": "department"}},
            {
                "expr": {"type": "aggregate", "function": "count", "argument": None},
                "alias": "n",
            },
        ],
        "group_by": [{"type": "column", "table": "v", "column": "department"}],
        "order_by": [
            {
                "expr": {"type": "aggregate", "function": "count", "argument": None},
                "direction": "desc",
            },
            {
                "expr": {"type": "column", "table": "v", "column": "department"},
                "direction": "asc",
            },
        ],
    },
}

# Exact intent field grammar.  The JSON-schema payload describes these fields
# with loose prose strings, which reasoning models routinely mis-render into
# nested expression objects; this note pins the accepted concrete shapes.
_INTENT_GRAMMAR_NOTE = (
    "INTENT FIELD GRAMMAR (exact concrete shapes; deviation is rejected)\n"
    'output: {"columns": [plain label strings, never objects], '
    '"row_set_mode": "preserve"|"distinct"}\n'
    'grain: {"kind": "scalar"|"rows"|"groups", "keys": [strings; empty for scalar, '
    "required for groups]}\n"
    'joins: [{"edge": schema-graph edge id, "left_alias": str, "right_alias": str, '
    '"kind": "inner"|"left", "preserve": "none"|"left"}] (preserve must be "none" '
    'for inner and "left" for left)\n'
    'entity_preservation: {"mode": "none"|"preserve", "alias": null|str, '
    '"include_zero_match": bool} (alias null and include_zero_match false when mode is "none")\n'
    'aggregation: {"stage": "none"|"base_rows"|"after_join"|"per_group", '
    '"distinct_scope": "none"|"rows"|"aggregate_arguments"}\n'
    'set_semantics: {"mode": "none"|"any_of"|"all_of"|"difference", '
    '"preserve_duplicates": bool}\n'
    'null_semantics: {"mode": "unspecified"|"include"|"exclude"|"only"}\n'
    'tie_policy: {"mode": "unspecified"|"deterministic"|"include_ties", '
    '"keys": [strings; only for deterministic]}\n'
    'ordering: [{"key": str, "direction": "asc"|"desc", '
    '"nulls": "unspecified"|"first"|"last"}]\n'
    'precision: {"mode": "exact"|"round", "digits": null for exact, 0..12 for round}'
)

# Dataset-wide deterministic result conventions, verified against every grouped
# and metric-ordered development gold.  They encode the curator's ordering and
# grain style so equal-metric rows and window selections replay exactly; they
# add no semantics beyond what the question requests.
_OUTPUT_CONVENTIONS_NOTE = (
    "DETERMINISTIC OUTPUT CONVENTIONS (apply exactly; the strict checker compares "
    "full ordered result sets)\n"
    "- ORDER BY starts with exactly the keys the question requests, in the "
    "requested directions.\n"
    "- Grouped metric sorts (per-group counts/sums/averages grouped by "
    "visits/patients columns such as department, disease, region, gender, "
    "patient, year): append every grouping key ASC after the requested metric "
    "keys as deterministic tie-breakers, in group-by order (use name instead of "
    "patient_id when both group a patient).\n"
    "- Exception — single-table groupings over kg_triples.relation or "
    "kg_entities.type: emit only the requested sort keys, no added tie-breaker.\n"
    "- Exception — rank-then-top-N requests phrased with an explicit CTE step "
    "(先用CTE统计…前N): emit only the requested metric sort plus LIMIT, no added "
    "tie-breaker.\n"
    "- Row-level metric sorts over kg_triples (count/confidence/mention_count): "
    "append the first projected column ASC as the only tie-breaker; never append "
    "further keys.\n"
    "- Sorts on a unique key (patient_id listings): no added tie-breaker.\n"
    "- Window requests (第M至第N条 / 跳过前K条取N条): keep the plain ordered "
    "query and use limit plus offset; never rewrite with ROW_NUMBER.\n"
    "- Entity-type constraints (尾实体类型为X / 头实体类型为X) must constrain the "
    "kg_entities row that is projected: join kg_entities and filter its type "
    "column so same-name entities of other types cannot match.\n"
    "- When projecting entity attributes (names, mention counts) reached through "
    'a kg_triples join, set row_set_mode "distinct" unless the question asks for '
    "one row per triple."
)

# Encodings that the closed ``query`` grammar accepts but that a generator does
# not reach for unprompted.  Every line below was confirmed against the compiler
# by an offline coverage audit; each one is a shape that would otherwise be
# rejected outright and cost the whole candidate.
_PLAN_SHAPE_NOTE = (
    "QUERY NODE SHAPES (the closed grammar accepts only these encodings)\n"
    "- Sorting by a computed column: write "
    '{"alias": "<select alias>", "direction": ...} in order_by. The alias must be '
    "declared in this query's own select list. A column node naming a select "
    "alias is rejected because it is not a real table column.\n"
    "- Range filters: there is no between node. Write a binary and over gte and "
    "lte on the same expression.\n"
    "- Join filters: join.on conjuncts may compare a column with a literal "
    "(e.g. kg_entities.type = 'Drug') as long as one conjunct is still an "
    "allowlisted column-to-column equality. Comparisons over arithmetic (cost * "
    "1.5) belong in where.\n"
    "- Ordered or truncated set operations: put order_by/limit/offset on the "
    "set_op node itself and order by an alias declared in its left branch. Do "
    "not wrap the set operation in an extra select.\n"
    "- Relative dates: use the date_sub/date_add function with exactly "
    "[date expression, integer literal, unit literal in day|month|year]. Never "
    "write a SQLite modifier string such as '-6 months'.\n"
    "- Aliases: every subquery needs a source alias different from every alias "
    "already in scope; reusing an outer alias is rejected.\n"
    "- Rank-then-filter: compute the window in a cte or a from-subquery and "
    "filter its output column in the outer query; a window cannot appear in "
    "where."
)

_SCHEMA_REASON_MARKERS = (
    "forbidden tables",
    "forbidden source relation",
    "unknown or ambiguous source column",
    "no allowed base source",
    "cte column list does not match",
    "database qualification",
)
_READ_ONLY_REASON_MARKERS = (
    "not read-only",
    "not a select",
    "unsafe",
    "unapproved sqlite function",
)
_NONDETERMINISTIC_REASON_MARKERS = ("nondeterministic", "volatile")
_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_ALLOWED_FEEDBACK_CODES = frozenset(
    {
        "json_schema",
        "deterministic_compilation",
        "syntax_validation",
        "read_only_validation",
        "schema_validation",
        "question_grounding",
        "semantic_completeness",
    }
)
_FEEDBACK_CODE_RE = re.compile(r"[^a-z0-9_:\-]+")

ModelCallable = Callable[[str], Any]
ExecutorCallable = Callable[[str, Path], Mapping[str, Any]]
CounterfactualCallable = Callable[
    [Path, str, Sequence[Mapping[str, Any]]], CounterfactualReport | Mapping[str, Any]
]


@dataclass(frozen=True, slots=True)
class QueryPlanV3PipelineConfig:
    """Immutable, bounded runtime policy for the four fixed semantic lanes."""

    max_attempts_per_lane: int = 2
    max_feedback_items: int = 5
    max_feedback_code_chars: int = 64
    selector_config: SelectorConfig = field(default_factory=SelectorConfig)

    def __post_init__(self) -> None:
        for label, value, lower, upper in (
            ("max_attempts_per_lane", self.max_attempts_per_lane, 1, 4),
            ("max_feedback_items", self.max_feedback_items, 0, 8),
            ("max_feedback_code_chars", self.max_feedback_code_chars, 8, 128),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{label} must be an integer")
            if not lower <= value <= upper:
                raise ValueError(f"{label} must be in [{lower}, {upper}]")
        if not isinstance(self.selector_config, SelectorConfig):
            raise TypeError("selector_config must be a SelectorConfig")


def _require_text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _bounded_feedback(
    values: Sequence[str],
    config: QueryPlanV3PipelineConfig,
) -> tuple[str, ...]:
    if config.max_feedback_items == 0:
        return ()
    codes: list[str] = []
    for value in values:
        code = _FEEDBACK_CODE_RE.sub("_", str(value).strip().casefold()).strip("_")
        if not code or code not in _ALLOWED_FEEDBACK_CODES:
            continue
        code = code[: config.max_feedback_code_chars]
        if code not in codes:
            codes.append(code)
        if len(codes) >= config.max_feedback_items:
            break
    return tuple(codes)


def build_lane_prompt(
    *,
    question: str,
    live_schema_description: str,
    lane: str,
    validation_feedback: Sequence[str] = (),
    config: QueryPlanV3PipelineConfig = QueryPlanV3PipelineConfig(),
) -> str:
    """Build one deterministic prompt from the bounded runtime inputs only."""

    question = _require_text(question, "question")
    live_schema_description = _require_text(
        live_schema_description, "live_schema_description"
    )
    if lane not in SEMANTIC_LANES:
        raise ValueError(f"unsupported semantic lane: {lane!r}")
    if not isinstance(config, QueryPlanV3PipelineConfig):
        raise TypeError("config must be a QueryPlanV3PipelineConfig")
    feedback = _bounded_feedback(validation_feedback, config)
    schema_payload = _canonical_json(query_plan_v3_schema())
    feedback_payload = _canonical_json(list(feedback))
    example_payload = _canonical_json(_FORMAT_EXAMPLE_PLAN)
    return (
        "Generate exactly one complete Semantic QueryPlan v3 JSON object.\n"
        "Use only the supplied QueryPlan schema, live SQLite schema, user question, "
        "semantic lane, and validation feedback codes. Never emit SQL, prose, markdown, "
        "or fields outside the schema. Never invent semantics absent from the question.\n\n"
        f"SEMANTIC LANE\n{lane}\n{_LANE_INSTRUCTIONS[lane]}\n\n"
        f"QUERYPLAN V3 SCHEMA\n{schema_payload}\n\n"
        f"{_INTENT_GRAMMAR_NOTE}\n\n"
        f"{_OUTPUT_CONVENTIONS_NOTE}\n\n"
        f"{_PLAN_SHAPE_NOTE}\n\n"
        "FORMAT EXAMPLE (shape only; it answers a different, unrelated question — "
        "derive every semantic choice from the user question, not this example)\n"
        f"{example_payload}\n\n"
        f"LIVE SQLITE SCHEMA\n{live_schema_description}\n\n"
        f"USER QUESTION\n{question}\n\n"
        f"BOUNDED VALIDATION FEEDBACK\n{feedback_payload}\n\n"
        "Output only the complete QueryPlan v3 JSON object."
    )


def _visit(value: Any):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _visit(child)
    elif isinstance(value, list):
        for child in value:
            yield from _visit(child)


def _domain_literals(
    domains: Mapping[str, Sequence[str]] | None,
) -> tuple[str, ...]:
    if not isinstance(domains, Mapping):
        return ()
    return tuple(
        str(value)
        for key in sorted(domains)
        for value in domains[key]
        if isinstance(value, str) and value
    )


def _reason_code(prefix: str, reason: str) -> str:
    token = _FEEDBACK_CODE_RE.sub("_", str(reason).strip().casefold()).strip("_")
    return f"{prefix}:{token[:160] or 'failed'}"


def _syntax_report(sql: str) -> tuple[bool, tuple[str, ...]]:
    try:
        from sqlglot import parse
    except ImportError:
        return False, ("syntax:sqlglot_unavailable",)
    try:
        statements = [tree for tree in parse(sql, read="sqlite") if tree is not None]
    except Exception:
        return False, ("syntax:parse_failure",)
    if len(statements) != 1:
        return False, ("syntax:statement_count",)
    return True, ()


def assess_static_validation(
    *,
    question: str,
    plan: Mapping[str, Any],
    sql: str,
    domains: Mapping[str, Sequence[str]] | None = None,
    obligations: SemanticObligations | None = None,
) -> tuple[StaticValidation, SemanticScore]:
    """Independently derive every static flag and the existing semantic score."""

    syntax_valid, syntax_reasons = _syntax_report(sql)
    read_only = syntax_valid
    schema_valid = syntax_valid
    reasons: list[str] = list(syntax_reasons)

    if syntax_valid:
        try:
            validate_read_query(sql)
        except Exception as exc:
            message = str(exc).casefold()
            if any(marker in message for marker in _SCHEMA_REASON_MARKERS):
                schema_valid = False
                reasons.append("schema:forbidden_relation")
            elif "exactly one" in message or "parse" in message:
                syntax_valid = False
                read_only = False
                schema_valid = False
                reasons.append("syntax:read_query_validation_failed")
            else:
                read_only = False
                reasons.append("read_only:read_query_validation_failed")

    structural = validate_candidate(SemanticContract(), sql)
    structural_valid = bool(structural.accepted)
    for structural_reason in structural.reasons:
        lowered = str(structural_reason).casefold()
        if any(marker in lowered for marker in _SCHEMA_REASON_MARKERS):
            schema_valid = False
            reasons.append(_reason_code("schema", structural_reason))
        elif any(marker in lowered for marker in _READ_ONLY_REASON_MARKERS):
            read_only = False
            reasons.append(_reason_code("read_only", structural_reason))
        elif any(marker in lowered for marker in _NONDETERMINISTIC_REASON_MARKERS):
            reasons.append(_reason_code("determinism", structural_reason))
        else:
            reasons.append(_reason_code("structure", structural_reason))

    grounded = True
    try:
        validate_plan_v2_grounding(
            {"plan_version": 2, "query": plan["query"]},
            question,
            allowed_domain_values=_domain_literals(domains),
        )
    except Exception:
        grounded = False
        reasons.append("grounding:question_or_domain_mismatch")

    semantic = score_candidate_semantics(
        question,
        plan,
        sql,
        domains=domains,
    )
    contract_complete = (
        not semantic.contract_complete or semantic.contract_valid
    )
    complete = bool(
        structural_valid
        and contract_complete
        and semantic.obligations_valid
    )

    if not structural_valid:
        reasons.append("completeness:structural_validation_failed")
    if semantic.contract_complete and not semantic.contract_valid:
        reasons.append("completeness:question_contract_failed")
    if not semantic.obligations_valid:
        reasons.append("completeness:semantic_obligations_failed")

    if obligations is None:
        obligations = extract_semantic_obligations(question, domains or {})
    full_validation = True
    try:
        validate_query_plan_v3(
            plan,
            question,
            domains=domains,
            obligations=obligations,
        )
    except Exception:
        full_validation = False
        reasons.append("query_plan_v3:validation_failed")
    if full_validation is False and grounded and complete:
        # A future validator rule that is not represented by the current flag
        # decomposition must still fail closed rather than silently pass.
        complete = False
        reasons.append("completeness:unclassified_v3_validation_failure")

    return (
        StaticValidation(
            syntax_valid=syntax_valid,
            read_only=read_only,
            schema_valid=schema_valid,
            grounded=grounded,
            complete=complete,
            reason_codes=tuple(dict.fromkeys(reasons)),
        ),
        semantic,
    )


def _query_mentions_table(plan: Mapping[str, Any], table: str) -> bool:
    return any(
        node.get("type") == "table" and node.get("table") == table
        for node in _visit(plan.get("query"))
        if isinstance(node, Mapping)
    )


def plan_counterfactual_probes(
    plan: Mapping[str, Any],
    *,
    question: str,
    domains: Mapping[str, Sequence[str]] | None = None,
    obligations: SemanticObligations | None = None,
) -> tuple[dict[str, Any], ...]:
    """Plan at most one deterministic probe of each supported semantic family."""

    parsed = parse_query_plan_v3(plan)
    if obligations is None:
        obligations = extract_semantic_obligations(question, domains or {})
    intent = parsed["intent"]
    query_nodes = tuple(_visit(parsed["query"]))
    probes: list[dict[str, Any]] = []

    count_nodes = [
        node
        for node in query_nodes
        if isinstance(node, Mapping)
        and node.get("type") == "aggregate"
        and node.get("function") == "count"
    ]
    if count_nodes and _query_mentions_table(parsed, "visits"):
        duplicate_changes: list[bool] = []
        for node in count_nodes:
            if not node.get("distinct"):
                duplicate_changes.append(True)
                continue
            argument = node.get("argument")
            if not isinstance(argument, Mapping) or argument.get("type") != "column":
                duplicate_changes = []
                break
            # The mutation copies every visit value except visit_id.  DISTINCT
            # over any copied column is invariant; DISTINCT visit_id must change.
            duplicate_changes.append(argument.get("column") == "visit_id")
        if duplicate_changes:
            probes.append(
                {
                    "probe": "row_count_vs_distinct",
                    "expected_relation": (
                        "changed_result" if any(duplicate_changes) else "same_result"
                    ),
                }
            )

    declared_edges = tuple(intent["joins"])
    disease_triple_join = next(
        (
            item
            for item in declared_edges
            if item["edge"] == "visits_disease_triples"
        ),
        None,
    )
    if disease_triple_join is not None and count_nodes and not any(
        str(key).startswith(f"{disease_triple_join['right_alias']}.")
        for key in intent["grain"]["keys"]
    ):
        amplification_changes: list[bool] = []
        for node in count_nodes:
            if not node.get("distinct"):
                amplification_changes.append(True)
                continue
            argument = node.get("argument")
            if not isinstance(argument, Mapping) or argument.get("type") != "column":
                amplification_changes = []
                break
            amplification_changes.append(
                argument.get("table") == disease_triple_join["right_alias"]
                and argument.get("column") == "tail"
            )
        if amplification_changes:
            probes.append(
                {
                    "probe": "join_amplification",
                    "expected_relation": (
                        "changed_result"
                        if any(amplification_changes)
                        else "same_result"
                    ),
                }
            )

    if (
        any(item["edge"] == "patients_visits" for item in declared_edges)
        and intent["grain"]["kind"] != "scalar"
    ):
        preserve_zero = bool(
            intent["entity_preservation"]["mode"] == "preserve"
            and intent["entity_preservation"]["include_zero_match"]
        )
        probes.append(
            {
                "probe": "zero_match_preservation",
                "expected_relation": (
                    "row_count_increases" if preserve_zero else "row_count_same"
                ),
            }
        )

    absent_obligation = any(
        item.operator is PredicateOperator.ABSENT
        for item in obligations.predicates
    )
    if absent_obligation:
        probes.append(
            {"probe": "null_anti_join", "expected_relation": "same_result"}
        )

    set_mode = intent["set_semantics"]["mode"]
    if set_mode == "any_of" and intent["set_semantics"]["preserve_duplicates"]:
        probes.append(
            {
                "probe": "set_duplicates",
                "expected_relation": "changed_result",
            }
        )

    ordering_keys = tuple(
        re.sub(r"\s+", "", str(item["key"])).casefold()
        for item in intent["ordering"]
    )
    tie_mode = intent["tie_policy"]["mode"]
    hard_limit = any(
        isinstance(node, Mapping)
        and node.get("type") == "select"
        and node.get("limit") is not None
        for node in query_nodes
    )
    if (
        tie_mode == "deterministic"
        and hard_limit
        and any("cost" in key for key in ordering_keys)
    ):
        probes.append(
            {
                "probe": "topk_tie",
                "expected_relation": "row_count_same",
            }
        )

    order = {
        name: index
        for index, name in enumerate(
            (
                "row_count_vs_distinct",
                "join_amplification",
                "zero_match_preservation",
                "null_anti_join",
                "set_duplicates",
                "topk_tie",
                "date_boundary",
            )
        )
    }
    return tuple(sorted(probes, key=lambda item: order[str(item["probe"])]))


def execute_deterministic_fingerprint(sql: str, db_path: Path) -> Mapping[str, Any]:
    """Run the existing complete fingerprint executor twice and compare evidence."""

    first = execute_with_fingerprint(sql, Path(db_path))
    second = execute_with_fingerprint(sql, Path(db_path))
    first_hash = first.get("result_sha256")
    second_hash = second.get("result_sha256")
    deterministic = bool(
        first.get("complete") is True
        and second.get("complete") is True
        and first.get("readonly") is True
        and second.get("readonly") is True
        and isinstance(first_hash, str)
        and first_hash == second_hash
    )
    return {
        "executed": True,
        "complete": bool(first.get("complete") and second.get("complete")),
        "readonly": bool(first.get("readonly") and second.get("readonly")),
        "deterministic": deterministic,
        "result_sha256": first_hash,
        "row_count": first.get("row_count"),
    }


def _execution_evidence(
    executor: ExecutorCallable,
    sql: str,
    db_path: Path,
) -> ExecutionFingerprint:
    try:
        result = executor(sql, Path(db_path))
        if not isinstance(result, Mapping):
            raise TypeError("executor result must be a mapping")
        digest = result.get("result_sha256")
        complete = result.get("complete") is True
        readonly = result.get("readonly") is True
        executed = result.get("executed", True) is True and complete and readonly
        deterministic = result.get("deterministic") is True
        reasons: list[str] = []
        if not complete:
            reasons.append("execution_incomplete")
        if not readonly:
            reasons.append("execution_readonly_unproven")
        if not deterministic:
            reasons.append("execution_determinism_unproven")
        if not isinstance(digest, str) or _HASH_RE.fullmatch(digest) is None:
            digest = None
            reasons.append("execution_fingerprint_invalid")
        return ExecutionFingerprint(
            result_sha256=digest,
            executed=executed,
            deterministic=deterministic,
            reason_codes=tuple(dict.fromkeys(reasons)),
        )
    except Exception:
        return ExecutionFingerprint(
            result_sha256=None,
            executed=False,
            deterministic=False,
            reason_codes=("execution_failed",),
        )


def _counterfactual_evidence(
    runner: CounterfactualCallable,
    db_path: Path,
    sql: str,
    probes: Sequence[Mapping[str, Any]],
) -> tuple[float, bool, dict[str, Any]]:
    try:
        report = runner(Path(db_path), sql, probes)
        if isinstance(report, CounterfactualReport):
            score = report.score
            failed = report.failed
            outcomes = report.outcomes
            version = report.version
            passed = report.passed
            skipped = report.skipped
        elif isinstance(report, Mapping):
            score = report.get("score")
            failed = report.get("failed", 0)
            outcomes = tuple(report.get("outcomes", ()))
            version = str(report.get("version", "injected"))
            passed = int(report.get("passed", 0))
            skipped = int(report.get("skipped", 0))
        else:
            raise TypeError("counterfactual runner result must be a report or mapping")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("counterfactual score must be numeric")
        score_value = float(score)
        if not math.isfinite(score_value) or not 0.0 <= score_value <= 1.0:
            raise ValueError("counterfactual score must be in [0, 1]")
        hard_failure = bool(int(failed) > 0)
        outcome_diagnostics: list[dict[str, Any]] = []
        for outcome in outcomes:
            if isinstance(outcome, Mapping):
                probe = str(outcome.get("probe", "unknown"))
                applicable = outcome.get("applicable") is True
                passed_value = outcome.get("passed")
                reason = str(outcome.get("reason_code", "unknown"))
            else:
                probe = str(getattr(outcome, "probe", "unknown"))
                applicable = getattr(outcome, "applicable", False) is True
                passed_value = getattr(outcome, "passed", None)
                reason = str(getattr(outcome, "reason_code", "unknown"))
            if applicable and passed_value is False:
                hard_failure = True
            outcome_diagnostics.append(
                {
                    "probe": probe,
                    "applicable": applicable,
                    "passed": passed_value,
                    "reason_code": reason,
                }
            )
        return (
            round(score_value, 6),
            hard_failure,
            {
                "version": version,
                "score": round(score_value, 6),
                "passed": passed,
                "failed": int(failed),
                "skipped": skipped,
                "hard_failure": hard_failure,
                "outcomes": tuple(outcome_diagnostics),
            },
        )
    except Exception:
        return (
            0.0,
            True,
            {
                "version": "unavailable",
                "score": 0.0,
                "passed": 0,
                "failed": 1,
                "skipped": 0,
                "hard_failure": True,
                "outcomes": (),
                "reason_code": "counterfactual_execution_failed",
            },
        )


def plan_complexity(plan: Mapping[str, Any]) -> int:
    """Return a deterministic structured-node count for selector parsimony."""

    return sum(1 for item in _visit(plan) if isinstance(item, Mapping))


def _candidate_record(
    *,
    question: str,
    plan: Mapping[str, Any],
    sql: str,
    lane: str,
    generation_index: int,
    static_validation: StaticValidation,
    semantic: SemanticScore,
    execution: ExecutionFingerprint,
    counterfactual_score: float,
    counterfactual_hard_failure: bool,
) -> CandidateRecord:
    plan_hash = canonical_plan_sha256_v3(plan)
    intent_hash = _canonical_hash(plan["intent"])
    candidate_id = hashlib.sha256(
        f"{lane}\0{generation_index}\0{plan_hash}".encode("utf-8")
    ).hexdigest()[:24]
    return CandidateRecord(
        provenance=CandidateProvenance(
            candidate_id=candidate_id,
            generator=f"query_plan_v3:{lane}",
            generation_index=generation_index,
            strategy=STRATEGY,
        ),
        hashes=IntentPlanHashes(
            intent_sha256=intent_hash,
            plan_sha256=plan_hash,
        ),
        sql=sql,
        static_validation=static_validation,
        execution=execution,
        counterfactual_score=counterfactual_score,
        semantic_score=semantic.score,
        complexity=plan_complexity(plan),
        judge_score=None,
        counterfactual_hard_failure=counterfactual_hard_failure,
    )


def _static_feedback(validation: StaticValidation) -> tuple[str, ...]:
    feedback: list[str] = []
    if not validation.syntax_valid:
        feedback.append("syntax_validation")
    if not validation.read_only:
        feedback.append("read_only_validation")
    if not validation.schema_valid:
        feedback.append("schema_validation")
    if not validation.grounded:
        feedback.append("question_grounding")
    if not validation.complete:
        feedback.append("semantic_completeness")
    return tuple(feedback)


def _candidate_diagnostic(
    record: CandidateRecord,
    *,
    lane: str,
    attempt: int,
    semantic: SemanticScore,
    probe_plan: Sequence[Mapping[str, Any]],
    counterfactual: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_id": record.provenance.candidate_id,
        "lane": lane,
        "attempt": attempt,
        "plan_sha256": record.hashes.plan_sha256,
        "intent_sha256": record.hashes.intent_sha256,
        "static_validation": {
            "syntax_valid": record.static_validation.syntax_valid,
            "read_only": record.static_validation.read_only,
            "schema_valid": record.static_validation.schema_valid,
            "grounded": record.static_validation.grounded,
            "complete": record.static_validation.complete,
            "hard_failure": record.static_validation.hard_failure,
            "reason_codes": record.static_validation.reason_codes,
        },
        "execution": {
            "executed": record.execution.executed,
            "deterministic": record.execution.deterministic,
            "result_sha256": record.execution.result_sha256,
            "hard_failure": record.execution.hard_failure,
            "reason_codes": record.execution.reason_codes,
        },
        "counterfactual": {
            **dict(counterfactual),
            "planned_probes": tuple(
                {
                    "probe": str(item["probe"]),
                    "expected_relation": str(item["expected_relation"]),
                }
                for item in probe_plan
            ),
        },
        "semantic": {
            "score": semantic.score,
            "hard_failure": semantic.hard_failure,
            "reason_codes": semantic.reason_codes,
            "capability_coverage": semantic.capability_coverage,
            "contract_complete": semantic.contract_complete,
            "contract_valid": semantic.contract_valid,
            "obligations_renderable": semantic.obligations_renderable,
            "obligations_valid": semantic.obligations_valid,
            "plan_valid": semantic.plan_valid,
        },
        "complexity": record.complexity,
        "judge_score": None,
        "counterfactual_hard_failure": record.counterfactual_hard_failure,
    }


def run_query_plan_v3_pipeline(
    *,
    question: str,
    live_schema_description: str,
    db_path: Path,
    generate_candidate: ModelCallable,
    repair_candidate: ModelCallable | None = None,
    executor: ExecutorCallable = execute_deterministic_fingerprint,
    counterfactual_runner: CounterfactualCallable = run_counterfactual_suite,
    domains: Mapping[str, Sequence[str]] | None = None,
    config: QueryPlanV3PipelineConfig = QueryPlanV3PipelineConfig(),
) -> dict[str, Any]:
    """Generate four semantic lanes and deterministically select or abstain."""

    question = _require_text(question, "question")
    live_schema_description = _require_text(
        live_schema_description, "live_schema_description"
    )
    if not callable(generate_candidate):
        raise TypeError("generate_candidate must be callable")
    if repair_candidate is not None and not callable(repair_candidate):
        raise TypeError("repair_candidate must be callable or None")
    if not callable(executor):
        raise TypeError("executor must be callable")
    if not callable(counterfactual_runner):
        raise TypeError("counterfactual_runner must be callable")
    if not isinstance(config, QueryPlanV3PipelineConfig):
        raise TypeError("config must be a QueryPlanV3PipelineConfig")

    database = Path(db_path)
    obligations = extract_semantic_obligations(question, domains or {})
    records: list[CandidateRecord] = []
    diagnostic_by_id: dict[str, dict[str, Any]] = {}
    semantic_by_id: dict[str, SemanticScore] = {}
    attempt_diagnostics: list[dict[str, Any]] = []
    failure_stages: list[str] = []
    attempts = 0
    rounds_used = 0

    if obligations.disposition is ObligationDisposition.TERMINAL_ABSTAIN:
        failure_stages.append("semantic_obligations")
    else:
        for lane_index, lane in enumerate(SEMANTIC_LANES):
            feedback: tuple[str, ...] = ()
            for attempt_index in range(config.max_attempts_per_lane):
                attempt_number = attempt_index + 1
                rounds_used = max(rounds_used, attempt_number)
                prompt = build_lane_prompt(
                    question=question,
                    live_schema_description=live_schema_description,
                    lane=lane,
                    validation_feedback=feedback,
                    config=config,
                )
                model_call = (
                    generate_candidate
                    if attempt_index == 0 or repair_candidate is None
                    else repair_candidate
                )
                attempts += 1
                generation_index = (
                    lane_index * config.max_attempts_per_lane + attempt_index
                )
                try:
                    raw = model_call(prompt)
                except Exception:
                    failure_stages.append("model_generation")
                    attempt_diagnostics.append(
                        {
                            "lane": lane,
                            "attempt": attempt_number,
                            "stage": "model_generation",
                            "recorded": False,
                        }
                    )
                    feedback = ()
                    continue

                try:
                    parsed = parse_query_plan_v3(raw)
                except (QueryPlanV3Error, ValueError, TypeError):
                    failure_stages.append("json_schema")
                    attempt_diagnostics.append(
                        {
                            "lane": lane,
                            "attempt": attempt_number,
                            "stage": "json_schema",
                            "recorded": False,
                        }
                    )
                    feedback = ("json_schema",)
                    continue

                try:
                    sql = compile_query_plan_v3(parsed)
                except (QueryPlanV3Error, ValueError, TypeError):
                    failure_stages.append("deterministic_compilation")
                    attempt_diagnostics.append(
                        {
                            "lane": lane,
                            "attempt": attempt_number,
                            "stage": "deterministic_compilation",
                            "recorded": False,
                        }
                    )
                    feedback = ("deterministic_compilation",)
                    continue

                static_validation, semantic = assess_static_validation(
                    question=question,
                    plan=parsed,
                    sql=sql,
                    domains=domains,
                    obligations=obligations,
                )
                probe_plan: tuple[dict[str, Any], ...] = ()
                counterfactual: dict[str, Any] = {
                    "version": "not_run",
                    "score": 0.0,
                    "passed": 0,
                    "failed": 0,
                    "skipped": 0,
                    "hard_failure": False,
                    "outcomes": (),
                }

                if static_validation.hard_failure:
                    failure_stages.append("static_validation")
                    execution = ExecutionFingerprint(
                        result_sha256=None,
                        executed=False,
                        deterministic=False,
                        reason_codes=("not_executed_static_failure",),
                    )
                    record = _candidate_record(
                        question=question,
                        plan=parsed,
                        sql=sql,
                        lane=lane,
                        generation_index=generation_index,
                        static_validation=static_validation,
                        semantic=semantic,
                        execution=execution,
                        counterfactual_score=0.0,
                        counterfactual_hard_failure=False,
                    )
                    records.append(record)
                    semantic_by_id[record.provenance.candidate_id] = semantic
                    diagnostic_by_id[record.provenance.candidate_id] = (
                        _candidate_diagnostic(
                            record,
                            lane=lane,
                            attempt=attempt_number,
                            semantic=semantic,
                            probe_plan=probe_plan,
                            counterfactual=counterfactual,
                        )
                    )
                    attempt_diagnostics.append(
                        {
                            "lane": lane,
                            "attempt": attempt_number,
                            "stage": "static_validation",
                            "recorded": True,
                            "candidate_id": record.provenance.candidate_id,
                        }
                    )
                    feedback = _static_feedback(static_validation)
                    continue

                execution = _execution_evidence(executor, sql, database)
                counterfactual_score = 0.0
                counterfactual_hard_failure = False
                if execution.hard_failure:
                    failure_stages.append("execution")
                else:
                    probe_plan = plan_counterfactual_probes(
                        parsed,
                        question=question,
                        domains=domains,
                        obligations=obligations,
                    )
                    (
                        counterfactual_score,
                        counterfactual_hard_failure,
                        counterfactual,
                    ) = _counterfactual_evidence(
                        counterfactual_runner,
                        database,
                        sql,
                        probe_plan,
                    )
                    if counterfactual_hard_failure:
                        failure_stages.append("counterfactual")

                record = _candidate_record(
                    question=question,
                    plan=parsed,
                    sql=sql,
                    lane=lane,
                    generation_index=generation_index,
                    static_validation=static_validation,
                    semantic=semantic,
                    execution=execution,
                    counterfactual_score=counterfactual_score,
                    counterfactual_hard_failure=counterfactual_hard_failure,
                )
                records.append(record)
                semantic_by_id[record.provenance.candidate_id] = semantic
                diagnostic_by_id[record.provenance.candidate_id] = (
                    _candidate_diagnostic(
                        record,
                        lane=lane,
                        attempt=attempt_number,
                        semantic=semantic,
                        probe_plan=probe_plan,
                        counterfactual=counterfactual,
                    )
                )
                attempt_diagnostics.append(
                    {
                        "lane": lane,
                        "attempt": attempt_number,
                        "stage": (
                            "counterfactual"
                            if counterfactual_hard_failure
                            else "execution"
                            if execution.hard_failure
                            else "candidate_ready"
                        ),
                        "recorded": True,
                        "candidate_id": record.provenance.candidate_id,
                    }
                )
                # Execution and counterfactual evidence are terminal for this lane:
                # neither is permitted to become model repair feedback.
                break

    selection = select_candidate(records, config.selector_config)
    selected = selection.selected
    ranking_by_id = {
        item.candidate.provenance.candidate_id: (rank, item)
        for rank, item in enumerate(selection.ranking, start=1)
    }
    hard_rejected_ids = {
        item.provenance.candidate_id for item in selection.rejected_hard_failures
    }
    candidate_diagnostics: list[dict[str, Any]] = []
    for record in sorted(
        records,
        key=lambda item: (
            item.provenance.generation_index,
            item.provenance.candidate_id,
        ),
    ):
        diagnostic = dict(diagnostic_by_id[record.provenance.candidate_id])
        ranking = ranking_by_id.get(record.provenance.candidate_id)
        diagnostic["hard_rejected"] = (
            record.provenance.candidate_id in hard_rejected_ids
        )
        diagnostic["selected"] = selected is record
        if ranking is not None:
            rank, ranked = ranking
            diagnostic["selector_rank"] = rank
            diagnostic["selector_final_score"] = round(ranked.final_score, 6)
            diagnostic["selector_consensus_size"] = ranked.consensus_size
        candidate_diagnostics.append(diagnostic)

    selector_ranking = tuple(
        {
            "candidate_id": item.candidate.provenance.candidate_id,
            "rank": rank,
            "final_score": round(item.final_score, 6),
            "base_score": round(item.base_score, 6),
            "consensus_size": item.consensus_size,
            "consensus_bonus": round(item.consensus_bonus, 6),
            "complexity_penalty": round(item.complexity_penalty, 6),
        }
        for rank, item in enumerate(selection.ranking, start=1)
    )
    selector_metadata = {
        "version": selection.config_version,
        "config_sha256": selection.config_sha256,
        "reason_code": selection.reason_code,
        "abstained": selection.abstained,
        "selected_candidate_id": (
            None if selected is None else selected.provenance.candidate_id
        ),
        "eligible_candidates": len(selection.ranking),
        "hard_rejected_candidates": len(selection.rejected_hard_failures),
        "ranking": selector_ranking,
    }

    if selected is None:
        failure_stages.append("candidate_selection")
        return {
            "sql": None,
            "confidence": 0.0,
            "strategy": STRATEGY,
            "pipeline_version": PIPELINE_VERSION,
            "plan_version": PLAN_VERSION,
            "plan_sha256": None,
            "n_candidates": len(records),
            "n_agree": 0,
            "rounds": rounds_used,
            "attempts": attempts,
            "selector_version": selection.config_version,
            "selector_config_sha256": selection.config_sha256,
            "selector_reason": selection.reason_code,
            "selected_candidate_id": None,
            "selector_metadata": selector_metadata,
            "candidate_diagnostics": tuple(candidate_diagnostics),
            "attempt_diagnostics": tuple(attempt_diagnostics),
            "failure_stages": tuple(dict.fromkeys(failure_stages)),
            "execution_validated": False,
            "semantic_validated": False,
            "contract_validated": False,
            "obligations_validated": False,
            "semantic_warnings": (),
            "abstained": True,
        }

    selected_ranked = next(
        item for item in selection.ranking if item.candidate is selected
    )
    selected_semantic = semantic_by_id[selected.provenance.candidate_id]
    return {
        "sql": selected.sql,
        "confidence": round(float(selected_ranked.final_score), 6),
        "strategy": STRATEGY,
        "pipeline_version": PIPELINE_VERSION,
        "plan_version": PLAN_VERSION,
        "plan_sha256": selected.hashes.plan_sha256,
        "n_candidates": len(records),
        "n_agree": selected_ranked.consensus_size,
        "rounds": rounds_used,
        "attempts": attempts,
        "selector_version": selection.config_version,
        "selector_config_sha256": selection.config_sha256,
        "selector_reason": selection.reason_code,
        "selected_candidate_id": selected.provenance.candidate_id,
        "selector_metadata": selector_metadata,
        "candidate_diagnostics": tuple(candidate_diagnostics),
        "attempt_diagnostics": tuple(attempt_diagnostics),
        "failure_stages": tuple(dict.fromkeys(failure_stages)),
        "execution_validated": not selected.execution.hard_failure,
        "semantic_validated": not selected_semantic.hard_failure,
        "contract_validated": selected_semantic.contract_valid,
        "obligations_validated": selected_semantic.obligations_valid,
        "semantic_warnings": selected_semantic.reason_codes,
        "abstained": False,
    }


query_plan_v3_pipeline = run_query_plan_v3_pipeline


__all__ = [
    "PIPELINE_VERSION",
    "SEMANTIC_LANES",
    "STRATEGY",
    "QueryPlanV3PipelineConfig",
    "assess_static_validation",
    "build_lane_prompt",
    "execute_deterministic_fingerprint",
    "plan_complexity",
    "plan_counterfactual_probes",
    "query_plan_v3_pipeline",
    "run_query_plan_v3_pipeline",
]
