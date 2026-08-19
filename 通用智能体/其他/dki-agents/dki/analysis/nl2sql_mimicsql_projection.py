"""Fail-closed MIMICSQL projection onto the sealed four-table schema.

The projector consumes one already-loaded MIMICSQL ``format`` annotation.  It
does not know where the benchmark files live and therefore cannot open an
official test split by itself.  Development builders may pass train/dev rows;
the blind consumer may pass test rows only after its create-once consumption
marker has been published.

Only concepts with a defensible target equivalent are admitted.  Conditions
are rebound to a deterministic row in the sealed database, every SQL predicate
literal must remain explicit in the projected question, and the rendered query
must execute successfully through the shared read-only protocol.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from dki.analysis.nl2sql_css_projection import (
    ProjectedItem,
    TargetDomains,
)
from dki.analysis.nl2sql_protocol import (
    execute_complete_readonly,
    normalise_question,
    validate_gold_sql,
    validate_question_sql_grounding,
)


MIMICSQL_BENCHMARK = "MIMICSQL-natural-v2"
MIMICSQL_SOURCE_REVISION = "1c3bda58b08fce0015ae1c6680cc928b3c2580cf"
MIMICSQL_TEST_GIT_BLOB = "87767d6aef47c088d9374946e99297d5753704be"
MIMICSQL_PROJECTION_SEED = "20260726-mimicsql-four-table-v1"
NL2SQL_V3_PARTITION_SEED = "20260726-authoritative-public-v3"

SOURCE_TABLES = {
    0: "demographic",
    1: "diagnoses",
    2: "procedures",
    3: "prescriptions",
    4: "lab",
}
SOURCE_COLUMNS: dict[tuple[int, int], str] = {
    (0, 0): "patient_id",
    (0, 2): "name",
    (0, 4): "age",
    (0, 6): "gender",
    (0, 10): "los_days",
    (0, 16): "disease",
    (1, 3): "disease",
    (1, 4): "disease",
    (2, 3): "exam",
    (2, 4): "exam",
    (3, 4): "drug_prescribed",
    (4, 6): "exam",
}
TARGET_COLUMNS = {
    "patient_id": "p.patient_id",
    "name": "p.name",
    "gender": "p.gender",
    "age": "p.age",
    "los_days": "v.los_days",
    "disease": "v.disease",
    "drug_prescribed": "v.drug_prescribed",
    "exam": "v.exam",
}
PATIENT_CONCEPTS = {"patient_id", "name", "gender", "age"}
VISIT_CONCEPTS = {"los_days", "disease", "drug_prescribed", "exam"}
NUMERIC_CONCEPTS = {"age", "los_days"}
CONDITION_OPERATORS = ("=", ">", "<", ">=", "<=")
AGGREGATIONS = ("", "COUNT", "MAX", "MIN", "AVG")


class MIMICSQLProjectionRejected(ValueError):
    """A stable, aggregate-safe reason for rejecting a source row."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class MIMICCondition:
    source_table: int
    source_column: int
    concept: str
    operator: str
    original: Any


@dataclass(frozen=True)
class LiteralBinding:
    original: Any
    replacement: Any
    concept: str


def _stable_int(*parts: Any) -> int:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest()[:8], "big")


def _source_row_sha256(row: Mapping[str, Any]) -> str:
    payload = json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _concept(table_index: Any, column_index: Any) -> str:
    try:
        identity = (int(table_index), int(column_index))
    except (TypeError, ValueError) as exc:
        raise MIMICSQLProjectionRejected("invalid_source_column") from exc
    if identity[0] not in SOURCE_TABLES:
        raise MIMICSQLProjectionRejected("unsupported_source_table")
    concept = SOURCE_COLUMNS.get(identity)
    if concept is None:
        raise MIMICSQLProjectionRejected("unsupported_source_column")
    return concept


def _normalise_annotation(
    row: Mapping[str, Any],
) -> tuple[str, list[tuple[int, int, str]], list[MIMICCondition], int]:
    key = str(row.get("key", "")).strip()
    if not re.fullmatch(r"[0-9a-fA-F]{16,128}", key):
        raise MIMICSQLProjectionRejected("invalid_source_key")
    question = str(row.get("question_refine", "")).strip()
    if not question:
        raise MIMICSQLProjectionRejected("empty_question")
    annotation = row.get("format")
    if not isinstance(annotation, Mapping):
        raise MIMICSQLProjectionRejected("missing_structured_format")

    raw_tables = annotation.get("table")
    if (
        not isinstance(raw_tables, list)
        or not raw_tables
        or any(isinstance(value, bool) for value in raw_tables)
    ):
        raise MIMICSQLProjectionRejected("invalid_source_tables")
    try:
        tables = [int(value) for value in raw_tables]
    except (TypeError, ValueError) as exc:
        raise MIMICSQLProjectionRejected("invalid_source_tables") from exc
    if len(set(tables)) != len(tables) or any(
        table not in SOURCE_TABLES for table in tables
    ):
        raise MIMICSQLProjectionRejected("invalid_source_tables")

    raw_select = annotation.get("agg_col")
    if not isinstance(raw_select, list) or not raw_select:
        raise MIMICSQLProjectionRejected("invalid_aggregation_columns")
    select_columns: list[tuple[int, int, str]] = []
    for entry in raw_select:
        if not isinstance(entry, list) or len(entry) != 2:
            raise MIMICSQLProjectionRejected("invalid_aggregation_column")
        try:
            table_index = int(entry[0])
            column_index = int(entry[1])
        except (TypeError, ValueError) as exc:
            raise MIMICSQLProjectionRejected(
                "invalid_aggregation_column"
            ) from exc
        select_columns.append(
            (table_index, column_index, _concept(table_index, column_index))
        )
    if len({concept for _, _, concept in select_columns}) != len(
        select_columns
    ):
        raise MIMICSQLProjectionRejected("duplicate_target_select_concept")

    raw_conditions = annotation.get("cond")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise MIMICSQLProjectionRejected("invalid_conditions")
    conditions: list[MIMICCondition] = []
    for entry in raw_conditions:
        if not isinstance(entry, list) or len(entry) != 4:
            raise MIMICSQLProjectionRejected("invalid_condition")
        try:
            table_index = int(entry[0])
            column_index = int(entry[1])
            operator_index = int(entry[2])
        except (TypeError, ValueError) as exc:
            raise MIMICSQLProjectionRejected("invalid_condition") from exc
        if operator_index < 0 or operator_index >= len(CONDITION_OPERATORS):
            raise MIMICSQLProjectionRejected(
                "unsupported_condition_operator"
            )
        conditions.append(
            MIMICCondition(
                source_table=table_index,
                source_column=column_index,
                concept=_concept(table_index, column_index),
                operator=CONDITION_OPERATORS[operator_index],
                original=entry[3],
            )
        )
    condition_concepts = [condition.concept for condition in conditions]
    if len(set(condition_concepts)) != len(condition_concepts):
        # Primary diagnosis/diagnosis-title and procedure/lab concepts collapse
        # onto one target column.  Keeping both would silently replace a source
        # join relation with two predicates over one attribute.
        raise MIMICSQLProjectionRejected(
            "duplicate_target_condition_concept"
        )

    try:
        aggregation_index = int(annotation.get("sel"))
    except (TypeError, ValueError) as exc:
        raise MIMICSQLProjectionRejected("invalid_aggregation") from exc
    if aggregation_index < 0 or aggregation_index >= len(AGGREGATIONS):
        raise MIMICSQLProjectionRejected("invalid_aggregation")
    if aggregation_index == 1 and (
        len(select_columns) != 1
        or select_columns[0][:2] != (0, 0)
    ):
        raise MIMICSQLProjectionRejected(
            "unsupported_count_target"
        )
    if aggregation_index in {2, 3, 4} and (
        len(select_columns) != 1
        or select_columns[0][2] not in NUMERIC_CONCEPTS
    ):
        raise MIMICSQLProjectionRejected(
            "unsupported_numeric_aggregation"
        )

    referenced_tables = {
        *(table for table, _, _ in select_columns),
        *(condition.source_table for condition in conditions),
    }
    if set(tables) != referenced_tables:
        raise MIMICSQLProjectionRejected("source_table_binding_mismatch")
    return question, select_columns, conditions, aggregation_index


def _question_mentions_concept(question: str, concept: str) -> bool:
    patterns = {
        "patient_id": r"\b(?:patient|subject)\s*(?:id|identifier)\b|"
        r"\bwhich patients?\b|\blist (?:the )?patients?\b",
        "name": r"\bname\b|\bwho\b|\bwhich patient\b",
        "gender": r"\b(?:gender|sex|female|male|women|woman|men|man)\b",
        "age": r"\b(?:age|aged|years? old|oldest|youngest|older|younger)\b",
        "los_days": r"\b(?:hospital stay|stay(?:ed)?|hospitali[sz]ed|"
        r"admitted|duration|days? in (?:the )?hospital)\b",
        "disease": r"\b(?:disease|diagnos\w*|condition|suffering|"
        r"affected by|have|has)\b",
        "drug_prescribed": r"\b(?:drug|medication|prescri\w*|treated|"
        r"administer\w*|given|using|use)\b",
        "exam": r"\b(?:procedure|exam|test|laboratory|lab|underwent|"
        r"had|measure|checked)\b",
    }
    return bool(re.search(patterns[concept], question, flags=re.IGNORECASE))


def _validate_question_intent(
    question: str,
    select_columns: Sequence[tuple[int, int, str]],
    conditions: Sequence[MIMICCondition],
    aggregation_index: int,
) -> None:
    lowered = unicodedata.normalize("NFKC", question).casefold()
    aggregation = AGGREGATIONS[aggregation_index]
    count_cue = re.search(
        r"\b(?:how many|count|number of|what number|total number|"
        r"calculate (?:the )?(?:total )?number|"
        r"(?:give|provide|find|report|get|tell|let me know) "
        r"(?:me )?(?:the )?(?:total )?number)\b",
        lowered,
    )
    if aggregation == "COUNT" and not count_cue:
        raise MIMICSQLProjectionRejected("count_intent_not_grounded")
    aggregate_patterns = {
        "MAX": r"\b(?:maximum|max|highest|oldest|longest|most)\b",
        "MIN": r"\b(?:minimum|min|lowest|youngest|shortest|least)\b",
        "AVG": r"\b(?:average|avg|mean)\b",
    }
    if aggregation in aggregate_patterns and not re.search(
        aggregate_patterns[aggregation], lowered
    ):
        raise MIMICSQLProjectionRejected(
            "aggregation_intent_not_grounded"
        )
    if aggregation == "":
        for _, _, concept in select_columns:
            if not _question_mentions_concept(lowered, concept):
                raise MIMICSQLProjectionRejected(
                    "selected_concept_not_grounded"
                )
    for condition in conditions:
        if not _question_mentions_concept(lowered, condition.concept):
            raise MIMICSQLProjectionRejected(
                "condition_concept_not_grounded"
            )


def _anchor_for(
    domains: TargetDomains,
    source_sha256: str,
    attempt: int,
) -> Mapping[str, Any]:
    index = (
        _stable_int(
            MIMICSQL_PROJECTION_SEED,
            source_sha256,
            attempt,
        )
        % len(domains.joined_rows)
    )
    return domains.joined_rows[index]


def _numeric_replacement(
    concept: str,
    operator: str,
    anchor: Mapping[str, Any],
) -> int:
    raw = anchor.get(concept)
    if raw is None:
        raise MIMICSQLProjectionRejected("empty_target_literal")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise MIMICSQLProjectionRejected(
            "invalid_target_numeric_literal"
        ) from exc
    if operator == ">":
        if value <= 0:
            raise MIMICSQLProjectionRejected(
                "numeric_anchor_cannot_satisfy_operator"
            )
        return value - 1
    if operator == "<":
        return value + 1
    return value


def _replacement_for(
    condition: MIMICCondition,
    anchor: Mapping[str, Any],
) -> Any:
    if condition.concept in NUMERIC_CONCEPTS:
        try:
            float(condition.original)
        except (TypeError, ValueError) as exc:
            raise MIMICSQLProjectionRejected(
                "source_literal_type_mismatch"
            ) from exc
        return _numeric_replacement(
            condition.concept,
            condition.operator,
            anchor,
        )
    if condition.operator != "=":
        raise MIMICSQLProjectionRejected(
            "non_numeric_inequality_not_supported"
        )
    replacement = anchor.get(condition.concept)
    if replacement is None or str(replacement).strip() == "":
        raise MIMICSQLProjectionRejected("empty_target_literal")
    return replacement


def _sql_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return f"{value:g}" if isinstance(value, float) else str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _literal_patterns(binding: LiteralBinding) -> list[re.Pattern[str]]:
    raw = unicodedata.normalize("NFKC", str(binding.original).strip())
    if binding.concept == "gender":
        value = raw.casefold()
        if value in {"f", "female"}:
            variants = ("female", "females", "woman", "women", "f")
        elif value in {"m", "male"}:
            variants = ("male", "males", "man", "men", "m")
        else:
            raise MIMICSQLProjectionRejected(
                "unsupported_source_gender_literal"
            )
        return [
            re.compile(
                r"(?<![A-Za-z])(?:" + "|".join(variants) + r")(?![A-Za-z])",
                flags=re.IGNORECASE,
            )
        ]
    variants = [raw]
    try:
        numeric = float(raw)
    except ValueError:
        numeric = None
    if numeric is not None and numeric.is_integer():
        variants.append(str(int(numeric)))
    patterns: list[re.Pattern[str]] = []
    for variant in sorted(set(variants), key=len, reverse=True):
        pieces = [
            re.escape(piece)
            for piece in re.split(r"\s+", variant)
            if piece
        ]
        if not pieces:
            continue
        body = r"\s+".join(pieces)
        if variant[0].isalnum():
            body = r"(?<![\w.])" + body
        if variant[-1].isalnum():
            body += r"(?![\w.])"
        patterns.append(re.compile(body, flags=re.IGNORECASE))
    return patterns


def _replace_question_literals(
    question: str,
    bindings: Sequence[LiteralBinding],
) -> str:
    result = unicodedata.normalize("NFKC", question)
    replacement_by_literal: dict[str, str] = {}
    for binding in bindings:
        identity = unicodedata.normalize(
            "NFKC", str(binding.original)
        ).casefold()
        replacement = str(binding.replacement)
        previous = replacement_by_literal.get(identity)
        if previous is not None and previous != replacement:
            raise MIMICSQLProjectionRejected(
                "ambiguous_source_literal_binding"
            )
        replacement_by_literal[identity] = replacement
        replaced = False
        for pattern in _literal_patterns(binding):
            result, count = pattern.subn(
                lambda _match, value=replacement: value,
                result,
            )
            replaced = replaced or count > 0
        if not replaced:
            raise MIMICSQLProjectionRejected(
                "source_literal_not_found_in_question"
            )
    return re.sub(r"\s+", " ", result).strip()


def _from_sql(concepts: set[str]) -> str:
    uses_patient = bool(concepts & PATIENT_CONCEPTS)
    uses_visit = bool(concepts & VISIT_CONCEPTS)
    if uses_patient and uses_visit:
        return "patients p JOIN visits v ON p.patient_id = v.patient_id"
    if uses_patient:
        return "patients p"
    if uses_visit:
        return "visits v"
    raise MIMICSQLProjectionRejected("query_has_no_mapped_table")


def _render(
    select_columns: Sequence[tuple[int, int, str]],
    conditions: Sequence[MIMICCondition],
    aggregation_index: int,
    anchor: Mapping[str, Any],
) -> tuple[str, list[LiteralBinding], str, str]:
    aggregation = AGGREGATIONS[aggregation_index]
    selected = [TARGET_COLUMNS[concept] for _, _, concept in select_columns]
    if aggregation == "COUNT":
        select_sql = f"COUNT(DISTINCT {selected[0]})"
    elif aggregation:
        select_sql = f"{aggregation}({selected[0]})"
    else:
        select_sql = ", ".join(selected)

    bindings: list[LiteralBinding] = []
    predicates: list[str] = []
    concepts = {concept for _, _, concept in select_columns}
    for condition in conditions:
        replacement = _replacement_for(condition, anchor)
        bindings.append(
            LiteralBinding(
                original=condition.original,
                replacement=replacement,
                concept=condition.concept,
            )
        )
        concepts.add(condition.concept)
        predicates.append(
            f"{TARGET_COLUMNS[condition.concept]} "
            f"{condition.operator} {_sql_literal(replacement)}"
        )
    query = (
        f"SELECT {select_sql} FROM {_from_sql(concepts)} "
        f"WHERE {' AND '.join(predicates)}"
    )
    uses_join = bool(concepts & PATIENT_CONCEPTS) and bool(
        concepts & VISIT_CONCEPTS
    )
    if uses_join:
        category = "multi_table_join"
    elif aggregation:
        category = "aggregation_grouping"
    else:
        category = "filtering"
    complexity = (
        len(conditions)
        + (1 if uses_join else 0)
        + (1 if aggregation else 0)
    )
    if complexity >= 4:
        difficulty = "extra"
    elif complexity >= 2:
        difficulty = "hard"
    else:
        difficulty = "medium"
    return query, bindings, category, difficulty


def skeleton_role_v3(skeleton: str) -> str:
    """Partition complete target AST families before test is consumed."""
    digest = hashlib.sha256(
        f"{NL2SQL_V3_PARTITION_SEED}\0{skeleton}".encode("utf-8")
    ).hexdigest()
    return "holdout" if int(digest[:8], 16) % 20 < 13 else "development"


def project_mimicsql_row(
    row: Mapping[str, Any],
    domains: TargetDomains,
    db_path: Path,
    *,
    source_split: str,
    required_role: str | None = None,
    source_revision: str = MIMICSQL_SOURCE_REVISION,
    max_attempts: int = 64,
) -> ProjectedItem:
    """Project one MIMICSQL row or reject it without semantic guessing."""
    if source_split not in {"train", "dev", "test"}:
        raise ValueError("source_split must be train, dev, or test")
    if required_role not in {None, "development", "holdout"}:
        raise ValueError("required_role must be development, holdout, or None")
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("source_revision must be a lowercase Git SHA-1")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if not isinstance(row, Mapping):
        raise MIMICSQLProjectionRejected("source_row_not_mapping")

    question, select_columns, conditions, aggregation_index = (
        _normalise_annotation(row)
    )
    _validate_question_intent(
        question,
        select_columns,
        conditions,
        aggregation_index,
    )
    source_sha = _source_row_sha256(row)
    last_reason = "no_nonempty_projection"
    for attempt in range(max_attempts):
        try:
            anchor = _anchor_for(domains, source_sha, attempt)
            gold_sql, bindings, category, difficulty = _render(
                select_columns,
                conditions,
                aggregation_index,
                anchor,
            )
            projected_question = _replace_question_literals(
                question,
                bindings,
            )
            skeleton = validate_gold_sql(gold_sql)
            role = skeleton_role_v3(skeleton)
            if required_role is not None and role != required_role:
                raise MIMICSQLProjectionRejected(
                    "skeleton_reserved_for_other_role"
                )
            validate_question_sql_grounding(
                projected_question,
                gold_sql,
            )
            _, result_rows = execute_complete_readonly(gold_sql, db_path)
            if not result_rows:
                raise MIMICSQLProjectionRejected("empty_result")
            if (
                AGGREGATIONS[aggregation_index] == "COUNT"
                and (
                    result_rows[0][0] is None
                    or int(result_rows[0][0]) <= 0
                )
            ):
                raise MIMICSQLProjectionRejected("empty_count_result")

            normalized = normalise_question(projected_question)
            identity = hashlib.sha256(
                (
                    f"{MIMICSQL_PROJECTION_SEED}\0{source_split}\0"
                    f"{source_sha}\0{normalized}\0{gold_sql}"
                ).encode("utf-8")
            ).hexdigest()
            skeleton_sha256 = hashlib.sha256(
                skeleton.encode("utf-8")
            ).hexdigest()
            item = {
                "id": f"mimicsql-{source_split}-{identity[:20]}",
                "question": projected_question,
                "gold_sql": gold_sql,
                "difficulty": difficulty,
                "category": category,
                "provenance": {
                    "benchmark": MIMICSQL_BENCHMARK,
                    "source_revision": source_revision,
                    "source_split": source_split,
                    "source_key_sha256": hashlib.sha256(
                        str(row["key"]).encode("utf-8")
                    ).hexdigest(),
                    "source_sql_sha256": hashlib.sha256(
                        str(row.get("sql", "")).encode("utf-8")
                    ).hexdigest(),
                    "source_row_sha256": source_sha,
                    "projection_seed": MIMICSQL_PROJECTION_SEED,
                    "partition_seed": NL2SQL_V3_PARTITION_SEED,
                    "literal_attempt": attempt,
                    "skeleton_sha256": skeleton_sha256,
                    "skeleton_role": role,
                },
            }
            return ProjectedItem(
                item=item,
                skeleton_role=role,
                skeleton_sha256=skeleton_sha256,
            )
        except MIMICSQLProjectionRejected as exc:
            last_reason = exc.reason
            if exc.reason == "skeleton_reserved_for_other_role":
                raise
            if exc.reason not in {
                "empty_result",
                "empty_count_result",
                "empty_target_literal",
                "numeric_anchor_cannot_satisfy_operator",
            }:
                raise
        except (RuntimeError, ValueError, sqlite3.Error) as exc:
            raise MIMICSQLProjectionRejected(
                f"validation_or_execution:{type(exc).__name__}"
            ) from exc
    raise MIMICSQLProjectionRejected(last_reason)


def deduplicate_mimicsql_items(
    items: Iterable[ProjectedItem],
) -> tuple[list[ProjectedItem], dict[str, int]]:
    """Drop duplicate questions and fail closed on ambiguous annotations."""
    groups: dict[str, list[ProjectedItem]] = {}
    ids: dict[str, ProjectedItem] = {}
    for projected in items:
        item_id = str(projected.item["id"])
        if item_id in ids and ids[item_id] != projected:
            raise ValueError(f"conflicting projected item id: {item_id}")
        ids[item_id] = projected
        normalized = normalise_question(str(projected.item["question"]))
        groups.setdefault(normalized, []).append(projected)
    selected: list[ProjectedItem] = []
    exact_duplicates = 0
    ambiguous_questions = 0
    ambiguous_rows = 0
    for group in groups.values():
        gold_skeletons = {
            validate_gold_sql(str(projected.item["gold_sql"]))
            for projected in group
        }
        if len(gold_skeletons) != 1:
            ambiguous_questions += 1
            ambiguous_rows += len(group)
            continue
        ordered = sorted(group, key=lambda value: value.item["id"])
        selected.append(ordered[0])
        exact_duplicates += len(ordered) - 1
    return (
        sorted(selected, key=lambda value: value.item["id"]),
        {
            "exact_duplicate_rows_removed": exact_duplicates,
            "ambiguous_questions_removed": ambiguous_questions,
            "ambiguous_rows_removed": ambiguous_rows,
        },
    )
