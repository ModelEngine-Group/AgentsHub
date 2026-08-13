"""Fail-closed EHRSQL projection onto the sealed four-table medical schema.

Only official train/validation rows may be used for development selection.
The same deterministic code can later be applied to official test rows inside
the one-shot freeze process.  It never imports the project's NL2SQL engine.

The projection keeps upstream English paraphrases, replaces only typed values
with values from the target database, and admits a row only when its semantics
have an unambiguous four-table equivalent.  Temporal source rows are rejected
in this version; silently dropping a time condition would create a fluent but
incorrect benchmark item.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from dki.analysis.nl2sql_css_projection import (
    PROJECTION_SEED_V2,
    ProjectedItem,
    skeleton_role_v2,
)
from dki.analysis.nl2sql_protocol import (
    execute_complete_readonly,
    normalise_question,
    validate_gold_sql,
    validate_question_sql_grounding,
)
from dki.analysis.nl2sql_sqlite import connect_readonly


TIME_SQL_PATTERN = re.compile(
    r"current_time|strftime\s*\(|datetime\s*\(|"
    r"(?:admittime|dischargetime|starttime|treatmenttime|diagnosistime|"
    r"labresulttime|observationtime|culturetakentime|chargetime)\b",
    re.IGNORECASE,
)

TIME_QUESTION_PATTERN = re.compile(
    r"\b(?:first|last|current|previous|most recent)\s+"
    r"(?:hospital\s+)?(?:visit|encounter|admission)\b|"
    r"\b(?:when\s+(?:they|the patient)\s+(?:first|last)\s+visited|"
    r"(?:first|last)\s+time)\b|"
    r"\b(?:within|during|over|in)\s+(?:the\s+)?"
    r"(?:last|past|previous|first|current)\s+\d+\s*"
    r"(?:hours?|days?|weeks?|months?|years?)\b|"
    r"\b(?:less than|under|within)\s+\d+\s*"
    r"(?:hours?|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)

TAG_KIND: dict[str, str] = {
    "what is the cost of diagnosing {diagnosis_name}?": "cost_disease",
    "what is the cost of a drug named {drug_name}?": "cost_drug",
    "what is the cost of a {lab_name} lab test?": "cost_exam",
    "what is the cost of a procedure named {procedure_name}?": "cost_exam",
    "what is the gender of patient {patient_id}?": "patient_gender",
    "what is_verb the age of patient {patient_id} [time_filter_global1]?": "patient_age",
    "count the number of {procedure_name} procedure cases [time_filter_global1].": "count_exam_cases",
    "count the number of {drug_name} prescription cases [time_filter_global1].": "count_drug_cases",
    "count the number of patients who were diagnosed with {diagnosis_name} [time_filter_global1].": "count_disease_patients",
    "count the number of patients who were prescribed {drug_name} [time_filter_global1].": "count_drug_patients",
    "count the number of patients who received a {procedure_name} procedure [time_filter_global1].": "count_exam_patients",
    "count the number of patients who received a {lab_name} lab test [time_filter_global1].": "count_exam_patients",
    "list the ids of patients diagnosed with {diagnosis_name} [time_filter_global1].": "list_disease_patients",
    "has_verb patient {patient_id} been diagnosed with {diagnosis_name} [time_filter_global1]?": "has_disease",
    "has_verb patient {patient_id} been prescribed {drug_name} [time_filter_global1]?": "has_drug",
    "has_verb patient {patient_id} received a {procedure_name} procedure [time_filter_global1]?": "has_exam",
    "has_verb patient {patient_id} received a {lab_name} lab test [time_filter_global1]?": "has_exam",
    "has_verb patient {patient_id} received any diagnosis [time_filter_global1]?": "has_any_disease",
    "has_verb patient {patient_id} been prescribed any medication [time_filter_global1]?": "has_any_drug",
    "has_verb patient {patient_id} received any procedure [time_filter_global1]?": "has_any_exam",
    "has_verb patient {patient_id} received any lab test [time_filter_global1]?": "has_any_exam",
    "what is_verb the total hospital cost of patient {patient_id} [time_filter_global1]?": "total_cost_patient",
    "what is_verb the [agg_function] total hospital cost that involves a diagnosis named {diagnosis_name} [time_filter_global1]?": "aggregate_cost_disease",
    "what is_verb the [agg_function] total hospital cost that involves a drug named {drug_name} [time_filter_global1]?": "aggregate_cost_drug",
    "what is_verb the [agg_function] total hospital cost that involves a procedure named {procedure_name} [time_filter_global1]?": "aggregate_cost_exam",
    "what is_verb the [agg_function] total hospital cost that involves a {lab_name} lab test [time_filter_global1]?": "aggregate_cost_exam",
    "count the number of drugs patient {patient_id} were prescribed [time_filter_global1].": "count_patient_drugs",
    "count the number of times that patient {patient_id} were prescribed {drug_name} [time_filter_global1].": "count_patient_drug",
    "count the number of times that patient {patient_id} received a {procedure_name} procedure [time_filter_global1].": "count_patient_exam",
    "count the number of hospital visits of patient {patient_id} [time_filter_global1].": "count_patient_visits",
    "has_verb patient {patient_id} been admitted to the hospital [time_filter_global1]?": "has_patient_visit",
    "what are_verb the top [n_rank] frequently prescribed drugs [time_filter_global1]?": "top_drugs",
    "what are_verb the top [n_rank] frequent diagnoses [time_filter_global1]?": "top_diseases",
    "what are_verb the top [n_rank] frequent procedures [time_filter_global1]?": "top_exams",
    "what are_verb the top [n_rank] frequent lab tests [time_filter_global1]?": "top_exams",
    "what are_verb the top [n_rank] frequently prescribed drugs of patients aged [age_group] [time_filter_global1]?": "top_drugs_by_age",
    "what are_verb the top [n_rank] frequent diagnoses of patients aged [age_group] [time_filter_global1]?": "top_diseases_by_age",
    "what are_verb the top [n_rank] frequent procedures of patients aged [age_group] [time_filter_global1]?": "top_exams_by_age",
    "what are_verb the top [n_rank] frequent lab tests of patients aged [age_group] [time_filter_global1]?": "top_exams_by_age",
}

VALUE_TARGET = {
    "patient_id": "patient_id",
    "diagnosis_name": "disease",
    "diagnosis_name1": "disease",
    "diagnosis_name2": "disease",
    "drug_name": "drug_prescribed",
    "drug_name1": "drug_prescribed",
    "drug_name2": "drug_prescribed",
    "drug_name3": "drug_prescribed",
    "procedure_name": "exam",
    "procedure_name1": "exam",
    "procedure_name2": "exam",
    "lab_name": "exam",
    "culture_name": "exam",
}

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


class EHRProjectionRejected(ValueError):
    """A fail-closed rejection with an aggregate-safe reason."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class EHRTargetDomains:
    joined_rows: tuple[Mapping[str, Any], ...]


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_int(*parts: Any) -> int:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest()[:8], "big")


def load_ehr_target_domains(db_path: Path) -> EHRTargetDomains:
    connection = connect_readonly(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = tuple(
            dict(row)
            for row in connection.execute(
                """
                SELECT
                    p.patient_id,
                    p.gender,
                    p.age,
                    v.visit_id,
                    v.visit_date,
                    v.department,
                    v.disease,
                    v.drug_prescribed,
                    v.exam,
                    v.cost,
                    v.los_days
                FROM visits v
                JOIN patients p ON p.patient_id = v.patient_id
                WHERE
                    length(v.disease) > 0
                    AND length(v.drug_prescribed) > 0
                    AND length(v.exam) > 0
                ORDER BY v.visit_id
                """
            )
        )
    finally:
        connection.close()
    if not rows:
        raise ValueError("sealed target database has no complete visit anchors")
    return EHRTargetDomains(joined_rows=rows)


def _sql_string(value: Any) -> str:
    return str(value).replace("'", "''")


def _anchor(
    domains: EHRTargetDomains,
    source_sha256: str,
    attempt: int,
    offset: int = 0,
) -> Mapping[str, Any]:
    index = (
        _stable_int(PROJECTION_SEED_V2, source_sha256, attempt) + offset * 37
    ) % len(domains.joined_rows)
    return domains.joined_rows[index]


def _replace_once_casefold(question: str, source: Any, target: Any) -> str:
    source_text = str(source).strip()
    if not source_text:
        raise EHRProjectionRejected("empty_source_value")
    pattern = re.compile(re.escape(source_text), re.IGNORECASE)
    replaced, count = pattern.subn(lambda _: str(target), question)
    if count == 0:
        raise EHRProjectionRejected("source_value_not_found_in_question")
    return replaced


def _replace_typed_values(
    row: Mapping[str, Any],
    question: str,
    anchor: Mapping[str, Any],
    domains: EHRTargetDomains,
    source_sha256: str,
    attempt: int,
) -> tuple[str, dict[str, Any]]:
    values = row.get("value")
    if not isinstance(values, Mapping):
        raise EHRProjectionRejected("missing_value_map")
    replacements: dict[str, Any] = {}
    result = question
    for position, key in enumerate(sorted(values)):
        if key not in VALUE_TARGET:
            continue
        target_column = VALUE_TARGET[key]
        selected = (
            anchor
            if position == 0
            else _anchor(domains, source_sha256, attempt, position)
        )
        replacement = selected.get(target_column)
        if replacement is None or str(replacement) == "":
            raise EHRProjectionRejected("empty_target_value")
        result = _replace_once_casefold(result, values[key], replacement)
        replacements[key] = replacement
    return re.sub(r"\s+", " ", result).strip(), replacements


def _require(replacements: Mapping[str, Any], *keys: str) -> list[Any]:
    result = []
    for key in keys:
        if key not in replacements:
            raise EHRProjectionRejected(f"missing_typed_value:{key}")
        result.append(replacements[key])
    return result


def _source_aggregate(source_sql: str) -> str:
    match = re.search(r"\b(sum|avg|max|min)\s*\(", source_sql, re.IGNORECASE)
    if not match:
        raise EHRProjectionRejected("source_aggregate_missing")
    return match.group(1).upper()


def _source_limit(source_sql: str) -> int:
    match = re.search(
        r"(?:\bc1\s*<=|\blimit)\s*(\d+)",
        source_sql,
        re.IGNORECASE,
    )
    if not match:
        raise EHRProjectionRejected("source_top_limit_missing")
    return max(1, min(int(match.group(1)), 10))


def _ground_limit(question: str, limit: int) -> str:
    result = question
    number_forms = [str(limit)]
    number_forms.extend(
        word for word, value in NUMBER_WORDS.items() if value == limit
    )
    ranked_phrase = "|".join(re.escape(value) for value in number_forms)
    result, count = re.subn(
        rf"\bthe\s+(?:{ranked_phrase})\s+most\s+",
        f"the top {limit} ",
        result,
        count=1,
        flags=re.IGNORECASE,
    )
    if count:
        return result
    for word, value in NUMBER_WORDS.items():
        if value == limit:
            result, count = re.subn(
                rf"\b{word}\b",
                str(limit),
                result,
                count=1,
                flags=re.IGNORECASE,
            )
            if count:
                return result
    if re.search(rf"\b{limit}\b", result):
        return result
    return f"{result.rstrip(' ?.')}; return the top {limit}."


def _source_age_range(source_sql: str) -> tuple[int, int]:
    between = re.search(
        r"\bage\s+between\s+(\d+)\s+and\s+(\d+)",
        source_sql,
        re.IGNORECASE,
    )
    if between:
        return int(between.group(1)), int(between.group(2))
    minimum = re.search(r"\bage\s*>=\s*(\d+)", source_sql, re.IGNORECASE)
    if minimum:
        return int(minimum.group(1)), 120
    maximum = re.search(r"\bage\s*<=\s*(\d+)", source_sql, re.IGNORECASE)
    if maximum:
        return 0, int(maximum.group(1))
    raise EHRProjectionRejected("source_age_range_missing")


def _ground_age(question: str, low: int, high: int) -> str:
    return (
        f"{question.rstrip(' ?.')} (age between {low} and {high})."
        if high < 120
        else f"{question.rstrip(' ?.')} (age at least {low})."
    )


def _render(
    *,
    kind: str,
    question: str,
    replacements: Mapping[str, Any],
    source_sql: str,
) -> tuple[str, str, str, str]:
    category = "filtering"
    difficulty = "medium"
    if kind == "patient_gender":
        (patient_id,) = _require(replacements, "patient_id")
        sql = (
            "SELECT DISTINCT p.gender FROM patients p "
            f"WHERE p.patient_id = '{_sql_string(patient_id)}'"
        )
    elif kind == "patient_age":
        (patient_id,) = _require(replacements, "patient_id")
        sql = (
            "SELECT p.age FROM patients p "
            f"WHERE p.patient_id = '{_sql_string(patient_id)}'"
        )
    elif kind.startswith("cost_"):
        key, column = {
            "cost_disease": ("diagnosis_name", "disease"),
            "cost_drug": ("drug_name", "drug_prescribed"),
            "cost_exam": (
                "lab_name" if "lab_name" in replacements else "procedure_name",
                "exam",
            ),
        }[kind]
        (value,) = _require(replacements, key)
        sql = (
            "SELECT DISTINCT v.cost FROM visits v "
            f"WHERE v.{column} = '{_sql_string(value)}' ORDER BY v.cost"
        )
    elif kind in {"count_exam_cases", "count_drug_cases"}:
        key, column = (
            ("drug_name", "drug_prescribed")
            if kind == "count_drug_cases"
            else (
                (
                    "lab_name"
                    if "lab_name" in replacements
                    else "procedure_name"
                ),
                "exam",
            )
        )
        (value,) = _require(replacements, key)
        sql = (
            "SELECT COUNT(*) FROM visits v "
            f"WHERE v.{column} = '{_sql_string(value)}'"
        )
    elif kind.startswith("count_") and kind.endswith("_patients"):
        key, column = {
            "count_disease_patients": ("diagnosis_name", "disease"),
            "count_drug_patients": ("drug_name", "drug_prescribed"),
            "count_exam_patients": (
                "lab_name" if "lab_name" in replacements else "procedure_name",
                "exam",
            ),
        }[kind]
        (value,) = _require(replacements, key)
        sql = (
            "SELECT COUNT(DISTINCT v.patient_id) FROM visits v "
            f"WHERE v.{column} = '{_sql_string(value)}'"
        )
        difficulty = "hard"
    elif kind == "list_disease_patients":
        (disease,) = _require(replacements, "diagnosis_name")
        sql = (
            "SELECT DISTINCT v.patient_id FROM visits v "
            f"WHERE v.disease = '{_sql_string(disease)}' "
            "ORDER BY v.patient_id"
        )
        difficulty = "hard"
    elif kind in {"has_disease", "has_drug", "has_exam"}:
        patient_id = _require(replacements, "patient_id")[0]
        key, column = {
            "has_disease": ("diagnosis_name", "disease"),
            "has_drug": ("drug_name", "drug_prescribed"),
            "has_exam": (
                "lab_name" if "lab_name" in replacements else "procedure_name",
                "exam",
            ),
        }[kind]
        value = _require(replacements, key)[0]
        sql = (
            "SELECT COUNT(*) > 0 FROM visits v "
            f"WHERE v.patient_id = '{_sql_string(patient_id)}' "
            f"AND v.{column} = '{_sql_string(value)}'"
        )
        difficulty = "hard"
    elif kind in {"has_any_disease", "has_any_drug", "has_any_exam"}:
        patient_id = _require(replacements, "patient_id")[0]
        column = {
            "has_any_disease": "disease",
            "has_any_drug": "drug_prescribed",
            "has_any_exam": "exam",
        }[kind]
        sql = (
            "SELECT COUNT(*) > 0 FROM visits v "
            f"WHERE v.patient_id = '{_sql_string(patient_id)}' "
            f"AND length(v.{column}) > 0"
        )
    elif kind == "total_cost_patient":
        (patient_id,) = _require(replacements, "patient_id")
        sql = (
            "SELECT SUM(v.cost) FROM visits v "
            f"WHERE v.patient_id = '{_sql_string(patient_id)}'"
        )
        category = "aggregation_grouping"
        difficulty = "hard"
    elif kind.startswith("aggregate_cost_"):
        aggregate = _source_aggregate(source_sql)
        key, column = {
            "aggregate_cost_disease": ("diagnosis_name", "disease"),
            "aggregate_cost_drug": ("drug_name", "drug_prescribed"),
            "aggregate_cost_exam": (
                "lab_name" if "lab_name" in replacements else "procedure_name",
                "exam",
            ),
        }[kind]
        value = _require(replacements, key)[0]
        sql = (
            f"SELECT {aggregate}(v.cost) FROM visits v "
            f"WHERE v.{column} = '{_sql_string(value)}'"
        )
        category = "aggregation_grouping"
        difficulty = "hard"
    elif kind in {
        "count_patient_drugs",
        "count_patient_drug",
        "count_patient_exam",
        "count_patient_visits",
    }:
        patient_id = _require(replacements, "patient_id")[0]
        predicate = ""
        expression = "COUNT(*)"
        if kind == "count_patient_drugs":
            expression = "COUNT(DISTINCT v.drug_prescribed)"
            predicate = " AND length(v.drug_prescribed) > 0"
        elif kind == "count_patient_drug":
            drug = _require(replacements, "drug_name")[0]
            predicate = (
                f" AND v.drug_prescribed = '{_sql_string(drug)}'"
            )
        elif kind == "count_patient_exam":
            key = "lab_name" if "lab_name" in replacements else "procedure_name"
            exam = _require(replacements, key)[0]
            predicate = f" AND v.exam = '{_sql_string(exam)}'"
        sql = (
            f"SELECT {expression} FROM visits v "
            f"WHERE v.patient_id = '{_sql_string(patient_id)}'{predicate}"
        )
    elif kind == "has_patient_visit":
        (patient_id,) = _require(replacements, "patient_id")
        sql = (
            "SELECT COUNT(*) > 0 FROM visits v "
            f"WHERE v.patient_id = '{_sql_string(patient_id)}'"
        )
    elif kind.startswith("top_"):
        limit = _source_limit(source_sql)
        question = _ground_limit(question, limit)
        column = (
            "drug_prescribed"
            if "drugs" in kind
            else ("disease" if "diseases" in kind else "exam")
        )
        if kind.endswith("_by_age"):
            low, high = _source_age_range(source_sql)
            question = _ground_age(question, low, high)
            age_predicate = (
                f"p.age BETWEEN {low} AND {high}"
                if high < 120
                else f"p.age >= {low}"
            )
            sql = (
                f"SELECT v.{column} FROM patients p "
                "JOIN visits v ON p.patient_id = v.patient_id "
                f"WHERE {age_predicate} "
                f"AND NULLIF(TRIM(v.{column}), '') IS NOT NULL "
                f"GROUP BY v.{column} ORDER BY COUNT(*) DESC, v.{column} "
                f"LIMIT {limit}"
            )
            difficulty = "extra"
        else:
            sql = (
                f"SELECT v.{column} FROM visits v "
                f"WHERE NULLIF(TRIM(v.{column}), '') IS NOT NULL "
                f"GROUP BY v.{column} "
                f"ORDER BY COUNT(*) DESC, v.{column} LIMIT {limit}"
            )
            difficulty = "hard"
        category = "top_k"
    else:
        raise EHRProjectionRejected("unsupported_handler")
    return question, sql, category, difficulty


def _validate_rendered_question_intent(kind: str, question: str) -> None:
    if kind in {"cost_disease", "cost_drug", "cost_exam"} and not re.search(
        r"\b(?:costs?|prices?|pricing|charge|pay|bill|amount)\b|"
        r"\bhow much\b",
        question,
        re.IGNORECASE,
    ):
        raise EHRProjectionRejected("question_intent_not_grounded")


def project_ehrsql_row(
    row: Mapping[str, Any],
    domains: EHRTargetDomains,
    db_path: Path,
    *,
    source_dataset: str,
    source_split: str,
    required_role: str | None = None,
    max_attempts: int = 32,
) -> ProjectedItem:
    """Project one EHRSQL row, or reject it without guessing."""
    if source_dataset not in {"eicu", "mimic_iii"}:
        raise ValueError("source_dataset must be eicu or mimic_iii")
    if source_split not in {"train", "valid", "test"}:
        raise ValueError("source_split must be train, valid, or test")
    if required_role not in {None, "development", "holdout"}:
        raise ValueError("required_role must be development, holdout, or None")
    if row.get("is_impossible") is True:
        raise EHRProjectionRejected("upstream_unanswerable")
    tag = str(row.get("q_tag", "")).strip()
    kind = TAG_KIND.get(tag)
    if not kind:
        raise EHRProjectionRejected("unsupported_q_tag")
    source_sql = str(row.get("query", "")).strip()
    if not source_sql:
        raise EHRProjectionRejected("missing_source_sql")
    if TIME_SQL_PATTERN.search(source_sql):
        raise EHRProjectionRejected("temporal_semantics_not_supported")
    question = str(row.get("question", "")).strip()
    if not question:
        raise EHRProjectionRejected("missing_source_question")
    source_sha256 = _sha256_payload(row)
    last_reason = "no_nonempty_projection"
    for attempt in range(max_attempts):
        try:
            anchor = _anchor(domains, source_sha256, attempt)
            projected_question, replacements = _replace_typed_values(
                row,
                question,
                anchor,
                domains,
                source_sha256,
                attempt,
            )
            if TIME_QUESTION_PATTERN.search(projected_question):
                raise EHRProjectionRejected(
                    "temporal_question_not_supported"
                )
            projected_question, gold_sql, category, difficulty = _render(
                kind=kind,
                question=projected_question,
                replacements=replacements,
                source_sql=source_sql,
            )
            _validate_rendered_question_intent(kind, projected_question)
            skeleton = validate_gold_sql(gold_sql)
            role = skeleton_role_v2(skeleton)
            if required_role is not None and role != required_role:
                raise EHRProjectionRejected("skeleton_reserved_for_other_role")
            validate_question_sql_grounding(projected_question, gold_sql)
            _, result_rows = execute_complete_readonly(gold_sql, db_path)
            if not result_rows:
                raise EHRProjectionRejected("empty_result")
            identity = hashlib.sha256(
                (
                    f"{PROJECTION_SEED_V2}\0EHRSQL-v1.5.1\0{source_dataset}\0"
                    f"{source_split}\0{source_sha256}\0"
                    f"{normalise_question(projected_question)}\0{gold_sql}"
                ).encode("utf-8")
            ).hexdigest()
            return ProjectedItem(
                item={
                    "id": f"ehrsql-{source_dataset}-{source_split}-{identity[:18]}",
                    "question": projected_question,
                    "gold_sql": gold_sql,
                    "difficulty": difficulty,
                    "category": category,
                    "provenance": {
                        "benchmark": "EHRSQL-v1.5.1",
                        "source_dataset": source_dataset,
                        "source_split": source_split,
                        "source_row_sha256": source_sha256,
                        "source_id_sha256": hashlib.sha256(
                            str(row.get("id", "")).encode("utf-8")
                        ).hexdigest(),
                        "source_q_tag_sha256": hashlib.sha256(
                            tag.encode("utf-8")
                        ).hexdigest(),
                        "projection_seed": PROJECTION_SEED_V2,
                        "partition_version": "v2",
                        "literal_attempt": attempt,
                        "skeleton_sha256": hashlib.sha256(
                            skeleton.encode("utf-8")
                        ).hexdigest(),
                        "skeleton_role": role,
                    },
                },
                skeleton_role=role,
                skeleton_sha256=hashlib.sha256(
                    skeleton.encode("utf-8")
                ).hexdigest(),
            )
        except EHRProjectionRejected as exc:
            last_reason = exc.reason
            if exc.reason == "skeleton_reserved_for_other_role":
                raise
            if exc.reason not in {
                "empty_result",
                "empty_target_value",
                "source_value_not_found_in_question",
            }:
                raise
        except (RuntimeError, ValueError, sqlite3.Error) as exc:
            last_reason = f"validation_or_execution:{type(exc).__name__}"
    raise EHRProjectionRejected(last_reason)


def deduplicate_ehrsql_items(
    items: Iterable[ProjectedItem],
) -> list[ProjectedItem]:
    by_question: dict[str, ProjectedItem] = {}
    by_id: dict[str, ProjectedItem] = {}
    for projected in items:
        normalized = normalise_question(str(projected.item["question"]))
        if normalized in by_question:
            continue
        item_id = str(projected.item["id"])
        if item_id in by_id:
            raise ValueError(f"duplicate projected EHRSQL id: {item_id}")
        by_question[normalized] = projected
        by_id[item_id] = projected
    return sorted(
        by_id.values(),
        key=lambda item: (
            item.item["difficulty"],
            item.item["category"],
            item.item["id"],
        ),
    )
