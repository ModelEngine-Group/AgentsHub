"""Triple validation operators for task 2."""

from __future__ import annotations

from typing import Any

from src.operators.kg_ops.relation_extractor import RELATION_SCHEMA


def validate_triples(triples: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate triples against the task-2 schema and remove duplicates."""

    valid_triples: list[dict[str, Any]] = []
    invalid_triples: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for triple in triples:
        error = _validation_error(triple)
        dedupe_key = (
            str(triple.get("subject", "")),
            str(triple.get("predicate", "")),
            str(triple.get("object", "")),
        )
        if error:
            invalid = dict(triple)
            invalid["error"] = error
            invalid_triples.append(invalid)
            continue
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        valid_triples.append(triple)

    return {
        "status": "passed" if not invalid_triples else "warning",
        "valid_count": len(valid_triples),
        "invalid_count": len(invalid_triples),
        "duplicate_count": max(len(triples) - len(valid_triples) - len(invalid_triples), 0),
        "triples": valid_triples,
        "invalid_triples": invalid_triples,
    }


def _validation_error(triple: dict[str, Any]) -> str | None:
    for key in ("subject", "predicate", "object", "record_id"):
        if not triple.get(key):
            return f"missing {key}"

    predicate = triple["predicate"]
    expected = RELATION_SCHEMA.get(predicate)
    if not expected:
        return f"unsupported predicate: {predicate}"

    subject_type, object_type = expected
    if triple.get("subject_type") != subject_type:
        return f"subject_type must be {subject_type}"
    if triple.get("object_type") != object_type:
        return f"object_type must be {object_type}"

    confidence = triple.get("confidence", 0)
    if not isinstance(confidence, int | float) or confidence < 0 or confidence > 1:
        return "confidence must be between 0 and 1"

    return None
