"""Held-out extraction-quality evaluation for the task-2 KG operators.

Computes precision / recall / F1 of medical entity extraction against a
hand-annotated gold corpus that is independent from the bundled sample and the
fine-tuning data. The metric is set-based per entity type: for each record the
predicted entity set is compared with the gold set, and true/false positives and
false negatives are aggregated into micro (corpus-level) and per-type scores.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.operators.kg_ops.entity_extractor import (
    ENTITY_ALIASES,
    ENTITY_DICTIONARY,
    extract_medical_entities,
)
from src.operators.kg_ops.relation_extractor import (
    RELATION_SCHEMA,
    extract_relations,
    extract_relations_tensorized,
)

_ENTITY_TYPES = list(ENTITY_DICTIONARY.keys())
_PREDICATES = list(RELATION_SCHEMA.keys())


def entity_in_vocabulary(name: str, entity_type: str) -> bool:
    """Return whether a gold entity label is covered by the built-in dictionary."""

    terms = set(ENTITY_DICTIONARY.get(entity_type, []))
    aliases = ENTITY_ALIASES.get(entity_type, {})
    canonical = aliases.get(name, name)
    return canonical in terms or name in terms


def _split_bucket(entity_name: str, entity_type: str) -> str:
    return "in_vocabulary" if entity_in_vocabulary(entity_name, entity_type) else "out_of_vocabulary"


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _predict_entities(
    text: str,
    extractor: Callable[[str], dict[str, Any]],
) -> dict[str, set[str]]:
    result = extractor(text)
    records = result.get("records", [])
    predicted: dict[str, set[str]] = {etype: set() for etype in _ENTITY_TYPES}
    for record in records:
        for etype in _ENTITY_TYPES:
            predicted[etype].update(record.get("entities", {}).get(etype, []))
    return predicted


def evaluate_extraction_quality(
    gold_records: list[dict[str, Any]],
    extractor: Callable[[str], dict[str, Any]] = extract_medical_entities,
) -> dict[str, Any]:
    """Evaluate entity extraction against a gold-annotated corpus.

    ``gold_records`` is a list of ``{"record_id", "text", "entities": {type: [...]}}``
    records. Returns overall micro precision/recall/F1, per-type scores, and the
    list of false positives / false negatives per record for transparency.
    """

    per_type_counts: dict[str, dict[str, int]] = {
        etype: {"tp": 0, "fp": 0, "fn": 0} for etype in _ENTITY_TYPES
    }
    record_diagnostics: list[dict[str, Any]] = []

    for record in gold_records:
        text = record.get("text", "")
        gold = {
            etype: set(record.get("entities", {}).get(etype, []))
            for etype in _ENTITY_TYPES
        }
        predicted = _predict_entities(text, extractor)

        false_positives: dict[str, list[str]] = {}
        false_negatives: dict[str, list[str]] = {}
        for etype in _ENTITY_TYPES:
            tp = predicted[etype] & gold[etype]
            fp = predicted[etype] - gold[etype]
            fn = gold[etype] - predicted[etype]
            per_type_counts[etype]["tp"] += len(tp)
            per_type_counts[etype]["fp"] += len(fp)
            per_type_counts[etype]["fn"] += len(fn)
            if fp:
                false_positives[etype] = sorted(fp)
            if fn:
                false_negatives[etype] = sorted(fn)

        record_diagnostics.append({
            "record_id": record.get("record_id", ""),
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        })

    per_type = {etype: _prf(**counts) for etype, counts in per_type_counts.items()}
    total_tp = sum(counts["tp"] for counts in per_type_counts.values())
    total_fp = sum(counts["fp"] for counts in per_type_counts.values())
    total_fn = sum(counts["fn"] for counts in per_type_counts.values())

    return {
        "record_count": len(gold_records),
        "overall": _prf(total_tp, total_fp, total_fn),
        "per_type": per_type,
        "records": record_diagnostics,
    }


def evaluate_extraction_vocabulary_split(
    gold_records: list[dict[str, Any]],
    extractor: Callable[[str], dict[str, Any]] = extract_medical_entities,
) -> dict[str, Any]:
    """Evaluate extraction with separate in-vocabulary vs out-of-vocabulary metrics.

    Out-of-vocabulary gold entities are expected to be missed by the rule-based
    dictionary extractor; this split makes generalization limits explicit.
    """

    base = evaluate_extraction_quality(gold_records, extractor=extractor)
    split_counts: dict[str, dict[str, int]] = {
        "in_vocabulary": {"tp": 0, "fp": 0, "fn": 0},
        "out_of_vocabulary": {"tp": 0, "fp": 0, "fn": 0},
    }

    for record in gold_records:
        text = record.get("text", "")
        gold = {
            etype: set(record.get("entities", {}).get(etype, []))
            for etype in _ENTITY_TYPES
        }
        predicted = _predict_entities(text, extractor)

        for etype in _ENTITY_TYPES:
            for entity in gold[etype]:
                bucket = _split_bucket(entity, etype)
                if entity in predicted[etype]:
                    split_counts[bucket]["tp"] += 1
                else:
                    split_counts[bucket]["fn"] += 1
            for entity in predicted[etype] - gold[etype]:
                bucket = _split_bucket(entity, etype)
                split_counts[bucket]["fp"] += 1

    vocabulary_split = {
        bucket: _prf(**counts) for bucket, counts in split_counts.items()
    }
    oov = vocabulary_split["out_of_vocabulary"]
    base["vocabulary_split"] = vocabulary_split
    base["oov_entity_count"] = (
        oov["tp"] + oov["fn"]
    )
    base["interpretation"] = (
        "Hybrid dictionary + suffix-pattern extraction reports separate "
        "in-vocabulary and out-of-vocabulary recall."
    )
    return base


def _predict_relations(
    text: str,
    backend: str,
) -> set[tuple[str, str, str]]:
    """Extract relation triples for a single record's text via the chosen backend."""

    extraction = extract_medical_entities(text)
    records = extraction.get("records", [])
    if backend == "rule":
        triples = extract_relations(records)
    else:
        triples = extract_relations_tensorized(records, backend=backend)["triples"]
    return {(t["subject"], t["predicate"], t["object"]) for t in triples}


def evaluate_relation_quality(
    gold_records: list[dict[str, Any]],
    backend: str = "rule",
) -> dict[str, Any]:
    """Evaluate relation extraction against a gold-annotated relation corpus.

    ``gold_records`` is a list of ``{"record_id", "text", "relations": [
    {"subject", "predicate", "object"}, ...]}`` records. The predicted relation
    triples (from the selected ``backend``: ``rule`` / ``cpu`` / ``npu``) are
    compared set-wise per record. Returns overall micro precision/recall/F1,
    per-predicate scores, and per-record false positives / false negatives.
    """

    if backend not in {"rule", "cpu", "npu"}:
        raise ValueError(f"unsupported relation backend: {backend!r}")

    per_pred_counts: dict[str, dict[str, int]] = {
        pred: {"tp": 0, "fp": 0, "fn": 0} for pred in _PREDICATES
    }
    record_diagnostics: list[dict[str, Any]] = []

    for record in gold_records:
        text = record.get("text", "")
        gold = {
            (rel["subject"], rel["predicate"], rel["object"])
            for rel in record.get("relations", [])
        }
        predicted = _predict_relations(text, backend)

        tp = gold & predicted
        fp = predicted - gold
        fn = gold - predicted
        for triple in tp:
            per_pred_counts.setdefault(triple[1], {"tp": 0, "fp": 0, "fn": 0})["tp"] += 1
        for triple in fp:
            per_pred_counts.setdefault(triple[1], {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1
        for triple in fn:
            per_pred_counts.setdefault(triple[1], {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1

        record_diagnostics.append({
            "record_id": record.get("record_id", ""),
            "false_positives": sorted(f"{s} -{p}-> {o}" for (s, p, o) in fp),
            "false_negatives": sorted(f"{s} -{p}-> {o}" for (s, p, o) in fn),
        })

    per_predicate = {pred: _prf(**counts) for pred, counts in per_pred_counts.items()}
    total_tp = sum(counts["tp"] for counts in per_pred_counts.values())
    total_fp = sum(counts["fp"] for counts in per_pred_counts.values())
    total_fn = sum(counts["fn"] for counts in per_pred_counts.values())

    return {
        "record_count": len(gold_records),
        "backend": backend,
        "overall": _prf(total_tp, total_fp, total_fn),
        "per_predicate": per_predicate,
        "records": record_diagnostics,
    }
