"""Assertion-aware entity schema and evaluation."""

from __future__ import annotations

from typing import Mapping, Sequence

from dki.nlp.evaluation import Span, evaluate_spans, validate_span

POLARITY_CERTAINTY = {"present", "negated", "possible", "conditional", "unresolved"}
TEMPORALITY = {"current", "historical", "planned", "unresolved"}
EXPERIENCER = {"patient", "family", "other", "unresolved"}


class AssertionValidationError(ValueError):
    pass


def validate_assertion(entity: Mapping[str, object], text: str | None = None) -> dict:
    start, end = validate_span(entity.get("span"), text)
    assertion = entity.get("assertion")
    if not isinstance(assertion, Mapping):
        raise AssertionValidationError("assertion object is required")
    values = {
        "polarity_certainty": assertion.get("polarity_certainty"),
        "temporality": assertion.get("temporality"),
        "experiencer": assertion.get("experiencer"),
    }
    for name, allowed in (
        ("polarity_certainty", POLARITY_CERTAINTY),
        ("temporality", TEMPORALITY),
        ("experiencer", EXPERIENCER),
    ):
        if values[name] not in allowed:
            raise AssertionValidationError(f"invalid {name}: {values[name]!r}")
    return {"span": [start, end], "type": entity["type"], "assertion": values}


def evaluate_assertions(pred_by_doc, gold_by_doc) -> dict:
    """Report oracle-span assertion and end-to-end joint correctness separately."""
    oracle_correct = oracle_total = joint_correct = 0
    pred_spans, gold_spans = {}, {}
    for doc_id in sorted(set(pred_by_doc) | set(gold_by_doc)):
        predictions = list(pred_by_doc.get(doc_id, []))
        gold = list(gold_by_doc.get(doc_id, []))
        pred_spans[doc_id] = [(e["span"][0], e["span"][1], e["type"]) for e in predictions]
        gold_spans[doc_id] = [(e["span"][0], e["span"][1], e["type"]) for e in gold]
        pred_lookup = {(e["span"][0], e["span"][1], e["type"]): e for e in predictions}
        for entity in gold:
            key = (entity["span"][0], entity["span"][1], entity["type"])
            matched = pred_lookup.get(key)
            if matched is not None:
                oracle_total += 1
                if matched.get("assertion") == entity.get("assertion"):
                    oracle_correct += 1
                    joint_correct += 1
    extraction = evaluate_spans(pred_spans, gold_spans, match="exact")
    gold_total = sum(len(items) for items in gold_by_doc.values())
    return {
        "extraction_exact": extraction,
        "oracle_span_assertion_accuracy": round(oracle_correct / oracle_total, 4) if oracle_total else None,
        "oracle_span_support": oracle_total,
        "end_to_end_joint_accuracy": round(joint_correct / gold_total, 4) if gold_total else None,
        "gold_support": gold_total,
    }
