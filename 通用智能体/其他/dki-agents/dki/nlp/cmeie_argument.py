"""Native CMeIE argument spans for the fallback NER + relation pipeline.

The fallback deliberately uses the eleven native CMeIE argument types rather
than projecting through the nine CMeEE types.  Role is not part of this NER
head: a mention may be a subject for one SPO and an object for another.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from dki.nlp.cmeie_spo import all_occurrences, char_span_to_tokens, same_sentence


ARGUMENT_MODEL_CONFIG = "argument_model.json"
ARGUMENT_HEAD_WEIGHTS = "globalpointer.pt"
ARGUMENT_BILSTM_WEIGHTS = "bilstm.pt"
ARGUMENT_SOFTLEXICON = "softlexicon_surfaces.json"
ARGUMENT_SOFTLEXICON_WEIGHTS = "softlexicon_proj.pt"
ARGUMENT_MODEL_SCHEMA_VERSION = 3
ARGUMENT_SOFTLEX_MODEL_SCHEMA_VERSION = 4
ARGUMENT_POINTER_VARIANT = "efficient"
ARGUMENT_INNER_DIM = 64


def argument_model_contract(
    *,
    labels: list[str],
    max_length: int,
    selected_threshold: float,
    pointer_variant: str = ARGUMENT_POINTER_VARIANT,
    use_bilstm: bool = False,
    softlexicon_sha256: str | None = None,
    softlexicon_min_chars: int | None = None,
) -> dict[str, Any]:
    """Build the portable contract required to reconstruct an argument head."""
    if (
        not labels
        or labels != sorted(set(labels))
        or any(not isinstance(label, str) or not label for label in labels)
    ):
        raise ValueError("argument labels must be sorted unique non-empty strings")
    if len(labels) != 11:
        raise ValueError("a CMeIE argument model must expose exactly 11 types")
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    if (
        isinstance(selected_threshold, bool)
        or not isinstance(selected_threshold, (int, float))
        or not math.isfinite(float(selected_threshold))
    ):
        raise ValueError("selected_threshold must be a finite number")
    pointer_variant = str(pointer_variant).casefold()
    if pointer_variant not in {"classic", "efficient"}:
        raise ValueError("pointer_variant must be classic or efficient")
    if not isinstance(use_bilstm, bool):
        raise ValueError("use_bilstm must be boolean")
    base = {
        "schema_version": ARGUMENT_MODEL_SCHEMA_VERSION,
        "kind": "cmeie_native_argument_globalpointer",
        "pointer_variant": pointer_variant,
        "inner_dim": ARGUMENT_INNER_DIM,
        "labels": labels,
        "n_labels": len(labels),
        "max_length": max_length,
        "selected_threshold": float(selected_threshold),
        "head_weights": ARGUMENT_HEAD_WEIGHTS,
        "use_bilstm": use_bilstm,
        "bilstm_weights": ARGUMENT_BILSTM_WEIGHTS if use_bilstm else None,
    }
    if softlexicon_sha256 is None:
        if softlexicon_min_chars is not None:
            raise ValueError(
                "softlexicon_min_chars requires softlexicon_sha256"
            )
        return base
    if (
        not isinstance(softlexicon_sha256, str)
        or len(softlexicon_sha256) != 64
        or any(character not in "0123456789abcdef" for character in softlexicon_sha256)
    ):
        raise ValueError("softlexicon_sha256 must be a lowercase SHA-256")
    if (
        isinstance(softlexicon_min_chars, bool)
        or not isinstance(softlexicon_min_chars, int)
        or softlexicon_min_chars < 1
    ):
        raise ValueError("softlexicon_min_chars must be a positive integer")
    return {
        **base,
        "schema_version": ARGUMENT_SOFTLEX_MODEL_SCHEMA_VERSION,
        "softlexicon": True,
        "softlexicon_mode": "add",
        "softlexicon_dim": len(labels) * 4,
        "softlexicon_min_chars": softlexicon_min_chars,
        "softlexicon_sha256": softlexicon_sha256,
        "softlexicon_surfaces": ARGUMENT_SOFTLEXICON,
        "softlexicon_projection_weights": ARGUMENT_SOFTLEXICON_WEIGHTS,
    }


def validate_argument_model_contract(payload: Any) -> dict[str, Any]:
    """Fail closed on incomplete or ambiguous argument model metadata."""
    if not isinstance(payload, dict):
        raise ValueError("argument model contract must be a JSON object")
    legacy_required = {
        "schema_version",
        "kind",
        "pointer_variant",
        "inner_dim",
        "labels",
        "n_labels",
        "max_length",
        "selected_threshold",
        "head_weights",
    }
    schema_version = payload.get("schema_version")
    if schema_version in {1, 2}:
        if set(payload) != legacy_required:
            raise ValueError("argument model contract fields differ from schema")
        threshold = payload.get("selected_threshold")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or (schema_version == 1 and float(threshold) < 0.0)
        ):
            raise ValueError(
                "schema-v1 selected_threshold must be non-negative and finite"
            )
        labels = payload.get("labels")
        if (
            not isinstance(labels, list)
            or labels != sorted(set(labels))
            or len(labels) != 11
            or any(not isinstance(label, str) or not label for label in labels)
        ):
            raise ValueError("argument labels must be 11 sorted unique strings")
        max_length = payload.get("max_length")
        if (
            isinstance(max_length, bool)
            or not isinstance(max_length, int)
            or max_length < 1
        ):
            raise ValueError("max_length must be a positive integer")
        expected = {
            "schema_version": schema_version,
            "kind": "cmeie_native_argument_globalpointer",
            "pointer_variant": ARGUMENT_POINTER_VARIANT,
            "inner_dim": ARGUMENT_INNER_DIM,
            "labels": labels,
            "n_labels": len(labels),
            "max_length": max_length,
            "selected_threshold": float(threshold),
            "head_weights": ARGUMENT_HEAD_WEIGHTS,
        }
    elif schema_version in {
        ARGUMENT_MODEL_SCHEMA_VERSION,
        ARGUMENT_SOFTLEX_MODEL_SCHEMA_VERSION,
    }:
        required = legacy_required | {"use_bilstm", "bilstm_weights"}
        if schema_version == ARGUMENT_SOFTLEX_MODEL_SCHEMA_VERSION:
            required |= {
                "softlexicon",
                "softlexicon_mode",
                "softlexicon_dim",
                "softlexicon_min_chars",
                "softlexicon_sha256",
                "softlexicon_surfaces",
                "softlexicon_projection_weights",
            }
        if set(payload) != required:
            raise ValueError("argument model contract fields differ from schema")
        expected = argument_model_contract(
            labels=payload["labels"],
            max_length=payload["max_length"],
            selected_threshold=payload["selected_threshold"],
            pointer_variant=payload["pointer_variant"],
            use_bilstm=payload["use_bilstm"],
            softlexicon_sha256=(
                payload["softlexicon_sha256"]
                if schema_version == ARGUMENT_SOFTLEX_MODEL_SCHEMA_VERSION
                else None
            ),
            softlexicon_min_chars=(
                payload["softlexicon_min_chars"]
                if schema_version == ARGUMENT_SOFTLEX_MODEL_SCHEMA_VERSION
                else None
            ),
        )
    else:
        raise ValueError("unsupported argument model contract schema")
    if payload != expected:
        raise ValueError("argument model contract values differ from schema")
    return expected


def load_argument_model_contract(model_dir: Path) -> dict[str, Any]:
    """Load and validate the reconstruction contract without loading weights."""
    path = Path(model_dir) / ARGUMENT_MODEL_CONFIG
    if not path.is_file():
        raise FileNotFoundError(f"missing argument model contract: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid argument model contract: {path}") from error
    contract = validate_argument_model_contract(payload)
    weights = Path(model_dir) / contract["head_weights"]
    if not weights.is_file():
        raise FileNotFoundError(f"missing argument head weights: {weights}")
    bilstm_weights = contract.get("bilstm_weights")
    if bilstm_weights is not None:
        bilstm_path = Path(model_dir) / bilstm_weights
        if not bilstm_path.is_file():
            raise FileNotFoundError(
                f"missing argument BiLSTM weights: {bilstm_path}"
            )
    if contract.get("softlexicon"):
        surfaces = Path(model_dir) / contract["softlexicon_surfaces"]
        projection = Path(model_dir) / contract["softlexicon_projection_weights"]
        if not surfaces.is_file():
            raise FileNotFoundError(
                f"missing argument SoftLexicon surfaces: {surfaces}"
            )
        if not projection.is_file():
            raise FileNotFoundError(
                f"missing argument SoftLexicon projection weights: {projection}"
            )
        digest = hashlib.sha256(surfaces.read_bytes()).hexdigest()
        if digest != contract["softlexicon_sha256"]:
            raise ValueError("argument SoftLexicon SHA-256 mismatch")
    return contract


def labels_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    labels = set()
    for row in rows:
        for spo in row.get("spo_list", []):
            for role in ("subject", "object"):
                value = str(spo.get(role, ""))
                label = str(spo.get(f"{role}_type", ""))
                if value and label:
                    labels.add(label)
    return sorted(labels)


def _argument_occurrences(row: dict[str, Any]):
    """Yield every legal native-type surface occurrence from one SPO row."""
    text = str(row.get("text", ""))
    for spo in row.get("spo_list", []):
        subject, obj = str(spo.get("subject", "")), str(spo.get("object", ""))
        subject_type, object_type = str(spo.get("subject_type", "")), str(spo.get("object_type", ""))
        if not subject or not obj or not subject_type or not object_type:
            continue
        # Retain all same-sentence occurrences of each argument.  This mirrors
        # the direct-SPO label policy and avoids silently selecting a repeated
        # mention by position alone.
        subject_spans = all_occurrences(text, subject)
        object_spans = all_occurrences(text, obj)
        valid_subject = {
            span for span in subject_spans
            if any(same_sentence(text, span, other) for other in object_spans)
        }
        valid_object = {
            span for span in object_spans
            if any(same_sentence(text, other, span) for other in subject_spans)
        }
        for span in valid_subject:
            yield subject_type, span
        for span in valid_object:
            yield object_type, span


def build_labels(
    row: dict[str, Any],
    offsets: list[tuple[int, int]],
    type2id: dict[str, int],
) -> tuple[torch.Tensor, int]:
    """Build a dense GlobalPointer label tensor and count unalignable spans."""
    labels = torch.zeros((len(type2id), len(offsets), len(offsets)), dtype=torch.float32)
    dropped = 0
    for label, span in _argument_occurrences(row):
        token_span = char_span_to_tokens(offsets, span)
        if label not in type2id or token_span is None:
            dropped += 1
            continue
        start, end = token_span
        labels[type2id[label], start, end] = 1.0
    return labels, dropped


def gold_mentions(row: dict[str, Any]) -> set[tuple[str, int, int, str]]:
    text = str(row.get("text", ""))
    return {
        (label, start, end, text[start:end])
        for label, (start, end) in _argument_occurrences(row)
    }


def decode_mentions(
    text: str,
    offsets: list[tuple[int, int]],
    logits: torch.Tensor,
    labels: list[str],
    threshold: float,
) -> set[tuple[str, int, int, str]]:
    decoded = set()
    for label_id, token_start, token_end in (logits > threshold).nonzero(as_tuple=False).tolist():
        if token_start > token_end:
            continue
        left, right = offsets[token_start], offsets[token_end]
        if left[1] <= left[0] or right[1] <= right[0]:
            continue
        start, end = left[0], right[1]
        value = text[start:end]
        if value:
            decoded.add((labels[label_id], start, end, value))
    return decoded


def mention_metric(predicted: list[set], gold: list[set]) -> dict[str, float | int]:
    tp = sum(len(one & two) for one, two in zip(predicted, gold))
    fp = sum(len(one - two) for one, two in zip(predicted, gold))
    fn = sum(len(two - one) for one, two in zip(predicted, gold))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "tp": tp, "fp": fp, "fn": fn, "precision": precision,
        "recall": recall, "f1": 2 * precision * recall / max(1e-12, precision + recall),
    }


def mention_metric_by_label(
    predicted: list[set[tuple[str, int, int, str]]],
    gold: list[set[tuple[str, int, int, str]]],
    labels: list[str],
) -> dict[str, dict[str, float | int]]:
    """Return deterministic exact typed-span metrics for every native type."""
    if len(predicted) != len(gold):
        raise ValueError("predicted and gold row counts differ")
    result: dict[str, dict[str, float | int]] = {}
    for label in labels:
        predicted_rows = [
            {mention for mention in row if mention[0] == label}
            for row in predicted
        ]
        gold_rows = [
            {mention for mention in row if mention[0] == label}
            for row in gold
        ]
        result[label] = mention_metric(predicted_rows, gold_rows)
    return result
