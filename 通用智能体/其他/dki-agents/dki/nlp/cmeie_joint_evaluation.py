"""Shared, fail-closed CMeIE joint-evaluation protocol.

This module contains the security-sensitive pieces shared by the batched
outer evaluation, the outer seed gate, and the official development
confirmation.  In particular it:

* derives metrics exclusively from integer counts;
* evaluates all supplied seed runtimes while traversing one in-memory row
  sequence once;
* loads ``gplinker.pt`` through :mod:`dki.evidence_io` and
  ``torch.load(..., weights_only=True)``;
* re-verifies the complete checkpoint bundle and extraction contract after
  loading the tokenizer and encoder and before the first inference; and
* publishes and consumes canonical JSON receipts without replacement.

Dataset access markers remain owned by :mod:`dki.nlp.cmeie_run_protocol`.
Callers must publish the appropriate marker before opening a protected
dataset.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import secrets
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

from dki.evidence_io import EvidenceBinding, load_bound_json, read_bound_bytes
from dki.nlp.cmeie_artifacts import verify_checkpoint_bundle
from dki.nlp.cmeie_extraction_contract import (
    load_and_validate_extraction_contract,
)
from dki.nlp.cmeie_run_protocol import (
    OFFICIAL_DEV_DATASET_ROLE,
    OUTER_DATASET_ROLE,
    RunBinding,
    canonical_run_paths,
    dataset_key_sha256,
    load_run_binding,
)


ALL_SEEDS = (42, 43, 44)
OUTER_BATCH_SCHEMA_VERSION = 1
OUTER_BATCH_KIND = "cmeie_gplinker_outer_batch_evaluation"
OFFICIAL_CONFIRMATION_SCHEMA_VERSION = 1
OFFICIAL_CONFIRMATION_KIND = (
    "cmeie_gplinker_official_dev_development_confirmation"
)
DATASET_MARKER_SCHEMA_VERSION = 1
DATASET_MARKER_KIND = "cmeie_dataset_access_started"
OFFICIAL_DEV_EXPECTED_ROWS = 3585

OUTER_MEDIAN_F1_FLOOR = Fraction(1, 2)
OUTER_WORST_F1_FLOOR = Fraction(12, 25)
OUTER_PRECISION_RECALL_FLOOR = Fraction(9, 20)
OFFICIAL_DEV_F1_FLOOR = Fraction(1, 2)

_SHA256_CHARS = frozenset("0123456789abcdef")
_COMMON_SELECTION_FIELDS = (
    "rid",
    "head",
    "tree",
    "run_root",
    "source_manifest_path",
    "source_manifest_sha256",
    "source_manifest_size",
    "source_root",
    "source_tree_sha256",
    "source_tree_manifest_sha256",
    "train_sha256",
    "source_train_sha256",
    "fit_shard_sha256",
    "calibration_shard_sha256",
    "split_manifest_sha256",
    "base_model",
    "base_model_tree_sha256",
    "tokenizer_vocabulary_sha256",
    "label_table_sha256",
    "schema_constraints_sha256",
    "max_length",
    "parameter_count",
    "training_script_sha256",
    "evaluator_sha256",
    "model_module_sha256",
    "alignment_module_sha256",
    "globalpointer_module_sha256",
    "artifact_integrity_module_sha256",
    "extraction_contract_module_sha256",
    "selector_sha256",
    "selection_protocol_sha256",
    "run_protocol_sha256",
)


class JointEvaluationError(ValueError):
    """Evidence cannot authorize the requested evaluation transition."""


class Predictor(Protocol):
    """One frozen seed runtime used by :func:`evaluate_rows_once`."""

    seed: int

    def predict(self, row: Mapping[str, Any]) -> set[tuple[str, str, str]]:
        """Return exact subject/predicate/object triples for ``row``."""


@dataclass(frozen=True, slots=True)
class LoadedPredictor:
    """A verified model runtime plus the evidence needed by a receipt."""

    seed: int
    predict_fn: Callable[
        [Mapping[str, Any]], set[tuple[str, str, str]]
    ]
    bundle_binding: dict[str, Any]
    contract_binding: dict[str, Any]
    checkpoint_binding: EvidenceBinding

    def predict(
        self,
        row: Mapping[str, Any],
    ) -> set[tuple[str, str, str]]:
        return self.predict_fn(row)


def canonical_json_bytes(value: object) -> bytes:
    """Return the sole JSON byte representation accepted by this protocol."""

    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise JointEvaluationError(
            "evaluation evidence is not finite canonical JSON"
        ) from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise JointEvaluationError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def require_integer(
    value: object,
    *,
    label: str,
    minimum: int = 0,
) -> int:
    if type(value) is not int or value < minimum:
        raise JointEvaluationError(
            f"{label} must be a non-boolean integer >= {minimum}"
        )
    return value


def ratio_payload(value: Fraction) -> dict[str, int]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
    }


def parse_ratio(value: object, *, label: str) -> Fraction:
    if type(value) is not dict or set(value) != {
        "numerator",
        "denominator",
    }:
        raise JointEvaluationError(
            f"{label} must contain exactly numerator and denominator"
        )
    numerator = require_integer(
        value["numerator"],
        label=f"{label}.numerator",
    )
    denominator = require_integer(
        value["denominator"],
        label=f"{label}.denominator",
        minimum=1,
    )
    return Fraction(numerator, denominator)


def metric_from_counts(
    tp: int,
    fp: int,
    fn: int,
) -> dict[str, Any]:
    """Build one metric using exact ratios derived only from counts."""

    tp = require_integer(tp, label="tp")
    fp = require_integer(fp, label="fp")
    fn = require_integer(fn, label="fn")
    precision = Fraction(tp, tp + fp) if tp + fp else Fraction(0)
    recall = Fraction(tp, tp + fn) if tp + fn else Fraction(0)
    f1 = (
        Fraction(2 * tp, 2 * tp + fp + fn)
        if 2 * tp + fp + fn
        else Fraction(0)
    )
    return {
        "counts": {"tp": tp, "fp": fp, "fn": fn},
        "precision": ratio_payload(precision),
        "recall": ratio_payload(recall),
        "f1": ratio_payload(f1),
    }


def validate_metric(
    value: object,
    *,
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "counts",
        "precision",
        "recall",
        "f1",
    }:
        raise JointEvaluationError(
            f"{label} must contain exactly counts, precision, recall, f1"
        )
    counts = value["counts"]
    if type(counts) is not dict or set(counts) != {"tp", "fp", "fn"}:
        raise JointEvaluationError(
            f"{label}.counts must contain exactly tp, fp, fn"
        )
    expected = metric_from_counts(
        require_integer(counts["tp"], label=f"{label}.counts.tp"),
        require_integer(counts["fp"], label=f"{label}.counts.fp"),
        require_integer(counts["fn"], label=f"{label}.counts.fn"),
    )
    for field in ("precision", "recall", "f1"):
        if parse_ratio(value[field], label=f"{label}.{field}") != (
            parse_ratio(expected[field], label=f"{label}.{field}.expected")
        ):
            raise JointEvaluationError(
                f"{label}.{field} disagrees with integer counts"
            )
    return expected


def metric_ratio(metric: Mapping[str, Any], field: str) -> Fraction:
    return parse_ratio(metric.get(field), label=f"metric.{field}")


def evaluate_rows_once(
    rows: Sequence[Mapping[str, Any]],
    predictors: Sequence[Predictor],
    *,
    gold_fn: Callable[
        [Mapping[str, Any]], set[tuple[str, str, str]]
    ],
) -> dict[int, dict[str, Any]]:
    """Evaluate all seeds in one traversal of a preloaded row sequence."""

    if type(rows) not in {list, tuple}:
        raise TypeError("rows must be a concrete list or tuple")
    by_seed: dict[int, Predictor] = {}
    for predictor in predictors:
        seed = getattr(predictor, "seed", None)
        if type(seed) is not int or seed not in ALL_SEEDS:
            raise JointEvaluationError(
                "predictor seed must be 42, 43, or 44"
            )
        if seed in by_seed:
            raise JointEvaluationError(f"duplicate predictor seed {seed}")
        by_seed[seed] = predictor
    if tuple(sorted(by_seed)) != ALL_SEEDS:
        raise JointEvaluationError(
            "outer batch predictors must be exactly seeds 42, 43, and 44"
        )

    counts = {
        seed: {"tp": 0, "fp": 0, "fn": 0}
        for seed in ALL_SEEDS
    }
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise JointEvaluationError(
                f"evaluation row {index} must be an object"
            )
        gold = gold_fn(row)
        if type(gold) is not set:
            gold = set(gold)
        for seed in ALL_SEEDS:
            predicted = by_seed[seed].predict(row)
            if type(predicted) is not set:
                predicted = set(predicted)
            counts[seed]["tp"] += len(predicted & gold)
            counts[seed]["fp"] += len(predicted - gold)
            counts[seed]["fn"] += len(gold - predicted)
    return {
        seed: metric_from_counts(**counts[seed])
        for seed in ALL_SEEDS
    }


def evaluate_one_model(
    rows: Sequence[Mapping[str, Any]],
    predictor: Predictor,
    *,
    gold_fn: Callable[
        [Mapping[str, Any]], set[tuple[str, str, str]]
    ],
) -> dict[str, Any]:
    counts = {"tp": 0, "fp": 0, "fn": 0}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise JointEvaluationError(
                f"evaluation row {index} must be an object"
            )
        gold = set(gold_fn(row))
        predicted = set(predictor.predict(row))
        counts["tp"] += len(predicted & gold)
        counts["fp"] += len(predicted - gold)
        counts["fn"] += len(gold - predicted)
    return metric_from_counts(**counts)


def outer_metrics_pass(metrics: Mapping[int, Mapping[str, Any]]) -> bool:
    if set(metrics) != set(ALL_SEEDS):
        raise JointEvaluationError(
            "outer metrics must contain exactly seeds 42, 43, and 44"
        )
    validated = {
        seed: validate_metric(metrics[seed], label=f"outer metric seed {seed}")
        for seed in ALL_SEEDS
    }
    f1_values = sorted(
        metric_ratio(validated[seed], "f1")
        for seed in ALL_SEEDS
    )
    return (
        f1_values[1] >= OUTER_MEDIAN_F1_FLOOR
        and f1_values[0] >= OUTER_WORST_F1_FLOOR
        and all(
            metric_ratio(validated[seed], field)
            >= OUTER_PRECISION_RECALL_FLOOR
            for seed in ALL_SEEDS
            for field in ("precision", "recall")
        )
    )


def official_metric_pass(metric: Mapping[str, Any]) -> bool:
    validated = validate_metric(metric, label="official dev metric")
    return metric_ratio(validated, "f1") >= OFFICIAL_DEV_F1_FLOOR


def validate_selection_set(
    selections: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, Mapping[str, Any]], dict[str, Any]]:
    """Require exact seeds and exact common bindings across selections."""

    by_seed: dict[int, Mapping[str, Any]] = {}
    for index, selection in enumerate(selections):
        if not isinstance(selection, Mapping):
            raise JointEvaluationError(
                f"selection[{index}] must be an object"
            )
        if selection.get("schema_version") != 3:
            raise JointEvaluationError(
                f"selection[{index}] must use schema v3"
            )
        if (
            selection.get("kind")
            != "cmeie_gplinker_checkpoint_selection"
            or selection.get("dataset_role") != "calibration_selection"
            or selection.get("selection_used") is not True
            or selection.get("outer_accessed") is not False
            or selection.get("official_dev_accessed") is not False
        ):
            raise JointEvaluationError(
                f"selection[{index}] is not a frozen calibration selection"
            )
        seed = selection.get("seed")
        if type(seed) is not int or seed not in ALL_SEEDS or seed in by_seed:
            raise JointEvaluationError(
                f"selection[{index}] has an invalid or duplicate seed"
            )
        thresholds = selection.get("thresholds")
        if (
            type(thresholds) is not list
            or len(thresholds) != 3
            or any(value not in {-0.5, 0.0, 0.5} for value in thresholds)
            or selection.get("schema_constraint") not in {"off", "on"}
            or selection.get("grid_size") != 54
        ):
            raise JointEvaluationError(
                f"selection[{index}] has an invalid declared-grid choice"
            )
        by_seed[seed] = selection
    if tuple(sorted(by_seed)) != ALL_SEEDS:
        raise JointEvaluationError(
            "selection receipts must be exactly seeds 42, 43, and 44"
        )

    first = by_seed[ALL_SEEDS[0]]
    common: dict[str, Any] = {}
    for field in _COMMON_SELECTION_FIELDS:
        if field not in first:
            raise JointEvaluationError(
                f"selection seed 42 lacks common binding {field}"
            )
        expected = first[field]
        for seed in ALL_SEEDS[1:]:
            if by_seed[seed].get(field) != expected:
                raise JointEvaluationError(
                    f"selection seed {seed} differs on common binding {field}"
                )
        common[field] = expected
    return by_seed, common


def selection_evidence(
    selections: Mapping[int, Mapping[str, Any]],
    paths: Mapping[int, Path],
    bindings: Mapping[int, EvidenceBinding],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for seed in ALL_SEEDS:
        selection = selections[seed]
        binding = bindings[seed]
        result.append(
            {
                "seed": seed,
                "path": str(paths[seed]),
                "sha256": require_sha256(
                    binding.get("sha256"),
                    label=f"selection seed {seed} SHA-256",
                ),
                "size": require_integer(
                    binding.get("size"),
                    label=f"selection seed {seed} size",
                ),
                "training_receipt_sha256": require_sha256(
                    selection.get("training_receipt_sha256"),
                    label=f"selection seed {seed} training receipt SHA-256",
                ),
                "checkpoint_sha256": require_sha256(
                    selection.get("checkpoint_sha256"),
                    label=f"selection seed {seed} checkpoint SHA-256",
                ),
                "checkpoint_bundle_manifest_sha256": require_sha256(
                    selection.get("checkpoint_bundle_manifest_sha256"),
                    label=(
                        f"selection seed {seed} bundle manifest SHA-256"
                    ),
                ),
                "checkpoint_bundle_tree_sha256": require_sha256(
                    selection.get("checkpoint_bundle_tree_sha256"),
                    label=f"selection seed {seed} bundle tree SHA-256",
                ),
                "extraction_contract_sha256": require_sha256(
                    selection.get("extraction_contract_sha256"),
                    label=f"selection seed {seed} contract SHA-256",
                ),
                "semantic_contract_sha256": require_sha256(
                    selection.get("semantic_contract_sha256"),
                    label=f"selection seed {seed} semantic SHA-256",
                ),
            }
        )
    return result


def _verify_live_run(run_binding: RunBinding) -> RunBinding:
    if type(run_binding) is not RunBinding:
        raise TypeError("run_binding must be a RunBinding")
    live = load_run_binding(run_binding.run_root)
    if live != run_binding:
        raise JointEvaluationError(
            "run binding changed during joint evaluation"
        )
    return live


def load_and_validate_started_marker(
    run_binding: RunBinding,
    *,
    role: str,
    dataset_sha256: str,
    split_sha256: str,
) -> tuple[dict[str, Any], EvidenceBinding]:
    """Descriptor-bind and independently validate one canonical marker."""

    live = _verify_live_run(run_binding)
    path = canonical_run_paths(live).dataset_started(role)
    payload, binding = load_bound_json(
        path,
        f"CMeIE {role} access marker",
        reject_multiple_links=True,
        require_object=True,
    )
    if (
        payload.get("schema_version") != DATASET_MARKER_SCHEMA_VERSION
        or payload.get("kind") != DATASET_MARKER_KIND
        or payload.get("started") is not True
        or payload.get("completion_recorded") is not False
        or payload.get("rid") != live.rid
        or payload.get("head") != live.git_head
        or payload.get("tree") != live.included_tree_sha256
        or payload.get("run_root") != str(live.run_root)
        or payload.get("source_manifest_path")
        != str(live.source_manifest)
        or payload.get("source_manifest_sha256")
        != live.source_manifest_sha256
        or payload.get("dataset_role") != role
        or payload.get("dataset_sha256") != dataset_sha256
        or payload.get("split_sha256") != split_sha256
        or payload.get("dataset_key_sha256")
        != dataset_key_sha256(role, dataset_sha256, split_sha256)
    ):
        raise JointEvaluationError(
            f"CMeIE {role} access marker binding is invalid"
        )
    evidence = payload.get("input_model_evidence")
    if type(evidence) is not dict:
        raise JointEvaluationError(
            f"CMeIE {role} access marker lacks model evidence"
        )
    if payload.get("input_model_evidence_sha256") != sha256_bytes(
        canonical_json_bytes(evidence)
    ):
        raise JointEvaluationError(
            f"CMeIE {role} access marker model evidence hash is invalid"
        )
    return payload, binding


def publish_json_once(
    path: Path,
    payload: Mapping[str, Any],
    *,
    label: str,
) -> EvidenceBinding:
    """Publish canonical JSON by a same-directory hard link, never replace."""

    target = Path(os.path.abspath(os.fspath(path)))
    if os.path.lexists(target):
        raise FileExistsError(f"{label} already exists: {target}")
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise JointEvaluationError(
            f"{label} parent must be an existing real directory"
        )
    encoded = canonical_json_bytes(dict(payload))
    stage = target.parent / (
        f".{target.name}.stage-{os.getpid()}-{secrets.token_hex(12)}"
    )
    descriptor = os.open(
        stage,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError(f"cannot write {label} staging file")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        try:
            os.link(stage, target, follow_symlinks=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"{label} already exists: {target}"
            ) from exc
        try:
            directory_descriptor = os.open(
                target.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
        except OSError:
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            except OSError:
                pass
            finally:
                os.close(directory_descriptor)
    finally:
        try:
            os.unlink(stage)
        except FileNotFoundError:
            pass
    actual, binding = read_bound_bytes(
        target,
        label,
        reject_multiple_links=True,
    )
    if actual != encoded:
        raise RuntimeError(f"published {label} bytes changed")
    return binding


def load_canonical_completion(
    run_binding: RunBinding,
    *,
    role: str,
    expected_kind: str,
    expected_schema_version: int,
) -> tuple[dict[str, Any], EvidenceBinding]:
    live = _verify_live_run(run_binding)
    path = canonical_run_paths(live).dataset_completion(role)
    payload, binding = load_bound_json(
        path,
        f"CMeIE {role} completion",
        reject_multiple_links=True,
        require_object=True,
    )
    canonical = canonical_json_bytes(payload)
    if (
        payload.get("schema_version") != expected_schema_version
        or payload.get("kind") != expected_kind
        or payload.get("dataset_role") != role
        or payload.get("rid") != live.rid
        or payload.get("head") != live.git_head
        or payload.get("tree") != live.included_tree_sha256
        or payload.get("run_root") != str(live.run_root)
        or binding.get("size") != len(canonical)
        or binding.get("sha256") != sha256_bytes(canonical)
    ):
        raise JointEvaluationError(
            f"CMeIE {role} completion binding is invalid"
        )
    return payload, binding


def _verify_bundle(selection: Mapping[str, Any]) -> dict[str, Any]:
    directory = Path(str(selection["checkpoint_directory"]))
    return verify_checkpoint_bundle(
        directory,
        expected_manifest_sha256=require_sha256(
            selection.get("checkpoint_bundle_manifest_sha256"),
            label="selection checkpoint bundle manifest SHA-256",
        ),
        expected_tree_sha256=require_sha256(
            selection.get("checkpoint_bundle_tree_sha256"),
            label="selection checkpoint bundle tree SHA-256",
        ),
        expected_checkpoint_sha256=require_sha256(
            selection.get("checkpoint_sha256"),
            label="selection checkpoint SHA-256",
        ),
    )


def load_predictor(
    selection: Mapping[str, Any],
    *,
    device_name: str,
) -> LoadedPredictor:
    """Load one selected GPLinker model under its complete frozen contract."""

    seed = selection.get("seed")
    if type(seed) is not int or seed not in ALL_SEEDS:
        raise JointEvaluationError("selection seed is invalid")
    if device_name not in {"cpu", "npu"}:
        raise JointEvaluationError("device must be exactly cpu or npu")
    checkpoint_directory = Path(str(selection["checkpoint_directory"]))
    checkpoint_path = Path(str(selection["checkpoint"]))
    if checkpoint_path != checkpoint_directory / "gplinker.pt":
        raise JointEvaluationError(
            "selection checkpoint is not canonical gplinker.pt"
        )

    bundle_before = _verify_bundle(selection)
    checkpoint_raw, checkpoint_binding = read_bound_bytes(
        checkpoint_path,
        f"CMeIE seed {seed} gplinker checkpoint",
        reject_multiple_links=True,
    )
    if (
        checkpoint_binding.get("sha256")
        != selection.get("checkpoint_sha256")
        or checkpoint_binding.get("canonical_path") != str(checkpoint_path)
    ):
        raise JointEvaluationError(
            f"seed {seed} checkpoint descriptor binding differs from selection"
        )

    import torch
    from transformers import AutoModel, AutoTokenizer

    from dki.nlp.cmeie_gplinker import CMeIEGPLinker
    from dki.nlp.cmeie_spo import decode_spo

    saved = torch.load(
        io.BytesIO(checkpoint_raw),
        map_location="cpu",
        weights_only=True,
    )
    if type(saved) is not dict:
        raise JointEvaluationError(
            f"seed {seed} checkpoint root must be a mapping"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_directory,
        use_fast=True,
    )
    vocabulary = tokenizer.get_vocab()
    contract_before = load_and_validate_extraction_contract(
        checkpoint_directory,
        tokenizer_vocabulary=vocabulary,
        expected_binding=selection,
    )
    encoder = AutoModel.from_pretrained(checkpoint_directory)
    if device_name == "npu" and not (
        hasattr(torch, "npu") and torch.npu.is_available()
    ):
        raise JointEvaluationError("NPU evaluation requested but unavailable")
    device = torch.device("npu:0" if device_name == "npu" else "cpu")
    entity_labels = saved.get("entity_labels")
    predicates = saved.get("predicates")
    model_state = saved.get("model")
    if (
        type(entity_labels) is not list
        or type(predicates) is not list
        or not entity_labels
        or not predicates
        or not isinstance(model_state, Mapping)
    ):
        raise JointEvaluationError(
            f"seed {seed} checkpoint lacks frozen model metadata"
        )
    model = CMeIEGPLinker(
        encoder,
        int(encoder.config.hidden_size),
        len(entity_labels),
        len(predicates),
        pointer_variant=saved.get("pointer_variant", "efficient"),
    ).to(device)
    model.load_state_dict(model_state)
    model.eval()

    # This is deliberately after both tokenizer and encoder loading and before
    # the closure can perform its first inference.
    bundle_after = _verify_bundle(selection)
    contract_after = load_and_validate_extraction_contract(
        checkpoint_directory,
        tokenizer_vocabulary=tokenizer.get_vocab(),
        expected_binding=selection,
    )
    if bundle_before != bundle_after or contract_before != contract_after:
        raise JointEvaluationError(
            f"seed {seed} model evidence changed while loading"
        )

    thresholds = tuple(selection["thresholds"])
    contract_payload, contract_payload_binding = load_bound_json(
        checkpoint_directory / "extraction_contract.json",
        f"CMeIE seed {seed} extraction contract payload",
        reject_multiple_links=True,
        require_object=True,
    )
    contract_final = load_and_validate_extraction_contract(
        checkpoint_directory,
        tokenizer_vocabulary=tokenizer.get_vocab(),
        expected_binding=selection,
    )
    if (
        contract_payload_binding.get("sha256")
        != selection.get("extraction_contract_sha256")
        or contract_final != contract_after
        or saved.get("entity_labels") != contract_payload.get("entity_labels")
        or saved.get("predicates") != contract_payload.get("predicates")
        or saved.get("pointer_variant", "efficient")
        != contract_payload.get("pointer_variant")
        or selection.get("max_length") != contract_payload.get("max_length")
    ):
        raise JointEvaluationError(
            f"seed {seed} checkpoint semantics differ from its contract"
        )
    schema_constraints = (
        {
            predicate: [tuple(pair) for pair in pairs]
            for predicate, pairs in contract_payload[
                "schema_constraints"
            ].items()
        }
        if selection.get("schema_constraint") == "on"
        else None
    )
    if selection.get("schema_constraint") == "on" and not isinstance(
        schema_constraints,
        Mapping,
    ):
        raise JointEvaluationError(
            f"seed {seed} schema-on selection lacks constraints"
        )

    def predict(row: Mapping[str, Any]) -> set[tuple[str, str, str]]:
        text = row.get("text")
        if type(text) is not str:
            raise JointEvaluationError("CMeIE row text must be a string")
        encoded = tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=require_integer(
                selection.get("max_length"),
                label="selection.max_length",
                minimum=1,
            ),
        )
        offsets = [
            tuple(item)
            for item in encoded.pop("offset_mapping")[0].tolist()
        ]
        inputs = {
            key: value.to(device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }
        with torch.no_grad():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "npu",
            ):
                output = model(**inputs)
        return decode_spo(
            text,
            offsets,
            output["entity"][0].float().cpu(),
            output["relation_head"][0].float().cpu(),
            output["relation_tail"][0].float().cpu(),
            entity_labels,
            predicates,
            thresholds,
            schema_constraints,
        )

    return LoadedPredictor(
        seed=seed,
        predict_fn=predict,
        bundle_binding=bundle_after,
        contract_binding=contract_after,
        checkpoint_binding=checkpoint_binding,
    )


def source_member_binding(
    run_binding: RunBinding,
    relative_path: str,
) -> dict[str, Any]:
    """Verify one evaluator dependency against the included-tree manifest."""

    live = _verify_live_run(run_binding)
    manifest, _ = load_bound_json(
        live.source_manifest,
        "CMeIE run source manifest",
        require_object=True,
    )
    entries = manifest.get("entries")
    if type(entries) is not list:
        raise JointEvaluationError("source manifest entries are invalid")
    matching = [
        item
        for item in entries
        if type(item) is dict and item.get("path") == relative_path
    ]
    if len(matching) != 1:
        raise JointEvaluationError(
            f"source manifest does not uniquely bind {relative_path}"
        )
    expected = matching[0]
    path = live.source_root.joinpath(*relative_path.split("/"))
    raw, binding = read_bound_bytes(
        path,
        f"CMeIE source member {relative_path}",
        reject_multiple_links=True,
    )
    if (
        expected.get("size") != len(raw)
        or expected.get("sha256") != sha256_bytes(raw)
        or binding.get("sha256") != expected.get("sha256")
    ):
        raise JointEvaluationError(
            f"source member differs from included-tree manifest: {relative_path}"
        )
    return {
        "relative_path": relative_path,
        "path": str(path),
        "size": len(raw),
        "sha256": binding["sha256"],
    }


def exact_outer_gate_observation(
    metrics: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    validated = {
        seed: validate_metric(metrics[seed], label=f"outer metric seed {seed}")
        for seed in ALL_SEEDS
    }
    f1 = sorted(
        metric_ratio(validated[seed], "f1")
        for seed in ALL_SEEDS
    )
    return {
        "median_f1": ratio_payload(f1[1]),
        "worst_f1": ratio_payload(f1[0]),
        "passed": outer_metrics_pass(validated),
    }


def ensure_finite_json(value: object, *, label: str) -> None:
    """Validate recursively without accepting NaN/Infinity or alien values."""

    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise JointEvaluationError(f"{label} contains a non-finite value")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            ensure_finite_json(item, label=f"{label}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str or not key or "\0" in key:
                raise JointEvaluationError(
                    f"{label} contains an invalid object key"
                )
            ensure_finite_json(item, label=f"{label}.{key}")
        return
    raise JointEvaluationError(
        f"{label} contains unsupported type {type(value).__name__}"
    )


__all__ = [
    "ALL_SEEDS",
    "JointEvaluationError",
    "LoadedPredictor",
    "OFFICIAL_CONFIRMATION_KIND",
    "OFFICIAL_CONFIRMATION_SCHEMA_VERSION",
    "OFFICIAL_DEV_EXPECTED_ROWS",
    "OFFICIAL_DEV_F1_FLOOR",
    "OUTER_BATCH_KIND",
    "OUTER_BATCH_SCHEMA_VERSION",
    "OUTER_MEDIAN_F1_FLOOR",
    "OUTER_PRECISION_RECALL_FLOOR",
    "OUTER_WORST_F1_FLOOR",
    "canonical_json_bytes",
    "ensure_finite_json",
    "evaluate_one_model",
    "evaluate_rows_once",
    "exact_outer_gate_observation",
    "load_and_validate_started_marker",
    "load_canonical_completion",
    "load_predictor",
    "metric_from_counts",
    "metric_ratio",
    "official_metric_pass",
    "outer_metrics_pass",
    "parse_ratio",
    "publish_json_once",
    "ratio_payload",
    "selection_evidence",
    "sha256_bytes",
    "source_member_binding",
    "validate_metric",
    "validate_selection_set",
]
