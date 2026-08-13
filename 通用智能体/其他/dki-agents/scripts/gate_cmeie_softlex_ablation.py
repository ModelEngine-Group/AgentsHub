"""Gate the single declared CMeIE fit-only SoftLexicon ablation.

The gate compares two seed-43, four-epoch, frozen-encoder calibration receipts
that differ only by the fit-only SoftLexicon feature branch.  It never reads
outer or official-dev data and publishes one create-once aggregate decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MIN_RELATIVE_F1_GAIN = 0.02
MAX_CANDIDATE_RATIO = 2.0
MAX_PRECISION_DROP = 0.03
MIN_ABSOLUTE_PRECISION = 0.70
MIN_ABSOLUTE_RECALL = 0.88
MIN_ABSOLUTE_F1 = 0.75


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_receipt(path: Path, role: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("kind") != "cmeie_argument_feasibility_training":
        raise ValueError(f"{role} is not an argument training receipt")
    if payload.get("dataset_role") != "fit_training_calibration_selection":
        raise ValueError(f"{role} used the wrong dataset role")
    if payload.get("outer_accessed") is not False:
        raise ValueError(f"{role} accessed outer data")
    if payload.get("official_dev_accessed") is not False:
        raise ValueError(f"{role} accessed official dev data")
    return payload


def _metric(receipt: dict[str, Any]) -> dict[str, float | int]:
    selected = receipt.get("selected_calibration")
    if not isinstance(selected, dict) or not isinstance(selected.get("aggregate"), dict):
        raise ValueError("receipt lacks selected calibration aggregate")
    aggregate = selected["aggregate"]
    result: dict[str, float | int] = {}
    for key in ("tp", "fp", "fn"):
        value = aggregate.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid aggregate count: {key}")
        result[key] = value
    for key in ("precision", "recall", "f1"):
        value = aggregate.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"invalid aggregate metric: {key}")
        result[key] = float(value)
    predicted_mentions = selected.get("predicted_mentions")
    if (
        isinstance(predicted_mentions, bool)
        or not isinstance(predicted_mentions, int)
        or predicted_mentions < 1
    ):
        raise ValueError("invalid predicted mention count")
    result["predicted_mentions"] = predicted_mentions
    return result


def ablation_decision(
    baseline_metric: dict[str, float | int],
    augmented_metric: dict[str, float | int],
) -> dict[str, Any]:
    baseline_count = int(baseline_metric["predicted_mentions"])
    augmented_count = int(augmented_metric["predicted_mentions"])
    relative_f1_gain = float(augmented_metric["f1"]) - float(baseline_metric["f1"])
    candidate_ratio = augmented_count / baseline_count
    precision_drop = float(baseline_metric["precision"]) - float(
        augmented_metric["precision"]
    )
    gates = {
        "relative_f1_gain_at_least_0_02": relative_f1_gain >= MIN_RELATIVE_F1_GAIN,
        "candidate_ratio_at_most_2": candidate_ratio <= MAX_CANDIDATE_RATIO,
        "precision_drop_at_most_0_03": precision_drop <= MAX_PRECISION_DROP,
        "absolute_precision_at_least_0_70": float(augmented_metric["precision"])
        >= MIN_ABSOLUTE_PRECISION,
        "absolute_recall_at_least_0_88": float(augmented_metric["recall"])
        >= MIN_ABSOLUTE_RECALL,
        "absolute_f1_at_least_0_75": float(augmented_metric["f1"])
        >= MIN_ABSOLUTE_F1,
    }
    return {
        "relative_f1_gain": relative_f1_gain,
        "candidate_ratio": candidate_ratio,
        "precision_drop": precision_drop,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _validate_pair(
    baseline: dict[str, Any],
    augmented: dict[str, Any],
) -> None:
    exact_fields = (
        "rid",
        "git_head",
        "source_tree_sha256",
        "source_manifest_sha256",
        "seed",
        "fit_validation",
        "calibration_validation",
        "base_model_tree",
        "labels",
        "n_labels",
        "training_strategy",
        "encoder_scope",
        "overfit_preflight",
    )
    for field in exact_fields:
        if baseline.get(field) != augmented.get(field):
            raise ValueError(f"ablation receipts differ in bound field: {field}")
    if baseline.get("seed") != 43:
        raise ValueError("the declared SoftLexicon ablation uses seed 43")
    if baseline.get("ablation_role") != "baseline":
        raise ValueError("baseline receipt has the wrong ablation role")
    if augmented.get("ablation_role") != "softlex":
        raise ValueError("augmented receipt has the wrong ablation role")
    expected = {
        "pointer_variant": "classic",
        "use_bilstm": True,
        "epochs": 4,
        "head_warmup_epochs": 4,
        "encoder_scope": "none",
        "training_strategy": "frozen_encoder_ablation",
        "precision": "bf16",
    }
    for role, receipt in (("baseline", baseline), ("augmented", augmented)):
        configuration = receipt.get("configuration")
        if not isinstance(configuration, dict):
            raise ValueError(f"{role} lacks configuration")
        for field, value in expected.items():
            if configuration.get(field) != value:
                raise ValueError(f"{role} violates declared configuration: {field}")
    baseline_softlex = baseline["configuration"].get("softlexicon")
    augmented_softlex = augmented["configuration"].get("softlexicon")
    if not isinstance(baseline_softlex, dict) or baseline_softlex.get("enabled") is not False:
        raise ValueError("baseline must disable SoftLexicon")
    if (
        not isinstance(augmented_softlex, dict)
        or augmented_softlex.get("enabled") is not True
        or augmented_softlex.get("mode") != "add"
        or augmented_softlex.get("min_chars") != 2
        or augmented_softlex.get("source_role") != "fit"
        or not augmented_softlex.get("payload_sha256")
    ):
        raise ValueError("augmented receipt lacks the declared fit-only SoftLexicon")
    ignored = {"softlexicon", "ablation_role"}
    baseline_config = {
        key: value for key, value in baseline["configuration"].items() if key not in ignored
    }
    augmented_config = {
        key: value for key, value in augmented["configuration"].items() if key not in ignored
    }
    if baseline_config != augmented_config:
        raise ValueError("ablation configurations differ beyond SoftLexicon")


def _publish_json_create_once(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            if hasattr(os, "fchmod"):
                os.fchmod(handle.fileno(), 0o444)
        os.link(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--augmented-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from dki.nlp.cmeie_run_protocol import load_run_binding

    run_binding = load_run_binding(args.run_root)
    output = args.output.resolve()
    run_root = args.run_root.resolve()
    if run_root not in output.parents:
        parser.error("output must be inside the bound run root")
    if os.path.lexists(output):
        parser.error("output is a create-once ablation decision")
    baseline = _load_receipt(args.baseline_receipt, "baseline")
    augmented = _load_receipt(args.augmented_receipt, "augmented")
    _validate_pair(baseline, augmented)
    if baseline["rid"] != run_binding.rid:
        raise ValueError("receipts are not bound to this run root")
    baseline_metric = _metric(baseline)
    augmented_metric = _metric(augmented)
    decision = ablation_decision(baseline_metric, augmented_metric)
    payload = {
        "schema_version": 1,
        "kind": "cmeie_fit_softlexicon_ablation_gate",
        "dataset_role": "calibration_ablation_selection",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "rid": run_binding.rid,
        "git_head": run_binding.git_head,
        "source_tree_sha256": run_binding.included_tree_sha256,
        "source_manifest_sha256": run_binding.source_manifest_sha256,
        "baseline_receipt": {
            "path": str(args.baseline_receipt.resolve()),
            "sha256": _sha256(args.baseline_receipt),
            "metric": baseline_metric,
        },
        "augmented_receipt": {
            "path": str(args.augmented_receipt.resolve()),
            "sha256": _sha256(args.augmented_receipt),
            "metric": augmented_metric,
            "softlexicon": augmented["configuration"]["softlexicon"],
        },
        "thresholds": {
            "min_relative_f1_gain": MIN_RELATIVE_F1_GAIN,
            "max_candidate_ratio": MAX_CANDIDATE_RATIO,
            "max_precision_drop": MAX_PRECISION_DROP,
            "min_absolute_precision": MIN_ABSOLUTE_PRECISION,
            "min_absolute_recall": MIN_ABSOLUTE_RECALL,
            "min_absolute_f1": MIN_ABSOLUTE_F1,
        },
        "decision": decision,
        "promotion": "resume_oof" if decision["passed"] else "rejected_ablation",
        "outer_accessed": False,
        "official_dev_accessed": False,
        "gate_script_sha256": _sha256(Path(__file__)),
    }
    _publish_json_create_once(output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if decision["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
