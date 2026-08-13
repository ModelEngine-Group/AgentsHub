"""Fail-closed, hash-bound public evidence for the browser demo.

The demo must never infer the newest result from directory mtimes or scrape a
number from human-readable Markdown.  Release evidence is authoritative only
when ``FINAL_STATUS.json`` binds it by repository-relative path and SHA-256.
Before a final status exists, operators may explicitly opt in to one public
aggregate JSON file per quality gate through the documented environment
variables below.

Canonical ``FINAL_STATUS.json`` evidence bindings::

    {
      "schema_version": 1,
      "status": "NOT_READY_FOR_SUBMISSION",
      "source": {"included_tree_sha256": "..."},
      "evidence": {
        "cmeie": {"path": "outputs/remote_runs/.../cmeie.json",
                  "sha256": "..."},
        "nl2sql": {"path": "outputs/remote_runs/.../nl2sql.json",
                   "sha256": "..."}
      }
    }

``cmeie_public_summary`` and ``nl2sql_public_attestation`` are accepted as
unambiguous aliases for the two evidence keys.  A configured but invalid final
status is a hard failure; it never falls back to pre-final evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
NPU_ACCEPTANCE = ROOT / "docs" / "remote_npu_acceptance_20260725.md"
NPU_ACCEPTANCE_SHA256 = (
    "f0e0a1feedada43fb24ab5fe4629c36e0e9f166753b6fc38910f1a39f7c5d015"
)

FINAL_STATUS_ENV = "DKI_FINAL_STATUS_PATH"
CMEIE_SUMMARY_ENV = "DKI_CMEIE_PUBLIC_SUMMARY"
NL2SQL_ATTESTATION_ENV = "DKI_NL2SQL_PUBLIC_ATTESTATION"
DEVICE_SNAPSHOT_ENV = "DKI_DEMO_DEVICE_SNAPSHOT"

_FINAL_STATUSES = {
    "DEMO_READY",
    "QUALITY_GATES_PASSED",
    "READY_FOR_SUBMISSION",
    "NOT_READY_FOR_SUBMISSION",
}
_QUALITY_PASSED_FINAL_STATUSES = {
    "QUALITY_GATES_PASSED",
    "READY_FOR_SUBMISSION",
}
_FORBIDDEN_AGGREGATE_KEYS = {
    "question",
    "questions",
    "gold_sql",
    "gold_spo",
    "predicted_sql",
    "prediction_sql",
    "predictions",
    "details",
    "detailed_results",
    "error_samples",
    "examples",
    "holdout_rows",
    "development_rows",
}
_HEX = frozenset("0123456789abcdef")
_MAX_PUBLIC_JSON_BYTES = 4 * 1024 * 1024
_MAX_DEVICE_SNAPSHOT_BYTES = 128 * 1024
_DEVICE_SNAPSHOT_TOP_LEVEL_KEYS = {
    "schema_version",
    "kind",
    "created_at",
    "requested_logical_device",
    "detected_available",
    "detected_device_name",
    "environment_label",
    "detection_sources",
    "streamlit_runtime",
    "commands",
    "privacy",
}
_DEVICE_SNAPSHOT_COMMAND_KEYS = {"npu_smi", "python_runtime"}
_DEVICE_SNAPSHOT_NPU_SMI_KEYS = {
    "binary_available",
    "returncode",
    "stdout_sha256",
    "summary_lines",
    "error_type",
}
_DEVICE_SNAPSHOT_RUNTIME_KEYS = {
    "python",
    "torch",
    "torch_npu",
    "npu_available",
    "device_name",
    "error_type",
}
_DEVICE_SNAPSHOT_STREAMLIT_KEYS = {
    "version",
    "module_path",
    "distribution_path",
    "record_sha256",
    "metadata_sha256",
}
_DEVICE_SNAPSHOT_PRIVACY_KEYS = {
    "contains_environment",
    "contains_secrets",
}
_DEVICE_SNAPSHOT_CREATED_AT = re.compile(
    r"\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z"
)
_DEVICE_SNAPSHOT_ERROR_TYPE = re.compile(r"\A[A-Za-z_][A-Za-z0-9_]{0,127}\Z")
_DEVICE_SNAPSHOT_910 = re.compile(
    r"(?:Ascend\s*910|Ascend910|\b910[A-Z0-9_]*\b)",
    re.IGNORECASE,
)
_SENSITIVE_SNAPSHOT_TEXT = re.compile(
    r"(?:"
    r"-----BEGIN [^-\r\n]*PRIVATE KEY-----|"
    r"(?:api[_-]?key|access[_-]?token|client[_-]?secret|password|passwd|"
    r"private[_-]?key|authorization|bearer|cookie)\s*[:=]|"
    r"(?:sk|pk)-[A-Za-z0-9_-]{16,}|"
    r"(?:^|[\s;])(?:HOME|PATH|PYTHONPATH|LD_LIBRARY_PATH|"
    r"ASCEND_[A-Z0-9_]+|DKI_[A-Z0-9_]+|CUDA_VISIBLE_DEVICES)\s*=|"
    r"\.env(?:\s|$)"
    r")",
    re.IGNORECASE,
)


class EvidenceError(ValueError):
    """A validation error whose code is safe to expose in the demo."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in _HEX for char in value)
    )


def _require_sha256(value: Any, code: str = "invalid_sha256") -> str:
    if not _is_sha256(value):
        raise EvidenceError(code)
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError("duplicate_json_key")
        result[key] = value
    return result


def _walk_public_payload(value: Any) -> None:
    """Reject row-level material and malformed hash fields recursively."""

    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = str(key).strip().lower()
            if normalized_key in _FORBIDDEN_AGGREGATE_KEYS:
                raise EvidenceError("non_aggregate_content")
            if normalized_key.endswith("sha256"):
                _require_sha256(child)
            _walk_public_payload(child)
    elif isinstance(value, list):
        for child in value:
            _walk_public_payload(child)


def _public_json_path(raw_path: str | Path) -> tuple[Path, str]:
    root = ROOT.resolve(strict=True)
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise EvidenceError("configured_file_missing") from None
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        raise EvidenceError("path_outside_repository") from None
    parts = tuple(part.casefold() for part in relative.parts)
    if len(parts) >= 2 and parts[:2] == ("outputs", "private"):
        raise EvidenceError("private_path_rejected")
    if not resolved.is_file():
        raise EvidenceError("configured_path_not_file")
    if resolved.suffix.casefold() != ".json":
        raise EvidenceError("public_evidence_must_be_json")
    return resolved, relative.as_posix()


def _load_public_json(raw_path: str | Path) -> tuple[dict[str, Any], Path, str, str]:
    path, relative = _public_json_path(raw_path)
    try:
        raw = path.read_bytes()
    except OSError:
        raise EvidenceError("configured_file_unreadable") from None
    if len(raw) > _MAX_PUBLIC_JSON_BYTES:
        raise EvidenceError("public_json_too_large")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except EvidenceError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise EvidenceError("invalid_public_json") from None
    if not isinstance(payload, dict):
        raise EvidenceError("public_json_not_object")
    _walk_public_payload(payload)
    # Parse and hash the exact same byte snapshot.  Reopening the file here
    # would create a TOCTOU gap between the validated payload and its binding.
    return payload, path, relative, hashlib.sha256(raw).hexdigest()


def _configured_final_status() -> tuple[str | Path, str] | None:
    configured = os.getenv(FINAL_STATUS_ENV)
    if configured is not None:
        if not configured.strip():
            raise EvidenceError("empty_final_status_path")
        return configured, "environment_final_status"
    default = ROOT / "FINAL_STATUS.json"
    if default.exists():
        return default, "repository_final_status"
    return None


def _final_binding(
    evidence: dict[str, Any], aliases: tuple[str, ...]
) -> dict[str, Any]:
    present = [name for name in aliases if name in evidence]
    if len(present) != 1:
        raise EvidenceError("ambiguous_or_missing_evidence_binding")
    binding = evidence[present[0]]
    if not isinstance(binding, dict):
        raise EvidenceError("invalid_evidence_binding")
    path = binding.get("path")
    if not isinstance(path, str) or not path.strip():
        raise EvidenceError("missing_evidence_path")
    _require_sha256(binding.get("sha256"), "missing_evidence_sha256")
    return binding


def _load_final_status() -> dict[str, Any] | None:
    configured = _configured_final_status()
    if configured is None:
        return None
    raw_path, authority = configured
    payload, path, relative, file_sha256 = _load_public_json(raw_path)
    if payload.get("schema_version") != 1:
        raise EvidenceError("unsupported_final_status_schema")
    status = payload.get("status")
    if status not in _FINAL_STATUSES:
        raise EvidenceError("invalid_final_status")
    source = payload.get("source")
    if not isinstance(source, dict):
        raise EvidenceError("missing_final_source_binding")
    source_tree_sha256 = _require_sha256(
        source.get("included_tree_sha256"),
        "missing_final_included_tree_sha256",
    )
    bindings = payload.get("evidence")
    if not isinstance(bindings, dict):
        raise EvidenceError("missing_final_evidence_bindings")
    # A final status represents both quality gates, including blocked evidence.
    _final_binding(bindings, ("cmeie", "cmeie_public_summary"))
    _final_binding(bindings, ("nl2sql", "nl2sql_public_attestation"))
    return {
        "payload": payload,
        "path": path,
        "relative_path": relative,
        "sha256": file_sha256,
        "status": status,
        "source_tree_sha256": source_tree_sha256,
        "authority": authority,
    }


def _number(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(code)
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise EvidenceError(code)
    return result


def _metric_triplet(value: Any, code: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise EvidenceError(code)
    precision = _number(value.get("precision"), code)
    recall = _number(value.get("recall"), code)
    f1 = _number(value.get("f1"), code)
    expected = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    if abs(f1 - expected) > 0.001:
        raise EvidenceError("inconsistent_f1")
    return {"precision": precision, "recall": recall, "f1": f1}


def _validate_cmeie(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != 1:
        raise EvidenceError("unsupported_cmeie_summary_schema")
    kind = payload.get("kind")
    if kind is not None and kind != "cmeie_public_summary":
        raise EvidenceError("invalid_cmeie_summary_kind")
    dataset_role = payload.get("dataset_role")
    status = payload.get("status")
    if not isinstance(dataset_role, str) or not dataset_role.strip():
        raise EvidenceError("missing_cmeie_dataset_role")
    if not isinstance(status, str) or not status.strip():
        raise EvidenceError("missing_cmeie_status")
    _require_sha256(payload.get("source_tree_sha256"), "missing_cmeie_source_hash")

    official_accessed = payload.get("official_dev_accessed", False)
    outer_accessed = payload.get("outer_accessed", False)
    if not isinstance(official_accessed, bool) or not isinstance(outer_accessed, bool):
        raise EvidenceError("invalid_cmeie_access_flags")

    metrics: dict[str, float] | None = None
    metric_scope: str | None = None
    if "official_dev" in payload:
        metrics = _metric_triplet(payload["official_dev"], "invalid_cmeie_metrics")
        metric_scope = "official_dev_development_confirmation"
    elif "metrics" in payload:
        metrics = _metric_triplet(payload["metrics"], "invalid_cmeie_metrics")
        metric_scope = dataset_role
    if official_accessed and metrics is None:
        raise EvidenceError("official_dev_metrics_missing")
    if (
        metrics is not None
        and "official_dev" in str(metric_scope)
        and not official_accessed
    ):
        raise EvidenceError("official_dev_access_flag_mismatch")

    calibration = payload.get("calibration_gate")
    calibration_best_f1: float | None = None
    if calibration is not None:
        if not isinstance(calibration, dict) or not isinstance(calibration.get("passed"), bool):
            raise EvidenceError("invalid_cmeie_calibration_gate")
        _number(calibration.get("threshold"), "invalid_cmeie_calibration_gate")
        seed_metrics: list[dict[str, float]] = []
        for key, value in calibration.items():
            if key.startswith("seed"):
                seed_metrics.append(_metric_triplet(value, "invalid_cmeie_seed_metrics"))
        if not seed_metrics:
            raise EvidenceError("missing_cmeie_seed_metrics")
        calibration_best_f1 = max(item["f1"] for item in seed_metrics)
    if metrics is None and calibration is None:
        raise EvidenceError("missing_cmeie_aggregate_metrics")

    explicit_quality_gate = payload.get(
        "quality_gate_passed", payload.get("gate_passed", False)
    )
    if not isinstance(explicit_quality_gate, bool):
        raise EvidenceError("invalid_cmeie_gate_status")
    status_claims_pass = status in {"PASSED", "QUALITY_GATES_PASSED"}
    quality_gate_passed = explicit_quality_gate or status_claims_pass
    if quality_gate_passed:
        if (
            not official_accessed
            or metrics is None
            or metric_scope != "official_dev_development_confirmation"
            or metrics["f1"] < 0.5
        ):
            raise EvidenceError("unsupported_cmeie_pass_claim")

    result: dict[str, Any] = {
        "status": status,
        "gate_status": "PASSED" if quality_gate_passed else "BLOCKED",
        "gate_passed": quality_gate_passed,
        "available": True,
        "dataset_role": dataset_role,
        "model_family": payload.get("model_family"),
        "outer_accessed": outer_accessed,
        "official_dev_accessed": official_accessed,
        "source_tree_sha256": payload["source_tree_sha256"],
        "f1": metrics["f1"] if metrics else None,
        "metrics": metrics,
        "metric_scope": metric_scope,
        "calibration_best_f1": calibration_best_f1,
    }
    if calibration is not None:
        result["calibration_gate"] = calibration
    return result


def _count(value: Any, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceError(code)
    return value


def _nl2sql_counts(value: Any, code: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise EvidenceError(code)
    n = _count(value.get("n"), code)
    correct = _count(value.get("n_correct"), code)
    answered = _count(value.get("n_answered"), code)
    if not (correct <= answered <= n):
        raise EvidenceError("inconsistent_nl2sql_counts")
    aliases = {
        "accuracy_num": correct,
        "accuracy_den": n,
        "coverage_num": answered,
        "coverage_den": n,
    }
    for key, expected in aliases.items():
        if key in value and _count(value[key], code) != expected:
            raise EvidenceError("inconsistent_nl2sql_count_alias")
    return {"n": n, "correct": correct, "answered": answered}


def _validate_nl2sql(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != 1:
        raise EvidenceError("unsupported_nl2sql_attestation_schema")
    if payload.get("kind") != "blind_evaluation_public_attestation":
        raise EvidenceError("invalid_nl2sql_attestation_kind")
    if payload.get("status") != "consumed":
        raise EvidenceError("nl2sql_holdout_not_consumed")
    for key in (
        "source_tree_sha256",
        "candidate_seal_sha256",
        "freeze_manifest_sha256",
        "holdout_sha256",
        "evaluator_sha256",
    ):
        _require_sha256(payload.get(key), f"missing_{key}")

    rules = payload.get("public_attestation_rules")
    expected_rules = {
        "strict_scoring": True,
        "no_question_or_gold_sql_leakage": True,
        "single_inference_pass": True,
        "selection_used": False,
    }
    if not isinstance(rules, dict) or any(
        rules.get(key) is not value for key, value in expected_rules.items()
    ):
        raise EvidenceError("invalid_nl2sql_public_rules")

    overall_payload = payload.get("overall")
    hard_payload = payload.get("hard")
    if (
        not isinstance(overall_payload, dict)
        or overall_payload.get("strict_result_contract") is not True
    ):
        raise EvidenceError("missing_strict_result_contract")
    if (
        not isinstance(hard_payload, dict)
        or hard_payload.get("derived_from_overall_single_inference") is not True
    ):
        raise EvidenceError("hard_result_not_single_pass_derived")
    overall = _nl2sql_counts(overall_payload, "invalid_nl2sql_overall")
    hard = _nl2sql_counts(hard_payload, "invalid_nl2sql_hard")
    protocol_passed = (
        overall["n"] >= 120
        and hard["n"] >= 40
        and 100 * overall["correct"] > 85 * overall["n"]
        and 100 * hard["correct"] >= 70 * hard["n"]
    )
    gate_passed = payload.get("gate_passed")
    if not isinstance(gate_passed, bool) or gate_passed is not protocol_passed:
        raise EvidenceError("inconsistent_nl2sql_gate")
    gate = payload.get("gate")
    if gate is not None:
        if not isinstance(gate, dict) or gate.get("passed") is not gate_passed:
            raise EvidenceError("inconsistent_nl2sql_gate_object")
    gate_reason = payload.get("gate_reason")
    if gate_passed and gate_reason not in {None, "passed"}:
        raise EvidenceError("inconsistent_nl2sql_gate_reason")
    if not gate_passed and (not isinstance(gate_reason, str) or not gate_reason):
        raise EvidenceError("missing_nl2sql_gate_reason")

    return {
        "status": "PASSED" if gate_passed else "BLOCKED",
        "gate_status": "PASSED" if gate_passed else "BLOCKED",
        "gate_passed": gate_passed,
        "gate_reason": gate_reason,
        "consumption_status": "consumed",
        "available": True,
        "dataset_role": "prospective_independent_holdout",
        "engine": payload.get("engine"),
        "n": overall["n"],
        "correct": overall["correct"],
        "accuracy": overall["correct"] / overall["n"] if overall["n"] else 0.0,
        "coverage": overall["answered"] / overall["n"] if overall["n"] else 0.0,
        "hard_n": hard["n"],
        "hard_correct": hard["correct"],
        "hard_accuracy": hard["correct"] / hard["n"] if hard["n"] else 0.0,
        "source_tree_sha256": payload["source_tree_sha256"],
        "candidate_seal_sha256": payload["candidate_seal_sha256"],
        "holdout_sha256": payload["holdout_sha256"],
    }


def _blocked(status: str, error_code: str, authority: str) -> dict[str, Any]:
    return {
        "status": status,
        "gate_status": "BLOCKED",
        "gate_passed": False,
        "available": False,
        "error_code": error_code,
        "authority": authority,
        "f1": None,
    }


def _component_evidence(
    *,
    aliases: tuple[str, ...],
    prefinal_env: str,
    validator: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    try:
        final = _load_final_status()
    except EvidenceError as exc:
        return _blocked("invalid", exc.code, "final_status")
    except Exception:
        return _blocked("invalid", "final_status_validation_failed", "final_status")

    if final is not None:
        try:
            bindings = final["payload"]["evidence"]
            binding = _final_binding(bindings, aliases)
            payload, path, relative, file_sha256 = _load_public_json(binding["path"])
            if file_sha256 != binding["sha256"]:
                raise EvidenceError("evidence_sha256_mismatch")
            if path == final["path"]:
                raise EvidenceError("self_referential_final_status")
            result = validator(payload)
            if result.get("source_tree_sha256") != final["source_tree_sha256"]:
                raise EvidenceError("component_source_tree_mismatch")
            if (
                final["status"] in _QUALITY_PASSED_FINAL_STATUSES
                and not result["gate_passed"]
            ):
                raise EvidenceError("final_status_quality_claim_mismatch")
            result.update(
                {
                    "authority": "final_status",
                    "path": relative,
                    "sha256": file_sha256,
                    "final_status": final["status"],
                    "final_status_path": final["relative_path"],
                    "final_status_sha256": final["sha256"],
                    "final_source_tree_sha256": final["source_tree_sha256"],
                }
            )
            return result
        except EvidenceError as exc:
            return _blocked("invalid", exc.code, "final_status")
        except Exception:
            return _blocked(
                "invalid", "final_evidence_validation_failed", "final_status"
            )

    configured = os.getenv(prefinal_env)
    if configured is None:
        return _blocked("unavailable", "authoritative_evidence_not_configured", "none")
    if not configured.strip():
        return _blocked("invalid", "empty_prefinal_evidence_path", "explicit_prefinal")
    try:
        payload, _, relative, file_sha256 = _load_public_json(configured)
        result = validator(payload)
        # Pre-final files are useful for metric inspection, but they cannot
        # promote the UI's quality/release gates without FINAL_STATUS binding.
        result.update(
            {
                "reported_status": result.get("status"),
                "reported_gate_status": result.get("gate_status"),
                "reported_gate_passed": result.get("gate_passed"),
                "status": "PRE_FINAL_ONLY",
                "gate_status": "BLOCKED",
                "gate_passed": False,
                "release_eligible": False,
            }
        )
        result.update(
            {
                "authority": "explicit_prefinal",
                "path": relative,
                "sha256": file_sha256,
            }
        )
        return result
    except EvidenceError as exc:
        return _blocked("invalid", exc.code, "explicit_prefinal")
    except Exception:
        return _blocked(
            "invalid", "prefinal_evidence_validation_failed", "explicit_prefinal"
        )


def _require_exact_keys(
    value: Any,
    expected: set[str],
    code: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise EvidenceError(code)
    return value


def _snapshot_text(
    value: Any,
    code: str,
    *,
    maximum_length: int,
    optional: bool = False,
) -> str | None:
    if value is None and optional:
        return None
    if (
        not isinstance(value, str)
        or not value
        or not value.strip()
        or len(value) > maximum_length
        or "\r" in value
        or "\n" in value
        or any(ord(character) < 32 and character != "\t" for character in value)
    ):
        raise EvidenceError(code)
    if _SENSITIVE_SNAPSHOT_TEXT.search(value):
        raise EvidenceError("device_snapshot_sensitive_content")
    return value


def _device_snapshot_path(raw_path: str | Path) -> tuple[Path, str]:
    """Resolve one non-symlink JSON below ``outputs/demo_runs``."""

    root = ROOT.resolve(strict=True)
    provided = Path(raw_path)
    if any(part == ".." for part in provided.parts):
        raise EvidenceError("device_snapshot_path_traversal")
    candidate = provided if provided.is_absolute() else root / provided
    lexical = Path(os.path.abspath(os.fspath(candidate)))
    try:
        relative = lexical.relative_to(root)
    except ValueError:
        raise EvidenceError("device_snapshot_outside_repository") from None
    folded_parts = tuple(part.casefold() for part in relative.parts)
    if len(folded_parts) < 3 or folded_parts[:2] != ("outputs", "demo_runs"):
        raise EvidenceError("device_snapshot_outside_demo_runs")
    if "private" in folded_parts[2:]:
        raise EvidenceError("device_snapshot_private_path_rejected")
    if lexical.suffix.casefold() != ".json":
        raise EvidenceError("device_snapshot_must_be_json")

    current = root
    try:
        for part in relative.parts:
            current = current / part
            if stat.S_ISLNK(current.lstat().st_mode):
                raise EvidenceError("device_snapshot_symlink_rejected")
        resolved = lexical.resolve(strict=True)
        demo_root = (root / "outputs" / "demo_runs").resolve(strict=True)
    except EvidenceError:
        raise
    except (OSError, RuntimeError):
        raise EvidenceError("device_snapshot_missing") from None
    try:
        resolved.relative_to(demo_root)
    except ValueError:
        raise EvidenceError("device_snapshot_outside_demo_runs") from None
    if os.path.normcase(os.fspath(resolved)) != os.path.normcase(os.fspath(lexical)):
        # This also fails closed for junction/reparse traversal on platforms
        # where ``is_symlink`` does not identify every indirection type.
        raise EvidenceError("device_snapshot_symlink_rejected")
    try:
        if not stat.S_ISREG(resolved.stat().st_mode):
            raise EvidenceError("device_snapshot_not_regular_file")
    except EvidenceError:
        raise
    except OSError:
        raise EvidenceError("device_snapshot_unreadable") from None
    return resolved, relative.as_posix()


def _read_device_snapshot(
    raw_path: str | Path,
) -> tuple[dict[str, Any], str, str]:
    """Read, validate and hash one immutable-sized JSON byte snapshot."""

    path, relative = _device_snapshot_path(raw_path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise EvidenceError("device_snapshot_unreadable") from None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise EvidenceError("device_snapshot_not_regular_file")
        if before.st_size > _MAX_DEVICE_SNAPSHOT_BYTES:
            raise EvidenceError("device_snapshot_too_large")
        chunks: list[bytes] = []
        remaining = _MAX_DEVICE_SNAPSHOT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
    except EvidenceError:
        raise
    except OSError:
        raise EvidenceError("device_snapshot_unreadable") from None
    finally:
        os.close(descriptor)
    if len(raw) > _MAX_DEVICE_SNAPSHOT_BYTES:
        raise EvidenceError("device_snapshot_too_large")
    try:
        after = path.stat(follow_symlinks=False)
    except OSError:
        raise EvidenceError("device_snapshot_changed_during_read") from None
    if (
        (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(raw) != before.st_size
    ):
        raise EvidenceError("device_snapshot_changed_during_read")
    # Revalidate every path component after reading so a parent cannot be
    # swapped for a symlink without failing this evidence read.
    verified_path, verified_relative = _device_snapshot_path(raw_path)
    if verified_path != path or verified_relative != relative:
        raise EvidenceError("device_snapshot_changed_during_read")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except EvidenceError:
        raise
    except (UnicodeError, json.JSONDecodeError):
        raise EvidenceError("invalid_device_snapshot_json") from None
    if not isinstance(payload, dict):
        raise EvidenceError("device_snapshot_not_object")
    return payload, relative, hashlib.sha256(raw).hexdigest()


def _snapshot_error_type(value: Any, code: str) -> str | None:
    value = _snapshot_text(value, code, maximum_length=128, optional=True)
    if value is not None and not _DEVICE_SNAPSHOT_ERROR_TYPE.fullmatch(value):
        raise EvidenceError(code)
    return value


def _streamlit_runtime_path(
    raw_path: Any,
    *,
    expected: Path,
    kind: str,
) -> tuple[Path, str]:
    text = _snapshot_text(
        raw_path,
        f"invalid_device_snapshot_streamlit_{kind}_path",
        maximum_length=4096,
    )
    provided = Path(text)
    if not provided.is_absolute() or any(part == ".." for part in provided.parts):
        raise EvidenceError(f"invalid_device_snapshot_streamlit_{kind}_path")
    lexical = Path(os.path.abspath(os.fspath(provided)))
    expected = Path(os.path.abspath(os.fspath(expected)))
    if os.path.normcase(os.fspath(lexical)) != os.path.normcase(os.fspath(expected)):
        raise EvidenceError(f"invalid_device_snapshot_streamlit_{kind}_path")
    root = ROOT.resolve(strict=True)
    try:
        relative = lexical.relative_to(root)
    except ValueError:
        raise EvidenceError("device_snapshot_streamlit_path_outside_repository") from None
    current = root
    try:
        for part in relative.parts:
            current = current / part
            if stat.S_ISLNK(current.lstat().st_mode):
                raise EvidenceError("device_snapshot_streamlit_symlink_rejected")
        resolved = lexical.resolve(strict=True)
    except EvidenceError:
        raise
    except (OSError, RuntimeError):
        raise EvidenceError("device_snapshot_streamlit_path_missing") from None
    if os.path.normcase(os.fspath(resolved)) != os.path.normcase(os.fspath(lexical)):
        raise EvidenceError("device_snapshot_streamlit_symlink_rejected")
    if kind == "module":
        if not resolved.is_file():
            raise EvidenceError("device_snapshot_streamlit_module_not_file")
    elif not resolved.is_dir():
        raise EvidenceError("device_snapshot_streamlit_distribution_not_directory")
    return resolved, relative.as_posix()


def _stable_optional_file_hash(path: Path, claimed: Any, *, label: str) -> str | None:
    if claimed is not None:
        claimed = _require_sha256(
            claimed,
            f"invalid_device_snapshot_streamlit_{label}_sha256",
        )
    if not path.exists():
        if claimed is not None:
            raise EvidenceError(f"device_snapshot_streamlit_{label}_missing")
        return None
    if claimed is None:
        raise EvidenceError(f"device_snapshot_streamlit_{label}_hash_missing")
    if path.is_symlink():
        raise EvidenceError("device_snapshot_streamlit_symlink_rejected")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise EvidenceError(f"device_snapshot_streamlit_{label}_unreadable") from None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise EvidenceError(f"device_snapshot_streamlit_{label}_not_file")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
    except EvidenceError:
        raise
    except OSError:
        raise EvidenceError(f"device_snapshot_streamlit_{label}_unreadable") from None
    finally:
        os.close(descriptor)
    try:
        after = path.stat(follow_symlinks=False)
    except OSError:
        raise EvidenceError("device_snapshot_streamlit_changed_during_read") from None
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise EvidenceError("device_snapshot_streamlit_changed_during_read")
    if digest.hexdigest() != claimed:
        raise EvidenceError(f"device_snapshot_streamlit_{label}_sha256_mismatch")
    return claimed


def _validate_device_snapshot(
    payload: dict[str, Any],
    *,
    snapshot_relative: str,
) -> dict[str, Any]:
    _require_exact_keys(
        payload,
        _DEVICE_SNAPSHOT_TOP_LEVEL_KEYS,
        "invalid_device_snapshot_schema",
    )
    if payload["schema_version"] != 2:
        raise EvidenceError("unsupported_device_snapshot_schema")
    if payload["kind"] != "dki_demo_device_snapshot":
        raise EvidenceError("invalid_device_snapshot_kind")
    created_at = payload["created_at"]
    if not isinstance(created_at, str) or not _DEVICE_SNAPSHOT_CREATED_AT.fullmatch(
        created_at
    ):
        raise EvidenceError("invalid_device_snapshot_created_at")
    try:
        datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise EvidenceError("invalid_device_snapshot_created_at") from None
    if payload["requested_logical_device"] != "npu:0":
        raise EvidenceError("invalid_device_snapshot_logical_device")

    detected_available = payload["detected_available"]
    if not isinstance(detected_available, bool):
        raise EvidenceError("invalid_device_snapshot_detected_available")
    detected_device_name = _snapshot_text(
        payload["detected_device_name"],
        "invalid_device_snapshot_detected_device_name",
        maximum_length=256,
        optional=True,
    )
    environment_label = payload["environment_label"]
    if environment_label is not None:
        environment_label = _snapshot_text(
            environment_label,
            "invalid_device_snapshot_environment_label",
            maximum_length=128,
        )
    raw_sources = payload["detection_sources"]
    if (
        not isinstance(raw_sources, list)
        or any(not isinstance(source, str) for source in raw_sources)
        or any(source not in {"torch_npu", "npu-smi"} for source in raw_sources)
        or len(raw_sources) != len(set(raw_sources))
    ):
        raise EvidenceError("invalid_device_snapshot_detection_sources")

    commands = _require_exact_keys(
        payload["commands"],
        _DEVICE_SNAPSHOT_COMMAND_KEYS,
        "invalid_device_snapshot_commands",
    )
    npu_smi = _require_exact_keys(
        commands["npu_smi"],
        _DEVICE_SNAPSHOT_NPU_SMI_KEYS,
        "invalid_device_snapshot_npu_smi",
    )
    binary_available = npu_smi["binary_available"]
    if not isinstance(binary_available, bool):
        raise EvidenceError("invalid_device_snapshot_npu_smi_available")
    returncode = npu_smi["returncode"]
    if returncode is not None and (
        isinstance(returncode, bool)
        or not isinstance(returncode, int)
        or not -(2**31) <= returncode < 2**31
    ):
        raise EvidenceError("invalid_device_snapshot_npu_smi_returncode")
    stdout_sha256 = npu_smi["stdout_sha256"]
    if stdout_sha256 is not None:
        _require_sha256(
            stdout_sha256,
            "invalid_device_snapshot_npu_smi_stdout_sha256",
        )
    summary_lines = npu_smi["summary_lines"]
    if not isinstance(summary_lines, list) or len(summary_lines) > 80:
        raise EvidenceError("invalid_device_snapshot_summary_lines")
    clean_summary = [
        _snapshot_text(
            line,
            "invalid_device_snapshot_summary_line",
            maximum_length=500,
        )
        for line in summary_lines
    ]
    npu_smi_error = _snapshot_error_type(
        npu_smi["error_type"],
        "invalid_device_snapshot_npu_smi_error_type",
    )
    if not binary_available and (
        returncode is not None
        or stdout_sha256 is not None
        or clean_summary
        or npu_smi_error is not None
    ):
        raise EvidenceError("inconsistent_unavailable_npu_smi_snapshot")
    if binary_available and returncode is None and npu_smi_error is None:
        raise EvidenceError("inconsistent_unexecuted_npu_smi_snapshot")
    if returncode is None and (stdout_sha256 is not None or clean_summary):
        raise EvidenceError("inconsistent_unexecuted_npu_smi_snapshot")
    if returncode is not None and stdout_sha256 is None:
        raise EvidenceError("missing_device_snapshot_npu_smi_stdout_sha256")
    if returncode is not None and npu_smi_error is not None:
        raise EvidenceError("inconsistent_device_snapshot_npu_smi_error")

    runtime = _require_exact_keys(
        commands["python_runtime"],
        _DEVICE_SNAPSHOT_RUNTIME_KEYS,
        "invalid_device_snapshot_python_runtime",
    )
    python_version = _snapshot_text(
        runtime["python"],
        "invalid_device_snapshot_python_version",
        maximum_length=128,
    )
    torch_version = _snapshot_text(
        runtime["torch"],
        "invalid_device_snapshot_torch_version",
        maximum_length=128,
        optional=True,
    )
    torch_npu_version = _snapshot_text(
        runtime["torch_npu"],
        "invalid_device_snapshot_torch_npu_version",
        maximum_length=128,
        optional=True,
    )
    device_name = _snapshot_text(
        runtime["device_name"],
        "invalid_device_snapshot_device_name",
        maximum_length=256,
        optional=True,
    )
    npu_available = runtime["npu_available"]
    if not isinstance(npu_available, bool):
        raise EvidenceError("invalid_device_snapshot_npu_available")
    runtime_error = _snapshot_error_type(
        runtime["error_type"],
        "invalid_device_snapshot_python_runtime_error_type",
    )
    if torch_version is None and (
        torch_npu_version is not None or device_name is not None
    ):
        raise EvidenceError("inconsistent_device_snapshot_python_runtime")
    if torch_npu_version is None and device_name is not None:
        raise EvidenceError("inconsistent_device_snapshot_python_runtime")
    if npu_available and (
        torch_version is None
        or torch_npu_version is None
        or device_name is None
        or runtime_error is not None
    ):
        raise EvidenceError("inconsistent_device_snapshot_python_runtime")
    if not npu_available and device_name is not None:
        raise EvidenceError("inconsistent_device_snapshot_python_runtime")

    smi_detected = bool(
        returncode == 0 and _DEVICE_SNAPSHOT_910.search("\n".join(clean_summary))
    )
    expected_sources = [
        source
        for source, active in (("torch_npu", npu_available), ("npu-smi", smi_detected))
        if active
    ]
    if raw_sources != expected_sources or detected_available is not bool(expected_sources):
        raise EvidenceError("inconsistent_device_snapshot_detection")
    if detected_available:
        if environment_label != "910C 验收环境":
            raise EvidenceError("invalid_device_snapshot_environment_label")
        expected_name = device_name if npu_available else None
        if detected_device_name != expected_name:
            raise EvidenceError("inconsistent_device_snapshot_detected_device_name")
    elif (
        detected_device_name is not None
        or environment_label is not None
        or raw_sources
    ):
        raise EvidenceError("undetected_device_snapshot_claims_npu")

    streamlit = _require_exact_keys(
        payload["streamlit_runtime"],
        _DEVICE_SNAPSHOT_STREAMLIT_KEYS,
        "invalid_device_snapshot_streamlit_runtime",
    )
    if streamlit["version"] != "1.54.0":
        raise EvidenceError("invalid_device_snapshot_streamlit_version")
    snapshot_path = ROOT.joinpath(*Path(snapshot_relative).parts)
    demo_root = snapshot_path.parent
    dependency_root = demo_root / ".demo_deps"
    module, module_relative = _streamlit_runtime_path(
        streamlit["module_path"],
        expected=dependency_root / "streamlit" / "__init__.py",
        kind="module",
    )
    distribution, distribution_relative = _streamlit_runtime_path(
        streamlit["distribution_path"],
        expected=dependency_root / "streamlit-1.54.0.dist-info",
        kind="distribution",
    )
    del module
    record_sha256 = _stable_optional_file_hash(
        distribution / "RECORD",
        streamlit["record_sha256"],
        label="record",
    )
    metadata_sha256 = _stable_optional_file_hash(
        distribution / "METADATA",
        streamlit["metadata_sha256"],
        label="metadata",
    )

    privacy = _require_exact_keys(
        payload["privacy"],
        _DEVICE_SNAPSHOT_PRIVACY_KEYS,
        "invalid_device_snapshot_privacy",
    )
    if (
        privacy["contains_environment"] is not False
        or privacy["contains_secrets"] is not False
    ):
        raise EvidenceError("unsafe_device_snapshot_privacy")
    return {
        "schema_version": 2,
        "kind": "dki_demo_device_snapshot",
        "created_at": created_at,
        "requested_logical_device": "npu:0",
        "detected_available": detected_available,
        "detected_device_name": detected_device_name,
        "environment_label": environment_label,
        "detection_sources": list(raw_sources),
        "streamlit_runtime": {
            "version": "1.54.0",
            "module_path": module_relative,
            "distribution_path": distribution_relative,
            "record_sha256": record_sha256,
            "metadata_sha256": metadata_sha256,
        },
        "commands": {
            "npu_smi": {
                "binary_available": binary_available,
                "returncode": returncode,
                "stdout_sha256": stdout_sha256,
                "summary_lines": clean_summary,
                "error_type": npu_smi_error,
            },
            "python_runtime": {
                "python": python_version,
                "torch": torch_version,
                "torch_npu": torch_npu_version,
                "npu_available": npu_available,
                "device_name": device_name,
                "error_type": runtime_error,
            },
        },
        "privacy": {
            "contains_environment": False,
            "contains_secrets": False,
        },
    }


def current_device_snapshot_evidence() -> dict[str, Any]:
    """Return optional, non-authoritative live device evidence fail-closed."""

    configured = os.getenv(DEVICE_SNAPSHOT_ENV)
    if configured is None:
        return {
            "status": "unavailable",
            "available": False,
            "authority": "none",
            "sha256": None,
            "error_code": "device_snapshot_not_configured",
        }
    if not configured.strip():
        return {
            "status": "unavailable",
            "available": False,
            "authority": "explicit_device_snapshot",
            "sha256": None,
            "error_code": "empty_device_snapshot_path",
        }
    try:
        payload, relative, file_sha256 = _read_device_snapshot(configured)
        result = _validate_device_snapshot(
            payload,
            snapshot_relative=relative,
        )
        result.update(
            {
                "status": "available",
                "available": True,
                "authority": "explicit_device_snapshot",
                "path": relative,
                "sha256": file_sha256,
            }
        )
        return result
    except EvidenceError as exc:
        return {
            "status": "unavailable",
            "available": False,
            "authority": "explicit_device_snapshot",
            "sha256": None,
            "error_code": exc.code,
        }
    except Exception:
        return {
            "status": "unavailable",
            "available": False,
            "authority": "explicit_device_snapshot",
            "sha256": None,
            "error_code": "device_snapshot_validation_failed",
        }


def npu_evidence() -> dict[str, Any]:
    try:
        exists = NPU_ACCEPTANCE.is_file()
        actual = _sha256(NPU_ACCEPTANCE) if exists else None
    except OSError:
        exists = False
        actual = None
    bound = actual == NPU_ACCEPTANCE_SHA256
    return {
        "status": "bound" if bound else "hash_mismatch",
        "hash_bound": bound,
        "dataset_role": "historical_official_dev_and_npu_acceptance",
        "environment_label": "910C 验收环境",
        "reported_device": "Ascend 910 / Ascend910_9382",
        "device_claim": "910C 验收环境（驱动报告 Ascend 910 / Ascend910_9382）",
        "driver": "npu-smi 25.5.1",
        "cann": "9.1.0 beta 3",
        "torch": "2.7.1",
        "torch_npu": "2.7.1.post6",
        "official_dev_f1": 0.613020,
        # Keep the original field while exposing the stable app schema.
        "npu_docs_per_second": 326.098,
        "micro_batch_docs_per_second": 326.098,
        "npu_cpu_speedup": 17.7295,
        "energy_j_per_document": 0.664121,
        "idle_subtracted_energy_j_per_document": 0.157099,
        "acceptance_path": str(NPU_ACCEPTANCE),
        "acceptance_sha256": actual,
        "expected_acceptance_sha256": NPU_ACCEPTANCE_SHA256,
        # The current snapshot is display-only evidence.  It is deliberately
        # excluded from ``hash_bound`` and every release/quality calculation.
        "current_device_snapshot": current_device_snapshot_evidence(),
    }


def cmeie_evidence() -> dict[str, Any]:
    return _component_evidence(
        aliases=("cmeie", "cmeie_public_summary"),
        prefinal_env=CMEIE_SUMMARY_ENV,
        validator=_validate_cmeie,
    )


def nl2sql_evidence() -> dict[str, Any]:
    return _component_evidence(
        aliases=("nl2sql", "nl2sql_public_attestation"),
        prefinal_env=NL2SQL_ATTESTATION_ENV,
        validator=_validate_nl2sql,
    )


def _release_blocked(
    status: str,
    error_code: str,
    *,
    final: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": status,
        "gate_status": "BLOCKED",
        "available": False,
        "hash_bound": False,
        "authority": "none",
        "sha256": None,
        "quality_gates_passed": False,
        "ready_for_submission": False,
        "error_code": error_code,
    }
    if final is not None:
        result.update(
            {
                "final_status": final["status"],
                "authority": final["authority"],
                "sha256": final["sha256"],
                "final_status_path": final["relative_path"],
                "final_status_sha256": final["sha256"],
                "source_tree_sha256": final["source_tree_sha256"],
            }
        )
    return result


def release_evidence(
    *,
    cmeie: dict[str, Any] | None = None,
    nl2sql: dict[str, Any] | None = None,
    npu: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one hash-bound release verdict without promoting pre-final data."""

    try:
        final = _load_final_status()
    except EvidenceError as exc:
        return _release_blocked("invalid", exc.code)
    except Exception:
        return _release_blocked("invalid", "final_status_validation_failed")
    if final is None:
        return _release_blocked(
            "unavailable", "authoritative_final_status_not_configured"
        )

    cmeie = cmeie if cmeie is not None else cmeie_evidence()
    nl2sql = nl2sql if nl2sql is not None else nl2sql_evidence()
    npu = npu if npu is not None else npu_evidence()
    components = (cmeie, nl2sql)
    components_bound = all(
        component.get("available") is True
        and component.get("authority") == "final_status"
        and component.get("final_status_sha256") == final["sha256"]
        and component.get("final_source_tree_sha256")
        == final["source_tree_sha256"]
        and component.get("source_tree_sha256") == final["source_tree_sha256"]
        and _is_sha256(component.get("sha256"))
        for component in components
    )
    if not components_bound:
        return _release_blocked(
            "BLOCKED", "incomplete_or_unbound_release_evidence", final=final
        )

    both_component_gates = all(
        component.get("gate_passed") is True for component in components
    )
    declared_quality = final["status"] in _QUALITY_PASSED_FINAL_STATUSES
    if declared_quality and not both_component_gates:
        return _release_blocked(
            "BLOCKED", "final_status_quality_claim_mismatch", final=final
        )

    # A complete final status may be NOT_READY because packaging/demo gates
    # failed even though both model-quality components passed.  Keep those two
    # facts separate: component quality is true only after final binding, while
    # submission readiness still requires the explicit READY status.
    quality_gates_passed = both_component_gates
    npu_bound = npu.get("status") == "bound" and npu.get("hash_bound") is True
    ready_for_submission = (
        final["status"] == "READY_FOR_SUBMISSION"
        and quality_gates_passed
        and npu_bound
    )
    if final["status"] == "READY_FOR_SUBMISSION" and not ready_for_submission:
        blocked = _release_blocked(
            "BLOCKED", "ready_release_not_fully_hash_bound", final=final
        )
        blocked.update(
            {
                "available": True,
                "quality_gates_passed": quality_gates_passed,
                "cmeie_sha256": cmeie["sha256"],
                "nl2sql_sha256": nl2sql["sha256"],
                "npu_acceptance_sha256": npu.get("acceptance_sha256"),
                "npu_hash_bound": False,
            }
        )
        return blocked

    return {
        "status": final["status"],
        "gate_status": "PASSED" if quality_gates_passed else "BLOCKED",
        "available": True,
        "hash_bound": npu_bound,
        "authority": final["authority"],
        "sha256": final["sha256"],
        "quality_gates_passed": quality_gates_passed,
        "ready_for_submission": ready_for_submission,
        "final_status": final["status"],
        "final_status_path": final["relative_path"],
        "final_status_sha256": final["sha256"],
        "source_tree_sha256": final["source_tree_sha256"],
        "cmeie_sha256": cmeie["sha256"],
        "nl2sql_sha256": nl2sql["sha256"],
        "npu_acceptance_sha256": npu.get("acceptance_sha256"),
        "npu_hash_bound": npu_bound,
    }


def public_registry() -> dict[str, Any]:
    npu = npu_evidence()
    cmeie = cmeie_evidence()
    nl2sql = nl2sql_evidence()
    return {
        "cmeee": {
            "dataset_role": "official_dev_historical_acceptance",
            "f1": 0.613020,
            "status": (
                "historical_hash_bound"
                if npu["hash_bound"]
                else "historical_evidence_invalid"
            ),
            "acceptance_sha256": NPU_ACCEPTANCE_SHA256,
        },
        "cmeie": cmeie,
        "nl2sql": nl2sql,
        "npu": npu,
        "release": release_evidence(cmeie=cmeie, nl2sql=nl2sql, npu=npu),
    }


__all__ = [
    "cmeie_evidence",
    "current_device_snapshot_evidence",
    "nl2sql_evidence",
    "npu_evidence",
    "public_registry",
    "release_evidence",
]
