"""Build the sealed, aggregate-only final competition evidence directory.

The builder never discovers a "latest" run.  Every authority is passed
explicitly, validated against one included-tree hash, and reduced to public
aggregate data before a sibling staging directory is atomically renamed into
its create-once destination.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping

from dki.demo.service import DemoRunService


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
HEAD_RE = re.compile(r"^[0-9a-f]{40}$")
RID_RE = re.compile(
    r"^(?P<stamp>\d{8}T\d{6}Z)-g(?P<head>[0-9a-f]{7})-t(?P<tree>[0-9a-f]{12})$"
)
TREE_ALGORITHM = "sha256 of sorted path NUL size NUL file_sha256 LF"
NPU_ACCEPTANCE_SHA256 = "f0e0a1feedada43fb24ab5fe4629c36e0e9f166753b6fc38910f1a39f7c5d015"
OUTPUT_FILENAMES = (
    "FINAL_STATUS.json",
    "逐项验收报告.md",
    "CMEIE_PUBLIC_SUMMARY.json",
    "NL2SQL_PUBLIC_ATTESTATION.json",
    "DEVICE_ENVIRONMENT.json",
    "SHA256SUMS.txt",
)
FORBIDDEN_OUTPUT_KEYS = {
    "question",
    "questions",
    "goldsql",
    "predictedsql",
    "predictionsql",
    "goldspo",
    "predictedspo",
    "holdoutrows",
    "developmentrows",
    "detailedresults",
    "errorsamples",
    "examples",
    "rows",
}
FORBIDDEN_PATH_FRAGMENTS = (
    "outputs/private/",
    "holdout.jsonl",
    "/.env",
    "cblue/",
)
FORBIDDEN_PACKAGE_PATH_FRAGMENTS = FORBIDDEN_PATH_FRAGMENTS + ("outputs/demo_runs/",)
FORBIDDEN_PACKAGE_SUFFIXES = {
    ".om", ".onnx", ".pt", ".pth", ".ckpt", ".safetensors", ".pem", ".key"
}
SECRET_TEXT_RE = re.compile(
    r"-----BEGIN [^-\r\n]*PRIVATE KEY-----|"
    r"(?:api[_-]?key|access[_-]?token|client[_-]?secret|password|passwd|"
    r"private[_-]?key|authorization|bearer)\s*[:=]",
    re.IGNORECASE,
)
SQL_TEXT_RE = re.compile(r"\b(?:select|with)\b[\s\S]{0,2000}\bfrom\b", re.IGNORECASE)


class FinalReportError(ValueError):
    """Fail-closed validation or publication error."""


@dataclass(frozen=True)
class BuildInputs:
    repo_root: Path
    output_dir: Path
    included_tree_manifest: Path
    demo_run_dir: Path
    cmeie_summary: Path
    nl2sql_attestation: Path
    npu_acceptance: Path
    device_snapshot: Path
    source_test_receipt: Path
    cleanroom_test_receipt: Path
    package_receipt: Path
    expected_npu_sha256: str = NPU_ACCEPTANCE_SHA256


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise FinalReportError(f"{label} must be lowercase SHA-256")
    return value


def _require_count(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise FinalReportError(f"{label} must be a non-negative integer")
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FinalReportError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise FinalReportError(f"non-finite JSON number: {value}")


def _load_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except FinalReportError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FinalReportError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise FinalReportError(f"{label} must be a JSON object")
    _walk_numbers(payload, label=label)
    return payload


def _walk_numbers(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise FinalReportError(f"{label} contains a non-finite number")
    if isinstance(value, dict):
        for child in value.values():
            _walk_numbers(child, label=label)
    elif isinstance(value, list):
        for child in value:
            _walk_numbers(child, label=label)


def _path_has_link(path: Path, stop: Path) -> bool:
    # Keep the lexical path until every component has been inspected.  Calling
    # resolve() first would erase the evidence that an input traversed a link.
    current = Path(os.path.abspath(path))
    stop = stop.resolve(strict=True)
    chain: list[Path] = []
    while True:
        chain.append(current)
        if current == stop:
            break
        if current.parent == current:
            raise FinalReportError(f"path is outside repository: {path}")
        current = current.parent
    for item in reversed(chain):
        metadata = os.lstat(item)
        attributes = getattr(metadata, "st_file_attributes", 0)
        if stat.S_ISLNK(metadata.st_mode) or bool(attributes & 0x0400):
            return True
    return False


def _public_path(
    repo_root: Path,
    raw_path: Path | str,
    *,
    label: str,
    want_directory: bool = False,
) -> tuple[Path, str]:
    root = Path(repo_root).resolve(strict=True)
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        candidate = Path(os.path.abspath(candidate))
        candidate.relative_to(root)
        if _path_has_link(candidate, root):
            raise FinalReportError(f"{label} traverses a symlink or reparse point")
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(root).as_posix()
    except FinalReportError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise FinalReportError(f"{label} is missing or outside the repository") from exc
    if want_directory:
        if not resolved.is_dir():
            raise FinalReportError(f"{label} must be a directory")
    elif not resolved.is_file():
        raise FinalReportError(f"{label} must be a regular file")
    normalized = relative.casefold().replace("\\", "/")
    if normalized == "outputs/private" or normalized.startswith("outputs/private/"):
        raise FinalReportError(f"{label} points into outputs/private")
    return resolved, relative


def _read_json(
    repo_root: Path, raw_path: Path | str, *, label: str
) -> tuple[dict[str, Any], Path, str, str]:
    path, relative = _public_path(repo_root, raw_path, label=label)
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ino != after.st_ino
    ):
        raise FinalReportError(f"{label} changed while being read")
    payload = _load_json_bytes(raw, label=label)
    _reject_private_payload(payload, label=label)
    return payload, path, relative, _sha256_bytes(raw)


def _canonical_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _reject_private_payload(value: Any, *, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if _canonical_key(key) in FORBIDDEN_OUTPUT_KEYS:
                raise FinalReportError(f"{label} contains forbidden row-level key: {key}")
            _reject_private_payload(child, label=label)
    elif isinstance(value, list):
        for child in value:
            _reject_private_payload(child, label=label)
    elif isinstance(value, str):
        normalized = value.casefold().replace("\\", "/")
        if any(fragment in normalized for fragment in FORBIDDEN_PATH_FRAGMENTS):
            raise FinalReportError(f"{label} contains a forbidden private path")
        if SECRET_TEXT_RE.search(value):
            raise FinalReportError(f"{label} contains sensitive text")


def _tree_hash(entries: Iterable[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(str(entry["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(entry["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _canonical_member_path(raw: object, *, label: str) -> str:
    if not isinstance(raw, str) or not raw or "\\" in raw or "\0" in raw:
        raise FinalReportError(f"{label} is not a canonical POSIX path")
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in raw.split("/"))
        or posix.as_posix() != raw
    ):
        raise FinalReportError(f"{label} is unsafe or noncanonical")
    return raw


def _validate_included_tree(
    repo_root: Path, path: Path
) -> dict[str, Any]:
    payload, manifest_path, relative, manifest_sha = _read_json(
        repo_root, path, label="included-tree manifest"
    )
    if payload.get("schema_version") != 1 or payload.get("algorithm") != TREE_ALGORITHM:
        raise FinalReportError("unsupported included-tree manifest schema/algorithm")
    rid = payload.get("rid")
    head = payload.get("git_head")
    claimed_tree = _require_sha256(payload.get("included_tree_sha256"), "included-tree hash")
    match = RID_RE.fullmatch(str(rid))
    if match is None or not isinstance(head, str) or HEAD_RE.fullmatch(head) is None:
        raise FinalReportError("included-tree RID or Git HEAD is invalid")
    try:
        time.strptime(match.group("stamp"), "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise FinalReportError("included-tree RID timestamp is invalid") from exc
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise FinalReportError("included-tree entries must be non-empty")
    entries: list[dict[str, Any]] = []
    identities: set[str] = set()
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise FinalReportError(f"included-tree entry {index} is invalid")
        relative_entry = _canonical_member_path(
            raw_entry.get("path"), label=f"included-tree entry {index}"
        )
        identity = relative_entry.casefold()
        if identity in identities:
            raise FinalReportError("included-tree contains duplicate/case-alias paths")
        identities.add(identity)
        size = _require_count(raw_entry.get("size"), f"included-tree size {index}")
        digest = _require_sha256(raw_entry.get("sha256"), f"included-tree hash {index}")
        source, actual_relative = _public_path(
            repo_root, relative_entry, label=f"included-tree source {relative_entry}"
        )
        if actual_relative != relative_entry:
            raise FinalReportError("included-tree source path is not canonical")
        if source.stat().st_size != size or _sha256_file(source) != digest:
            raise FinalReportError(f"included-tree source differs: {relative_entry}")
        entries.append({"path": relative_entry, "size": size, "sha256": digest})
    if [entry["path"] for entry in entries] != sorted(entry["path"] for entry in entries):
        raise FinalReportError("included-tree entries are not sorted")
    actual_tree = _tree_hash(entries)
    if actual_tree != claimed_tree:
        raise FinalReportError("included-tree aggregate hash mismatch")
    if (
        match.group("head") != head[:7]
        or match.group("tree") != claimed_tree[:12]
    ):
        raise FinalReportError("included-tree RID does not bind HEAD/tree prefixes")
    return {
        "rid": rid,
        "git_head": head,
        "included_tree_sha256": claimed_tree,
        "manifest_path": relative,
        "manifest_sha256": manifest_sha,
        "n_files": len(entries),
        "path": manifest_path,
    }


def _validate_demo(repo_root: Path, run_dir: Path, source_tree: str) -> dict[str, Any]:
    directory, relative = _public_path(
        repo_root, run_dir, label="demo run directory", want_directory=True
    )
    receipt = DemoRunService.read_receipt(directory)
    if not isinstance(receipt, dict):
        raise FinalReportError("demo receipt is missing, malformed, or failed live validation")
    if receipt.get("source_tree_sha256") != source_tree:
        raise FinalReportError("demo source tree differs from the final source tree")
    completed_path = directory / "completed.json"
    completed_sha = _sha256_file(completed_path) if completed_path.is_file() else None
    status = receipt.get("status")
    datamate_mode = receipt.get("datamate_mode")
    passed = (
        status == "completed"
        and receipt.get("returncode") == 0
        and receipt.get("_service_verified") is True
        and receipt.get("mode") == "agent-offline"
        and datamate_mode in {"real-datamate", "compat-shim"}
        and isinstance(receipt.get("artifacts"), dict)
        and len(receipt["artifacts"]) == 7
    )
    if status == "completed" and not passed:
        raise FinalReportError("completed demo receipt does not satisfy the verified contract")
    return {
        "passed": passed,
        "status": status,
        "run_id": receipt.get("run_id"),
        "mode": receipt.get("mode"),
        "run_directory": relative,
        "receipt_sha256": completed_sha,
        "datamate_mode": datamate_mode,
    }


def _metric_triplet(raw: object, *, label: str) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise FinalReportError(f"{label} must be a metric object")
    values: dict[str, float] = {}
    for name in ("precision", "recall", "f1"):
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FinalReportError(f"{label}.{name} is invalid")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise FinalReportError(f"{label}.{name} is out of range")
        values[name] = number
    expected = (
        0.0
        if values["precision"] + values["recall"] == 0
        else 2 * values["precision"] * values["recall"]
        / (values["precision"] + values["recall"])
    )
    if abs(values["f1"] - expected) > 0.001:
        raise FinalReportError(f"{label}.f1 is inconsistent with precision/recall")
    return values


def _seed_metrics(raw: object, *, label: str) -> dict[str, dict[str, float]]:
    if not isinstance(raw, dict):
        raise FinalReportError(f"{label} must be an object")
    result: dict[str, dict[str, float]] = {}
    for seed in (42, 43, 44):
        key = f"seed{seed}"
        if key not in raw:
            raise FinalReportError(f"{label} lacks {key}")
        result[key] = _metric_triplet(raw[key], label=f"{label}.{key}")
    return result


def _validate_cmeie(
    repo_root: Path, path: Path, source_tree: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, _path, relative, digest = _read_json(
        repo_root, path, label="CMeIE public summary"
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "cmeie_public_summary"
        or payload.get("source_tree_sha256") != source_tree
    ):
        raise FinalReportError("CMeIE public summary schema/source binding failed")
    declared = payload.get("quality_gate_passed", payload.get("gate_passed"))
    if not isinstance(declared, bool):
        raise FinalReportError("CMeIE public summary lacks a boolean gate")
    calibration = payload.get("calibration_gate")
    if not isinstance(calibration, dict) or not isinstance(calibration.get("passed"), bool):
        raise FinalReportError("CMeIE calibration gate is missing")
    threshold = calibration.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise FinalReportError("CMeIE calibration threshold is invalid")
    calibration_seeds = _seed_metrics(calibration, label="CMeIE calibration")
    calibration_pass = bool(
        calibration.get("passed")
        and float(threshold) >= 0.45
        and max(row["f1"] for row in calibration_seeds.values()) >= 0.45
    )

    outer_accessed = payload.get("outer_accessed")
    official_accessed = payload.get("official_dev_accessed")
    if type(outer_accessed) is not bool or type(official_accessed) is not bool:
        raise FinalReportError("CMeIE access flags are invalid")
    outer = payload.get("outer_gate")
    outer_seeds: dict[str, dict[str, float]] | None = None
    outer_pass = False
    median_f1: float | None = None
    worst_f1: float | None = None
    if outer_accessed:
        if not isinstance(outer, dict) or not isinstance(outer.get("passed"), bool):
            raise FinalReportError("CMeIE outer gate is missing")
        outer_seeds = _seed_metrics(outer, label="CMeIE outer")
        f1s = sorted(row["f1"] for row in outer_seeds.values())
        median_f1 = f1s[1]
        worst_f1 = f1s[0]
        outer_pass = bool(
            outer.get("passed")
            and median_f1 >= 0.50
            and worst_f1 >= 0.48
            and all(
                row["precision"] >= 0.45 and row["recall"] >= 0.45
                for row in outer_seeds.values()
            )
        )
        for field, actual in (("median_f1", median_f1), ("worst_f1", worst_f1)):
            if field in outer and abs(float(outer[field]) - actual) > 1e-9:
                raise FinalReportError(f"CMeIE outer {field} claim is inconsistent")
    elif outer is not None:
        raise FinalReportError("CMeIE outer metrics exist while outer_accessed is false")

    official_raw = payload.get("official_dev", payload.get("metrics"))
    official = (
        _metric_triplet(official_raw, label="CMeIE official dev")
        if official_accessed
        else None
    )
    if not official_accessed and official_raw is not None:
        raise FinalReportError("CMeIE official metrics exist while access flag is false")
    dataset_role = payload.get("dataset_role")
    model_family = payload.get("model_family")
    selected_seed = payload.get("selected_seed")
    selection_basis = payload.get("selection_basis")
    computed = bool(
        calibration_pass
        and outer_accessed
        and outer_pass
        and official_accessed
        and official is not None
        and official["f1"] >= 0.50
        and dataset_role == "official_dev_development_confirmation"
        and selected_seed in {42, 43, 44}
        and selection_basis == "calibration_f1"
    )
    if declared is not computed:
        raise FinalReportError("CMeIE declared gate is inconsistent with hard thresholds")
    public: dict[str, Any] = {
        "schema_version": 1,
        "kind": "cmeie_public_summary",
        "status": "PASSED" if computed else "BLOCKED",
        "quality_gate_passed": computed,
        "dataset_role": dataset_role,
        "model_family": model_family,
        "source_tree_sha256": source_tree,
        "outer_accessed": outer_accessed,
        "official_dev_accessed": official_accessed,
        "selected_seed": selected_seed,
        "selection_basis": selection_basis,
        "calibration_gate": {
            "passed": calibration_pass,
            "threshold": float(threshold),
            **calibration_seeds,
        },
        "outer_gate": (
            {
                "passed": outer_pass,
                "median_f1": median_f1,
                "worst_f1": worst_f1,
                **(outer_seeds or {}),
            }
            if outer_accessed
            else None
        ),
    }
    if official is not None:
        public["official_dev"] = official
    return public, {"passed": computed, "input_path": relative, "input_sha256": digest}


def _nl_counts(raw: object, *, label: str) -> dict[str, int]:
    if not isinstance(raw, dict):
        raise FinalReportError(f"{label} must be an object")
    n = _require_count(raw.get("n"), f"{label}.n")
    correct = _require_count(raw.get("n_correct"), f"{label}.n_correct")
    answered = _require_count(raw.get("n_answered"), f"{label}.n_answered")
    if not correct <= answered <= n:
        raise FinalReportError(f"{label} counts are inconsistent")
    for key, expected in (
        ("accuracy_num", correct), ("accuracy_den", n),
        ("coverage_num", answered), ("coverage_den", n),
    ):
        if key in raw and raw[key] != expected:
            raise FinalReportError(f"{label}.{key} is inconsistent")
    return {"n": n, "correct": correct, "answered": answered}


def _validate_nl2sql(
    repo_root: Path, path: Path, source_tree: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, _path, relative, digest = _read_json(
        repo_root, path, label="NL2SQL public attestation"
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "blind_evaluation_public_attestation"
        or payload.get("status") != "consumed"
        or payload.get("source_tree_sha256") != source_tree
    ):
        raise FinalReportError("NL2SQL public attestation schema/source binding failed")
    for key in (
        "candidate_seal_sha256", "freeze_manifest_sha256", "holdout_sha256",
        "evaluator_sha256",
    ):
        _require_sha256(payload.get(key), f"NL2SQL {key}")
    rules = payload.get("public_attestation_rules")
    expected_rules = {
        "strict_scoring": True,
        "no_question_or_gold_sql_leakage": True,
        "single_inference_pass": True,
        "selection_used": False,
    }
    if not isinstance(rules, dict) or any(rules.get(k) is not v for k, v in expected_rules.items()):
        raise FinalReportError("NL2SQL public rules are invalid")
    overall_raw = payload.get("overall")
    hard_raw = payload.get("hard")
    if not isinstance(overall_raw, dict) or overall_raw.get("strict_result_contract") is not True:
        raise FinalReportError("NL2SQL strict result contract is missing")
    if not isinstance(hard_raw, dict) or hard_raw.get("derived_from_overall_single_inference") is not True:
        raise FinalReportError("NL2SQL hard metric was not derived from the single pass")
    overall = _nl_counts(overall_raw, label="NL2SQL overall")
    hard = _nl_counts(hard_raw, label="NL2SQL hard")
    computed = bool(
        overall["n"] >= 120
        and hard["n"] >= 40
        and 100 * overall["correct"] > 85 * overall["n"]
        and 100 * hard["correct"] >= 70 * hard["n"]
    )
    declared = payload.get("gate_passed")
    if type(declared) is not bool or declared is not computed:
        raise FinalReportError("NL2SQL declared gate is inconsistent with integer thresholds")
    gate = payload.get("gate")
    if gate is not None and (not isinstance(gate, dict) or gate.get("passed") is not computed):
        raise FinalReportError("NL2SQL nested gate is inconsistent")
    reason = payload.get("gate_reason")
    if computed and reason not in {None, "passed"}:
        raise FinalReportError("NL2SQL passed gate has an invalid reason")
    if not computed and (not isinstance(reason, str) or not reason):
        raise FinalReportError("NL2SQL failed gate lacks a reason")
    public = {
        "schema_version": 1,
        "kind": "blind_evaluation_public_attestation",
        "status": "consumed",
        "gate_passed": computed,
        "gate_reason": "passed" if computed else reason,
        "engine": payload.get("engine"),
        "source_tree_sha256": source_tree,
        "candidate_seal_sha256": payload["candidate_seal_sha256"],
        "freeze_manifest_sha256": payload["freeze_manifest_sha256"],
        "holdout_sha256": payload["holdout_sha256"],
        "evaluator_sha256": payload["evaluator_sha256"],
        "overall": {
            "n": overall["n"], "n_correct": overall["correct"],
            "n_answered": overall["answered"],
            "accuracy_num": overall["correct"], "accuracy_den": overall["n"],
            "coverage_num": overall["answered"], "coverage_den": overall["n"],
            "strict_result_contract": True,
        },
        "hard": {
            "n": hard["n"], "n_correct": hard["correct"],
            "n_answered": hard["answered"],
            "accuracy_num": hard["correct"], "accuracy_den": hard["n"],
            "coverage_num": hard["answered"], "coverage_den": hard["n"],
            "derived_from_overall_single_inference": True,
        },
        "gate": {"passed": computed},
        "public_attestation_rules": expected_rules,
    }
    return public, {"passed": computed, "input_path": relative, "input_sha256": digest}


def _validate_binding(
    repo_root: Path, raw: object, *, label: str
) -> tuple[Path, str, str]:
    if not isinstance(raw, dict):
        raise FinalReportError(f"{label} binding is missing")
    path, relative = _public_path(repo_root, raw.get("path", ""), label=label)
    size = _require_count(raw.get("bytes"), f"{label}.bytes")
    expected = _require_sha256(raw.get("sha256"), f"{label}.sha256")
    actual = _sha256_file(path)
    if path.stat().st_size != size or actual != expected:
        raise FinalReportError(f"{label} binding differs from live bytes")
    return path, relative, actual


def _validate_test_receipt(
    repo_root: Path, path: Path, source_tree: str, *, expected_stage: str
) -> dict[str, Any]:
    payload, _path, relative, digest = _read_json(
        repo_root, path, label=f"{expected_stage} full-test receipt"
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "dki_full_test_receipt"
        or payload.get("stage") != expected_stage
        or payload.get("source_tree_sha256") != source_tree
    ):
        raise FinalReportError(f"{expected_stage} full-test schema/source binding failed")
    counts = payload.get("counts")
    if not isinstance(counts, dict):
        raise FinalReportError(f"{expected_stage} test counts are missing")
    parsed = {
        name: _require_count(counts.get(name), f"{expected_stage}.{name}")
        for name in ("collected", "passed", "failed", "errors", "skipped", "xpassed")
    }
    if parsed["passed"] + parsed["skipped"] + parsed["failed"] + parsed["errors"] != parsed["collected"]:
        raise FinalReportError(f"{expected_stage} test counts are inconsistent")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise FinalReportError(f"{expected_stage} test artifacts are missing")
    artifact_hashes: dict[str, str] = {}
    for name in ("stdout", "stderr", "junit", "node_ids", "skip_reasons"):
        _bound, _relative, artifact_hashes[name] = _validate_binding(
            repo_root, artifacts.get(name), label=f"{expected_stage} test {name}"
        )
    returncode = payload.get("returncode")
    status = payload.get("status")
    expected_pass = bool(
        status == "passed"
        and returncode == 0
        and parsed["failed"] == parsed["errors"] == parsed["xpassed"] == 0
        and payload.get("node_ids_complete") is True
        and payload.get("missing_baseline_nodes") == 0
        and payload.get("new_skips") == 0
        and payload.get("non_agent_returncode") == 0
        and payload.get("agent_returncode") == 0
    )
    if status == "passed" and not expected_pass:
        raise FinalReportError(f"{expected_stage} test receipt has an unsupported pass claim")
    if status not in {"passed", "failed"}:
        raise FinalReportError(f"{expected_stage} test status is invalid")
    return {
        "passed": expected_pass,
        "status": status,
        "path": relative,
        "sha256": digest,
        "counts": parsed,
        "artifact_hashes": artifact_hashes,
    }


def _validate_device(repo_root: Path, path: Path) -> dict[str, Any]:
    payload, _path, relative, digest = _read_json(
        repo_root, path, label="device snapshot"
    )
    if payload.get("schema_version") != 2 or payload.get("kind") != "dki_demo_device_snapshot":
        raise FinalReportError("device snapshot v2 schema is required")
    if payload.get("requested_logical_device") != "npu:0":
        raise FinalReportError("device snapshot must request npu:0")
    detected = payload.get("detected_available")
    if type(detected) is not bool:
        raise FinalReportError("device detected_available must be boolean")
    sources = payload.get("detection_sources")
    if not isinstance(sources, list) or any(item not in {"torch_npu", "npu-smi"} for item in sources):
        raise FinalReportError("device detection_sources are invalid")
    commands = payload.get("commands")
    runtime = commands.get("python_runtime") if isinstance(commands, dict) else None
    smi = commands.get("npu_smi") if isinstance(commands, dict) else None
    if not isinstance(runtime, dict) or not isinstance(smi, dict):
        raise FinalReportError("device command summaries are missing")
    privacy = payload.get("privacy")
    if not isinstance(privacy, dict) or privacy.get("contains_environment") is not False or privacy.get("contains_secrets") is not False:
        raise FinalReportError("device snapshot privacy flags are invalid")
    if "stdout_sha256" in smi and smi.get("stdout_sha256") is not None:
        _require_sha256(smi.get("stdout_sha256"), "npu-smi stdout hash")
    streamlit = payload.get("streamlit_runtime")
    if not isinstance(streamlit, dict) or streamlit.get("version") != "1.54.0":
        raise FinalReportError("device snapshot lacks the pinned Streamlit runtime")
    passed = bool(
        detected
        and sources
        and isinstance(payload.get("detected_device_name"), str)
        and payload.get("detected_device_name")
        and isinstance(runtime.get("python"), str)
        and isinstance(runtime.get("torch"), str)
        and isinstance(runtime.get("torch_npu"), str)
    )
    return {
        "passed": passed,
        "input_path": relative,
        "input_sha256": digest,
        "public": {
            "schema_version": 1,
            "kind": "dki_device_environment_public_summary",
            "created_at": payload.get("created_at"),
            "requested_logical_device": "npu:0",
            "detected_available": detected,
            "detected_device_name": payload.get("detected_device_name"),
            "environment_label": payload.get("environment_label"),
            "detection_sources": sources,
            "python": runtime.get("python"),
            "torch": runtime.get("torch"),
            "torch_npu": runtime.get("torch_npu"),
            "npu_smi_returncode": smi.get("returncode"),
            "npu_smi_stdout_sha256": smi.get("stdout_sha256"),
            "streamlit": streamlit.get("version"),
            "privacy": {
                "contains_environment": False,
                "contains_secrets": False,
            },
        },
    }


def _validate_manifest_payload(payload: dict[str, Any], source_tree: str) -> list[dict[str, Any]]:
    if payload.get("schema_version") != 1 or payload.get("source_tree_sha256") != source_tree:
        raise FinalReportError("package manifest schema/source binding failed")
    rows = payload.get("files")
    if not isinstance(rows, list):
        raise FinalReportError("package manifest files are missing")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise FinalReportError(f"package manifest row {index} is invalid")
        member = _canonical_member_path(row.get("path"), label=f"package manifest row {index}")
        identity = member.casefold()
        if identity in seen:
            raise FinalReportError("package manifest contains duplicate/case-alias paths")
        seen.add(identity)
        normalized = member.casefold()
        if (
            any(fragment in normalized for fragment in FORBIDDEN_PACKAGE_PATH_FRAGMENTS)
            or Path(member).suffix.casefold() in FORBIDDEN_PACKAGE_SUFFIXES
        ):
            raise FinalReportError(f"package manifest contains forbidden artifact: {member}")
        result.append({
            "path": member,
            "bytes": _require_count(row.get("bytes"), f"package bytes {index}"),
            "sha256": _require_sha256(row.get("sha256"), f"package hash {index}"),
        })
    if payload.get("n_files") != len(result):
        raise FinalReportError("package manifest n_files is inconsistent")
    return result


def _validate_package(
    repo_root: Path,
    path: Path,
    source_tree: str,
    cleanroom_test: dict[str, Any],
) -> dict[str, Any]:
    payload, _path, relative, digest = _read_json(
        repo_root, path, label="clean-room package receipt"
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "clean_room_package_verification"
        or payload.get("source_tree_sha256") != source_tree
    ):
        raise FinalReportError("package receipt schema/source binding failed")
    archive, _archive_relative, _archive_sha = _validate_binding(
        repo_root, payload.get("archive"), label="submission archive"
    )
    manifest_path, _manifest_relative, _manifest_sha = _validate_binding(
        repo_root, payload.get("manifest"), label="external package manifest"
    )
    manifest_raw = manifest_path.read_bytes()
    manifest_payload = _load_json_bytes(manifest_raw, label="external package manifest")
    _reject_private_payload(manifest_payload, label="external package manifest")
    rows = _validate_manifest_payload(manifest_payload, source_tree)
    manifest_claim = payload.get("manifest_checks")
    required_manifest_claims = {
        "external_matches_internal": True,
        "duplicate_members": 0,
        "missing_members": 0,
        "extra_members": 0,
        "path_traversal_members": 0,
        "absolute_members": 0,
        "symlink_members": 0,
    }
    if not isinstance(manifest_claim, dict) or any(
        manifest_claim.get(key) != value for key, value in required_manifest_claims.items()
    ):
        raise FinalReportError("package manifest-check receipt is not clean")
    privacy = payload.get("privacy")
    if not isinstance(privacy, dict) or any(
        privacy.get(key) != value
        for key, value in {
            "passed": True,
            "secret_hits": 0,
            "private_data_hits": 0,
            "forbidden_artifact_hits": 0,
        }.items()
    ):
        raise FinalReportError("package privacy receipt is not clean")
    clean = payload.get("clean_room")
    if (
        not isinstance(clean, dict)
        or clean.get("test_receipt_sha256") != cleanroom_test["sha256"]
        or clean.get("passed") is not cleanroom_test["passed"]
    ):
        raise FinalReportError("package receipt is not bound to clean-room tests")
    internal_name = payload.get("internal_manifest_path")
    internal_name = _canonical_member_path(internal_name, label="internal manifest path")
    try:
        with zipfile.ZipFile(archive, "r") as bundle:
            infos = bundle.infolist()
            names = [info.filename for info in infos if not info.is_dir()]
            if len(names) != len(set(names)) or len({name.casefold() for name in names}) != len(names):
                raise FinalReportError("submission ZIP contains duplicate/case-alias members")
            for info in infos:
                if info.is_dir():
                    continue
                _canonical_member_path(info.filename, label="submission ZIP member")
                if stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK:
                    raise FinalReportError("submission ZIP contains a symlink")
            if internal_name not in names:
                raise FinalReportError("submission ZIP lacks its internal manifest")
            if bundle.read(internal_name) != manifest_raw:
                raise FinalReportError("external and internal package manifests differ")
            expected_names = {row["path"] for row in rows} | {internal_name}
            if set(names) != expected_names:
                raise FinalReportError("submission ZIP members differ from the manifest")
            for row in rows:
                content = bundle.read(row["path"])
                if len(content) != row["bytes"] or _sha256_bytes(content) != row["sha256"]:
                    raise FinalReportError(f"submission ZIP member hash differs: {row['path']}")
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise FinalReportError("submission archive is not a valid manifest-bound ZIP") from exc
    declared = payload.get("status") == "passed"
    passed = bool(declared and cleanroom_test["passed"])
    if declared and not passed:
        raise FinalReportError("package receipt has an unsupported pass claim")
    return {
        "passed": passed,
        "status": payload.get("status"),
        "path": relative,
        "sha256": digest,
        "archive_sha256": _sha256_file(archive),
        "manifest_sha256": _sha256_bytes(manifest_raw),
        "n_files": len(rows),
    }


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    _reject_private_payload(payload, label="generated public evidence")
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    # Parse exactly what will be published as a final self-check.
    _load_json_bytes(encoded, label="generated public evidence")
    return encoded


def _binding(path: str, encoded: bytes) -> dict[str, Any]:
    return {"path": path, "bytes": len(encoded), "sha256": _sha256_bytes(encoded)}


SCORING_POINTS = (
    ("任务一", "自然语言任务与 Agent 规划", 5, "Nexent CoreAgent 调度唯一 CLI 业务链", "项目总览→运行完整闭环；任务一→查看 Agent 轨迹", "demo"),
    ("任务一", "DataMate 数据处理算子", 10, "真实 DataMate 或兼容 shim，实际模式进入回执", "任务一→查看 DataMate 模式与 manifest", "datamate"),
    ("任务一", "状态、产物与可复现性", 10, "create-once run、输入/源码/七项产物哈希", "总览→选择运行；任务一→运行证据", "demo"),
    ("任务一", "异常恢复与执行轨迹", 5, "events 与恢复结果的 aggregate-only 摘要", "任务一→恢复轨迹", "demo"),
    ("任务二", "医疗实体抽取", 10, "KG 实体流水线；CMeEE 正式质量独立分栏", "任务二→实体统计；NPU 性能证据", "npu"),
    ("任务二", "关系与三元组抽取", 10, "联合 text-to-SPO 模型通过显式配置启用", "任务二→关系、三元组与核心子图", "cmeie"),
    ("任务二", "Schema 校验、去重与可追溯存储", 10, "KG 与 provenance 逐级哈希绑定", "任务二→统计、证据与子图", "demo"),
    ("任务二", "知识图谱问答与可视化", 10, "结构化答案、置信度、原文证据与 ECharts 子图", "任务二→图谱问答", "demo"),
    ("任务二", "Ascend NPU 性能加分", 10, "910C 验收环境历史正式 CMeEE 证据", "NPU 性能证据→设备与指标", "npu"),
    ("任务三", "复用任务一、二数据血缘", 5, "Task1 manifest、KG、SQLite provenance 串联", "任务三→数据库 provenance", "demo"),
    ("任务三", "自然语言到安全 SQL", 10, "QueryPlan/rule 路由、只读完整执行与独立门槛", "任务三→生成并执行 SQL", "nl2sql"),
    ("任务三", "统计分析与完整结果", 10, "完整结果可下载，页面仅预览 200 行", "任务三→预览与 CSV", "demo"),
    ("任务三", "BI 图表与图文说明", 5, "ECharts BI 报告及哈希绑定", "任务三→打开 BI 报告", "demo"),
)


def _score_rows(gates: Mapping[str, bool], hashes: Mapping[str, str | None]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task, point, score, implementation, step, authority in SCORING_POINTS:
        gate = gates[authority]
        if authority in {"cmeie", "nl2sql", "datamate"} and gates["demo"] and not gate:
            verdict = "PARTIAL"
        else:
            verdict = "PASS" if gate else "BLOCKED"
        rows.append({
            "task": task,
            "point": point,
            "score": score,
            "implementation": implementation,
            "demo_step": step,
            "data_role": {
                "demo": "controlled synthetic / verified demo",
                "datamate": "controlled synthetic / verified demo runtime",
                "cmeie": "official dev development confirmation + family-disjoint outer",
                "nl2sql": "prospective independent holdout / aggregate-only",
                "npu": "historical official dev and 910C NPU acceptance",
            }[authority],
            "evidence_sha256": hashes.get(authority),
            "verdict": verdict,
            "limitation": {
                "demo": "演示正确性不替代独立质量门",
                "datamate": "compat-shim 明确记为 PARTIAL，不冒充真实 DataMate",
                "cmeie": "官方 dev 已用于开发确认，不宣称真正盲测",
                "nl2sql": "不公开题目、SQL 或逐题错误",
                "npu": "历史验收与当前设备快照分栏展示",
            }[authority],
        })
    return rows


def _report_markdown(
    *,
    status: str,
    blockers: list[str],
    rows: list[dict[str, Any]],
    cmeie: Mapping[str, Any],
    nl2sql: Mapping[str, Any],
    npu_hash: str,
    final_status_sha: str,
) -> str:
    overall = nl2sql["overall"]
    hard = nl2sql["hard"]
    overall_ratio = overall["n_correct"] / overall["n"] if overall["n"] else 0.0
    hard_ratio = hard["n_correct"] / hard["n"] if hard["n"] else 0.0
    official = cmeie.get("official_dev") or {}
    lines = [
        "# dki-agents 比赛逐项验收报告",
        "",
        f"- 最终状态：`{status}`",
        f"- FINAL_STATUS SHA-256：`{final_status_sha}`",
        f"- 阻塞项：{('、'.join(blockers) if blockers else '无')}",
        "- 隐私边界：本文仅含聚合指标，不含题目、金标准/预测 SQL、SPO 明细或逐条错误。",
        "",
        "## 核心质量与性能",
        "",
        f"- CMeIE 完整 text-to-SPO 官方 dev 开发确认：P/R/F1="
        f"{official.get('precision', 0):.6f}/{official.get('recall', 0):.6f}/{official.get('f1', 0):.6f}。",
        f"- NL2SQL prospective independent：overall {overall['n_correct']}/{overall['n']}="
        f"{overall_ratio:.2%}；hard+ {hard['n_correct']}/{hard['n']}="
        f"{hard_ratio:.2%}。",
        "- CMeEE 历史正式验收：官方 dev F1 0.613020；NPU 326.098 docs/s；NPU/CPU 17.7295×。",
        f"- NPU 验收文档 SHA-256：`{npu_hash}`。",
        "",
        "## 13 个评分点逐项对照（总分 110，含 NPU 加分）",
        "",
        "| 任务 | 评分项 | 分值 | 当前实现 | 浏览器演示步骤 | 数据角色 | 证据 SHA-256 | 判定 | 已知限制 |",
        "|---|---|---:|---|---|---|---|---|---|",
    ]
    for row in rows:
        values = [
            row["task"], row["point"], str(row["score"]), row["implementation"],
            row["demo_step"], row["data_role"], f"`{row['evidence_sha256'] or 'unavailable'}`",
            row["verdict"], row["limitation"],
        ]
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |")
    lines.extend([
        "",
        "## 最终判定规则",
        "",
        "只有演示、CMeIE、NL2SQL、NPU、密封源码全测、clean-room 全测以及包隐私/manifest "
        "全部通过时才可声明 `READY_FOR_SUBMISSION`。任何其他状态均为 "
        "`NOT_READY_FOR_SUBMISSION`，开发集、gold-subject 或历史 ZIP 不得替代缺失门槛。",
        "",
    ])
    text = "\n".join(lines)
    if SQL_TEXT_RE.search(text) or SECRET_TEXT_RE.search(text):
        raise FinalReportError("generated report unexpectedly contains sensitive detail")
    return text


def _write_new(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_rename_directory(staging: Path, output: Path) -> None:
    if os.name == "nt":
        move = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
        move.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
        move.restype = ctypes.c_int
        if move(str(staging), str(output), 0x00000008):
            return
        error = ctypes.get_last_error()
        if error in {80, 183}:
            raise FileExistsError(f"final report directory already exists: {output}")
        raise OSError(error, ctypes.FormatError(error), str(output))
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace directory rename is unavailable")
    renameat2.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(staging),
        -100,
        os.fsencode(output),
        1,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(f"final report directory already exists: {output}")
    raise OSError(error, os.strerror(error), str(output))


def _output_destination(repo_root: Path, output_dir: Path) -> Path:
    root = Path(repo_root).resolve(strict=True)
    output = Path(output_dir)
    if not output.is_absolute():
        output = root / output
    output = Path(os.path.abspath(output))
    try:
        relative = output.relative_to(root).as_posix().casefold()
    except ValueError as exc:
        raise FinalReportError("final report output must remain inside the repository") from exc
    if relative == "outputs/private" or relative.startswith("outputs/private/"):
        raise FinalReportError("final report output must not be private")
    if output.exists():
        raise FileExistsError(f"final report directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if _path_has_link(output.parent, root):
        raise FinalReportError("final report output parent traverses a link")
    return output


def build_final_report(inputs: BuildInputs) -> dict[str, Any]:
    repo_root = Path(inputs.repo_root).resolve(strict=True)
    output_dir = _output_destination(repo_root, inputs.output_dir)
    source = _validate_included_tree(repo_root, inputs.included_tree_manifest)
    tree = source["included_tree_sha256"]
    demo = _validate_demo(repo_root, inputs.demo_run_dir, tree)
    cmeie_public, cmeie_input = _validate_cmeie(repo_root, inputs.cmeie_summary, tree)
    nl_public, nl_input = _validate_nl2sql(repo_root, inputs.nl2sql_attestation, tree)
    expected_npu = _require_sha256(inputs.expected_npu_sha256, "expected NPU acceptance hash")
    npu_path, npu_relative = _public_path(
        repo_root, inputs.npu_acceptance, label="NPU acceptance document"
    )
    npu_hash = _sha256_file(npu_path)
    if npu_hash != expected_npu:
        raise FinalReportError("historical NPU acceptance hash changed")
    device = _validate_device(repo_root, inputs.device_snapshot)
    source_tests = _validate_test_receipt(
        repo_root, inputs.source_test_receipt, tree, expected_stage="sealed_source"
    )
    clean_tests = _validate_test_receipt(
        repo_root, inputs.cleanroom_test_receipt, tree, expected_stage="clean_room"
    )
    package = _validate_package(
        repo_root, inputs.package_receipt, tree, clean_tests
    )

    gates = {
        "demo": demo["passed"],
        "cmeie": cmeie_input["passed"],
        "nl2sql": nl_input["passed"],
        "npu": True,
        "device": device["passed"],
        "source_tests": source_tests["passed"],
        "clean_room": clean_tests["passed"],
        "package": package["passed"],
    }
    blocker_names = {
        "demo": "demo_not_verified",
        "cmeie": "cmeie_quality_gate_failed",
        "nl2sql": "nl2sql_quality_gate_failed",
        "npu": "npu_acceptance_invalid",
        "device": "current_npu_device_not_detected",
        "source_tests": "sealed_source_full_tests_failed",
        "clean_room": "clean_room_full_tests_failed",
        "package": "package_privacy_or_manifest_failed",
    }
    blockers = [blocker_names[name] for name, passed in gates.items() if not passed]
    ready = not blockers
    status = "READY_FOR_SUBMISSION" if ready else "NOT_READY_FOR_SUBMISSION"

    final_root_relative = output_dir.relative_to(repo_root).as_posix()
    cmeie_bytes = _json_bytes(cmeie_public)
    nl_bytes = _json_bytes(nl_public)
    device_bytes = _json_bytes(device["public"])
    cmeie_output = f"{final_root_relative}/CMEIE_PUBLIC_SUMMARY.json"
    nl_output = f"{final_root_relative}/NL2SQL_PUBLIC_ATTESTATION.json"
    device_output = f"{final_root_relative}/DEVICE_ENVIRONMENT.json"
    final_payload = {
        "schema_version": 1,
        "kind": "dki_final_competition_status",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
        "quality_gates_passed": bool(gates["cmeie"] and gates["nl2sql"]),
        "ready_for_submission": ready,
        "blockers": blockers,
        "source": {
            "rid": source["rid"],
            "git_head": source["git_head"],
            "included_tree_sha256": tree,
            "manifest_path": source["manifest_path"],
            "manifest_sha256": source["manifest_sha256"],
            "n_files": source["n_files"],
        },
        "gates": gates,
        "evidence": {
            "cmeie": _binding(cmeie_output, cmeie_bytes),
            "nl2sql": _binding(nl_output, nl_bytes),
            "demo": {
                "run_id": demo["run_id"],
                "mode": demo["mode"],
                "datamate_mode": demo["datamate_mode"],
                "completed_receipt_sha256": demo["receipt_sha256"],
            },
            "npu": {"path": npu_relative, "sha256": npu_hash},
            "device": _binding(device_output, device_bytes),
            "sealed_source_tests": {
                "path": source_tests["path"], "sha256": source_tests["sha256"]
            },
            "clean_room_tests": {
                "path": clean_tests["path"], "sha256": clean_tests["sha256"]
            },
            "package_verification": {
                "path": package["path"], "sha256": package["sha256"],
                "archive_sha256": package["archive_sha256"],
                "manifest_sha256": package["manifest_sha256"],
            },
        },
        "privacy": {
            "aggregate_only": True,
            "contains_questions": False,
            "contains_sql": False,
            "contains_spo_details": False,
            "contains_private_paths": False,
            "contains_secrets": False,
        },
    }
    final_bytes = _json_bytes(final_payload)
    score_hashes = {
        "demo": demo["receipt_sha256"],
        "datamate": demo["receipt_sha256"],
        "cmeie": _sha256_bytes(cmeie_bytes),
        "nl2sql": _sha256_bytes(nl_bytes),
        "npu": npu_hash,
    }
    score_gates = {
        **gates,
        "datamate": bool(demo["passed"] and demo["datamate_mode"] == "real-datamate"),
    }
    rows = _score_rows(score_gates, score_hashes)
    report = _report_markdown(
        status=status,
        blockers=blockers,
        rows=rows,
        cmeie=cmeie_public,
        nl2sql=nl_public,
        npu_hash=npu_hash,
        final_status_sha=_sha256_bytes(final_bytes),
    ).encode("utf-8")
    files: dict[str, bytes] = {
        "FINAL_STATUS.json": final_bytes,
        "逐项验收报告.md": report,
        "CMEIE_PUBLIC_SUMMARY.json": cmeie_bytes,
        "NL2SQL_PUBLIC_ATTESTATION.json": nl_bytes,
        "DEVICE_ENVIRONMENT.json": device_bytes,
    }
    sums = "".join(
        f"{_sha256_bytes(content)}  {name}\n"
        for name, content in sorted(files.items())
    ).encode("utf-8")
    files["SHA256SUMS.txt"] = sums
    if tuple(files) != OUTPUT_FILENAMES:
        raise AssertionError("final output filename contract drifted")

    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    published = False
    try:
        for name, content in files.items():
            _write_new(staging / name, content)
        if os.name != "nt":
            descriptor = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        _atomic_rename_directory(staging, output_dir)
        published = True
        if os.name != "nt":
            descriptor = os.open(output_dir.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
    return {
        "status": status,
        "ready_for_submission": ready,
        "blockers": blockers,
        "output_dir": str(output_dir),
        "final_status_sha256": _sha256_bytes(final_bytes),
        "files": {name: _sha256_bytes(content) for name, content in files.items()},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an atomic aggregate-only final competition evidence directory"
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--included-tree-manifest", type=Path, required=True)
    parser.add_argument("--demo-run-dir", type=Path, required=True)
    parser.add_argument("--cmeie-summary", type=Path, required=True)
    parser.add_argument("--nl2sql-attestation", type=Path, required=True)
    parser.add_argument("--npu-acceptance", type=Path, required=True)
    parser.add_argument("--device-snapshot", type=Path, required=True)
    parser.add_argument("--source-test-receipt", type=Path, required=True)
    parser.add_argument("--cleanroom-test-receipt", type=Path, required=True)
    parser.add_argument("--package-receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_final_report(
        BuildInputs(
            repo_root=args.repo_root,
            output_dir=args.output_dir,
            included_tree_manifest=args.included_tree_manifest,
            demo_run_dir=args.demo_run_dir,
            cmeie_summary=args.cmeie_summary,
            nl2sql_attestation=args.nl2sql_attestation,
            npu_acceptance=args.npu_acceptance,
            device_snapshot=args.device_snapshot,
            source_test_receipt=args.source_test_receipt,
            cleanroom_test_receipt=args.cleanroom_test_receipt,
            package_receipt=args.package_receipt,
        )
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
