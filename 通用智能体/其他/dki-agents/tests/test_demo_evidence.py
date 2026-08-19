from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dki.demo import evidence


HASH = "a" * 64


@pytest.fixture
def isolated_repository(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(evidence, "ROOT", tmp_path)
    for name in (
        evidence.FINAL_STATUS_ENV,
        evidence.CMEIE_SUMMARY_ENV,
        evidence.NL2SQL_ATTESTATION_ENV,
        evidence.DEVICE_SNAPSHOT_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cmeie_summary(
    *, f1: float = 0.6, source_hash: str = HASH
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "cmeie_public_summary",
        "dataset_role": "official_dev_development_confirmation",
        "model_family": "span_pair_cross_encoder",
        "source_tree_sha256": source_hash,
        "outer_accessed": True,
        "official_dev_accessed": True,
        "quality_gate_passed": True,
        "status": "PASSED",
        "metrics": {"precision": f1, "recall": f1, "f1": f1},
    }


def _nl2sql_attestation(*, source_hash: str = HASH) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "blind_evaluation_public_attestation",
        "status": "consumed",
        "gate_passed": True,
        "gate_reason": "passed",
        "engine": "agent",
        "source_tree_sha256": source_hash,
        "candidate_seal_sha256": HASH,
        "freeze_manifest_sha256": HASH,
        "holdout_sha256": HASH,
        "evaluator_sha256": HASH,
        "overall": {
            "n": 120,
            "n_correct": 103,
            "n_answered": 120,
            "accuracy_num": 103,
            "accuracy_den": 120,
            "coverage_num": 120,
            "coverage_den": 120,
            "strict_result_contract": True,
        },
        "hard": {
            "n": 40,
            "n_correct": 28,
            "n_answered": 40,
            "accuracy_num": 28,
            "accuracy_den": 40,
            "coverage_num": 40,
            "coverage_den": 40,
            "derived_from_overall_single_inference": True,
        },
        "gate": {"passed": True},
        "public_attestation_rules": {
            "strict_scoring": True,
            "no_question_or_gold_sql_leakage": True,
            "single_inference_pass": True,
            "selection_used": False,
        },
    }


def _write_final_status(
    root: Path,
    cmeie_path: Path,
    nl2sql_path: Path,
    *,
    status: str = "READY_FOR_SUBMISSION",
    source_hash: str = HASH,
) -> Path:
    return _write_json(
        root / "FINAL_STATUS.json",
        {
            "schema_version": 1,
            "status": status,
            "source": {"included_tree_sha256": source_hash},
            "evidence": {
                "cmeie": {
                    "path": cmeie_path.relative_to(root).as_posix(),
                    "sha256": _sha256(cmeie_path),
                },
                "nl2sql": {
                    "path": nl2sql_path.relative_to(root).as_posix(),
                    "sha256": _sha256(nl2sql_path),
                },
            },
        },
    )


def _bind_test_npu(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    acceptance = root / "docs/remote_npu_acceptance_20260725.md"
    acceptance.parent.mkdir(parents=True, exist_ok=True)
    acceptance.write_text("hash-bound NPU acceptance\n", encoding="utf-8")
    monkeypatch.setattr(evidence, "NPU_ACCEPTANCE", acceptance)
    monkeypatch.setattr(evidence, "NPU_ACCEPTANCE_SHA256", _sha256(acceptance))
    return acceptance


def _device_snapshot(
    snapshot_path: Path,
    *,
    device_name: str | None = "Ascend910_9382",
    summary_lines: list[str] | None = None,
    detected: bool = True,
) -> dict[str, object]:
    demo_root = snapshot_path.parent
    module = demo_root / ".demo_deps" / "streamlit" / "__init__.py"
    distribution = demo_root / ".demo_deps" / "streamlit-1.54.0.dist-info"
    module.parent.mkdir(parents=True, exist_ok=True)
    distribution.mkdir(parents=True, exist_ok=True)
    module.write_text("__version__='1.54.0'\n", encoding="utf-8")
    record = distribution / "RECORD"
    metadata = distribution / "METADATA"
    record.write_text("streamlit/__init__.py\n", encoding="utf-8")
    metadata.write_text("Name: streamlit\nVersion: 1.54.0\n", encoding="utf-8")
    lines = (
        ["Ascend 910 / Ascend910_9382", "npu-smi 25.5.1"]
        if summary_lines is None else summary_lines
    )
    smi_detected = any("910" in line for line in lines)
    sources = []
    if detected:
        sources.append("torch_npu")
        if smi_detected:
            sources.append("npu-smi")
    return {
        "schema_version": 2,
        "kind": "dki_demo_device_snapshot",
        "created_at": "2026-08-07T12:34:56Z",
        "requested_logical_device": "npu:0",
        "detected_available": detected,
        "detected_device_name": device_name if detected else None,
        "environment_label": "910C 验收环境" if detected else None,
        "detection_sources": sources if detected else [],
        "streamlit_runtime": {
            "version": "1.54.0",
            "module_path": str(module.resolve()),
            "distribution_path": str(distribution.resolve()),
            "record_sha256": _sha256(record),
            "metadata_sha256": _sha256(metadata),
        },
        "commands": {
            "npu_smi": {
                "binary_available": detected,
                "returncode": 0 if detected else None,
                "stdout_sha256": HASH if detected else None,
                "summary_lines": lines if detected else [],
                "error_type": None,
            },
            "python_runtime": {
                "python": "3.11.6",
                "torch": "2.7.1" if detected else None,
                "torch_npu": "2.7.1.post6" if detected else None,
                "npu_available": detected,
                "device_name": device_name if detected else None,
                "error_type": None if detected else "ImportError",
            },
        },
        "privacy": {
            "contains_environment": False,
            "contains_secrets": False,
        },
    }


def test_missing_authoritative_files_fail_closed(
    isolated_repository: Path,
) -> None:
    del isolated_repository

    cmeie = evidence.cmeie_evidence()
    nl2sql = evidence.nl2sql_evidence()

    assert cmeie["status"] == "unavailable"
    assert cmeie["gate_status"] == "BLOCKED"
    assert nl2sql["status"] == "unavailable"
    assert nl2sql["gate_status"] == "BLOCKED"


def test_newest_mtime_lures_are_never_discovered(
    isolated_repository: Path,
) -> None:
    _write_json(
        isolated_repository
        / "outputs/remote_runs/lure/evidence/cmeie/newest/selection.json",
        {"metrics": {"f1": 0.999}},
    )
    markdown = (
        isolated_repository
        / "outputs/remote_runs/lure/nl2sql/newest_blind_summary.md"
    )
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("overall 120/120\nhard 90/90\n", encoding="utf-8")

    assert evidence.cmeie_evidence()["status"] == "unavailable"
    assert evidence.nl2sql_evidence()["status"] == "unavailable"


def test_explicit_private_evidence_path_is_rejected(
    isolated_repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_summary = _write_json(
        isolated_repository / "outputs/private/run/cmeie_summary.json",
        _cmeie_summary(),
    )
    monkeypatch.setenv(evidence.CMEIE_SUMMARY_ENV, str(private_summary))

    result = evidence.cmeie_evidence()

    assert result["status"] == "invalid"
    assert result["gate_status"] == "BLOCKED"
    assert result["error_code"] == "private_path_rejected"
    assert "outputs/private" not in json.dumps(result)


def test_final_status_hash_tampering_blocks_without_prefinal_fallback(
    isolated_repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cmeie = _write_json(
        isolated_repository / "outputs/remote_runs/final/cmeie.json",
        _cmeie_summary(),
    )
    nl2sql = _write_json(
        isolated_repository / "outputs/remote_runs/final/nl2sql.json",
        _nl2sql_attestation(),
    )
    _write_final_status(
        isolated_repository,
        cmeie,
        nl2sql,
        status="NOT_READY_FOR_SUBMISSION",
    )
    decoy = _write_json(
        isolated_repository / "outputs/remote_runs/decoy/cmeie.json",
        _cmeie_summary(f1=0.55),
    )
    monkeypatch.setenv(evidence.CMEIE_SUMMARY_ENV, str(decoy))
    cmeie.write_text(cmeie.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    result = evidence.cmeie_evidence()

    assert result["status"] == "invalid"
    assert result["gate_status"] == "BLOCKED"
    assert result["authority"] == "final_status"
    assert result["error_code"] == "evidence_sha256_mismatch"


def test_explicit_public_nl2sql_attestation_is_normalized(
    isolated_repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attestation = _write_json(
        isolated_repository / "outputs/remote_runs/pre-final/nl2sql.json",
        _nl2sql_attestation(),
    )
    monkeypatch.setenv(evidence.NL2SQL_ATTESTATION_ENV, str(attestation))

    result = evidence.nl2sql_evidence()

    assert result["status"] == "PRE_FINAL_ONLY"
    assert result["gate_status"] == "BLOCKED"
    assert result["gate_passed"] is False
    assert result["reported_gate_passed"] is True
    assert result["authority"] == "explicit_prefinal"
    assert result["correct"] == 103
    assert result["n"] == 120
    assert result["hard_correct"] == 28
    assert result["sha256"] == _sha256(attestation)


def test_repository_final_status_has_priority_over_prefinal_environment(
    isolated_repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_cmeie = _write_json(
        isolated_repository / "outputs/remote_runs/final/cmeie.json",
        _cmeie_summary(f1=0.6),
    )
    final_nl2sql = _write_json(
        isolated_repository / "outputs/remote_runs/final/nl2sql.json",
        _nl2sql_attestation(),
    )
    _write_final_status(isolated_repository, final_cmeie, final_nl2sql)
    prefinal_cmeie = _write_json(
        isolated_repository / "outputs/remote_runs/pre-final/cmeie.json",
        _cmeie_summary(f1=0.55),
    )
    monkeypatch.setenv(evidence.CMEIE_SUMMARY_ENV, str(prefinal_cmeie))

    result = evidence.cmeie_evidence()

    assert result["authority"] == "final_status"
    assert result["final_status"] == "READY_FOR_SUBMISSION"
    assert result["path"] == "outputs/remote_runs/final/cmeie.json"
    assert result["f1"] == pytest.approx(0.6)


def test_final_status_without_included_tree_binding_is_rejected(
    isolated_repository: Path,
) -> None:
    cmeie = _write_json(
        isolated_repository / "outputs/remote_runs/final/cmeie.json",
        _cmeie_summary(),
    )
    nl2sql = _write_json(
        isolated_repository / "outputs/remote_runs/final/nl2sql.json",
        _nl2sql_attestation(),
    )
    _write_json(
        isolated_repository / "FINAL_STATUS.json",
        {
            "schema_version": 1,
            "status": "READY_FOR_SUBMISSION",
            "evidence": {
                "cmeie": {
                    "path": cmeie.relative_to(isolated_repository).as_posix(),
                    "sha256": _sha256(cmeie),
                },
                "nl2sql": {
                    "path": nl2sql.relative_to(isolated_repository).as_posix(),
                    "sha256": _sha256(nl2sql),
                },
            },
        },
    )

    component = evidence.cmeie_evidence()
    release = evidence.release_evidence()

    assert component["status"] == "invalid"
    assert component["error_code"] == "missing_final_source_binding"
    assert release["status"] == "invalid"
    assert release["quality_gates_passed"] is False
    assert release["ready_for_submission"] is False


def test_component_source_tree_must_match_final_status(
    isolated_repository: Path,
) -> None:
    final_source = "b" * 64
    cmeie = _write_json(
        isolated_repository / "outputs/remote_runs/final/cmeie.json",
        _cmeie_summary(source_hash=HASH),
    )
    nl2sql = _write_json(
        isolated_repository / "outputs/remote_runs/final/nl2sql.json",
        _nl2sql_attestation(source_hash=HASH),
    )
    _write_final_status(
        isolated_repository,
        cmeie,
        nl2sql,
        source_hash=final_source,
    )

    component = evidence.cmeie_evidence()
    nl2sql_component = evidence.nl2sql_evidence()
    release = evidence.release_evidence()

    assert component["status"] == "invalid"
    assert component["error_code"] == "component_source_tree_mismatch"
    assert nl2sql_component["status"] == "invalid"
    assert nl2sql_component["error_code"] == "component_source_tree_mismatch"
    assert release["status"] == "BLOCKED"
    assert release["quality_gates_passed"] is False
    assert release["ready_for_submission"] is False


def test_npu_evidence_exposes_legacy_and_safe_app_schema(
    isolated_repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    acceptance = _bind_test_npu(isolated_repository, monkeypatch)

    result = evidence.npu_evidence()

    assert result["status"] == "bound"
    assert result["hash_bound"] is True
    assert result["npu_docs_per_second"] == pytest.approx(326.098)
    assert result["micro_batch_docs_per_second"] == pytest.approx(326.098)
    assert result["energy_j_per_document"] == pytest.approx(0.664121)
    assert result["device_claim"].startswith("910C 验收环境")
    assert result["driver"] == "npu-smi 25.5.1"
    assert result["cann"] == "9.1.0 beta 3"
    assert result["torch_npu"] == "2.7.1.post6"
    assert result["acceptance_path"] == str(acceptance)
    assert result["current_device_snapshot"] == {
        "status": "unavailable",
        "available": False,
        "authority": "none",
        "sha256": None,
        "error_code": "device_snapshot_not_configured",
    }


def test_valid_current_device_snapshot_is_additive_and_not_fabricated(
    isolated_repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot_path = (
        isolated_repository / "outputs/demo_runs/server-rid/device_snapshot.json"
    )
    snapshot = _write_json(
        snapshot_path,
        _device_snapshot(snapshot_path, device_name="Ascend910_9382-live"),
    )
    monkeypatch.setenv(evidence.DEVICE_SNAPSHOT_ENV, str(snapshot))
    _bind_test_npu(isolated_repository, monkeypatch)

    npu = evidence.public_registry()["npu"]
    current = npu["current_device_snapshot"]

    assert npu["status"] == "bound"
    assert npu["official_dev_f1"] == pytest.approx(0.613020)
    assert npu["micro_batch_docs_per_second"] == pytest.approx(326.098)
    assert npu["reported_device"] == "Ascend 910 / Ascend910_9382"
    assert current["status"] == "available"
    assert current["available"] is True
    assert current["authority"] == "explicit_device_snapshot"
    assert current["path"] == (
        "outputs/demo_runs/server-rid/device_snapshot.json"
    )
    assert current["sha256"] == _sha256(snapshot)
    assert current["schema_version"] == 2
    assert current["requested_logical_device"] == "npu:0"
    assert current["detected_available"] is True
    assert current["detected_device_name"] == "Ascend910_9382-live"
    assert current["detection_sources"] == ["torch_npu", "npu-smi"]
    assert current["environment_label"] == "910C 验收环境"
    assert current["streamlit_runtime"] == {
        "version": "1.54.0",
        "module_path": (
            "outputs/demo_runs/server-rid/.demo_deps/streamlit/__init__.py"
        ),
        "distribution_path": (
            "outputs/demo_runs/server-rid/.demo_deps/streamlit-1.54.0.dist-info"
        ),
        "record_sha256": _sha256(
            snapshot_path.parent
            / ".demo_deps/streamlit-1.54.0.dist-info/RECORD"
        ),
        "metadata_sha256": _sha256(
            snapshot_path.parent
            / ".demo_deps/streamlit-1.54.0.dist-info/METADATA"
        ),
    }
    assert (
        current["commands"]["python_runtime"]["device_name"]
        == "Ascend910_9382-live"
    )
    assert current["privacy"] == {
        "contains_environment": False,
        "contains_secrets": False,
    }


def test_valid_undetected_snapshot_is_available_without_claiming_910c(
    isolated_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_path = (
        isolated_repository / "outputs/demo_runs/no-npu/device_snapshot.json"
    )
    snapshot = _write_json(
        snapshot_path,
        _device_snapshot(snapshot_path, detected=False),
    )
    monkeypatch.setenv(evidence.DEVICE_SNAPSHOT_ENV, str(snapshot))

    current = evidence.current_device_snapshot_evidence()

    assert current["status"] == "available"
    assert current["available"] is True
    assert current["detected_available"] is False
    assert current["detected_device_name"] is None
    assert current["environment_label"] is None
    assert current["detection_sources"] == []
    assert current["commands"]["python_runtime"]["error_type"] == "ImportError"


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    [
        ("extra_key", "invalid_device_snapshot_schema"),
        ("wrong_logical_device", "invalid_device_snapshot_logical_device"),
        ("source_mismatch", "inconsistent_device_snapshot_detection"),
        ("device_name_mismatch", "inconsistent_device_snapshot_detected_device_name"),
        ("invalid_stdout_sha", "invalid_device_snapshot_npu_smi_stdout_sha256"),
        ("invalid_streamlit_hash", "device_snapshot_streamlit_record_sha256_mismatch"),
        ("extra_streamlit_key", "invalid_device_snapshot_streamlit_runtime"),
        ("streamlit_path_escape", "invalid_device_snapshot_streamlit_module_path"),
        ("forged_910c", "undetected_device_snapshot_claims_npu"),
        ("unsafe_privacy", "unsafe_device_snapshot_privacy"),
        ("sensitive_summary", "device_snapshot_sensitive_content"),
    ],
)
def test_invalid_current_device_snapshot_schema_fails_unavailable(
    isolated_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    error_code: str,
) -> None:
    snapshot_path = (
        isolated_repository
        / f"outputs/demo_runs/server-rid/{mutation}.json"
    )
    payload = _device_snapshot(
        snapshot_path,
        detected=mutation != "forged_910c",
    )
    if mutation == "extra_key":
        payload["unexpected"] = True
    elif mutation == "wrong_logical_device":
        payload["requested_logical_device"] = "npu:1"
    elif mutation == "source_mismatch":
        payload["detection_sources"] = ["torch_npu"]
    elif mutation == "device_name_mismatch":
        payload["detected_device_name"] = "forged-device"
    elif mutation == "invalid_stdout_sha":
        payload["commands"]["npu_smi"]["stdout_sha256"] = "not-a-hash"
    elif mutation == "invalid_streamlit_hash":
        payload["streamlit_runtime"]["record_sha256"] = "0" * 64
    elif mutation == "extra_streamlit_key":
        payload["streamlit_runtime"]["unexpected"] = True
    elif mutation == "streamlit_path_escape":
        payload["streamlit_runtime"]["module_path"] = str(
            (isolated_repository / "outside.py").resolve()
        )
    elif mutation == "forged_910c":
        payload["environment_label"] = "910C 验收环境"
    elif mutation == "unsafe_privacy":
        payload["privacy"]["contains_secrets"] = True
    else:
        payload["commands"]["npu_smi"]["summary_lines"] = [
            "API_KEY=must-not-be-exposed"
        ]
    snapshot = _write_json(
        snapshot_path,
        payload,
    )
    monkeypatch.setenv(evidence.DEVICE_SNAPSHOT_ENV, str(snapshot))
    _bind_test_npu(isolated_repository, monkeypatch)

    npu = evidence.public_registry()["npu"]
    current = npu["current_device_snapshot"]

    assert npu["status"] == "bound"
    assert npu["hash_bound"] is True
    assert current["status"] == "unavailable"
    assert current["available"] is False
    assert current["error_code"] == error_code
    assert current["sha256"] is None
    assert "path" not in current


@pytest.mark.parametrize(
    ("relative_path", "error_code"),
    [
        (
            "outputs/remote_runs/server-rid/device_snapshot.json",
            "device_snapshot_outside_demo_runs",
        ),
        (
            "outputs/demo_runs/server-rid/private/device_snapshot.json",
            "device_snapshot_private_path_rejected",
        ),
    ],
)
def test_current_device_snapshot_rejects_unsafe_locations(
    isolated_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
    error_code: str,
) -> None:
    snapshot_path = isolated_repository / relative_path
    snapshot = _write_json(
        snapshot_path,
        _device_snapshot(snapshot_path),
    )
    monkeypatch.setenv(evidence.DEVICE_SNAPSHOT_ENV, str(snapshot))

    current = evidence.npu_evidence()["current_device_snapshot"]

    assert current["status"] == "unavailable"
    assert current["error_code"] == error_code
    assert "path" not in current


def test_current_device_snapshot_rejects_symlink(
    isolated_repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_path = (
        isolated_repository
        / "outputs/demo_runs/server-rid/device_snapshot-target.json"
    )
    target = _write_json(
        target_path,
        _device_snapshot(target_path),
    )
    link = target.with_name("device_snapshot.json")
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are not available in this test environment")
    monkeypatch.setenv(evidence.DEVICE_SNAPSHOT_ENV, str(link))

    current = evidence.npu_evidence()["current_device_snapshot"]

    assert current["status"] == "unavailable"
    assert current["error_code"] == "device_snapshot_symlink_rejected"
    assert "path" not in current


def test_current_device_snapshot_has_independent_size_limit(
    isolated_repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot_path = (
        isolated_repository / "outputs/demo_runs/server-rid/device_snapshot.json"
    )
    snapshot = _write_json(
        snapshot_path,
        _device_snapshot(snapshot_path, summary_lines=["x" * (129 * 1024)]),
    )
    monkeypatch.setenv(evidence.DEVICE_SNAPSHOT_ENV, str(snapshot))

    current = evidence.npu_evidence()["current_device_snapshot"]

    assert current["status"] == "unavailable"
    assert current["error_code"] == "device_snapshot_too_large"
    assert current["sha256"] is None


def test_prefinal_component_gates_never_promote_release(
    isolated_repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cmeie = _write_json(
        isolated_repository / "outputs/remote_runs/pre-final/cmeie.json",
        _cmeie_summary(),
    )
    nl2sql = _write_json(
        isolated_repository / "outputs/remote_runs/pre-final/nl2sql.json",
        _nl2sql_attestation(),
    )
    monkeypatch.setenv(evidence.CMEIE_SUMMARY_ENV, str(cmeie))
    monkeypatch.setenv(evidence.NL2SQL_ATTESTATION_ENV, str(nl2sql))
    _bind_test_npu(isolated_repository, monkeypatch)

    registry = evidence.public_registry()

    assert registry["cmeie"]["gate_passed"] is False
    assert registry["cmeie"]["reported_gate_passed"] is True
    assert registry["nl2sql"]["gate_passed"] is False
    assert registry["nl2sql"]["reported_gate_passed"] is True
    assert registry["release"]["status"] == "unavailable"
    assert registry["release"]["quality_gates_passed"] is False
    assert registry["release"]["ready_for_submission"] is False


def test_complete_hash_bound_final_status_promotes_release(
    isolated_repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cmeie = _write_json(
        isolated_repository / "outputs/remote_runs/final/cmeie.json",
        _cmeie_summary(),
    )
    nl2sql = _write_json(
        isolated_repository / "outputs/remote_runs/final/nl2sql.json",
        _nl2sql_attestation(),
    )
    final = _write_final_status(isolated_repository, cmeie, nl2sql)
    acceptance = _bind_test_npu(isolated_repository, monkeypatch)

    registry = evidence.public_registry()
    release = registry["release"]

    assert release["status"] == "READY_FOR_SUBMISSION"
    assert release["gate_status"] == "PASSED"
    assert release["hash_bound"] is True
    assert release["authority"] == "repository_final_status"
    assert release["sha256"] == _sha256(final)
    assert release["quality_gates_passed"] is True
    assert release["ready_for_submission"] is True
    assert release["source_tree_sha256"] == HASH
    assert release["final_status_sha256"] == _sha256(final)
    assert release["cmeie_sha256"] == _sha256(cmeie)
    assert release["nl2sql_sha256"] == _sha256(nl2sql)
    assert release["npu_acceptance_sha256"] == _sha256(acceptance)


def test_complete_not_ready_final_can_report_quality_without_ready(
    isolated_repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cmeie = _write_json(
        isolated_repository / "outputs/remote_runs/final/cmeie.json",
        _cmeie_summary(),
    )
    nl2sql = _write_json(
        isolated_repository / "outputs/remote_runs/final/nl2sql.json",
        _nl2sql_attestation(),
    )
    _write_final_status(
        isolated_repository,
        cmeie,
        nl2sql,
        status="NOT_READY_FOR_SUBMISSION",
    )
    _bind_test_npu(isolated_repository, monkeypatch)

    release = evidence.public_registry()["release"]

    assert release["status"] == "NOT_READY_FOR_SUBMISSION"
    assert release["quality_gates_passed"] is True
    assert release["ready_for_submission"] is False


def test_ready_final_is_blocked_when_npu_document_is_not_hash_bound(
    isolated_repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cmeie = _write_json(
        isolated_repository / "outputs/remote_runs/final/cmeie.json",
        _cmeie_summary(),
    )
    nl2sql = _write_json(
        isolated_repository / "outputs/remote_runs/final/nl2sql.json",
        _nl2sql_attestation(),
    )
    _write_final_status(isolated_repository, cmeie, nl2sql)
    _bind_test_npu(isolated_repository, monkeypatch)
    monkeypatch.setattr(evidence, "NPU_ACCEPTANCE_SHA256", "b" * 64)

    release = evidence.public_registry()["release"]

    assert release["status"] == "BLOCKED"
    assert release["hash_bound"] is False
    assert release["quality_gates_passed"] is True
    assert release["ready_for_submission"] is False
    assert release["error_code"] == "ready_release_not_fully_hash_bound"
