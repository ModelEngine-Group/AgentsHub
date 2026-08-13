from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import sys
import zipfile
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_final_competition_report.py"
SPEC = importlib.util.spec_from_file_location("dki_final_report_builder", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
report_builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = report_builder
SPEC.loader.exec_module(report_builder)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _binding(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": _rel(root, path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _metric(value: float) -> dict[str, float]:
    return {"precision": value, "recall": value, "f1": value}


def _included_tree(root: Path) -> tuple[Path, str]:
    source = root / "src" / "source.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    entry = {
        "path": _rel(root, source),
        "size": source.stat().st_size,
        "sha256": _sha256(source),
    }
    tree = report_builder._tree_hash([entry])
    head = "1234567" + "0" * 33
    manifest = {
        "schema_version": 1,
        "rid": f"20260807T120000Z-g{head[:7]}-t{tree[:12]}",
        "git_head": head,
        "algorithm": report_builder.TREE_ALGORITHM,
        "included_tree_sha256": tree,
        "entries": [entry],
    }
    return _write_json(root / "evidence" / "included-tree.json", manifest), tree


def _cmeie_summary(root: Path, tree: str, *, official_f1: float) -> Path:
    passed = official_f1 >= 0.50
    payload = {
        "schema_version": 1,
        "kind": "cmeie_public_summary",
        "source_tree_sha256": tree,
        "quality_gate_passed": passed,
        "model_family": "span_pair_cross_encoder",
        "selected_seed": 42,
        "selection_basis": "calibration_f1",
        "dataset_role": "official_dev_development_confirmation",
        "outer_accessed": True,
        "official_dev_accessed": True,
        "calibration_gate": {
            "passed": True,
            "threshold": 0.45,
            "seed42": _metric(0.48),
            "seed43": _metric(0.47),
            "seed44": _metric(0.46),
        },
        "outer_gate": {
            "passed": True,
            "median_f1": 0.50,
            "worst_f1": 0.48,
            "seed42": _metric(0.52),
            "seed43": _metric(0.50),
            "seed44": _metric(0.48),
        },
        "official_dev": _metric(official_f1),
    }
    return _write_json(root / "evidence" / "cmeie-public.json", payload)


def _nl2sql_attestation(
    root: Path,
    tree: str,
    *,
    overall_correct: int,
    hard_correct: int,
) -> Path:
    passed = 100 * overall_correct > 85 * 120 and 100 * hard_correct >= 70 * 40
    payload = {
        "schema_version": 1,
        "kind": "blind_evaluation_public_attestation",
        "status": "consumed",
        "gate_passed": passed,
        "gate_reason": "passed" if passed else "one_or_more_quality_gates_failed",
        "engine": "rule",
        "source_tree_sha256": tree,
        "candidate_seal_sha256": "a" * 64,
        "freeze_manifest_sha256": "b" * 64,
        "holdout_sha256": "c" * 64,
        "evaluator_sha256": "d" * 64,
        "overall": {
            "n": 120,
            "n_correct": overall_correct,
            "n_answered": 120,
            "accuracy_num": overall_correct,
            "accuracy_den": 120,
            "coverage_num": 120,
            "coverage_den": 120,
            "strict_result_contract": True,
        },
        "hard": {
            "n": 40,
            "n_correct": hard_correct,
            "n_answered": 40,
            "accuracy_num": hard_correct,
            "accuracy_den": 40,
            "coverage_num": 40,
            "coverage_den": 40,
            "derived_from_overall_single_inference": True,
        },
        "gate": {"passed": passed},
        "public_attestation_rules": {
            "strict_scoring": True,
            "no_question_or_gold_sql_leakage": True,
            "single_inference_pass": True,
            "selection_used": False,
        },
    }
    return _write_json(root / "evidence" / "nl2sql-public.json", payload)


def _device_snapshot(root: Path) -> Path:
    payload = {
        "schema_version": 2,
        "kind": "dki_demo_device_snapshot",
        "created_at": "2026-08-07T12:00:00Z",
        "requested_logical_device": "npu:0",
        "detected_available": True,
        "detected_device_name": "Ascend910_9382",
        "environment_label": "910C 验收环境",
        "detection_sources": ["torch_npu", "npu-smi"],
        "streamlit_runtime": {"version": "1.54.0"},
        "commands": {
            "npu_smi": {
                "returncode": 0,
                "stdout_sha256": "e" * 64,
            },
            "python_runtime": {
                "python": "3.11.6",
                "torch": "2.7.1",
                "torch_npu": "2.7.1.post6",
            },
        },
        "privacy": {
            "contains_environment": False,
            "contains_secrets": False,
        },
    }
    return _write_json(root / "evidence" / "device.json", payload)


def _test_receipt(root: Path, tree: str, *, stage: str) -> Path:
    artifact_dir = root / "evidence" / "tests" / stage
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_files = {}
    for name, content in {
        "stdout": "3 passed\n",
        "stderr": "",
        "junit": "<testsuite tests='3' failures='0' errors='0' />\n",
        "node_ids": "tests/test_a.py::test_a\ntests/test_b.py::test_b\ntests/test_c.py::test_c\n",
        "skip_reasons": "[]\n",
    }.items():
        target = artifact_dir / f"{name}.txt"
        target.write_text(content, encoding="utf-8")
        artifact_files[name] = _binding(root, target)
    payload = {
        "schema_version": 1,
        "kind": "dki_full_test_receipt",
        "stage": stage,
        "source_tree_sha256": tree,
        "status": "passed",
        "returncode": 0,
        "non_agent_returncode": 0,
        "agent_returncode": 0,
        "node_ids_complete": True,
        "missing_baseline_nodes": 0,
        "new_skips": 0,
        "counts": {
            "collected": 3,
            "passed": 3,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "xpassed": 0,
        },
        "artifacts": artifact_files,
    }
    return _write_json(root / "evidence" / f"{stage}-tests.json", payload)


def _package_receipt(root: Path, tree: str, clean_receipt: Path) -> Path:
    source = root / "src" / "source.py"
    member = _rel(root, source)
    external_manifest = root / "evidence" / "package-manifest.json"
    manifest_payload = {
        "schema_version": 1,
        "source_tree_sha256": tree,
        "n_files": 1,
        "files": [
            {
                "path": member,
                "bytes": source.stat().st_size,
                "sha256": _sha256(source),
            }
        ],
    }
    _write_json(external_manifest, manifest_payload)
    archive = root / "dist" / "submission.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(member, source.read_bytes())
        bundle.writestr("MANIFEST.json", external_manifest.read_bytes())
    payload = {
        "schema_version": 1,
        "kind": "clean_room_package_verification",
        "source_tree_sha256": tree,
        "status": "passed",
        "archive": _binding(root, archive),
        "manifest": _binding(root, external_manifest),
        "manifest_checks": {
            "external_matches_internal": True,
            "duplicate_members": 0,
            "missing_members": 0,
            "extra_members": 0,
            "path_traversal_members": 0,
            "absolute_members": 0,
            "symlink_members": 0,
        },
        "privacy": {
            "passed": True,
            "secret_hits": 0,
            "private_data_hits": 0,
            "forbidden_artifact_hits": 0,
        },
        "clean_room": {
            "passed": True,
            "test_receipt_sha256": _sha256(clean_receipt),
        },
        "internal_manifest_path": "MANIFEST.json",
    }
    return _write_json(root / "evidence" / "package-verification.json", payload)


def _inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    overall_correct: int = 103,
    hard_correct: int = 28,
    official_f1: float = 0.51,
) -> tuple[object, dict[str, Path | str]]:
    root = tmp_path / "repo"
    root.mkdir(parents=True)
    included, tree = _included_tree(root)
    cmeie = _cmeie_summary(root, tree, official_f1=official_f1)
    nl2sql = _nl2sql_attestation(
        root,
        tree,
        overall_correct=overall_correct,
        hard_correct=hard_correct,
    )
    device = _device_snapshot(root)
    source_tests = _test_receipt(root, tree, stage="sealed_source")
    clean_tests = _test_receipt(root, tree, stage="clean_room")
    package = _package_receipt(root, tree, clean_tests)
    npu = root / "docs" / "remote_npu_acceptance.md"
    npu.parent.mkdir(parents=True)
    npu.write_text("immutable historical NPU acceptance\n", encoding="utf-8")

    demo_run = root / "outputs" / "demo_runs" / "server-rid" / "runs" / "run-1"
    demo_run.mkdir(parents=True)
    _write_json(demo_run / "completed.json", {"status": "completed"})
    demo_receipt = {
        "status": "completed",
        "returncode": 0,
        "_service_verified": True,
        "mode": "agent-offline",
        "source_tree_sha256": tree,
        "run_id": "run-1",
        "datamate_mode": "real-datamate",
        "artifacts": {f"artifact-{index}": {} for index in range(7)},
    }
    monkeypatch.setattr(
        report_builder.DemoRunService,
        "read_receipt",
        staticmethod(lambda _directory: dict(demo_receipt)),
    )
    output = root / "reports" / "final"
    inputs = report_builder.BuildInputs(
        repo_root=root,
        output_dir=output,
        included_tree_manifest=included,
        demo_run_dir=demo_run,
        cmeie_summary=cmeie,
        nl2sql_attestation=nl2sql,
        npu_acceptance=npu,
        device_snapshot=device,
        source_test_receipt=source_tests,
        cleanroom_test_receipt=clean_tests,
        package_receipt=package,
        expected_npu_sha256=_sha256(npu),
    )
    return inputs, {
        "root": root,
        "tree": tree,
        "output": output,
        "source": root / "src" / "source.py",
        "cmeie": cmeie,
        "nl2sql": nl2sql,
        "archive": root / "dist" / "submission.zip",
        "package": package,
    }


def test_ready_report_is_atomic_aggregate_only_and_covers_13_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, paths = _inputs(tmp_path, monkeypatch)

    result = report_builder.build_final_report(inputs)

    output = Path(paths["output"])
    assert result["status"] == "READY_FOR_SUBMISSION"
    assert result["ready_for_submission"] is True
    assert set(path.name for path in output.iterdir()) == set(report_builder.OUTPUT_FILENAMES)
    final = json.loads((output / "FINAL_STATUS.json").read_text(encoding="utf-8"))
    assert final["ready_for_submission"] is True
    assert final["quality_gates_passed"] is True
    assert all(final["gates"].values())
    markdown = (output / "逐项验收报告.md").read_text(encoding="utf-8")
    assert sum(markdown.count(f"| 任务{label} |") for label in "一二三") == 13
    assert "outputs/private" not in markdown
    assert "gold_sql" not in markdown.casefold()
    sums = (output / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    assert len(sums) == 5
    for row in sums:
        digest, name = row.split("  ", 1)
        assert _sha256(output / name) == digest


@pytest.mark.parametrize(
    ("correct", "expected_ready"),
    [(102, False), (103, True)],
)
def test_nl2sql_overall_gate_is_strictly_above_85_percent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    correct: int,
    expected_ready: bool,
) -> None:
    inputs, paths = _inputs(
        tmp_path,
        monkeypatch,
        overall_correct=correct,
        hard_correct=28,
    )

    report_builder.build_final_report(inputs)

    output = Path(paths["output"])
    attestation = json.loads(
        (output / "NL2SQL_PUBLIC_ATTESTATION.json").read_text(encoding="utf-8")
    )
    final = json.loads((output / "FINAL_STATUS.json").read_text(encoding="utf-8"))
    assert attestation["gate_passed"] is expected_ready
    assert final["ready_for_submission"] is expected_ready


@pytest.mark.parametrize(
    ("correct", "expected_ready"),
    [(27, False), (28, True)],
)
def test_nl2sql_hard_gate_accepts_exactly_70_percent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    correct: int,
    expected_ready: bool,
) -> None:
    inputs, paths = _inputs(
        tmp_path,
        monkeypatch,
        overall_correct=103,
        hard_correct=correct,
    )

    report_builder.build_final_report(inputs)

    output = Path(paths["output"])
    attestation = json.loads(
        (output / "NL2SQL_PUBLIC_ATTESTATION.json").read_text(encoding="utf-8")
    )
    assert attestation["gate_passed"] is expected_ready


@pytest.mark.parametrize(
    ("official_f1", "expected_ready"),
    [(0.499, False), (0.50, True)],
)
def test_cmeie_official_dev_gate_is_exactly_point_five(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    official_f1: float,
    expected_ready: bool,
) -> None:
    inputs, paths = _inputs(tmp_path, monkeypatch, official_f1=official_f1)

    report_builder.build_final_report(inputs)

    output = Path(paths["output"])
    summary = json.loads(
        (output / "CMEIE_PUBLIC_SUMMARY.json").read_text(encoding="utf-8")
    )
    final = json.loads((output / "FINAL_STATUS.json").read_text(encoding="utf-8"))
    assert summary["quality_gate_passed"] is expected_ready
    assert final["ready_for_submission"] is expected_ready


def test_source_or_source_binding_tamper_fails_closed_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, paths = _inputs(tmp_path, monkeypatch)
    Path(paths["source"]).write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(report_builder.FinalReportError, match="included-tree source differs"):
        report_builder.build_final_report(inputs)

    assert not Path(paths["output"]).exists()


def test_mismatched_authority_duplicate_json_and_nan_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, paths = _inputs(tmp_path, monkeypatch)
    nl2sql = Path(paths["nl2sql"])
    payload = json.loads(nl2sql.read_text(encoding="utf-8"))
    payload["source_tree_sha256"] = "f" * 64
    _write_json(nl2sql, payload)
    with pytest.raises(report_builder.FinalReportError, match="source binding"):
        report_builder.build_final_report(inputs)

    duplicate = b'{"schema_version":1,"schema_version":1}'
    with pytest.raises(report_builder.FinalReportError, match="duplicate JSON key"):
        report_builder._load_json_bytes(duplicate, label="duplicate")
    with pytest.raises(report_builder.FinalReportError, match="non-finite"):
        report_builder._load_json_bytes(b'{"value":NaN}', label="nan")


def test_private_row_level_content_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, paths = _inputs(tmp_path, monkeypatch)
    cmeie = Path(paths["cmeie"])
    payload = json.loads(cmeie.read_text(encoding="utf-8"))
    payload["gold_sql"] = "sensitive row-level value"
    _write_json(cmeie, payload)

    with pytest.raises(report_builder.FinalReportError, match="forbidden row-level key"):
        report_builder.build_final_report(inputs)

    assert not Path(paths["output"]).exists()


def test_package_byte_tamper_and_zip_symlink_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, paths = _inputs(tmp_path, monkeypatch)
    archive = Path(paths["archive"])
    archive.write_bytes(archive.read_bytes() + b"tamper")
    with pytest.raises(report_builder.FinalReportError, match="binding differs"):
        report_builder.build_final_report(inputs)

    inputs, paths = _inputs(tmp_path / "symlink-case", monkeypatch)
    archive = Path(paths["archive"])
    package = Path(paths["package"])
    source = Path(paths["source"])
    external = Path(inputs.package_receipt).parent / "package-manifest.json"
    info = zipfile.ZipInfo("src/source.py")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(info, b"src/target.py")
        bundle.writestr("MANIFEST.json", external.read_bytes())
    receipt = json.loads(package.read_text(encoding="utf-8"))
    receipt["archive"] = _binding(Path(paths["root"]), archive)
    _write_json(package, receipt)
    with pytest.raises(report_builder.FinalReportError, match="symlink"):
        report_builder.build_final_report(inputs)


def test_create_once_preserves_first_publication_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, paths = _inputs(tmp_path, monkeypatch)
    report_builder.build_final_report(inputs)
    output = Path(paths["output"])
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    with pytest.raises(FileExistsError, match="already exists"):
        report_builder.build_final_report(inputs)

    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
