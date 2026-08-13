"""Local-only checks for the credential-free NPU preparation workflow."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import shutil
import sys
import tarfile
import threading
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    script = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUNDLE = _load("prepare_remote_npu_bundle", "scripts/prepare_remote_npu_bundle.py")
PREFLIGHT = _load("remote_npu_preflight", "scripts/remote_npu_preflight.py")
RESCORE = _load("eval_ner_cmeee_checkpoint", "scripts/eval_ner_cmeee_checkpoint.py")
EVIDENCE = _load("build_evidence_claims", "scripts/build_evidence_claims.py")
LEGACY_LAUNCH = _load("launch_remote_theoretical_limit", "scripts/launch_remote_theoretical_limit.py")


def test_legacy_remote_launcher_fails_closed_without_connecting(capsys):
    assert LEGACY_LAUNCH.main() == 2
    assert "Refusing obsolete automatic remote launch" in capsys.readouterr().err


def test_source_bundle_is_credential_free_and_contains_runbook(tmp_path):
    result = BUNDLE.build(tmp_path / "bundle", ROOT, include_cblue=False, overwrite=False)
    archive = Path(result["source_archive"])
    assert archive.is_file()
    assert len(result["source_archive_sha256"]) == 64
    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
        assert "dki-agents/REMOTE_NPU_RUNBOOK.json" in names
        assert "dki-agents/REMOTE_NPU_SOURCE_MANIFEST.json" in names
        assert "dki-agents/scripts/download_cblue.py" in names
        assert "dki-agents/tests/test_cblue_normalization.py" in names
        assert all("/.env" not in name for name in names)
        assert all("/data/" not in name for name in names)
        assert all("/models/" not in name for name in names)
        runbook = json.load(tar.extractfile("dki-agents/REMOTE_NPU_RUNBOOK.json"))
    assert runbook["source_tree_sha256"] == result["source_tree_sha256"]
    assert [step["id"] for step in runbook["ordered_steps"]] == [
        "host_preflight",
        "cmeee_onnx_export",
        "cmeee_atc_compile",
        "inspect_real_om_io",
        "bind_checkpoint_to_om",
        "bound_artifact_preflight",
        "cmeee_frozen_rescore",
        "cmeie_fit_only_seed42",
        "cmeie_fit_only_seed43",
        "om_cpu_npu_energy_acceptance",
    ]
    preflight = next(
        step["command"]
        for step in runbook["ordered_steps"]
        if step["id"] == "bound_artifact_preflight"
    )
    for flag in ("--cmeee-model", "--om-manifest", "--om-path", "--tokenizer-path", "--labels-path"):
        assert flag in preflight
    assert "--required-path" not in preflight
    assert "OM_TOKENIZER_PATH" in runbook["required_variables"]
    assert "OM_LABELS_PATH" in runbook["required_variables"]


def test_cblue_archive_inputs_are_only_normalized_benchmarks():
    assert list(BUNDLE.CBLUE_FILES) == [
        "data/cblue/CMeEE-V2/train.jsonl",
        "data/cblue/CMeEE-V2/dev.jsonl",
        "data/cblue/CMeIE-V2/train.jsonl",
        "data/cblue/CMeIE-V2/dev.jsonl",
    ]


def test_preflight_reports_missing_normalized_data_without_npu_import(tmp_path, monkeypatch):
    fake_root = tmp_path / "repo"
    fake_root.mkdir()
    monkeypatch.setattr(PREFLIGHT, "EXPECTED_CBLUE_ROWS", {"CMeEE-V2/train.jsonl": 1})
    report, passed = PREFLIGHT._cblue_report(fake_root)
    assert passed is False
    assert report["CMeEE-V2/train.jsonl"] == {"exists": False, "expected_rows": 1}


def test_preflight_write_refuses_existing_output_without_mutation(tmp_path):
    output = tmp_path / "preflight.json"
    output.write_text('{"owner": "existing"}', encoding="utf-8")

    with pytest.raises(FileExistsError):
        PREFLIGHT._write(output, {"owner": "new"})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "owner": "existing"
    }


def test_concurrent_preflight_write_has_exactly_one_winner(tmp_path):
    output = tmp_path / "preflight.json"
    barrier = threading.Barrier(2)

    def publish(owner: str) -> tuple[str, str]:
        barrier.wait()
        try:
            PREFLIGHT._write(output, {"owner": owner})
        except FileExistsError:
            return owner, "lost"
        return owner, "won"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, ("one", "two")))

    assert sorted(status for _, status in outcomes) == ["lost", "won"]
    winner = next(owner for owner, status in outcomes if status == "won")
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "owner": winner
    }


def _source_tree_fixture(tmp_path: Path):
    source_root = tmp_path / "source"
    script = source_root / "scripts" / "remote_npu_preflight.py"
    script.parent.mkdir(parents=True)
    shutil.copyfile(Path(PREFLIGHT.__file__), script)
    module = source_root / "dki" / "example.py"
    module.parent.mkdir()
    module.write_text("VALUE = 1\n", encoding="utf-8")

    entries = []
    for relative in ("dki/example.py", "scripts/remote_npu_preflight.py"):
        path = source_root / Path(relative)
        entries.append({
            "path": relative,
            "size": path.stat().st_size,
            "sha256": PREFLIGHT._sha256(path),
        })
    entries.sort(key=lambda entry: entry["path"])
    tree_sha256 = PREFLIGHT._tree_sha256(entries)
    git_head = "a" * 40
    rid = f"20260729T120000Z-g{git_head[:7]}-t{tree_sha256[:12]}"
    manifest = tmp_path / "included_tree.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": PREFLIGHT.SOURCE_TREE_SCHEMA_VERSION,
                "rid": rid,
                "git_head": git_head,
                "algorithm": PREFLIGHT.SOURCE_TREE_ALGORITHM,
                "included_tree_sha256": tree_sha256,
                "entries": entries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return source_root, manifest, tree_sha256


def _base_model_fixture(tmp_path: Path) -> Path:
    model = tmp_path / "base-model"
    (model / "tokenizer").mkdir(parents=True)
    (model / "config.json").write_text('{"model_type":"bert"}\n', encoding="utf-8")
    (model / "tokenizer" / "vocab.txt").write_text(
        "[PAD]\n[UNK]\n", encoding="utf-8"
    )
    return model


def test_cmeie_preflight_binds_source_manifest_script_and_base_model_tree(tmp_path):
    source_root, manifest, source_tree_sha256 = _source_tree_fixture(tmp_path)
    model = _base_model_fixture(tmp_path)

    report, passed = PREFLIGHT._cmeie_binding_report(
        source_root=source_root,
        source_tree_sha256=source_tree_sha256,
        source_tree_manifest=manifest,
        cmeie_base_model=model,
        strict=True,
    )

    assert passed is True
    assert report["complete_arguments"] is True
    assert report["valid"] is True
    source = report["source_tree"]
    assert source["source_tree_sha256"] == source_tree_sha256
    assert source["manifest_sha256"] == PREFLIGHT._sha256(manifest)
    assert source["preflight_script"]["sha256"] == PREFLIGHT._sha256(
        Path(PREFLIGHT.__file__)
    )
    base_model = report["cmeie_base_model"]
    assert [entry["path"] for entry in base_model["files"]] == [
        "config.json",
        "tokenizer/vocab.txt",
    ]
    assert base_model["tree_sha256"] == PREFLIGHT._tree_sha256(
        base_model["files"]
    )


def test_cmeie_preflight_rejects_source_manifest_and_file_hash_tampering(tmp_path):
    source_root, manifest, source_tree_sha256 = _source_tree_fixture(tmp_path)
    model = _base_model_fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["entries"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    report, passed = PREFLIGHT._cmeie_binding_report(
        source_root=source_root,
        source_tree_sha256=source_tree_sha256,
        source_tree_manifest=manifest,
        cmeie_base_model=model,
        strict=True,
    )
    assert passed is False
    assert report["source_tree"]["valid"] is False
    assert "manifest hash" in report["source_tree"]["message"]

    source_root, manifest, source_tree_sha256 = _source_tree_fixture(
        tmp_path / "second"
    )
    (source_root / "dki" / "example.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )
    report, passed = PREFLIGHT._cmeie_binding_report(
        source_root=source_root,
        source_tree_sha256=source_tree_sha256,
        source_tree_manifest=manifest,
        cmeie_base_model=model,
        strict=True,
    )
    assert passed is False
    assert report["source_tree"]["valid"] is False
    assert "current source file" in report["source_tree"]["message"]


def test_cmeie_base_model_report_rejects_tree_tampering_between_passes(
    tmp_path, monkeypatch
):
    model = _base_model_fixture(tmp_path)
    calls = []

    def reject_changed_tree(root, *, label):
        calls.append((Path(root), label))
        raise ValueError(
            "CMeIE base-model tree changed while it was inventoried and hashed"
        )

    monkeypatch.setattr(PREFLIGHT, "scan_bound_regular_tree", reject_changed_tree)
    report, passed = PREFLIGHT._cmeie_base_model_report(model)
    assert passed is False
    assert report["valid"] is False
    assert "changed while it was inventoried and hashed" in report["message"]
    assert calls == [(model, "CMeIE base-model")]


def test_strict_cmeie_preflight_requires_all_binding_arguments(tmp_path):
    report, passed = PREFLIGHT._cmeie_binding_report(
        source_root=tmp_path,
        source_tree_sha256=None,
        source_tree_manifest=None,
        cmeie_base_model=None,
        strict=True,
    )
    assert passed is False
    assert report == {
        "complete_arguments": False,
        "valid": False,
        "missing_arguments": [
            "--source-tree-sha256",
            "--source-tree-manifest",
            "--cmeie-base-model",
        ],
    }


def test_cmeie_binding_keeps_non_strict_legacy_preflight_optional(tmp_path):
    report, passed = PREFLIGHT._cmeie_binding_report(
        source_root=tmp_path,
        source_tree_sha256=None,
        source_tree_manifest=None,
        cmeie_base_model=None,
        strict=False,
    )
    assert report is None
    assert passed is True


def test_main_keeps_cblue_data_root_separate_from_executing_source_root(
    tmp_path, monkeypatch
):
    data_root = tmp_path / "cblue-checkout"
    source_root = tmp_path / "source-checkout"
    data_root.mkdir()
    source_root.mkdir()
    output = tmp_path / "preflight.json"
    captured = {}

    monkeypatch.setattr(PREFLIGHT, "ROOT", source_root)
    monkeypatch.setattr(PREFLIGHT, "_cblue_report", lambda root: ({}, True))
    monkeypatch.setattr(PREFLIGHT, "_required_paths", lambda paths: ([], True))
    monkeypatch.setattr(
        PREFLIGHT, "_cmeee_checkpoint_report", lambda model: (None, True)
    )
    monkeypatch.setattr(
        PREFLIGHT,
        "_om_manifest_report",
        lambda *args: (None, True),
    )
    monkeypatch.setattr(PREFLIGHT, "_runtime_report", lambda: ({}, True))

    def capture_binding(**kwargs):
        captured.update(kwargs)
        return {"valid": True}, True

    monkeypatch.setattr(PREFLIGHT, "_cmeie_binding_report", capture_binding)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remote_npu_preflight.py",
            "--root",
            str(data_root),
            "--output",
            str(output),
        ],
    )

    assert PREFLIGHT.main() == 0
    assert captured["source_root"] == source_root
    assert captured["source_root"] != data_root


def test_runtime_report_isolates_a_failed_torch_npu_child_probe(monkeypatch):
    monkeypatch.setattr(PREFLIGHT, "_package_version", lambda name: {"available": True, "version": "x"})
    monkeypatch.setattr(PREFLIGHT, "_torch_npu_probe", lambda: {
        "npu_available": False,
        "probe": {"returncode": 3, "stderr": "extension conflict"},
    })
    monkeypatch.setattr(PREFLIGHT, "_run", lambda command: {"command": command, "returncode": 0})
    report, passed = PREFLIGHT._runtime_report()
    assert passed is False
    assert report["torch_npu_available"] is False
    assert report["torch_npu_probe"]["probe"]["returncode"] == 3


def test_checkpoint_manifest_hashes_all_regular_files(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    nested = model / "tokenizer"
    nested.mkdir()
    (nested / "tokenizer.json").write_text("{}", encoding="utf-8")
    manifest = RESCORE._model_manifest(model)
    assert [entry["path"] for entry in manifest] == ["config.json", "tokenizer/tokenizer.json"]
    assert all(len(entry["sha256"]) == 64 for entry in manifest)


def _complete_cmeee_checkpoint(model: Path) -> None:
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "globalpointer.pt").write_bytes(b"frozen-head")
    (model / "globalpointer_config.json").write_text(
        json.dumps({"max_len": 256, "inner_dim": 64, "selected_threshold": -1.0}),
        encoding="utf-8",
    )
    (model / "types.json").write_text(
        json.dumps(list(PREFLIGHT.EXPECTED_CMEEE_TYPES)), encoding="utf-8",
    )


def test_preflight_accepts_complete_cmeee_checkpoint_without_model_load(tmp_path):
    model = tmp_path / "model"
    _complete_cmeee_checkpoint(model)
    report, passed = PREFLIGHT._cmeee_checkpoint_report(model)
    assert passed is True
    assert report["complete"] is True
    assert {entry["path"] for entry in report["files"]} == {
        str(model / "config.json"),
        str(model / "globalpointer.pt"),
        str(model / "globalpointer_config.json"),
        str(model / "types.json"),
    }
    assert all(len(entry["sha256"]) == 64 for entry in report["files"])


def test_preflight_rejects_incomplete_or_wrong_type_cmeee_checkpoint(tmp_path):
    model = tmp_path / "model"
    _complete_cmeee_checkpoint(model)
    (model / "types.json").write_text(json.dumps(["wrong"]), encoding="utf-8")
    report, passed = PREFLIGHT._cmeee_checkpoint_report(model)
    assert passed is False
    assert report["complete"] is False

    (model / "globalpointer.pt").unlink()
    report, passed = PREFLIGHT._cmeee_checkpoint_report(model)
    assert passed is False
    assert report["complete"] is False


def test_preflight_requires_complete_om_manifest_arguments(tmp_path):
    report, passed = PREFLIGHT._om_manifest_report(tmp_path / "manifest.json", None, None, None)
    assert passed is False
    assert report == {
        "complete_arguments": False,
        "error": "manifest, OM, tokenizer, and labels paths must be supplied together",
    }


def test_preflight_hash_verifies_complete_om_manifest_without_om_load(tmp_path):
    om = tmp_path / "model.om"
    tokenizer = tmp_path / "tokenizer.json"
    labels = tmp_path / "labels.json"
    om.write_bytes(b"om")
    tokenizer.write_text("tokenizer", encoding="utf-8")
    labels.write_text("labels", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "model_id": "medical_ner_bert",
        "om_sha256": PREFLIGHT._sha256(om),
        "tokenizer_sha256": PREFLIGHT._sha256(tokenizer),
        "labels_sha256": PREFLIGHT._sha256(labels),
        "soc_version": "Ascend910B2C",
        "cann_version": "8.5.1",
        "inputs": [{"name": "input_ids", "dtype": "int64", "shape": [1, 16]}],
        "outputs": [{"name": "logits", "dtype": "float32", "shape": [1, 16, 9]}],
    }), encoding="utf-8")
    report, passed = PREFLIGHT._om_manifest_report(
        manifest_path, om, tokenizer, labels,
    )
    assert passed is True
    assert report["valid"] is True
    assert report["manifest_sha256"] == PREFLIGHT._sha256(manifest_path)
    assert [item["role"] for item in report["artifacts"]] == ["om", "tokenizer", "labels"]


def test_cmeie_claim_requires_explicit_gold_type_signature_scope():
    artifact = {
        "schema_version": 2,
        "dataset_role": "development_locked_report",
        "n_dev": 3585,
        "evaluation_scope": "gold_entity_pairs_and_gold_type_signatures_not_end_to_end_spo",
        "gold_type_signature_conditioning": True,
        "selection_protocol": {"official_dev_used_for_selection": False},
        "schema_constraint_scope": "fit_partition_only",
        "train_config": {"objective": "multi_label_bce"},
        "micro_f1": 0.6,
        "precision": 0.6,
        "recall": 0.6,
        "pred_na": 0,
        "pred_rel": 10,
        "selected_threshold": 0.5,
    }
    claim = EVIDENCE._extract_cmeie_claim(artifact)
    assert "gold 主客体类型签名" in claim["notes"]
    artifact["gold_type_signature_conditioning"] = False
    try:
        EVIDENCE._extract_cmeie_claim(artifact)
    except ValueError as exc:
        assert "type-signature" in str(exc)
    else:
        raise AssertionError("CMeIE claim gate accepted unspecified type conditioning")
