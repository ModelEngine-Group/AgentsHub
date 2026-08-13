from __future__ import annotations

import hashlib
import json
import shutil
import stat
import zipfile
from pathlib import Path

import pytest

from scripts.build_submission_bundle import (
    OPERATOR_ARCHIVE_NAMES,
    PUBLIC_AGGREGATE_CBLUE_RECEIPTS,
    REQUIRED_PUBLIC_EVIDENCE,
    _is_hard_excluded,
    _package_tree_sha,
    build,
)
from scripts.verify_submission_bundle import (
    AGENT_TEST_PATHS,
    BundleVerificationError,
    CLEANROOM_RUNTIME_ARTIFACTS,
    CLEANROOM_RUNTIME_COMMAND,
    PROMOTED_CLAIM_RECEIPTS,
    cleanroom_runtime_command,
    non_agent_pytest_args,
    validate_archive,
)


RID = "20260802T120000Z-gabcdef0-t123456789abc"
HEAD = "a" * 40


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_manifest(root: Path, source_files: dict[str, bytes]) -> Path:
    entries = [
        {"path": path, "size": len(data), "sha256": _sha(data)}
        for path, data in sorted(source_files.items())
    ]
    payload = {
        "schema_version": 1,
        "rid": RID,
        "git_head": HEAD,
        "algorithm": "sha256 of sorted path NUL size NUL file_sha256 LF",
        "included_tree_sha256": _package_tree_sha(entries),
        "entries": entries,
    }
    path = root / "outputs" / "remote_runs" / RID / "included_tree_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _operator_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "rid-operators"
    directory.mkdir(parents=True)
    for name in OPERATOR_ARCHIVE_NAMES:
        with zipfile.ZipFile(directory / name, "x", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("metadata.yml", f"name: {name}\n")
            archive.writestr("__init__.py", "# standalone operator\n")
            archive.writestr("process.py", "VALUE = 1\n")
    return directory


def _release_tree(tmp_path: Path) -> tuple[Path, Path, dict[str, Path], Path]:
    root = tmp_path / "repo"
    source_files = {
        "README.md": b"# synthetic release\n",
        "dki/hello.py": b"VALUE = 'clean-room'\n",
        "outputs/cblue/cmeee_metrics.json": b'{"aggregate_only": true}\n',
        "outputs/cblue/cmeie_metrics.json": b'{"aggregate_only": true}\n',
    }
    for rel, data in source_files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    included_tree = _source_manifest(root, source_files)
    evidence: dict[str, Path] = {}
    for index, kind in enumerate(REQUIRED_PUBLIC_EVIDENCE, start=1):
        path = root / "outputs" / "remote_runs" / RID / "nl2sql" / f"{kind}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "aggregate_only",
                    "gate_passed": True,
                    "correct": index,
                    "sha256": "b" * 64,
                }
            ),
            encoding="utf-8",
        )
        evidence[kind] = path
    return root, included_tree, evidence, _operator_dir(tmp_path)


def _build_valid(tmp_path: Path) -> tuple[Path, Path]:
    root, included_tree, evidence, operators = _release_tree(tmp_path)
    output = root / "dist" / "submission" / RID / "candidate.zip"
    build(
        output,
        root,
        included_tree_manifest=included_tree,
        public_evidence=evidence,
        operator_dir=operators,
        git_head=HEAD,
    )
    return output, output.with_suffix(".manifest.json")


def _copy_manifest(source: Path, target_bundle: Path) -> Path:
    target_manifest = target_bundle.with_suffix(".manifest.json")
    shutil.copyfile(source.with_suffix(".manifest.json"), target_manifest)
    return target_manifest


def test_builds_bound_archive_and_refuses_second_publication(tmp_path: Path) -> None:
    bundle, manifest_path = _build_valid(tmp_path)
    snapshot = validate_archive(bundle, external_manifest=manifest_path)
    assert snapshot["secret_scan_hits"] == []
    assert snapshot["manifest"]["n_files"] == len(snapshot["payloads"])
    with zipfile.ZipFile(bundle) as archive:
        assert archive.read("dki-agents/SUBMISSION_MANIFEST.json") == manifest_path.read_bytes()
        for rel in PUBLIC_AGGREGATE_CBLUE_RECEIPTS:
            assert archive.read(f"dki-agents/{rel}") == b'{"aggregate_only": true}\n'

    root = bundle.parents[3]
    included_tree = root / "outputs" / "remote_runs" / RID / "included_tree_manifest.json"
    evidence = {
        kind: root / "outputs" / "remote_runs" / RID / "nl2sql" / f"{kind}.json"
        for kind in REQUIRED_PUBLIC_EVIDENCE
    }
    with pytest.raises(FileExistsError):
        build(
            bundle,
            root,
            included_tree_manifest=included_tree,
            public_evidence=evidence,
            operator_dir=root.parent / "rid-operators",
            git_head=HEAD,
        )


def test_only_explicit_aggregate_cblue_receipts_bypass_the_raw_data_exclusion() -> None:
    assert all(not _is_hard_excluded(rel) for rel in PUBLIC_AGGREGATE_CBLUE_RECEIPTS)
    assert _is_hard_excluded("outputs/cblue/train.jsonl")
    assert _is_hard_excluded("outputs/cblue/checkpoints/model.safetensors")


@pytest.mark.parametrize("kind", ["duplicate", "traversal", "symlink", "unrecorded"])
def test_verifier_rejects_unsafe_or_unrecorded_members(tmp_path: Path, kind: str) -> None:
    bundle, _ = _build_valid(tmp_path)
    corrupt = bundle.with_name(f"{kind}.zip")
    shutil.copyfile(bundle, corrupt)
    external = _copy_manifest(bundle, corrupt)
    with zipfile.ZipFile(corrupt, "a") as archive:
        if kind == "duplicate":
            archive.writestr("dki-agents/README.md", b"duplicate")
        elif kind == "traversal":
            archive.writestr("dki-agents/../escaped.txt", b"no")
        elif kind == "symlink":
            info = zipfile.ZipInfo("dki-agents/link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, b"target")
        else:
            archive.writestr("dki-agents/unrecorded.txt", b"no")
    with pytest.raises(BundleVerificationError):
        validate_archive(corrupt, external_manifest=external)


def test_verifier_rejects_external_manifest_drift(tmp_path: Path) -> None:
    bundle, manifest = _build_valid(tmp_path)
    manifest.write_bytes(b"{}\n")
    with pytest.raises(BundleVerificationError, match="external and internal manifest bytes differ"):
        validate_archive(bundle, external_manifest=manifest)


def test_builder_rejects_private_or_row_level_public_evidence(tmp_path: Path) -> None:
    root, included_tree, evidence, operators = _release_tree(tmp_path)
    private = root / "outputs" / "private" / RID / "holdout.jsonl"
    private.parent.mkdir(parents=True, exist_ok=True)
    private.write_text('{"opaque": true}\n', encoding="utf-8")
    private_evidence = dict(evidence)
    private_evidence["candidate_seal"] = private
    with pytest.raises(ValueError, match="public remote-run evidence"):
        build(
            root / "dist" / "submission" / RID / "private.zip",
            root,
            included_tree_manifest=included_tree,
            public_evidence=private_evidence,
            operator_dir=operators,
            git_head=HEAD,
        )

    evidence["blind_attestation"].write_text(
        json.dumps({"status": "bad", "question": "private item"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="private blind field"):
        build(
            root / "dist" / "submission" / RID / "rows.zip",
            root,
            included_tree_manifest=included_tree,
            public_evidence=evidence,
            operator_dir=operators,
            git_head=HEAD,
        )


def test_builder_rejects_source_tree_that_no_longer_matches_payload(tmp_path: Path) -> None:
    root, included_tree, evidence, operators = _release_tree(tmp_path)
    (root / "README.md").write_text("changed after source binding\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not exactly bind"):
        build(
            root / "dist" / "submission" / RID / "changed.zip",
            root,
            included_tree_manifest=included_tree,
            public_evidence=evidence,
            operator_dir=operators,
            git_head=HEAD,
        )


def test_release_uses_only_explicit_fresh_operator_archives(tmp_path: Path) -> None:
    root, included_tree, evidence, operators = _release_tree(tmp_path)
    stale = root / "dist" / "operators" / OPERATOR_ARCHIVE_NAMES[0]
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale historical archive")
    output = root / "dist" / "submission" / RID / "fresh.zip"
    build(
        output,
        root,
        included_tree_manifest=included_tree,
        public_evidence=evidence,
        operator_dir=operators,
        git_head=HEAD,
    )
    snapshot = validate_archive(output, external_manifest=output.with_suffix(".manifest.json"))
    binding = snapshot["manifest"]["required_operator_archives"]
    assert set(binding) == set(OPERATOR_ARCHIVE_NAMES)
    with zipfile.ZipFile(output) as archive:
        embedded = archive.read(f"dki-agents/dist/operators/{OPERATOR_ARCHIVE_NAMES[0]}")
    assert embedded == (operators / OPERATOR_ARCHIVE_NAMES[0]).read_bytes()
    assert embedded != stale.read_bytes()


def test_builder_rejects_unsafe_operator_archive(tmp_path: Path) -> None:
    root, included_tree, evidence, operators = _release_tree(tmp_path)
    target = operators / OPERATOR_ARCHIVE_NAMES[0]
    target.unlink()
    with zipfile.ZipFile(target, "x") as archive:
        archive.writestr("metadata.yml", "name: unsafe\n")
        archive.writestr("__init__.py", "")
        archive.writestr("process.py", "")
        archive.writestr("../escaped.txt", "no")
    with pytest.raises(ValueError, match="unsafe member"):
        build(
            root / "dist" / "submission" / RID / "unsafe.zip",
            root,
            included_tree_manifest=included_tree,
            public_evidence=evidence,
            operator_dir=operators,
            git_head=HEAD,
        )


def test_clean_room_verifier_separates_all_agent_dependency_tests() -> None:
    args = non_agent_pytest_args()
    assert args[0] == "tests"
    assert set(AGENT_TEST_PATHS) == {
        "tests/test_agents.py",
        "tests/test_insight_agent_nl2sql_e2e.py",
    }
    assert {argument.removeprefix("--ignore=") for argument in args[1:]} == set(
        AGENT_TEST_PATHS
    )


def test_clean_room_verifier_has_fixed_runtime_lineage_setup() -> None:
    python = Path("/opt/venv/bin/python")
    assert CLEANROOM_RUNTIME_COMMAND == ("-m", "dki", "demo-all")
    assert cleanroom_runtime_command(python) == [str(python), "-m", "dki", "demo-all"]
    assert set(CLEANROOM_RUNTIME_ARTIFACTS) >= {
        "outputs/processed/manifest.json",
        "outputs/kg/medical_kg.json",
        "outputs/kg/medical_kg.provenance.json",
        "outputs/insight.db",
        "outputs/insight.provenance.json",
        "outputs/runtime_eval/kg_extraction_eval.json",
        "outputs/runtime_eval/nl2sql_eval.json",
    }
    assert set(PROMOTED_CLAIM_RECEIPTS) == {
        "outputs/eval/kg_extraction_eval.json",
        "outputs/eval/nl2sql_eval.json",
    }
