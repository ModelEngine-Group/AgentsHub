import hashlib
import json
import os
import shutil
import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from dki.nlp import cmeie_run_protocol as protocol


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _make_run(
    parent: Path,
    *,
    timestamp: str = "20260729T120000Z",
    head: str = "30fce24" + ("a" * 33),
) -> tuple[Path, protocol.RunBinding]:
    source_bytes = b"authoritative source\n"
    entries = [
        {
            "path": "dki/example.py",
            "size": len(source_bytes),
            "sha256": _sha(source_bytes),
        }
    ]
    tree = protocol.included_tree_sha256(entries)
    rid = f"{timestamp}-g{head[:7]}-t{tree[:12]}"
    root = parent / f"dki-agents-{rid}"
    source_root = root / "source" / "dki-agents"
    source_root.mkdir(parents=True)
    (source_root / "dki").mkdir()
    (source_root / "dki" / "example.py").write_bytes(source_bytes)
    manifest_dir = root / "evidence" / "source"
    manifest_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "rid": rid,
        "git_head": head,
        "algorithm": protocol.SOURCE_MANIFEST_ALGORITHM,
        "included_tree_sha256": tree,
        "entries": entries,
    }
    (manifest_dir / "included_tree.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return root, protocol.load_run_binding(root)


def _model_evidence(
    binding: protocol.RunBinding,
    *,
    name: str = "model-a",
) -> dict:
    directory = protocol.canonical_paths(binding).seed_model(42) / name
    directory.mkdir(parents=True)
    checkpoint = directory / "gplinker.pt"
    checkpoint.write_bytes(name.encode("ascii"))
    selection = directory / "selection.json"
    selection.write_text('{"selected":true}\n', encoding="utf-8")
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha(checkpoint.read_bytes()),
        "selection_receipt_path": str(selection),
        "selection_receipt_sha256": _sha(selection.read_bytes()),
        "selection_used": True,
    }


def test_parse_rid_is_strict_and_calendar_valid():
    parts = protocol.parse_rid(
        "20260729T104110Z-g30fce24-tf26bf851cc2b"
    )
    assert parts.timestamp == "20260729T104110Z"
    assert parts.head_prefix == "30fce24"
    assert parts.tree_prefix == "f26bf851cc2b"

    with pytest.raises(TypeError, match="RID must be text"):
        protocol.parse_rid(True)
    with pytest.raises(ValueError, match="lowercase hex"):
        protocol.parse_rid(
            "20260729T104110Z-g30FCE24-tf26bf851cc2b"
        )
    with pytest.raises(ValueError, match="calendar"):
        protocol.parse_rid(
            "20260230T104110Z-g30fce24-tf26bf851cc2b"
        )


def test_run_binding_requires_fixed_manifest_and_source_locations(tmp_path):
    root, binding = _make_run(tmp_path)

    assert binding.run_root == root
    assert binding.source_manifest == (
        root / "evidence" / "source" / "included_tree.json"
    )
    assert binding.source_root == root / "source" / "dki-agents"
    assert binding.git_head.startswith(protocol.parse_rid(binding.rid).head_prefix)
    assert binding.included_tree_sha256.startswith(
        protocol.parse_rid(binding.rid).tree_prefix
    )

    moved = root / "evidence" / "source" / "moved.json"
    binding.source_manifest.rename(moved)
    with pytest.raises(FileNotFoundError, match="source manifest"):
        protocol.load_run_binding(root)


def test_run_binding_recomputes_manifest_tree_and_rejects_rid_tamper(
    tmp_path,
):
    root, binding = _make_run(tmp_path)
    manifest_path = binding.source_manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"][0]["size"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="differs from its entries"):
        protocol.load_run_binding(root)


def test_run_binding_hashes_each_installed_manifest_entry(
    monkeypatch,
    tmp_path,
):
    root, _ = _make_run(tmp_path)
    real_read = protocol.read_bound_bytes
    source_reads = []

    def recording_read(path, label, *args, **kwargs):
        result = real_read(path, label, *args, **kwargs)
        if label.startswith("CMeIE run source file "):
            source_reads.append((Path(path), result[1]))
        return result

    monkeypatch.setattr(protocol, "read_bound_bytes", recording_read)
    binding = protocol.load_run_binding(root)

    assert [path for path, _ in source_reads] == [
        binding.source_root / "dki" / "example.py"
    ]
    assert source_reads[0][1]["sha256"] == _sha(
        b"authoritative source\n"
    )
    assert source_reads[0][1]["nlink"] == 1


def test_run_binding_rejects_missing_installed_source_file(tmp_path):
    root, binding = _make_run(tmp_path)
    (binding.source_root / "dki" / "example.py").unlink()

    with pytest.raises(ValueError, match="missing_files"):
        protocol.load_run_binding(root)


def test_run_binding_rejects_extra_source_file_and_empty_directory(tmp_path):
    root, binding = _make_run(tmp_path)
    (binding.source_root / "unlisted.txt").write_bytes(b"not in manifest")
    (binding.source_root / "empty-extra").mkdir()

    with pytest.raises(
        ValueError,
        match=r"extra_files=.*unlisted\.txt.*extra_directories=.*empty-extra",
    ):
        protocol.load_run_binding(root)


def test_run_binding_rejects_extra_pycache_with_formal_command_guidance(
    tmp_path,
):
    root, binding = _make_run(tmp_path)
    cache = binding.source_root / "dki" / "__pycache__"
    cache.mkdir()
    (cache / "example.cpython-311.pyc").write_bytes(b"bytecode")

    with pytest.raises(
        ValueError,
        match="PYTHONDONTWRITEBYTECODE=1",
    ):
        protocol.load_run_binding(root)


def test_run_binding_rejects_same_size_source_replacement(tmp_path):
    root, binding = _make_run(tmp_path)
    source = binding.source_root / "dki" / "example.py"
    replacement = source.with_name("replacement.tmp")
    replacement.write_bytes(b"x" * source.stat().st_size)
    os.replace(replacement, source)

    with pytest.raises(ValueError, match="SHA-256 differs"):
        protocol.load_run_binding(root)


def test_run_binding_rejects_source_hard_link_outside_tree(tmp_path):
    root, binding = _make_run(tmp_path)
    source = binding.source_root / "dki" / "example.py"
    alias = tmp_path / "outside-source-alias.py"
    os.link(source, alias)

    with pytest.raises(ValueError, match="multiple hard links"):
        protocol.load_run_binding(root)


def test_run_binding_rejects_hard_linked_source_manifest(tmp_path):
    root, binding = _make_run(tmp_path)
    alias = tmp_path / "source-manifest-alias.json"
    os.link(binding.source_manifest, alias)

    with pytest.raises(ValueError, match="multiple hard links"):
        protocol.load_run_binding(root)


def test_run_binding_rejects_source_file_symlink_or_reparse(
    monkeypatch,
    tmp_path,
):
    root, binding = _make_run(tmp_path)
    source = binding.source_root / "dki" / "example.py"
    real_lstat = protocol.os.lstat

    class _LinkStatProxy:
        def __init__(self, original):
            self._original = original
            self.st_mode = (
                stat.S_IFLNK | stat.S_IMODE(original.st_mode)
            )

        def __getattr__(self, name):
            return getattr(self._original, name)

    def marked_lstat(path):
        metadata = real_lstat(path)
        if Path(path) == source:
            return _LinkStatProxy(metadata)
        return metadata

    monkeypatch.setattr(protocol.os, "lstat", marked_lstat)
    with pytest.raises(ValueError, match="symlink or reparse"):
        protocol.load_run_binding(root)


def test_run_binding_rejects_special_source_entry(monkeypatch, tmp_path):
    root, binding = _make_run(tmp_path)
    source = binding.source_root / "dki" / "example.py"
    real_lstat = protocol.os.lstat

    class _SpecialStatProxy:
        def __init__(self, original):
            self._original = original
            self.st_mode = (
                stat.S_IFIFO | stat.S_IMODE(original.st_mode)
            )

        def __getattr__(self, name):
            return getattr(self._original, name)

    def marked_lstat(path):
        metadata = real_lstat(path)
        if Path(path) == source:
            return _SpecialStatProxy(metadata)
        return metadata

    monkeypatch.setattr(protocol.os, "lstat", marked_lstat)
    with pytest.raises(ValueError, match="special or unstable"):
        protocol.load_run_binding(root)


def test_run_binding_rejects_directory_replacement_during_file_reads(
    monkeypatch,
    tmp_path,
):
    root, binding = _make_run(tmp_path)
    source_directory = binding.source_root / "dki"
    real_read = protocol.read_bound_bytes
    replaced = False

    def replace_directory_after_read(path, label, *args, **kwargs):
        nonlocal replaced
        result = real_read(path, label, *args, **kwargs)
        if (
            not replaced
            and label.startswith("CMeIE run source file ")
        ):
            replaced = True
            displaced = binding.source_root / "dki-displaced"
            source_directory.rename(displaced)
            source_directory.mkdir()
            (source_directory / "example.py").write_bytes(result[0])
            shutil.rmtree(displaced)
        return result

    monkeypatch.setattr(
        protocol,
        "read_bound_bytes",
        replace_directory_after_read,
    )
    with pytest.raises(ValueError, match="source tree changed"):
        protocol.load_run_binding(root)
    assert replaced is True


def test_manifest_rejects_duplicate_case_alias_and_file_directory_collision():
    entry = {
        "path": "dki/example.py",
        "size": 1,
        "sha256": _sha(b"x"),
    }
    with pytest.raises(ValueError, match="duplicate paths"):
        protocol.included_tree_sha256([entry, dict(entry)])

    alias = dict(entry, path="DKI/EXAMPLE.py")
    with pytest.raises(ValueError, match="casefold-alias"):
        protocol.included_tree_sha256([entry, alias])

    file_entry = dict(entry, path="dki")
    nested_entry = dict(entry, path="dki/example.py")
    with pytest.raises(ValueError, match="both a file and directory"):
        protocol._expected_source_layout([file_entry, nested_entry])


def test_canonical_paths_are_derived_only_from_run_binding(tmp_path):
    root, binding = _make_run(tmp_path)
    paths = protocol.canonical_run_paths(binding)

    assert paths.split_dir == root / "evidence" / "cmeie" / "split"
    assert paths.preflight_receipt == (
        root / "evidence" / "cmeie" / "preflight.json"
    )
    assert paths.overfit == root / "evidence" / "cmeie" / "overfit.json"
    assert paths.base_model == root / "models" / "cmeie_base_model"
    assert paths.seed_model(42) == root / "models" / "cmeie" / "seed-42"
    assert paths.training_attestation(42) == (
        root / "evidence" / "cmeie" / "seed-42-training.json"
    )
    assert paths.outer_started.name == "outer_access_started.json"
    assert paths.outer_completion.name == "outer_completion.json"
    assert paths.outer_gate.name == "outer_gate.json"
    assert (
        paths.official_dev_marker.name
        == "official_dev_confirmation_started.json"
    )
    assert (
        paths.official_dev_completion.name
        == "official_dev_confirmation_completed.json"
    )
    with pytest.raises(ValueError, match="one of 42, 43, or 44"):
        paths.seed_model(41)
    with pytest.raises(TypeError, match="non-boolean"):
        paths.seed_model(True)
    with pytest.raises(ValueError, match="one of 42, 43, or 44"):
        paths.training_attestation(41)


def _training_evidence(receipt_sha256: str) -> dict:
    return {
        "training_receipt_sha256": receipt_sha256,
        "epoch_bindings": [
            {
                "epoch": 1,
                "checkpoint_bundle_manifest": "checkpoint_bundle.json",
                "checkpoint_bundle_manifest_sha256": "a" * 64,
                "checkpoint_bundle_tree_sha256": "b" * 64,
                "checkpoint_sha256": "c" * 64,
                "extraction_contract": "extraction_contract.json",
                "extraction_contract_sha256": "d" * 64,
                "semantic_contract_sha256": "e" * 64,
            }
        ],
        "preflight_common": {
            "preflight_receipt_sha256": "f" * 64,
        },
        "source_binding": {
            "source_tree_sha256": "1" * 64,
        },
        "data_binding": {
            "fit_shard_sha256": "2" * 64,
            "calibration_shard_sha256": "3" * 64,
        },
        "configuration": {
            "epochs": 8,
            "fp16": True,
        },
    }


def test_training_attestation_is_create_once_and_binds_candidate(tmp_path):
    _, binding = _make_run(tmp_path)
    paths = protocol.canonical_run_paths(binding)
    candidate = paths.seed_model(42)
    candidate.mkdir(parents=True)
    receipt = candidate / "training_receipt.json"
    receipt.write_text('{"schema_version":6}\n', encoding="utf-8")
    receipt_sha256 = _sha(receipt.read_bytes())

    attestation = protocol.publish_training_attestation_once(
        binding,
        42,
        _training_evidence(receipt_sha256),
    )
    payload = json.loads(attestation.read_text(encoding="utf-8"))

    assert attestation == paths.training_attestation(42)
    assert payload["candidate_directory"] == str(candidate)
    assert payload["training_receipt_sha256"] == receipt_sha256
    assert payload["rid"] == binding.rid
    assert payload["head"] == binding.git_head
    assert payload["tree"] == binding.included_tree_sha256
    original = attestation.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        protocol.publish_training_attestation_once(
            binding,
            42,
            _training_evidence(receipt_sha256),
        )
    assert attestation.read_bytes() == original


def test_training_attestation_failure_preserves_published_candidate(tmp_path):
    _, binding = _make_run(tmp_path)
    paths = protocol.canonical_run_paths(binding)
    candidate = paths.seed_model(42)
    candidate.mkdir(parents=True)
    receipt = candidate / "training_receipt.json"
    receipt.write_text('{"schema_version":6}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="differs"):
        protocol.publish_training_attestation_once(
            binding,
            42,
            _training_evidence("0" * 64),
        )

    assert candidate.is_dir()
    assert receipt.is_file()
    assert not paths.training_attestation(42).exists()


def test_tampered_binding_canonical_path_is_rejected(tmp_path):
    _, binding = _make_run(tmp_path)
    altered = replace(
        binding,
        source_root=binding.run_root / "source" / "wrong",
    )
    with pytest.raises(ValueError, match="source root path is not canonical"):
        protocol.canonical_paths(altered)
    with pytest.raises(ValueError, match="source root path is not canonical"):
        protocol.publish_dataset_started_once(
            altered,
            protocol.OUTER_DATASET_ROLE,
            "a" * 64,
            "b" * 64,
            {},
        )


def test_dataset_key_excludes_all_model_evidence(tmp_path):
    _, binding = _make_run(tmp_path)
    first_evidence = _model_evidence(binding, name="model-a")
    marker = protocol.publish_dataset_started_once(
        binding,
        protocol.OUTER_DATASET_ROLE,
        "a" * 64,
        "b" * 64,
        first_evidence,
    )
    payload = json.loads(marker.read_text(encoding="utf-8"))

    expected_material = (
        f"{protocol.OUTER_DATASET_ROLE}\0{'a' * 64}\0{'b' * 64}"
    )
    assert payload["dataset_key"] == expected_material
    assert payload["dataset_key_sha256"] == _sha(
        expected_material.encode("utf-8")
    )
    assert payload["rid"] == binding.rid
    assert payload["head"] == binding.git_head
    assert payload["tree"] == binding.included_tree_sha256
    assert payload["run_root"] == str(binding.run_root)
    assert payload["input_model_evidence"] == first_evidence
    assert payload["started"] is True
    assert payload["completion_recorded"] is False


def test_copied_candidate_and_changed_selection_cannot_replay_dataset(
    tmp_path,
):
    _, binding = _make_run(tmp_path)
    first = _model_evidence(binding, name="model-a")
    marker = protocol.publish_dataset_started_once(
        binding,
        protocol.OUTER_DATASET_ROLE,
        "c" * 64,
        "d" * 64,
        first,
    )
    original = marker.read_bytes()

    copied = _model_evidence(binding, name="model-copy")
    shutil.copyfile(
        Path(first["checkpoint_path"]),
        Path(copied["checkpoint_path"]),
    )
    copied["checkpoint_sha256"] = _sha(
        Path(copied["checkpoint_path"]).read_bytes()
    )
    with pytest.raises(FileExistsError, match="already exists"):
        protocol.publish_dataset_started_once(
            binding,
            protocol.OUTER_DATASET_ROLE,
            "c" * 64,
            "d" * 64,
            copied,
        )

    changed_selection = dict(first)
    changed_selection["selection_receipt_path"] = copied[
        "selection_receipt_path"
    ]
    changed_selection["selection_receipt_sha256"] = copied[
        "selection_receipt_sha256"
    ]
    with pytest.raises(FileExistsError, match="already exists"):
        protocol.publish_dataset_started_once(
            binding,
            protocol.OUTER_DATASET_ROLE,
            "c" * 64,
            "d" * 64,
            changed_selection,
        )
    assert marker.read_bytes() == original


def test_distinct_checkpoint_same_dataset_key_is_refused(tmp_path):
    _, binding = _make_run(tmp_path)
    protocol.publish_dataset_started_once(
        binding,
        protocol.OUTER_DATASET_ROLE,
        "1" * 64,
        "2" * 64,
        _model_evidence(binding, name="model-a"),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        protocol.publish_dataset_started_once(
            binding,
            protocol.OUTER_DATASET_ROLE,
            "1" * 64,
            "2" * 64,
            _model_evidence(binding, name="model-b"),
        )


def test_different_rids_have_isolated_dataset_ledgers(tmp_path):
    _, first = _make_run(tmp_path / "one", timestamp="20260729T120001Z")
    _, second = _make_run(tmp_path / "two", timestamp="20260729T120002Z")

    first_marker = protocol.publish_dataset_started_once(
        first,
        protocol.OUTER_DATASET_ROLE,
        "3" * 64,
        "4" * 64,
        _model_evidence(first),
    )
    second_marker = protocol.publish_dataset_started_once(
        second,
        protocol.OUTER_DATASET_ROLE,
        "3" * 64,
        "4" * 64,
        _model_evidence(second),
    )

    assert first_marker != second_marker
    assert first_marker.exists()
    assert second_marker.exists()


def test_dataset_roles_have_separate_fixed_markers(tmp_path):
    _, binding = _make_run(tmp_path)
    evidence = _model_evidence(binding)

    outer = protocol.publish_dataset_started_once(
        binding,
        protocol.OUTER_DATASET_ROLE,
        "5" * 64,
        "6" * 64,
        evidence,
    )
    official = protocol.publish_dataset_started_once(
        binding,
        protocol.OFFICIAL_DEV_DATASET_ROLE,
        "7" * 64,
        "8" * 64,
        evidence,
    )

    assert outer == protocol.dataset_started_path(
        binding, protocol.OUTER_DATASET_ROLE
    )
    assert official == protocol.dataset_started_path(
        binding, protocol.OFFICIAL_DEV_DATASET_ROLE
    )
    assert outer != official


def test_evidence_sha_path_and_boolean_types_are_strict(tmp_path):
    _, binding = _make_run(tmp_path)
    checkpoint = (
        protocol.canonical_paths(binding).seed_model(42) / "strict.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"x")

    with pytest.raises(ValueError, match="lowercase"):
        protocol.publish_dataset_started_once(
            binding,
            protocol.OUTER_DATASET_ROLE,
            "A" * 64,
            "b" * 64,
            {},
        )
    with pytest.raises(ValueError, match="JSON boolean"):
        protocol.publish_dataset_started_once(
            binding,
            protocol.OUTER_DATASET_ROLE,
            "a" * 64,
            "b" * 64,
            {"selection_used": 1},
        )
    escaping = str(
        binding.run_root / "models" / ".." / "outside" / "model.pt"
    )
    with pytest.raises(ValueError, match="lexical traversal"):
        protocol.publish_dataset_started_once(
            binding,
            protocol.OUTER_DATASET_ROLE,
            "a" * 64,
            "b" * 64,
            {
                "checkpoint_path": escaping,
                "checkpoint_sha256": "c" * 64,
            },
        )


def test_reparse_source_directory_is_rejected(monkeypatch, tmp_path):
    root, _ = _make_run(tmp_path)
    source_root = root / "source" / "dki-agents"
    real_lstat = protocol.os.lstat

    class _StatProxy:
        def __init__(self, original):
            self._original = original
            self.st_file_attributes = protocol._REPARSE_POINT_ATTRIBUTE

        def __getattr__(self, name):
            return getattr(self._original, name)

    def marked_lstat(path):
        metadata = real_lstat(path)
        if Path(path) == source_root:
            return _StatProxy(metadata)
        return metadata

    monkeypatch.setattr(protocol.os, "lstat", marked_lstat)
    with pytest.raises(ValueError, match="symlink or reparse"):
        protocol.load_run_binding(root)


def test_concurrent_dataset_marker_has_one_winner(tmp_path):
    _, binding = _make_run(tmp_path)
    evidence = _model_evidence(binding)

    def publish(_):
        try:
            return protocol.publish_dataset_started_once(
                binding,
                protocol.OUTER_DATASET_ROLE,
                "9" * 64,
                "a" * 64,
                evidence,
            )
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(publish, range(8)))

    winners = [path for path in results if path is not None]
    assert len(winners) == 1
    assert winners[0].exists()
    assert sum(path is None for path in results) == 7


def test_post_link_crash_keeps_terminal_marker(monkeypatch, tmp_path):
    _, binding = _make_run(tmp_path)
    evidence = _model_evidence(binding)
    real_read = protocol.read_bound_bytes

    def crash_after_link(path, label, *args, **kwargs):
        if label == "published dataset access marker":
            raise RuntimeError("simulated post-link crash")
        return real_read(path, label, *args, **kwargs)

    monkeypatch.setattr(protocol, "read_bound_bytes", crash_after_link)
    with pytest.raises(RuntimeError, match="post-link crash"):
        protocol.publish_dataset_started_once(
            binding,
            protocol.OUTER_DATASET_ROLE,
            "b" * 64,
            "c" * 64,
            evidence,
        )

    marker = protocol.dataset_started_path(
        binding, protocol.OUTER_DATASET_ROLE
    )
    assert marker.exists()
    monkeypatch.setattr(protocol, "read_bound_bytes", real_read)
    with pytest.raises(FileExistsError, match="already exists"):
        protocol.publish_dataset_started_once(
            binding,
            protocol.OUTER_DATASET_ROLE,
            "b" * 64,
            "c" * 64,
            evidence,
        )
