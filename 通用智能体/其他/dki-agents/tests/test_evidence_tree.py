import hashlib
import importlib.util
import os
import stat
from pathlib import Path

import pytest

from dki import evidence_tree


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree_digest(entries):
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _model_tree(tmp_path):
    model = tmp_path / "model"
    nested = model / "nested"
    nested.mkdir(parents=True)
    (model / "config.json").write_bytes(b'{"model":"test"}\n')
    (model / "materialization_receipt.json").write_bytes(
        b'{"materialized":true}\n'
    )
    (nested / "weights.bin").write_bytes(b"\x00\x01weights")
    return model


def test_tree_report_is_deterministic_and_recomputable(tmp_path):
    model = _model_tree(tmp_path)

    first = evidence_tree.scan_bound_regular_tree(
        model, label="base model"
    )
    os.utime(model / "config.json", None)
    second = evidence_tree.scan_bound_regular_tree(
        model, label="base model"
    )

    assert first == second
    assert first["root"] == str(model.absolute())
    assert first["algorithm"] == evidence_tree.TREE_HASH_ALGORITHM
    assert [entry["path"] for entry in first["entries"]] == [
        "config.json",
        "materialization_receipt.json",
        "nested/weights.bin",
    ]
    assert first["sha256"] == _tree_digest(first["entries"])
    assert first["n_files"] == 3
    assert first["total_bytes"] == sum(
        entry["size"] for entry in first["entries"]
    )


def test_outside_hardlink_and_receipt_hardlink_are_rejected(tmp_path):
    model = _model_tree(tmp_path)
    outside = tmp_path / "outside-alias.bin"
    os.link(model / "nested" / "weights.bin", outside)
    with pytest.raises(ValueError, match="multiple hard links"):
        evidence_tree.scan_bound_regular_tree(model, label="base model")

    outside.unlink()
    receipt_alias = tmp_path / "receipt-alias.json"
    os.link(model / "materialization_receipt.json", receipt_alias)
    with pytest.raises(ValueError, match="multiple hard links"):
        evidence_tree.scan_bound_regular_tree(model, label="base model")


def test_path_replacement_after_descriptor_hash_is_rejected(
    monkeypatch, tmp_path
):
    model = _model_tree(tmp_path)
    target = model / "config.json"
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(b'{"evil!":"yes"}\n')
    real_hash = evidence_tree.hash_bound_file
    replaced = False

    def hash_then_replace(path, label, reject_multiple_links=True):
        nonlocal replaced
        binding = real_hash(
            path,
            label,
            reject_multiple_links=reject_multiple_links,
        )
        if Path(path) == target and not replaced:
            replaced = True
            os.replace(replacement, target)
        return binding

    monkeypatch.setattr(evidence_tree, "hash_bound_file", hash_then_replace)
    with pytest.raises(ValueError, match="tree changed"):
        evidence_tree.scan_bound_regular_tree(model, label="base model")
    assert replaced


def test_same_size_in_place_tampering_after_hash_is_rejected(
    monkeypatch, tmp_path
):
    model = _model_tree(tmp_path)
    target = model / "config.json"
    original = target.read_bytes()
    replacement = b"x" * len(original)
    real_hash = evidence_tree.hash_bound_file
    changed = False

    def hash_then_change(path, label, reject_multiple_links=True):
        nonlocal changed
        binding = real_hash(
            path,
            label,
            reject_multiple_links=reject_multiple_links,
        )
        if Path(path) == target and not changed:
            changed = True
            before = target.stat()
            target.write_bytes(replacement)
            os.utime(
                target,
                ns=(
                    before.st_atime_ns,
                    before.st_mtime_ns + 2_000_000_000,
                ),
            )
        return binding

    monkeypatch.setattr(evidence_tree, "hash_bound_file", hash_then_change)
    with pytest.raises(ValueError, match="tree changed"):
        evidence_tree.scan_bound_regular_tree(model, label="base model")
    assert changed
    assert target.stat().st_size == len(original)


def test_directory_replacement_between_inventories_is_rejected(
    monkeypatch, tmp_path
):
    model = _model_tree(tmp_path)
    nested = model / "nested"
    parked = model / "parked"
    replacement = model / "replacement"
    replacement.mkdir()
    (replacement / "weights.bin").write_bytes(b"\x00\x01weights")
    real_inventory = evidence_tree._inventory_once
    calls = 0

    def replace_before_second(root, *, label):
        nonlocal calls
        calls += 1
        if calls == 2:
            nested.rename(parked)
            replacement.rename(nested)
        return real_inventory(root, label=label)

    monkeypatch.setattr(
        evidence_tree, "_inventory_once", replace_before_second
    )
    with pytest.raises(ValueError, match="tree changed"):
        evidence_tree.scan_bound_regular_tree(model, label="base model")
    assert calls == 2


def test_nonordinary_file_is_rejected_without_platform_skips(tmp_path):
    model = _model_tree(tmp_path)
    target = model / "special"
    if os.name == "nt":
        target.write_bytes(b"ordinary")
        real_lstat = evidence_tree.os.lstat

        class _SpecialStat:
            def __init__(self, original):
                self._original = original

            def __getattr__(self, name):
                if name == "st_mode":
                    return stat.S_IFIFO
                return getattr(self._original, name)

        def special_lstat(path):
            metadata = real_lstat(path)
            if Path(path) == target:
                return _SpecialStat(metadata)
            return metadata

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(evidence_tree.os, "lstat", special_lstat)
        try:
            with pytest.raises(ValueError, match="non-ordinary"):
                evidence_tree.scan_bound_regular_tree(
                    model, label="base model"
                )
        finally:
            monkeypatch.undo()
    else:
        os.mkfifo(target)
        with pytest.raises(ValueError, match="non-ordinary"):
            evidence_tree.scan_bound_regular_tree(model, label="base model")


def test_all_three_consumers_return_the_same_tree_binding(tmp_path):
    model = _model_tree(tmp_path)
    preflight = _load_script(
        "tree_preflight_consumer",
        "scripts/remote_npu_preflight.py",
    )
    smoke = _load_script(
        "tree_smoke_consumer",
        "scripts/smoke_cmeie_gplinker_overfit.py",
    )
    train = _load_script(
        "tree_train_consumer",
        "scripts/train_spo_cmeie.py",
    )

    preflight_report, passed = preflight._cmeie_base_model_report(model)
    smoke_report = smoke._tree_report(model)
    train_report = train._tree_report(model)

    assert passed
    assert smoke_report == train_report
    assert preflight_report["tree_sha256"] == smoke_report["sha256"]
    assert preflight_report["file_count"] == smoke_report["n_files"]
    assert preflight_report["files"] == smoke_report["entries"]
