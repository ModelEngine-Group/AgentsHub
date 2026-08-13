import errno
import hashlib
import importlib.util
import json
import os
import stat
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "materialize_hf_snapshot.py"
)
SPEC = importlib.util.spec_from_file_location("materialize_hf_snapshot", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _make_symlink(link, target, *, directory=False):
    link.symlink_to(target, target_is_directory=directory)


def _cache_snapshot(tmp_path, *, data=b'{"model":"safe"}\n'):
    allowed = tmp_path / "cache"
    model_root = allowed / "models--example--model"
    blobs = model_root / "blobs"
    snapshot = model_root / "snapshots" / "revision"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    blob = blobs / hashlib.sha256(data).hexdigest()
    blob.write_bytes(data)
    link = snapshot / "config.json"
    _make_symlink(link, Path("..") / ".." / "blobs" / blob.name)
    return allowed, snapshot, blob, link


def _regular_snapshot(tmp_path, *, data=b"regular snapshot\n"):
    allowed = tmp_path / "cache"
    snapshot = allowed / "models--example--model" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    source = snapshot / "weights.bin"
    source.write_bytes(data)
    return allowed, snapshot, source


def _receipt(output):
    return json.loads(
        (output / MODULE.RECEIPT_NAME).read_text(encoding="utf-8")
    )


def _staging_paths(output):
    return list(
        output.parent.glob(f".{output.name}.materializing-*")
    )


def test_relative_file_symlink_is_materialized_as_regular_file(tmp_path):
    allowed, snapshot, blob, _ = _cache_snapshot(tmp_path)
    regular = snapshot / "tokenizer.json"
    regular.write_bytes(b'{"tokens":[]}\n')
    output = tmp_path / "materialized"

    summary = MODULE.materialize_snapshot(
        source_snapshot=snapshot,
        allowed_root=allowed,
        output=output,
    )

    config = output / "config.json"
    tokenizer = output / "tokenizer.json"
    assert stat.S_ISREG(os.lstat(config).st_mode)
    assert stat.S_ISREG(os.lstat(tokenizer).st_mode)
    assert not config.is_symlink()
    assert config.read_bytes() == blob.read_bytes()
    assert tokenizer.read_bytes() == regular.read_bytes()
    receipt_path = output / MODULE.RECEIPT_NAME
    assert stat.S_ISREG(os.lstat(receipt_path).st_mode)

    receipt = _receipt(output)
    assert receipt["content_tree"]["receipt_excluded"] is True
    assert receipt["content_tree"]["n_files"] == 2
    assert [entry["relative_path"] for entry in receipt["entries"]] == [
        "config.json",
        "tokenizer.json",
    ]
    link_entry = receipt["entries"][0]
    assert link_entry["kind"] == "symlink"
    assert link_entry["link_target"].replace("\\", "/").startswith(
        "../../blobs/"
    )
    assert Path(link_entry["resolved_source"]) == blob
    assert link_entry["sha256"] == hashlib.sha256(blob.read_bytes()).hexdigest()
    assert receipt["entries"][1]["kind"] == "regular_file"
    assert "link_target" not in receipt["entries"][1]
    assert summary["content_tree_sha256"] == receipt["content_tree"]["sha256"]
    assert summary["receipt_sha256"] == hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()


def test_link_escape_is_rejected_without_output(tmp_path):
    allowed = tmp_path / "cache"
    snapshot = allowed / "model" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    link = snapshot / "weights.bin"
    _make_symlink(link, os.path.relpath(outside, link.parent))
    output = tmp_path / "escape-output"

    with pytest.raises(ValueError, match="escapes the allowed root"):
        MODULE.materialize_snapshot(
            source_snapshot=snapshot,
            allowed_root=allowed,
            output=output,
        )

    assert not output.exists()
    assert _staging_paths(output) == []


def test_directory_link_is_rejected(tmp_path):
    allowed = tmp_path / "cache"
    target = allowed / "model" / "blobs"
    snapshot = allowed / "model" / "snapshots" / "revision"
    target.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (target / "file.bin").write_bytes(b"x")
    link = snapshot / "linked-directory"
    _make_symlink(
        link,
        Path("..") / ".." / "blobs",
        directory=True,
    )

    with pytest.raises(ValueError, match="directory symlink"):
        MODULE.materialize_snapshot(
            source_snapshot=snapshot,
            allowed_root=allowed,
            output=tmp_path / "directory-output",
        )


def test_link_cycle_is_rejected(tmp_path):
    allowed = tmp_path / "cache"
    snapshot = allowed / "model" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    first = snapshot / "first.bin"
    second = snapshot / "second.bin"
    _make_symlink(first, "second.bin")
    _make_symlink(second, "first.bin")

    with pytest.raises(ValueError, match="cycle"):
        MODULE.materialize_snapshot(
            source_snapshot=snapshot,
            allowed_root=allowed,
            output=tmp_path / "cycle-output",
        )


def test_target_directory_link_component_is_rejected(tmp_path):
    allowed = tmp_path / "cache"
    actual = allowed / "actual"
    actual.mkdir(parents=True)
    (actual / "blob").write_bytes(b"blob")
    alias = allowed / "alias"
    _make_symlink(alias, actual, directory=True)
    snapshot = allowed / "model" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    link = snapshot / "weights.bin"
    _make_symlink(link, os.path.relpath(alias / "blob", link.parent))

    with pytest.raises(ValueError, match="directory link"):
        MODULE.materialize_snapshot(
            source_snapshot=snapshot,
            allowed_root=allowed,
            output=tmp_path / "component-output",
        )


def test_source_tamper_after_copy_is_rejected_and_staging_is_removed(
    monkeypatch,
    tmp_path,
):
    allowed, snapshot, blob = _regular_snapshot(tmp_path)
    output = tmp_path / "tamper-output"
    original_copy = MODULE._copy_entry_to_staging
    changed = False

    def copy_then_change(entry, *, staging, allowed_root):
        nonlocal changed
        result = original_copy(
            entry,
            staging=staging,
            allowed_root=allowed_root,
        )
        if not changed:
            changed = True
            blob.write_bytes(b"changed after copy")
        return result

    monkeypatch.setattr(
        MODULE,
        "_copy_entry_to_staging",
        copy_then_change,
    )
    with pytest.raises(ValueError, match="inventory changed|source changed"):
        MODULE.materialize_snapshot(
            source_snapshot=snapshot,
            allowed_root=allowed,
            output=output,
        )

    assert not output.exists()
    assert _staging_paths(output) == []


def test_extra_source_alias_after_copy_is_rejected(
    monkeypatch,
    tmp_path,
):
    allowed, snapshot, _ = _regular_snapshot(tmp_path)
    output = tmp_path / "extra-output"
    original_copy = MODULE._copy_entry_to_staging
    added = False

    def copy_then_add(entry, *, staging, allowed_root):
        nonlocal added
        result = original_copy(
            entry,
            staging=staging,
            allowed_root=allowed_root,
        )
        if not added:
            added = True
            (snapshot / "extra.json").write_bytes(b"extra")
        return result

    monkeypatch.setattr(
        MODULE,
        "_copy_entry_to_staging",
        copy_then_add,
    )
    with pytest.raises(ValueError, match="inventory changed"):
        MODULE.materialize_snapshot(
            source_snapshot=snapshot,
            allowed_root=allowed,
            output=output,
        )

    assert not output.exists()
    assert _staging_paths(output) == []


def test_casefold_alias_and_noncanonical_unicode_are_rejected(tmp_path):
    allowed = tmp_path / "cache"
    snapshot = allowed / "model" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    if os.name != "nt":
        # A case-sensitive filesystem can represent both aliases and therefore
        # exercises the collision directly.  Windows cannot create this pair.
        (snapshot / "Case.json").write_bytes(b"a")
        (snapshot / "case.json").write_bytes(b"b")
        with pytest.raises(ValueError, match="casefold-alias"):
            MODULE.materialize_snapshot(
                source_snapshot=snapshot,
                allowed_root=allowed,
                output=tmp_path / "alias-output",
            )

    other = allowed / "model" / "snapshots" / "other"
    other.mkdir()
    (other / "e\u0301.json").write_bytes(b"x")
    with pytest.raises(ValueError, match="noncanonical"):
        MODULE.materialize_snapshot(
            source_snapshot=other,
            allowed_root=allowed,
            output=tmp_path / "unicode-output",
        )


def test_deterministic_receipt_after_normalizing_absolute_roots(tmp_path):
    outputs = []
    receipts = []
    for name in ("one", "two"):
        case_root = tmp_path / name
        allowed, snapshot, _ = _regular_snapshot(case_root)
        (snapshot / "tokenizer.json").write_bytes(b'{"tokens":[]}\n')
        output = case_root / "materialized"
        MODULE.materialize_snapshot(
            source_snapshot=snapshot,
            allowed_root=allowed,
            output=output,
        )
        outputs.append(output)
        receipt = _receipt(output)
        receipt["source_snapshot"] = "<SOURCE>"
        receipt["allowed_root"] = "<ALLOWED>"
        for entry in receipt["entries"]:
            resolved = Path(entry["resolved_source"])
            entry["resolved_source"] = (
                "<ALLOWED>/"
                + resolved.relative_to(allowed).as_posix()
            )
        receipts.append(receipt)

    assert receipts[0] == receipts[1]
    for output in outputs:
        assert _staging_paths(output) == []


def test_non_symlink_reparse_source_is_rejected(monkeypatch, tmp_path):
    allowed, snapshot, source = _regular_snapshot(tmp_path)
    original_lstat = MODULE.os.lstat

    def marked_lstat(path):
        metadata = original_lstat(path)
        if Path(path) == source:
            return SimpleNamespace(
                st_mode=metadata.st_mode,
                st_file_attributes=MODULE._REPARSE_POINT_ATTRIBUTE,
            )
        return metadata

    monkeypatch.setattr(MODULE.os, "lstat", marked_lstat)
    with pytest.raises(ValueError, match="non-symlink reparse"):
        MODULE.materialize_snapshot(
            source_snapshot=snapshot,
            allowed_root=allowed,
            output=tmp_path / "reparse-output",
        )


def test_concurrent_materialization_has_at_most_one_winner(
    monkeypatch,
    tmp_path,
):
    allowed, snapshot, _ = _regular_snapshot(tmp_path)
    output = tmp_path / "race-output"
    barrier = threading.Barrier(2)
    original_publish = MODULE._publish_directory_create_once

    def synchronized_publish(staging, destination):
        barrier.wait(timeout=10)
        return original_publish(staging, destination)

    monkeypatch.setattr(
        MODULE,
        "_publish_directory_create_once",
        synchronized_publish,
    )

    def run():
        return MODULE.materialize_snapshot(
            source_snapshot=snapshot,
            allowed_root=allowed,
            output=output,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run) for _ in range(2)]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(("ok", future.result()))
            except FileExistsError as exc:
                outcomes.append(("exists", exc))

    assert sorted(kind for kind, _ in outcomes) == ["exists", "ok"]
    assert (output / "weights.bin").read_bytes()
    assert _receipt(output)["content_tree"]["n_files"] == 1
    assert _staging_paths(output) == []


def test_existing_output_is_rejected_without_modification(tmp_path):
    allowed, snapshot, _ = _regular_snapshot(tmp_path)
    output = tmp_path / "existing-output"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_bytes(b"keep")

    with pytest.raises(FileExistsError):
        MODULE.materialize_snapshot(
            source_snapshot=snapshot,
            allowed_root=allowed,
            output=output,
        )

    assert marker.read_bytes() == b"keep"
    assert _staging_paths(output) == []


def test_unsupported_publication_preserves_completed_staging(
    monkeypatch,
    tmp_path,
):
    allowed, snapshot, _ = _regular_snapshot(tmp_path)
    output = tmp_path / "unsupported-output"

    def unsupported(staging, destination):
        raise OSError(errno.ENOTSUP, "unsupported", str(destination))

    monkeypatch.setattr(
        MODULE,
        "_publish_directory_create_once",
        unsupported,
    )
    with pytest.raises(OSError) as captured:
        MODULE.materialize_snapshot(
            source_snapshot=snapshot,
            allowed_root=allowed,
            output=output,
        )

    assert captured.value.errno == errno.ENOTSUP
    assert not output.exists()
    staging = _staging_paths(output)
    assert len(staging) == 1
    assert (staging[0] / "weights.bin").read_bytes()
    assert (staging[0] / MODULE.RECEIPT_NAME).is_file()


def test_special_source_file_is_rejected(tmp_path):
    allowed = tmp_path / "cache"
    snapshot = allowed / "model" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    os.mkfifo(snapshot / "pipe")

    with pytest.raises(ValueError, match="special file"):
        MODULE.materialize_snapshot(
            source_snapshot=snapshot,
            allowed_root=allowed,
            output=tmp_path / "special-output",
        )


def test_cli_requires_all_paths_and_emits_summary(tmp_path, capsys):
    allowed, snapshot, _ = _regular_snapshot(tmp_path)
    output = tmp_path / "cli-output"

    assert (
        MODULE.main(
            [
                "--source-snapshot",
                str(snapshot),
                "--allowed-root",
                str(allowed),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["output"] == str(output)
    assert summary["n_files"] == 1
    assert len(summary["content_tree_sha256"]) == 64


# These nodes require filesystem objects that an ordinary Windows process
# cannot create.  Marking them as non-tests avoids adding acceptance skips;
# Linux still collects and executes the real filesystem cases.
if os.name == "nt":
    for _platform_only_node in (
        test_relative_file_symlink_is_materialized_as_regular_file,
        test_link_escape_is_rejected_without_output,
        test_directory_link_is_rejected,
        test_link_cycle_is_rejected,
        test_target_directory_link_component_is_rejected,
        test_special_source_file_is_rejected,
    ):
        _platform_only_node.__test__ = False
