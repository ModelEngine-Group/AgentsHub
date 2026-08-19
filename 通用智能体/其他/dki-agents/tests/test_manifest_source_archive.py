import errno
import gzip
import hashlib
import importlib.util
import io
import json
import os
import stat
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from dki.nlp import cmeie_run_protocol


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_manifest_source_archive.py"
)
SPEC = importlib.util.spec_from_file_location("build_manifest_source_archive", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
MANIFEST_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_included_tree_manifest.py"
)
MANIFEST_SPEC = importlib.util.spec_from_file_location(
    "build_included_tree_manifest_for_archive_test", MANIFEST_SCRIPT
)
MANIFEST_MODULE = importlib.util.module_from_spec(MANIFEST_SPEC)
MANIFEST_SPEC.loader.exec_module(MANIFEST_MODULE)


def _tree_hash(entries):
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item["path"]):
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write(root, relative, data, *, executable=False):
    path = root.joinpath(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _entry(relative, data):
    return {
        "path": relative,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _manifest(
    tmp_path,
    entries,
    *,
    tree=None,
    name="included_tree.json",
    head="a" * 40,
    **overrides,
):
    actual_tree = tree or _tree_hash(entries)
    payload = {
        "schema_version": 1,
        "rid": f"20260726T090000Z-g{head[:7]}-t{actual_tree[:12]}",
        "git_head": head,
        "algorithm": MODULE.TREE_ALGORITHM,
        "entries": sorted(entries, key=lambda item: item["path"]),
        "included_tree_sha256": actual_tree,
    }
    payload.update(overrides)
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _archive_members(path):
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        contents = {
            member.name: archive.extractfile(member).read()
            for member in members
            if member.isfile()
        }
    return members, contents


def _safe_extract_archive(path, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = PurePosixPath(member.name)
            if (
                member_path.is_absolute()
                or not member_path.parts
                or member_path.parts[0] != MODULE.ARCHIVE_ROOT
                or any(part in {"", ".", ".."} for part in member_path.parts)
                or not (member.isdir() or member.isfile())
            ):
                raise ValueError(f"unsafe archive member: {member.name}")
            target = destination.joinpath(*member_path.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"missing archive member bytes: {member.name}")
            target.write_bytes(source.read())


def test_deterministic_archive_has_exact_members_content_and_safe_metadata(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    alpha = b"alpha\n"
    runner = b"#!/bin/sh\nexit 0\n"
    _write(root, "alpha.txt", alpha)
    _write(root, "bin/run.sh", runner, executable=True)
    entries = [_entry("bin/run.sh", runner), _entry("alpha.txt", alpha)]
    manifest = _manifest(tmp_path, entries)
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"

    first_summary = MODULE.build_archive(
        root=root, manifest=manifest, output=first
    )
    os.utime(root / "alpha.txt", None)
    (root / "bin" / "run.sh").chmod(0o600)
    second_summary = MODULE.build_archive(
        root=root, manifest=manifest, output=second
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_summary["archive_sha256"] == hashlib.sha256(
        first.read_bytes()
    ).hexdigest()
    assert second_summary["archive_sha256"] == first_summary["archive_sha256"]
    assert first_summary["included_tree_sha256"] == _tree_hash(entries)

    members, contents = _archive_members(first)
    assert [member.name for member in members] == [
        "dki-agents",
        "dki-agents/alpha.txt",
        "dki-agents/bin/run.sh",
    ]
    assert contents == {
        "dki-agents/alpha.txt": alpha,
        "dki-agents/bin/run.sh": runner,
    }
    assert members[0].isdir()
    assert members[0].mode == 0o755
    assert members[1].mode == 0o644
    assert members[2].mode == 0o644
    for member in members:
        assert member.uid == 0
        assert member.gid == 0
        assert member.uname == ""
        assert member.gname == ""
        assert member.mtime == 0

    raw = first.read_bytes()
    assert raw[:3] == b"\x1f\x8b\x08"
    assert raw[4:8] == b"\0\0\0\0"
    assert raw[9] == 255
    with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as compressed:
        assert compressed.read()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_version": 2}, "schema_version"),
        ({"algorithm": "sha256"}, "algorithm"),
        ({"rid": "20260726T090000Z-gbbbbbbb-tbbbbbbbbbbbb"}, "Git prefix"),
        ({"rid": "20260230T090000Z-gaaaaaaa-taaaaaaaaaaaa"}, "timestamp"),
        ({"unexpected": True}, "exactly"),
    ],
)
def test_manifest_schema1_fields_are_strict(tmp_path, overrides, message):
    data = b"x"
    manifest = _manifest(tmp_path, [_entry("source.txt", data)], **overrides)

    with pytest.raises(ValueError, match=message):
        MODULE.load_manifest(manifest)


def test_manifest_rejects_duplicate_json_keys_and_noncanonical_order(tmp_path):
    data = b"x"
    entry = _entry("source.txt", data)
    tree = _tree_hash([entry])
    head = "a" * 40
    rid = f"20260726T090000Z-g{head[:7]}-t{tree[:12]}"
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        (
            '{"schema_version":1,"schema_version":1,'
            f'"rid":"{rid}","git_head":"{head}",'
            f'"algorithm":"{MODULE.TREE_ALGORITHM}",'
            f'"included_tree_sha256":"{tree}",'
            f'"entries":{json.dumps([entry])}}}'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate key"):
        MODULE.load_manifest(duplicate)

    alpha = _entry("alpha.txt", b"a")
    beta = _entry("beta.txt", b"b")
    unordered = _manifest(tmp_path, [alpha, beta], name="unordered.json")
    payload = json.loads(unordered.read_text(encoding="utf-8"))
    payload["entries"].reverse()
    unordered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical sorted order"):
        MODULE.load_manifest(unordered)


def test_manifest_file_and_source_hardlinks_are_rejected(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    data = b"source\n"
    source = _write(root, "source.txt", data)
    manifest = _manifest(tmp_path, [_entry("source.txt", data)])
    manifest_alias = tmp_path / "manifest-alias.json"
    os.link(manifest, manifest_alias)

    with pytest.raises(ValueError, match="hard link"):
        MODULE.build_archive(
            root=root,
            manifest=manifest_alias,
            output=tmp_path / "manifest-hardlink.tar.gz",
        )

    manifest_alias.unlink()
    source_alias = root / "source-alias.txt"
    os.link(source, source_alias)
    with pytest.raises(ValueError, match="hard link"):
        MODULE.build_archive(
            root=root,
            manifest=manifest,
            output=tmp_path / "source-hardlink.tar.gz",
        )
    assert not (tmp_path / "source-hardlink.tar.gz").exists()


@pytest.mark.parametrize("kind", ["content", "size", "digest", "tree"])
def test_tampering_is_rejected_without_output_or_staging(tmp_path, kind):
    root = tmp_path / "repo"
    root.mkdir()
    original = b"trusted\n"
    source = _write(root, "source.txt", original)
    entries = [_entry("source.txt", original)]
    tree = None
    if kind == "content":
        source.write_bytes(b"changed\n")
    elif kind == "size":
        entries[0]["size"] += 1
        tree = _tree_hash(entries)
    elif kind == "digest":
        entries[0]["sha256"] = "a" * 64
        tree = _tree_hash(entries)
    else:
        tree = "b" * 64
    manifest = _manifest(tmp_path, entries, tree=tree)
    output = tmp_path / "tampered.tar.gz"

    with pytest.raises(ValueError):
        MODULE.build_archive(root=root, manifest=manifest, output=output)

    assert not output.exists()
    assert list(tmp_path.glob(".tampered.tar.gz.*.tmp")) == []


@pytest.mark.parametrize(
    "unsafe",
    [
        "../escape.txt",
        "/absolute.txt",
        "C:/absolute.txt",
        r"nested\child.txt",
        "./source.txt",
        "nested//child.txt",
        "nested/../source.txt",
        "",
    ],
)
def test_unsafe_manifest_paths_are_rejected(tmp_path, unsafe):
    root = tmp_path / "repo"
    root.mkdir()
    data = b"x"
    entries = [_entry(unsafe, data)]
    manifest = _manifest(tmp_path, entries)
    output = tmp_path / "unsafe.tar.gz"

    with pytest.raises(ValueError):
        MODULE.build_archive(root=root, manifest=manifest, output=output)

    assert not output.exists()


def test_duplicate_and_casefold_alias_paths_are_rejected(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    data = b"x"
    for entries in (
        [_entry("same.txt", data), _entry("same.txt", data)],
        [_entry("Case.txt", data), _entry("case.txt", data)],
        [_entry("\u00e9.txt", data), _entry("e\u0301.txt", data)],
    ):
        manifest = _manifest(tmp_path, entries)
        with pytest.raises(ValueError):
            MODULE.build_archive(
                root=root,
                manifest=manifest,
                output=tmp_path / "alias.tar.gz",
            )


def test_symlink_source_and_symlink_component_are_rejected_where_supported(
    tmp_path,
):
    root = tmp_path / "repo"
    root.mkdir()
    real = _write(root, "real.txt", b"real\n")
    directory = root / "real-dir"
    directory.mkdir()
    nested = _write(root, "real-dir/nested.txt", b"nested\n")
    file_link = root / "file-link.txt"
    directory_link = root / "directory-link"
    try:
        file_link.symlink_to(real)
        directory_link.symlink_to(directory, target_is_directory=True)
    except (NotImplementedError, OSError):
        # Some locked-down Windows test volumes cannot create symlinks.  Keep
        # this node green there without weakening coverage: the following
        # reparse-attribute test exercises build_archive's rejection path, and
        # this assertion still verifies the shared link/reparse predicate.
        assert MODULE._is_link_or_reparse(
            SimpleNamespace(
                st_mode=stat.S_IFREG,
                st_file_attributes=MODULE._REPARSE_POINT_ATTRIBUTE,
            )
        )
        return

    for relative, data in (
        ("file-link.txt", real.read_bytes()),
        ("directory-link/nested.txt", nested.read_bytes()),
    ):
        entries = [_entry(relative, data)]
        manifest = _manifest(tmp_path, entries)
        with pytest.raises(ValueError, match="symlink or reparse"):
            MODULE.build_archive(
                root=root,
                manifest=manifest,
                output=tmp_path / f"{relative.replace('/', '-')}.tar.gz",
            )


def test_reparse_attribute_is_rejected_before_source_open(
    monkeypatch, tmp_path
):
    root = tmp_path / "repo"
    root.mkdir()
    source = _write(root, "source.txt", b"source\n")
    entries = [_entry("source.txt", source.read_bytes())]
    manifest = _manifest(tmp_path, entries)
    real_lstat = MODULE.os.lstat

    def marked_lstat(path):
        metadata = real_lstat(path)
        if Path(path) == source:
            return SimpleNamespace(
                st_mode=metadata.st_mode,
                st_file_attributes=MODULE._REPARSE_POINT_ATTRIBUTE,
                st_ino=metadata.st_ino,
                st_dev=metadata.st_dev,
                st_size=metadata.st_size,
                st_mtime_ns=metadata.st_mtime_ns,
                st_ctime_ns=metadata.st_ctime_ns,
            )
        return metadata

    monkeypatch.setattr(MODULE.os, "lstat", marked_lstat)
    with pytest.raises(ValueError, match="symlink or reparse"):
        MODULE.build_archive(
            root=root,
            manifest=manifest,
            output=tmp_path / "reparse.tar.gz",
        )


def test_existing_output_is_rejected_without_modification_or_staging(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    data = b"source\n"
    _write(root, "source.txt", data)
    manifest = _manifest(tmp_path, [_entry("source.txt", data)])
    output = tmp_path / "existing.tar.gz"
    output.write_bytes(b"keep me")

    with pytest.raises(FileExistsError):
        MODULE.build_archive(root=root, manifest=manifest, output=output)

    assert output.read_bytes() == b"keep me"
    assert list(tmp_path.glob(".existing.tar.gz.*.tmp")) == []


def test_archive_write_failure_leaves_no_partial_final_or_staging(
    monkeypatch, tmp_path
):
    root = tmp_path / "repo"
    root.mkdir()
    data = b"source\n"
    _write(root, "source.txt", data)
    manifest = _manifest(tmp_path, [_entry("source.txt", data)])
    output = tmp_path / "interrupted.tar.gz"

    def interrupted_write(output_file, **kwargs):
        output_file.write(b"partial")
        raise OSError("simulated archive write interruption")

    monkeypatch.setattr(MODULE, "_write_archive", interrupted_write)
    with pytest.raises(OSError, match="simulated archive write interruption"):
        MODULE.build_archive(root=root, manifest=manifest, output=output)

    assert not output.exists()
    assert list(tmp_path.glob(".interrupted.tar.gz.*.tmp")) == []


def test_terminal_archive_tampering_is_detected(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    data = b"source\n"
    _write(root, "source.txt", data)
    manifest = _manifest(tmp_path, [_entry("source.txt", data)])
    output = tmp_path / "tampered-terminal.tar.gz"

    if os.name == "nt":
        real_publish = MODULE._move_no_replace_windows

        def publish_then_tamper(staging, destination):
            real_publish(staging, destination)
            Path(destination).write_bytes(b"attacker")

        monkeypatch.setattr(
            MODULE, "_move_no_replace_windows", publish_then_tamper
        )
    else:
        real_publish = MODULE._publish_link_no_replace

        def publish_then_tamper(**kwargs):
            real_publish(**kwargs)
            (
                kwargs["parent"] / kwargs["output_name"]
            ).write_bytes(b"attacker")

        monkeypatch.setattr(
            MODULE, "_publish_link_no_replace", publish_then_tamper
        )

    with pytest.raises(ValueError, match="differs from staging"):
        MODULE.build_archive(root=root, manifest=manifest, output=output)
    assert output.read_bytes() == b"attacker"
    assert list(tmp_path.glob(".tampered-terminal.tar.gz.*.tmp")) == []


def test_concurrent_publication_has_one_winner_and_no_staging_files(
    monkeypatch, tmp_path
):
    root = tmp_path / "repo"
    root.mkdir()
    data = b"source\n"
    _write(root, "source.txt", data)
    manifest = _manifest(tmp_path, [_entry("source.txt", data)])
    output = tmp_path / "race.tar.gz"
    barrier = threading.Barrier(2)
    if os.name == "nt":
        real_publish = MODULE._move_no_replace_windows

        def synchronized_publish(staging, destination):
            barrier.wait(timeout=10)
            return real_publish(staging, destination)

        monkeypatch.setattr(
            MODULE, "_move_no_replace_windows", synchronized_publish
        )
    else:
        real_publish = MODULE._publish_link_no_replace

        def synchronized_publish(**kwargs):
            barrier.wait(timeout=10)
            return real_publish(**kwargs)

        monkeypatch.setattr(
            MODULE, "_publish_link_no_replace", synchronized_publish
        )

    def build():
        return MODULE.build_archive(
            root=root, manifest=manifest, output=output
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(build) for _ in range(2)]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(("ok", future.result()))
            except FileExistsError as exc:
                outcomes.append(("exists", exc))

    assert sorted(kind for kind, _ in outcomes) == ["exists", "ok"]
    members, contents = _archive_members(output)
    assert [member.name for member in members] == [
        "dki-agents",
        "dki-agents/source.txt",
    ]
    assert contents["dki-agents/source.txt"] == data
    assert list(tmp_path.glob(".race.tar.gz.*.tmp")) == []


def test_publication_race_does_not_overwrite_competing_output(
    monkeypatch, tmp_path
):
    root = tmp_path / "repo"
    root.mkdir()
    data = b"source\n"
    _write(root, "source.txt", data)
    manifest = _manifest(tmp_path, [_entry("source.txt", data)])
    output = tmp_path / "race-loser.tar.gz"

    if os.name == "nt":
        def lose_race(staging, destination):
            Path(destination).write_bytes(b"winner")
            raise FileExistsError(errno.EEXIST, "exists", destination)

        monkeypatch.setattr(MODULE, "_move_no_replace_windows", lose_race)
    else:
        def lose_race(**kwargs):
            destination = output.parent / kwargs["output_name"]
            destination.write_bytes(b"winner")
            raise FileExistsError(errno.EEXIST, "exists", destination)

        monkeypatch.setattr(MODULE, "_publish_link_no_replace", lose_race)
    with pytest.raises(FileExistsError):
        MODULE.build_archive(root=root, manifest=manifest, output=output)

    assert output.read_bytes() == b"winner"
    assert list(tmp_path.glob(".race-loser.tar.gz.*.tmp")) == []


def test_cli_requires_explicit_paths_and_reports_summary(tmp_path, capsys):
    root = tmp_path / "repo"
    root.mkdir()
    data = b"source\n"
    _write(root, "source.txt", data)
    manifest = _manifest(tmp_path, [_entry("source.txt", data)])
    output = tmp_path / "cli.tar.gz"

    assert (
        MODULE.main(
            [
                "--root",
                str(root),
                "--manifest",
                str(manifest),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["output"] == str(output)
    assert summary["n_files"] == 1
    assert summary["archive_sha256"] == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()


def test_manifest_archive_installs_exact_tree_for_run_binding(
    monkeypatch, tmp_path
):
    repository = tmp_path / "repository"
    repository.mkdir()
    source_bytes = b"authoritative source\n"
    _write(repository, "dki/example.py", source_bytes)
    monkeypatch.setattr(MANIFEST_MODULE, "ROOT", repository)
    entries = MANIFEST_MODULE.build_entries(["dki/example.py"])
    tree = MANIFEST_MODULE.tree_sha256(entries)
    head = "30fce24" + ("a" * 33)
    rid = f"20260729T120000Z-g{head[:7]}-t{tree[:12]}"
    manifest = tmp_path / "included_tree.json"
    MANIFEST_MODULE.publish_manifest(
        output=manifest,
        rid=rid,
        entries=entries,
        head=head,
    )

    archive_path = tmp_path / "source.tar.gz"
    MODULE.build_archive(
        root=repository,
        manifest=manifest,
        output=archive_path,
    )
    run_root = tmp_path / f"dki-agents-{rid}"
    _safe_extract_archive(archive_path, run_root / "source")
    evidence_source = run_root / "evidence" / "source"
    evidence_source.mkdir(parents=True)
    installed_manifest = evidence_source / "included_tree.json"
    installed_manifest.write_bytes(manifest.read_bytes())

    binding = cmeie_run_protocol.load_run_binding(run_root)
    assert binding.rid == rid
    assert binding.included_tree_sha256 == tree
    assert binding.source_root == run_root / "source" / "dki-agents"

    installed_source = binding.source_root / "dki" / "example.py"
    installed_source.write_bytes(b"tampered source\n")
    with pytest.raises(ValueError, match="differs from the manifest"):
        cmeie_run_protocol.load_run_binding(run_root)

    installed_source.write_bytes(source_bytes)
    (binding.source_root / "extra.txt").write_text(
        "extra\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exact manifest tree"):
        cmeie_run_protocol.load_run_binding(run_root)
