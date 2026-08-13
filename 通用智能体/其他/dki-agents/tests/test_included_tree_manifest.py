import hashlib
import importlib.util
import threading
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_included_tree_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_included_tree_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
REQUIRES_SECURE_PUBLICATION = pytest.mark.skipif(
    not MODULE._HAS_SECURE_DIR_FD,
    reason="secure direct publication requires no-follow directory descriptors",
)
REQUIRES_WINDOWS_PUBLICATION = pytest.mark.skipif(
    not MODULE._HAS_WINDOWS_CREATE_ONCE,
    reason="test requires Windows atomic create-once publication",
)


def _repository(monkeypatch, tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    monkeypatch.setattr(MODULE, "ROOT", root)
    return root


def _write(root, relative, content):
    path = root.joinpath(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _expected_tree(entries):
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _publish_arguments(entries):
    head = "a" * 40
    tree = MODULE.tree_sha256(entries)
    return f"20260726T090000Z-g{head[:7]}-t{tree[:12]}", head


@REQUIRES_SECURE_PUBLICATION
def test_included_tree_manifest_is_canonical_deterministic_and_create_once(
    monkeypatch, tmp_path
):
    root = _repository(monkeypatch, tmp_path)
    _write(root, "alpha.txt", "alpha\n")
    _write(root, "nested/beta.txt", "beta\n")

    entries = MODULE.build_entries([r"nested\beta.txt", "alpha.txt"])
    assert [entry["path"] for entry in entries] == ["alpha.txt", "nested/beta.txt"]
    tree = _expected_tree(entries)
    assert MODULE.tree_sha256(list(reversed(entries))) == tree

    head = "a" * 40
    rid = f"20260726T090000Z-g{head[:7]}-t{tree[:12]}"
    output = tmp_path / "included_tree_manifest.json"
    payload = MODULE.publish_manifest(
        output=output,
        rid=rid,
        entries=list(reversed(entries)),
        head=head,
    )

    assert payload["included_tree_sha256"] == tree
    assert payload["entries"] == entries
    assert json.loads(output.read_text(encoding="utf-8"))["entries"] == entries
    with pytest.raises(FileExistsError):
        MODULE.publish_manifest(
            output=output,
            rid=rid,
            entries=entries,
            head=head,
        )


@REQUIRES_SECURE_PUBLICATION
def test_publish_manifest_allows_exactly_one_competing_writer(monkeypatch, tmp_path):
    entries = [{"path": "alpha.txt", "size": 1, "sha256": "a" * 64}]
    rid, head = _publish_arguments(entries)
    output = tmp_path / "manifest.json"
    barrier = threading.Barrier(2)
    real_open_new_output = MODULE._open_new_output

    def simultaneous_output_open(parent, descriptor, name):
        barrier.wait(timeout=5)
        return real_open_new_output(parent, descriptor, name)

    def publish_once():
        try:
            return MODULE.publish_manifest(
                output=output,
                rid=rid,
                entries=entries,
                head=head,
            )
        except FileExistsError as exc:
            return exc

    monkeypatch.setattr(MODULE, "_open_new_output", simultaneous_output_open)
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: publish_once(), range(2)))

    payloads = [outcome for outcome in outcomes if isinstance(outcome, dict)]
    collisions = [outcome for outcome in outcomes if isinstance(outcome, FileExistsError)]
    assert len(payloads) == 1
    assert len(collisions) == 1
    assert json.loads(output.read_text(encoding="utf-8")) == payloads[0]


@REQUIRES_SECURE_PUBLICATION
def test_publish_manifest_creates_checked_output_parent(tmp_path):
    entries = [{"path": "alpha.txt", "size": 1, "sha256": "a" * 64}]
    rid, head = _publish_arguments(entries)
    output = tmp_path / "new-parent" / "nested" / "manifest.json"

    payload = MODULE.publish_manifest(output=output, rid=rid, entries=entries, head=head)

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == payload


@REQUIRES_SECURE_PUBLICATION
def test_publish_manifest_rejects_untrusted_writable_output_parent(tmp_path):
    entries = [{"path": "alpha.txt", "size": 1, "sha256": "a" * 64}]
    rid, head = _publish_arguments(entries)
    parent = tmp_path / "shared-output-parent"
    parent.mkdir()
    parent.chmod(0o777)
    output = parent / "manifest.json"

    with pytest.raises(ValueError, match="writable by an untrusted principal"):
        MODULE.publish_manifest(output=output, rid=rid, entries=entries, head=head)
    assert not output.exists()


def test_publish_manifest_fails_closed_without_bound_parent_support(
    monkeypatch, tmp_path
):
    entries = [{"path": "alpha.txt", "size": 1, "sha256": "a" * 64}]
    rid, head = _publish_arguments(entries)
    output = tmp_path / "new-parent" / "manifest.json"
    monkeypatch.setattr(MODULE, "_HAS_SECURE_PUBLICATION", False)
    monkeypatch.setattr(MODULE, "_HAS_WINDOWS_CREATE_ONCE", False)
    monkeypatch.setattr(MODULE, "_HAS_SECURE_DIR_FD", False)

    with pytest.raises(OSError, match="create-once support"):
        MODULE.publish_manifest(output=output, rid=rid, entries=entries, head=head)
    assert not output.parent.exists()


@REQUIRES_WINDOWS_PUBLICATION
def test_windows_publish_manifest_is_deterministic_and_create_once(
    monkeypatch, tmp_path
):
    root = _repository(monkeypatch, tmp_path)
    _write(root, "alpha.txt", "alpha\n")
    _write(root, "nested/beta.txt", "beta\n")
    entries = MODULE.build_entries([r"nested\beta.txt", "alpha.txt"])
    rid, head = _publish_arguments(entries)
    output = tmp_path / "new-parent" / "nested" / "manifest.json"

    payload = MODULE.publish_manifest(
        output=output,
        rid=rid,
        entries=list(reversed(entries)),
        head=head,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert payload["entries"] == sorted(entries, key=lambda item: item["path"])
    with pytest.raises(FileExistsError, match="already exists"):
        MODULE.publish_manifest(
            output=output,
            rid=rid,
            entries=entries,
            head=head,
        )


@REQUIRES_WINDOWS_PUBLICATION
def test_windows_publish_manifest_allows_exactly_one_competing_writer(
    monkeypatch, tmp_path
):
    entries = [{"path": "alpha.txt", "size": 1, "sha256": "a" * 64}]
    rid, head = _publish_arguments(entries)
    output = tmp_path / "manifest.json"
    barrier = threading.Barrier(2)
    real_open_new_output = MODULE._open_new_output_windows

    def simultaneous_output_open(path):
        barrier.wait(timeout=5)
        return real_open_new_output(path)

    def publish_once():
        try:
            return MODULE.publish_manifest(
                output=output,
                rid=rid,
                entries=entries,
                head=head,
            )
        except FileExistsError as exc:
            return exc

    monkeypatch.setattr(
        MODULE,
        "_open_new_output_windows",
        simultaneous_output_open,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: publish_once(), range(2)))

    payloads = [outcome for outcome in outcomes if isinstance(outcome, dict)]
    collisions = [
        outcome for outcome in outcomes if isinstance(outcome, FileExistsError)
    ]
    assert len(payloads) == 1
    assert len(collisions) == 1
    assert json.loads(output.read_text(encoding="utf-8")) == payloads[0]


@REQUIRES_WINDOWS_PUBLICATION
def test_windows_publish_manifest_rejects_output_name_substitution(
    monkeypatch, tmp_path
):
    entries = [{"path": "alpha.txt", "size": 1, "sha256": "a" * 64}]
    rid, head = _publish_arguments(entries)
    output = tmp_path / "manifest.json"
    real_lstat = MODULE._lstat
    output_lstat_calls = 0

    def substituted_lstat(path):
        nonlocal output_lstat_calls
        metadata = real_lstat(path)
        if Path(path) == output:
            output_lstat_calls += 1
            if output_lstat_calls >= 2:
                return SimpleNamespace(
                    st_mode=metadata.st_mode,
                    st_dev=metadata.st_dev,
                    st_ino=metadata.st_ino + 1,
                    st_size=metadata.st_size,
                    st_mtime_ns=metadata.st_mtime_ns,
                    st_birthtime_ns=getattr(metadata, "st_birthtime_ns", None),
                )
        return metadata

    monkeypatch.setattr(MODULE, "_lstat", substituted_lstat)
    with pytest.raises(ValueError, match="output changed during publication"):
        MODULE.publish_manifest(
            output=output,
            rid=rid,
            entries=entries,
            head=head,
        )
    assert output.exists()


@REQUIRES_SECURE_PUBLICATION
def test_publish_manifest_rejects_output_path_substitution(monkeypatch, tmp_path):
    entries = [{"path": "alpha.txt", "size": 1, "sha256": "a" * 64}]
    rid, head = _publish_arguments(entries)
    output = tmp_path / "manifest.json"
    replacement = tmp_path / "replacement.json"
    replacement_bytes = b"attacker-controlled output replacement\n"
    replacement.write_bytes(replacement_bytes)
    real_replace = MODULE.os.replace
    real_write_descriptor = MODULE._write_descriptor

    # Install a competing final file after the sibling staging file is durable.
    # The no-replace publication must lose without modifying that final file.
    def write_then_replace(descriptor, content):
        result = real_write_descriptor(descriptor, content)
        real_replace(replacement, output)
        return result

    monkeypatch.setattr(MODULE, "_write_descriptor", write_then_replace)
    with pytest.raises(FileExistsError, match="already exists"):
        MODULE.publish_manifest(output=output, rid=rid, entries=entries, head=head)

    assert output.read_bytes() == replacement_bytes
    assert list(tmp_path.glob(".manifest.json.*.tmp")) == []


def test_publish_manifest_write_failure_leaves_no_partial_final(
    monkeypatch, tmp_path
):
    entries = [{"path": "alpha.txt", "size": 1, "sha256": "a" * 64}]
    rid, head = _publish_arguments(entries)
    output = tmp_path / "manifest.json"
    real_write = MODULE.os.write
    writes = 0

    # A write interruption affects only the sibling staging file. The final name
    # remains absent, the staging file is cleaned, and a clean retry can publish.
    def interrupted_write(descriptor, content):
        nonlocal writes
        writes += 1
        if writes == 1:
            return real_write(descriptor, content[:1])
        raise OSError("simulated write interruption")

    monkeypatch.setattr(MODULE.os, "write", interrupted_write)
    with pytest.raises(OSError, match="simulated write interruption"):
        MODULE.publish_manifest(output=output, rid=rid, entries=entries, head=head)
    assert not output.exists()
    assert list(tmp_path.glob(".manifest.json.*.tmp")) == []

    monkeypatch.setattr(MODULE.os, "write", real_write)
    payload = MODULE.publish_manifest(
        output=output, rid=rid, entries=entries, head=head
    )
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_publish_manifest_rejects_terminal_final_tampering(monkeypatch, tmp_path):
    entries = [{"path": "alpha.txt", "size": 1, "sha256": "a" * 64}]
    rid, head = _publish_arguments(entries)
    output = tmp_path / "manifest.json"

    if os.name == "nt":
        real_verify = MODULE._read_bound_final_windows

        def tamper_then_verify(**kwargs):
            kwargs["output"].write_bytes(b"attacker")
            return real_verify(**kwargs)

        monkeypatch.setattr(
            MODULE, "_read_bound_final_windows", tamper_then_verify
        )
    else:
        real_verify = MODULE._read_bound_final_posix

        def tamper_then_verify(**kwargs):
            (kwargs["parent"] / kwargs["name"]).write_bytes(b"attacker")
            return real_verify(**kwargs)

        monkeypatch.setattr(
            MODULE, "_read_bound_final_posix", tamper_then_verify
        )

    with pytest.raises(ValueError, match="content differs"):
        MODULE.publish_manifest(
            output=output, rid=rid, entries=entries, head=head
        )
    assert output.read_bytes() == b"attacker"
    assert list(tmp_path.glob(".manifest.json.*.tmp")) == []


@REQUIRES_SECURE_PUBLICATION
def test_publish_manifest_rejects_output_parent_swap(monkeypatch, tmp_path):
    entries = [{"path": "alpha.txt", "size": 1, "sha256": "a" * 64}]
    rid, head = _publish_arguments(entries)
    parent = tmp_path / "manifest-parent"
    replacement_parent = tmp_path / "replacement-parent"
    parked_parent = tmp_path / "parked-parent"
    parent.mkdir()
    replacement_parent.mkdir()
    output = parent / "manifest.json"
    real_open_new_output = MODULE._open_new_output
    real_replace = MODULE.os.replace
    swapped = False

    # Replace the lexical parent after the original has been bound by an open
    # directory descriptor. The temporary file is created only in that bound
    # original, then removed when lexical identity validation fails.
    def open_after_parent_swap(parent_path, descriptor, name):
        nonlocal swapped
        real_replace(parent, parked_parent)
        real_replace(replacement_parent, parent)
        swapped = True
        return real_open_new_output(parent_path, descriptor, name)

    monkeypatch.setattr(MODULE, "_open_new_output", open_after_parent_swap)
    with pytest.raises(ValueError, match="output parent changed during publication"):
        MODULE.publish_manifest(output=output, rid=rid, entries=entries, head=head)

    assert swapped
    assert not (parent / output.name).exists()
    assert list(parked_parent.glob(".manifest.json.*.tmp")) == []


@REQUIRES_SECURE_PUBLICATION
def test_publish_manifest_rejects_parent_creation_redirect(monkeypatch, tmp_path):
    entries = [{"path": "alpha.txt", "size": 1, "sha256": "a" * 64}]
    rid, head = _publish_arguments(entries)
    parent = tmp_path / "manifest-parent"
    redirected_target = tmp_path / "redirected-target"
    parked_parent = tmp_path / "parked-parent"
    parent.mkdir()
    redirected_target.mkdir()
    output = parent / "new-parent" / "manifest.json"
    real_mkdir = MODULE.os.mkdir
    real_replace = MODULE.os.replace
    redirected = False

    # Move the lexical parent just before creation of its missing child. The
    # mkdir must remain bound to the held parent descriptor, never follow the
    # newly installed symlink, and later lexical validation must reject it.
    def mkdir_after_parent_redirect(path, mode=0o777, *, dir_fd=None):
        nonlocal redirected
        if path == "new-parent" and dir_fd is not None and not redirected:
            real_replace(parent, parked_parent)
            parent.symlink_to(redirected_target, target_is_directory=True)
            redirected = True
        return real_mkdir(path, mode=mode, dir_fd=dir_fd)

    monkeypatch.setattr(MODULE.os, "mkdir", mkdir_after_parent_redirect)
    with pytest.raises(ValueError, match="symlink or reparse component"):
        MODULE.publish_manifest(output=output, rid=rid, entries=entries, head=head)

    assert redirected
    assert (parked_parent / "new-parent").is_dir()
    assert not (redirected_target / "new-parent").exists()


@REQUIRES_SECURE_PUBLICATION
def test_publish_manifest_rejects_reparse_output_parent_before_direct_create(
    monkeypatch, tmp_path
):
    entries = [{"path": "alpha.txt", "size": 1, "sha256": "a" * 64}]
    rid, head = _publish_arguments(entries)
    parent = tmp_path / "manifest-parent"
    parent.mkdir()
    output = parent / "manifest.json"
    real_lstat = MODULE._lstat

    def reparse_lstat(path):
        metadata = real_lstat(path)
        if Path(path) == parent:
            return SimpleNamespace(
                st_mode=metadata.st_mode,
                st_file_attributes=MODULE._REPARSE_POINT_ATTRIBUTE,
            )
        return metadata

    monkeypatch.setattr(MODULE, "_lstat", reparse_lstat)
    with pytest.raises(ValueError, match="symlink or reparse component"):
        MODULE.publish_manifest(output=output, rid=rid, entries=entries, head=head)
    assert not output.exists()


def test_publish_manifest_rejects_lexical_output_traversal(tmp_path):
    entries = [{"path": "alpha.txt", "size": 1, "sha256": "a" * 64}]
    rid, head = _publish_arguments(entries)
    output = tmp_path / "new-parent" / ".." / "manifest.json"

    with pytest.raises(ValueError, match="lexical traversal"):
        MODULE.publish_manifest(output=output, rid=rid, entries=entries, head=head)
    assert not (tmp_path / "new-parent").exists()


def test_publish_manifest_rejects_invalid_rid_calendar_date(tmp_path):
    entries = [{"path": "alpha.txt", "size": 1, "sha256": "a" * 64}]
    tree = MODULE.tree_sha256(entries)
    head = "a" * 40
    rid = f"20260230T090000Z-g{head[:7]}-t{tree[:12]}"
    output = tmp_path / "manifest.json"

    with pytest.raises(ValueError, match="calendar"):
        MODULE.publish_manifest(
            output=output, rid=rid, entries=entries, head=head
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("../safe.txt", "lexical traversal"),
        ("nested/../safe.txt", "lexical traversal"),
        ("/safe.txt", "repository-relative"),
        ("C:/safe.txt", "repository-relative"),
        ("safe.txt\0suffix", "NUL"),
    ],
)
def test_included_tree_manifest_rejects_unsafe_lexical_paths(
    monkeypatch, tmp_path, path, message
):
    root = _repository(monkeypatch, tmp_path)
    _write(root, "safe.txt", "safe\n")

    with pytest.raises(ValueError, match=message):
        MODULE.build_entries([path])


def test_included_tree_manifest_rejects_absolute_paths(monkeypatch, tmp_path):
    root = _repository(monkeypatch, tmp_path)
    source = _write(root, "safe.txt", "safe\n")

    with pytest.raises(ValueError, match="repository-relative"):
        MODULE.build_entries([source])


def test_included_tree_manifest_rejects_duplicate_and_case_alias_paths(
    monkeypatch, tmp_path
):
    root = _repository(monkeypatch, tmp_path)
    _write(root, "nested/child.txt", "child\n")
    _write(root, "case.txt", "case\n")

    with pytest.raises(ValueError, match="duplicate root-relative"):
        MODULE.build_entries([r"nested\child.txt", "nested/child.txt"])
    with pytest.raises(ValueError, match="case alias"):
        MODULE.build_entries(["case.txt", "CASE.TXT"])


def test_included_tree_manifest_rejects_hardlinked_source(monkeypatch, tmp_path):
    root = _repository(monkeypatch, tmp_path)
    source = _write(root, "source.txt", "trusted\n")
    alias = root / "alias.txt"
    os.link(source, alias)

    with pytest.raises(ValueError, match="hard link"):
        MODULE.build_entries(["source.txt"])
    with pytest.raises(ValueError, match="hard link"):
        MODULE.build_entries(["alias.txt"])


def test_included_tree_manifest_rejects_symlink_components_before_resolution(
    monkeypatch, tmp_path
):
    root = _repository(monkeypatch, tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    file_link = root / "outside-link.txt"
    try:
        file_link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable in this test environment: {exc}")

    with pytest.raises(ValueError, match="symlink or reparse component"):
        MODULE.build_entries(["outside-link.txt"])

    _write(root, "safe/inside.txt", "inside\n")
    directory_link = root / "safe-link"
    try:
        directory_link.symlink_to(root / "safe", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable in this test environment: {exc}")
    with pytest.raises(ValueError, match="symlink or reparse component"):
        MODULE.build_entries(["safe-link/inside.txt"])


def test_included_tree_manifest_rejects_reparse_before_resolution(
    monkeypatch, tmp_path
):
    root = _repository(monkeypatch, tmp_path)
    source = _write(root, "safe.txt", "safe\n")
    real_lstat = MODULE._lstat

    def reparse_lstat(path):
        metadata = real_lstat(path)
        if Path(path) == source:
            return SimpleNamespace(
                st_mode=metadata.st_mode,
                st_file_attributes=MODULE._REPARSE_POINT_ATTRIBUTE,
            )
        return metadata

    def unexpected_resolve(self, *args, **kwargs):
        raise AssertionError("resolve must not run before reparse validation")

    monkeypatch.setattr(MODULE, "_lstat", reparse_lstat)
    monkeypatch.setattr(MODULE.Path, "resolve", unexpected_resolve)
    with pytest.raises(ValueError, match="symlink or reparse component"):
        MODULE.build_entries(["safe.txt"])


def test_included_tree_manifest_recognizes_reparse_attributes():
    metadata = SimpleNamespace(
        st_mode=stat.S_IFREG,
        st_file_attributes=MODULE._REPARSE_POINT_ATTRIBUTE,
    )
    assert MODULE._is_link_or_reparse(metadata)


def test_included_tree_manifest_hashes_verified_descriptor_after_pathname_swap(
    monkeypatch, tmp_path
):
    root = _repository(monkeypatch, tmp_path)
    source = _write(root, "source.txt", "trusted\n")
    replacement = _write(root, "replacement.txt", "replacement\n")
    trusted_bytes = source.read_bytes()
    replacement_bytes = replacement.read_bytes()
    trusted_digest = hashlib.sha256(trusted_bytes).hexdigest()
    real_open = MODULE.os.open
    real_path_open = MODULE.Path.open
    source_open_count = 0
    swapped = False

    # The first open captures the checked source descriptor. Subsequent path-based
    # opens model source.txt having been replaced by replacement.txt. This is a
    # deterministic, cross-platform name-swap model without relying on Windows
    # rename semantics for open files.
    def swapped_open(path, flags, *args):
        nonlocal source_open_count, swapped
        if Path(path) != source:
            return real_open(path, flags, *args)
        source_open_count += 1
        target = source if source_open_count == 1 else replacement
        descriptor = real_open(target, flags, *args)
        swapped = True
        return descriptor

    def swapped_path_open(path, *args, **kwargs):
        if Path(path) == source and swapped:
            return io.open(replacement, *args, **kwargs)
        return real_path_open(path, *args, **kwargs)

    monkeypatch.setattr(MODULE.os, "open", swapped_open)
    monkeypatch.setattr(MODULE.Path, "open", swapped_path_open)
    entries = MODULE.build_entries(["source.txt"])

    assert entries == [
        {
            "path": "source.txt",
            "size": len(trusted_bytes),
            "sha256": trusted_digest,
        }
    ]
    assert source_open_count == 1
    with source.open("rb") as handle:
        assert handle.read() == replacement_bytes


def test_included_tree_manifest_rejects_descriptor_substitution(
    monkeypatch, tmp_path
):
    root = _repository(monkeypatch, tmp_path)
    source = _write(root, "source.txt", "trusted\n")
    replacement = _write(root, "replacement.txt", "replacement\n")
    real_open = MODULE.os.open

    # Model the check/open race: source.txt passes lexical and lstat checks, but
    # os.open returns a descriptor for replacement.txt. The descriptor identity
    # mismatch must fail closed before its bytes can be accepted into a manifest.
    def swapped_open(path, flags, *args):
        if Path(path) == source:
            return real_open(replacement, flags, *args)
        return real_open(path, flags, *args)

    monkeypatch.setattr(MODULE.os, "open", swapped_open)
    with pytest.raises(ValueError, match="changed before hashing"):
        MODULE.build_entries(["source.txt"])
    assert source.read_text(encoding="utf-8") == "trusted\n"


def test_included_tree_manifest_rejects_descriptor_change_during_read(
    monkeypatch, tmp_path
):
    root = _repository(monkeypatch, tmp_path)
    _write(root, "source.txt", "trusted\n")
    real_fstat = MODULE.os.fstat
    calls = 0

    def changed_fstat(descriptor):
        nonlocal calls
        metadata = real_fstat(descriptor)
        calls += 1
        if calls == 2:
            return SimpleNamespace(
                st_mode=metadata.st_mode,
                st_dev=metadata.st_dev,
                st_ino=metadata.st_ino,
                st_size=metadata.st_size + 1,
                st_mtime_ns=metadata.st_mtime_ns,
                st_ctime_ns=metadata.st_ctime_ns,
                st_birthtime_ns=getattr(metadata, "st_birthtime_ns", None),
            )
        return metadata

    monkeypatch.setattr(MODULE.os, "fstat", changed_fstat)
    with pytest.raises(ValueError, match="changed while hashing"):
        MODULE.build_entries(["source.txt"])
    assert calls == 2


def test_tree_sha256_rejects_noncanonical_and_case_alias_entries():
    digest = "a" * 64
    with pytest.raises(ValueError, match="POSIX separators"):
        MODULE.tree_sha256([{"path": r"nested\child.txt", "size": 1, "sha256": digest}])
    with pytest.raises(ValueError, match="case-alias"):
        MODULE.tree_sha256(
            [
                {"path": "nested/child.txt", "size": 1, "sha256": digest},
                {"path": "nested/CHILD.txt", "size": 1, "sha256": digest},
            ]
        )


def test_hash_only_cli_accepts_relative_paths_without_reading_project_files(
    monkeypatch, tmp_path, capsys
):
    root = _repository(monkeypatch, tmp_path)
    _write(root, "alpha.txt", "alpha\n")
    _write(root, "nested/beta.txt", "beta\n")
    monkeypatch.setattr(MODULE, "git_head", lambda: "b" * 40)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_included_tree_manifest.py",
            "--path",
            "alpha.txt",
            "--path",
            r"nested\beta.txt",
            "--hash-only",
        ],
    )

    assert MODULE.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["git_head"] == "b" * 40
    assert result["n_files"] == 2
