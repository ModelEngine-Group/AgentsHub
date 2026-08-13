import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from dki import evidence_io


class _StatProxy:
    def __init__(self, original, **overrides):
        self._original = original
        self._overrides = overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._original, name)


def test_read_bound_bytes_returns_payload_and_complete_binding(tmp_path):
    path = tmp_path / "evidence.bin"
    payload = (b"0123456789abcdef" * 196_608) + b"tail"
    path.write_bytes(payload)

    actual, binding = evidence_io.read_bound_bytes(path, "large evidence")

    assert actual == payload
    assert binding["canonical_path"] == str(path.absolute())
    assert binding["size"] == len(payload)
    assert binding["sha256"] == hashlib.sha256(payload).hexdigest()
    assert binding["device"] == path.stat().st_dev
    assert binding["inode"] == path.stat().st_ino
    assert binding["nlink"] == path.stat().st_nlink
    assert isinstance(binding["mtime_ns"], int)
    assert isinstance(binding["ctime_ns"], int)
    assert "birthtime_ns" in binding
    assert "path_ctime_ns" in binding


def test_digest_and_payload_use_one_descriptor_read(monkeypatch, tmp_path):
    path = tmp_path / "single-open.bin"
    payload = b"same descriptor"
    path.write_bytes(payload)
    real_open = evidence_io.os.open
    opened = []

    def counted_open(target, flags):
        opened.append(Path(target))
        return real_open(target, flags)

    monkeypatch.setattr(evidence_io.os, "open", counted_open)
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda self: (_ for _ in ()).throw(
            AssertionError("path was reopened for hashing or payload")
        ),
    )

    actual, binding = evidence_io.read_bound_bytes(path, "single open")

    assert actual == payload
    assert binding["sha256"] == hashlib.sha256(actual).hexdigest()
    assert opened == [path.absolute()]


def test_path_replacement_during_descriptor_read_is_rejected(
    monkeypatch, tmp_path
):
    path = tmp_path / "replace.bin"
    replacement = tmp_path / "replacement.bin"
    path.write_bytes(b"trusted")
    replacement.write_bytes(b"attacker")
    real_read = evidence_io.os.read
    replaced = False

    def replace_before_read(descriptor, size):
        nonlocal replaced
        if not replaced:
            replaced = True
            if os.name == "nt":
                # Windows can deny renaming an open file.  Model the same race
                # at the pathname boundary and let the rebound check catch it.
                original_lstat = evidence_io.os.lstat

                def rebound_lstat(target):
                    if Path(target) == path:
                        return original_lstat(replacement)
                    return original_lstat(target)

                monkeypatch.setattr(evidence_io.os, "lstat", rebound_lstat)
            else:
                os.replace(replacement, path)
        return real_read(descriptor, size)

    monkeypatch.setattr(evidence_io.os, "read", replace_before_read)

    with pytest.raises(
        ValueError,
        match="changed while reading|pathname changed",
    ):
        evidence_io.read_bound_bytes(path, "replace race")


def test_in_place_mutation_during_read_is_rejected(monkeypatch, tmp_path):
    path = tmp_path / "mutable.bin"
    path.write_bytes(b"a" * (evidence_io._READ_CHUNK_SIZE + 32))
    real_read = evidence_io.os.read
    calls = 0

    def mutate_after_first_chunk(descriptor, size):
        nonlocal calls
        chunk = real_read(descriptor, size)
        calls += 1
        if calls == 1:
            with path.open("r+b") as stream:
                stream.seek(evidence_io._READ_CHUNK_SIZE + 1)
                stream.write(b"z")
                stream.flush()
                os.fsync(stream.fileno())
        return chunk

    monkeypatch.setattr(evidence_io.os, "read", mutate_after_first_chunk)

    with pytest.raises(ValueError, match="changed while reading"):
        evidence_io.read_bound_bytes(path, "mutable")


def test_hard_link_policy_is_optional(tmp_path):
    path = tmp_path / "source.bin"
    link = tmp_path / "alias.bin"
    path.write_bytes(b"linked")
    os.link(path, link)

    payload, binding = evidence_io.read_bound_bytes(path, "linked")
    assert payload == b"linked"
    assert binding["nlink"] == 2

    with pytest.raises(ValueError, match="multiple hard links"):
        evidence_io.read_bound_bytes(
            path, "linked", reject_multiple_links=True
        )


def test_reparse_attribute_is_rejected_before_open(monkeypatch, tmp_path):
    path = tmp_path / "reparse.bin"
    path.write_bytes(b"x")
    real_lstat = evidence_io.os.lstat

    def marked_lstat(target):
        metadata = real_lstat(target)
        if Path(target) == path:
            return _StatProxy(
                metadata,
                st_file_attributes=evidence_io._REPARSE_POINT_ATTRIBUTE,
            )
        return metadata

    monkeypatch.setattr(evidence_io.os, "lstat", marked_lstat)
    monkeypatch.setattr(
        evidence_io.os,
        "open",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("reparse target was opened")
        ),
    )

    with pytest.raises(ValueError, match="symlink or reparse"):
        evidence_io.read_bound_bytes(path, "reparse")


def test_reparse_directory_component_is_rejected(monkeypatch, tmp_path):
    directory = tmp_path / "component"
    directory.mkdir()
    path = directory / "evidence.bin"
    path.write_bytes(b"x")
    real_lstat = evidence_io.os.lstat

    def marked_lstat(target):
        metadata = real_lstat(target)
        if Path(target) == directory:
            return _StatProxy(
                metadata,
                st_file_attributes=evidence_io._REPARSE_POINT_ATTRIBUTE,
            )
        return metadata

    monkeypatch.setattr(evidence_io.os, "lstat", marked_lstat)

    with pytest.raises(ValueError, match="symlink or reparse"):
        evidence_io.read_bound_bytes(path, "component")


if os.name != "nt":

    def test_symlink_file_and_directory_component_are_rejected(tmp_path):
        target = tmp_path / "target.bin"
        target.write_bytes(b"target")
        file_link = tmp_path / "file-link.bin"
        file_link.symlink_to(target)

        directory = tmp_path / "real-directory"
        directory.mkdir()
        nested = directory / "nested.bin"
        nested.write_bytes(b"nested")
        directory_link = tmp_path / "directory-link"
        directory_link.symlink_to(directory, target_is_directory=True)

        with pytest.raises(ValueError, match="symlink or reparse"):
            evidence_io.read_bound_bytes(file_link, "file link")
        with pytest.raises(ValueError, match="symlink or reparse"):
            evidence_io.read_bound_bytes(
                directory_link / "nested.bin", "directory link"
            )


def test_lexical_traversal_is_rejected(tmp_path):
    escaped = tmp_path / "outside.bin"
    escaped.write_bytes(b"x")
    traversal = tmp_path / "child" / ".." / "outside.bin"

    with pytest.raises(ValueError, match="lexical traversal"):
        evidence_io.read_bound_bytes(traversal, "traversal")


def test_json_rejects_duplicate_keys_at_any_depth(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"outer":{"same":1,"same":2}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate key: same"):
        evidence_io.load_bound_json(path, "duplicate")


def test_json_rejects_invalid_utf8(tmp_path):
    path = tmp_path / "encoding.json"
    path.write_bytes(b'{"value":"\xff"}')

    with pytest.raises(ValueError, match="UTF-8 JSON"):
        evidence_io.load_bound_json(path, "encoding")


def test_json_object_and_mapping_schema(tmp_path):
    path = tmp_path / "receipt.json"
    raw = b'{"count":3,"name":"run","items":[1,2]}'
    path.write_bytes(raw)

    payload, binding = evidence_io.load_bound_json(
        path,
        "receipt",
        require_object=True,
        schema={
            "count": int,
            "name": str,
            "items": lambda value: len(value) == 2,
        },
    )

    assert payload == {"count": 3, "name": "run", "items": [1, 2]}
    assert binding["sha256"] == hashlib.sha256(raw).hexdigest()


def test_json_schema_rejects_missing_or_invalid_values(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text(json.dumps({"count": "3"}), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid type: count"):
        evidence_io.load_bound_json(
            path, "schema", schema={"count": int}
        )
    with pytest.raises(ValueError, match="missing required key: name"):
        evidence_io.load_bound_json(
            path, "schema", schema={"name": str}
        )


def test_json_callable_schema_and_object_requirement(tmp_path):
    array_path = tmp_path / "array.json"
    array_path.write_text("[1,2,3]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        evidence_io.load_bound_json(
            array_path, "array", require_object=True
        )

    object_path = tmp_path / "object.json"
    object_path.write_text('{"ok":false}', encoding="utf-8")
    with pytest.raises(ValueError, match="does not satisfy its schema"):
        evidence_io.load_bound_json(
            object_path,
            "object",
            schema=lambda value: value.get("ok") is True,
        )


def test_non_regular_file_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="regular file"):
        evidence_io.read_bound_bytes(tmp_path, "directory")
