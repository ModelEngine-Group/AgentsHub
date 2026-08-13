from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from dki.nlp import cmeie_artifacts as artifacts
from dki.nlp.cmeie_artifacts import (
    CHECKPOINT_BUNDLE_MANIFEST,
    CHECKPOINT_TREE_HASH_ALGORITHM,
    CheckpointBundleError,
    load_checkpoint_bundle_manifest,
    publish_checkpoint_bundle_manifest,
    verify_checkpoint_bundle,
)


def _bundle(tmp_path: Path) -> Path:
    directory = tmp_path / "epoch-01"
    (directory / "encoder").mkdir(parents=True)
    (directory / "gplinker.pt").write_bytes(b"pointer-state")
    (directory / "config.json").write_text(
        '{"hidden_size":8}\n',
        encoding="utf-8",
    )
    (directory / "encoder" / "model.safetensors").write_bytes(
        b"encoder-state"
    )
    (directory / "tokenizer.json").write_text(
        '{"tokens":["a","b"]}\n',
        encoding="utf-8",
    )
    return directory


def _manual_tree(entries: list[dict]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def test_publish_and_verify_checkpoint_bundle_is_deterministic(
    tmp_path: Path,
):
    directory = _bundle(tmp_path)
    binding = publish_checkpoint_bundle_manifest(directory)
    payload = load_checkpoint_bundle_manifest(directory)

    assert payload["tree_hash_algorithm"] == CHECKPOINT_TREE_HASH_ALGORITHM
    assert payload["entries"] == sorted(
        payload["entries"],
        key=lambda entry: entry["path"],
    )
    assert payload["tree_sha256"] == _manual_tree(payload["entries"])
    assert binding["checkpoint_bundle_manifest"] == (
        CHECKPOINT_BUNDLE_MANIFEST
    )
    assert binding["checkpoint_bundle_tree_sha256"] == (
        payload["tree_sha256"]
    )
    assert binding["checkpoint_bundle_n_files"] == len(payload["entries"])
    assert binding == verify_checkpoint_bundle(
        directory,
        expected_manifest_sha256=binding[
            "checkpoint_bundle_manifest_sha256"
        ],
        expected_tree_sha256=binding["checkpoint_bundle_tree_sha256"],
        expected_checkpoint_sha256=binding["checkpoint_sha256"],
    )


def test_checkpoint_bundle_manifest_is_create_once(tmp_path: Path):
    directory = _bundle(tmp_path)
    first = publish_checkpoint_bundle_manifest(directory)

    with pytest.raises(CheckpointBundleError, match="already exists"):
        publish_checkpoint_bundle_manifest(directory)

    assert verify_checkpoint_bundle(directory) == first


@pytest.mark.parametrize(
    "relative",
    [
        "gplinker.pt",
        "encoder/model.safetensors",
        "tokenizer.json",
    ],
)
def test_checkpoint_bundle_rejects_payload_tampering(
    tmp_path: Path,
    relative: str,
):
    directory = _bundle(tmp_path)
    binding = publish_checkpoint_bundle_manifest(directory)
    (directory / Path(relative)).write_bytes(b"tampered")

    with pytest.raises(
        CheckpointBundleError,
        match="exact-file verification failed",
    ):
        verify_checkpoint_bundle(
            directory,
            expected_manifest_sha256=binding[
                "checkpoint_bundle_manifest_sha256"
            ],
            expected_tree_sha256=binding[
                "checkpoint_bundle_tree_sha256"
            ],
            expected_checkpoint_sha256=binding["checkpoint_sha256"],
        )


def test_checkpoint_bundle_rejects_missing_and_unexpected_files(
    tmp_path: Path,
):
    missing_directory = _bundle(tmp_path / "missing")
    missing_binding = publish_checkpoint_bundle_manifest(missing_directory)
    (missing_directory / "tokenizer.json").unlink()
    with pytest.raises(CheckpointBundleError, match="missing=tokenizer.json"):
        verify_checkpoint_bundle(
            missing_directory,
            expected_manifest_sha256=missing_binding[
                "checkpoint_bundle_manifest_sha256"
            ],
        )

    extra_directory = _bundle(tmp_path / "extra")
    extra_binding = publish_checkpoint_bundle_manifest(extra_directory)
    (extra_directory / "unrecorded.bin").write_bytes(b"extra")
    with pytest.raises(
        CheckpointBundleError,
        match="unexpected=unrecorded.bin",
    ):
        verify_checkpoint_bundle(
            extra_directory,
            expected_manifest_sha256=extra_binding[
                "checkpoint_bundle_manifest_sha256"
            ],
        )


def test_checkpoint_bundle_rejects_hard_linked_payload(tmp_path: Path):
    directory = _bundle(tmp_path)
    os.link(
        directory / "gplinker.pt",
        directory / "gplinker-alias.pt",
    )

    with pytest.raises(
        CheckpointBundleError,
        match="hard-linked file",
    ):
        publish_checkpoint_bundle_manifest(directory)


def test_checkpoint_bundle_rejects_manifest_tampering(tmp_path: Path):
    directory = _bundle(tmp_path)
    binding = publish_checkpoint_bundle_manifest(directory)
    manifest = directory / CHECKPOINT_BUNDLE_MANIFEST
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["kind"] = "changed"
    manifest.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CheckpointBundleError, match="manifest SHA-256"):
        verify_checkpoint_bundle(
            directory,
            expected_manifest_sha256=binding[
                "checkpoint_bundle_manifest_sha256"
            ],
        )
    with pytest.raises(CheckpointBundleError, match="kind is invalid"):
        verify_checkpoint_bundle(directory)


def test_checkpoint_bundle_rejects_empty_tree(tmp_path: Path):
    directory = tmp_path / "epoch-01"
    directory.mkdir()

    with pytest.raises(CheckpointBundleError, match="tree is empty"):
        publish_checkpoint_bundle_manifest(directory)


def test_checkpoint_manifest_rejects_casefold_alias(tmp_path: Path):
    directory = _bundle(tmp_path)
    publish_checkpoint_bundle_manifest(directory)
    manifest = directory / CHECKPOINT_BUNDLE_MANIFEST
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    original = payload["entries"][0]
    alias = {
        **original,
        "path": original["path"].swapcase(),
    }
    assert alias["path"].casefold() == original["path"].casefold()
    payload["entries"].append(alias)
    payload["n_files"] += 1
    payload["total_bytes"] += alias["size"]
    manifest.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CheckpointBundleError, match="casefold-alias"):
        load_checkpoint_bundle_manifest(directory)


if os.name == "posix":

    def test_checkpoint_bundle_rejects_real_symlink(tmp_path: Path):
        directory = _bundle(tmp_path)
        link = directory / "encoder-link"
        os.symlink(directory / "encoder", link, target_is_directory=True)

        with pytest.raises(
            CheckpointBundleError,
            match="symlink or reparse point",
        ):
            publish_checkpoint_bundle_manifest(directory)

else:

    def test_checkpoint_bundle_rejects_mocked_reparse_root(
        tmp_path: Path,
        monkeypatch,
    ):
        status = SimpleNamespace(st_file_attributes=0x400)
        assert artifacts._is_reparse_point(status)
        directory = _bundle(tmp_path)
        with monkeypatch.context() as context:
            context.setattr(
                artifacts,
                "_is_reparse_point",
                lambda _status: True,
            )
            with pytest.raises(
                CheckpointBundleError,
                match="root must be a real directory",
            ):
                publish_checkpoint_bundle_manifest(directory)
