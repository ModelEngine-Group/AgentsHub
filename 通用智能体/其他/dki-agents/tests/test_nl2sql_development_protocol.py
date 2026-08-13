import json
from pathlib import Path

import pytest

import dki.analysis.nl2sql_development_evidence as evidence


RID = "20260726T122144Z-g30fce24-t963fd4161f4f"
SOURCE_TREE_SHA256 = "a" * 64


def _project_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root


def _pair_paths(root: Path) -> tuple[Path, Path]:
    return (
        root / "outputs" / "private" / RID / "private.json",
        root / "outputs" / "remote_runs" / RID / "public.json",
    )


def _pair_payloads() -> tuple[dict[str, object], dict[str, object]]:
    return (
        {
            "schema_version": 1,
            "rid": RID,
            "kind": "synthetic_private_development_artifact",
        },
        {
            "schema_version": 1,
            "rid": RID,
            "kind": "synthetic_public_development_aggregate",
        },
    )


@pytest.mark.parametrize(
    "document",
    [
        b'{"field": 1, "field": 2}',
        b'{"field": NaN}',
        b'{"field": Infinity}',
        b'{"field": -Infinity}',
    ],
)
def test_strict_json_rejects_duplicate_and_nonfinite_values(document):
    with pytest.raises(ValueError, match="strict JSON"):
        evidence.strict_json_loads(document, label="synthetic receipt")


@pytest.mark.parametrize(
    "row",
    [
        b'{"id":"one","id":"two","question":"q","gold_sql":"SELECT 1","difficulty":"hard","category":"filtering"}\n',
        b'{"id":"one","question":"q","gold_sql":"SELECT NaN","difficulty":"hard","category":"filtering","score":NaN}\n',
        b'{"id":true,"question":"q","gold_sql":"SELECT 1","difficulty":"hard","category":"filtering"}\n',
        b'{"id":"one","question":1,"gold_sql":"SELECT 1","difficulty":"hard","category":"filtering"}\n',
    ],
)
def test_strict_development_jsonl_rejects_ambiguous_rows(row):
    with pytest.raises(ValueError):
        evidence.strict_development_jsonl_rows(row, label="synthetic development")


def test_unknown_public_receipt_is_rejected_before_private_artifacts(tmp_path):
    root = _project_root(tmp_path)
    input_path = root / "outputs" / "private" / RID / "development.jsonl"
    database_path = root / "data" / "development.sqlite"
    evidence_path = (
        root
        / "outputs"
        / "remote_runs"
        / RID
        / "nl2sql"
        / "development_projection_receipt.json"
    )
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_bytes(b"{}")

    with pytest.raises(ValueError, match="source-controlled allowlist"):
        with evidence.verified_evaluation_context(
            input_path=input_path,
            database_path=database_path,
            evidence_path=evidence_path,
            source_tree_sha256=SOURCE_TREE_SHA256,
            root=root,
        ):
            raise AssertionError("unapproved public evidence must not yield a context")

    assert not input_path.exists()
    assert not database_path.exists()


def test_caller_digest_authorization_is_rejected(tmp_path):
    root = _project_root(tmp_path)

    with pytest.raises(ValueError, match="caller-provided expected evidence"):
        with evidence.verified_evaluation_context(
            input_path=root / "outputs" / "private" / RID / "development.jsonl",
            database_path=root / "data" / "development.sqlite",
            evidence_path=(
                root
                / "outputs"
                / "remote_runs"
                / RID
                / "development_projection_receipt.json"
            ),
            source_tree_sha256=SOURCE_TREE_SHA256,
            root=root,
            expected_evidence_sha256="b" * 64,
        ):
            raise AssertionError("caller authority must not yield a context")


def test_direct_rid_output_paths_reject_nested_and_mismatched_rids(tmp_path):
    root = _project_root(tmp_path)
    private_path, public_path = _pair_paths(root)

    assert evidence.private_development_output_path(
        private_path,
        root=root,
        rid=RID,
    ) == private_path
    assert evidence.public_development_output_path(
        public_path,
        root=root,
        rid=RID,
    ) == public_path

    with pytest.raises(ValueError, match="directly below"):
        evidence.private_development_output_path(
            private_path.parent / "nested" / "private.json",
            root=root,
            rid=RID,
        )
    with pytest.raises(ValueError, match="directly below"):
        evidence.public_development_output_path(
            root / "outputs" / "remote_runs" / "different-rid" / "public.json",
            root=root,
            rid=RID,
        )


def test_output_path_rejects_protected_tokens_and_symlink_components(tmp_path, monkeypatch):
    root = _project_root(tmp_path)
    protected = root / "outputs" / "private" / RID / "blind" / "private.json"
    with pytest.raises(ValueError, match="forbidden path token"):
        evidence.private_development_output_path(protected, root=root, rid=RID)

    private_path, _ = _pair_paths(root)
    original_is_symlink = Path.is_symlink

    def synthetic_is_symlink(path: Path) -> bool:
        if Path(path) == private_path:
            return True
        return original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", synthetic_is_symlink)
    with pytest.raises(ValueError, match="must not use a symlink"):
        evidence.private_development_output_path(private_path, root=root, rid=RID)


def test_pair_publication_preserves_private_half_and_recovers(tmp_path, monkeypatch):
    root = _project_root(tmp_path)
    private_path, public_path = _pair_paths(root)
    private_payload, public_payload = _pair_payloads()
    original_link = evidence.os.link

    def fail_public_link(source, destination, *args, **kwargs):
        if Path(destination) == public_path:
            raise OSError("synthetic public link failure")
        return original_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(evidence.os, "link", fail_public_link)
    with pytest.raises(OSError, match="synthetic public link failure"):
        evidence.publish_development_json_pair_once(
            private_path=private_path,
            private_payload=private_payload,
            public_path=public_path,
            public_payload=public_payload,
            root=root,
            rid=RID,
        )

    private_bytes = private_path.read_bytes()
    assert private_path.is_file()
    assert not public_path.exists()

    monkeypatch.setattr(evidence.os, "link", original_link)
    assert evidence.recover_development_json_pair(
        private_path=private_path,
        public_path=public_path,
        root=root,
        rid=RID,
    )
    assert private_path.read_bytes() == private_bytes
    public_document = json.loads(public_path.read_text(encoding="utf-8"))
    assert public_document["rid"] == RID
    assert public_document["pairing"]["published_from_private"] is True


def test_pair_recovery_detects_private_mutation_after_public_link(tmp_path, monkeypatch):
    root = _project_root(tmp_path)
    private_path, public_path = _pair_paths(root)
    private_payload, public_payload = _pair_payloads()
    original_link = evidence.os.link

    def fail_public_link(source, destination, *args, **kwargs):
        if Path(destination) == public_path:
            raise OSError("synthetic public link failure")
        return original_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(evidence.os, "link", fail_public_link)
    with pytest.raises(OSError, match="synthetic public link failure"):
        evidence.publish_development_json_pair_once(
            private_path=private_path,
            private_payload=private_payload,
            public_path=public_path,
            public_payload=public_payload,
            root=root,
            rid=RID,
        )

    def mutate_private_during_public_link(source, destination, *args, **kwargs):
        if Path(destination) == public_path:
            private_path.write_bytes(b"synthetically mutated private artifact")
        return original_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(evidence.os, "link", mutate_private_during_public_link)
    with pytest.raises(ValueError, match="changed during pair publication"):
        evidence.recover_development_json_pair(
            private_path=private_path,
            public_path=public_path,
            root=root,
            rid=RID,
        )

    assert public_path.is_file()


def test_pair_payload_rid_must_match_its_direct_namespace(tmp_path):
    root = _project_root(tmp_path)
    private_path, public_path = _pair_paths(root)
    private_payload, public_payload = _pair_payloads()
    private_payload["rid"] = "different-rid"

    with pytest.raises(ValueError, match="private development payload RID"):
        evidence.publish_development_json_pair_once(
            private_path=private_path,
            private_payload=private_payload,
            public_path=public_path,
            public_payload=public_payload,
            root=root,
            rid=RID,
        )
