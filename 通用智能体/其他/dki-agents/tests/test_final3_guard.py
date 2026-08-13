"""Content-hash final3/blind guard: rejects protected content for consumer
purposes even after rename; allows innocent files; fails closed on missing
manifest."""

import json

import pytest

from dki.evidence import EvidenceValidationError
from dki.evidence.final3_guard import CONSUMER_PURPOSES, guard_dir, guard_path


def _manifest(tmp_path, protected_file):
    from dki.evidence import sha256_file
    manifest = {
        "schema_version": 1,
        "protected": [{
            "path": "data/ood_blind/final3_blind_eval.jsonl",
            "role": "historical_blind_read_only",
            "sha256": sha256_file(protected_file),
            "size": protected_file.stat().st_size,
            "forbidden_for": list(CONSUMER_PURPOSES),
        }],
    }
    path = tmp_path / "protected.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return path


def test_guard_rejects_protected_content_even_when_renamed(tmp_path):
    blind = tmp_path / "final3_blind_eval.jsonl"
    blind.write_text('{"text":"secret blind doc"}\n', encoding="utf-8")
    manifest = _manifest(tmp_path, blind)
    renamed = tmp_path / "totally_innocent_train.jsonl"
    renamed.write_text(blind.read_text(encoding="utf-8"), encoding="utf-8")  # same bytes, new name
    with pytest.raises(EvidenceValidationError, match="forbidden"):
        guard_path(renamed, "training", manifest_path=manifest)


def test_guard_allows_non_protected_file(tmp_path):
    blind = tmp_path / "final3_blind_eval.jsonl"
    blind.write_text('{"text":"secret"}\n', encoding="utf-8")
    manifest = _manifest(tmp_path, blind)
    ok = tmp_path / "train.jsonl"
    ok.write_text('{"text":"ordinary training doc"}\n', encoding="utf-8")
    guard_path(ok, "training", manifest_path=manifest)  # must not raise


def test_guard_dir_scans_all_matching_files(tmp_path):
    blind = tmp_path / "final3_blind_eval.jsonl"
    blind.write_text('{"text":"secret blind"}\n', encoding="utf-8")
    manifest = _manifest(tmp_path, blind)
    data_dir = tmp_path / "ood_train"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text('{"text":"ok"}\n', encoding="utf-8")
    (data_dir / "leaked.jsonl").write_text(blind.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(EvidenceValidationError):
        guard_dir(data_dir, "training", manifest_path=manifest)


def test_guard_unknown_purpose_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown consumer purpose"):
        guard_path(tmp_path / "x.jsonl", "sightseeing")


def test_guard_missing_manifest_fails_closed(tmp_path):
    victim = tmp_path / "train.jsonl"
    victim.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        guard_path(victim, "training", manifest_path=tmp_path / "nope.json")


def test_real_manifest_blocks_final3_for_all_consumers():
    """The shipped manifest must forbid final3 for every consumer purpose.

    The private payload is intentionally absent from clean source and release
    bundles.  Its absence must not turn the manifest-policy assertion into a
    skip; when the payload is mounted on a controlled host we additionally
    exercise the content-hash guard.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    manifest = root / "evidence" / "manifests" / "protected_baseline.json"
    blind = root / "data" / "ood_blind" / "final3_blind_eval.jsonl"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    protected = {
        item["path"]: item
        for item in payload.get("protected", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    policy = protected["data/ood_blind/final3_blind_eval.jsonl"]
    assert policy["role"] == "historical_blind_read_only"
    assert set(policy["forbidden_for"]) == set(CONSUMER_PURPOSES)
    assert len(policy["sha256"]) == 64
    assert policy["size"] > 0

    if blind.is_file():
        for purpose in CONSUMER_PURPOSES:
            with pytest.raises(EvidenceValidationError):
                guard_path(blind, purpose, manifest_path=manifest)
