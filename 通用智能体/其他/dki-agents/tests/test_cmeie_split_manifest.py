from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
import sys
import threading
from pathlib import Path

import pytest

from dki import evidence_io
from dki.nlp import cmeie_run_protocol as run_protocol


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_cmeie_split_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_cmeie_split_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

SOURCE_TREE_SHA256 = "a" * 64
SOURCE_TREE_MANIFEST_SHA256 = "b" * 64


def _replace_path_when_opened(
    monkeypatch,
    *,
    target: Path,
    replacement: Path,
):
    original_open = evidence_io.os.open
    target = Path(os.path.abspath(target))
    replaced = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal replaced
        candidate = Path(os.path.abspath(path))
        if not replaced and candidate == target:
            os.replace(replacement, target)
            replaced = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(evidence_io.os, "open", racing_open)


def _write_source_tree_manifest(path, tree_sha256=SOURCE_TREE_SHA256):
    raw = json.dumps(
        {
            "schema_version": 1,
            "included_tree_sha256": tree_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(raw)
    return raw


def _make_run_binding(tmp_path):
    head = "a" * 40
    fixture = b"x"
    entries = [
        {
            "path": "fixture.txt",
            "size": len(fixture),
            "sha256": hashlib.sha256(fixture).hexdigest(),
        }
    ]
    tree = run_protocol.included_tree_sha256(entries)
    rid = f"20260729T120000Z-g{head[:7]}-t{tree[:12]}"
    root = tmp_path / f"dki-agents-{rid}"
    source_root = root / "source" / "dki-agents"
    source_root.mkdir(parents=True)
    (source_root / "fixture.txt").write_bytes(fixture)
    source_manifest = (
        root / "evidence" / "source" / "included_tree.json"
    )
    source_manifest.parent.mkdir(parents=True)
    source_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rid": rid,
                "git_head": head,
                "algorithm": run_protocol.SOURCE_MANIFEST_ALGORITHM,
                "included_tree_sha256": tree,
                "entries": entries,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return run_protocol.load_run_binding(root)


def test_family_split_keeps_title_and_duplicate_text_together(tmp_path):
    rows = [
        {"text": "甲病@正文甲", "spo_list": [{"predicate": "P"}]},
        {"text": "甲病@另一段", "spo_list": [{"predicate": "P"}]},
        {"text": "无标题的共同前缀一二三四五六七八九十甲", "spo_list": []},
        {"text": "无标题的共同前缀一二三四五六七八九十乙", "spo_list": []},
    ] + [{"text": f"独立文本{i}", "spo_list": []} for i in range(20)]
    split = MODULE.build_split(rows, "20260725")
    owner = {index: name for name, indices in split.items() for index in indices}
    assert owner[0] == owner[1]
    assert owner[2] == owner[3]


def test_manifest_is_deterministic_and_has_no_intersections(tmp_path):
    rows = [{"text": f"标题{i}@内容{i}", "spo_list": [{"predicate": "P" if i % 2 else "Q"}]} for i in range(30)]
    source = tmp_path / "train.jsonl"
    source.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    source_tree_sha256 = "A" * 64
    source_tree_manifest_sha256 = "B" * 64
    first = MODULE.make_manifest(
        rows,
        source,
        "20260725",
        source_tree_sha256,
        source_tree_manifest_sha256,
    )
    second = MODULE.make_manifest(
        rows,
        source,
        "20260725",
        source_tree_sha256,
        source_tree_manifest_sha256,
    )
    assert first == second
    assert first["schema_version"] == 6
    assert first["source_tree_sha256"] == source_tree_sha256.lower()
    assert (
        first["source_tree_manifest_sha256"]
        == source_tree_manifest_sha256.lower()
    )
    assert all(not values for group in first["intersections"].values() for values in group.values())


def test_manifest_builder_rejects_nonfixed_seed(tmp_path):
    rows = [
        {"text": f"标题{i}@内容{i}", "spo_list": [{"predicate": "P"}]}
        for i in range(30)
    ]
    source = tmp_path / "train.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in rows
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="seed must be exactly"):
        MODULE.make_manifest(
            rows,
            source,
            "42",
            SOURCE_TREE_SHA256,
            SOURCE_TREE_MANIFEST_SHA256,
        )


def test_manifest_builder_self_validates_before_returning(
    tmp_path,
    monkeypatch,
):
    rows = [
        {"text": f"标题{i}@内容{i}", "spo_list": [{"predicate": "P"}]}
        for i in range(30)
    ]
    source = tmp_path / "train.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in rows
        ),
        encoding="utf-8",
    )
    calls = []

    def reject(payload, actual_rows, **kwargs):
        calls.append((payload, actual_rows, kwargs))
        raise ValueError("shared validator sentinel")

    monkeypatch.setattr(MODULE, "validate_split_manifest", reject)
    with pytest.raises(ValueError, match="shared validator sentinel"):
        MODULE.make_manifest(
            rows,
            source,
            "20260725",
            SOURCE_TREE_SHA256,
            SOURCE_TREE_MANIFEST_SHA256,
        )
    assert len(calls) == 1
    assert calls[0][1] == rows
    assert calls[0][1] is not rows
    assert calls[0][2]["_bound_source"][0] is calls[0][1]


def test_builder_rejects_source_hardlink(tmp_path):
    rows = [
        {"text": f"title{i}@text{i}", "spo_list": []}
        for i in range(3)
    ]
    source = tmp_path / "train.jsonl"
    source.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )
    os.link(source, tmp_path / "train-hardlink.jsonl")

    with pytest.raises(ValueError, match="multiple hard links"):
        MODULE.build_split_directory(
            source=source,
            output_dir=tmp_path / "split",
            seed="20260725",
            source_tree_sha256=SOURCE_TREE_SHA256,
            source_tree_manifest_sha256=SOURCE_TREE_MANIFEST_SHA256,
        )


def test_builder_rejects_source_path_replacement(tmp_path, monkeypatch):
    rows = [
        {"text": f"title{i}@text{i}", "spo_list": []}
        for i in range(3)
    ]
    source = tmp_path / "train.jsonl"
    source.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_bytes(source.read_bytes())
    _replace_path_when_opened(
        monkeypatch,
        target=source,
        replacement=replacement,
    )

    with pytest.raises(ValueError, match="changed before reading"):
        MODULE.build_split_directory(
            source=source,
            output_dir=tmp_path / "split",
            seed="20260725",
            source_tree_sha256=SOURCE_TREE_SHA256,
            source_tree_manifest_sha256=SOURCE_TREE_MANIFEST_SHA256,
        )


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (
            b'{"text":"first","text":"second","spo_list":[]}\n',
            "duplicate JSON object key",
        ),
        (b"\xff\n", "must be UTF-8"),
        (
            b'{"text":"one","spo_list":[]}\n'
            b"\n"
            b'{"text":"two","spo_list":[]}\n'
            b'{"text":"three","spo_list":[]}\n',
            "must not be blank",
        ),
    ],
)
def test_builder_rejects_noncanonical_source_jsonl(tmp_path, raw, message):
    source = tmp_path / "train.jsonl"
    source.write_bytes(raw)

    with pytest.raises(ValueError, match=message):
        MODULE.build_split_directory(
            source=source,
            output_dir=tmp_path / "split",
            seed="20260725",
            source_tree_sha256=SOURCE_TREE_SHA256,
            source_tree_manifest_sha256=SOURCE_TREE_MANIFEST_SHA256,
        )


@pytest.mark.parametrize(
    "source_tree_sha256",
    [
        "",
        "a" * 63,
        "a" * 65,
        "g" * 64,
        ("a" * 63) + " ",
        None,
    ],
)
def test_manifest_rejects_invalid_source_tree_sha256(tmp_path, source_tree_sha256):
    rows = [{"text": f"title{i}@text{i}", "spo_list": []} for i in range(3)]
    source = tmp_path / "train.jsonl"
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly 64 hexadecimal"):
        MODULE.make_manifest(
            rows,
            source,
            "20260725",
            source_tree_sha256,
            SOURCE_TREE_MANIFEST_SHA256,
        )


@pytest.mark.parametrize(
    "source_tree_manifest_hash",
    ["", "b" * 63, "b" * 65, "z" * 64, None],
)
def test_manifest_rejects_invalid_source_tree_manifest_sha256(
    tmp_path,
    source_tree_manifest_hash,
):
    rows = [{"text": f"title{i}@text{i}", "spo_list": []} for i in range(3)]
    source = tmp_path / "train.jsonl"
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="source_tree_manifest_sha256 must be exactly 64 hexadecimal",
    ):
        MODULE.make_manifest(
            rows,
            source,
            "20260725",
            SOURCE_TREE_SHA256,
            source_tree_manifest_hash,
        )


def test_cli_requires_source_tree_sha256(tmp_path, monkeypatch, capsys):
    source = tmp_path / "train.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--train",
            str(source),
            "--output-dir",
            str(tmp_path / "split-dir"),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        MODULE.main()
    assert exc_info.value.code == 2
    assert "--source-tree-sha256" in capsys.readouterr().err


def test_cli_requires_source_tree_manifest(tmp_path, monkeypatch, capsys):
    source = tmp_path / "train.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--train",
            str(source),
            "--output-dir",
            str(tmp_path / "split-dir"),
            "--source-tree-sha256",
            SOURCE_TREE_SHA256,
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        MODULE.main()
    assert exc_info.value.code == 2
    assert "--source-tree-manifest" in capsys.readouterr().err


def test_cli_requires_run_root(tmp_path, monkeypatch, capsys):
    source = tmp_path / "train.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--train",
            str(source),
            "--output-dir",
            str(tmp_path / "split"),
            "--source-tree-sha256",
            SOURCE_TREE_SHA256,
            "--source-tree-manifest",
            str(tmp_path / "included_tree.json"),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        MODULE.main()
    assert exc_info.value.code == 2
    assert "--run-root" in capsys.readouterr().err


def test_source_tree_manifest_hashes_exact_validated_bytes(tmp_path):
    path = tmp_path / "included_tree.json"
    raw = _write_source_tree_manifest(path, SOURCE_TREE_SHA256.upper())
    assert MODULE.source_tree_manifest_sha256(path, SOURCE_TREE_SHA256) == (
        MODULE.sha256_bytes(raw)
    )


def test_source_tree_manifest_rejects_tree_hash_mismatch(tmp_path):
    path = tmp_path / "included_tree.json"
    _write_source_tree_manifest(path, "1" * 64)
    with pytest.raises(ValueError, match="does not match"):
        MODULE.source_tree_manifest_sha256(path, "2" * 64)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"{", "valid UTF-8 JSON"),
        (b"[]", "root must be a JSON object"),
        (b'{"schema_version":2,"included_tree_sha256":"' + b"a" * 64 + b'"}', "schema_version must be 1"),
        (b'{"schema_version":1}', "missing included_tree_sha256"),
    ],
)
def test_source_tree_manifest_rejects_invalid_payload(tmp_path, raw, message):
    path = tmp_path / "included_tree.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match=message):
        MODULE.source_tree_manifest_sha256(path, SOURCE_TREE_SHA256)


def test_cli_publishes_only_canonical_run_bound_split_directory(
    tmp_path,
    monkeypatch,
):
    rows = [
        {"text": f"title{i}@text{i}", "spo_list": [{"predicate": "P"}]}
        for i in range(30)
    ]
    source = tmp_path / "train.jsonl"
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    binding = _make_run_binding(tmp_path)
    output = run_protocol.canonical_run_paths(binding).split_dir
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--train",
            str(source),
            "--run-root",
            str(binding.run_root),
            "--output-dir",
            str(output),
            "--source-tree-sha256",
            binding.included_tree_sha256,
            "--source-tree-manifest",
            str(binding.source_manifest),
        ],
    )
    assert MODULE.main() == 0
    fixed_files = (
        "split_manifest.json",
        "fit.jsonl",
        "calibration.jsonl",
        "outer.jsonl",
    )
    assert sorted(path.name for path in output.iterdir()) == sorted(
        fixed_files
    )
    payload = json.loads(
        (output / "split_manifest.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == 6
    assert payload["rid"] == binding.rid
    assert payload["git_head"] == binding.git_head
    assert payload["canonical_run_root"] == str(binding.run_root)
    assert (
        payload["source_tree_sha256"]
        == binding.included_tree_sha256
    )
    assert payload["source_tree_manifest_sha256"] == MODULE.sha256_bytes(
        binding.source_manifest.read_bytes()
    )


@pytest.mark.parametrize(
    "tamper",
    ["output_dir", "source_tree", "source_manifest"],
)
def test_cli_rejects_paths_or_hashes_outside_run_binding(
    tmp_path,
    monkeypatch,
    tamper,
):
    rows = [
        {"text": f"title{i}@text{i}", "spo_list": [{"predicate": "P"}]}
        for i in range(30)
    ]
    source = tmp_path / "train.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False) for row in rows
        ),
        encoding="utf-8",
    )
    binding = _make_run_binding(tmp_path)
    output = run_protocol.canonical_run_paths(binding).split_dir
    tree = binding.included_tree_sha256
    source_manifest = binding.source_manifest
    if tamper == "output_dir":
        output = tmp_path / "copied-split"
    elif tamper == "source_tree":
        tree = "0" * 64
    else:
        source_manifest = tmp_path / "copied-source-manifest.json"
        source_manifest.write_bytes(binding.source_manifest.read_bytes())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--train",
            str(source),
            "--run-root",
            str(binding.run_root),
            "--output-dir",
            str(output),
            "--source-tree-sha256",
            tree,
            "--source-tree-manifest",
            str(source_manifest),
        ],
    )

    with pytest.raises(SystemExit, match="run binding"):
        MODULE.main()


def test_manifest_binds_distinct_source_trees_without_changing_split(tmp_path):
    rows = [
        {"text": f"title{i}@text{i}", "spo_list": [{"predicate": "P"}]}
        for i in range(30)
    ]
    source = tmp_path / "train.jsonl"
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    first_hash = "1" * 64
    second_hash = "2" * 64
    first_manifest_hash = "3" * 64
    second_manifest_hash = "4" * 64
    first = MODULE.make_manifest(
        rows,
        source,
        "20260725",
        first_hash,
        first_manifest_hash,
    )
    second = MODULE.make_manifest(
        rows,
        source,
        "20260725",
        second_hash,
        second_manifest_hash,
    )
    assert first["source_tree_sha256"] == first_hash
    assert second["source_tree_sha256"] == second_hash
    assert first["source_tree_manifest_sha256"] == first_manifest_hash
    assert second["source_tree_manifest_sha256"] == second_manifest_hash
    assert first != second
    assert first["splits"] == second["splits"]


def test_split_realises_declared_80_10_10_ratio_for_small_families():
    rows = [
        {
            "text": f"唯一标题{i}@唯一内容{i}",
            "spo_list": [{"predicate": f"P{i % 4}"}],
        }
        for i in range(1000)
    ]
    split = MODULE.build_split(rows, "20260725")
    sizes = {name: len(indices) for name, indices in split.items()}
    assert abs(sizes["fit"] / len(rows) - 0.80) <= 0.01
    assert abs(sizes["calibration"] / len(rows) - 0.10) <= 0.01
    assert abs(sizes["outer"] / len(rows) - 0.10) <= 0.01


def test_split_directory_publication_refuses_existing_target_and_keeps_staging(
    tmp_path,
):
    staging = tmp_path / ".split.candidate.staging"
    staging.mkdir()
    (staging / "owner.txt").write_text("new", encoding="utf-8")
    target = tmp_path / "split"
    target.mkdir()
    (target / "owner.txt").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        MODULE._publish_directory_create_once(staging, target)

    assert (staging / "owner.txt").read_text(encoding="utf-8") == "new"
    assert (target / "owner.txt").read_text(encoding="utf-8") == "existing"


def test_split_directory_concurrent_publication_has_one_winner(tmp_path):
    target = tmp_path / "split"
    barrier = threading.Barrier(2)

    def publish(name):
        staging = tmp_path / f".split.{name}.staging"
        staging.mkdir()
        (staging / "owner.txt").write_text(name, encoding="utf-8")
        barrier.wait()
        try:
            MODULE._publish_directory_create_once(staging, target)
        except FileExistsError:
            return name, "lost", staging
        return name, "won", staging

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, ("one", "two")))

    assert sorted(status for _, status, _ in outcomes) == ["lost", "won"]
    winner = next(
        name for name, status, _ in outcomes if status == "won"
    )
    loser_staging = next(
        staging for _, status, staging in outcomes if status == "lost"
    )
    assert (target / "owner.txt").read_text(encoding="utf-8") == winner
    assert loser_staging.is_dir()


def test_split_build_validation_failure_preserves_complete_staging(
    tmp_path,
    monkeypatch,
):
    rows = [
        {"text": f"标题{i}@正文{i}", "spo_list": [{"predicate": "P"}]}
        for i in range(30)
    ]
    source = tmp_path / "train.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in rows
        ),
        encoding="utf-8",
    )
    target = tmp_path / "split"

    def reject(*args, **kwargs):
        raise ValueError("post-write validation sentinel")

    monkeypatch.setattr(MODULE, "load_and_validate_role_shard", reject)
    with pytest.raises(ValueError, match="post-write validation sentinel"):
        MODULE.build_split_directory(
            source=source,
            output_dir=target,
            seed="20260725",
            source_tree_sha256=SOURCE_TREE_SHA256,
            source_tree_manifest_sha256=SOURCE_TREE_MANIFEST_SHA256,
        )

    staging = [
        path
        for path in tmp_path.iterdir()
        if path.name.startswith(".split.") and path.name.endswith(".staging")
    ]
    assert len(staging) == 1
    assert sorted(path.name for path in staging[0].iterdir()) == [
        "calibration.jsonl",
        "fit.jsonl",
        "outer.jsonl",
        "split_manifest.json",
    ]
    assert not target.exists()
