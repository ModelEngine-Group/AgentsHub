import copy
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest

from dki import evidence_io
from dki.nlp import cmeie_split_protocol as protocol
from dki.nlp import cmeie_run_protocol as run_protocol


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts" / "build_cmeie_split_manifest.py"
SPEC = importlib.util.spec_from_file_location(
    "cmeie_split_builder_for_protocol_tests",
    BUILDER_PATH,
)
BUILDER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BUILDER)
TREE_SHA256 = "a" * 64
TREE_MANIFEST_SHA256 = "b" * 64


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


def _make_run_binding(base: Path, timestamp="20260729T120000Z"):
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
    rid = f"{timestamp}-g{head[:7]}-t{tree[:12]}"
    root = base / f"dki-agents-{rid}"
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


def _rows(n_rows: int = 30) -> list[dict]:
    return [
        {
            "text": f"唯一标题{i}@唯一正文{i}",
            "spo_list": [{"predicate": "P"}],
        }
        for i in range(n_rows)
    ]


def _fixture(tmp_path: Path, rows=None):
    rows = _rows() if rows is None else rows
    train = tmp_path / "train.jsonl"
    train.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = BUILDER.make_manifest(
        rows,
        train,
        protocol.FIXED_SEED,
        TREE_SHA256,
        TREE_MANIFEST_SHA256,
    )
    return rows, train, manifest


def _directory_fixture(tmp_path: Path, rows=None):
    rows = _rows() if rows is None else rows
    train = tmp_path / "train.jsonl"
    train.write_bytes(
        b"\r\n".join(
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            for row in rows
        )
    )
    output = tmp_path / "split-v6"
    manifest = BUILDER.build_split_directory(
        source=train,
        output_dir=output,
        seed=protocol.FIXED_SEED,
        source_tree_sha256=TREE_SHA256,
        source_tree_manifest_sha256=TREE_MANIFEST_SHA256,
    )
    return rows, train, output, manifest


def _run_directory_fixture(tmp_path: Path):
    rows = _rows()
    train = tmp_path / "train.jsonl"
    train.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in rows
        ),
        encoding="utf-8",
    )
    binding = _make_run_binding(tmp_path / "run-one")
    output = run_protocol.canonical_run_paths(binding).split_dir
    manifest = BUILDER.build_split_directory(
        source=train,
        output_dir=output,
        seed=protocol.FIXED_SEED,
        source_tree_sha256=binding.included_tree_sha256,
        source_tree_manifest_sha256=(
            binding.source_manifest_sha256
        ),
        run_binding=binding,
    )
    return rows, train, output, manifest, binding


def _validate(manifest, rows, train, **updates):
    arguments = {
        "train_path": train,
        "expected_source_tree_sha256": TREE_SHA256,
        "expected_source_tree_manifest_sha256": TREE_MANIFEST_SHA256,
        "generator_path": BUILDER_PATH,
    }
    arguments.update(updates)
    return protocol.validate_split_manifest(
        manifest,
        rows,
        **arguments,
    )


def test_full_validator_returns_deterministic_receipt_safe_binding(tmp_path):
    rows, train, manifest = _fixture(tmp_path)

    first = _validate(manifest, rows, train)
    second = _validate(copy.deepcopy(manifest), rows, train)

    assert first == second
    assert first["validated"] is True
    assert first["seed"] == protocol.FIXED_SEED
    assert first["n_rows"] == len(rows)
    assert set(first["split_n_rows"]) == set(protocol.SPLIT_NAMES)
    assert "indices" not in first
    assert first["validator_sha256"] == protocol.sha256_file(
        Path(protocol.__file__)
    )


def test_full_validator_rejects_rows_not_loaded_from_bound_train_file(
    tmp_path,
):
    rows, train, manifest = _fixture(tmp_path)
    rows[0]["text"] = "被替换的内存行"

    with pytest.raises(ValueError, match="differ from the current train file"):
        _validate(manifest, rows, train)


def test_full_validator_rejects_source_hardlink(tmp_path):
    rows, train, manifest = _fixture(tmp_path)
    os.link(train, tmp_path / "train-hardlink.jsonl")

    with pytest.raises(ValueError, match="multiple hard links"):
        _validate(manifest, rows, train)


def test_full_validator_rejects_source_path_replacement(
    tmp_path,
    monkeypatch,
):
    rows, train, manifest = _fixture(tmp_path)
    replacement = tmp_path / "replacement-train.jsonl"
    replacement.write_bytes(train.read_bytes())
    _replace_path_when_opened(
        monkeypatch,
        target=train,
        replacement=replacement,
    )

    with pytest.raises(ValueError, match="changed before reading"):
        _validate(manifest, rows, train)


def test_file_loader_binds_exact_bytes_and_rejects_duplicate_keys(tmp_path):
    rows, train, manifest = _fixture(tmp_path)
    path = tmp_path / "split.json"
    raw = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(raw)

    loaded, binding = protocol.load_and_validate_split_manifest(
        path,
        rows,
        train_path=train,
        expected_source_tree_sha256=TREE_SHA256,
        expected_source_tree_manifest_sha256=TREE_MANIFEST_SHA256,
        generator_path=BUILDER_PATH,
    )

    assert loaded == manifest
    assert binding["split_manifest_sha256"] == protocol.sha256_bytes(raw)

    duplicate = raw.replace(
        b'"seed":"20260725"',
        b'"seed":"20260725","seed":"20260725"',
        1,
    )
    path.write_bytes(duplicate)
    with pytest.raises(ValueError, match="duplicate key"):
        protocol.load_split_manifest(path)


def test_file_loader_rejects_nonfinite_json_number(tmp_path):
    path = tmp_path / "split.json"
    path.write_text(
        '{"schema_version":4,"realized_ratio":NaN}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite"):
        protocol.load_split_manifest(path)


def test_split_manifest_descriptor_rejects_hardlink(tmp_path):
    path = tmp_path / "split_manifest.json"
    path.write_text('{"schema_version":6}', encoding="utf-8")
    os.link(path, tmp_path / "manifest-hardlink.json")

    with pytest.raises(ValueError, match="multiple hard links"):
        protocol.load_split_manifest(path)


def test_split_manifest_descriptor_rejects_path_replacement(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "split_manifest.json"
    replacement = tmp_path / "replacement.json"
    path.write_bytes(b'{"schema_version":6}')
    replacement.write_bytes(b'{"schema_version":7}')
    assert path.stat().st_size == replacement.stat().st_size
    _replace_path_when_opened(
        monkeypatch,
        target=path,
        replacement=replacement,
    )

    with pytest.raises(ValueError, match="changed before reading"):
        protocol.load_split_manifest(path)


def test_train_jsonl_loader_rejects_duplicate_keys(tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text(
        '{"text":"甲","text":"乙","spo_list":[]}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        protocol.load_cmeie_rows(train)


def test_v6_shards_preserve_exact_selected_source_line_bytes(tmp_path):
    _, train, output, manifest = _directory_fixture(tmp_path)
    _, source_row_bytes, _ = protocol.load_cmeie_source(train)

    for role in protocol.SPLIT_NAMES:
        expected = b"".join(
            source_row_bytes[index]
            for index in manifest["splits"][role]["indices"]
        )
        assert (
            output / protocol.SHARD_FILENAMES[role]
        ).read_bytes() == expected


@pytest.mark.parametrize("role", ["fit", "calibration"])
def test_single_role_validation_never_opens_source_or_outer(
    tmp_path,
    monkeypatch,
    role,
):
    _, train, output, _ = _directory_fixture(tmp_path)
    forbidden = {
        train.absolute(),
        (output / protocol.SHARD_FILENAMES["outer"]).absolute(),
    }
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.absolute() in forbidden:
            raise AssertionError(f"forbidden data access: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    _, rows, binding = protocol.load_and_validate_role_shard(
        output / protocol.MANIFEST_FILENAME,
        output / protocol.SHARD_FILENAMES[role],
        role=role,
        expected_source_tree_sha256=TREE_SHA256,
        expected_source_tree_manifest_sha256=TREE_MANIFEST_SHA256,
        generator_path=BUILDER_PATH,
    )

    assert rows
    assert binding["source_train_accessed"] is False
    assert binding["outer_accessed"] is False
    assert binding["role"] == role


def test_single_role_validation_rejects_shard_byte_tamper(tmp_path):
    _, _, output, _ = _directory_fixture(tmp_path)
    fit = output / protocol.SHARD_FILENAMES["fit"]
    fit.write_bytes(fit.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="size differs|byte binding differs"):
        protocol.load_and_validate_role_shard(
            output / protocol.MANIFEST_FILENAME,
            fit,
            role="fit",
            expected_source_tree_sha256=TREE_SHA256,
            expected_source_tree_manifest_sha256=TREE_MANIFEST_SHA256,
            generator_path=BUILDER_PATH,
        )


def test_single_role_validation_rejects_shard_hardlink(tmp_path):
    _, _, output, _ = _directory_fixture(tmp_path)
    fit = output / protocol.SHARD_FILENAMES["fit"]
    os.link(fit, tmp_path / "fit-hardlink.jsonl")

    with pytest.raises(ValueError, match="multiple hard links"):
        protocol.load_and_validate_role_shard(
            output / protocol.MANIFEST_FILENAME,
            fit,
            role="fit",
            expected_source_tree_sha256=TREE_SHA256,
            expected_source_tree_manifest_sha256=TREE_MANIFEST_SHA256,
            generator_path=BUILDER_PATH,
        )


def test_single_role_validation_rejects_equal_size_shard_tamper(tmp_path):
    _, _, output, _ = _directory_fixture(tmp_path)
    fit = output / protocol.SHARD_FILENAMES["fit"]
    original = fit.read_bytes()
    tampered = original.replace(b'"predicate":"P"', b'"predicate":"Q"', 1)
    assert tampered != original
    assert len(tampered) == len(original)
    fit.write_bytes(tampered)

    with pytest.raises(ValueError, match="row_sha256 differs"):
        protocol.load_and_validate_role_shard(
            output / protocol.MANIFEST_FILENAME,
            fit,
            role="fit",
            expected_source_tree_sha256=TREE_SHA256,
            expected_source_tree_manifest_sha256=TREE_MANIFEST_SHA256,
            generator_path=BUILDER_PATH,
        )


def test_single_role_validation_rejects_duplicate_row_key(tmp_path):
    _, _, output, _ = _directory_fixture(tmp_path)
    manifest_path = output / protocol.MANIFEST_FILENAME
    fit = output / protocol.SHARD_FILENAMES["fit"]
    duplicate = fit.read_bytes().replace(
        b'{"text":',
        b'{"text":"shadow","text":',
        1,
    )
    assert duplicate != fit.read_bytes()
    fit.write_bytes(duplicate)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest["splits"]["fit"]["shard"]
    declared["size_bytes"] = len(duplicate)
    declared["sha256"] = protocol.sha256_bytes(duplicate)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        protocol.load_and_validate_role_shard(
            manifest_path,
            fit,
            role="fit",
            expected_source_tree_sha256=TREE_SHA256,
            expected_source_tree_manifest_sha256=TREE_MANIFEST_SHA256,
            generator_path=BUILDER_PATH,
        )


def test_single_role_validation_rejects_shard_path_replacement(
    tmp_path,
    monkeypatch,
):
    _, _, output, _ = _directory_fixture(tmp_path)
    fit = output / protocol.SHARD_FILENAMES["fit"]
    replacement = tmp_path / "replacement-fit.jsonl"
    replacement.write_bytes(fit.read_bytes())
    _replace_path_when_opened(
        monkeypatch,
        target=fit,
        replacement=replacement,
    )

    with pytest.raises(ValueError, match="changed before reading"):
        protocol.load_and_validate_role_shard(
            output / protocol.MANIFEST_FILENAME,
            fit,
            role="fit",
            expected_source_tree_sha256=TREE_SHA256,
            expected_source_tree_manifest_sha256=TREE_MANIFEST_SHA256,
            generator_path=BUILDER_PATH,
        )


def test_run_bound_full_and_role_validation_return_rid_binding(tmp_path):
    rows, train, output, manifest, binding = _run_directory_fixture(
        tmp_path
    )
    full = protocol.validate_split_manifest(
        manifest,
        rows,
        train_path=train,
        expected_source_tree_sha256=binding.included_tree_sha256,
        expected_source_tree_manifest_sha256=(
            binding.source_manifest_sha256
        ),
        generator_path=BUILDER_PATH,
        expected_run_binding=binding,
    )
    _, fit, role = protocol.load_and_validate_role_shard(
        output / protocol.MANIFEST_FILENAME,
        output / protocol.SHARD_FILENAMES["fit"],
        role="fit",
        expected_source_tree_sha256=binding.included_tree_sha256,
        expected_source_tree_manifest_sha256=(
            binding.source_manifest_sha256
        ),
        generator_path=BUILDER_PATH,
        expected_run_binding=binding,
    )

    assert fit
    for receipt in (full, role):
        assert receipt["rid"] == binding.rid
        assert receipt["git_head"] == binding.git_head
        assert receipt["canonical_run_root"] == str(binding.run_root)


@pytest.mark.parametrize("allow_same_parent_staging", [False, True])
def test_role_validator_rejects_copied_split_outside_canonical_path(
    tmp_path,
    allow_same_parent_staging,
):
    _, _, output, _, binding = _run_directory_fixture(tmp_path)
    copied = tmp_path / "copied-split"
    shutil.copytree(output, copied)

    with pytest.raises(ValueError, match="canonical split directory"):
        protocol.load_and_validate_role_shard(
            copied / protocol.MANIFEST_FILENAME,
            copied / protocol.SHARD_FILENAMES["fit"],
            role="fit",
            expected_source_tree_sha256=binding.included_tree_sha256,
            expected_source_tree_manifest_sha256=(
                binding.source_manifest_sha256
            ),
            generator_path=BUILDER_PATH,
            expected_run_binding=binding,
            allow_same_parent_staging=allow_same_parent_staging,
        )


def test_role_validator_rejects_noncanonical_path_before_open(
    tmp_path,
    monkeypatch,
):
    _, _, output, _, binding = _run_directory_fixture(tmp_path)
    copied = tmp_path / "copied-split"
    shutil.copytree(output, copied)

    def forbidden_open(*args, **kwargs):
        raise AssertionError("manifest was opened before path rejection")

    monkeypatch.setattr(protocol, "load_bound_json", forbidden_open)
    with pytest.raises(ValueError, match="canonical split directory"):
        protocol.load_and_validate_role_shard(
            copied / protocol.MANIFEST_FILENAME,
            copied / protocol.SHARD_FILENAMES["fit"],
            role="fit",
            expected_source_tree_sha256=binding.included_tree_sha256,
            expected_source_tree_manifest_sha256=(
                binding.source_manifest_sha256
            ),
            generator_path=BUILDER_PATH,
            expected_run_binding=binding,
        )


def test_role_validator_rejects_manifest_replayed_into_another_rid(
    tmp_path,
):
    _, _, output, _, first = _run_directory_fixture(tmp_path)
    second = _make_run_binding(
        tmp_path / "run-two",
        timestamp="20260729T120001Z",
    )
    second_split = run_protocol.canonical_run_paths(second).split_dir
    shutil.copytree(output, second_split)

    with pytest.raises(ValueError, match="expected RID"):
        protocol.load_and_validate_role_shard(
            second_split / protocol.MANIFEST_FILENAME,
            second_split / protocol.SHARD_FILENAMES["fit"],
            role="fit",
            expected_source_tree_sha256=second.included_tree_sha256,
            expected_source_tree_manifest_sha256=(
                second.source_manifest_sha256
            ),
            generator_path=BUILDER_PATH,
            expected_run_binding=second,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.__setitem__("seed", "42"),
            "seed must be exactly",
        ),
        (
            lambda payload: payload.__setitem__(
                "generator_sha256",
                "0" * 64,
            ),
            "generator hash differs",
        ),
        (
            lambda payload: payload.__setitem__("unexpected", 1),
            "fields differ from the v6 schema",
        ),
        (
            lambda payload: payload.pop("grouping"),
            "fields differ from the v6 schema",
        ),
        (
            lambda payload: payload["splits"]["fit"].__setitem__(
                "realized_ratio",
                float("nan"),
            ),
            "non-finite",
        ),
    ],
)
def test_full_validator_rejects_false_root_and_numeric_evidence(
    tmp_path,
    mutation,
    message,
):
    rows, train, manifest = _fixture(tmp_path)
    mutation(manifest)
    with pytest.raises(ValueError, match=message):
        _validate(manifest, rows, train)


def test_full_validator_rejects_overlap_even_with_forged_empty_intersections(
    tmp_path,
):
    rows, train, manifest = _fixture(tmp_path)
    shared = manifest["splits"]["calibration"]["indices"][0]
    manifest["splits"]["fit"]["indices"].append(shared)
    manifest["splits"]["fit"]["indices"].sort()
    manifest["intersections"] = {
        "families": {
            key: [] for key in protocol.INTERSECTION_KEYS
        },
        "normalized_text_hashes": {
            key: [] for key in protocol.INTERSECTION_KEYS
        },
    }

    with pytest.raises(ValueError, match="split indices overlap"):
        _validate(manifest, rows, train)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (False, "non-boolean integers"),
        (-1, "out-of-range"),
        (10_000, "out-of-range"),
    ],
)
def test_full_validator_rejects_boolean_and_out_of_range_indices(
    tmp_path,
    replacement,
    message,
):
    rows, train, manifest = _fixture(tmp_path)
    manifest["splits"]["fit"]["indices"][0] = replacement
    with pytest.raises(ValueError, match=message):
        _validate(manifest, rows, train)


def test_full_validator_rejects_unsorted_duplicate_and_incomplete_indices(
    tmp_path,
):
    rows, train, original = _fixture(tmp_path)

    unsorted = copy.deepcopy(original)
    unsorted_indices = unsorted["splits"]["fit"]["indices"]
    unsorted_indices[0], unsorted_indices[1] = (
        unsorted_indices[1],
        unsorted_indices[0],
    )
    with pytest.raises(ValueError, match="must be sorted"):
        _validate(unsorted, rows, train)

    duplicate = copy.deepcopy(original)
    duplicate["splits"]["fit"]["indices"].insert(
        0,
        duplicate["splits"]["fit"]["indices"][0],
    )
    with pytest.raises(ValueError, match="must be unique"):
        _validate(duplicate, rows, train)

    incomplete = copy.deepcopy(original)
    incomplete["splits"]["fit"]["indices"].pop()
    with pytest.raises(ValueError, match="exactly cover"):
        _validate(incomplete, rows, train)


def test_full_validator_rejects_forged_intersection_evidence(tmp_path):
    rows, train, manifest = _fixture(tmp_path)
    manifest["intersections"]["families"]["fit:outer"] = ["forged"]

    with pytest.raises(ValueError, match="intersections do not match"):
        _validate(manifest, rows, train)


def test_full_validator_rejects_boolean_predicate_count(tmp_path):
    rows, train, manifest = _fixture(tmp_path)
    manifest["splits"]["calibration"]["predicate_counts"]["P"] = True

    with pytest.raises(ValueError, match="positive non-boolean integers"):
        _validate(manifest, rows, train)


@pytest.mark.parametrize(
    "path",
    [
        ("splits", "fit"),
        ("intersections", "families"),
        ("predicate_coverage_policy",),
    ],
)
def test_full_validator_rejects_extra_nested_fields(tmp_path, path):
    rows, train, manifest = _fixture(tmp_path)
    target = manifest
    for component in path:
        target = target[component]
    target["unexpected"] = "forged"

    with pytest.raises(ValueError, match="fields differ from the v6 schema"):
        _validate(manifest, rows, train)


@pytest.mark.parametrize(
    "field",
    [
        "n_rows",
        "target_ratio",
        "realized_ratio",
        "absolute_ratio_error",
        "row_sha256",
        "family_sha256",
        "normalized_text_sha256",
        "predicate_counts",
    ],
)
def test_full_validator_recomputes_every_split_evidence_field(
    tmp_path,
    field,
):
    rows, train, manifest = _fixture(tmp_path)
    current = manifest["splits"]["fit"][field]
    manifest["splits"]["fit"][field] = (
        {"forged": 1}
        if isinstance(current, dict)
        else "forged"
        if isinstance(current, str)
        else current + 1
    )

    with pytest.raises(ValueError):
        _validate(manifest, rows, train)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("relative_path", "wrong.jsonl"),
        ("n_rows", 999),
        ("size_bytes", 0),
        ("sha256", "0" * 64),
    ],
)
def test_full_validator_recomputes_every_shard_binding(
    tmp_path,
    field,
    value,
):
    rows, train, manifest = _fixture(tmp_path)
    manifest["splits"]["fit"]["shard"][field] = value

    with pytest.raises(ValueError, match="shard evidence"):
        _validate(manifest, rows, train)


def test_full_validator_recomputes_supported_predicate_coverage(tmp_path):
    rows = _rows(30)
    rows, train, manifest = _fixture(tmp_path, rows)
    malicious_split = {
        "fit": list(range(0, 9)) + list(range(10, 20)),
        "calibration": [9],
        "outer": list(range(20, 30)),
    }
    malicious_split = {
        name: sorted(indices)
        for name, indices in malicious_split.items()
    }
    evidence, intersections, supported = (
        protocol.compute_manifest_evidence(rows, malicious_split)
    )
    assert supported == {"P"}
    assert evidence["outer"]["predicate_counts"] == {"P": 10}

    # Make the last ten rows unlabelled while keeping P sufficiently supported
    # by the first twenty source families, then recompute an internally
    # consistent but protocol-invalid split with no P in outer.
    for index in range(20, 30):
        rows[index]["spo_list"] = []
    train.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    manifest["source_train_sha256"] = protocol.sha256_file(train)
    evidence, intersections, supported = (
        protocol.compute_manifest_evidence(rows, malicious_split)
    )
    assert supported == {"P"}
    assert evidence["outer"]["predicate_counts"] == {}
    _, source_row_bytes, _ = protocol.load_cmeie_source(train)
    shards = protocol.build_shard_bytes(
        source_row_bytes,
        malicious_split,
    )
    for role in protocol.SPLIT_NAMES:
        evidence[role]["shard"] = protocol.shard_evidence(
            role,
            shards[role],
            len(malicious_split[role]),
        )
    manifest["splits"] = evidence
    manifest["intersections"] = intersections

    with pytest.raises(
        ValueError,
        match="sufficiently supported predicates are absent",
    ):
        _validate(manifest, rows, train)


def test_three_row_boundary_is_valid_and_deterministic(tmp_path):
    rows = [
        {"text": f"边界标题{i}@正文{i}", "spo_list": []}
        for i in range(3)
    ]
    rows, train, first = _fixture(tmp_path, rows)
    second = BUILDER.make_manifest(
        rows,
        train,
        protocol.FIXED_SEED,
        TREE_SHA256,
        TREE_MANIFEST_SHA256,
    )

    assert first == second
    binding = _validate(first, rows, train)
    assert binding["n_rows"] == 3
    assert sum(binding["split_n_rows"].values()) == 3
