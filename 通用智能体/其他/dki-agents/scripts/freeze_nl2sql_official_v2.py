"""One-shot CSS + EHRSQL official-test projection for the NL2SQL v2 blind set.

The official test files are deliberately not opened or hashed during
preflight.  A create-once ``projection-started.json`` marker is published
first.  Any failure after that point permanently invalidates the RID and is
recorded without exposing questions, SQL, or failing examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.analysis.database import database_matches_manifest
from dki.analysis.nl2sql_css_projection import (
    PROJECTION_SEED_V2,
    ProjectedItem,
    ProjectionRejected,
    load_css_schemas,
    load_target_domains,
    project_css_row,
)
from dki.analysis.nl2sql_ehrsql_projection import (
    EHRProjectionRejected,
    load_ehr_target_domains,
    project_ehrsql_row,
)
from dki.analysis.nl2sql_protocol import (
    ALLOWED_CATEGORIES,
    ALLOWED_DIFFICULTIES,
    normalise_question,
    validate_gold_sql,
)
from dki.analysis.nl2sql_seal import (
    _validate_development_gate,
    _validate_regression_gate,
    _verify_source_tree_manifest,
)
from dki.pipeline.artifacts import validate_task2_artifact


EHR_DATABASES = ("eicu", "mimic_iii")
HARD_DIFFICULTIES = {"hard", "extra", "extra-hard"}
RID_PATTERN = re.compile(
    r"^\d{8}T\d{6}Z-g(?P<head>[0-9a-f]{7})-t(?P<tree>[0-9a-f]{12})$"
)
UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                dict(row),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(
                        f"{label} row {line_number} must be a JSON object"
                    )
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    return rows


def _stage_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)


def _publish_bytes_once(path: Path, data: bytes) -> None:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"create-once artifact already exists: {path}")
    temporary = _stage_bytes(path, data)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    _publish_bytes_once(path, _json_bytes(payload))


def _write_bytes_synced(path: Path, data: bytes) -> None:
    with Path(path).open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _copy_file_synced(source: Path, destination: Path) -> None:
    with Path(source).open("rb") as reader, Path(destination).open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def _bound_path(base: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe source-manifest path: {relative}")
    return Path(base).joinpath(*rel.parts)


def _binding_entry(binding: Any, label: str) -> dict[str, Any]:
    if (
        not isinstance(binding, Mapping)
        or not re.fullmatch(r"[0-9a-f]{64}", str(binding.get("sha256", "")))
        or not isinstance(binding.get("size"), int)
        or int(binding["size"]) < 0
    ):
        raise ValueError(f"{label} has an invalid frozen binding")
    return {
        "sha256": str(binding["sha256"]),
        "size": int(binding["size"]),
    }


def _verify_bound_nonholdout(
    path: Path,
    binding: Mapping[str, Any],
    label: str,
) -> Path:
    path = Path(path).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(binding["size"])
        or _sha256(path) != str(binding["sha256"])
    ):
        raise ValueError(f"{label} binding failed")
    return path


def _preflight_unopened_official_file(
    path: Path,
    binding: Mapping[str, Any],
    label: str,
) -> Path:
    """Check existence and frozen byte count without opening or hashing a test."""
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size != int(binding["size"]):
        raise ValueError(f"{label} is absent or differs in size from its freeze")
    return path


def _read_official_json_after_marker(
    path: Path,
    binding: Mapping[str, Any],
    marker_path: Path,
    label: str,
) -> list[dict[str, Any]]:
    """First content access for an official test; marker existence is mandatory."""
    if not Path(marker_path).is_file():
        raise RuntimeError("official test access attempted before projection marker")
    path = Path(path).resolve()
    raw = path.read_bytes()
    if (
        len(raw) != int(binding["size"])
        or hashlib.sha256(raw).hexdigest() != str(binding["sha256"])
    ):
        raise RuntimeError(f"{label} hash differs from the predeclared freeze")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, list) or any(
        not isinstance(row, dict) for row in payload
    ):
        raise RuntimeError(f"{label} must be a JSON list of objects")
    return payload


def _scope_rid(path: Path, scope: tuple[str, str]) -> str:
    parts = Path(path).resolve().parts
    for index in range(len(parts) - 2):
        if parts[index : index + 2] == scope:
            return parts[index + 2]
    raise ValueError(f"path is not below {'/'.join(scope)}/<RID>: {path}")


def _validate_output_layout(
    *,
    output_manifest: Path,
    public_output: Path,
    source_tree_manifest: Path,
    development_path: Path,
    development_projection_receipt: Path,
    development_evaluation_receipt: Path,
    regression_receipt: Path,
) -> tuple[str, Path, Path, Path, Path]:
    output_manifest = Path(output_manifest).resolve()
    public_output = Path(public_output).resolve()
    if (
        output_manifest.name != "freeze_manifest.json"
        or output_manifest.parent.name != "frozen_set"
        or output_manifest.parent.parent.name != "nl2sql_blind_v2"
    ):
        raise ValueError(
            "official-v2 output must be "
            "outputs/private/<RID>/nl2sql_blind_v2/frozen_set/"
            "freeze_manifest.json"
        )
    rid = _scope_rid(output_manifest, ("outputs", "private"))
    match = RID_PATTERN.fullmatch(rid)
    if not match:
        raise ValueError("output RID does not follow timestamp/head/tree form")
    bound_paths = (
        (public_output, ("outputs", "remote_runs")),
        (source_tree_manifest, ("outputs", "remote_runs")),
        (development_path, ("outputs", "private")),
        (development_projection_receipt, ("outputs", "remote_runs")),
        (development_evaluation_receipt, ("outputs", "private")),
        (regression_receipt, ("outputs", "private")),
    )
    for path, scope in bound_paths:
        if _scope_rid(path, scope) != rid:
            raise ValueError(f"RID differs across official-v2 artifacts: {path}")
    control_root = output_manifest.parent.parent
    started = control_root / "projection-started.json"
    invalid = control_root / "projection-invalid.json"
    completed = control_root / "projection-completed.json"
    occupied = [
        path
        for path in (
            output_manifest.parent,
            public_output,
            started,
            invalid,
            completed,
        )
        if path.exists()
    ]
    staging = (
        list(control_root.glob(".frozen_set.*.freeze-staging"))
        if control_root.is_dir()
        else []
    )
    if occupied or staging:
        raise FileExistsError(
            "official-v2 RID is already started, completed, invalid, or occupied: "
            + ", ".join(str(path) for path in [*occupied, *staging])
        )
    return rid, control_root, started, invalid, completed


def _validate_css_freeze(
    manifest_path: Path,
) -> tuple[dict[str, Any], Path, Path, dict[str, Any], dict[str, Any]]:
    manifest = _load_json(manifest_path, "CSS freeze manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind")
        != "external_authoritative_nl2sql_benchmark_freeze"
        or manifest.get("benchmark") != "CSS"
        or manifest.get("official_split") != "schema"
        or manifest.get("holdout_consumed") is not False
        or manifest.get("selection_used") is not False
    ):
        raise ValueError("CSS manifest is not an unconsumed official schema freeze")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("CSS freeze lacks file bindings")
    test_binding = _binding_entry(
        files.get("holdout/test.json"),
        "CSS official test",
    )
    tables_binding = _binding_entry(
        files.get("metadata/tables.json"),
        "CSS schema metadata",
    )
    base = Path(manifest_path).resolve().parent
    tables_path = _verify_bound_nonholdout(
        _bound_path(base, "metadata/tables.json"),
        tables_binding,
        "CSS schema metadata",
    )
    test_path = _preflight_unopened_official_file(
        _bound_path(base, "holdout/test.json"),
        test_binding,
        "CSS official test",
    )
    return manifest, base, test_path, test_binding, load_css_schemas(tables_path)


def _validate_ehr_audit(
    root: Path,
    audit_path: Path,
) -> tuple[
    dict[str, Any],
    tuple[tuple[str, Path, dict[str, Any]], ...],
]:
    audit = _load_json(audit_path, "EHRSQL source audit")
    if (
        audit.get("schema_version") != 1
        or audit.get("kind") != "ehrsql_public_repository_structure_audit"
        or audit.get("benchmark") != "EHRSQL-v1.5.1"
        or audit.get("contains_questions_or_sql") is not False
        or not re.fullmatch(
            r"[0-9a-f]{40}",
            str(audit.get("source_revision", "")),
        )
    ):
        raise ValueError("EHRSQL audit is not the pinned aggregate-only source")
    files = audit.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("EHRSQL audit lacks file bindings")
    root = Path(root).resolve()
    tests: list[tuple[str, Path, dict[str, Any]]] = []
    for database in EHR_DATABASES:
        relative = f"dataset/ehrsql/{database}/test.json"
        binding = _binding_entry(files.get(relative), f"EHRSQL {database} test")
        path = _preflight_unopened_official_file(
            _bound_path(root, relative),
            binding,
            f"EHRSQL {database} official test",
        )
        tests.append((database, path, binding))
    for relative in ("dataset/ehrsql/tables.json", "LICENSE", "README.md"):
        binding = _binding_entry(files.get(relative), f"EHRSQL {relative}")
        _verify_bound_nonholdout(
            _bound_path(root, relative),
            binding,
            f"EHRSQL {relative}",
        )
    return audit, tuple(tests)


def _validate_development_rows(
    development_path: Path,
) -> tuple[list[dict[str, Any]], frozenset[str], frozenset[str]]:
    rows = _load_jsonl(development_path, "official-v2 development projection")
    if len(rows) < 150:
        raise ValueError("official-v2 development projection has fewer than 150 rows")
    ids: set[str] = set()
    questions: set[str] = set()
    skeletons: set[str] = set()
    required = {"id", "question", "gold_sql", "difficulty", "category"}
    for index, row in enumerate(rows, start=1):
        if required - set(row) or any(
            not str(row.get(key, "")).strip() for key in required
        ):
            raise ValueError(f"development row {index} lacks required fields")
        identity = str(row["id"])
        question = normalise_question(str(row["question"]))
        if identity in ids or question in questions:
            raise ValueError("development projection has duplicate id or question")
        if str(row["difficulty"]).lower() not in ALLOWED_DIFFICULTIES:
            raise ValueError("development projection has invalid difficulty")
        if str(row["category"]).lower() not in ALLOWED_CATEGORIES:
            raise ValueError("development projection has invalid category")
        skeleton = validate_gold_sql(str(row["gold_sql"]))
        skeleton_sha256 = hashlib.sha256(skeleton.encode("utf-8")).hexdigest()
        provenance = row.get("provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("partition_version") != "v2"
            or provenance.get("skeleton_role") != "development"
            or provenance.get("skeleton_sha256") != skeleton_sha256
        ):
            raise ValueError("development projection has invalid v2 provenance")
        ids.add(identity)
        questions.add(question)
        skeletons.add(skeleton_sha256)
    return rows, frozenset(questions), frozenset(skeletons)


def _validate_projection_receipt(
    *,
    receipt_path: Path,
    source_tree_sha256: str,
    css_manifest_path: Path,
    css_manifest: Mapping[str, Any],
    ehr_audit_path: Path,
    ehr_audit: Mapping[str, Any],
    db_path: Path,
    development_path: Path,
    development_count: int,
) -> dict[str, Any]:
    receipt = _load_json(receipt_path, "official-v2 development receipt")
    hashes = receipt.get("hashes")
    selected = receipt.get("development_selected")
    official_tests = receipt.get("official_tests")
    partition = receipt.get("partition")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind")
        != "official_public_v2_four_table_development_receipt"
        or receipt.get("dataset_role") != "development_selection_used"
        or receipt.get("source_tree_sha256") != source_tree_sha256
        or not isinstance(hashes, Mapping)
        or not isinstance(selected, Mapping)
        or int(selected.get("n", 0)) != development_count
        or not isinstance(official_tests, Mapping)
        or official_tests.get("content_parsed_by_this_builder") is not False
        or official_tests.get("consumed") is not False
        or not isinstance(partition, Mapping)
        or partition.get("version") != "v2"
        or partition.get("seed") != PROJECTION_SEED_V2
        or partition.get("development_holdout_skeleton_intersection") != 0
    ):
        raise ValueError("development projection receipt violates the v2 contract")
    expected_hashes = {
        "builder_sha256": _sha256(
            ROOT / "scripts" / "project_nl2sql_official_v2_development.py"
        ),
        "css_projection_module_sha256": _sha256(
            ROOT / "dki" / "analysis" / "nl2sql_css_projection.py"
        ),
        "ehrsql_projection_module_sha256": _sha256(
            ROOT / "dki" / "analysis" / "nl2sql_ehrsql_projection.py"
        ),
        "protocol_sha256": _sha256(
            ROOT / "dki" / "analysis" / "nl2sql_protocol.py"
        ),
        "css_freeze_manifest_sha256": _sha256(css_manifest_path),
        "ehrsql_audit_sha256": _sha256(ehr_audit_path),
        "target_db_sha256": _sha256(db_path),
        "private_development_sha256": _sha256(development_path),
    }
    if any(hashes.get(key) != value for key, value in expected_hashes.items()):
        raise ValueError("development projection receipt hash binding failed")
    expected_tests = {
        "CSS/test": css_manifest["files"]["holdout/test.json"],
        "EHRSQL/eicu/test": ehr_audit["files"][
            "dataset/ehrsql/eicu/test.json"
        ],
        "EHRSQL/mimic_iii/test": ehr_audit["files"][
            "dataset/ehrsql/mimic_iii/test.json"
        ],
    }
    if official_tests.get("files") != expected_tests:
        raise ValueError("development receipt official-test declarations differ")
    privacy = receipt.get("privacy")
    if (
        not isinstance(privacy, Mapping)
        or privacy.get("public_receipt_contains_questions") is not False
        or privacy.get("public_receipt_contains_gold_sql") is not False
        or privacy.get("public_receipt_contains_error_samples") is not False
        or privacy.get("private_projection_allowed_in_submission") is not False
    ):
        raise ValueError("development projection privacy declaration is invalid")
    return receipt


def _preflight_freeze(
    *,
    css_freeze_manifest: Path,
    ehrsql_root: Path,
    ehrsql_audit: Path,
    db_path: Path,
    task1_manifest: Path,
    kg_path: Path,
    development_path: Path,
    development_projection_receipt: Path,
    development_evaluation_receipt: Path,
    regression_receipt: Path,
    source_tree_manifest: Path,
    output_manifest: Path,
    public_output: Path,
    created_at: str,
) -> dict[str, Any]:
    if not UTC_PATTERN.fullmatch(created_at):
        raise ValueError("created-at must be UTC RFC3339 without fractional seconds")
    rid, control_root, started, invalid, completed = _validate_output_layout(
        output_manifest=output_manifest,
        public_output=public_output,
        source_tree_manifest=source_tree_manifest,
        development_path=development_path,
        development_projection_receipt=development_projection_receipt,
        development_evaluation_receipt=development_evaluation_receipt,
        regression_receipt=regression_receipt,
    )
    tree = _verify_source_tree_manifest(Path(source_tree_manifest).resolve())
    tree_sha256 = str(tree["included_tree_sha256"])
    match = RID_PATTERN.fullmatch(rid)
    assert match is not None
    if (
        match.group("tree") != tree_sha256[:12]
        or tree.get("rid") != rid
        or match.group("head") != str(tree.get("git_head", ""))[:7]
    ):
        raise ValueError("RID differs from the verified source-tree manifest")

    task1_manifest = Path(task1_manifest).resolve()
    kg_path = Path(kg_path).resolve()
    db_path = Path(db_path).resolve()
    validate_task2_artifact(kg_path, task1_path=task1_manifest)
    if not database_matches_manifest(db_path, task1_manifest, kg_path):
        raise ValueError("database does not match the supplied Task 1/KG lineage")

    css_manifest, css_root, css_test_path, css_test_binding, css_schemas = (
        _validate_css_freeze(css_freeze_manifest)
    )
    ehr_audit_payload, ehr_test_specs = _validate_ehr_audit(
        ehrsql_root,
        ehrsql_audit,
    )
    development_rows, development_questions, development_skeletons = (
        _validate_development_rows(development_path)
    )
    projection_receipt = _validate_projection_receipt(
        receipt_path=development_projection_receipt,
        source_tree_sha256=tree_sha256,
        css_manifest_path=css_freeze_manifest,
        css_manifest=css_manifest,
        ehr_audit_path=ehrsql_audit,
        ehr_audit=ehr_audit_payload,
        db_path=db_path,
        development_path=development_path,
        development_count=len(development_rows),
    )
    development_gate = _validate_development_gate(
        development_evaluation_receipt,
        freeze={
            "development": {
                "sha256": _sha256(development_path),
            }
        },
        db_path=db_path,
        source_tree_sha256=tree_sha256,
    )
    if development_gate["n"] != len(development_rows):
        raise ValueError("development gate count differs from the projection")
    regression_gate = _validate_regression_gate(
        regression_receipt,
        db_path=db_path,
        source_tree_sha256=tree_sha256,
    )
    return {
        "rid": rid,
        "control_root": control_root,
        "started_marker": started,
        "invalid_marker": invalid,
        "completed_marker": completed,
        "output_manifest": Path(output_manifest).resolve(),
        "public_output": Path(public_output).resolve(),
        "created_at": created_at,
        "source_tree_manifest": Path(source_tree_manifest).resolve(),
        "source_tree": tree,
        "source_tree_sha256": tree_sha256,
        "task1_manifest": task1_manifest,
        "kg_path": kg_path,
        "db_path": db_path,
        "css_freeze_manifest": Path(css_freeze_manifest).resolve(),
        "css_manifest": css_manifest,
        "css_root": css_root,
        "css_test_path": css_test_path,
        "css_test_binding": css_test_binding,
        "css_schemas": css_schemas,
        "ehrsql_root": Path(ehrsql_root).resolve(),
        "ehrsql_audit": Path(ehrsql_audit).resolve(),
        "ehr_audit_payload": ehr_audit_payload,
        "ehr_test_specs": ehr_test_specs,
        "development_path": Path(development_path).resolve(),
        "development_rows": development_rows,
        "development_questions": development_questions,
        "development_skeletons": development_skeletons,
        "development_projection_receipt": Path(
            development_projection_receipt
        ).resolve(),
        "projection_receipt": projection_receipt,
        "development_evaluation_receipt": Path(
            development_evaluation_receipt
        ).resolve(),
        "regression_receipt": Path(regression_receipt).resolve(),
        "development_gate": development_gate,
        "regression_gate": regression_gate,
        "css_domains": load_target_domains(db_path),
        "ehr_domains": load_ehr_target_domains(db_path),
    }


def _after_started_marker(_context: Mapping[str, Any]) -> None:
    """Test seam immediately after the irreversible create-once marker."""


def _load_official_test_rows(
    context: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
]:
    marker = Path(context["started_marker"])
    css_rows = _read_official_json_after_marker(
        Path(context["css_test_path"]),
        context["css_test_binding"],
        marker,
        "CSS official test",
    )
    ehr_rows: dict[str, list[dict[str, Any]]] = {}
    source_files = {
        "CSS/test": dict(context["css_test_binding"]),
    }
    for database, path, binding in context["ehr_test_specs"]:
        rows = _read_official_json_after_marker(
            path,
            binding,
            marker,
            f"EHRSQL {database} official test",
        )
        for row in rows:
            if row.get("db_id") != database or row.get("split") != "test":
                raise RuntimeError(
                    f"EHRSQL {database} official-test metadata is inconsistent"
                )
        ehr_rows[database] = rows
        source_files[f"EHRSQL/{database}/test"] = dict(binding)
    return css_rows, ehr_rows, source_files


def _deduplicate_global(
    items: Iterable[ProjectedItem],
) -> tuple[list[ProjectedItem], dict[str, int]]:
    by_id: dict[str, ProjectedItem] = {}
    by_question: dict[str, list[ProjectedItem]] = defaultdict(list)
    for projected in items:
        identity = str(projected.item["id"])
        if identity in by_id and by_id[identity] != projected:
            raise ValueError(f"conflicting projected id: {identity}")
        by_id[identity] = projected
        by_question[
            normalise_question(str(projected.item["question"]))
        ].append(projected)
    selected: list[ProjectedItem] = []
    exact_duplicates = 0
    ambiguous_rows = 0
    ambiguous_questions = 0
    for group in by_question.values():
        canonical = {
            validate_gold_sql(str(projected.item["gold_sql"]))
            for projected in group
        }
        if len(canonical) != 1:
            ambiguous_questions += 1
            ambiguous_rows += len(group)
            continue
        group = sorted(group, key=lambda projected: projected.item["id"])
        selected.append(group[0])
        exact_duplicates += len(group) - 1
    return (
        sorted(selected, key=lambda projected: projected.item["id"]),
        {
            "exact_duplicate_rows_removed": exact_duplicates,
            "ambiguous_questions_removed": ambiguous_questions,
            "ambiguous_rows_removed": ambiguous_rows,
        },
    )


def _project_official_holdout(
    context: Mapping[str, Any],
    css_rows: list[dict[str, Any]],
    ehr_rows: Mapping[str, list[dict[str, Any]]],
) -> tuple[list[ProjectedItem], dict[str, Any]]:
    projected: list[ProjectedItem] = []
    rejection_counts: Counter[str] = Counter()
    source_attempts: Counter[str] = Counter()
    css_schemas = context["css_schemas"]
    for row in css_rows:
        source_attempts["CSS/test"] += 1
        db_id = str(row.get("db_id", ""))
        schema = css_schemas.get(db_id)
        if schema is None:
            rejection_counts["CSS/test/schema_not_in_frozen_metadata"] += 1
            continue
        try:
            projected.append(
                project_css_row(
                    row,
                    schema,
                    context["css_domains"],
                    context["db_path"],
                    source_split="test",
                    required_role="holdout",
                    partition_version="v2",
                )
            )
        except ProjectionRejected as exc:
            rejection_counts[f"CSS/test/{exc.reason}"] += 1
    for database in EHR_DATABASES:
        for row in ehr_rows.get(database, []):
            source_attempts[f"EHRSQL/{database}/test"] += 1
            try:
                projected.append(
                    project_ehrsql_row(
                        row,
                        context["ehr_domains"],
                        context["db_path"],
                        source_dataset=database,
                        source_split="test",
                        required_role="holdout",
                    )
                )
            except EHRProjectionRejected as exc:
                rejection_counts[
                    f"EHRSQL/{database}/test/{exc.reason}"
                ] += 1
    deduplicated, deduplication = _deduplicate_global(projected)
    if any(item.skeleton_role != "holdout" for item in deduplicated):
        raise RuntimeError("official projection contains a non-holdout AST role")
    return deduplicated, {
        "source_attempts": dict(sorted(source_attempts.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "n_before_global_deduplication": len(projected),
        "n_after_global_deduplication": len(deduplicated),
        "deduplication": deduplication,
    }


def _aggregate(items: list[ProjectedItem]) -> dict[str, Any]:
    difficulties = Counter(str(item.item["difficulty"]) for item in items)
    categories = Counter(str(item.item["category"]) for item in items)
    benchmarks = Counter(
        str(item.item["provenance"]["benchmark"]) for item in items
    )
    return {
        "n": len(items),
        "n_hard": sum(
            count
            for difficulty, count in difficulties.items()
            if difficulty in HARD_DIFFICULTIES
        ),
        "benchmark_counts": dict(sorted(benchmarks.items())),
        "difficulty_counts": dict(sorted(difficulties.items())),
        "category_counts": dict(sorted(categories.items())),
        "unique_target_ast_skeletons": len(
            {item.skeleton_sha256 for item in items}
        ),
    }


def _validate_holdout_against_development(
    context: Mapping[str, Any],
    items: list[ProjectedItem],
) -> tuple[int, int]:
    ids: set[str] = set()
    questions: set[str] = set()
    skeletons: set[str] = set()
    for projected in items:
        row = projected.item
        identity = str(row.get("id", ""))
        question = normalise_question(str(row.get("question", "")))
        if (
            not identity
            or not question
            or identity in ids
            or question in questions
            or str(row.get("difficulty", "")).lower()
            not in ALLOWED_DIFFICULTIES
            or str(row.get("category", "")).lower() not in ALLOWED_CATEGORIES
        ):
            raise RuntimeError("projected holdout violates item uniqueness/schema")
        skeleton = validate_gold_sql(str(row.get("gold_sql", "")))
        skeleton_sha256 = hashlib.sha256(
            skeleton.encode("utf-8")
        ).hexdigest()
        if (
            projected.skeleton_sha256 != skeleton_sha256
            or projected.skeleton_role != "holdout"
        ):
            raise RuntimeError("projected holdout has inconsistent AST provenance")
        ids.add(identity)
        questions.add(question)
        skeletons.add(skeleton_sha256)
    question_intersection = len(questions & context["development_questions"])
    skeleton_intersection = len(skeletons & context["development_skeletons"])
    if question_intersection or skeleton_intersection:
        raise RuntimeError(
            "official holdout overlaps development by question or AST skeleton"
        )
    hard_count = sum(
        str(item.item["difficulty"]).lower() in HARD_DIFFICULTIES
        for item in items
    )
    if len(items) < 120 or hard_count < 40:
        raise RuntimeError(
            "official projection does not satisfy the 120/40 blind floors"
        )
    return question_intersection, skeleton_intersection


def _publish_frozen_set(
    *,
    context: Mapping[str, Any],
    items: list[ProjectedItem],
    source_files: Mapping[str, Mapping[str, Any]],
    projection_stats: Mapping[str, Any],
    question_intersection: int,
    skeleton_intersection: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_manifest = Path(context["output_manifest"])
    final_dir = output_manifest.parent
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{final_dir.name}.",
            suffix=".freeze-staging",
            dir=final_dir.parent,
        )
    )
    development_staged = staging_dir / "development.jsonl"
    holdout_staged = staging_dir / "holdout.jsonl"
    manifest_staged = staging_dir / "freeze_manifest.json"
    _copy_file_synced(context["development_path"], development_staged)
    if _sha256(development_staged) != _sha256(context["development_path"]):
        raise RuntimeError("staged development copy differs from source")
    holdout_bytes = _jsonl_bytes(item.item for item in items)
    _write_bytes_synced(holdout_staged, holdout_bytes)
    holdout_aggregate = _aggregate(items)
    marker_binding = {
        "path": str(Path(context["started_marker"]).resolve()),
        "sha256": _sha256(context["started_marker"]),
        "size": Path(context["started_marker"]).stat().st_size,
    }
    manifest = {
        "schema_version": 4,
        "kind": "official_public_v2_four_table_blind_freeze",
        "created_once": True,
        "created_at": context["created_at"],
        "selection_used": False,
        "source_version": context["source_tree_sha256"],
        "source_tree": {
            "path": str(Path(context["source_tree_manifest"]).resolve()),
            "sha256": _sha256(context["source_tree_manifest"]),
        },
        "partition": {
            "version": "v2",
            "seed": PROJECTION_SEED_V2,
            "development_holdout_normalised_question_intersection": (
                question_intersection
            ),
            "development_holdout_skeleton_intersection": (
                skeleton_intersection
            ),
        },
        "task1_manifest": {
            "path": str(Path(context["task1_manifest"]).resolve()),
            "sha256": _sha256(context["task1_manifest"]),
        },
        "kg": {
            "path": str(Path(context["kg_path"]).resolve()),
            "sha256": _sha256(context["kg_path"]),
        },
        "database": {
            "path": str(Path(context["db_path"]).resolve()),
            "sha256": _sha256(context["db_path"]),
        },
        "n_total": len(context["development_rows"]) + len(items),
        "n_development": len(context["development_rows"]),
        "n_holdout": len(items),
        "n_hard": holdout_aggregate["n_hard"],
        "n_sql_ast_groups": holdout_aggregate[
            "unique_target_ast_skeletons"
        ],
        "development": {
            "path": development_staged.name,
            "sha256": _sha256(development_staged),
            "size": development_staged.stat().st_size,
        },
        "holdout": {
            "path": holdout_staged.name,
            "sha256": _sha256(holdout_staged),
            "size": holdout_staged.stat().st_size,
        },
        "official_sources": {
            "css_freeze_manifest": {
                "path": str(Path(context["css_freeze_manifest"]).resolve()),
                "sha256": _sha256(context["css_freeze_manifest"]),
            },
            "ehrsql_audit": {
                "path": str(Path(context["ehrsql_audit"]).resolve()),
                "sha256": _sha256(context["ehrsql_audit"]),
            },
            "official_test_files": dict(source_files),
        },
        "projection": {
            **dict(projection_stats),
            "holdout": holdout_aggregate,
        },
        "preconsumption_gates": {
            "development": dict(context["development_gate"]),
            "known_120_regression": dict(context["regression_gate"]),
        },
        "evidence": {
            "development_projection_receipt": {
                "path": str(
                    Path(
                        context["development_projection_receipt"]
                    ).resolve()
                ),
                "sha256": _sha256(
                    context["development_projection_receipt"]
                ),
            },
            "development_evaluation_receipt": {
                "path": str(
                    Path(
                        context["development_evaluation_receipt"]
                    ).resolve()
                ),
                "sha256": _sha256(
                    context["development_evaluation_receipt"]
                ),
            },
            "known_120_regression_receipt": {
                "path": str(Path(context["regression_receipt"]).resolve()),
                "sha256": _sha256(context["regression_receipt"]),
            },
            "projection_started": marker_binding,
        },
        "independence": {
            "official_test_content_opened_before_marker": False,
            "official_test_content_opened_after_marker": True,
            "project_nl2sql_engine_imported_by_projector": False,
            "holdout_used_for_selection": False,
            "holdout_reusable_after_failure": False,
        },
        "privacy": {
            "manifest_contains_questions": False,
            "manifest_contains_gold_sql": False,
            "manifest_contains_error_samples": False,
            "official_source_or_projection_allowed_in_submission": False,
        },
        "policy": (
            "the shared v2 target-AST partition was fixed from train/valid "
            "before official test access; the create-once projection marker "
            "was published before test content was opened; any post-marker "
            "failure permanently invalidates this RID"
        ),
    }
    _write_bytes_synced(manifest_staged, _json_bytes(manifest))
    try:
        os.rename(staging_dir, final_dir)
    except BaseException:
        # Preserve staging evidence after the irreversible marker.  It must
        # never be silently cleaned or reused.
        raise
    public_receipt = {
        "schema_version": 1,
        "kind": "official_public_v2_blind_freeze_attestation",
        "status": "frozen-unconsumed",
        "created_at": context["created_at"],
        "rid": context["rid"],
        "source_tree_sha256": context["source_tree_sha256"],
        "freeze_manifest_sha256": _sha256(output_manifest),
        "counts": {
            "n_development": manifest["n_development"],
            "n_holdout": manifest["n_holdout"],
            "n_hard": manifest["n_hard"],
        },
        "partition": manifest["partition"],
        "source_attempts": projection_stats["source_attempts"],
        "projection_counts": {
            "n_before_global_deduplication": projection_stats[
                "n_before_global_deduplication"
            ],
            "n_after_global_deduplication": projection_stats[
                "n_after_global_deduplication"
            ],
            "deduplication": projection_stats["deduplication"],
            "rejection_counts": projection_stats["rejection_counts"],
            "holdout_aggregate": holdout_aggregate,
        },
        "bindings": {
            "css_freeze_manifest_sha256": _sha256(
                context["css_freeze_manifest"]
            ),
            "ehrsql_audit_sha256": _sha256(context["ehrsql_audit"]),
            "database_sha256": _sha256(context["db_path"]),
            "projection_started_sha256": marker_binding["sha256"],
            "freezer_sha256": _sha256(Path(__file__)),
            "css_projection_module_sha256": _sha256(
                ROOT / "dki" / "analysis" / "nl2sql_css_projection.py"
            ),
            "ehrsql_projection_module_sha256": _sha256(
                ROOT / "dki" / "analysis" / "nl2sql_ehrsql_projection.py"
            ),
            "protocol_sha256": _sha256(
                ROOT / "dki" / "analysis" / "nl2sql_protocol.py"
            ),
        },
        "selection_used": False,
        "official_test_content_opened_after_marker": True,
        "contains_questions": False,
        "contains_gold_sql": False,
        "contains_predictions": False,
        "contains_error_samples": False,
        "private_projection_allowed_in_submission": False,
    }
    _publish_json_once(context["public_output"], public_receipt)
    return manifest, public_receipt


def freeze_official_v2(
    *,
    css_freeze_manifest: Path,
    ehrsql_root: Path,
    ehrsql_audit: Path,
    db_path: Path,
    task1_manifest: Path,
    kg_path: Path,
    development_path: Path,
    development_projection_receipt: Path,
    development_evaluation_receipt: Path,
    regression_receipt: Path,
    source_tree_manifest: Path,
    output_manifest: Path,
    public_output: Path,
    created_at: str,
) -> dict[str, Any]:
    """Freeze official tests once, returning only aggregate public information."""
    context = _preflight_freeze(
        css_freeze_manifest=css_freeze_manifest,
        ehrsql_root=ehrsql_root,
        ehrsql_audit=ehrsql_audit,
        db_path=db_path,
        task1_manifest=task1_manifest,
        kg_path=kg_path,
        development_path=development_path,
        development_projection_receipt=development_projection_receipt,
        development_evaluation_receipt=development_evaluation_receipt,
        regression_receipt=regression_receipt,
        source_tree_manifest=source_tree_manifest,
        output_manifest=output_manifest,
        public_output=public_output,
        created_at=created_at,
    )
    started_payload = {
        "schema_version": 1,
        "kind": "official_public_v2_projection_control",
        "status": "projection-started",
        "started_at": created_at,
        "rid": context["rid"],
        "source_tree_sha256": context["source_tree_sha256"],
        "css_freeze_manifest_sha256": _sha256(css_freeze_manifest),
        "ehrsql_audit_sha256": _sha256(ehrsql_audit),
        "development_sha256": _sha256(development_path),
        "database_sha256": _sha256(db_path),
        "official_test_content_opened_before_this_marker": False,
        "reuse_permitted_after_this_marker": False,
    }
    marker_published = False
    try:
        _publish_json_once(context["started_marker"], started_payload)
        marker_published = True
        _after_started_marker(context)
        css_rows, ehr_rows, source_files = _load_official_test_rows(context)
        items, projection_stats = _project_official_holdout(
            context,
            css_rows,
            ehr_rows,
        )
        question_intersection, skeleton_intersection = (
            _validate_holdout_against_development(context, items)
        )
        manifest, public_receipt = _publish_frozen_set(
            context=context,
            items=items,
            source_files=source_files,
            projection_stats=projection_stats,
            question_intersection=question_intersection,
            skeleton_intersection=skeleton_intersection,
        )
        completed_payload = {
            "schema_version": 1,
            "kind": "official_public_v2_projection_control",
            "status": "projection-completed",
            "completed_at": created_at,
            "rid": context["rid"],
            "projection_started_sha256": _sha256(context["started_marker"]),
            "freeze_manifest_sha256": _sha256(context["output_manifest"]),
            "public_attestation_sha256": _sha256(context["public_output"]),
            "n_holdout": manifest["n_holdout"],
            "n_hard": manifest["n_hard"],
            "selection_used": False,
        }
        _publish_json_once(context["completed_marker"], completed_payload)
        return public_receipt
    except BaseException as exc:
        if marker_published:
            invalid_payload = {
                "schema_version": 1,
                "kind": "official_public_v2_projection_control",
                "status": "projection-invalid",
                "invalidated_at": created_at,
                "rid": context["rid"],
                "projection_started_sha256": _sha256(
                    context["started_marker"]
                ),
                "failure_class": type(exc).__name__,
                "contains_questions_or_sql": False,
                "reuse_permitted": False,
            }
            try:
                _publish_json_once(context["invalid_marker"], invalid_payload)
            except FileExistsError:
                pass
        raise


def preflight_official_v2(**kwargs: Any) -> dict[str, Any]:
    """Run the complete non-holdout preflight without creating any marker."""
    context = _preflight_freeze(**kwargs)
    return {
        "schema_version": 1,
        "kind": "official_public_v2_projection_preflight",
        "status": "preflight-passed",
        "rid": context["rid"],
        "source_tree_sha256": context["source_tree_sha256"],
        "n_development": len(context["development_rows"]),
        "development_gate": dict(context["development_gate"]),
        "known_120_regression_gate": dict(context["regression_gate"]),
        "official_test_content_opened": False,
        "projection_marker_created": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--css-freeze-manifest", type=Path, required=True)
    parser.add_argument("--ehrsql-root", type=Path, required=True)
    parser.add_argument("--ehrsql-audit", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--task1-manifest", type=Path, required=True)
    parser.add_argument("--kg", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument(
        "--development-projection-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--development-evaluation-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument("--regression-receipt", type=Path, required=True)
    parser.add_argument("--source-tree-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    kwargs = {
        "css_freeze_manifest": args.css_freeze_manifest,
        "ehrsql_root": args.ehrsql_root,
        "ehrsql_audit": args.ehrsql_audit,
        "db_path": args.db,
        "task1_manifest": args.task1_manifest,
        "kg_path": args.kg,
        "development_path": args.development,
        "development_projection_receipt": (
            args.development_projection_receipt
        ),
        "development_evaluation_receipt": (
            args.development_evaluation_receipt
        ),
        "regression_receipt": args.regression_receipt,
        "source_tree_manifest": args.source_tree_manifest,
        "output_manifest": args.output,
        "public_output": args.public_output,
        "created_at": args.created_at,
    }
    try:
        receipt = (
            preflight_official_v2(**kwargs)
            if args.preflight_only
            else freeze_official_v2(**kwargs)
        )
    except (
        FileExistsError,
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
