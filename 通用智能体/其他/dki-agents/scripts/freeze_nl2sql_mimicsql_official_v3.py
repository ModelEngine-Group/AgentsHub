"""Freeze the pinned MIMICSQL official test as a one-shot v3 holdout.

The official ``test.json`` is not opened or hashed during preflight.  The
script first verifies only Git tree metadata and every non-holdout artifact,
then publishes an irreversible create-once marker.  Official test bytes may
be read only after that marker exists.  Any subsequent failure permanently
invalidates the RID.

Questions, gold SQL, and the projected holdout remain below
``outputs/private/<RID>``.  The public receipt contains aggregate counts and
hash bindings only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.analysis.database import database_matches_manifest  # noqa: E402
from dki.analysis.nl2sql_css_projection import (  # noqa: E402
    ProjectedItem,
    load_target_domains,
)
from dki.analysis.nl2sql_mimicsql_projection import (  # noqa: E402
    MIMICSQL_BENCHMARK,
    MIMICSQL_PROJECTION_SEED,
    MIMICSQL_SOURCE_REVISION,
    MIMICSQL_TEST_GIT_BLOB,
    NL2SQL_V3_PARTITION_SEED,
    MIMICSQLProjectionRejected,
    deduplicate_mimicsql_items,
    project_mimicsql_row,
)
from dki.analysis.nl2sql_protocol import (  # noqa: E402
    normalise_question,
    validate_gold_sql,
)
from dki.analysis.nl2sql_seal import (  # noqa: E402
    _verify_source_tree_manifest,
)
from dki.pipeline.artifacts import validate_task2_artifact  # noqa: E402


TEST_RELATIVE_PATH = Path(
    "mimicsql_data/mimicsql_natural_v2/test.json"
)
HARD_DIFFICULTIES = {"hard", "extra", "extra-hard"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _write_bytes_synced(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"create-once artifact exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _binding(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size": path.stat().st_size,
    }


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
                        f"{label} line {line_number} is not an object"
                    )
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    return rows


def _rid_below(path: Path, scope: str) -> str:
    parts = Path(path).resolve().parts
    for index in range(len(parts) - 2):
        if (
            parts[index].casefold() == "outputs"
            and parts[index + 1].casefold() == scope.casefold()
        ):
            return parts[index + 2]
    raise ValueError(f"artifact is not below outputs/{scope}/<RID>: {path}")


def _run_git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _official_test_metadata(
    source_repo: Path,
    *,
    source_revision: str,
    official_test_git_blob: str,
) -> dict[str, Any]:
    """Verify Git metadata and stat the test path without opening its bytes."""
    source_repo = Path(source_repo).resolve()
    if "outputs/private/" not in source_repo.as_posix():
        raise ValueError("the pinned public-source clone must remain private")
    if _run_git(source_repo, "rev-parse", "HEAD") != source_revision:
        raise ValueError("MIMICSQL clone is not at the pinned source revision")
    tree_line = _run_git(
        source_repo,
        "ls-tree",
        "HEAD",
        "--",
        TEST_RELATIVE_PATH.as_posix(),
    )
    match = re.fullmatch(
        r"100644 blob ([0-9a-f]{40})\t"
        + re.escape(TEST_RELATIVE_PATH.as_posix()),
        tree_line,
    )
    if not match or match.group(1) != official_test_git_blob:
        raise ValueError("MIMICSQL official test Git-blob binding failed")
    test_path = (source_repo / TEST_RELATIVE_PATH).resolve()
    try:
        test_path.relative_to(source_repo)
    except ValueError as exc:
        raise ValueError("unsafe MIMICSQL official test path") from exc
    if test_path.is_symlink() or not test_path.is_file():
        raise ValueError("MIMICSQL official test worktree file is missing")
    return {
        "path": test_path,
        "size_before_marker": test_path.stat().st_size,
        "git_blob_sha1": official_test_git_blob,
    }


def _validate_development(
    *,
    development_path: Path,
    development_receipt: Path,
    db_path: Path,
    source_tree_sha256: str,
    source_revision: str,
    official_test_git_blob: str,
) -> dict[str, Any]:
    development_path = Path(development_path).resolve()
    if "outputs/private/" not in development_path.as_posix():
        raise ValueError("MIMICSQL development projection must remain private")
    receipt = _load_json(
        development_receipt,
        "MIMICSQL development receipt",
    )
    benchmark = receipt.get("benchmark")
    official_test = receipt.get("official_test")
    hashes = receipt.get("hashes")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind")
        != "mimicsql_v3_four_table_development_receipt"
        or receipt.get("dataset_role") != "development_selection_used"
        or receipt.get("source_tree_sha256") != source_tree_sha256
        or not isinstance(benchmark, Mapping)
        or benchmark.get("name") != MIMICSQL_BENCHMARK
        or benchmark.get("source_revision") != source_revision
        or not isinstance(official_test, Mapping)
        or official_test.get("source_split") != "test"
        or official_test.get("git_blob_sha1") != official_test_git_blob
        or official_test.get("path_argument_accepted_by_this_builder")
        is not False
        or official_test.get("content_opened_or_hashed_by_this_builder")
        is not False
        or official_test.get("consumed") is not False
        or not isinstance(hashes, Mapping)
        or hashes.get("target_db_sha256") != _sha256(db_path)
        or hashes.get("private_development_sha256")
        != _sha256(development_path)
        or hashes.get("projection_module_sha256")
        != _sha256(
            ROOT / "dki" / "analysis" / "nl2sql_mimicsql_projection.py"
        )
        or hashes.get("builder_sha256")
        != _sha256(
            ROOT / "scripts" / "build_nl2sql_mimicsql_development.py"
        )
    ):
        raise ValueError(
            "MIMICSQL development receipt does not bind the candidate"
        )
    rows = _load_jsonl(development_path, "MIMICSQL development projection")
    if len(rows) < 150:
        raise ValueError("MIMICSQL development projection is below 150")
    questions: set[str] = set()
    skeletons: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if (
            not row.get("question")
            or not row.get("gold_sql")
            or row.get("difficulty")
            not in {"medium", "hard", "extra", "extra-hard"}
        ):
            raise ValueError(
                f"MIMICSQL development item {index} is malformed"
            )
        provenance = row.get("provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("skeleton_role") != "development"
        ):
            raise ValueError(
                "MIMICSQL development contains a non-development AST role"
            )
        question = normalise_question(str(row["question"]))
        skeleton = validate_gold_sql(str(row["gold_sql"]))
        skeleton_sha256 = hashlib.sha256(
            skeleton.encode("utf-8")
        ).hexdigest()
        if question in questions:
            raise ValueError("duplicate MIMICSQL development question")
        if provenance.get("skeleton_sha256") != skeleton_sha256:
            raise ValueError("MIMICSQL development AST binding failed")
        questions.add(question)
        skeletons.add(skeleton_sha256)
    return {
        "rows": rows,
        "questions": frozenset(questions),
        "skeletons": frozenset(skeletons),
        "receipt": receipt,
        "path_sha256": _sha256(development_path),
        "path_size": development_path.stat().st_size,
    }


def _preflight_freeze(
    *,
    source_repo: Path,
    db_path: Path,
    task1_manifest: Path,
    kg_path: Path,
    development_path: Path,
    development_receipt: Path,
    source_tree_manifest: Path,
    output_manifest: Path,
    public_output: Path,
    created_at: str,
    source_revision: str,
    official_test_git_blob: str,
) -> dict[str, Any]:
    if source_revision != MIMICSQL_SOURCE_REVISION:
        raise ValueError("MIMICSQL source revision is not pinned")
    if official_test_git_blob != MIMICSQL_TEST_GIT_BLOB:
        raise ValueError("MIMICSQL official test Git blob is not pinned")
    if not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
        created_at,
    ):
        raise ValueError("created_at must be UTC YYYY-MM-DDTHH:MM:SSZ")

    output_manifest = Path(output_manifest).resolve()
    public_output = Path(public_output).resolve()
    rid = _rid_below(output_manifest, "private")
    if _rid_below(public_output, "remote_runs") != rid:
        raise ValueError("private and public MIMICSQL freeze RID differ")
    if _rid_below(development_path, "private") != rid:
        raise ValueError("development and official freeze RID differ")
    if _rid_below(development_receipt, "remote_runs") != rid:
        raise ValueError("development receipt and official freeze RID differ")
    if _rid_below(source_tree_manifest, "remote_runs") != rid:
        raise ValueError("source tree and official freeze RID differ")

    control_root = output_manifest.parent.parent
    final_dir = output_manifest.parent
    if output_manifest.name != "freeze_manifest.json":
        raise ValueError("official freeze manifest name must be fixed")
    if final_dir.name != "frozen":
        raise ValueError("official freeze directory must be named frozen")
    if control_root.name != "nl2sql_mimicsql_official_v3":
        raise ValueError(
            "official freeze control directory name must be fixed"
        )
    expected_public = (
        public_output.parents[2]
        / rid
        / "nl2sql"
        / "mimicsql_official_freeze_attestation.json"
    )
    if public_output != expected_public:
        raise ValueError("official public attestation path must be fixed")
    started_marker = control_root / "projection-started.json"
    completed_marker = control_root / "projection-completed.json"
    invalid_marker = control_root / "projection-invalid.json"
    staging_dir = final_dir.parent / f".{final_dir.name}.freeze-staging"
    occupied = [
        path
        for path in (
            final_dir,
            staging_dir,
            started_marker,
            completed_marker,
            invalid_marker,
            public_output,
        )
        if path.exists()
    ]
    if occupied:
        raise FileExistsError(
            "MIMICSQL official projection RID is already occupied: "
            + ", ".join(str(path) for path in occupied)
        )

    source_tree = _verify_source_tree_manifest(source_tree_manifest)
    source_tree_sha256 = source_tree["included_tree_sha256"]
    task1_manifest = Path(task1_manifest).resolve()
    kg_path = Path(kg_path).resolve()
    db_path = Path(db_path).resolve()
    validate_task2_artifact(kg_path, task1_path=task1_manifest)
    if not database_matches_manifest(db_path, task1_manifest, kg_path):
        raise ValueError("database does not match Task 1 / KG provenance")

    development = _validate_development(
        development_path=development_path,
        development_receipt=development_receipt,
        db_path=db_path,
        source_tree_sha256=source_tree_sha256,
        source_revision=source_revision,
        official_test_git_blob=official_test_git_blob,
    )
    official_test = _official_test_metadata(
        source_repo,
        source_revision=source_revision,
        official_test_git_blob=official_test_git_blob,
    )
    return {
        "rid": rid,
        "created_at": created_at,
        "control_root": control_root,
        "final_dir": final_dir,
        "staging_dir": staging_dir,
        "started_marker": started_marker,
        "completed_marker": completed_marker,
        "invalid_marker": invalid_marker,
        "output_manifest": output_manifest,
        "public_output": public_output,
        "source_tree_manifest": Path(source_tree_manifest).resolve(),
        "source_tree_sha256": source_tree_sha256,
        "task1_manifest": task1_manifest,
        "kg_path": kg_path,
        "db_path": db_path,
        "development_path": Path(development_path).resolve(),
        "development_receipt": Path(development_receipt).resolve(),
        "development_rows": development["rows"],
        "development_questions": development["questions"],
        "development_skeletons": development["skeletons"],
        "development_sha256": development["path_sha256"],
        "development_size": development["path_size"],
        "official_test": official_test,
        "source_revision": source_revision,
        "official_test_git_blob": official_test_git_blob,
        "preflight_bindings": {
            "development_receipt": _binding(development_receipt),
            "source_tree_manifest": _binding(source_tree_manifest),
            "task1_manifest": _binding(task1_manifest),
            "kg": _binding(kg_path),
            "database": _binding(db_path),
        },
    }


def preflight_mimicsql_official_v3(**kwargs: Any) -> dict[str, Any]:
    context = _preflight_freeze(**kwargs)
    return {
        "schema_version": 1,
        "kind": "mimicsql_v3_official_projection_preflight",
        "status": "preflight-passed",
        "rid": context["rid"],
        "source_tree_sha256": context["source_tree_sha256"],
        "n_development": len(context["development_rows"]),
        "official_test_git_blob": context["official_test_git_blob"],
        "official_test_size_from_stat": context["official_test"][
            "size_before_marker"
        ],
        "official_test_content_opened": False,
        "projection_marker_created": False,
    }


def _read_official_test_after_marker(
    path: Path,
    *,
    marker: Path,
    expected_git_blob: str,
    expected_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not Path(marker).is_file():
        raise RuntimeError(
            "refusing to open MIMICSQL official test before projection marker"
        )
    data = Path(path).read_bytes()
    if len(data) != expected_size:
        raise RuntimeError(
            "MIMICSQL official test size changed after preflight"
        )
    git_blob = _git_blob_sha1(data)
    if git_blob != expected_git_blob:
        raise RuntimeError("MIMICSQL official test Git blob differs")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(data.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"MIMICSQL official test line {line_number} is invalid"
            ) from exc
        if not isinstance(row, dict):
            raise RuntimeError(
                f"MIMICSQL official test line {line_number} is not an object"
            )
        rows.append(row)
    if not rows:
        raise RuntimeError("MIMICSQL official test is empty")
    return rows, {
        "git_blob_sha1": git_blob,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
        "n_source_rows": len(rows),
    }


def _project_holdout(
    rows: Iterable[Mapping[str, Any]],
    *,
    db_path: Path,
    source_revision: str,
) -> tuple[list[ProjectedItem], dict[str, Any]]:
    domains = load_target_domains(db_path)
    projected: list[ProjectedItem] = []
    rejected: Counter[str] = Counter()
    attempts = 0
    for row in rows:
        attempts += 1
        try:
            projected.append(
                project_mimicsql_row(
                    row,
                    domains,
                    db_path,
                    source_split="test",
                    required_role="holdout",
                    source_revision=source_revision,
                )
            )
        except MIMICSQLProjectionRejected as exc:
            rejected[exc.reason] += 1
    deduplicated, deduplication = deduplicate_mimicsql_items(projected)
    return deduplicated, {
        "source_attempts": attempts,
        "rejection_counts": dict(sorted(rejected.items())),
        "n_before_deduplication": len(projected),
        "n_after_deduplication": len(deduplicated),
        "deduplication": deduplication,
    }


def _validate_holdout(
    context: Mapping[str, Any],
    items: Iterable[ProjectedItem],
) -> dict[str, Any]:
    rows = list(items)
    questions: set[str] = set()
    skeletons: set[str] = set()
    difficulties: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    for projected in rows:
        item = projected.item
        provenance = item.get("provenance")
        if (
            projected.skeleton_role != "holdout"
            or not isinstance(provenance, Mapping)
            or provenance.get("source_split") != "test"
            or provenance.get("skeleton_role") != "holdout"
            or provenance.get("skeleton_sha256")
            != projected.skeleton_sha256
        ):
            raise RuntimeError(
                "MIMICSQL official projection contains a non-holdout item"
            )
        question = normalise_question(str(item["question"]))
        if question in questions:
            raise RuntimeError(
                "MIMICSQL official projection has duplicate questions"
            )
        questions.add(question)
        skeletons.add(projected.skeleton_sha256)
        difficulties[str(item["difficulty"])] += 1
        categories[str(item["category"])] += 1
    question_overlap = questions & set(context["development_questions"])
    skeleton_overlap = skeletons & set(context["development_skeletons"])
    hard = sum(
        count
        for difficulty, count in difficulties.items()
        if difficulty in HARD_DIFFICULTIES
    )
    if question_overlap or skeleton_overlap:
        raise RuntimeError(
            "MIMICSQL official holdout overlaps development"
        )
    if len(rows) < 120 or hard < 40:
        raise RuntimeError(
            "MIMICSQL official projection is below the 120/40 blind floor"
        )
    return {
        "n": len(rows),
        "n_hard": hard,
        "difficulty_counts": dict(sorted(difficulties.items())),
        "category_counts": dict(sorted(categories.items())),
        "unique_target_ast_skeletons": len(skeletons),
        "development_holdout_normalised_question_intersection": 0,
        "development_holdout_skeleton_intersection": 0,
    }


def _after_started_marker(_context: Mapping[str, Any]) -> None:
    """Test hook executed after the irreversible marker is durable."""


def _verify_nonholdout_stability(context: Mapping[str, Any]) -> None:
    if (
        _sha256(context["development_path"])
        != context["development_sha256"]
        or Path(context["development_path"]).stat().st_size
        != context["development_size"]
    ):
        raise RuntimeError(
            "MIMICSQL development changed after projection preflight"
        )
    for label, expected in context["preflight_bindings"].items():
        current = _binding(expected["path"])
        if (
            current["path"] != expected["path"]
            or current["sha256"] != expected["sha256"]
            or current["size"] != expected["size"]
        ):
            raise RuntimeError(
                f"MIMICSQL non-holdout binding changed: {label}"
            )
    source_tree = _verify_source_tree_manifest(
        context["source_tree_manifest"]
    )
    if (
        source_tree["included_tree_sha256"]
        != context["source_tree_sha256"]
    ):
        raise RuntimeError("MIMICSQL source tree changed after preflight")


def _publish_frozen_set(
    context: Mapping[str, Any],
    *,
    items: list[ProjectedItem],
    source_binding: Mapping[str, Any],
    projection_stats: Mapping[str, Any],
    holdout_aggregate: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    staging_dir = Path(context["staging_dir"])
    final_dir = Path(context["final_dir"])
    staging_dir.mkdir(parents=True)
    development_path = staging_dir / "development.jsonl"
    holdout_path = staging_dir / "holdout.jsonl"
    manifest_path = staging_dir / "freeze_manifest.json"
    development_bytes = Path(context["development_path"]).read_bytes()
    holdout_bytes = _jsonl_bytes(item.item for item in items)
    _write_bytes_synced(development_path, development_bytes)
    _write_bytes_synced(holdout_path, holdout_bytes)

    manifest = {
        "schema_version": 4,
        "kind": "mimicsql_v3_four_table_blind_freeze",
        "created_once": True,
        "created_at": context["created_at"],
        "source_version": context["source_tree_sha256"],
        "task1_manifest": _binding(context["task1_manifest"]),
        "kg": _binding(context["kg_path"]),
        "database": _binding(context["db_path"]),
        "n_total": (
            len(context["development_rows"])
            + int(holdout_aggregate["n"])
        ),
        "n_development": len(context["development_rows"]),
        "n_holdout": int(holdout_aggregate["n"]),
        "n_hard": int(holdout_aggregate["n_hard"]),
        "n_sql_ast_groups": (
            len(context["development_skeletons"])
            + int(holdout_aggregate["unique_target_ast_skeletons"])
        ),
        "development": {
            "path": development_path.name,
            "sha256": hashlib.sha256(development_bytes).hexdigest(),
            "size": len(development_bytes),
        },
        "holdout": {
            "path": holdout_path.name,
            "sha256": hashlib.sha256(holdout_bytes).hexdigest(),
            "size": len(holdout_bytes),
        },
        "partition": {
            "version": "v3",
            "seed": NL2SQL_V3_PARTITION_SEED,
            "projection_seed": MIMICSQL_PROJECTION_SEED,
            "development_holdout_normalised_question_intersection": 0,
            "development_holdout_skeleton_intersection": 0,
        },
        "benchmark": {
            "name": MIMICSQL_BENCHMARK,
            "source_revision": context["source_revision"],
            "official_test": dict(source_binding),
            "license": "MIT",
        },
        "projection": dict(projection_stats),
        "holdout_aggregate": dict(holdout_aggregate),
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
        "evidence": {
            "projection_started": _binding(context["started_marker"]),
            "development_projection_receipt": _binding(
                context["development_receipt"]
            ),
            "source_tree_manifest": _binding(
                context["source_tree_manifest"]
            ),
        },
        "selection_used": False,
    }
    _write_bytes_synced(manifest_path, _json_bytes(manifest))
    try:
        os.rename(staging_dir, final_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    final_manifest = Path(context["output_manifest"])
    if final_manifest != final_dir / "freeze_manifest.json":
        raise RuntimeError("published MIMICSQL manifest path differs")

    public = {
        "schema_version": 1,
        "kind": "mimicsql_v3_official_blind_freeze_attestation",
        "status": "frozen-unconsumed",
        "rid": context["rid"],
        "created_at": context["created_at"],
        "source_tree_sha256": context["source_tree_sha256"],
        "freeze_manifest_sha256": _sha256(final_manifest),
        "benchmark": {
            "name": MIMICSQL_BENCHMARK,
            "source_revision": context["source_revision"],
            "official_test_git_blob": context["official_test_git_blob"],
        },
        "counts": {
            "n_development": manifest["n_development"],
            "n_holdout": manifest["n_holdout"],
            "n_hard": manifest["n_hard"],
            "difficulty_counts": holdout_aggregate[
                "difficulty_counts"
            ],
            "category_counts": holdout_aggregate["category_counts"],
            "unique_target_ast_skeletons": holdout_aggregate[
                "unique_target_ast_skeletons"
            ],
        },
        "selection_used": False,
        "official_test_content_opened_after_marker": True,
        "contains_questions": False,
        "contains_gold_sql": False,
        "contains_predictions": False,
        "contains_error_samples": False,
    }
    _publish_json_once(context["public_output"], public)
    return manifest, public


def freeze_mimicsql_official_v3(**kwargs: Any) -> dict[str, Any]:
    context = _preflight_freeze(**kwargs)
    started = {
        "schema_version": 1,
        "kind": "mimicsql_v3_official_projection_control",
        "status": "projection-started",
        "rid": context["rid"],
        "started_at": context["created_at"],
        "source_tree_sha256": context["source_tree_sha256"],
        "source_revision": context["source_revision"],
        "official_test_git_blob": context["official_test_git_blob"],
        "official_test_size_from_stat": context["official_test"][
            "size_before_marker"
        ],
        "official_test_content_opened_before_this_marker": False,
        "reuse_permitted_after_this_marker": False,
    }
    _publish_json_once(context["started_marker"], started)
    try:
        _after_started_marker(context)
        _verify_nonholdout_stability(context)
        source_rows, source_binding = _read_official_test_after_marker(
            context["official_test"]["path"],
            marker=context["started_marker"],
            expected_git_blob=context["official_test_git_blob"],
            expected_size=context["official_test"][
                "size_before_marker"
            ],
        )
        items, projection_stats = _project_holdout(
            source_rows,
            db_path=context["db_path"],
            source_revision=context["source_revision"],
        )
        holdout_aggregate = _validate_holdout(context, items)
        _verify_nonholdout_stability(context)
        manifest, public = _publish_frozen_set(
            context,
            items=items,
            source_binding=source_binding,
            projection_stats=projection_stats,
            holdout_aggregate=holdout_aggregate,
        )
        completed = {
            "schema_version": 1,
            "kind": "mimicsql_v3_official_projection_control",
            "status": "projection-completed",
            "rid": context["rid"],
            "completed_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
            "projection_started_sha256": _sha256(
                context["started_marker"]
            ),
            "freeze_manifest_sha256": _sha256(
                context["output_manifest"]
            ),
            "public_attestation_sha256": _sha256(
                context["public_output"]
            ),
            "n_holdout": manifest["n_holdout"],
            "n_hard": manifest["n_hard"],
            "selection_used": False,
        }
        _publish_json_once(context["completed_marker"], completed)
        return public
    except BaseException as exc:
        invalid = {
            "schema_version": 1,
            "kind": "mimicsql_v3_official_projection_control",
            "status": "projection-invalid",
            "rid": context["rid"],
            "invalidated_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
            "projection_started_sha256": _sha256(
                context["started_marker"]
            ),
            "failure_class": type(exc).__name__,
            "reuse_permitted": False,
        }
        if not Path(context["invalid_marker"]).exists():
            _publish_json_once(context["invalid_marker"], invalid)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--task1-manifest", type=Path, required=True)
    parser.add_argument("--kg", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument(
        "--development-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--source-tree-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument(
        "--source-revision",
        default=MIMICSQL_SOURCE_REVISION,
    )
    parser.add_argument(
        "--official-test-git-blob",
        default=MIMICSQL_TEST_GIT_BLOB,
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    kwargs = {
        "source_repo": args.source_repo,
        "db_path": args.db,
        "task1_manifest": args.task1_manifest,
        "kg_path": args.kg,
        "development_path": args.development,
        "development_receipt": args.development_receipt,
        "source_tree_manifest": args.source_tree_manifest,
        "output_manifest": args.output_manifest,
        "public_output": args.public_output,
        "created_at": args.created_at,
        "source_revision": args.source_revision,
        "official_test_git_blob": args.official_test_git_blob,
    }
    try:
        if args.preflight_only:
            result = preflight_mimicsql_official_v3(**kwargs)
        else:
            result = freeze_mimicsql_official_v3(**kwargs)
    except (
        FileExistsError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
