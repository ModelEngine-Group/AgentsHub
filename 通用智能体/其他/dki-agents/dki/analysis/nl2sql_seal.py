"""Hash-bound candidate sealing for one-shot NL2SQL blind evaluation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

from dki.analysis.database import database_matches_manifest
from dki.analysis.nl2sql_development_evidence import (
    require_lowercase_sha256,
    strict_development_jsonl_rows,
    strict_json_loads,
)
from dki.analysis.nl2sql_eval import evaluate_set
from dki.analysis.nl2sql_source_closure import (
    DEFAULT_ENTRY_MODULES,
    closure_sha256,
)
from dki.analysis.nl2sql_official_v2_development_anchor import (
    anchor_from_trusted_evidence_binding,
    trusted_evidence_binding,
)
from dki.analysis.nl2sql_protocol import (
    normalise_development_metadata,
    normalise_question,
    validate_gold_sql,
    validate_question_sql_grounding,
)
from dki.pipeline.artifacts import task2_receipt_path, validate_task2_artifact


ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT_MIN_N = 180
DEVELOPMENT_OVERALL_PERCENT = 92
DEVELOPMENT_HARD_PLUS_PERCENT = 85
DEVELOPMENT_EXTRA_HARD_PERCENT = 70
DEVELOPMENT_COVERAGE_PERCENT = 97
ORIGINAL_DEVELOPMENT_GATE = {
    "minimum_n": DEVELOPMENT_MIN_N,
    "overall_minimum_percent": DEVELOPMENT_OVERALL_PERCENT,
    "hard_plus_minimum_percent": DEVELOPMENT_HARD_PLUS_PERCENT,
    "extra_hard_minimum_percent": DEVELOPMENT_EXTRA_HARD_PERCENT,
    "coverage_minimum_percent": DEVELOPMENT_COVERAGE_PERCENT,
}
# Historical record of the explicit operator-authorized exception used for the
# already-consumed sixth holdout probe.  This is evidence only: current release
# sealing always enforces ORIGINAL_DEVELOPMENT_GATE and never admits this
# override.  Keeping the declaration lets old receipts remain interpretable
# without leaving a zero-threshold path in the active candidate sealer.
HISTORICAL_DEVELOPMENT_GATE_OVERRIDE = {
    "kind": "operator_authorized_holdout_performance_probe",
    "authorized_on": "2026-08-09",
    "reason": (
        "The operator explicitly requested consumption of the sixth holdout "
        "despite failure of the original development pre-seal gate."
    ),
    "minimum_n": 180,
    "overall_minimum_percent": 0,
    "hard_plus_minimum_percent": 0,
    "extra_hard_minimum_percent": 0,
    "coverage_minimum_percent": 0,
    "applies_only_when_original_gate_fails": True,
}
KNOWN_120_REQUIRED_N = 120
KNOWN_120_MIN_CORRECT = 112
EASY_44_REQUIRED_N = 44
EASY_44_REQUIRED_CORRECT = 44
REQUIRED_SOURCE_PATHS = {
    "dki/analysis/database.py",
    "dki/analysis/nl2sql.py",
    "dki/llm.py",
    "dki/analysis/nl2sql_compositional.py",
    "dki/analysis/nl2sql_css_projection.py",
    "dki/analysis/nl2sql_development_evidence.py",
    "dki/analysis/nl2sql_ehrsql_projection.py",
    "dki/analysis/nl2sql_eval.py",
    "dki/analysis/nl2sql_source_closure.py",
    "dki/analysis/nl2sql_official_v2_development_anchor.py",
    "dki/analysis/nl2sql_explicit.py",
    "dki/analysis/nl2sql_mimicsql_projection.py",
    "dki/analysis/nl2sql_protocol.py",
    "dki/analysis/nl2sql_seal.py",
    "dki/analysis/nl2sql_sketch.py",
    "dki/analysis/nl2sql_sqlite.py",
    "dki/kg/dictionaries/medical_dict.json",
    "dki/kg/schema.py",
    "dki/kg/store.py",
    "dki/paths.py",
    "dki/pipeline/artifacts.py",
    "scripts/build_included_tree_manifest.py",
    "scripts/build_nl2sql_source_closure.py",
    "scripts/build_nl2sql_curator_pack.py",
    "scripts/audit_nl2sql_development.py",
    "scripts/build_nl2sql_mimicsql_development.py",
    "scripts/eval_nl2sql_blind.py",
    "scripts/eval_nl2sql_development.py",
    "scripts/eval_nl2sql_frozen_development.py",
    "scripts/eval_nl2sql_layered.py",
    "scripts/freeze_nl2sql_blind.py",
    "scripts/freeze_nl2sql_mimicsql_official_v3.py",
    "scripts/freeze_nl2sql_official_v2.py",
    "scripts/freeze_python_environment.py",
    "scripts/index_nl2sql_public_donors.py",
    "scripts/project_nl2sql_official_v2_development.py",
    "scripts/project_nl2sql_public_donors.py",
    "scripts/seal_nl2sql_blind_candidate.py",
    "scripts/setup_env.ps1",
    "scripts/setup_env.sh",
    "pyproject.toml",
}

# ``verify_candidate_seal`` consumes these bindings directly. Validate their
# presence before resolving any path so a malformed or downgraded seal cannot
# turn into a KeyError (or silently skip an execution-closure check).
REQUIRED_SEAL_ARTIFACT_BINDINGS = (
    "freeze_manifest",
    "task1_manifest",
    "task2_kg",
    "database",
    "blind_evaluator",
    "source_tree_manifest",
    "execution_bundle",
    "dependency_lock",
    "development_gate",
    "known_120_regression_gate",
)

# These handoffs have the same fully attested five-file contract. Keep the
# release sealer in lockstep with the freezer so a valid newer curator version
# cannot freeze successfully only to be rejected later during candidate seal.
INDEPENDENT_FIVE_FILE_CURATOR_FORMATS = frozenset(
    {
        "independent-v11",
        "independent-v12",
        "independent-v13",
        "independent-v14",
        "independent-v15",
        "independent-v16",
        "independent-v17",
        "independent-v18",
        "independent-v19",
        "independent-v20",
        "independent-v21",
        "independent-v22",
        "independent-v23",
        "independent-v24",
        "independent-v25",
        "independent-v26",
        "independent-v27",
        "independent-v28",
        "independent-v29",
        "independent-v30",
    }
)
INDEPENDENT_FIVE_FILE_CURATOR_SOURCE_VERSIONS = frozenset(
    {
        "independent-curator-v11",
        "independent-curator-v11b",
        "independent-curator-v12",
        "independent-curator-v13",
        "independent-curator-v14",
        "independent-curator-v15",
        "independent-curator-v16",
        "independent-curator-v17",
        "independent-curator-v18",
        "independent-curator-v19",
        "independent-curator-v20",
        "independent-curator-v21",
        "independent-curator-v22",
        "independent-curator-v23",
        "independent-curator-v24",
        "independent-curator-v25",
        "independent-curator-v26",
        "independent-curator-v27",
        "independent-curator-v28",
        "independent-curator-v29",
        "independent-curator-v30",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def _publish_once(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"candidate seal already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _assert_freeze_unconsumed(freeze_manifest: Path) -> None:
    """Reject a blind set once a create-once consumption marker exists.

    The immutable public freeze attestation can still say
    ``frozen-unconsumed`` after a later evaluation.  The markers beside the
    private freeze manifest are the authoritative live state.  This check
    stats marker names only and never opens the holdout.
    """

    freeze_dir = Path(freeze_manifest).resolve().parent
    marker_names = (
        "consumed-started.json",
        "evaluation_receipt.json",
        "consumed-invalid.json",
    )
    occupied = [
        name for name in marker_names if (freeze_dir / name).exists()
    ]
    if occupied:
        raise ValueError(
            "blind freeze has already been consumed or invalidated: "
            + ", ".join(occupied)
        )


def _verify_source_tree_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = strict_json_loads(
            Path(path).read_bytes(),
            label="included-tree manifest",
        )
    except (OSError, ValueError) as exc:
        raise ValueError("included-tree manifest is unreadable") from exc
    if type(payload) is not dict:
        raise ValueError("included-tree manifest must be a JSON object")
    tree_hash = str(payload.get("included_tree_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", tree_hash):
        raise ValueError("included-tree manifest lacks a SHA-256 tree identity")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("included-tree manifest lacks entries")
    seen = set()
    canonical = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item.get("path", ""))):
        relative = str(entry.get("path", "")).replace("\\", "/")
        candidate = (ROOT / relative).resolve()
        try:
            candidate.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"unsafe included-tree path: {relative}") from exc
        if relative in seen:
            raise ValueError(f"duplicate included-tree path: {relative}")
        seen.add(relative)
        if (
            not candidate.is_file()
            or candidate.stat().st_size != entry.get("size")
            or sha256_file(candidate) != entry.get("sha256")
        ):
            raise ValueError(f"included-tree binding failed: {relative}")
        canonical.update(relative.encode("utf-8"))
        canonical.update(b"\0")
        canonical.update(str(candidate.stat().st_size).encode("ascii"))
        canonical.update(b"\0")
        canonical.update(str(entry["sha256"]).encode("ascii"))
        canonical.update(b"\n")
    if canonical.hexdigest() != tree_hash:
        raise ValueError("included-tree aggregate hash is invalid")
    missing = REQUIRED_SOURCE_PATHS - seen
    if missing:
        raise ValueError(
            f"included-tree manifest omits sealed source files: {sorted(missing)}"
        )
    return payload


def _verify_execution_bundle_closure(
    execution_bundle: Path,
    *,
    source_tree: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a dynamic-import-free source closure to the included tree."""

    try:
        manifest = strict_json_loads(
            Path(execution_bundle).read_bytes(),
            label="NL2SQL execution closure",
        )
    except (OSError, ValueError) as exc:
        raise ValueError("execution bundle closure is unreadable") from exc
    if (
        type(manifest) is not dict
        or manifest.get("schema_version") != 1
        or manifest.get("kind") != "nl2sql_static_execution_closure"
        or manifest.get("created_once") is not True
        or tuple(manifest.get("entry_modules", ())) != DEFAULT_ENTRY_MODULES
        or manifest.get("algorithm")
        != "sha256 of sorted module NUL path NUL size NUL file_sha256 LF"
    ):
        raise ValueError("execution bundle closure has the wrong NL2SQL contract")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("execution bundle closure lacks files")
    modules: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    for raw in raw_files:
        if not isinstance(raw, Mapping):
            raise ValueError("execution bundle closure file entry is invalid")
        module = raw.get("module")
        path = raw.get("path")
        size = raw.get("size")
        digest = raw.get("sha256")
        is_package = raw.get("is_package")
        if (
            not isinstance(module, str)
            or not isinstance(path, str)
            or type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or type(is_package) is not bool
            or module in modules
        ):
            raise ValueError("execution bundle closure file entry is malformed")
        entry = {
            "module": module,
            "path": path,
            "size": size,
            "sha256": digest,
            "is_package": is_package,
        }
        modules[module] = entry
        entries.append(entry)
    if set(DEFAULT_ENTRY_MODULES) - set(modules):
        raise ValueError("execution bundle closure omits required entry modules")
    if entries != sorted(entries, key=lambda item: (item["path"], item["module"])):
        raise ValueError("execution bundle closure files are not canonically sorted")
    if closure_sha256(entries) != manifest.get("closure_sha256"):
        raise ValueError("execution bundle closure aggregate hash is invalid")
    tree_entries = {
        str(entry["path"]): entry
        for entry in source_tree.get("entries", [])
        if isinstance(entry, Mapping)
    }
    for raw in entries:
        path = raw["path"]
        bound = tree_entries.get(path)
        if (
            not isinstance(bound, Mapping)
            or bound.get("sha256") != raw["sha256"]
            or bound.get("size") != raw["size"]
        ):
            raise ValueError(
                "execution bundle closure is not fully bound by included tree: "
                + path
            )
    return {
        "archive": _binding(Path(execution_bundle)),
        "closure_sha256": manifest["closure_sha256"],
        "entry_modules": list(DEFAULT_ENTRY_MODULES),
        "n_modules": len(modules),
        "n_files": len(entries),
    }


def _freeze_contract(
    *,
    freeze_manifest: Path,
    task1_manifest: Path,
    kg_path: Path,
    db_path: Path,
) -> dict[str, Any]:
    try:
        freeze = strict_json_loads(
            Path(freeze_manifest).read_bytes(),
            label="blind freeze manifest",
        )
    except (OSError, ValueError) as exc:
        raise ValueError("blind freeze manifest is unreadable") from exc
    if type(freeze) is not dict:
        raise ValueError("blind freeze manifest must be a JSON object")
    if (
        freeze.get("schema_version") != 4
        or freeze.get("created_once") is not True
        or freeze.get("selection_used") is not False
        or freeze.get("n_holdout", 0) < 120
        or freeze.get("n_hard", 0) < 40
        or freeze.get("n_development", 0) < DEVELOPMENT_MIN_N
    ):
        raise ValueError("blind freeze manifest does not satisfy the v4 gate contract")
    expected = {
        "task1_manifest": (task1_manifest, freeze.get("task1_manifest", {})),
        "kg": (kg_path, freeze.get("kg", {})),
        "database": (db_path, freeze.get("database", {})),
    }
    for label, (path, binding) in expected.items():
        if (
            not isinstance(binding, dict)
            or binding.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"blind freeze {label} binding failed")
    if freeze.get("kind") == "official_public_v2_four_table_blind_freeze":
        _official_v2_freeze_evidence(freeze_manifest, freeze)
    if freeze.get("kind") == "mimicsql_v3_four_table_blind_freeze":
        _mimicsql_v3_freeze_evidence(freeze_manifest, freeze)
    if (
        freeze.get("kind")
        == "independent_curated_v3_four_table_blind_freeze"
    ):
        _independent_v3_freeze_evidence(freeze_manifest, freeze)
    return freeze


def _scoped_output_root_and_rid(
    path: Path,
    *,
    scope: str,
) -> tuple[Path, str]:
    resolved = Path(path).resolve()
    parts = resolved.parts
    for index in range(len(parts) - 2):
        if parts[index : index + 2] == ("outputs", scope):
            return Path(*parts[:index]), parts[index + 2]
    raise ValueError(
        f"official-v2 artifact is not below outputs/{scope}/<RID>"
    )


def _safe_frozen_member(root: Path, relative: Any, label: str) -> Path:
    relative_path = Path(str(relative))
    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or len(relative_path.parts) != 1
    ):
        raise ValueError(f"official-v2 {label} path is unsafe")
    return root / relative_path


def _official_v2_freeze_evidence(
    freeze_manifest: Path,
    freeze: Mapping[str, Any],
) -> dict[str, Path]:
    """Validate aggregate projection evidence without opening the holdout."""
    freeze_manifest = Path(freeze_manifest).resolve()
    frozen_root = freeze_manifest.parent
    control_root = frozen_root.parent
    output_root, rid = _scoped_output_root_and_rid(
        freeze_manifest,
        scope="private",
    )
    partition = freeze.get("partition")
    independence = freeze.get("independence")
    privacy = freeze.get("privacy")
    if (
        not isinstance(partition, Mapping)
        or partition.get("version") != "v2"
        or partition.get("development_holdout_normalised_question_intersection")
        != 0
        or partition.get("development_holdout_skeleton_intersection") != 0
        or not isinstance(independence, Mapping)
        or independence.get("official_test_content_opened_before_marker")
        is not False
        or independence.get("official_test_content_opened_after_marker")
        is not True
        or independence.get("project_nl2sql_engine_imported_by_projector")
        is not False
        or independence.get("holdout_used_for_selection") is not False
        or independence.get("holdout_reusable_after_failure") is not False
        or not isinstance(privacy, Mapping)
        or privacy.get("manifest_contains_questions") is not False
        or privacy.get("manifest_contains_gold_sql") is not False
        or privacy.get("manifest_contains_error_samples") is not False
        or privacy.get("official_source_or_projection_allowed_in_submission")
        is not False
    ):
        raise ValueError("official-v2 freeze isolation contract is invalid")

    development = freeze.get("development")
    holdout = freeze.get("holdout")
    if not isinstance(development, Mapping) or not isinstance(holdout, Mapping):
        raise ValueError("official-v2 freeze lacks development/holdout bindings")
    development_path = _safe_frozen_member(
        frozen_root,
        development.get("path"),
        "development",
    )
    if (
        not development_path.is_file()
        or development_path.stat().st_size != development.get("size")
        or sha256_file(development_path) != development.get("sha256")
    ):
        raise ValueError("official-v2 frozen development binding failed")
    # The sealer may stat the holdout but must never open or hash it.  Its
    # content identity was created after the projection marker and is carried
    # by the immutable freeze/completion attestations.
    holdout_path = _safe_frozen_member(
        frozen_root,
        holdout.get("path"),
        "holdout",
    )
    if (
        not re.fullmatch(r"[0-9a-f]{64}", str(holdout.get("sha256", "")))
        or not holdout_path.is_file()
        or holdout_path.stat().st_size != holdout.get("size")
    ):
        raise ValueError("official-v2 holdout declaration or byte count failed")

    evidence = freeze.get("evidence")
    started_binding = (
        evidence.get("projection_started")
        if isinstance(evidence, Mapping)
        else None
    )
    started = control_root / "projection-started.json"
    completed = control_root / "projection-completed.json"
    invalid = control_root / "projection-invalid.json"
    if (
        not isinstance(started_binding, Mapping)
        or Path(str(started_binding.get("path", ""))).resolve()
        != started.resolve()
        or not started.is_file()
        or started.stat().st_size != started_binding.get("size")
        or sha256_file(started) != started_binding.get("sha256")
        or invalid.exists()
        or not completed.is_file()
    ):
        raise ValueError("official-v2 projection control evidence is invalid")
    started_payload = _load_receipt(started, "official-v2 projection start")
    completed_payload = _load_receipt(
        completed,
        "official-v2 projection completion",
    )
    if (
        started_payload.get("status") != "projection-started"
        or started_payload.get("rid") != rid
        or started_payload.get(
            "official_test_content_opened_before_this_marker"
        )
        is not False
        or started_payload.get("reuse_permitted_after_this_marker") is not False
        or completed_payload.get("status") != "projection-completed"
        or completed_payload.get("rid") != rid
        or completed_payload.get("projection_started_sha256")
        != sha256_file(started)
        or completed_payload.get("freeze_manifest_sha256")
        != sha256_file(freeze_manifest)
        or completed_payload.get("n_holdout") != freeze.get("n_holdout")
        or completed_payload.get("n_hard") != freeze.get("n_hard")
        or completed_payload.get("selection_used") is not False
    ):
        raise ValueError("official-v2 projection completion binding failed")

    public_attestation = (
        output_root
        / "outputs"
        / "remote_runs"
        / rid
        / "nl2sql"
        / "official_v2_freeze_attestation.json"
    )
    public = _load_receipt(
        public_attestation,
        "official-v2 public freeze attestation",
    )
    counts = public.get("counts")
    if (
        sha256_file(public_attestation)
        != completed_payload.get("public_attestation_sha256")
        or public.get("kind")
        != "official_public_v2_blind_freeze_attestation"
        or public.get("status") != "frozen-unconsumed"
        or public.get("rid") != rid
        or public.get("freeze_manifest_sha256")
        != sha256_file(freeze_manifest)
        or public.get("selection_used") is not False
        or public.get("official_test_content_opened_after_marker") is not True
        or public.get("contains_questions") is not False
        or public.get("contains_gold_sql") is not False
        or public.get("contains_predictions") is not False
        or public.get("contains_error_samples") is not False
        or not isinstance(counts, Mapping)
        or counts.get("n_development") != freeze.get("n_development")
        or counts.get("n_holdout") != freeze.get("n_holdout")
        or counts.get("n_hard") != freeze.get("n_hard")
    ):
        raise ValueError("official-v2 public freeze attestation is invalid")
    return {
        "projection_started": started,
        "projection_completed": completed,
        "public_freeze_attestation": public_attestation,
    }


def _mimicsql_v3_freeze_evidence(
    freeze_manifest: Path,
    freeze: Mapping[str, Any],
) -> dict[str, Path]:
    """Validate MIMICSQL aggregate evidence without opening its holdout."""
    freeze_manifest = Path(freeze_manifest).resolve()
    frozen_root = freeze_manifest.parent
    control_root = frozen_root.parent
    output_root, rid = _scoped_output_root_and_rid(
        freeze_manifest,
        scope="private",
    )
    partition = freeze.get("partition")
    independence = freeze.get("independence")
    privacy = freeze.get("privacy")
    benchmark = freeze.get("benchmark")
    if (
        not isinstance(partition, Mapping)
        or partition.get("version") != "v3"
        or partition.get(
            "development_holdout_normalised_question_intersection"
        )
        != 0
        or partition.get("development_holdout_skeleton_intersection")
        != 0
        or not isinstance(independence, Mapping)
        or independence.get("official_test_content_opened_before_marker")
        is not False
        or independence.get("official_test_content_opened_after_marker")
        is not True
        or independence.get("project_nl2sql_engine_imported_by_projector")
        is not False
        or independence.get("holdout_used_for_selection") is not False
        or independence.get("holdout_reusable_after_failure") is not False
        or not isinstance(privacy, Mapping)
        or privacy.get("manifest_contains_questions") is not False
        or privacy.get("manifest_contains_gold_sql") is not False
        or privacy.get("manifest_contains_error_samples") is not False
        or privacy.get("official_source_or_projection_allowed_in_submission")
        is not False
        or not isinstance(benchmark, Mapping)
        or benchmark.get("name") != "MIMICSQL-natural-v2"
        or not re.fullmatch(
            r"[0-9a-f]{40}",
            str(benchmark.get("source_revision", "")),
        )
    ):
        raise ValueError("MIMICSQL-v3 freeze isolation contract is invalid")

    development = freeze.get("development")
    holdout = freeze.get("holdout")
    if not isinstance(development, Mapping) or not isinstance(
        holdout,
        Mapping,
    ):
        raise ValueError(
            "MIMICSQL-v3 freeze lacks development/holdout bindings"
        )
    development_path = _safe_frozen_member(
        frozen_root,
        development.get("path"),
        "MIMICSQL development",
    )
    if (
        not development_path.is_file()
        or development_path.stat().st_size != development.get("size")
        or sha256_file(development_path) != development.get("sha256")
    ):
        raise ValueError("MIMICSQL-v3 frozen development binding failed")
    # Do not open or hash the holdout here.  Its identity is carried by the
    # create-once freeze/completion chain and checked only after the evaluator
    # has published consumed-started.json.
    holdout_path = _safe_frozen_member(
        frozen_root,
        holdout.get("path"),
        "MIMICSQL holdout",
    )
    if (
        not re.fullmatch(r"[0-9a-f]{64}", str(holdout.get("sha256", "")))
        or not holdout_path.is_file()
        or holdout_path.stat().st_size != holdout.get("size")
    ):
        raise ValueError("MIMICSQL-v3 holdout declaration failed")

    evidence = freeze.get("evidence")
    started_binding = (
        evidence.get("projection_started")
        if isinstance(evidence, Mapping)
        else None
    )
    started = control_root / "projection-started.json"
    completed = control_root / "projection-completed.json"
    invalid = control_root / "projection-invalid.json"
    if (
        not isinstance(started_binding, Mapping)
        or Path(str(started_binding.get("path", ""))).resolve()
        != started.resolve()
        or not started.is_file()
        or started.stat().st_size != started_binding.get("size")
        or sha256_file(started) != started_binding.get("sha256")
        or invalid.exists()
        or not completed.is_file()
    ):
        raise ValueError("MIMICSQL-v3 projection control evidence is invalid")
    started_payload = _load_receipt(
        started,
        "MIMICSQL-v3 projection start",
    )
    completed_payload = _load_receipt(
        completed,
        "MIMICSQL-v3 projection completion",
    )
    if (
        started_payload.get("status") != "projection-started"
        or started_payload.get("rid") != rid
        or started_payload.get(
            "official_test_content_opened_before_this_marker"
        )
        is not False
        or started_payload.get("reuse_permitted_after_this_marker")
        is not False
        or completed_payload.get("status") != "projection-completed"
        or completed_payload.get("rid") != rid
        or completed_payload.get("projection_started_sha256")
        != sha256_file(started)
        or completed_payload.get("freeze_manifest_sha256")
        != sha256_file(freeze_manifest)
        or completed_payload.get("n_holdout") != freeze.get("n_holdout")
        or completed_payload.get("n_hard") != freeze.get("n_hard")
        or completed_payload.get("selection_used") is not False
    ):
        raise ValueError("MIMICSQL-v3 projection completion binding failed")

    public_attestation = (
        output_root
        / "outputs"
        / "remote_runs"
        / rid
        / "nl2sql"
        / "mimicsql_official_freeze_attestation.json"
    )
    public = _load_receipt(
        public_attestation,
        "MIMICSQL-v3 public freeze attestation",
    )
    counts = public.get("counts")
    if (
        sha256_file(public_attestation)
        != completed_payload.get("public_attestation_sha256")
        or public.get("kind")
        != "mimicsql_v3_official_blind_freeze_attestation"
        or public.get("status") != "frozen-unconsumed"
        or public.get("rid") != rid
        or public.get("freeze_manifest_sha256")
        != sha256_file(freeze_manifest)
        or public.get("selection_used") is not False
        or public.get("official_test_content_opened_after_marker")
        is not True
        or public.get("contains_questions") is not False
        or public.get("contains_gold_sql") is not False
        or public.get("contains_predictions") is not False
        or public.get("contains_error_samples") is not False
        or not isinstance(counts, Mapping)
        or counts.get("n_development") != freeze.get("n_development")
        or counts.get("n_holdout") != freeze.get("n_holdout")
        or counts.get("n_hard") != freeze.get("n_hard")
    ):
        raise ValueError(
            "MIMICSQL-v3 public freeze attestation is invalid"
        )
    return {
        "mimicsql_projection_started": started,
        "mimicsql_projection_completed": completed,
        "mimicsql_public_freeze_attestation": public_attestation,
    }


def _independent_v3_freeze_evidence(
    freeze_manifest: Path,
    freeze: Mapping[str, Any],
) -> dict[str, Path]:
    """Validate independent-curator evidence without opening the holdout."""
    freeze_manifest = Path(freeze_manifest).resolve()
    frozen_root = freeze_manifest.parent
    output_root, rid = _scoped_output_root_and_rid(
        freeze_manifest,
        scope="private",
    )
    if freeze.get("rid") != rid:
        raise ValueError("independent-v3 freeze RID binding failed")
    partition = freeze.get("partition")
    privacy = freeze.get("privacy")
    curation = freeze.get("independent_curation")
    declaration = (
        curation.get("declaration")
        if isinstance(curation, Mapping)
        else None
    )
    partition_is_v3 = (
        isinstance(partition, Mapping)
        and partition.get("version") == "v3"
        and partition.get("normalised_question_intersection") == 0
        and partition.get("sql_ast_skeleton_intersection") == 0
        and partition.get("whole_ast_groups_only") is True
    )
    partition_is_v4 = (
        isinstance(partition, Mapping)
        and partition.get("version") == "v4"
        and partition.get("normalised_question_intersection") == 0
        and partition.get("near_duplicate_question_intersection") == 0
        and partition.get("sql_ast_skeleton_intersection") == 0
        and partition.get("whole_ast_groups_only") is True
        and partition.get("whole_partition_families_only") is True
        and isinstance(partition.get("near_duplicate_policy"), Mapping)
        and partition["near_duplicate_policy"].get(
            "character_3gram_jaccard_gte"
        ) == 0.85
        and partition["near_duplicate_policy"].get(
            "sequence_matcher_ratio_gte"
        ) == 0.90
        and partition["near_duplicate_policy"].get("transitive_union_find")
        is True
    )
    if (
        not (partition_is_v3 or partition_is_v4)
        or not isinstance(privacy, Mapping)
        or privacy.get("manifest_contains_questions") is not False
        or privacy.get("manifest_contains_gold_sql") is not False
        or privacy.get("manifest_contains_predictions") is not False
        or privacy.get("manifest_contains_error_samples") is not False
        or privacy.get(
            "raw_corpus_or_frozen_rows_allowed_in_submission"
        )
        is not False
        or not isinstance(curation, Mapping)
        or not isinstance(declaration, Mapping)
        or freeze.get("freeze_tool_sha256")
        != sha256_file(ROOT / "scripts" / "freeze_nl2sql_blind.py")
        or freeze.get("protocol_sha256")
        != sha256_file(ROOT / "dki" / "analysis" / "nl2sql_protocol.py")
    ):
        raise ValueError(
            "independent-v3 curation/isolation contract is invalid"
        )
    curator_format = str(
        curation.get("curator_format", "legacy-pack")
    )
    if curator_format == "legacy-pack":
        if (
            declaration.get("kind")
            != "independent_four_table_nl2sql_curator_attestation"
            or declaration.get("independent_curation") is not True
            or declaration.get("all_gold_sql_executed") is not True
            or declaration.get("all_pairs_semantically_reviewed") is not True
            or declaration.get("engine_context_provided") is not False
            or declaration.get("local_evaluation_context_provided") is not False
            or declaration.get("project_local_question_reuse") is not False
            or declaration.get("project_local_question_paraphrase") is not False
            or declaration.get("development_or_holdout_preassigned") is not False
            or declaration.get("corpus_sha256") != freeze.get("input_sha256")
        ):
            raise ValueError(
                "independent-v3 curation/isolation contract is invalid"
            )
        curator_identity = str(
            declaration.get("curator_identity", "")
        ).strip()
        reviewer_identity = str(
            declaration.get("semantic_reviewer_identity", "")
        ).strip()
        if (
            not curator_identity
            or not reviewer_identity
            or curator_identity == reviewer_identity
        ):
            raise ValueError("independent-v3 reviewer identities are invalid")
    elif curator_format in INDEPENDENT_FIVE_FILE_CURATOR_FORMATS:
        independence = declaration.get("independence_declarations")
        execution = declaration.get("execution_attestation")
        corpus = declaration.get("corpus")
        database = declaration.get("database")
        false_aliases = (
            ("implementation_or_source_code_accessed",),
            ("nl2sql_implementation_accessed",),
            ("tests_accessed",),
            ("existing_nl2sql_eval_accessed",),
            ("outputs_private_accessed",),
            ("prior_curator_files_accessed", "other_curator_files_accessed"),
            ("prior_holdout_accessed",),
            ("failure_details_accessed", "failure_detail_files_accessed"),
            ("conversation_records_accessed",),
            (
                "existing_questions_copied_or_paraphrased",
                "external_existing_questions_copied_or_paraphrased",
            ),
        )
        freeze_database = freeze.get("database")
        if (
            declaration.get("source_version")
            not in INDEPENDENT_FIVE_FILE_CURATOR_SOURCE_VERSIONS
            or not isinstance(independence, Mapping)
            or any(
                not any(
                    alias in independence and independence.get(alias) is False
                    for alias in aliases
                )
                for aliases in false_aliases
            )
            or independence.get(
                "database_schema_and_self_generated_query_results_only"
            )
            is not True
            or not isinstance(execution, Mapping)
            or execution.get("all_queries_executed_successfully") is not True
            or execution.get("all_results_fully_materialized") is not True
            or execution.get("single_read_only_sqlite_query_per_record")
            is not True
            or execution.get("allowed_tables_only") is not True
            or not isinstance(corpus, Mapping)
            or corpus.get("sha256") != freeze.get("input_sha256")
            or corpus.get("record_count") != freeze.get("n_total")
            or not isinstance(database, Mapping)
            or not isinstance(freeze_database, Mapping)
            or database.get("sha256") != freeze_database.get("sha256")
        ):
            raise ValueError(
                "independent-v3 curation/isolation contract is invalid"
            )
    else:
        raise ValueError("independent-v3 curator format is unsupported")

    pack_path = Path(
        str(curation.get("curator_pack_manifest_path", ""))
    ).resolve()
    attestation_path = Path(
        str(curation.get("attestation_path", ""))
    ).resolve()
    if (
        not pack_path.is_file()
        or sha256_file(pack_path)
        != curation.get("curator_pack_manifest_sha256")
        or not attestation_path.is_file()
        or sha256_file(attestation_path)
        != curation.get("attestation_sha256")
    ):
        raise ValueError(
            "independent-v3 curator pack/attestation binding failed"
        )
    attestation_payload = _load_receipt(
        attestation_path,
        "independent-v3 curator attestation",
    )
    if curator_format == "legacy-pack":
        pack_payload = _load_receipt(
            pack_path,
            "independent-v3 curator pack manifest",
        )
        if (
            declaration.get("curator_pack_manifest_sha256")
            != curation.get("curator_pack_manifest_sha256")
            or pack_payload.get("kind") != "engine_blind_nl2sql_curator_pack"
            or pack_payload.get("dataset_role")
            != "independent_curation_input_only"
            or pack_payload.get("created_once") is not True
            or pack_payload.get("contains_engine_implementation") is not False
            or pack_payload.get("contains_questions_or_gold_sql") is not False
            or attestation_payload != declaration
        ):
            raise ValueError(
                "independent-v3 curator declaration differs from evidence"
            )
    else:
        def valid_audit(binding: object) -> bool:
            if not isinstance(binding, Mapping):
                return False
            path = Path(str(binding.get("path", ""))).resolve()
            return (
                path.is_file()
                and path.stat().st_size == binding.get("size")
                and sha256_file(path) == binding.get("sha256")
            )

        if (
            curation.get("pack_dataset_role")
            != "independent_curation_input_only"
            or attestation_payload != declaration
            or not valid_audit(curation.get("aggregate_audit"))
            or not valid_audit(curation.get("execution_audit"))
            or (
                curation.get("sha256s") is not None
                and not valid_audit(curation.get("sha256s"))
            )
        ):
            raise ValueError(
                "independent-v3 curator declaration differs from evidence"
            )

    development = freeze.get("development")
    holdout = freeze.get("holdout")
    if not isinstance(development, Mapping) or not isinstance(
        holdout,
        Mapping,
    ):
        raise ValueError(
            "independent-v3 freeze lacks development/holdout bindings"
        )
    development_path = _safe_frozen_member(
        frozen_root,
        development.get("path"),
        "independent-v3 development",
    )
    if (
        not development_path.is_file()
        or development_path.stat().st_size != development.get("size")
        or sha256_file(development_path) != development.get("sha256")
    ):
        raise ValueError(
            "independent-v3 frozen development binding failed"
        )
    # Stat only.  Hashing or parsing the holdout is reserved for the
    # evaluator after consumed-started.json has been published.
    holdout_path = _safe_frozen_member(
        frozen_root,
        holdout.get("path"),
        "independent-v3 holdout",
    )
    if (
        not re.fullmatch(r"[0-9a-f]{64}", str(holdout.get("sha256", "")))
        or not holdout_path.is_file()
        or holdout_path.stat().st_size != holdout.get("size")
    ):
        raise ValueError("independent-v3 holdout declaration failed")

    public_attestation = (
        output_root
        / "outputs"
        / "remote_runs"
        / rid
        / "nl2sql"
        / "independent_v3_freeze_attestation.json"
    )
    public = _load_receipt(
        public_attestation,
        "independent-v3 public freeze attestation",
    )
    counts = public.get("counts")
    bindings = public.get("bindings")
    if (
        public.get("kind")
        != "independent_curated_v3_blind_freeze_attestation"
        or public.get("status") != "frozen-unconsumed"
        or public.get("rid") != rid
        or public.get("source_tree_sha256") != freeze.get("source_version")
        or public.get("freeze_manifest_sha256")
        != sha256_file(freeze_manifest)
        or public.get("selection_used") is not False
        or public.get("contains_questions") is not False
        or public.get("contains_gold_sql") is not False
        or public.get("contains_predictions") is not False
        or public.get("contains_error_samples") is not False
        or not isinstance(counts, Mapping)
        or counts.get("n_total") != freeze.get("n_total")
        or counts.get("n_development") != freeze.get("n_development")
        or counts.get("n_holdout") != freeze.get("n_holdout")
        or counts.get("n_hard") != freeze.get("n_hard")
        or counts.get("n_sql_ast_groups")
        != freeze.get("n_sql_ast_groups")
        or not isinstance(bindings, Mapping)
        or bindings.get("curator_pack_manifest_sha256")
        != curation.get("curator_pack_manifest_sha256")
        or bindings.get("curator_attestation_sha256")
        != curation.get("attestation_sha256")
        or bindings.get("corpus_sha256") != freeze.get("input_sha256")
        or bindings.get("task1_manifest_sha256")
        != freeze.get("task1_manifest", {}).get("sha256")
        or bindings.get("kg_sha256") != freeze.get("kg", {}).get("sha256")
        or bindings.get("database_sha256")
        != freeze.get("database", {}).get("sha256")
        or (
            curation.get("sha256s") is not None
            and bindings.get("curator_sha256s_sha256")
            != curation.get("sha256s", {}).get("sha256")
        )
    ):
        raise ValueError(
            "independent-v3 public freeze attestation is invalid"
        )
    return {
        "independent_v3_public_freeze_attestation": public_attestation,
        "independent_v3_curator_pack_manifest": pack_path,
        "independent_v3_curator_attestation": attestation_path,
    }


def _environment() -> dict[str, str]:
    try:
        sqlglot_version = importlib.metadata.version("sqlglot")
    except importlib.metadata.PackageNotFoundError:
        sqlglot_version = ""
    if not sqlglot_version:
        raise RuntimeError("sqlglot must be installed before sealing")
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "sqlglot": sqlglot_version,
    }


def _agent_runtime_profile(
    llm_strategy: str = "legacy_sql",
    agent_rule_policy: str = "rule_first",
    llm_reasoning_mode: str = "disabled",
    llm_completion_tokens: int = 1024,
) -> dict[str, Any]:
    """Return the secret-free, hash-bound Agent inference profile.

    The API key is required to exist at seal/evaluation time but is never
    copied into an evidence artifact.  The endpoint is represented only by a
    SHA-256 digest so public receipts bind the provider without publishing a
    private gateway URL.  The profile schema remains v1: the strategy and
    inference-control fields are additive bindings, while QueryPlan-specific
    bindings are present only when that strategy is selected.
    """

    from dki.paths import LLM_API_KEY, LLM_API_URL, LLM_MODEL_NAME

    if llm_strategy not in {
        "legacy_sql",
        "query_plan",
        "query_plan_v2",
        "query_plan_v3",
    }:
        raise ValueError("unsupported Agent LLM strategy")
    if agent_rule_policy not in {"rule_first", "model_first", "plan_first"}:
        raise ValueError("unsupported Agent rule policy")
    if (llm_strategy, agent_rule_policy) not in {
        ("legacy_sql", "rule_first"),
        ("legacy_sql", "model_first"),
        ("query_plan", "rule_first"),
        ("query_plan", "plan_first"),
        ("query_plan_v2", "rule_first"),
        ("query_plan_v2", "plan_first"),
        ("query_plan_v3", "rule_first"),
        ("query_plan_v3", "plan_first"),
    }:
        raise ValueError("unsupported Agent LLM strategy/rule-policy combination")
    if llm_reasoning_mode not in {"disabled", "adaptive"}:
        raise ValueError("unsupported Agent LLM reasoning mode")
    if (
        type(llm_completion_tokens) is not int
        or not 1 <= llm_completion_tokens <= 131072
    ):
        raise ValueError(
            "Agent LLM completion tokens must be an integer between 1 and 131072"
        )
    if not (LLM_API_KEY and LLM_API_URL and LLM_MODEL_NAME):
        raise RuntimeError(
            "engine=agent requires LLM_API_KEY, LLM_API_URL and LLM_MODEL_NAME"
        )
    endpoint = LLM_API_URL.strip()
    model = LLM_MODEL_NAME.strip()
    lowered_endpoint = endpoint.casefold()
    lowered_model = model.casefold()
    if "xiaomimimo" in lowered_endpoint or lowered_model.startswith("mimo-"):
        provider_profile = (
            "mimo_adaptive_reasoning"
            if llm_reasoning_mode == "adaptive"
            else "mimo_thinking_disabled"
        )
        reasoning_mode_effective = llm_reasoning_mode
        reasoning_control_supported = True
    elif "minimax" in lowered_endpoint or lowered_model.startswith("minimax-"):
        if lowered_model.startswith("minimax-m3"):
            provider_profile = "minimax_m3_controlled_reasoning"
            reasoning_mode_effective = llm_reasoning_mode
            reasoning_control_supported = True
        else:
            # MiniMax M2.x always reasons and exposes only separated reasoning
            # content.  Record that provider reality instead of claiming the
            # requested adaptive/disabled switch was actually enforced.
            provider_profile = "minimax_m2_provider_default_reasoning"
            reasoning_mode_effective = "provider_default_reasoning"
            reasoning_control_supported = False
    else:
        provider_profile = "openai_compatible"
        reasoning_mode_effective = "provider_default_or_opaque"
        reasoning_control_supported = False
    profile = {
        "schema_version": 1,
        "engine": "agent",
        "llm_strategy": llm_strategy,
        "agent_rule_policy": agent_rule_policy,
        "llm_reasoning_mode": llm_reasoning_mode,
        "llm_reasoning_mode_effective": reasoning_mode_effective,
        "llm_reasoning_control_supported": reasoning_control_supported,
        "llm_completion_tokens": llm_completion_tokens,
        "model_name": model,
        "api_url_sha256": hashlib.sha256(endpoint.encode("utf-8")).hexdigest(),
        "provider_profile": provider_profile,
        "llm_samples": 1,
        "llm_rounds": 2,
        "llm_temperature": 0.0,
        "runtime_error_policy": "fail_closed",
        "engine_source_sha256": sha256_file(
            ROOT / "dki" / "analysis" / "nl2sql.py"
        ),
        "prompt_source_sha256": sha256_file(
            ROOT / "dki" / "analysis" / "nl2sql.py"
        ),
        "provider_adapter_sha256": sha256_file(ROOT / "dki" / "llm.py"),
        "contains_api_key": False,
    }
    if llm_strategy in {"query_plan", "query_plan_v2"}:
        query_plan_source = (
            ROOT / "dki" / "analysis" / "nl2sql_query_plan_v2.py"
            if llm_strategy == "query_plan_v2"
            else ROOT / "dki" / "analysis" / "nl2sql_query_plan.py"
        )
        profile.update(
            {
                "query_plan_schema_version": 2 if llm_strategy == "query_plan_v2" else 1,
                "query_plan_source_sha256": sha256_file(query_plan_source),
            }
        )
    elif llm_strategy == "query_plan_v3":
        # QueryPlan v3 compiles through the bounded v2 grammar and selects
        # among multi-lane candidates, so the seal binds that full closure.
        profile.update(
            {
                "query_plan_schema_version": 3,
                "query_plan_source_sha256": sha256_file(
                    ROOT / "dki" / "analysis" / "nl2sql_query_plan_v3.py"
                ),
                "query_plan_v2_source_sha256": sha256_file(
                    ROOT / "dki" / "analysis" / "nl2sql_query_plan_v2.py"
                ),
                "query_plan_v3_pipeline_sha256": sha256_file(
                    ROOT / "dki" / "analysis" / "nl2sql_query_plan_v3_pipeline.py"
                ),
                "candidate_selector_sha256": sha256_file(
                    ROOT / "dki" / "analysis" / "nl2sql_candidate_selector.py"
                ),
                "semantic_scorer_sha256": sha256_file(
                    ROOT / "dki" / "analysis" / "nl2sql_semantic_scorer.py"
                ),
                "counterfactual_sha256": sha256_file(
                    ROOT / "dki" / "analysis" / "nl2sql_counterfactual.py"
                ),
                "schema_graph_sha256": sha256_file(
                    ROOT / "dki" / "analysis" / "nl2sql_schema_graph.py"
                ),
            }
        )
    return profile


def _runtime_profile_for_engine(
    *,
    engine: str,
    llm_strategy: str = "legacy_sql",
    agent_rule_policy: str = "rule_first",
    llm_reasoning_mode: str = "disabled",
    llm_completion_tokens: int = 1024,
) -> dict[str, Any] | None:
    """Resolve the exact secret-free inference profile selected for a gate.

    Rule candidates must not carry dormant Agent overrides.  Agent candidates
    are bound to the same four inference controls used by development and
    blind evaluation, in addition to the provider/model hashes produced by
    :func:`_agent_runtime_profile`.
    """

    if engine not in {"rule", "agent"}:
        raise ValueError("prospective blind evaluation requires engine=rule or agent")
    if engine == "rule":
        if (
            llm_strategy != "legacy_sql"
            or agent_rule_policy != "rule_first"
            or llm_reasoning_mode != "disabled"
            or llm_completion_tokens != 1024
        ):
            raise ValueError("non-default Agent runtime options require engine=agent")
        return None
    return _agent_runtime_profile(
        llm_strategy=llm_strategy,
        agent_rule_policy=agent_rule_policy,
        llm_reasoning_mode=llm_reasoning_mode,
        llm_completion_tokens=llm_completion_tokens,
    )


def _load_receipt(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = strict_json_loads(
            Path(path).read_bytes(),
            label=f"{label} receipt",
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} receipt is unreadable") from exc
    if type(payload) is not dict:
        raise ValueError(f"{label} receipt must be a JSON object")
    return payload


def _require_nonnegative_int(value: Any, label: str) -> int:
    """Reject numeric coercion, booleans, and negative gate counts."""
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _difficulty_counts(
    overall: Mapping[str, Any],
) -> dict[str, tuple[int, int]]:
    layers = overall.get("by_difficulty")
    if not isinstance(layers, Mapping):
        raise ValueError("development receipt lacks difficulty layers")
    counts: dict[str, tuple[int, int]] = {}
    for raw_name, layer in layers.items():
        if type(raw_name) is not str or not raw_name:
            raise ValueError("development difficulty layer name is invalid")
        name = raw_name
        if not isinstance(layer, Mapping):
            raise ValueError(f"development difficulty layer is invalid: {name}")
        n = _require_nonnegative_int(
            layer.get("n"),
            f"development difficulty layer {name}.n",
        )
        correct = _require_nonnegative_int(
            layer.get("correct"),
            f"development difficulty layer {name}.correct",
        )
        if correct > n:
            raise ValueError(
                f"development difficulty layer has impossible counts: {name}"
            )
        counts[name] = (n, correct)
    return counts


def _development_quality_gate(overall: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the prospective development gate using exact integer ratios.

    Displayed decimal metrics are intentionally ignored: all boundaries are
    checked from counts so rounding cannot promote a failing candidate.  The
    stable failure codes make the pre-consumption reason actionable without
    exposing any development question or SQL text.
    """

    n = _require_nonnegative_int(overall.get("n"), "development overall.n")
    correct = _require_nonnegative_int(
        overall.get("n_correct"),
        "development overall.n_correct",
    )
    answered = _require_nonnegative_int(
        overall.get("n_answered"),
        "development overall.n_answered",
    )
    if correct > answered or answered > n:
        raise ValueError("development receipt overall counts are impossible")

    layers = _difficulty_counts(overall)
    layer_n = sum(value[0] for value in layers.values())
    layer_correct = sum(value[1] for value in layers.values())
    if layer_n != n or layer_correct != correct:
        raise ValueError(
            "development receipt difficulty totals differ from overall counts"
        )

    hard_n = sum(
        layers.get(name, (0, 0))[0]
        for name in ("hard", "extra", "extra-hard")
    )
    hard_correct = sum(
        layers.get(name, (0, 0))[1]
        for name in ("hard", "extra", "extra-hard")
    )
    extra_hard_n, extra_hard_correct = layers.get("extra-hard", (0, 0))

    active_gate = ORIGINAL_DEVELOPMENT_GATE
    failures: list[str] = []
    if n < active_gate["minimum_n"]:
        failures.append(
            f"development_n_below_{active_gate['minimum_n']}"
        )
    if correct * 100 < active_gate["overall_minimum_percent"] * n:
        failures.append(
            "development_overall_below_"
            f"0_{active_gate['overall_minimum_percent']:02d}"
        )
    if answered * 100 < active_gate["coverage_minimum_percent"] * n:
        failures.append(
            "development_coverage_below_"
            f"0_{active_gate['coverage_minimum_percent']:02d}"
        )
    if hard_n == 0:
        failures.append("development_hard_plus_missing")
    elif (
        hard_correct * 100
        < active_gate["hard_plus_minimum_percent"] * hard_n
    ):
        failures.append(
            "development_hard_plus_below_"
            f"0_{active_gate['hard_plus_minimum_percent']:02d}"
        )
    if extra_hard_n == 0:
        failures.append("development_extra_hard_missing")
    elif (
        extra_hard_correct * 100
        < active_gate["extra_hard_minimum_percent"] * extra_hard_n
    ):
        failures.append(
            "development_extra_hard_below_"
            f"0_{active_gate['extra_hard_minimum_percent']:02d}"
        )
    if failures:
        raise ValueError("development gate failed: " + ",".join(failures))

    return {
        "n": n,
        "n_correct": correct,
        "n_answered": answered,
        "hard_n": hard_n,
        "hard_correct": hard_correct,
        "extra_hard_n": extra_hard_n,
        "extra_hard_correct": extra_hard_correct,
        "gate_thresholds": dict(ORIGINAL_DEVELOPMENT_GATE),
        "original_gate_thresholds": dict(ORIGINAL_DEVELOPMENT_GATE),
        "original_gate_passed": True,
        "release_eligible": True,
        "gate_override": None,
    }


def _validate_development_gate(
    path: Path,
    *,
    freeze: Mapping[str, Any],
    db_path: Path,
    source_tree_sha256: str,
    freeze_manifest: Path | None = None,
    candidate_rid: str | None = None,
    engine: str = "rule",
    agent_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate either the anchored legacy or recomputed independent gate."""
    payload = _load_receipt(path, "development gate")
    if (
        payload.get("development_evidence_kind")
        == "independent_curated_v3_four_table_development_receipt"
    ):
        if freeze_manifest is None:
            raise ValueError(
                "independent development receipt requires its freeze manifest"
            )
        return _validate_independent_development_gate(
            payload,
            freeze=freeze,
            freeze_manifest=freeze_manifest,
            db_path=db_path,
            source_tree_sha256=source_tree_sha256,
            candidate_rid=candidate_rid,
            engine=engine,
            agent_profile=agent_profile,
        )
    overall = payload.get("overall")
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
        or payload.get("dataset_role") != "development_selection_used"
        or payload.get("selection_used") is not True
        or payload.get("created_once") is not True
        or engine != "rule"
        or payload.get("engine") != "rule"
        or payload.get("development_evidence_kind")
        != "official_public_v2_four_table_development_receipt"
        or not isinstance(overall, Mapping)
        or overall.get("strict_result_contract") is not True
    ):
        raise ValueError("development receipt does not prove the strict rule gate")
    try:
        anchor = anchor_from_trusted_evidence_binding(
            payload.get("trusted_evidence")
        )
    except ValueError as exc:
        raise ValueError("development receipt lacks a valid trusted evidence binding") from exc

    source_tree_sha256 = require_lowercase_sha256(
        source_tree_sha256,
        "candidate source tree SHA-256",
    )
    development = freeze.get("development")
    database = freeze.get("database")
    if not isinstance(development, Mapping) or not isinstance(database, Mapping):
        raise ValueError("blind freeze lacks its paired development/database binding")
    freeze_input_sha256 = require_lowercase_sha256(
        development.get("sha256"),
        "blind freeze development SHA-256",
    )
    freeze_database_sha256 = require_lowercase_sha256(
        database.get("sha256"),
        "blind freeze database SHA-256",
    )
    receipt_input_sha256 = require_lowercase_sha256(
        payload.get("input_sha256"),
        "development receipt input SHA-256",
    )
    receipt_database_sha256 = require_lowercase_sha256(
        payload.get("database_sha256"),
        "development receipt database SHA-256",
    )
    receipt_evidence_sha256 = require_lowercase_sha256(
        payload.get("development_evidence_sha256"),
        "development receipt evidence SHA-256",
    )
    receipt_engine_sha256 = require_lowercase_sha256(
        payload.get("engine_source_sha256"),
        "development receipt engine SHA-256",
    )
    receipt_source_tree_sha256 = require_lowercase_sha256(
        payload.get("source_tree_sha256"),
        "development receipt source tree SHA-256",
    )
    if (
        type(payload.get("rid")) is not str
        or payload["rid"] != anchor.rid
        or receipt_input_sha256 != anchor.private_development_sha256
        or freeze_input_sha256 != anchor.private_development_sha256
        or receipt_database_sha256 != anchor.target_db_sha256
        or freeze_database_sha256 != anchor.target_db_sha256
        or sha256_file(db_path) != anchor.target_db_sha256
        or receipt_evidence_sha256 != anchor.receipt_sha256
        or receipt_engine_sha256
        != sha256_file(ROOT / "dki" / "analysis" / "nl2sql.py")
        or receipt_source_tree_sha256 != source_tree_sha256
    ):
        raise ValueError("development receipt differs from its trusted anchor or sealed candidate")

    quality_gate = _development_quality_gate(overall)
    return {
        **quality_gate,
        "trusted_evidence": trusted_evidence_binding(anchor),
    }


def _validate_independent_development_gate(
    payload: Mapping[str, Any],
    *,
    freeze: Mapping[str, Any],
    freeze_manifest: Path,
    db_path: Path,
    source_tree_sha256: str,
    candidate_rid: str | None,
    engine: str,
    agent_profile: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Recompute the frozen independent development gate before sealing.

    An independent freeze is already a create-once, curator-attested split.
    Unlike the historical official-v2 path, it therefore does not need a
    source-controlled receipt allowlist.  The sealer still refuses to trust
    self-reported metrics: it opens only the frozen development member,
    validates every row, runs the sealed rule engine again, and requires the
    receipt aggregate to equal that recomputation exactly.  The holdout is
    never opened or hashed here.
    """
    overall = payload.get("overall")
    schema_version = payload.get("schema_version")
    freeze_rid = freeze.get("rid")
    effective_candidate_rid = candidate_rid or str(freeze_rid)
    if schema_version == 1:
        identity_ok = (
            effective_candidate_rid == freeze_rid
            and payload.get("rid") == freeze_rid
        )
    elif schema_version == 2:
        identity_ok = (
            payload.get("rid") == effective_candidate_rid
            and payload.get("candidate_rid") == effective_candidate_rid
            and payload.get("freeze_rid") == freeze_rid
        )
    else:
        identity_ok = False
    if (
        freeze.get("kind")
        != "independent_curated_v3_four_table_blind_freeze"
        or not identity_ok
        or payload.get("dataset_role") != "development_selection_used"
        or payload.get("selection_used") is not True
        or payload.get("created_once") is not True
        or engine not in {"rule", "agent"}
        or payload.get("engine") != engine
        or not isinstance(overall, Mapping)
        or overall.get("strict_result_contract") is not True
    ):
        raise ValueError(
            "independent development receipt does not prove the strict engine gate"
        )

    freeze_manifest = Path(freeze_manifest).resolve()
    if (
        not freeze_manifest.is_file()
        or payload.get("freeze_manifest_sha256")
        != sha256_file(freeze_manifest)
    ):
        raise ValueError(
            "independent development receipt freeze binding failed"
        )
    development = freeze.get("development")
    database = freeze.get("database")
    if not isinstance(development, Mapping) or not isinstance(
        database,
        Mapping,
    ):
        raise ValueError("independent freeze lacks development/database bindings")
    development_path = _safe_frozen_member(
        freeze_manifest.parent,
        development.get("path"),
        "independent-v3 development",
    )
    source_tree_sha256 = require_lowercase_sha256(
        source_tree_sha256,
        "candidate source tree SHA-256",
    )
    expected = {
        "input_sha256": development.get("sha256"),
        "database_sha256": database.get("sha256"),
        "source_tree_sha256": source_tree_sha256,
        "engine_source_sha256": sha256_file(
            ROOT / "dki" / "analysis" / "nl2sql.py"
        ),
        "evaluator_sha256": sha256_file(
            ROOT / "scripts" / "eval_nl2sql_frozen_development.py"
        ),
        "protocol_sha256": sha256_file(
            ROOT / "dki" / "analysis" / "nl2sql_protocol.py"
        ),
        "freeze_source_tree_sha256": freeze.get("source_version"),
        "candidate_source_differs_from_freeze": (
            freeze.get("source_version") != source_tree_sha256
        ),
    }
    if engine == "agent":
        if not isinstance(agent_profile, Mapping):
            raise ValueError("Agent development gate requires a sealed runtime profile")
        expected["agent_profile"] = dict(agent_profile)
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "independent development receipt differs from the frozen candidate"
        )
    if (
        not development_path.is_file()
        or development_path.stat().st_size != development.get("size")
        or sha256_file(development_path) != development.get("sha256")
        or sha256_file(db_path) != database.get("sha256")
    ):
        raise ValueError("independent development data/database binding failed")

    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    questions: set[str] = set()
    for raw in strict_development_jsonl_rows(
        development_path.read_bytes(),
        label="independent frozen development",
    ):
        row = normalise_development_metadata(raw)
        validate_gold_sql(row["gold_sql"])
        validate_question_sql_grounding(row["question"], row["gold_sql"])
        item_id = str(row["id"])
        question_key = normalise_question(str(row["question"]))
        if item_id in ids or question_key in questions:
            raise ValueError(
                "independent development ids/questions must be unique"
            )
        ids.add(item_id)
        questions.add(question_key)
        rows.append(row)
    if engine == "rule":
        recomputed = evaluate_set(rows, "rule", db_path, strict=True)
        recomputed_overall = {
            key: value for key, value in recomputed.items() if key != "details"
        }
        if dict(overall) != recomputed_overall:
            raise ValueError(
                "independent development receipt metrics differ from recomputation"
            )
    else:
        private_path_value = payload.get("private_results_path")
        private_sha256 = payload.get("private_results_sha256")
        if type(private_path_value) is not str or type(private_sha256) is not str:
            raise ValueError("agent development receipt lacks private result binding")
        private_path = Path(private_path_value).resolve()
        expected_private = (
            ROOT
            / "outputs"
            / "private"
            / effective_candidate_rid
            / "nl2sql_development"
            / str(freeze_rid)
            / "development_selection_used.json"
        ).resolve()
        if (
            private_path != expected_private
            or not private_path.is_file()
            or sha256_file(private_path) != private_sha256
        ):
            raise ValueError("agent development private result binding failed")
        private_payload = _load_receipt(private_path, "private development")
        details = private_payload.get("details")
        if (
            private_payload.get("engine") != "agent"
            or private_payload.get("overall") != dict(overall)
            or private_payload.get("agent_profile") != expected["agent_profile"]
            or not isinstance(details, list)
            or len(details) != overall.get("n")
        ):
            raise ValueError("agent development private results are inconsistent")
        detail_correct = sum(bool(item.get("correct")) for item in details if isinstance(item, Mapping))
        detail_answered = sum(
            item.get("engine") not in (None, "none")
            for item in details
            if isinstance(item, Mapping)
        )
        if (
            detail_correct != overall.get("n_correct")
            or detail_answered != overall.get("n_answered")
        ):
            raise ValueError("agent development detail counts differ from aggregate")

    quality_gate = _development_quality_gate(overall)
    curation = freeze.get("independent_curation", {})
    trusted = {
        "schema_version": 1,
        "authority": "independent_curated_v3_freeze_recomputation",
        "rid": freeze.get("rid"),
        "freeze_manifest_sha256": sha256_file(freeze_manifest),
        "development_sha256": development.get("sha256"),
        "database_sha256": database.get("sha256"),
        "curator_pack_manifest_sha256": (
            curation.get("curator_pack_manifest_sha256")
            if isinstance(curation, Mapping)
            else None
        ),
        "curator_attestation_sha256": (
            curation.get("attestation_sha256")
            if isinstance(curation, Mapping)
            else None
        ),
    }
    return {
        **quality_gate,
        "trusted_evidence": trusted,
    }


def _regression_quality_gate(payload: Mapping[str, Any]) -> dict[str, int]:
    """Validate the frozen known-120 and original easy-44 count gates."""

    n = _require_nonnegative_int(payload.get("n"), "known-120 receipt n")
    correct = _require_nonnegative_int(
        payload.get("n_correct"),
        "known-120 receipt n_correct",
    )
    answered = _require_nonnegative_int(
        payload.get("n_answered"),
        "known-120 receipt n_answered",
    )
    if correct > answered or answered > n:
        raise ValueError("known-120 receipt overall counts are impossible")

    layers = payload.get("by_difficulty")
    if not isinstance(layers, Mapping):
        raise ValueError("known-120 receipt lacks difficulty layers")
    easy = layers.get("easy")
    failures: list[str] = []
    if not isinstance(easy, Mapping):
        easy_n = 0
        easy_correct = 0
        failures.append("easy_44_missing")
    else:
        easy_n = _require_nonnegative_int(easy.get("n"), "known-120 easy n")
        easy_correct = _require_nonnegative_int(
            easy.get("correct"),
            "known-120 easy correct",
        )
        if easy_correct > easy_n:
            raise ValueError("known-120 easy layer counts are impossible")

    if n != KNOWN_120_REQUIRED_N:
        failures.append("known_120_n_not_120")
    if correct < KNOWN_120_MIN_CORRECT:
        failures.append("known_120_correct_below_112")
    if easy_n != EASY_44_REQUIRED_N:
        failures.append("easy_44_n_not_44")
    if easy_correct != EASY_44_REQUIRED_CORRECT:
        failures.append("easy_44_not_44_of_44")
    if failures:
        raise ValueError("NL2SQL regression gate failed: " + ",".join(failures))

    return {
        "n": n,
        "n_correct": correct,
        "n_answered": answered,
        "easy_n": easy_n,
        "easy_correct": easy_correct,
    }


def _validate_regression_gate(
    path: Path,
    *,
    db_path: Path,
    source_tree_sha256: str,
) -> dict[str, Any]:
    payload = _load_receipt(path, "known-120 regression")
    layers = payload.get("by_difficulty")
    if (
        payload.get("schema_version") != 3
        or payload.get("dataset_role") != "development_selection_used"
        or payload.get("selection_used") is not True
        or payload.get("engine") != "rule"
        or payload.get("strict_result_contract") is not True
        or not isinstance(layers, Mapping)
        or payload.get("source_tree_sha256") != source_tree_sha256
        or payload.get("engine_source_sha256")
        != sha256_file(ROOT / "dki" / "analysis" / "nl2sql.py")
    ):
        raise ValueError("known-120 receipt does not prove the strict rule regression")
    if payload.get("database_sha256") != sha256_file(db_path):
        raise ValueError("known-120 receipt database differs from the candidate")
    inputs = payload.get("input_files")
    if not isinstance(inputs, Mapping) or set(inputs) != {"base", "hard"}:
        raise ValueError("known-120 receipt lacks exact input-file bindings")
    for label, binding in inputs.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"known-120 input binding is invalid: {label}")
        candidate = Path(str(binding.get("path", ""))).resolve()
        if (
            not candidate.is_file()
            or binding.get("sha256") != sha256_file(candidate)
        ):
            raise ValueError(f"known-120 input binding failed: {label}")
    return _regression_quality_gate(payload)


def _validate_dependency_lock(path: Path) -> dict[str, Any]:
    try:
        payload = strict_json_loads(
            Path(path).read_bytes(),
            label="dependency lock",
        )
    except (OSError, ValueError) as exc:
        raise ValueError("dependency lock is unreadable") from exc
    if type(payload) is not dict:
        raise ValueError("dependency lock must be a JSON object")
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "python_environment_lock"
        or payload.get("created_once") is not True
    ):
        raise ValueError("unsupported dependency lock")
    python = payload.get("python", {})
    if (
        not isinstance(python, dict)
        or python.get("version") != platform.python_version()
        or python.get("implementation") != platform.python_implementation()
        or payload.get("platform") != platform.platform()
    ):
        raise ValueError("dependency lock runtime differs from the sealing runtime")
    packages = payload.get("packages")
    if not isinstance(packages, list) or not packages:
        raise ValueError("dependency lock lacks installed packages")
    package_map = {
        str(item.get("name", "")).casefold(): str(item.get("version", ""))
        for item in packages
        if isinstance(item, dict)
    }
    active_packages = {}
    for distribution in importlib.metadata.distributions():
        name = str(distribution.metadata.get("Name", "")).strip().casefold()
        version = str(distribution.version).strip()
        if name and version:
            active_packages[name] = version
    if package_map != active_packages:
        raise ValueError("installed distributions differ from the dependency lock")
    if package_map.get("sqlglot") != importlib.metadata.version("sqlglot"):
        raise ValueError("dependency lock does not bind the active sqlglot")
    canonical = "".join(
        f"{name}=={package_map[name]}\n" for name in sorted(package_map)
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != payload.get(
        "canonical_packages_sha256"
    ):
        raise ValueError("dependency lock aggregate hash is invalid")
    return payload


def seal_candidate(
    *,
    output: Path,
    freeze_manifest: Path,
    development_receipt: Path,
    regression_receipt: Path,
    task1_manifest: Path,
    kg_path: Path,
    db_path: Path,
    source_tree_manifest: Path,
    execution_bundle: Path,
    dependency_spec: Path,
    dependency_lock: Path,
    engine: str = "rule",
    llm_strategy: str = "legacy_sql",
    agent_rule_policy: str = "rule_first",
    llm_reasoning_mode: str = "disabled",
    llm_completion_tokens: int = 1024,
) -> dict[str, Any]:
    agent_profile = _runtime_profile_for_engine(
        engine=engine,
        llm_strategy=llm_strategy,
        agent_rule_policy=agent_rule_policy,
        llm_reasoning_mode=llm_reasoning_mode,
        llm_completion_tokens=llm_completion_tokens,
    )
    _assert_freeze_unconsumed(freeze_manifest)
    task1_manifest = Path(task1_manifest).resolve()
    kg_path = Path(kg_path).resolve()
    db_path = Path(db_path).resolve()
    validate_task2_artifact(kg_path, task1_path=task1_manifest)
    if not database_matches_manifest(db_path, task1_manifest, kg_path):
        raise ValueError("database does not match the supplied Task 1/KG lineage")
    freeze = _freeze_contract(
        freeze_manifest=freeze_manifest,
        task1_manifest=task1_manifest,
        kg_path=kg_path,
        db_path=db_path,
    )
    source_tree = _verify_source_tree_manifest(source_tree_manifest)
    execution_bundle_closure = _verify_execution_bundle_closure(
        execution_bundle,
        source_tree=source_tree,
    )
    freeze_source_tree = freeze.get("source_version")
    if freeze_source_tree is not None and not re.fullmatch(
        r"[0-9a-f]{64}",
        str(freeze_source_tree),
    ):
        raise ValueError("blind freeze source identity is malformed")
    development_gate = _validate_development_gate(
        development_receipt,
        freeze=freeze,
        db_path=db_path,
        source_tree_sha256=source_tree["included_tree_sha256"],
        freeze_manifest=freeze_manifest,
        candidate_rid=(
            str(source_tree.get("rid"))
            if source_tree.get("rid") is not None
            else None
        ),
        engine=engine,
        agent_profile=agent_profile,
    )
    regression_gate = _validate_regression_gate(
        regression_receipt,
        db_path=db_path,
        source_tree_sha256=source_tree["included_tree_sha256"],
    )
    dependency_lock_payload = _validate_dependency_lock(dependency_lock)
    if (
        dependency_lock_payload.get("source_tree_sha256")
        != source_tree["included_tree_sha256"]
    ):
        raise ValueError("dependency lock is not bound to the included source tree")
    artifacts = {
        "freeze_manifest": _binding(freeze_manifest),
        "development_gate": _binding(development_receipt),
        "known_120_regression_gate": _binding(regression_receipt),
        "task1_manifest": _binding(task1_manifest),
        "task2_kg": _binding(kg_path),
        "task2_provenance": _binding(task2_receipt_path(kg_path)),
        "database": _binding(db_path),
        "database_provenance": _binding(
            db_path.with_suffix(".provenance.json")
        ),
        "source_tree_manifest": _binding(source_tree_manifest),
        "execution_bundle": execution_bundle_closure["archive"],
        "dependency_spec": _binding(dependency_spec),
        "dependency_lock": _binding(dependency_lock),
        "engine_source": _binding(ROOT / "dki" / "analysis" / "nl2sql.py"),
        "nl2sql_eval": _binding(
            ROOT / "dki" / "analysis" / "nl2sql_eval.py"
        ),
        "nl2sql_protocol": _binding(
            ROOT / "dki" / "analysis" / "nl2sql_protocol.py"
        ),
        "nl2sql_seal": _binding(
            ROOT / "dki" / "analysis" / "nl2sql_seal.py"
        ),
        "blind_freezer": _binding(
            ROOT / "scripts" / "freeze_nl2sql_blind.py"
        ),
        "official_v2_blind_freezer": _binding(
            ROOT / "scripts" / "freeze_nl2sql_official_v2.py"
        ),
        "mimicsql_v3_blind_freezer": _binding(
            ROOT
            / "scripts"
            / "freeze_nl2sql_mimicsql_official_v3.py"
        ),
        "mimicsql_v3_development_builder": _binding(
            ROOT
            / "scripts"
            / "build_nl2sql_mimicsql_development.py"
        ),
        "mimicsql_v3_projection": _binding(
            ROOT
            / "dki"
            / "analysis"
            / "nl2sql_mimicsql_projection.py"
        ),
        "official_v2_development_builder": _binding(
            ROOT
            / "scripts"
            / "project_nl2sql_official_v2_development.py"
        ),
        "css_projection": _binding(
            ROOT / "dki" / "analysis" / "nl2sql_css_projection.py"
        ),
        "ehrsql_projection": _binding(
            ROOT / "dki" / "analysis" / "nl2sql_ehrsql_projection.py"
        ),
        "blind_evaluator": _binding(
            ROOT / "scripts" / "eval_nl2sql_blind.py"
        ),
        "candidate_sealer": _binding(
            ROOT / "scripts" / "seal_nl2sql_blind_candidate.py"
        ),
    }
    if engine == "agent":
        artifacts["llm_adapter"] = _binding(ROOT / "dki" / "llm.py")
    if freeze.get("kind") == "official_public_v2_four_table_blind_freeze":
        official_evidence = _official_v2_freeze_evidence(
            freeze_manifest,
            freeze,
        )
        artifacts.update(
            {
                key: _binding(path)
                for key, path in official_evidence.items()
            }
        )
    if freeze.get("kind") == "mimicsql_v3_four_table_blind_freeze":
        mimicsql_evidence = _mimicsql_v3_freeze_evidence(
            freeze_manifest,
            freeze,
        )
        artifacts.update(
            {
                key: _binding(path)
                for key, path in mimicsql_evidence.items()
            }
        )
    if (
        freeze.get("kind")
        == "independent_curated_v3_four_table_blind_freeze"
    ):
        independent_evidence = _independent_v3_freeze_evidence(
            freeze_manifest,
            freeze,
        )
        artifacts.update(
            {
                key: _binding(path)
                for key, path in independent_evidence.items()
            }
        )
    payload = {
        "schema_version": 1,
        "kind": "sealed_nl2sql_blind_candidate",
        "status": "sealed",
        "engine": engine,
        "agent_profile": agent_profile,
        "holdout_opened_by_sealer": False,
        "holdout_sha256_declared": freeze.get("holdout", {}).get("sha256"),
        "source_tree_sha256": source_tree["included_tree_sha256"],
        "candidate_rid": source_tree.get("rid"),
        "freeze_rid": freeze.get("rid"),
        "freeze_source_tree_sha256": freeze_source_tree,
        "candidate_source_differs_from_freeze": (
            freeze_source_tree is not None
            and freeze_source_tree != source_tree["included_tree_sha256"]
        ),
        "freeze_counts": {
            "n_development": freeze["n_development"],
            "n_holdout": freeze["n_holdout"],
            "n_hard": freeze["n_hard"],
        },
        "preconsumption_gates": {
            "development": development_gate,
            "known_120_regression": regression_gate,
        },
        "execution_bundle_closure": {
            key: value
            for key, value in execution_bundle_closure.items()
            if key != "archive"
        },
        "environment": _environment(),
        "artifacts": artifacts,
        "sealer_sha256": sha256_file(Path(__file__)),
    }
    _publish_once(output, payload)
    return payload


def verify_candidate_seal(
    seal_path: Path,
    *,
    freeze_manifest: Path,
    task1_manifest: Path,
    kg_path: Path,
    db_path: Path,
    engine: str,
    llm_strategy: str = "legacy_sql",
    agent_rule_policy: str = "rule_first",
    llm_reasoning_mode: str = "disabled",
    llm_completion_tokens: int = 1024,
) -> dict[str, Any]:
    agent_profile = _runtime_profile_for_engine(
        engine=engine,
        llm_strategy=llm_strategy,
        agent_rule_policy=agent_rule_policy,
        llm_reasoning_mode=llm_reasoning_mode,
        llm_completion_tokens=llm_completion_tokens,
    )
    _assert_freeze_unconsumed(freeze_manifest)
    try:
        seal = strict_json_loads(
            Path(seal_path).read_bytes(),
            label="candidate seal",
        )
    except (OSError, ValueError) as exc:
        raise ValueError("candidate seal is unreadable") from exc
    if type(seal) is not dict:
        raise ValueError("candidate seal must be a JSON object")
    if (
        seal.get("schema_version") != 1
        or seal.get("kind") != "sealed_nl2sql_blind_candidate"
        or seal.get("status") != "sealed"
        or seal.get("engine") != engine
        or seal.get("holdout_opened_by_sealer") is not False
    ):
        raise ValueError("unsupported or invalid candidate seal")
    if engine == "agent":
        if seal.get("agent_profile") != agent_profile:
            raise ValueError("candidate seal Agent runtime profile differs")
    elif seal.get("agent_profile") is not None:
        raise ValueError("rule candidate seal must not contain an Agent profile")
    artifacts = seal.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("candidate seal lacks artifact bindings")
    for key in REQUIRED_SEAL_ARTIFACT_BINDINGS:
        if not isinstance(artifacts.get(key), Mapping):
            raise ValueError(
                f"candidate seal lacks required artifact binding: {key}"
            )
    if engine == "agent" and not isinstance(artifacts.get("llm_adapter"), Mapping):
        raise ValueError("candidate seal lacks required artifact binding: llm_adapter")
    runtime_paths = {
        "freeze_manifest": Path(freeze_manifest).resolve(),
        "task1_manifest": Path(task1_manifest).resolve(),
        "task2_kg": Path(kg_path).resolve(),
        "database": Path(db_path).resolve(),
        "blind_evaluator": ROOT / "scripts" / "eval_nl2sql_blind.py",
    }
    for key, expected_path in runtime_paths.items():
        binding = artifacts.get(key, {})
        if Path(str(binding.get("path", ""))).resolve() != expected_path.resolve():
            raise ValueError(f"candidate seal runtime path differs: {key}")
    for key, binding in artifacts.items():
        if not isinstance(binding, dict):
            raise ValueError(f"candidate seal binding is invalid: {key}")
        candidate = Path(str(binding.get("path", "")))
        if (
            not candidate.is_file()
            or candidate.stat().st_size != binding.get("size")
            or sha256_file(candidate) != binding.get("sha256")
        ):
            raise ValueError(f"candidate seal artifact binding failed: {key}")
    source_manifest = Path(artifacts["source_tree_manifest"]["path"])
    source_tree = _verify_source_tree_manifest(source_manifest)
    if source_tree["included_tree_sha256"] != seal.get("source_tree_sha256"):
        raise ValueError("candidate seal source-tree identity differs")
    execution_bundle_closure = _verify_execution_bundle_closure(
        Path(artifacts["execution_bundle"]["path"]),
        source_tree=source_tree,
    )
    sealed_execution_bundle_closure = seal.get("execution_bundle_closure")
    if sealed_execution_bundle_closure != {
        key: value
        for key, value in execution_bundle_closure.items()
        if key != "archive"
    }:
        raise ValueError("candidate seal execution-bundle closure differs")
    if _environment() != seal.get("environment"):
        raise ValueError("candidate seal environment differs from runtime")
    dependency_lock = _validate_dependency_lock(
        Path(artifacts["dependency_lock"]["path"])
    )
    if dependency_lock.get("source_tree_sha256") != seal.get(
        "source_tree_sha256"
    ):
        raise ValueError("candidate seal dependency lock differs from source tree")
    validate_task2_artifact(kg_path, task1_path=task1_manifest)
    if not database_matches_manifest(db_path, task1_manifest, kg_path):
        raise ValueError("candidate seal database lineage no longer validates")
    freeze = _freeze_contract(
        freeze_manifest=freeze_manifest,
        task1_manifest=task1_manifest,
        kg_path=kg_path,
        db_path=db_path,
    )
    freeze_source_tree = freeze.get("source_version")
    if freeze_source_tree is not None and not re.fullmatch(
        r"[0-9a-f]{64}",
        str(freeze_source_tree),
    ):
        raise ValueError("blind freeze source identity is malformed")
    source_differs = (
        freeze_source_tree is not None
        and freeze_source_tree != seal.get("source_tree_sha256")
    )
    if (
        seal.get("freeze_source_tree_sha256") != freeze_source_tree
        or seal.get("candidate_source_differs_from_freeze")
        is not source_differs
        or seal.get("holdout_sha256_declared")
        != freeze.get("holdout", {}).get("sha256")
    ):
        raise ValueError("candidate seal differs from the frozen holdout declaration")
    refreshed_development_gate = _validate_development_gate(
        Path(artifacts["development_gate"]["path"]),
        freeze=freeze,
        db_path=db_path,
        source_tree_sha256=seal["source_tree_sha256"],
        freeze_manifest=freeze_manifest,
        candidate_rid=(
            str(source_tree.get("rid"))
            if source_tree.get("rid") is not None
            else seal.get("candidate_rid")
        ),
        engine=engine,
        agent_profile=agent_profile,
    )
    refreshed_regression_gate = _validate_regression_gate(
        Path(artifacts["known_120_regression_gate"]["path"]),
        db_path=db_path,
        source_tree_sha256=seal["source_tree_sha256"],
    )
    gates = seal.get("preconsumption_gates")
    if (
        not isinstance(gates, dict)
        or gates.get("development") != refreshed_development_gate
        or gates.get("known_120_regression") != refreshed_regression_gate
    ):
        raise ValueError("candidate seal preconsumption gates no longer validate")
    return seal
