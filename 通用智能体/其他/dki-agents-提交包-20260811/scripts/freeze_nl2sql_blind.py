"""Create-once NL2SQL blind-set freeze grouped by normalized SQL AST."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.analysis.database import database_matches_manifest
from dki.analysis.nl2sql_protocol import (
    ALLOWED_CATEGORIES,
    ALLOWED_DIFFICULTIES,
    execute_complete_readonly,
    normalise_question,
    validate_gold_sql,
    validate_question_sql_grounding,
)
from dki.pipeline.artifacts import validate_task2_artifact


EXPECTED_DIFFICULTY_QUOTAS = {
    "medium": 90,
    "hard": 150,
    "extra": 30,
    "extra-hard": 30,
}
EXPECTED_CATEGORY_QUOTAS = {
    "filtering": 30,
    "aggregation_grouping": 45,
    "top_k": 24,
    "multi_table_join": 45,
    "temporal": 30,
    "knowledge_graph": 36,
    "having": 15,
    "nested_correlated": 30,
    "cte": 20,
    "set_operation": 10,
    "window": 15,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sql_ast_skeleton(sql: str) -> str:
    try:
        from sqlglot import exp, parse_one
    except ImportError as exc:
        raise SystemExit(
            "缺少 sqlglot；请安装项目依赖后再冻结，禁止退化为正则 SQL 分组"
        ) from exc
    try:
        tree = parse_one(sql, read="sqlite")
    except Exception as exc:
        raise ValueError(f"gold_sql 不是合法 SQLite SQL: {exc}") from exc

    def anonymize(node):
        if isinstance(node, exp.Literal):
            return exp.Literal.string("__LITERAL__")
        return node

    normalized = tree.transform(anonymize).sql(dialect="sqlite")
    return re.sub(r"\s+", " ", normalized).strip().upper()


class _UnionFind:
    """Small deterministic union-find for non-overlapping blind families."""

    def __init__(self, count: int) -> None:
        self.parent = list(range(count))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root > right_root:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root


def _question_trigrams(question: str) -> set[str]:
    if len(question) < 3:
        return {question} if question else set()
    return {question[index : index + 3] for index in range(len(question) - 2)}


def _questions_are_near_duplicates(left: str, right: str) -> bool:
    """Apply the published deterministic near-duplicate contract."""

    left_grams = _question_trigrams(left)
    right_grams = _question_trigrams(right)
    union = len(left_grams | right_grams)
    jaccard = len(left_grams & right_grams) / union if union else 1.0
    if jaccard >= 0.85:
        return True
    return difflib.SequenceMatcher(None, left, right, autojunk=False).ratio() >= 0.90


def _partition_groups(rows: list[dict[str, Any]]) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Build transitive AST/exact-near-question families without exposing text.

    The returned keys are digests of immutable input IDs solely for deterministic
    selection.  They are never emitted in a public attestation.
    """

    normalized = [normalise_question(str(row["question"])) for row in rows]
    skeleton_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        skeleton_groups[_sql_ast_skeleton(str(row["gold_sql"]))].append(index)
    families = _UnionFind(len(rows))
    for indices in skeleton_groups.values():
        for index in indices[1:]:
            families.union(indices[0], index)
    near_links = 0
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if _questions_are_near_duplicates(normalized[left], normalized[right]):
                near_links += 1
                families.union(left, right)
    by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        by_root[families.find(index)].append(index)
    groups: dict[str, list[int]] = {}
    for indices in by_root.values():
        identity = "\0".join(sorted(str(rows[index]["id"]) for index in indices))
        groups[f"family:{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"] = sorted(indices)
    return groups, {
        "n_sql_ast_groups": len(skeleton_groups),
        "n_partition_families": len(groups),
        "n_near_duplicate_links": near_links,
    }


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    lines = [
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def _write_bytes_synced(path: Path, data: bytes) -> None:
    """Write a staging artifact and sync the writable descriptor before publish."""
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_bytes_once(path: Path, data: bytes) -> None:
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
        handle.write(data)
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


def _rid_below(path: Path, scope: str) -> str:
    parts = Path(path).resolve().parts
    for index in range(len(parts) - 2):
        if (
            parts[index].casefold() == "outputs"
            and parts[index + 1].casefold() == scope.casefold()
        ):
            return parts[index + 2]
    raise ValueError(f"artifact is not below outputs/{scope}/<RID>: {path}")


def _refuse_overwrite(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise SystemExit(
            "冻结产物已存在，禁止覆盖；请使用新的版本目录:\n  "
            + "\n  ".join(existing)
        )


def _validate_engine_blind_pack(pack_manifest_path: Path) -> dict[str, Any]:
    """Validate the immutable curator input pack independently of its handoff."""
    try:
        pack = json.loads(pack_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("curator pack manifest is unreadable") from exc
    if (
        pack.get("kind") != "engine_blind_nl2sql_curator_pack"
        or pack.get("dataset_role") != "independent_curation_input_only"
        or pack.get("created_once") is not True
        or pack.get("contains_engine_implementation") is not False
        or pack.get("contains_questions_or_gold_sql") is not False
    ):
        raise ValueError("curator pack does not prove the engine-blind input boundary")
    pack_root = pack_manifest_path.parent.resolve()
    file_bindings = pack.get("files")
    if not isinstance(file_bindings, dict) or not file_bindings:
        raise ValueError("curator pack lacks file bindings")
    for relative, binding in file_bindings.items():
        if not isinstance(relative, str) or not isinstance(binding, dict):
            raise ValueError("curator pack has an invalid file binding")
        candidate = (pack_root / relative).resolve()
        try:
            candidate.relative_to(pack_root)
        except ValueError as exc:
            raise ValueError(f"curator pack contains an unsafe path: {relative}") from exc
        if (
            not candidate.is_file()
            or _sha256(candidate) != binding.get("sha256")
            or candidate.stat().st_size != binding.get("size")
        ):
            raise ValueError(f"curator pack file binding failed: {relative}")
    return pack


def _validate_pack_and_attestation(
    *,
    pack_manifest_path: Path,
    curator_attestation_path: Path,
    corpus_path: Path,
    n_rows: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify that curation crossed the declared engine-blind boundary."""
    pack = _validate_engine_blind_pack(pack_manifest_path)
    file_bindings = pack["files"]
    try:
        curator = json.loads(curator_attestation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("curator attestation is unreadable") from exc

    required_false = (
        "engine_context_provided",
        "local_evaluation_context_provided",
        "project_local_question_reuse",
        "project_local_question_paraphrase",
        "development_or_holdout_preassigned",
    )
    kind = curator.get("kind")
    if kind == "independent_four_table_nl2sql_curator_attestation":
        required_true = (
            "independent_curation",
            "all_gold_sql_executed",
            "all_pairs_semantically_reviewed",
        )
        if any(curator.get(key) is not True for key in required_true):
            raise ValueError(
                "curator attestation lacks a required positive declaration"
            )
        if (
            curator.get("curator_pack_manifest_sha256")
            != _sha256(pack_manifest_path)
        ):
            raise ValueError("curator attestation is not bound to this pack")
    elif kind == "public_authoritative_four_table_projection_attestation":
        required_true = (
            "mechanical_projection",
            "all_gold_sql_executed",
            "all_pairs_semantically_validated_by_construction",
        )
        if any(curator.get(key) is not True for key in required_true):
            raise ValueError(
                "public projection attestation lacks a required positive "
                "declaration"
            )
        independence = curator.get("independence_basis")
        if (
            not isinstance(independence, dict)
            or independence.get("official_source_test_content_parsed")
            is not False
            or independence.get("project_nl2sql_engine_imported") is not False
            or independence.get("project_local_evaluation_questions_used")
            is not False
            or independence.get("development_or_holdout_preassigned")
            is not False
        ):
            raise ValueError(
                "public projection attestation violates source/engine isolation"
            )
        bindings = curator.get("bindings")
        database_binding = (
            bindings.get("database")
            if isinstance(bindings, dict)
            else None
        )
        pack_database_binding = file_bindings.get("four_table.sqlite")
        if (
            not isinstance(database_binding, dict)
            or not isinstance(pack_database_binding, dict)
            or database_binding.get("sha256")
            != pack_database_binding.get("sha256")
        ):
            raise ValueError(
                "public projection database differs from the curator pack"
            )
        aggregate = curator.get("aggregate")
        if (
            not isinstance(aggregate, dict)
            or aggregate.get("n_items") != n_rows
            or aggregate.get("all_gold_sql_validated") is not True
            or aggregate.get(
                "all_gold_sql_executed_readonly_without_truncation"
            )
            is not True
            or aggregate.get("all_result_sets_nonempty") is not True
            or aggregate.get(
                "all_predicate_literals_grounded_in_question"
            )
            is not True
        ):
            raise ValueError(
                "public projection aggregate lacks executable semantic proof"
            )
    else:
        raise ValueError("unsupported curator attestation kind")
    if any(curator.get(key) is not False for key in required_false):
        raise ValueError("curator attestation lacks a required negative declaration")
    if not str(curator.get("curator_identity", "")).strip():
        raise ValueError("curator identity is required")
    if not str(curator.get("semantic_reviewer_identity", "")).strip():
        raise ValueError("semantic reviewer identity is required")
    if (
        str(curator["curator_identity"]).strip()
        == str(curator["semantic_reviewer_identity"]).strip()
    ):
        raise ValueError("curator and semantic reviewer must be distinct")
    if not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
        r"(?:\.\d{1,9})?(?:Z|\+00:00)",
        str(curator.get("created_at", "")),
    ):
        raise ValueError("curator attestation created_at must be UTC RFC3339")
    if curator.get("n_items") != n_rows:
        raise ValueError("curator attestation item count differs from corpus")
    if curator.get("corpus_sha256") != _sha256(corpus_path):
        raise ValueError("curator attestation is not bound to this corpus")
    return pack, curator


def _choose_holdout_indices(
    *,
    rows: list[dict[str, Any]],
    groups: dict[str, list[int]],
    seed: str,
    minimum_holdout: int,
    maximum_holdout: int,
    minimum_hard: int,
) -> list[int]:
    """Choose whole AST groups with a deterministic bounded subset DP."""
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: hashlib.sha256(
            f"{seed}\0{item[0]}".encode("utf-8")
        ).hexdigest(),
    )
    # State is (row count, hard count capped at the gate); the tuple stores
    # ordered group indices. First arrival wins, making ties deterministic.
    states: dict[tuple[int, int], tuple[int, ...]] = {(0, 0): ()}
    hard_labels = {"hard", "extra", "extra-hard"}
    for group_index, (_, indices) in enumerate(ordered_groups):
        group_size = len(indices)
        group_hard = sum(
            rows[index]["difficulty"] in hard_labels for index in indices
        )
        next_states = dict(states)
        for (count, hard), chosen in states.items():
            new_count = count + group_size
            if new_count > maximum_holdout:
                continue
            key = (new_count, min(minimum_hard, hard + group_hard))
            next_states.setdefault(key, chosen + (group_index,))
        states = next_states
    candidates = [
        (count, chosen)
        for (count, hard), chosen in states.items()
        if minimum_holdout <= count <= maximum_holdout and hard >= minimum_hard
    ]
    if not candidates:
        raise ValueError(
            "AST groups cannot satisfy holdout/development size and hard floors"
        )
    _, chosen_groups = min(
        candidates,
        key=lambda item: (
            item[0] - minimum_holdout,
            len(item[1]),
            item[1],
        ),
    )
    return sorted(
        index
        for group_index in chosen_groups
        for index in ordered_groups[group_index][1]
    )


SUPPORTED_INDEPENDENT_CURATOR_VERSIONS = frozenset(
    {
        "independent-curator-v11",
        "independent-curator-v11b",
        # v12 is a newly commissioned, engine-isolated handoff.  It retains
        # the exact five-file/hash/isolation contract; accepting it here does
        # not relax any curation, execution, or duplicate-control gate.
        "independent-curator-v12",
        # v13 is a replacement commissioned after the v12 handoff failed
        # pre-freeze metadata validation.  It is deliberately a distinct
        # corpus and is checked by the same complete five-file contract.
        "independent-curator-v13",
        # v14 is a new isolated corpus after v13 failed its own immutable
        # corpus-to-attestation binding during pre-freeze validation.
        "independent-curator-v14",
        "independent-curator-v15",
        # v16 was the first prospective corpus consumed under the strict
        # overall gate.  v17 is its independently commissioned replacement
        # after that corpus failed on aggregate metrics.  Both retain the
        # exact same five-file, database-bound and isolation declarations;
        # version admission does not relax any row or execution check below.
        "independent-curator-v16",
        "independent-curator-v17",
        "independent-curator-v18",
        "independent-curator-v19",
        # v20 replaces the pre-freeze v19 delivery after its natural-language
        # questions failed the explicit SQL-literal grounding gate.  The
        # five-file, isolation and execution contracts remain unchanged.
        "independent-curator-v20",
        "independent-curator-v21",
        # v22 is a wholly new, engine-isolated corpus commissioned only after
        # v21 was irreversibly consumed.  It retains the identical five-file,
        # database-binding, grounding, and execution contract.
        "independent-curator-v22",
        # v23 is a new create-once replacement after v22 failed before
        # partition publication because one gold query exceeded the bounded
        # execution budget.  It is not a rewrite or rerun of v22.
        "independent-curator-v23",
        "independent-curator-v24",
        "independent-curator-v25",
        "independent-curator-v26",
        # v27 is the new engine-isolated prospective corpus commissioned
        # after v26 was irreversibly consumed.  It is admitted under the
        # unchanged five-file, database-bound and aggregate-only contract.
        "independent-curator-v27",
        # v28 is the engine-isolated prospective corpus commissioned after
        # v27 was irreversibly consumed.  It keeps the identical five-file,
        # database-bound and aggregate-only contract; version admission does
        # not relax any row, grounding, isolation, or execution gate.
        "independent-curator-v28",
        # v29 is a fresh engine-isolated corpus commissioned after v28's holdout
        # was consumed.  It retains the exact five-file/isolation/execution
        # contract; admission here does not relax any gate.
        "independent-curator-v29",
        # v30 is a fresh engine-isolated corpus commissioned after v29's holdout
        # was consumed under the LLM-first architecture.  It retains the exact
        # five-file/isolation/execution contract; admission here does not relax
        # any gate.
        "independent-curator-v30",
    }
)

CURATOR_FORMAT_CHOICES = (
    "legacy-pack",
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
)
INDEPENDENT_FIVE_FILE_FORMATS = frozenset(CURATOR_FORMAT_CHOICES[1:])


def _validate_independent_attestation(
    *,
    corpus_path: Path,
    database_path: Path,
    attestation_path: Path,
    aggregate_audit_path: Path,
    execution_audit_path: Path,
    sha256s_path: Path,
    n_rows: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate an engine-isolated five-file curator handoff.

    The main implementation process never needs to display the corpus or the
    row-level execution audit.  The freezer verifies their hashes and counts,
    while only aggregate declarations are copied to the freeze manifest.
    """

    try:
        curator = json.loads(attestation_path.read_text(encoding="utf-8"))
        aggregate = json.loads(
            aggregate_audit_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("independent v11 curator evidence is unreadable") from exc
    if curator.get("source_version") not in SUPPORTED_INDEPENDENT_CURATOR_VERSIONS:
        raise ValueError("unsupported independent curator source version")
    corpus = curator.get("corpus")
    database = curator.get("database")
    independence = curator.get("independence_declarations")
    execution = curator.get("execution_attestation")
    if (
        not isinstance(corpus, dict)
        or corpus.get("record_count") != n_rows
        or corpus.get("sha256") != _sha256(corpus_path)
    ):
        raise ValueError("independent v11 corpus binding failed")
    if (
        not isinstance(database, dict)
        or database.get("sha256") != _sha256(database_path)
    ):
        raise ValueError("independent v11 database binding failed")
    required_false_aliases = (
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
    if (
        not isinstance(independence, dict)
        or any(
            not any(
                alias in independence and independence.get(alias) is False
                for alias in aliases
            )
            for aliases in required_false_aliases
        )
        or independence.get("database_schema_and_self_generated_query_results_only")
        is not True
    ):
        raise ValueError("independent v11 isolation declarations are incomplete")
    required_execution = (
        "all_queries_executed_successfully",
        "all_results_fully_materialized",
        "single_read_only_sqlite_query_per_record",
        "allowed_tables_only",
    )
    if (
        not isinstance(execution, dict)
        or any(execution.get(key) is not True for key in required_execution)
    ):
        raise ValueError("independent v11 execution declarations are incomplete")
    if (
        aggregate.get("record_count") != n_rows
        or aggregate.get("unique_id_count") != n_rows
        or aggregate.get("unique_normalized_question_count") != n_rows
        or aggregate.get("unique_gold_sql_count") != n_rows
        or aggregate.get("execution", {}).get("success_count") != n_rows
        or aggregate.get("execution", {}).get("failure_count") != 0
    ):
        raise ValueError("independent v11 aggregate audit failed")

    declared: dict[str, str] = {}
    for line in sha256s_path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError("invalid independent v11 SHA256SUMS line")
        declared[fields[1].lstrip("*")] = fields[0].lower()
    expected_files = {
        corpus_path.name: corpus_path,
        attestation_path.name: attestation_path,
        aggregate_audit_path.name: aggregate_audit_path,
        execution_audit_path.name: execution_audit_path,
    }
    if set(declared) != set(expected_files):
        raise ValueError("independent v11 SHA256SUMS file set differs")
    for name, path in expected_files.items():
        if declared[name] != _sha256(path):
            raise ValueError(f"independent v11 SHA256SUMS mismatch: {name}")
    with execution_audit_path.open("rb") as stream:
        execution_rows = sum(1 for line in stream if line.strip())
    if execution_rows != n_rows:
        raise ValueError("independent v11 execution audit row count differs")
    pack = {
        "dataset_role": "independent_curation_input_only",
        "format": "independent-five-file-handoff",
    }
    return pack, curator


def _validate_independent_v11_attestation(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Backward-compatible name for the original v11 handoff validator.

    Existing frozen v11/v11b evidence remains readable.  New callers should
    use the generic function so a separately attested v12 handoff is checked
    against exactly the same strict contract rather than being silently
    treated as legacy input.
    """

    return _validate_independent_attestation(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="独立编写的 JSONL")
    parser.add_argument("--output", type=Path, required=True, help="冻结 manifest JSON")
    parser.add_argument(
        "--holdout-count",
        type=int,
        default=120,
        help="盲测集最少题数，赛题门槛默认 120",
    )
    parser.add_argument("--seed", default="dki-nl2sql-blind-v2")
    parser.add_argument("--db", type=Path, required=True, help="sealed read-only SQLite database")
    parser.add_argument("--task1-manifest", type=Path, required=True)
    parser.add_argument("--kg", type=Path, required=True)
    parser.add_argument("--curator-pack-manifest", type=Path, required=True)
    parser.add_argument("--curator-attestation", type=Path, required=True)
    parser.add_argument(
        "--curator-format",
        choices=CURATOR_FORMAT_CHOICES,
        default="legacy-pack",
    )
    parser.add_argument("--curator-aggregate-audit", type=Path)
    parser.add_argument("--curator-execution-audit", type=Path)
    parser.add_argument(
        "--curator-sha256s",
        type=Path,
        help="SHA256SUMS.txt from an independent five-file handoff",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=300,
        help="exact independent corpus size",
    )
    parser.add_argument(
        "--quota-contract",
        choices=("legacy-300", "attested-distribution"),
        default="legacy-300",
    )
    parser.add_argument("--source-version", required=True, help="sealed source/included-tree identity")
    parser.add_argument(
        "--public-output",
        type=Path,
        required=True,
        help="aggregate-only public freeze attestation",
    )
    parser.add_argument("--created-at", required=True)
    args = parser.parse_args()

    private_root = "outputs/private/"
    private_paths = [
        args.input,
        args.output,
        args.curator_pack_manifest,
        args.curator_attestation,
    ]
    private_paths.extend(
        path
        for path in (
            args.curator_aggregate_audit,
            args.curator_execution_audit,
            args.curator_sha256s,
        )
        if path is not None
    )
    if any(private_root not in path.resolve().as_posix() for path in private_paths):
        raise SystemExit("raw independent corpus and frozen output must remain under outputs/private/<RID>")
    if private_root in args.public_output.resolve().as_posix():
        raise SystemExit("public freeze attestation must not be private")
    try:
        output_rid = _rid_below(args.output, "private")
        public_rid = _rid_below(args.public_output, "remote_runs")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if output_rid != public_rid:
        raise SystemExit("private freeze and public attestation RID differ")
    if not re.fullmatch(r"[0-9a-f]{64}", args.source_version):
        raise SystemExit("source-version must be a full lowercase SHA-256")
    if not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
        args.created_at,
    ):
        raise SystemExit("created-at must be UTC RFC3339")
    if args.holdout_count < 120:
        raise SystemExit("holdout-count 不得低于赛题审计门槛 120")
    validate_task2_artifact(args.kg, task1_path=args.task1_manifest)
    if not database_matches_manifest(args.db, args.task1_manifest, args.kg):
        raise SystemExit("sealed database does not match supplied Task 1 / KG provenance")
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != args.expected_count:
        raise SystemExit(
            "independent corpus must contain exactly "
            f"{args.expected_count} items, received {len(rows)}"
        )
    try:
        if args.curator_format in INDEPENDENT_FIVE_FILE_FORMATS:
            if (
                args.curator_aggregate_audit is None
                or args.curator_execution_audit is None
                or args.curator_sha256s is None
            ):
                raise ValueError(
                    "independent five-file handoff requires aggregate, execution, and SHA256SUMS evidence"
                )
            pack = _validate_engine_blind_pack(args.curator_pack_manifest)
            pack_database = pack.get("files", {}).get("four_table.sqlite")
            if (
                not isinstance(pack_database, dict)
                or pack_database.get("sha256") != _sha256(args.db)
            ):
                raise ValueError(
                    "independent curator pack database differs from the sealed database"
                )
            _, curator = _validate_independent_attestation(
                corpus_path=args.input,
                database_path=args.db,
                attestation_path=args.curator_attestation,
                aggregate_audit_path=args.curator_aggregate_audit,
                execution_audit_path=args.curator_execution_audit,
                sha256s_path=args.curator_sha256s,
                n_rows=len(rows),
            )
        else:
            pack, curator = _validate_pack_and_attestation(
                pack_manifest_path=args.curator_pack_manifest,
                curator_attestation_path=args.curator_attestation,
                corpus_path=args.input,
                n_rows=len(rows),
            )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if any(not row.get("question") or not row.get("gold_sql") for row in rows):
        raise SystemExit("每题必须包含 question 与 gold_sql")
    if len({row["question"] for row in rows}) != len(rows):
        raise SystemExit("盲测题 question 不可重复")

    required = {"id", "question", "gold_sql", "difficulty", "category"}
    for index, row in enumerate(rows, start=1):
        missing = required - set(row)
        if missing or any(not str(row.get(key, "")).strip() for key in required):
            raise SystemExit(f"item {index} lacks required fields: {sorted(missing)}")
        row["difficulty"] = str(row["difficulty"]).lower()
        if row["difficulty"] not in ALLOWED_DIFFICULTIES:
            raise SystemExit(f"item {index} has invalid difficulty: {row['difficulty']}")
        row["category"] = str(row["category"]).lower()
        if row["category"] not in ALLOWED_CATEGORIES:
            raise SystemExit(f"item {index} has invalid category: {row['category']}")
        try:
            validate_gold_sql(row["gold_sql"])
            validate_question_sql_grounding(row["question"], row["gold_sql"])
            execute_complete_readonly(row["gold_sql"], args.db)
        except Exception as exc:
            raise SystemExit(f"item {index} gold_sql failed validation/execution: {exc}") from exc
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise SystemExit("blind item id must be unique")
    if len({normalise_question(str(row["question"])) for row in rows}) != len(rows):
        raise SystemExit("normalised blind question must be unique")
    difficulty_counts = Counter(row["difficulty"] for row in rows)
    category_counts = Counter(row["category"] for row in rows)
    if args.quota_contract == "legacy-300":
        if dict(difficulty_counts) != EXPECTED_DIFFICULTY_QUOTAS:
            raise SystemExit(
                "independent corpus difficulty quotas differ from the declared contract"
            )
        if dict(category_counts) != EXPECTED_CATEGORY_QUOTAS:
            raise SystemExit(
                "independent corpus category quotas differ from the declared contract"
            )
    else:
        declared_corpus = curator.get("corpus", {})
        if (
            dict(difficulty_counts)
            != declared_corpus.get("difficulty_distribution")
            or dict(category_counts)
            != declared_corpus.get("category_distribution")
            or min(difficulty_counts.values(), default=0) < 40
            or min(category_counts.values(), default=0) < 15
        ):
            raise SystemExit(
                "independent corpus distribution differs from its attestation"
            )

    try:
        groups, partition_stats = _partition_groups(rows)
    except ValueError as exc:
        raise SystemExit(f"blind partition construction failed: {exc}") from exc

    try:
        holdout_indices = _choose_holdout_indices(
            rows=rows,
            groups=groups,
            seed=args.seed,
            minimum_holdout=args.holdout_count,
            maximum_holdout=len(rows) - 150,
            minimum_hard=40,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    holdout_index_set = set(holdout_indices)
    development_indices = [
        index for index in range(len(rows)) if index not in holdout_index_set
    ]

    holdout = [rows[index] for index in sorted(holdout_indices)]
    development = [rows[index] for index in sorted(development_indices)]
    if len(development) < 150:
        raise SystemExit("AST grouping left fewer than 150 development items")
    hard_count = sum(item["difficulty"] in {"hard", "extra", "extra-hard"} for item in holdout)
    if hard_count < 40:
        raise SystemExit(f"holdout has only {hard_count} hard/extra items; minimum is 40")
    if len(holdout) < args.holdout_count:
        raise SystemExit("AST 分组后盲测题数不足")

    final_output = args.output.resolve()
    final_dir = final_output.parent
    if (
        final_output.name != "freeze_manifest.json"
        or final_dir.name != "frozen"
        or final_dir.parent.name != "nl2sql_blind_v3"
    ):
        raise SystemExit(
            "freeze output must be "
            "outputs/private/<RID>/nl2sql_blind_v3/frozen/"
            "freeze_manifest.json"
        )
    expected_public = (
        args.public_output.resolve().parents[2]
        / output_rid
        / "nl2sql"
        / "independent_v3_freeze_attestation.json"
    )
    if args.public_output.resolve() != expected_public:
        raise SystemExit("independent-v3 public attestation path must be fixed")
    if final_dir.exists():
        raise SystemExit(f"freeze destination already exists; choose a new RID directory: {final_dir}")
    if args.public_output.exists():
        raise SystemExit(
            f"public freeze attestation already exists: {args.public_output}"
        )
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = final_dir.parent / f".{final_dir.name}.freeze-staging"
    if staging_dir.exists():
        raise SystemExit(f"stale freeze staging exists; inspect it without reusing it: {staging_dir}")
    staging_dir.mkdir()
    output = staging_dir / final_output.name
    holdout_path = staging_dir / "holdout.jsonl"
    development_path = staging_dir / "development.jsonl"

    _write_bytes_synced(holdout_path, _jsonl_bytes(holdout))
    _write_bytes_synced(development_path, _jsonl_bytes(development))
    payload = {
        "schema_version": 4,
        "kind": "independent_curated_v3_four_table_blind_freeze",
        "created_once": True,
        "created_at": args.created_at,
        "rid": output_rid,
        "input_path": str(args.input.resolve()),
        "input_sha256": _sha256(args.input),
        "independent_curation": {
            "curator_format": args.curator_format,
            "curator_pack_manifest_path": str(
                args.curator_pack_manifest.resolve()
            ),
            "curator_pack_manifest_sha256": _sha256(
                args.curator_pack_manifest
            ),
            "attestation_path": str(args.curator_attestation.resolve()),
            "attestation_sha256": _sha256(args.curator_attestation),
            "declaration": curator,
            "pack_dataset_role": pack["dataset_role"],
            "aggregate_audit": (
                _binding(args.curator_aggregate_audit)
                if args.curator_aggregate_audit is not None
                else None
            ),
            "execution_audit": (
                _binding(args.curator_execution_audit)
                if args.curator_execution_audit is not None
                else None
            ),
            "sha256s": (
                _binding(args.curator_sha256s)
                if args.curator_sha256s is not None
                else None
            ),
        },
        "source_version": args.source_version,
        "freeze_tool_sha256": _sha256(Path(__file__)),
        "protocol_sha256": _sha256(
            ROOT / "dki" / "analysis" / "nl2sql_protocol.py"
        ),
        "task1_manifest": _binding(args.task1_manifest),
        "kg": _binding(args.kg),
        "database": _binding(args.db),
        "seed": args.seed,
        "n_total": len(rows),
        "n_sql_ast_groups": partition_stats["n_sql_ast_groups"],
        "n_partition_families": partition_stats["n_partition_families"],
        "n_near_duplicate_links": partition_stats["n_near_duplicate_links"],
        "n_development": len(development),
        "n_holdout": len(holdout),
        "n_hard": hard_count,
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "minimum_holdout": args.holdout_count,
        "development": {
            "path": development_path.name,
            "sha256": _sha256(development_path),
            "size": development_path.stat().st_size,
        },
        "holdout": {
            "path": holdout_path.name,
            "sha256": _sha256(holdout_path),
            "size": holdout_path.stat().st_size,
        },
        "partition": {
            "version": "v4",
            "normalised_question_intersection": 0,
            "near_duplicate_question_intersection": 0,
            "sql_ast_skeleton_intersection": 0,
            "whole_ast_groups_only": True,
            "whole_partition_families_only": True,
            "near_duplicate_policy": {
                "character_3gram_jaccard_gte": 0.85,
                "sequence_matcher_ratio_gte": 0.90,
                "transitive_union_find": True,
            },
        },
        "development_aggregate": {
            "n": len(development),
            "n_hard": sum(
                item["difficulty"] in {"hard", "extra", "extra-hard"}
                for item in development
            ),
            "difficulty_counts": dict(
                sorted(
                    Counter(
                        item["difficulty"] for item in development
                    ).items()
                )
            ),
            "category_counts": dict(
                sorted(
                    Counter(
                        item["category"] for item in development
                    ).items()
                )
            ),
        },
        "holdout_aggregate": {
            "n": len(holdout),
            "n_hard": hard_count,
            "difficulty_counts": dict(
                sorted(
                    Counter(
                        item["difficulty"] for item in holdout
                    ).items()
                )
            ),
            "category_counts": dict(
                sorted(
                    Counter(
                        item["category"] for item in holdout
                    ).items()
                )
            ),
        },
        "policy": (
            "normalized SQL AST and deterministic near-duplicate question "
            "families are never split across development and holdout; deterministic "
            "bounded subset DP targets the minimum "
            "holdout size while preserving at least 150 development items and "
            "40 hard/extra holdout items"
        ),
        "semantic_grounding": {
            "validated": True,
            "policy": "every direct predicate and LIMIT literal must be explicit in its question",
        },
        "privacy": {
            "manifest_contains_questions": False,
            "manifest_contains_gold_sql": False,
            "manifest_contains_predictions": False,
            "manifest_contains_error_samples": False,
            "raw_corpus_or_frozen_rows_allowed_in_submission": False,
        },
        "selection_used": False,
    }
    _write_bytes_synced(output, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
    try:
        os.rename(staging_dir, final_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    output = final_output
    public_payload = {
        "schema_version": 1,
        "kind": "independent_curated_v3_blind_freeze_attestation",
        "status": "frozen-unconsumed",
        "rid": output_rid,
        "created_at": args.created_at,
        "source_tree_sha256": args.source_version,
        "freeze_manifest_sha256": _sha256(output),
        "counts": {
            "n_total": payload["n_total"],
            "n_development": payload["n_development"],
            "n_holdout": payload["n_holdout"],
            "n_hard": payload["n_hard"],
            "n_sql_ast_groups": payload["n_sql_ast_groups"],
            "n_partition_families": payload["n_partition_families"],
            "n_near_duplicate_links": payload["n_near_duplicate_links"],
            "holdout_difficulty_counts": payload[
                "holdout_aggregate"
            ]["difficulty_counts"],
            "holdout_category_counts": payload[
                "holdout_aggregate"
            ]["category_counts"],
        },
        "bindings": {
            "curator_pack_manifest_sha256": _sha256(
                args.curator_pack_manifest
            ),
            "curator_attestation_sha256": _sha256(
                args.curator_attestation
            ),
            "corpus_sha256": _sha256(args.input),
            "task1_manifest_sha256": _sha256(args.task1_manifest),
            "kg_sha256": _sha256(args.kg),
            "database_sha256": _sha256(args.db),
            "curator_aggregate_audit_sha256": (
                _sha256(args.curator_aggregate_audit)
                if args.curator_aggregate_audit is not None
                else None
            ),
            "curator_execution_audit_sha256": (
                _sha256(args.curator_execution_audit)
                if args.curator_execution_audit is not None
                else None
            ),
            "curator_sha256s_sha256": (
                _sha256(args.curator_sha256s)
                if args.curator_sha256s is not None
                else None
            ),
        },
        "partition": payload["partition"],
        "selection_used": False,
        "contains_questions": False,
        "contains_gold_sql": False,
        "contains_predictions": False,
        "contains_error_samples": False,
    }
    _publish_bytes_once(
        args.public_output,
        (
            json.dumps(
                public_payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
    )
    print(json.dumps({
        "manifest": str(output),
        "public_attestation": str(args.public_output.resolve()),
        "n_total": payload["n_total"],
        "n_holdout": payload["n_holdout"],
        "n_sql_ast_groups": payload["n_sql_ast_groups"],
        "selection_used": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
