"""Re-audit only development after an independent blind freeze was consumed.

This entry point is deliberately separate from
``eval_nl2sql_frozen_development.py``.  The prospective evaluator must keep
rejecting consumed freezes.  This script is retrospective development evidence
only: it never promotes a candidate, never creates or amends a blind seal, and
never opens or hashes the holdout member.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.analysis.database import database_matches_manifest
from dki.analysis.nl2sql import NL2SQL
from dki.analysis.nl2sql_development_evidence import (
    strict_development_jsonl_rows,
    strict_json_loads,
)
from dki.analysis.nl2sql_eval import evaluate_set
from dki.analysis.nl2sql_protocol import (
    normalise_development_metadata,
    normalise_question,
    validate_gold_sql,
    validate_question_sql_grounding,
)
from dki.analysis.nl2sql_seal import (
    DEVELOPMENT_COVERAGE_PERCENT,
    DEVELOPMENT_EXTRA_HARD_PERCENT,
    DEVELOPMENT_HARD_PLUS_PERCENT,
    DEVELOPMENT_MIN_N,
    DEVELOPMENT_OVERALL_PERCENT,
    _agent_runtime_profile,
    _verify_source_tree_manifest,
)
from dki.pipeline.artifacts import validate_task2_artifact


DATASET_ROLE = "development_selection_used_post_consumption_reaudit"
FREEZE_KIND = "independent_curated_v3_four_table_blind_freeze"
RECEIPT_NAME = "evaluation_receipt.json"
INVALID_MARKER_NAME = "consumed-invalid.json"
STARTED_MARKER_NAME = "consumed-started.json"
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_file(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = strict_json_loads(Path(path).read_bytes(), label=label)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if type(payload) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _rid_below(path: Path, scope: str) -> str:
    parts = Path(path).resolve().parts
    for index in range(len(parts) - 2):
        if (
            parts[index].casefold() == "outputs"
            and parts[index + 1].casefold() == scope.casefold()
        ):
            return parts[index + 2]
    raise ValueError(f"artifact is not below outputs/{scope}/<RID>: {path}")


def _safe_member(root: Path, relative: object, expected_name: str) -> Path:
    relative_path = Path(str(relative))
    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or len(relative_path.parts) != 1
        or relative_path.name != expected_name
    ):
        raise ValueError(f"frozen member must be exactly {expected_name}")
    return Path(root) / relative_path


def _binding_matches(path: Path, binding: object, label: str) -> None:
    if not isinstance(binding, Mapping):
        raise ValueError(f"freeze lacks {label} binding")
    digest = binding.get("sha256")
    size = binding.get("size")
    if (
        type(digest) is not str
        or _SHA256.fullmatch(digest) is None
        or type(size) is not int
        or size < 0
        or not Path(path).is_file()
        or Path(path).stat().st_size != size
        or _sha256(path) != digest
    ):
        raise ValueError(f"freeze {label} binding failed")


def _ordinary_file_identity(path: Path, label: str) -> tuple[int, int, Path]:
    """Return a stat-only identity and reject links before any byte read."""

    path = Path(path)
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"{label} is missing or not stat-readable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be an ordinary non-symlink file")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} cannot be resolved safely") from exc
    return int(metadata.st_dev), int(metadata.st_ino), resolved


def _reject_holdout_aliases(
    holdout_path: Path,
    candidates: Mapping[str, Path],
) -> None:
    """Ensure no readable input is the holdout under another name.

    Only filesystem metadata is inspected.  This function must run before a
    candidate is opened, parsed, copied, or hashed.
    """

    holdout_identity = _ordinary_file_identity(holdout_path, "frozen holdout")
    for label, candidate in candidates.items():
        identity = _ordinary_file_identity(Path(candidate), label)
        same_inode = (
            holdout_identity[0] == identity[0]
            and holdout_identity[1] != 0
            and holdout_identity[1] == identity[1]
        )
        if same_inode or holdout_identity[2] == identity[2]:
            raise ValueError(f"{label} aliases the frozen holdout")


def _reject_private_lineage_inputs(candidates: Mapping[str, Path]) -> None:
    """Keep non-dataset inputs outside every outputs/private subtree."""

    for label, candidate in candidates.items():
        _device, _inode, resolved = _ordinary_file_identity(candidate, label)
        folded = [part.casefold() for part in resolved.parts]
        if any(
            folded[index : index + 2] == ["outputs", "private"]
            for index in range(max(0, len(folded) - 1))
        ):
            raise ValueError(f"{label} must not come from outputs/private")


def _preflight_source_tree_without_private(
    source_tree_manifest: Path,
    *,
    holdout_path: Path,
    root: Path,
) -> None:
    """Reject private or holdout-alias entries before the seal verifier hashes."""

    payload = _strict_json_file(source_tree_manifest, "included-tree manifest")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("included-tree manifest lacks entries")
    holdout_identity = _ordinary_file_identity(holdout_path, "frozen holdout")
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"included-tree entry {index} is invalid")
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise ValueError(f"included-tree entry {index} path is invalid")
        parts = relative.split("/")
        if any(part in {"", ".", ".."} for part in parts) or Path(relative).is_absolute():
            raise ValueError(f"included-tree entry {index} path is unsafe")
        folded = [part.casefold() for part in parts]
        if len(folded) >= 2 and folded[0:2] == ["outputs", "private"]:
            raise ValueError("included-tree manifest must not contain outputs/private")
        candidate = Path(root).joinpath(*parts)
        current = Path(root)
        for part in parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(
                    f"included-tree entry traverses a symlink: {relative}"
                )
        try:
            metadata = os.lstat(candidate)
        except FileNotFoundError:
            # The authoritative verifier will report the missing source file.
            continue
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or int(metadata.st_nlink) != 1
        ):
            raise ValueError(f"included-tree entry is not an ordinary file: {relative}")
        same_inode = (
            int(metadata.st_dev) == holdout_identity[0]
            and int(metadata.st_ino) != 0
            and int(metadata.st_ino) == holdout_identity[1]
        )
        try:
            same_resolved = candidate.resolve(strict=True) == holdout_identity[2]
        except OSError as exc:
            raise ValueError(f"included-tree entry cannot be resolved: {relative}") from exc
        if same_inode or same_resolved:
            raise ValueError("included-tree entry aliases the frozen holdout")


def _load_post_consumption_contract(
    *,
    manifest: Path,
    task1_manifest: Path,
    kg_path: Path,
    db_path: Path,
    source_tree_manifest: Path,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Validate a consumed freeze while reading only its development dataset.

    The holdout path is resolved to a single frozen member and checked with
    ``is_file``/``stat`` only.  Its declared hash is syntax-checked because that
    value is already part of the immutable manifest, but the holdout bytes are
    never opened, parsed, copied, or hashed here.
    """

    manifest = Path(manifest).resolve()
    freeze_rid = _rid_below(manifest, "private")
    # The immutable consumed freeze may live under an older source RID.  Bind
    # its canonical outputs/private suffix instead of requiring private blind
    # bytes to be copied into the current candidate source tree.
    if (
        manifest.name != "freeze_manifest.json"
        or manifest.parent.name != "frozen"
        or manifest.parent.parent.name != "nl2sql_blind_v3"
        or manifest.parent.parent.parent.name != freeze_rid
        or manifest.parent.parent.parent.parent.name.casefold() != "private"
        or manifest.parent.parent.parent.parent.parent.name.casefold() != "outputs"
    ):
        raise ValueError("manifest must be the fixed independent-v3 freeze path")
    development_path = manifest.parent / "development.jsonl"
    holdout_path = manifest.parent / "holdout.jsonl"
    started_marker = manifest.parent / STARTED_MARKER_NAME
    consumption_receipt_path = manifest.parent / RECEIPT_NAME
    public_inputs = {
        "task1 manifest": Path(task1_manifest),
        "knowledge graph": Path(kg_path),
        "database": Path(db_path),
        "included-tree manifest": Path(source_tree_manifest),
    }
    # Alias detection outranks lineage placement: an input that IS the frozen
    # holdout under another name must be reported as such, not as a generic
    # private-lineage violation.  Both checks read only filesystem metadata.
    _reject_holdout_aliases(
        holdout_path,
        {
            "blind freeze manifest": manifest,
            "frozen development": development_path,
            "consumed-started marker": started_marker,
            "consumption receipt": consumption_receipt_path,
            **public_inputs,
        },
    )
    _reject_private_lineage_inputs(public_inputs)

    freeze = _strict_json_file(manifest, "blind freeze manifest")
    partition = freeze.get("partition")
    privacy = freeze.get("privacy")
    if (
        freeze.get("schema_version") != 4
        or freeze.get("kind") != FREEZE_KIND
        or freeze.get("created_once") is not True
        or freeze.get("selection_used") is not False
        or freeze.get("rid") != freeze_rid
        or type(freeze.get("n_development")) is not int
        or freeze["n_development"] < DEVELOPMENT_MIN_N
        or type(freeze.get("n_holdout")) is not int
        or freeze["n_holdout"] < 120
        or type(freeze.get("n_hard")) is not int
        or freeze["n_hard"] < 40
        or not isinstance(partition, Mapping)
        or partition.get("version") != "v4"
        or partition.get("normalised_question_intersection") != 0
        or partition.get("near_duplicate_question_intersection") != 0
        or partition.get("sql_ast_skeleton_intersection") != 0
        or not isinstance(privacy, Mapping)
        or privacy.get("manifest_contains_questions") is not False
        or privacy.get("manifest_contains_gold_sql") is not False
        or privacy.get("manifest_contains_predictions") is not False
        or privacy.get("manifest_contains_error_samples") is not False
    ):
        raise ValueError("blind freeze manifest violates the development re-audit contract")

    manifest_sha256 = _sha256(manifest)
    invalid_marker = manifest.parent / INVALID_MARKER_NAME
    if invalid_marker.exists():
        raise ValueError("consumed-invalid freeze cannot be used for development re-audit")
    if not started_marker.is_file() or not consumption_receipt_path.is_file():
        raise ValueError("development re-audit requires a completed consumed freeze")
    consumption = _strict_json_file(
        consumption_receipt_path,
        "blind aggregate consumption receipt",
    )
    if (
        consumption.get("schema_version") != 1
        or consumption.get("status") != "consumed"
        or consumption.get("freeze_manifest_sha256") != manifest_sha256
        or type(consumption.get("gate_passed")) is not bool
        or _SHA256.fullmatch(str(consumption.get("candidate_seal_sha256", "")))
        is None
    ):
        raise ValueError("blind aggregate consumption receipt is invalid")

    expected_bindings = {
        "task1_manifest": (Path(task1_manifest), freeze.get("task1_manifest")),
        "kg": (Path(kg_path), freeze.get("kg")),
        "database": (Path(db_path), freeze.get("database")),
    }
    for label, (path, binding) in expected_bindings.items():
        _binding_matches(path, binding, label)

    development = freeze.get("development")
    holdout = freeze.get("holdout")
    if not isinstance(development, Mapping) or not isinstance(holdout, Mapping):
        raise ValueError("freeze lacks development/holdout bindings")
    development_path = _safe_member(
        manifest.parent,
        development.get("path"),
        "development.jsonl",
    )
    _binding_matches(development_path, development, "development")

    # HOLDOUT SAFETY BOUNDARY: do not pass this path to _sha256, read_bytes,
    # open, SQLite, or any parser.  Only the immutable declaration and stat are
    # checked.  A dedicated regression test makes any byte read fail loudly.
    holdout_path = _safe_member(
        manifest.parent,
        holdout.get("path"),
        "holdout.jsonl",
    )
    if (
        _SHA256.fullmatch(str(holdout.get("sha256", ""))) is None
        or type(holdout.get("size")) is not int
        or holdout["size"] < 0
        or not holdout_path.is_file()
        or holdout_path.stat().st_size != holdout["size"]
    ):
        raise ValueError("holdout declaration/stat binding failed")

    return {
        "freeze": freeze,
        "freeze_rid": freeze_rid,
        "freeze_manifest_sha256": manifest_sha256,
        "development_path": development_path,
        "development_sha256": str(development["sha256"]),
        "consumption_receipt_sha256": _sha256(consumption_receipt_path),
        "historic_candidate_seal_sha256": str(
            consumption["candidate_seal_sha256"]
        ),
        "holdout_stat_checked": True,
        "holdout_opened": False,
        "holdout_hashed": False,
        "holdout_alias_preflight": True,
    }


def _load_development(path: Path, *, expected_n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    questions: set[str] = set()
    for raw in strict_development_jsonl_rows(
        Path(path).read_bytes(),
        label="independent frozen development post-consumption re-audit",
    ):
        row = normalise_development_metadata(raw)
        validate_gold_sql(row["gold_sql"])
        validate_question_sql_grounding(row["question"], row["gold_sql"])
        item_id = str(row["id"])
        question_key = normalise_question(str(row["question"]))
        if item_id in ids or question_key in questions:
            raise ValueError("development ids and normalised questions must be unique")
        ids.add(item_id)
        questions.add(question_key)
        rows.append(row)
    if len(rows) != expected_n:
        raise ValueError("development row count differs from freeze manifest")
    return rows


def _validate_engine_strategy(
    *,
    engine: str,
    llm_strategy: str,
    agent_rule_policy: str,
    llm_reasoning_mode: str,
    llm_completion_tokens: int,
) -> None:
    if llm_reasoning_mode not in {"disabled", "adaptive"}:
        raise ValueError("unsupported Agent LLM reasoning mode")
    if (
        type(llm_completion_tokens) is not int
        or not 1 <= llm_completion_tokens <= 131072
    ):
        raise ValueError("Agent LLM completion tokens are out of range")
    if engine != "agent" and (
        llm_strategy != "legacy_sql"
        or agent_rule_policy != "rule_first"
        or llm_reasoning_mode != "disabled"
        or llm_completion_tokens != 1024
    ):
        raise ValueError("non-default Agent LLM options require engine=agent")
    if engine == "agent" and (llm_strategy, agent_rule_policy) not in {
        ("legacy_sql", "rule_first"),
        ("legacy_sql", "model_first"),
        ("query_plan", "rule_first"),
        ("query_plan", "plan_first"),
        ("query_plan_v2", "rule_first"),
        ("query_plan_v2", "plan_first"),
    }:
        raise ValueError("unsupported Agent LLM strategy/rule-policy combination")


def _nl2sql_for_evaluation(
    db_path: Path,
    *,
    engine: str,
    llm_strategy: str,
    agent_rule_policy: str,
    llm_reasoning_mode: str,
    llm_completion_tokens: int,
) -> NL2SQL:
    _validate_engine_strategy(
        engine=engine,
        llm_strategy=llm_strategy,
        agent_rule_policy=agent_rule_policy,
        llm_reasoning_mode=llm_reasoning_mode,
        llm_completion_tokens=llm_completion_tokens,
    )
    return NL2SQL(
        db_path=db_path,
        llm_strategy=llm_strategy,
        agent_rule_policy=agent_rule_policy,
        llm_reasoning_mode=llm_reasoning_mode,
        llm_completion_tokens=llm_completion_tokens,
        llm_fail_closed=(engine == "agent"),
    )


def _required_source_paths(engine: str, llm_strategy: str) -> tuple[str, ...]:
    paths = [
        "dki/analysis/database.py",
        "dki/analysis/nl2sql.py",
        "dki/analysis/nl2sql_development_evidence.py",
        "dki/analysis/nl2sql_eval.py",
        "dki/analysis/nl2sql_protocol.py",
        "dki/analysis/nl2sql_seal.py",
        "dki/pipeline/artifacts.py",
        "scripts/eval_nl2sql_consumed_development.py",
    ]
    if engine == "agent":
        paths.append("dki/llm.py")
        if llm_strategy == "query_plan":
            paths.append("dki/analysis/nl2sql_query_plan.py")
        elif llm_strategy == "query_plan_v2":
            paths.append("dki/analysis/nl2sql_query_plan_v2.py")
    return tuple(paths)


def _current_source_bindings(
    source_tree: Mapping[str, Any],
    *,
    root: Path,
    engine: str,
    llm_strategy: str,
) -> dict[str, str]:
    entries = source_tree.get("entries")
    if not isinstance(entries, list):
        raise ValueError("included-tree manifest lacks entries")
    by_path = {
        str(entry.get("path", "")).replace("\\", "/"): entry
        for entry in entries
        if isinstance(entry, Mapping)
    }
    result: dict[str, str] = {}
    for relative in _required_source_paths(engine, llm_strategy):
        entry = by_path.get(relative)
        path = Path(root) / Path(relative)
        if (
            not isinstance(entry, Mapping)
            or not path.is_file()
            or entry.get("size") != path.stat().st_size
            or entry.get("sha256") != _sha256(path)
        ):
            raise ValueError(f"included tree does not bind current source: {relative}")
        result[relative] = str(entry["sha256"])
    return result


def _development_gate(overall: Mapping[str, Any]) -> dict[str, Any]:
    def integer(name: str) -> int:
        value = overall.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(f"development aggregate {name} must be a nonnegative integer")
        return value

    n = integer("n")
    correct = integer("n_correct")
    answered = integer("n_answered")
    if correct > answered or answered > n:
        raise ValueError("development aggregate counts are impossible")
    layers = overall.get("by_difficulty")
    if not isinstance(layers, Mapping):
        raise ValueError("development aggregate lacks difficulty layers")
    counts: dict[str, tuple[int, int]] = {}
    for name, raw in layers.items():
        if type(name) is not str or not isinstance(raw, Mapping):
            raise ValueError("development difficulty layer is invalid")
        layer_n = raw.get("n")
        layer_correct = raw.get("correct")
        if (
            type(layer_n) is not int
            or type(layer_correct) is not int
            or layer_n < 0
            or layer_correct < 0
            or layer_correct > layer_n
        ):
            raise ValueError("development difficulty counts are invalid")
        counts[name] = (layer_n, layer_correct)
    if sum(value[0] for value in counts.values()) != n or sum(
        value[1] for value in counts.values()
    ) != correct:
        raise ValueError("development difficulty totals differ from overall")
    hard_n = sum(counts.get(name, (0, 0))[0] for name in ("hard", "extra", "extra-hard"))
    hard_correct = sum(
        counts.get(name, (0, 0))[1] for name in ("hard", "extra", "extra-hard")
    )
    extra_hard_n, extra_hard_correct = counts.get("extra-hard", (0, 0))
    failures: list[str] = []
    if n < DEVELOPMENT_MIN_N:
        failures.append("development_n_below_180")
    if correct * 100 < DEVELOPMENT_OVERALL_PERCENT * n:
        failures.append("development_overall_below_0_92")
    if answered * 100 < DEVELOPMENT_COVERAGE_PERCENT * n:
        failures.append("development_coverage_below_0_97")
    if hard_n == 0:
        failures.append("development_hard_plus_missing")
    elif hard_correct * 100 < DEVELOPMENT_HARD_PLUS_PERCENT * hard_n:
        failures.append("development_hard_plus_below_0_85")
    if extra_hard_n == 0:
        failures.append("development_extra_hard_missing")
    elif extra_hard_correct * 100 < DEVELOPMENT_EXTRA_HARD_PERCENT * extra_hard_n:
        failures.append("development_extra_hard_below_0_70")
    return {
        "passed": not failures,
        "reasons": failures or ["development_quality_gate_passed"],
        "n": n,
        "n_correct": correct,
        "n_answered": answered,
        "hard_plus_n": hard_n,
        "hard_plus_correct": hard_correct,
        "extra_hard_n": extra_hard_n,
        "extra_hard_correct": extra_hard_correct,
        "thresholds": {
            "minimum_n": DEVELOPMENT_MIN_N,
            "overall_minimum_percent": DEVELOPMENT_OVERALL_PERCENT,
            "hard_plus_minimum_percent": DEVELOPMENT_HARD_PLUS_PERCENT,
            "extra_hard_minimum_percent": DEVELOPMENT_EXTRA_HARD_PERCENT,
            "coverage_minimum_percent": DEVELOPMENT_COVERAGE_PERCENT,
        },
    }


def _public_overall(result: Mapping[str, Any]) -> dict[str, Any]:
    allowed = (
        "engine",
        "n",
        "n_correct",
        "accuracy",
        "n_answered",
        "n_abstained",
        "coverage",
        "execution_accuracy_answered",
        "strict_result_contract",
        "result_order_policy",
        "by_difficulty",
    )
    return {name: result[name] for name in allowed}


def _build_receipts(
    *,
    common: Mapping[str, Any],
    result: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    gate = _development_gate(result)
    aggregate = _public_overall(result)
    private = {
        **dict(common),
        "kind": "private_post_consumption_development_reaudit",
        "overall": aggregate,
        "development_gate": gate,
        "details": result.get("details", []),
    }
    public = {
        **dict(common),
        "kind": "post_consumption_development_reaudit_public_attestation",
        "status": "completed",
        "overall": aggregate,
        "development_gate": gate,
        "contains_questions": False,
        "contains_gold_sql": False,
        "contains_predictions": False,
        "contains_error_samples": False,
    }
    return private, public


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _stage(path: Path, data: bytes) -> Path:
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


def _publish_pair_once(
    *,
    private_path: Path,
    private_payload: Mapping[str, Any],
    public_path: Path,
    public_payload: Mapping[str, Any],
) -> None:
    private_path = Path(private_path).resolve()
    public_path = Path(public_path).resolve()
    if private_path.exists() or public_path.exists():
        raise FileExistsError("post-consumption development receipt already exists")
    public_body = dict(public_payload)
    private_document = dict(private_payload)
    private_document["pairing"] = {
        "protocol": "post_consumption_development_private_first_v1",
        "public_body": public_body,
    }
    private_bytes = _json_bytes(private_document)
    public_document = {
        **public_body,
        "pairing": {
            "protocol": "post_consumption_development_private_first_v1",
            "private_sha256": hashlib.sha256(private_bytes).hexdigest(),
            "published_from_private": True,
        },
    }
    private_temporary = _stage(private_path, private_bytes)
    private_published = False
    try:
        public_temporary = _stage(public_path, _json_bytes(public_document))
        try:
            os.link(private_temporary, private_path)
            private_published = True
            os.link(public_temporary, public_path)
        except Exception:
            if private_published:
                try:
                    if os.path.samefile(private_path, private_temporary):
                        private_path.unlink()
                except OSError:
                    # Preserve an ambiguous external filesystem state for
                    # manual audit; never delete a path we cannot rebind to
                    # this invocation's staging inode.
                    pass
            raise
        finally:
            public_temporary.unlink(missing_ok=True)
    finally:
        private_temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--kg", type=Path, required=True)
    parser.add_argument("--task1-manifest", type=Path, required=True)
    parser.add_argument("--source-tree-manifest", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--engine", choices=["rule", "agent"], default="rule")
    parser.add_argument(
        "--llm-strategy",
        choices=["legacy_sql", "query_plan", "query_plan_v2"],
        default="legacy_sql",
    )
    parser.add_argument(
        "--agent-rule-policy",
        choices=["rule_first", "model_first", "plan_first"],
        default="rule_first",
    )
    parser.add_argument(
        "--llm-reasoning-mode",
        choices=["disabled", "adaptive"],
        default="disabled",
    )
    parser.add_argument("--llm-completion-tokens", type=int, default=1024)
    args = parser.parse_args()

    try:
        _validate_engine_strategy(
            engine=args.engine,
            llm_strategy=args.llm_strategy,
            agent_rule_policy=args.agent_rule_policy,
            llm_reasoning_mode=args.llm_reasoning_mode,
            llm_completion_tokens=args.llm_completion_tokens,
        )
        manifest = args.manifest.resolve()
        source_tree_manifest = args.source_tree_manifest.resolve()
        contract = _load_post_consumption_contract(
            manifest=manifest,
            task1_manifest=args.task1_manifest,
            kg_path=args.kg,
            db_path=args.db,
            source_tree_manifest=source_tree_manifest,
            root=ROOT,
        )
        _preflight_source_tree_without_private(
            source_tree_manifest,
            holdout_path=manifest.parent / "holdout.jsonl",
            root=ROOT,
        )
        source_tree = _verify_source_tree_manifest(source_tree_manifest)
        candidate_rid = source_tree.get("rid")
        if (
            type(candidate_rid) is not str
            or not candidate_rid
            or any(
                character
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                for character in candidate_rid
            )
        ):
            raise ValueError("included-tree manifest lacks a safe candidate RID")
        source_tree_sha256 = str(source_tree["included_tree_sha256"])
        source_bindings = _current_source_bindings(
            source_tree,
            root=ROOT,
            engine=args.engine,
            llm_strategy=args.llm_strategy,
        )

        freeze_rid = str(contract["freeze_rid"])
        private_output = args.private_output.resolve()
        public_output = args.public_output.resolve()
        expected_private = (
            ROOT
            / "outputs"
            / "private"
            / candidate_rid
            / "nl2sql_development_post_consumption"
            / freeze_rid
            / "development_selection_used_post_consumption_reaudit.json"
        ).resolve()
        expected_public = (
            ROOT
            / "outputs"
            / "remote_runs"
            / candidate_rid
            / "nl2sql"
            / "post_consumption_development_reaudit.json"
        ).resolve()
        if private_output != expected_private or public_output != expected_public:
            raise ValueError("post-consumption development output paths are not canonical")
        if (
            _rid_below(private_output, "private") != candidate_rid
            or _rid_below(public_output, "remote_runs") != candidate_rid
        ):
            raise ValueError("development outputs must use the candidate RID")
        if private_output.exists() or public_output.exists():
            raise FileExistsError("post-consumption development receipt already exists")

        validate_task2_artifact(args.kg, task1_path=args.task1_manifest)
        if not database_matches_manifest(
            args.db,
            args.task1_manifest,
            args.kg,
        ):
            raise ValueError("database does not match Task 1/KG lineage")
        rows = _load_development(
            contract["development_path"],
            expected_n=int(contract["freeze"]["n_development"]),
        )
        nl = _nl2sql_for_evaluation(
            args.db,
            engine=args.engine,
            llm_strategy=args.llm_strategy,
            agent_rule_policy=args.agent_rule_policy,
            llm_reasoning_mode=args.llm_reasoning_mode,
            llm_completion_tokens=args.llm_completion_tokens,
        )
        agent_profile = (
            _agent_runtime_profile(
                llm_strategy=args.llm_strategy,
                agent_rule_policy=args.agent_rule_policy,
                llm_reasoning_mode=args.llm_reasoning_mode,
                llm_completion_tokens=args.llm_completion_tokens,
            )
            if args.engine == "agent"
            else None
        )
        result = evaluate_set(
            rows,
            args.engine,
            args.db,
            nl=nl,
            strict=True,
        )
        common: dict[str, Any] = {
            "schema_version": 1,
            "rid": candidate_rid,
            "candidate_rid": candidate_rid,
            "freeze_rid": freeze_rid,
            "dataset_role": DATASET_ROLE,
            "selection_used": True,
            "created_once": True,
            "post_consumption_reaudit": True,
            "blind_claim": False,
            "candidate_seal_eligible": False,
            "blind_holdout_reused": False,
            "holdout_opened": False,
            "holdout_hashed": False,
            "holdout_alias_preflight": contract["holdout_alias_preflight"],
            "source_manifest_private_entries_rejected": True,
            "holdout_validation": "declared SHA-256 syntax plus stat size only",
            "dataset_content_access": ["development"],
            "engine": args.engine,
            "source_tree_sha256": source_tree_sha256,
            "source_tree_manifest_sha256": _sha256(source_tree_manifest),
            "source_bindings": source_bindings,
            "freeze_manifest_sha256": contract["freeze_manifest_sha256"],
            "development_sha256": contract["development_sha256"],
            "database_sha256": _sha256(args.db),
            "task1_manifest_sha256": _sha256(args.task1_manifest),
            "kg_sha256": _sha256(args.kg),
            "historic_consumption_status": "consumed",
            "historic_consumption_receipt_sha256": contract[
                "consumption_receipt_sha256"
            ],
            "historic_candidate_seal_sha256": contract[
                "historic_candidate_seal_sha256"
            ],
            "runner_sha256": _sha256(Path(__file__)),
            "evaluator_sha256": _sha256(
                ROOT / "dki" / "analysis" / "nl2sql_eval.py"
            ),
            "protocol_sha256": _sha256(
                ROOT / "dki" / "analysis" / "nl2sql_protocol.py"
            ),
            "engine_source_sha256": _sha256(
                ROOT / "dki" / "analysis" / "nl2sql.py"
            ),
        }
        if agent_profile is not None:
            common["agent_profile"] = agent_profile
        private_payload, public_payload = _build_receipts(
            common=common,
            result=result,
        )
        _publish_pair_once(
            private_path=private_output,
            private_payload=private_payload,
            public_path=public_output,
            public_payload=public_payload,
        )
        print(
            json.dumps(
                {
                    "rid": candidate_rid,
                    "freeze_rid": freeze_rid,
                    "dataset_role": DATASET_ROLE,
                    "n": result["n"],
                    "n_correct": result["n_correct"],
                    "accuracy": result["accuracy"],
                    "coverage": result["coverage"],
                    "development_gate_passed": public_payload[
                        "development_gate"
                    ]["passed"],
                    "blind_claim": False,
                    "candidate_seal_eligible": False,
                    "private_output": str(private_output),
                    "public_output": str(public_output),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
