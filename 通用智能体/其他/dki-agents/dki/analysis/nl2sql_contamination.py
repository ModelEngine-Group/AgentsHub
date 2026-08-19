"""Deterministic contamination audit for NL2SQL evaluation candidates.

Before any new blind or holdout set is consumed, every candidate item must be
proven disjoint from ALL historical/consumed assets under the published
fingerprint contract (exact question, near-duplicate question, normalized SQL,
AST skeleton, and transitive semantic family), and the runtime import closure
of the production engine must contain no history-only assets.

The private report carries item-ID-level hits for the operator.  The public
attestation carries only counts, policy constants, and hashes — never question
text, SQL text, or file contents.
"""
from __future__ import annotations

import ast as python_ast
import difflib
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from dki.analysis.nl2sql_protocol import normalise_question, validate_gold_sql


CONTAMINATION_AUDIT_VERSION = "nl2sql-contamination-v1"

# Published near-duplicate contract (mirrors scripts/freeze_nl2sql_blind.py).
NEAR_DUPLICATE_TRIGRAM_JACCARD = 0.85
NEAR_DUPLICATE_SEQUENCE_RATIO = 0.90

# History-only asset markers that must never appear in the runtime closure.
FORBIDDEN_RUNTIME_FRAGMENTS: tuple[str, ...] = (
    "_v28_local",
    "outputs/private",
    "outputs/remote_runs",
    "plans",
    "transcripts",
    "wrong_answers",
    "holdout",
)

DISALLOWED_CATEGORIES: tuple[str, ...] = (
    "exact_question",
    "near_question",
    "sql_exact",
    "ast_skeleton",
    "semantic_family",
)
ADVISORY_CATEGORIES: tuple[str, ...] = ("join_graph",)

_REQUIRED_ITEM_FIELDS = ("id", "question", "gold_sql")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_report_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sql_ast_skeleton(sql: str) -> str:
    """Literal-anonymized AST skeleton under the published freeze contract."""

    try:
        from sqlglot import exp, parse_one
    except ImportError as exc:  # pragma: no cover - dependency is mandatory
        raise RuntimeError(
            "sqlglot is required for AST-skeleton contamination checks; "
            "regex SQL grouping is forbidden"
        ) from exc
    try:
        tree = parse_one(sql, read="sqlite")
    except Exception as exc:
        raise ValueError(f"gold_sql is not valid SQLite SQL: {exc}") from exc

    def anonymize(node):
        if isinstance(node, exp.Literal):
            return exp.Literal.string("__LITERAL__")
        return node

    normalized = tree.transform(anonymize).sql(dialect="sqlite")
    return re.sub(r"\s+", " ", normalized).strip().upper()


def join_graph_fingerprint(sql: str) -> str:
    """Order-insensitive fingerprint of the tables referenced by one query."""

    from sqlglot import exp, parse_one

    tree = parse_one(sql, read="sqlite")
    tables = sorted(
        {
            str(node.name).casefold()
            for node in tree.find_all(exp.Table)
            if node.name
        }
    )
    return _sha256_text("\0".join(tables))


def question_trigrams(question: str) -> frozenset[str]:
    if len(question) < 3:
        return frozenset({question}) if question else frozenset()
    return frozenset(
        question[index : index + 3] for index in range(len(question) - 2)
    )


def questions_are_near_duplicates(left: str, right: str) -> bool:
    """Published deterministic near-duplicate contract (trigram + ratio)."""

    left_grams = question_trigrams(left)
    right_grams = question_trigrams(right)
    union = len(left_grams | right_grams)
    jaccard = len(left_grams & right_grams) / union if union else 1.0
    if jaccard >= NEAR_DUPLICATE_TRIGRAM_JACCARD:
        return True
    return (
        difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()
        >= NEAR_DUPLICATE_SEQUENCE_RATIO
    )


@dataclass(frozen=True)
class ItemFingerprint:
    """Typed per-item contamination fingerprints (text kept private)."""

    item_id: str
    normalized_question: str
    question_sha256: str
    trigrams: frozenset[str]
    sql_sha256: str
    ast_skeleton_sha256: str
    join_graph_sha256: str


def fingerprint_item(item: Mapping[str, Any]) -> ItemFingerprint:
    """Fail-closed fingerprinting for one {id, question, gold_sql} record."""

    if not isinstance(item, Mapping):
        raise TypeError("contamination item must be a mapping")
    missing = [key for key in _REQUIRED_ITEM_FIELDS if key not in item]
    if missing:
        raise ValueError(f"contamination item is missing fields: {missing}")
    item_id = item["id"]
    question = item["question"]
    gold_sql = item["gold_sql"]
    if not isinstance(item_id, str) or not item_id.strip():
        raise ValueError("contamination item id must be a non-empty string")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("contamination item question must be a non-empty string")
    if not isinstance(gold_sql, str) or not gold_sql.strip():
        raise ValueError("contamination item gold_sql must be a non-empty string")
    validate_gold_sql(gold_sql)
    normalized = unicodedata.normalize("NFKC", normalise_question(question))
    normalized_sql = re.sub(r"\s+", " ", gold_sql).strip().rstrip(";").upper()
    return ItemFingerprint(
        item_id=item_id,
        normalized_question=normalized,
        question_sha256=_sha256_text(normalized),
        trigrams=question_trigrams(normalized),
        sql_sha256=_sha256_text(normalized_sql),
        ast_skeleton_sha256=_sha256_text(sql_ast_skeleton(gold_sql)),
        join_graph_sha256=join_graph_fingerprint(gold_sql),
    )


class _UnionFind:
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


def _transitive_family_overlap(
    candidates: Sequence[ItemFingerprint],
    historical: Sequence[ItemFingerprint],
) -> list[tuple[str, str]]:
    """Candidate/historical pairs joined by transitive AST + near-dup links."""

    combined: list[ItemFingerprint] = list(candidates) + list(historical)
    families = _UnionFind(len(combined))
    by_skeleton: dict[str, list[int]] = defaultdict(list)
    for index, print_ in enumerate(combined):
        by_skeleton[print_.ast_skeleton_sha256].append(index)
    for indices in by_skeleton.values():
        for index in indices[1:]:
            families.union(indices[0], index)
    for left in range(len(combined)):
        for right in range(left + 1, len(combined)):
            if questions_are_near_duplicates(
                combined[left].normalized_question,
                combined[right].normalized_question,
            ):
                families.union(left, right)
    boundary = len(candidates)
    members: dict[int, dict[str, list[str]]] = defaultdict(
        lambda: {"candidates": [], "historical": []}
    )
    for index in range(len(combined)):
        root = families.find(index)
        side = "candidates" if index < boundary else "historical"
        members[root][side].append(combined[index].item_id)
    hits: list[tuple[str, str]] = []
    for group in members.values():
        if group["candidates"] and group["historical"]:
            for candidate_id in group["candidates"]:
                for historical_id in group["historical"]:
                    hits.append((candidate_id, historical_id))
    return sorted(set(hits))


@dataclass(frozen=True)
class ContaminationReport:
    """Private hits plus a text-free public attestation payload."""

    version: str
    clean: bool
    hit_counts: Mapping[str, int]
    private_hits: Mapping[str, tuple[tuple[str, str], ...]]
    n_candidates: int
    n_historical: int
    policy: Mapping[str, Any] = field(default_factory=dict)

    def public_attestation(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": "nl2sql_contamination_public_attestation",
            "version": self.version,
            "clean": self.clean,
            "n_candidates": self.n_candidates,
            "n_historical": self.n_historical,
            "disallowed_categories": list(DISALLOWED_CATEGORIES),
            "advisory_categories": list(ADVISORY_CATEGORIES),
            "hit_counts": dict(sorted(self.hit_counts.items())),
            "policy": dict(self.policy),
        }
        payload["report_sha256"] = canonical_report_sha256(payload)
        return payload

    def private_report(self) -> dict[str, Any]:
        return {
            "kind": "nl2sql_contamination_private_report",
            "version": self.version,
            "clean": self.clean,
            "hit_counts": dict(sorted(self.hit_counts.items())),
            "hits": {
                category: [list(pair) for pair in pairs]
                for category, pairs in sorted(self.private_hits.items())
            },
            "n_candidates": self.n_candidates,
            "n_historical": self.n_historical,
            "policy": dict(self.policy),
        }


def audit_contamination(
    candidate_items: Sequence[Mapping[str, Any]],
    historical_items: Sequence[Mapping[str, Any]],
) -> ContaminationReport:
    """Fail-closed disjointness audit of candidates against history.

    Any overlap in a disallowed category makes the result not clean.  Empty
    inputs are refused: an empty candidate set means the caller audited
    nothing, and an empty historical set means history was not loaded.
    """

    if not candidate_items:
        raise ValueError("candidate_items must not be empty (fail-closed)")
    if not historical_items:
        raise ValueError("historical_items must not be empty (fail-closed)")
    candidates = [fingerprint_item(item) for item in candidate_items]
    historical = [fingerprint_item(item) for item in historical_items]
    for label, prints in (("candidate", candidates), ("historical", historical)):
        ids = [print_.item_id for print_ in prints]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate {label} item IDs are forbidden")

    hits: dict[str, list[tuple[str, str]]] = {
        category: [] for category in DISALLOWED_CATEGORIES + ADVISORY_CATEGORIES
    }
    for candidate in candidates:
        for other in historical:
            if candidate.question_sha256 == other.question_sha256:
                hits["exact_question"].append((candidate.item_id, other.item_id))
            elif questions_are_near_duplicates(
                candidate.normalized_question,
                other.normalized_question,
            ):
                hits["near_question"].append((candidate.item_id, other.item_id))
            if candidate.sql_sha256 == other.sql_sha256:
                hits["sql_exact"].append((candidate.item_id, other.item_id))
            if candidate.ast_skeleton_sha256 == other.ast_skeleton_sha256:
                hits["ast_skeleton"].append((candidate.item_id, other.item_id))
            if candidate.join_graph_sha256 == other.join_graph_sha256:
                hits["join_graph"].append((candidate.item_id, other.item_id))
    hits["semantic_family"] = _transitive_family_overlap(candidates, historical)

    hit_counts = {category: len(pairs) for category, pairs in hits.items()}
    clean = all(hit_counts[category] == 0 for category in DISALLOWED_CATEGORIES)
    policy = {
        "near_duplicate_trigram_jaccard": NEAR_DUPLICATE_TRIGRAM_JACCARD,
        "near_duplicate_sequence_ratio": NEAR_DUPLICATE_SEQUENCE_RATIO,
    }
    return ContaminationReport(
        version=CONTAMINATION_AUDIT_VERSION,
        clean=clean,
        hit_counts=hit_counts,
        private_hits={
            category: tuple(sorted(set(pairs)))
            for category, pairs in hits.items()
        },
        n_candidates=len(candidates),
        n_historical=len(historical),
        policy=policy,
    )


# --------------------------------------------------------------------------
# Runtime import-closure scanner
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ClosureFinding:
    module_path: str
    line_number: int
    fragment: str


@dataclass(frozen=True)
class ClosureReport:
    version: str
    clean: bool
    n_modules: int
    findings: tuple[ClosureFinding, ...]

    def public_summary(self) -> dict[str, Any]:
        payload = {
            "kind": "nl2sql_runtime_closure_public_summary",
            "version": self.version,
            "clean": self.clean,
            "n_modules": self.n_modules,
            "n_findings": len(self.findings),
            "forbidden_fragments": list(FORBIDDEN_RUNTIME_FRAGMENTS),
        }
        payload["report_sha256"] = canonical_report_sha256(payload)
        return payload


def _module_imports(source: str, module_path: str) -> set[str]:
    tree = python_ast.parse(source, filename=module_path)
    names: set[str] = set()
    for node in python_ast.walk(tree):
        if isinstance(node, python_ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, python_ast.ImportFrom) and node.module:
            if node.level == 0:
                names.add(node.module)
                # ``from dki.analysis import helper`` also pulls in the
                # submodule ``dki.analysis.helper`` when that file exists.
                for alias in node.names:
                    names.add(f"{node.module}.{alias.name}")
    return names


def _fragment_matches_literal(fragment: str, literal: str) -> bool:
    """Match a forbidden fragment only as a path segment, never as prose.

    A docstring saying "model plans" must not fire; ``"plans/x.json"``,
    ``"x/plans"``, ``"_v28_local/history.json"`` and the bare directory name
    ``"plans"`` all must.
    """

    lowered = literal.casefold().replace("\\", "/")
    pattern = "(^|/)" + re.escape(fragment.casefold()) + "(/|$)"
    return re.search(pattern, lowered) is not None


def _string_literal_lines(source: str, module_path: str) -> list[tuple[int, str]]:
    tree = python_ast.parse(source, filename=module_path)
    literals: list[tuple[int, str]] = []
    for node in python_ast.walk(tree):
        if isinstance(node, python_ast.Constant) and isinstance(node.value, str):
            literals.append((node.lineno, node.value))
    return literals


def scan_runtime_closure(
    repo_root: Path,
    *,
    entry_module: str = "dki.analysis.nl2sql",
    extra_paths: Sequence[Path] = (),
    forbidden_fragments: Sequence[str] = FORBIDDEN_RUNTIME_FRAGMENTS,
) -> ClosureReport:
    """Walk the static ``dki.*`` import closure and flag history-only refs.

    Only string literals are inspected, so a docstring mentioning "plans" in
    prose never fires; a code path embedding ``_v28_local`` or a history
    directory does.
    """

    root = Path(repo_root)
    if not (root / "dki").is_dir():
        raise FileNotFoundError(f"repo root has no dki package: {root}")

    def module_file(module: str) -> Path | None:
        relative = Path(*module.split("."))
        for suffix in (relative.with_suffix(".py"), relative / "__init__.py"):
            path = root / suffix
            if path.is_file():
                return path
        return None

    pending = [entry_module]
    visited: dict[str, Path] = {}
    while pending:
        module = pending.pop()
        if module in visited or not module.startswith("dki"):
            continue
        path = module_file(module)
        if path is None:
            continue
        visited[module] = path
        source = path.read_text(encoding="utf-8")
        for imported in sorted(_module_imports(source, str(path))):
            if imported.startswith("dki"):
                pending.append(imported)

    findings: list[ClosureFinding] = []
    scan_targets = [
        (module, path) for module, path in sorted(visited.items())
    ] + [(str(path), Path(path)) for path in extra_paths]
    for label, path in scan_targets:
        source = Path(path).read_text(encoding="utf-8")
        for line_number, literal in _string_literal_lines(source, str(path)):
            for fragment in forbidden_fragments:
                if _fragment_matches_literal(fragment, literal):
                    findings.append(
                        ClosureFinding(
                            module_path=label,
                            line_number=line_number,
                            fragment=fragment,
                        )
                    )
    findings_tuple = tuple(
        sorted(
            findings,
            key=lambda item: (item.module_path, item.line_number, item.fragment),
        )
    )
    return ClosureReport(
        version=CONTAMINATION_AUDIT_VERSION,
        clean=not findings_tuple,
        n_modules=len(visited),
        findings=findings_tuple,
    )


__all__ = [
    "ADVISORY_CATEGORIES",
    "CONTAMINATION_AUDIT_VERSION",
    "DISALLOWED_CATEGORIES",
    "FORBIDDEN_RUNTIME_FRAGMENTS",
    "NEAR_DUPLICATE_SEQUENCE_RATIO",
    "NEAR_DUPLICATE_TRIGRAM_JACCARD",
    "ClosureFinding",
    "ClosureReport",
    "ContaminationReport",
    "ItemFingerprint",
    "audit_contamination",
    "canonical_report_sha256",
    "fingerprint_item",
    "join_graph_fingerprint",
    "question_trigrams",
    "questions_are_near_duplicates",
    "scan_runtime_closure",
    "sql_ast_skeleton",
]
