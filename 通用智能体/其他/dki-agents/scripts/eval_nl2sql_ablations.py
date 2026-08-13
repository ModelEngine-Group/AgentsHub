"""Evaluate manifest-bound, precomputed NL2SQL ablation JSONL results.

The runner never imports or invokes the NL2SQL engine.  It accepts strict JSONL
outcomes plus one immutable manifest per arm.  Manifests bind the rule registry,
effective policy, profile identity, source tree, dataset/family/run/database
profiles, expected populations, target ablation, and canonical result-row hash.
No acceptance decision is produced from unbound or mixed provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.analysis.nl2sql_ablation import (  # noqa: E402
    KNOWN_ERROR_CATEGORIES,
    AblationAcceptanceCriteria,
    DEFAULT_ABLATION_ACCEPTANCE_CRITERIA,
    NL2SQLAblationConfig,
    PairedStrictComparison,
    RouteProvenance,
    RuleRegistrySnapshot,
    assess_ablation_acceptance,
    canonical_sha256,
    compare_paired_correctness,
    compare_stability_populations,
    validate_rule_cluster_id,
    validate_rule_id,
    validate_sha256,
    validate_stable_id,
)


REPORT_SCHEMA_VERSION = 2
ARM_MANIFEST_SCHEMA_VERSION = 1
DEFAULT_DB_ID = "base"
DEFAULT_RUN_ID = "primary"
UNSPECIFIED = "unspecified"
_ERROR_CATEGORY_SET = frozenset(KNOWN_ERROR_CATEGORIES)
_ITEM_ID_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,512}$")

ROW_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "route",
        "registry_hash",
        "policy_hash",
        "ablation_config_hash",
        "profile_hash",
        "source_hash",
        "rule_id",
        "selected_rule_id",
        "cluster_id",
        "rule_cluster",
        "selected_rule_cluster",
        "reason_codes",
        "decisions",
        "rule_decisions",
    }
)
ROW_FIELDS = frozenset(
    {
        "id",
        "correct",
        "dataset_role",
        "slice_id",
        "shadow_id",
        "db_id",
        "database_id",
        "run_id",
        "difficulty",
        "category",
        "family_id",
        "family",
        "capabilities",
        "capability",
        "engine",
        "error_category",
        "plan",
        "provenance",
        *ROW_PROVENANCE_FIELDS,
    }
)
MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "arm_id",
        "arm_role",
        "registry",
        "registry_hash",
        "config",
        "policy_hash",
        "profile_hash",
        "target_disabled_clusters",
        "target_disabled_rule_ids",
        "source_hash",
        "dataset_manifest_hash",
        "family_manifest_hash",
        "run_manifest_hash",
        "database_manifest_hash",
        "selection_role",
        "shadow_role",
        "base_db_id",
        "primary_run_id",
        "database_ids",
        "family_ids",
        "run_ids",
        "shadow_groups",
        "result_rows_hash",
    }
)


class _DuplicateJSONKey(ValueError):
    pass


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Any:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _strict_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON value is forbidden: {value}")
    return parsed


def strict_json_loads(text: str, *, source: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
            parse_float=_strict_float,
        )
    except (json.JSONDecodeError, _DuplicateJSONKey, ValueError) as exc:
        raise ValueError(f"{source}: invalid strict JSON: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _item_id(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not _ITEM_ID_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a nonempty string of at most 512 characters")
    return value


def _stable_label(value: Any, *, field_name: str) -> str:
    return validate_stable_id(value, field_name=field_name)


def _alias_from_mapping(
    payload: Mapping[str, Any],
    aliases: Sequence[str],
    *,
    field_name: str,
    required: bool = False,
) -> Any:
    present = [(name, payload[name]) for name in aliases if name in payload]
    if not present:
        if required:
            raise ValueError(f"missing {field_name}")
        return None
    first = present[0][1]
    if any(value != first for _, value in present[1:]):
        raise ValueError(f"conflicting aliases for {field_name}: {present!r}")
    return first


def _nested_mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    if name not in payload:
        return None
    value = payload[name]
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    unknown = set(value) - ROW_PROVENANCE_FIELDS
    if unknown:
        raise ValueError(f"{name} has unknown fields: {sorted(unknown)!r}")
    return value


def _cross_level_alias(
    payload: Mapping[str, Any],
    aliases: Sequence[str],
    *,
    field_name: str,
    required: bool = False,
) -> Any:
    values: list[tuple[str, Any]] = []
    containers = [("row", payload)]
    for name in ("plan", "provenance"):
        nested = _nested_mapping(payload, name)
        if nested is not None:
            containers.append((name, nested))
    for container_name, container in containers:
        local = [(alias, container[alias]) for alias in aliases if alias in container]
        if local:
            first = local[0][1]
            if any(value != first for _, value in local[1:]):
                raise ValueError(
                    f"conflicting {field_name} aliases in {container_name}: {local!r}"
                )
            values.append((container_name, first))
    if not values:
        if required:
            raise ValueError(f"missing {field_name}")
        return None
    first = values[0][1]
    if any(value != first for _, value in values[1:]):
        raise ValueError(f"conflicting {field_name} values across levels: {values!r}")
    return first


def _capability_id(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("capability names must be strings")
    normalised = re.sub(r"[\s\-/]+", "_", value.strip().casefold())
    normalised = re.sub(r"_+", "_", normalised).strip("_")
    return validate_stable_id(normalised, field_name="capability")


def _capabilities(payload: Mapping[str, Any]) -> tuple[str, ...]:
    raw = _alias_from_mapping(
        payload,
        ("capabilities", "capability"),
        field_name="capabilities",
    )
    if raw is None:
        return ()
    if isinstance(raw, str):
        values = [part for part in re.split(r"[,;|]+", raw) if part.strip()]
    elif isinstance(raw, Mapping):
        if any(not isinstance(enabled, bool) for enabled in raw.values()):
            raise TypeError("capability mapping values must be booleans")
        values = [key for key, enabled in raw.items() if enabled]
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = list(raw)
    else:
        raise TypeError("capabilities must be a string, sequence, or bool mapping")
    return tuple(sorted({_capability_id(value) for value in values}))


def public_label_hash(namespace: str, value: str) -> str:
    """Hash bounded stable labels so report keys cannot leak their raw content."""

    validate_stable_id(namespace, field_name="label_namespace")
    if (
        not isinstance(value, str)
        or len(value) > 256
        or not re.fullmatch(r"[a-z0-9][a-z0-9._:-]*", value)
    ):
        raise ValueError(
            "public label values must be lowercase stable identifiers of at most 256 characters"
        )
    material = namespace + "\0" + value
    return f"sha256:{hashlib.sha256(material.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True)
class PrecomputedResult:
    item_id: str
    correct: bool
    dataset_role: str
    slice_id: str
    db_id: str
    run_id: str
    difficulty: str
    category: str
    family_id: str
    capabilities: tuple[str, ...]
    engine: str
    provenance: RouteProvenance
    error_category: str | None = None

    def __post_init__(self) -> None:
        _item_id(self.item_id, field_name="item_id")
        if not isinstance(self.correct, bool):
            raise TypeError("correct must be a JSON boolean")
        for name in (
            "dataset_role",
            "slice_id",
            "db_id",
            "run_id",
            "difficulty",
            "category",
            "family_id",
            "engine",
        ):
            _stable_label(getattr(self, name), field_name=name)
        if not isinstance(self.capabilities, tuple):
            raise TypeError("capabilities must be a tuple")
        for capability in self.capabilities:
            _capability_id(capability)
        if not isinstance(self.provenance, RouteProvenance):
            raise TypeError("provenance must be RouteProvenance")
        if self.error_category is not None:
            if self.error_category not in _ERROR_CATEGORY_SET:
                raise ValueError(
                    f"unknown error_category {self.error_category!r}; "
                    f"known={KNOWN_ERROR_CATEGORIES!r}"
                )
            if self.correct:
                raise ValueError("a correct row cannot have an error_category")

    @property
    def identity(self) -> tuple[str, str, str, str, str]:
        return (
            self.dataset_role,
            self.slice_id,
            self.db_id,
            self.run_id,
            self.item_id,
        )

    @property
    def invariant_metadata(self) -> tuple[Any, ...]:
        return (
            self.difficulty,
            self.category,
            self.family_id,
            self.capabilities,
        )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "id": self.item_id,
            "correct": self.correct,
            "dataset_role": self.dataset_role,
            "slice_id": self.slice_id,
            "db_id": self.db_id,
            "run_id": self.run_id,
            "difficulty": self.difficulty,
            "category": self.category,
            "family_id": self.family_id,
            "capabilities": list(self.capabilities),
            "engine": self.engine,
            "error_category": self.error_category,
            "provenance": self.provenance.as_dict(),
        }


def parse_precomputed_result(
    payload: Mapping[str, Any],
    *,
    registry: RuleRegistrySnapshot,
    source: str = "row",
) -> PrecomputedResult:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{source} must be a JSON object")
    unknown = set(payload) - ROW_FIELDS
    if unknown:
        raise ValueError(f"{source} has unknown fields: {sorted(unknown)!r}")
    item_id = _item_id(payload.get("id"), field_name=f"{source}.id")
    if "correct" not in payload or not isinstance(payload["correct"], bool):
        raise TypeError(f"{source}.correct must be a JSON boolean")

    family = _alias_from_mapping(
        payload,
        ("family_id", "family"),
        field_name="family_id",
        required=True,
    )
    slice_id = _alias_from_mapping(
        payload,
        ("slice_id", "shadow_id"),
        field_name="slice_id",
        required=True,
    )
    db_id = _alias_from_mapping(
        payload,
        ("db_id", "database_id"),
        field_name="db_id",
        required=True,
    )

    schema_version = _cross_level_alias(
        payload,
        ("schema_version",),
        field_name="provenance.schema_version",
    )
    rule_id = _cross_level_alias(
        payload,
        ("rule_id", "selected_rule_id"),
        field_name="rule_id",
    )
    cluster_id = _cross_level_alias(
        payload,
        ("cluster_id", "rule_cluster", "selected_rule_cluster"),
        field_name="cluster_id",
    )
    reason_codes = _cross_level_alias(
        payload,
        ("reason_codes",),
        field_name="reason_codes",
    )
    decisions = _cross_level_alias(
        payload,
        ("decisions", "rule_decisions"),
        field_name="decisions",
    )
    route_payload = {
        "route": _cross_level_alias(
            payload,
            ("route",),
            field_name="route",
            required=True,
        ),
        "registry_hash": _cross_level_alias(
            payload,
            ("registry_hash",),
            field_name="registry_hash",
            required=True,
        ),
        "policy_hash": _cross_level_alias(
            payload,
            ("policy_hash", "ablation_config_hash"),
            field_name="policy_hash",
            required=True,
        ),
        "profile_hash": _cross_level_alias(
            payload,
            ("profile_hash",),
            field_name="profile_hash",
            required=True,
        ),
        "source_hash": _cross_level_alias(
            payload,
            ("source_hash",),
            field_name="source_hash",
            required=True,
        ),
    }
    for field_name, value in (
        ("schema_version", schema_version),
        ("rule_id", rule_id),
        ("cluster_id", cluster_id),
        ("reason_codes", reason_codes),
        ("decisions", decisions),
    ):
        if value is not None:
            route_payload[field_name] = value
    provenance = RouteProvenance.from_dict(route_payload, registry=registry)
    error_category = payload.get("error_category")
    if error_category is not None:
        error_category = validate_stable_id(
            error_category,
            field_name="error_category",
        )

    return PrecomputedResult(
        item_id=item_id,
        correct=payload["correct"],
        dataset_role=_stable_label(
            payload.get("dataset_role"),
            field_name="dataset_role",
        ),
        slice_id=_stable_label(slice_id, field_name="slice_id"),
        db_id=_stable_label(db_id, field_name="db_id"),
        run_id=_stable_label(payload.get("run_id"), field_name="run_id"),
        difficulty=_stable_label(payload.get("difficulty"), field_name="difficulty"),
        category=_stable_label(payload.get("category"), field_name="category"),
        family_id=_stable_label(family, field_name="family_id"),
        capabilities=_capabilities(payload),
        engine=_stable_label(payload.get("engine"), field_name="engine"),
        provenance=provenance,
        error_category=error_category,
    )


@dataclass(frozen=True)
class LoadedResultSet:
    file_sha256: str
    rows: tuple[PrecomputedResult, ...]

    def __post_init__(self) -> None:
        validate_sha256(self.file_sha256, field_name="file_sha256")
        if not isinstance(self.rows, tuple):
            object.__setattr__(self, "rows", tuple(self.rows))
        if not self.rows:
            raise ValueError("result set must not be empty")
        seen: set[tuple[str, str, str, str, str]] = set()
        for row in self.rows:
            if not isinstance(row, PrecomputedResult):
                raise TypeError("result rows must be PrecomputedResult values")
            if row.identity in seen:
                raise ValueError(f"duplicate result identity: {row.identity!r}")
            seen.add(row.identity)
        object.__setattr__(
            self,
            "rows",
            tuple(sorted(self.rows, key=lambda row: row.identity)),
        )

    @property
    def canonical_rows_sha256(self) -> str:
        return canonical_rows_sha256(self.rows)

    def source_summary(self) -> dict[str, Any]:
        return {
            "file_sha256": self.file_sha256,
            "canonical_rows_sha256": self.canonical_rows_sha256,
            "n_rows": len(self.rows),
        }


def load_precomputed_results(
    path: Path,
    *,
    registry: RuleRegistrySnapshot,
) -> LoadedResultSet:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"result JSONL does not exist: {path}")
    rows: list[PrecomputedResult] = []
    seen: dict[tuple[str, str, str, str, str], int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = strict_json_loads(line, source=f"{path}:{line_number}")
            try:
                row = parse_precomputed_result(
                    payload,
                    registry=registry,
                    source=f"{path}:{line_number}",
                )
            except (TypeError, ValueError) as exc:
                raise type(exc)(f"{path}:{line_number}: {exc}") from exc
            if row.identity in seen:
                raise ValueError(
                    f"{path}:{line_number}: duplicate result identity; "
                    f"first seen on line {seen[row.identity]}: {row.identity!r}"
                )
            seen[row.identity] = line_number
            rows.append(row)
    if not rows:
        raise ValueError(f"result JSONL is empty: {path}")
    rows.sort(key=lambda row: row.identity)
    return LoadedResultSet(_sha256_file(path), tuple(rows))


def canonical_rows_sha256(rows: Sequence[PrecomputedResult]) -> str:
    return canonical_sha256(
        {
            "rows": [
                row.canonical_payload()
                for row in sorted(rows, key=lambda item: item.identity)
            ]
        }
    )


def _unique_stable_ids(values: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence")
    result: set[str] = set()
    for value in values:
        identifier = validate_stable_id(value, field_name=field_name)
        if identifier in result:
            raise ValueError(f"duplicate {field_name}: {identifier}")
        result.add(identifier)
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    return tuple(sorted(result))


def _cluster_ids(values: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence")
    result: set[str] = set()
    for value in values:
        cluster = validate_rule_cluster_id(value)
        if cluster in result:
            raise ValueError(f"duplicate {field_name}: {cluster}")
        result.add(cluster)
    return tuple(sorted(result))


def _rule_ids(
    values: Any,
    *,
    field_name: str,
    registry: RuleRegistrySnapshot,
) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence")
    result: set[str] = set()
    for value in values:
        rule_id = validate_rule_id(value)
        registry.cluster_for(rule_id)
        if rule_id in result:
            raise ValueError(f"duplicate {field_name}: {rule_id}")
        result.add(rule_id)
    return tuple(sorted(result))


def _shadow_groups(values: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("shadow_groups must be a sequence")
    result: set[tuple[str, str]] = set()
    for index, value in enumerate(values):
        if isinstance(value, Mapping):
            unknown = set(value) - {"slice_id", "family_id"}
            if unknown or set(value) != {"slice_id", "family_id"}:
                raise ValueError(
                    f"shadow_groups[{index}] must contain only slice_id and family_id"
                )
            raw_slice = value["slice_id"]
            raw_family = value["family_id"]
        elif (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and len(value) == 2
        ):
            raw_slice, raw_family = value
        else:
            raise TypeError(
                f"shadow_groups[{index}] must be an object or a two-item sequence"
            )
        group = (
            validate_stable_id(raw_slice, field_name="shadow.slice_id"),
            validate_stable_id(raw_family, field_name="shadow.family_id"),
        )
        if group in result:
            raise ValueError(f"duplicate shadow group: {group!r}")
        result.add(group)
    if not result:
        raise ValueError("shadow_groups must not be empty")
    return tuple(sorted(result))


@dataclass(frozen=True)
class AblationArmManifest:
    arm_id: str
    arm_role: str
    registry: RuleRegistrySnapshot
    config: NL2SQLAblationConfig
    target_disabled_clusters: tuple[str, ...]
    target_disabled_rule_ids: tuple[str, ...]
    source_hash: str
    dataset_manifest_hash: str
    family_manifest_hash: str
    run_manifest_hash: str
    database_manifest_hash: str
    selection_role: str
    shadow_role: str
    base_db_id: str
    primary_run_id: str
    database_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    run_ids: tuple[str, ...]
    shadow_groups: tuple[tuple[str, str], ...]
    result_rows_hash: str

    def __post_init__(self) -> None:
        validate_stable_id(self.arm_id, field_name="arm_id")
        if self.arm_role not in {"baseline", "variant"}:
            raise ValueError("arm_role must be baseline or variant")
        if not isinstance(self.registry, RuleRegistrySnapshot):
            raise TypeError("registry must be RuleRegistrySnapshot")
        if not isinstance(self.config, NL2SQLAblationConfig):
            raise TypeError("config must be NL2SQLAblationConfig")
        if self.config.registry.registry_hash != self.registry.registry_hash:
            raise ValueError("config registry does not match manifest registry")
        object.__setattr__(
            self,
            "target_disabled_clusters",
            _cluster_ids(
                self.target_disabled_clusters,
                field_name="target_disabled_clusters",
            ),
        )
        object.__setattr__(
            self,
            "target_disabled_rule_ids",
            _rule_ids(
                self.target_disabled_rule_ids,
                field_name="target_disabled_rule_ids",
                registry=self.registry,
            ),
        )
        object.__setattr__(
            self,
            "database_ids",
            _unique_stable_ids(self.database_ids, field_name="database_ids"),
        )
        object.__setattr__(
            self,
            "family_ids",
            _unique_stable_ids(self.family_ids, field_name="family_ids"),
        )
        object.__setattr__(
            self,
            "run_ids",
            _unique_stable_ids(self.run_ids, field_name="run_ids"),
        )
        object.__setattr__(self, "shadow_groups", _shadow_groups(self.shadow_groups))
        for field_name in (
            "source_hash",
            "dataset_manifest_hash",
            "family_manifest_hash",
            "run_manifest_hash",
            "database_manifest_hash",
            "result_rows_hash",
        ):
            validate_sha256(getattr(self, field_name), field_name=field_name)
        for field_name in (
            "selection_role",
            "shadow_role",
            "base_db_id",
            "primary_run_id",
        ):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        if self.selection_role == self.shadow_role:
            raise ValueError("selection_role and shadow_role must differ")
        if self.base_db_id not in self.database_ids:
            raise ValueError("base_db_id is absent from database_ids")
        if self.primary_run_id not in self.run_ids:
            raise ValueError("primary_run_id is absent from run_ids")
        if not set(family for _, family in self.shadow_groups) <= set(self.family_ids):
            raise ValueError("shadow_groups contain a family absent from family_ids")
        for rule_id in self.target_disabled_rule_ids:
            self.registry.cluster_for(rule_id)
        if not set(self.target_disabled_clusters) <= set(self.config.disabled_clusters):
            raise ValueError("target_disabled_clusters are not disabled by the config")
        if not set(self.target_disabled_rule_ids) <= set(self.config.disabled_rule_ids):
            raise ValueError("target_disabled_rule_ids are not disabled by the config")
        if self.arm_role == "baseline" and (
            self.target_disabled_clusters or self.target_disabled_rule_ids
        ):
            raise ValueError("baseline manifest cannot declare a target ablation")
        if self.arm_role == "variant" and not (
            self.target_disabled_clusters or self.target_disabled_rule_ids
        ):
            raise ValueError("variant manifest must declare a target ablation")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "AblationArmManifest":
        if not isinstance(payload, Mapping):
            raise TypeError("arm manifest must be an object")
        unknown = set(payload) - MANIFEST_FIELDS
        if unknown:
            raise ValueError(f"arm manifest has unknown fields: {sorted(unknown)!r}")
        missing = MANIFEST_FIELDS - set(payload)
        if missing:
            raise ValueError(f"arm manifest is missing fields: {sorted(missing)!r}")
        if payload["schema_version"] != ARM_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported arm manifest schema_version: {payload['schema_version']!r}"
            )
        registry = RuleRegistrySnapshot.from_payload(payload["registry"])
        supplied_registry_hash = validate_sha256(
            payload["registry_hash"],
            field_name="registry_hash",
        )
        if supplied_registry_hash != registry.registry_hash:
            raise ValueError("registry_hash does not match registry snapshot")
        config = NL2SQLAblationConfig.from_manifest_payload(
            payload["config"],
            registry=registry,
        )
        if validate_sha256(payload["policy_hash"], field_name="policy_hash") != config.policy_hash:
            raise ValueError("policy_hash does not match effective config")
        if validate_sha256(payload["profile_hash"], field_name="profile_hash") != config.profile_hash:
            raise ValueError("profile_hash does not match config profile")
        return cls(
            arm_id=payload["arm_id"],
            arm_role=payload["arm_role"],
            registry=registry,
            config=config,
            target_disabled_clusters=_cluster_ids(
                payload["target_disabled_clusters"],
                field_name="target_disabled_clusters",
            ),
            target_disabled_rule_ids=_rule_ids(
                payload["target_disabled_rule_ids"],
                field_name="target_disabled_rule_ids",
                registry=registry,
            ),
            source_hash=payload["source_hash"],
            dataset_manifest_hash=payload["dataset_manifest_hash"],
            family_manifest_hash=payload["family_manifest_hash"],
            run_manifest_hash=payload["run_manifest_hash"],
            database_manifest_hash=payload["database_manifest_hash"],
            selection_role=payload["selection_role"],
            shadow_role=payload["shadow_role"],
            base_db_id=payload["base_db_id"],
            primary_run_id=payload["primary_run_id"],
            database_ids=_unique_stable_ids(
                payload["database_ids"],
                field_name="database_ids",
            ),
            family_ids=_unique_stable_ids(
                payload["family_ids"],
                field_name="family_ids",
            ),
            run_ids=_unique_stable_ids(payload["run_ids"], field_name="run_ids"),
            shadow_groups=_shadow_groups(payload["shadow_groups"]),
            result_rows_hash=payload["result_rows_hash"],
        )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": ARM_MANIFEST_SCHEMA_VERSION,
            "arm_id": self.arm_id,
            "arm_role": self.arm_role,
            "registry": self.registry.canonical_payload()["registrations"],
            "registry_hash": self.registry.registry_hash,
            "config": self.config.manifest_payload(),
            "policy_hash": self.config.policy_hash,
            "profile_hash": self.config.profile_hash,
            "target_disabled_clusters": list(self.target_disabled_clusters),
            "target_disabled_rule_ids": list(self.target_disabled_rule_ids),
            "source_hash": self.source_hash,
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "family_manifest_hash": self.family_manifest_hash,
            "run_manifest_hash": self.run_manifest_hash,
            "database_manifest_hash": self.database_manifest_hash,
            "selection_role": self.selection_role,
            "shadow_role": self.shadow_role,
            "base_db_id": self.base_db_id,
            "primary_run_id": self.primary_run_id,
            "database_ids": list(self.database_ids),
            "family_ids": list(self.family_ids),
            "run_ids": list(self.run_ids),
            "shadow_groups": [
                {"slice_id": slice_id, "family_id": family_id}
                for slice_id, family_id in self.shadow_groups
            ],
            "result_rows_hash": self.result_rows_hash,
        }

    @property
    def manifest_hash(self) -> str:
        return canonical_sha256(self.canonical_payload())

    def public_summary(self) -> dict[str, Any]:
        return {
            "arm_id_hash": public_label_hash("arm", self.arm_id),
            "arm_role": self.arm_role,
            "manifest_hash": self.manifest_hash,
            "registry_hash": self.registry.registry_hash,
            "policy_hash": self.config.policy_hash,
            "profile_hash": self.config.profile_hash,
            "source_hash": self.source_hash,
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "family_manifest_hash": self.family_manifest_hash,
            "run_manifest_hash": self.run_manifest_hash,
            "database_manifest_hash": self.database_manifest_hash,
            "result_rows_hash": self.result_rows_hash,
            "target_disabled_clusters": list(self.target_disabled_clusters),
            "target_disabled_rule_ids": list(self.target_disabled_rule_ids),
            "n_databases": len(self.database_ids),
            "n_families": len(self.family_ids),
            "n_runs": len(self.run_ids),
            "n_shadow_groups": len(self.shadow_groups),
        }


def load_arm_manifest(path: Path) -> AblationArmManifest:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"arm manifest does not exist: {path}")
    payload = strict_json_loads(path.read_text(encoding="utf-8"), source=str(path))
    return AblationArmManifest.from_payload(payload)


@dataclass(frozen=True)
class BoundArm:
    manifest: AblationArmManifest
    results: LoadedResultSet

    def source_summary(self) -> dict[str, Any]:
        return {
            **self.results.source_summary(),
            "manifest_hash": self.manifest.manifest_hash,
        }


def _population_by_run(
    rows: Sequence[PrecomputedResult],
    *,
    manifest: AblationArmManifest,
) -> dict[str, set[tuple[str, str, str]]]:
    result: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for row in rows:
        if row.dataset_role == manifest.selection_role and row.db_id == manifest.base_db_id:
            result[row.run_id].add((row.slice_id, row.family_id, row.item_id))
    return result


def bind_arm(
    manifest: AblationArmManifest,
    results: LoadedResultSet,
) -> BoundArm:
    if results.canonical_rows_sha256 != manifest.result_rows_hash:
        raise ValueError("result_rows_hash does not match canonical JSONL rows")
    for row in results.rows:
        provenance = row.provenance
        if provenance.registry.registry_hash != manifest.registry.registry_hash:
            raise ValueError("row registry provenance does not match arm manifest")
        if provenance.policy_hash != manifest.config.policy_hash:
            raise ValueError("row policy provenance does not match arm manifest")
        if provenance.profile_hash != manifest.config.profile_hash:
            raise ValueError("row profile provenance does not match arm manifest")
        if provenance.source_hash != manifest.source_hash:
            raise ValueError("row source provenance does not match arm manifest")
        if provenance.rule_id is not None:
            manifest.registry.validate_membership(
                provenance.rule_id,
                provenance.cluster_id or "",
            )
            if not manifest.config.is_enabled(
                provenance.cluster_id or "",
                provenance.rule_id,
            ):
                raise ValueError("row selected rule is disabled by the arm policy")
        for decision in provenance.decisions:
            enabled = manifest.config.is_enabled(
                decision.cluster_id,
                decision.rule_id,
            )
            if not enabled and decision.decision.value == "sql":
                raise ValueError("disabled rule cannot have an SQL decision")

    actual_databases = {row.db_id for row in results.rows}
    actual_families = {row.family_id for row in results.rows}
    actual_runs = {row.run_id for row in results.rows}
    if actual_databases != set(manifest.database_ids):
        raise ValueError("JSONL database population does not match database manifest")
    if actual_families != set(manifest.family_ids):
        raise ValueError("JSONL family population does not match family manifest")
    if actual_runs != set(manifest.run_ids):
        raise ValueError("JSONL run population does not match run manifest")

    selection_primary_base = {
        (row.slice_id, row.family_id, row.item_id)
        for row in results.rows
        if row.dataset_role == manifest.selection_role
        and row.db_id == manifest.base_db_id
        and row.run_id == manifest.primary_run_id
    }
    if not selection_primary_base:
        raise ValueError("manifest-bound primary selection population is empty")
    for db_id in manifest.database_ids:
        population = {
            (row.slice_id, row.family_id, row.item_id)
            for row in results.rows
            if row.dataset_role == manifest.selection_role
            and row.db_id == db_id
            and row.run_id == manifest.primary_run_id
        }
        if population != selection_primary_base:
            raise ValueError(
                f"alternate DB {db_id} does not contain the identical selection population"
            )

    populations_by_run = _population_by_run(results.rows, manifest=manifest)
    if set(populations_by_run) != set(manifest.run_ids):
        raise ValueError("selection stability evidence is missing a manifest-bound run")
    for run_id, population in populations_by_run.items():
        if population != selection_primary_base:
            raise ValueError(
                f"stability run {run_id} does not contain the identical family/item population"
            )

    actual_shadow_groups = {
        (row.slice_id, row.family_id)
        for row in results.rows
        if row.dataset_role == manifest.shadow_role
        and row.db_id == manifest.base_db_id
        and row.run_id == manifest.primary_run_id
    }
    if actual_shadow_groups != set(manifest.shadow_groups):
        raise ValueError("shadow slice+family groups do not match the family manifest")
    return BoundArm(manifest, results)


Pair = tuple[PrecomputedResult, PrecomputedResult]


def _validate_manifest_pair(
    baseline: AblationArmManifest,
    variant: AblationArmManifest,
) -> None:
    if baseline.arm_role != "baseline" or variant.arm_role != "variant":
        raise ValueError("comparison requires baseline and variant arm roles")
    shared_fields = (
        "source_hash",
        "dataset_manifest_hash",
        "family_manifest_hash",
        "run_manifest_hash",
        "database_manifest_hash",
        "selection_role",
        "shadow_role",
        "base_db_id",
        "primary_run_id",
        "database_ids",
        "family_ids",
        "run_ids",
        "shadow_groups",
    )
    for field_name in shared_fields:
        if getattr(baseline, field_name) != getattr(variant, field_name):
            raise ValueError(f"baseline/variant manifests differ in {field_name}")
    if baseline.registry.registry_hash != variant.registry.registry_hash:
        raise ValueError("baseline/variant registry snapshots differ")
    enabled_rule_ids = {
        registration.rule_id
        for registration in baseline.registry.registrations
        if baseline.config.is_enabled(
            registration.cluster_id,
            registration.rule_id,
        )
    }
    variant_enabled_rule_ids = {
        registration.rule_id
        for registration in variant.registry.registrations
        if variant.config.is_enabled(
            registration.cluster_id,
            registration.rule_id,
        )
    }
    if not variant_enabled_rule_ids <= enabled_rule_ids:
        raise ValueError("variant re-enables baseline-disabled rule behavior")
    newly_disabled_rule_ids = enabled_rule_ids - variant_enabled_rule_ids
    added_clusters = set(variant.config.disabled_clusters) - set(
        baseline.config.disabled_clusters
    )
    expected_target_rule_ids = {
        rule_id
        for rule_id in newly_disabled_rule_ids
        if variant.registry.cluster_for(rule_id) not in added_clusters
    }
    if added_clusters != set(variant.target_disabled_clusters):
        raise ValueError("variant target clusters do not match the effective policy delta")
    if expected_target_rule_ids != set(variant.target_disabled_rule_ids):
        raise ValueError("variant target rules do not match the effective policy delta")


def pair_bound_arms(baseline: BoundArm, variant: BoundArm) -> tuple[Pair, ...]:
    baseline = bind_arm(baseline.manifest, baseline.results)
    variant = bind_arm(variant.manifest, variant.results)
    _validate_manifest_pair(baseline.manifest, variant.manifest)
    baseline_by_key = {row.identity: row for row in baseline.results.rows}
    variant_by_key = {row.identity: row for row in variant.results.rows}
    if baseline_by_key.keys() != variant_by_key.keys():
        missing = sorted(baseline_by_key.keys() - variant_by_key.keys())
        extra = sorted(variant_by_key.keys() - baseline_by_key.keys())
        raise ValueError(
            f"baseline/variant result identities differ: missing={missing!r}, extra={extra!r}"
        )
    pairs: list[Pair] = []
    for key in sorted(baseline_by_key):
        before = baseline_by_key[key]
        after = variant_by_key[key]
        if before.invariant_metadata != after.invariant_metadata:
            raise ValueError(
                f"paired metadata differs for {key!r}: "
                f"baseline={before.invariant_metadata!r}, "
                f"variant={after.invariant_metadata!r}"
            )
        pairs.append((before, after))
    if not pairs:
        raise ValueError("paired result set is empty")
    return tuple(pairs)


def _identity_token(identity: Sequence[str]) -> str:
    return json.dumps(list(identity), ensure_ascii=False, separators=(",", ":"))


def comparison_for_pairs(pairs: Sequence[Pair]) -> PairedStrictComparison:
    if not pairs:
        raise ValueError("cannot compare an empty pair group")
    return compare_paired_correctness(
        {_identity_token(before.identity): before.correct for before, _ in pairs},
        {_identity_token(after.identity): after.correct for _, after in pairs},
    )


def _public_group_comparisons(
    pairs: Sequence[Pair],
    *,
    namespace: str,
    labels: Any,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Pair]] = defaultdict(list)
    for pair in pairs:
        raw = labels(pair[0])
        values = (raw,) if isinstance(raw, str) else tuple(raw)
        for value in values or (UNSPECIFIED,):
            grouped[public_label_hash(namespace, value)].append(pair)
    return {
        label_hash: comparison_for_pairs(grouped[label_hash]).as_dict()
        for label_hash in sorted(grouped)
    }


def _hashed_counter(namespace: str, values: Sequence[str]) -> dict[str, int]:
    counter = Counter(public_label_hash(namespace, value) for value in values)
    return {key: counter[key] for key in sorted(counter)}


def _provenance_summary(pairs: Sequence[Pair]) -> dict[str, Any]:
    def arm(rows: Sequence[PrecomputedResult]) -> dict[str, Any]:
        policy_hashes = {row.provenance.policy_hash for row in rows}
        profile_hashes = {row.provenance.profile_hash for row in rows}
        source_hashes = {row.provenance.source_hash for row in rows}
        registry_hashes = {row.provenance.registry.registry_hash for row in rows}
        if any(len(values) != 1 for values in (policy_hashes, profile_hashes, source_hashes, registry_hashes)):
            raise ValueError("mixed arm provenance reached report generation")
        decision_counter = Counter(
            f"{decision.rule_id}:{decision.decision.value}"
            for row in rows
            for decision in row.provenance.decisions
        )
        return {
            "policy_hash": next(iter(policy_hashes)),
            "profile_hash": next(iter(profile_hashes)),
            "source_hash": next(iter(source_hashes)),
            "registry_hash": next(iter(registry_hashes)),
            "engine": _hashed_counter("engine", [row.engine for row in rows]),
            "route": _hashed_counter("route", [row.provenance.route for row in rows]),
            "rule_id": dict(
                sorted(
                    Counter(
                        row.provenance.rule_id or "no_selected_rule"
                        for row in rows
                    ).items()
                )
            ),
            "cluster_id": dict(
                sorted(
                    Counter(
                        row.provenance.cluster_id or "no_selected_rule"
                        for row in rows
                    ).items()
                )
            ),
            "rule_decisions": dict(sorted(decision_counter.items())),
        }

    return {
        "baseline": arm([before for before, _ in pairs]),
        "variant": arm([after for _, after in pairs]),
        "route_transitions": _hashed_counter(
            "route_transition",
            [
                f"{before.provenance.route}:{after.provenance.route}"
                for before, after in pairs
            ],
        ),
    }


def _error_summary(pairs: Sequence[Pair]) -> dict[str, Any]:
    def arm(rows: Sequence[PrecomputedResult]) -> dict[str, Any]:
        errors = [row.error_category or UNSPECIFIED for row in rows if not row.correct]
        return {
            "n_incorrect": len(errors),
            "by_error_category": _hashed_counter("error_category", errors),
        }

    return {
        "known_categories": list(KNOWN_ERROR_CATEGORIES),
        "baseline": arm([before for before, _ in pairs]),
        "variant": arm([after for _, after in pairs]),
    }


def _pairs_where(pairs: Sequence[Pair], predicate: Any) -> tuple[Pair, ...]:
    return tuple(pair for pair in pairs if predicate(pair[0]))


def _comparison_by(
    pairs: Sequence[Pair],
    *,
    namespace: str,
    key: Any,
) -> dict[str, PairedStrictComparison]:
    grouped: dict[str, list[Pair]] = defaultdict(list)
    for pair in pairs:
        label = public_label_hash(namespace, key(pair[0]))
        grouped[label].append(pair)
    return {
        label: comparison_for_pairs(grouped[label]) for label in sorted(grouped)
    }


def _stability_for_pairs(
    pairs: Sequence[Pair],
    *,
    manifest: AblationArmManifest,
):
    baseline: dict[str, dict[str, dict[str, bool]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    variant: dict[str, dict[str, dict[str, bool]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    selected = _pairs_where(
        pairs,
        lambda row: (
            row.dataset_role == manifest.selection_role
            and row.db_id == manifest.base_db_id
        ),
    )
    if not selected:
        return None
    for before, after in selected:
        family = public_label_hash("family", before.family_id)
        item = _identity_token((before.slice_id, before.item_id))
        run_label = public_label_hash("run", before.run_id)
        baseline[family][run_label][item] = before.correct
        variant[family][run_label][item] = after.correct
    return compare_stability_populations(baseline, variant)


def _missing_acceptance(
    criteria: AblationAcceptanceCriteria,
    reason: str,
) -> dict[str, Any]:
    requirements = {
        "pooled_positive_delta": "manifest-bound pooled selection evidence",
        "shadow_worst_family_loss": "manifest-bound fixed slice+family evidence",
        "alternate_db_no_loss": "manifest-bound identical alternate-DB populations",
        "stability_no_worsening": "identical family/run populations for both arms",
    }
    return {
        "accepted": False,
        "status": "not_evaluable",
        "criteria": criteria.as_dict(),
        "reason": reason,
        "gates": {
            name: {
                "passed": False,
                "measured": None,
                "requirement": requirement,
                "evidence_count": 0,
                "reason": reason,
            }
            for name, requirement in requirements.items()
        },
        "measurements": {
            "pooled": None,
            "shadow_by_slice_family": {},
            "alternate_db": {},
            "stability": None,
        },
    }


def acceptance_for_pairs(
    pairs: Sequence[Pair],
    *,
    manifest: AblationArmManifest,
    criteria: AblationAcceptanceCriteria,
) -> dict[str, Any]:
    pooled_pairs = _pairs_where(
        pairs,
        lambda row: (
            row.dataset_role == manifest.selection_role
            and row.db_id == manifest.base_db_id
            and row.run_id == manifest.primary_run_id
        ),
    )
    if not pooled_pairs:
        return _missing_acceptance(criteria, "manifest-bound pooled selection is empty")
    shadow_pairs = _pairs_where(
        pairs,
        lambda row: (
            row.dataset_role == manifest.shadow_role
            and row.db_id == manifest.base_db_id
            and row.run_id == manifest.primary_run_id
        ),
    )
    alternate_pairs = _pairs_where(
        pairs,
        lambda row: (
            row.dataset_role == manifest.selection_role
            and row.db_id != manifest.base_db_id
            and row.run_id == manifest.primary_run_id
        ),
    )
    shadow = _comparison_by(
        shadow_pairs,
        namespace="shadow_slice_family",
        key=lambda row: f"{row.slice_id}:{row.family_id}",
    )
    alternate_db = _comparison_by(
        alternate_pairs,
        namespace="database",
        key=lambda row: row.db_id,
    )
    decision = assess_ablation_acceptance(
        pooled=comparison_for_pairs(pooled_pairs),
        shadow=shadow,
        alternate_db=alternate_db,
        stability=_stability_for_pairs(pairs, manifest=manifest),
        criteria=criteria,
    ).as_dict()
    decision["status"] = "accepted" if decision["accepted"] else "rejected"
    return decision


def _variant_report(
    pairs: Sequence[Pair],
    *,
    manifest: AblationArmManifest,
    criteria: AblationAcceptanceCriteria,
) -> dict[str, Any]:
    return {
        "all_items": comparison_for_pairs(pairs).as_dict(),
        "by_dataset_role": _public_group_comparisons(
            pairs,
            namespace="dataset_role",
            labels=lambda row: row.dataset_role,
        ),
        "by_difficulty": _public_group_comparisons(
            pairs,
            namespace="difficulty",
            labels=lambda row: row.difficulty,
        ),
        "by_category": _public_group_comparisons(
            pairs,
            namespace="category",
            labels=lambda row: row.category,
        ),
        "by_family": _public_group_comparisons(
            pairs,
            namespace="family",
            labels=lambda row: row.family_id,
        ),
        "by_database": _public_group_comparisons(
            pairs,
            namespace="database",
            labels=lambda row: row.db_id,
        ),
        "by_capability": _public_group_comparisons(
            pairs,
            namespace="capability",
            labels=lambda row: row.capabilities,
        ),
        "error_taxonomy": _error_summary(pairs),
        "provenance": _provenance_summary(pairs),
        "acceptance": acceptance_for_pairs(
            pairs,
            manifest=manifest,
            criteria=criteria,
        ),
    }


def build_ablation_report(
    baseline: BoundArm,
    variants: Mapping[str, BoundArm],
    *,
    criteria: AblationAcceptanceCriteria | None = None,
) -> dict[str, Any]:
    if not isinstance(baseline, BoundArm):
        raise TypeError("baseline must be a BoundArm")
    if not isinstance(variants, Mapping) or not variants:
        raise ValueError("at least one named variant is required")
    criteria = criteria or DEFAULT_ABLATION_ACCEPTANCE_CRITERIA
    baseline = bind_arm(baseline.manifest, baseline.results)
    variant_reports: dict[str, Any] = {}
    variant_sources: dict[str, Any] = {}
    for name in sorted(variants):
        validate_stable_id(name, field_name="variant_name")
        arm = variants[name]
        if not isinstance(arm, BoundArm):
            raise TypeError("variant values must be BoundArm instances")
        if name != arm.manifest.arm_id:
            raise ValueError("variant mapping name does not match manifest arm_id")
        public_arm_id = public_label_hash("arm", name)
        pairs = pair_bound_arms(baseline, arm)
        variant_reports[public_arm_id] = _variant_report(
            pairs,
            manifest=arm.manifest,
            criteria=criteria,
        )
        variant_sources[public_arm_id] = {
            "manifest": arm.manifest.public_summary(),
            "results": arm.source_summary(),
        }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "nl2sql_manifest_bound_ablation_report",
        "strict_all_item": True,
        "private_item_fields_published": False,
        "public_dimension_labels_hashed": True,
        "input_contract": {
            "required_row_fields": [
                "id",
                "correct",
                "dataset_role",
                "slice_id",
                "db_id",
                "run_id",
                "difficulty",
                "category",
                "family_id",
                "engine",
                "route",
                "registry_hash",
                "policy_hash",
                "profile_hash",
                "source_hash",
                "rule_id/cluster_id when a deterministic rule is selected",
            ],
            "pair_identity": [
                "dataset_role",
                "slice_id",
                "db_id",
                "run_id",
                "id",
            ],
            "strict_json": True,
            "incorrect_includes": [
                "abstention",
                "inference_failure",
                "execution_failure",
                "malformed_prediction",
            ],
        },
        "criteria": criteria.as_dict(),
        "baseline": {
            "manifest": baseline.manifest.public_summary(),
            "results": baseline.source_summary(),
        },
        "variants": variant_reports,
        "variant_sources": variant_sources,
        "all_variants_accepted": all(
            payload["acceptance"]["accepted"]
            for payload in variant_reports.values()
        ),
    }


def encode_report(report: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def publish_report_create_once(path: Path, report: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(encode_report(report))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("argument must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    try:
        validate_stable_id(name, field_name="variant_name")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not raw_path:
        raise argparse.ArgumentTypeError("path must be nonempty")
    return name, Path(raw_path)


def _parser() -> argparse.ArgumentParser:
    defaults = DEFAULT_ABLATION_ACCEPTANCE_CRITERIA
    parser = argparse.ArgumentParser(
        description="Compare manifest-bound precomputed NL2SQL ablation results.",
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--variant", type=_named_path, action="append", required=True)
    parser.add_argument(
        "--variant-manifest",
        type=_named_path,
        action="append",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--min-pooled-delta",
        type=float,
        default=defaults.min_pooled_delta,
    )
    parser.add_argument(
        "--max-shadow-loss",
        type=float,
        default=defaults.max_shadow_loss,
    )
    parser.add_argument(
        "--min-shadow-groups",
        type=int,
        default=defaults.min_shadow_groups,
    )
    parser.add_argument(
        "--min-alternate-dbs",
        type=int,
        default=defaults.min_alternate_dbs,
    )
    parser.add_argument(
        "--max-stability-range",
        type=float,
        default=defaults.max_stability_range,
    )
    parser.add_argument(
        "--max-stability-stddev",
        type=float,
        default=defaults.max_stability_stddev,
    )
    parser.add_argument(
        "--min-stability-runs",
        type=int,
        default=defaults.min_stability_runs,
    )
    parser.add_argument("--require-acceptance", action="store_true")
    return parser


def _named_paths(values: Sequence[tuple[str, Path]], *, label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for name, path in values:
        if name in result:
            raise ValueError(f"duplicate {label} name: {name}")
        result[name] = path
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if os.path.lexists(args.output):
        parser.error("output is create-once and already exists")
    try:
        variant_paths = _named_paths(args.variant, label="variant")
        manifest_paths = _named_paths(args.variant_manifest, label="variant manifest")
        if variant_paths.keys() != manifest_paths.keys():
            raise ValueError("variant and variant-manifest names differ")
        criteria = AblationAcceptanceCriteria(
            min_pooled_delta=args.min_pooled_delta,
            max_shadow_loss=args.max_shadow_loss,
            min_shadow_groups=args.min_shadow_groups,
            min_alternate_dbs=args.min_alternate_dbs,
            max_stability_range=args.max_stability_range,
            max_stability_stddev=args.max_stability_stddev,
            min_stability_runs=args.min_stability_runs,
        )
        baseline_manifest = load_arm_manifest(args.baseline_manifest)
        baseline = bind_arm(
            baseline_manifest,
            load_precomputed_results(
                args.baseline,
                registry=baseline_manifest.registry,
            ),
        )
        variants: dict[str, BoundArm] = {}
        for name in sorted(variant_paths):
            manifest = load_arm_manifest(manifest_paths[name])
            if manifest.arm_id != name:
                raise ValueError(f"variant {name} does not match manifest arm_id")
            variants[name] = bind_arm(
                manifest,
                load_precomputed_results(
                    variant_paths[name],
                    registry=manifest.registry,
                ),
            )
        report = build_ablation_report(baseline, variants, criteria=criteria)
        publish_report_create_once(args.output, report)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "all_variants_accepted": report["all_variants_accepted"],
                "output_sha256": _sha256_file(args.output),
                "variants": {
                    name: {
                        "accepted": payload["acceptance"]["accepted"],
                        "delta_accuracy": payload["all_items"]["delta_accuracy"],
                        "n": payload["all_items"]["n"],
                    }
                    for name, payload in sorted(report["variants"].items())
                },
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    if args.require_acceptance and not report["all_variants_accepted"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
