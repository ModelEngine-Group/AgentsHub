"""Immutable semantic contract for one CMeIE GPLinker epoch bundle.

``extraction_contract.json`` deliberately contains no digest of itself.  The
caller stores the returned binding outside the contract (normally in the
training receipt), while ``checkpoint_bundle.json`` subsequently binds the
contract file as an ordinary epoch-bundle member.

The three legacy semantic digests in this module intentionally use the same
canonical encodings as the original CMeIE training entrypoint.  The aggregate
semantic-contract digest additionally binds the tokenizer vocabulary identity
to every extraction setting that affects label IDs or decoding.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dki.evidence_io import load_bound_json


EXTRACTION_CONTRACT_FILENAME = "extraction_contract.json"
EXTRACTION_CONTRACT_SCHEMA_VERSION = 1
EXTRACTION_CONTRACT_KIND = "cmeie_gplinker_extraction_contract"
CHECKPOINT_BUNDLE_MANIFEST = "checkpoint_bundle.json"

FIXED_MAX_LENGTH = 320
FIXED_POINTER_VARIANT = "efficient"
FIXED_HEAD_REDUCTION = "balanced"
FIXED_THRESHOLD_VALUES = (-0.5, 0.0, 0.5)
FIXED_GRID_SIZE = 54

TOKENIZER_VOCABULARY_HASH_ALGORITHM = (
    "sha256 of UTF-8 canonical JSON sorted [token,id] pairs"
)
LABEL_TABLE_HASH_ALGORITHM = (
    "sha256 of UTF-8 canonical JSON object "
    "{entity_labels,predicates} with sorted keys"
)
SCHEMA_CONSTRAINTS_HASH_ALGORITHM = (
    "sha256 of UTF-8 canonical JSON with sorted object keys"
)
SEMANTIC_CONTRACT_HASH_ALGORITHM = (
    "sha256 of UTF-8 canonical JSON object "
    "{contract,tokenizer_vocabulary_sha256} with sorted keys"
)
BINDING_ALGORITHMS = {
    "extraction_contract_sha256": (
        "sha256 of exact canonical extraction_contract.json bytes"
    ),
    "tokenizer_vocabulary_sha256": (
        TOKENIZER_VOCABULARY_HASH_ALGORITHM
    ),
    "label_table_sha256": LABEL_TABLE_HASH_ALGORITHM,
    "schema_constraints_sha256": (
        SCHEMA_CONSTRAINTS_HASH_ALGORITHM
    ),
    "semantic_contract_sha256": SEMANTIC_CONTRACT_HASH_ALGORITHM,
}

_REPARSE_POINT_ATTRIBUTE = getattr(
    stat,
    "FILE_ATTRIBUTE_REPARSE_POINT",
    0x0400,
)
_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "entity_labels",
        "predicates",
        "schema_constraints",
        "max_length",
        "pointer_variant",
        "head_reduction",
        "threshold_values",
        "grid_size",
    }
)
_HASH_KEYS = (
    "extraction_contract_sha256",
    "tokenizer_vocabulary_sha256",
    "label_table_sha256",
    "schema_constraints_sha256",
    "semantic_contract_sha256",
)
_SHA256_LENGTH = 64


class ExtractionContractError(ValueError):
    """The extraction contract cannot be published or validated."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _compact_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ExtractionContractError(
            "extraction contract contains a non-JSON or non-finite value"
        ) from exc
    return text.encode("utf-8")


def _contract_bytes(payload: Mapping[str, Any]) -> bytes:
    return _compact_json_bytes(payload) + b"\n"


def _is_real_directory(path: Path) -> bool:
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except OSError:
        return False
    return (
        stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and not bool(
            getattr(metadata, "st_file_attributes", 0)
            & _REPARSE_POINT_ATTRIBUTE
        )
    )


def _checkpoint_root(checkpoint_dir: str | os.PathLike[str]) -> Path:
    try:
        raw = os.fspath(checkpoint_dir)
    except TypeError as exc:
        raise ExtractionContractError(
            "checkpoint directory path is invalid"
        ) from exc
    if not isinstance(raw, str) or not raw or "\0" in raw:
        raise ExtractionContractError(
            "checkpoint directory path must be non-empty text"
        )
    root = Path(os.path.abspath(raw))
    if not _is_real_directory(root):
        raise ExtractionContractError(
            "checkpoint directory must be a real existing directory"
        )
    return root


def _nonempty_unique_strings(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ExtractionContractError(f"{field} must be a sequence")
    if not value:
        raise ExtractionContractError(f"{field} must not be empty")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if type(item) is not str or not item.strip():
            raise ExtractionContractError(
                f"{field} values must be non-empty strings"
            )
        if item in seen:
            raise ExtractionContractError(
                f"{field} values must be unique"
            )
        seen.add(item)
        result.append(item)
    return result


def _schema_constraints(
    value: Any,
    *,
    entity_labels: Sequence[str],
    predicates: Sequence[str],
) -> dict[str, list[list[str]]]:
    if not isinstance(value, Mapping) or not value:
        raise ExtractionContractError(
            "schema_constraints must be a non-empty object"
    )
    allowed_entity_labels = set(entity_labels)
    allowed_predicates = set(predicates)
    result: dict[str, list[list[str]]] = {}
    for predicate, pairs in value.items():
        if type(predicate) is not str or not predicate.strip():
            raise ExtractionContractError(
                "schema constraint predicates must be non-empty strings"
            )
        if predicate not in allowed_predicates:
            raise ExtractionContractError(
                "schema_constraints contains an unknown predicate: "
                f"{predicate!r}"
            )
        if not isinstance(pairs, (list, tuple)) or not pairs:
            raise ExtractionContractError(
                "each schema constraint must contain at least one type pair"
            )
        normalised_pairs: list[list[str]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ExtractionContractError(
                    "schema constraint entries must be two-item sequences"
                )
            subject, obj = pair
            if (
                type(subject) is not str
                or type(obj) is not str
                or not subject.strip()
                or not obj.strip()
            ):
                raise ExtractionContractError(
                    "schema constraint types must be non-empty strings"
                )
            if (
                subject not in allowed_entity_labels
                or obj not in allowed_entity_labels
            ):
                raise ExtractionContractError(
                    "schema constraint types must occur in entity_labels"
                )
            signature = (subject, obj)
            if signature in seen_pairs:
                raise ExtractionContractError(
                    "schema constraint type pairs must be unique"
                )
            seen_pairs.add(signature)
            normalised_pairs.append([subject, obj])
        result[predicate] = normalised_pairs
    return result


def _tokenizer_vocabulary(
    value: Any,
) -> list[list[str | int]]:
    if not isinstance(value, Mapping) or not value:
        raise ExtractionContractError(
            "tokenizer_vocabulary must be a non-empty mapping"
        )
    pairs: list[list[str | int]] = []
    identifiers: set[int] = set()
    for token, identifier in value.items():
        if type(token) is not str or not token:
            raise ExtractionContractError(
                "tokenizer vocabulary tokens must be non-empty strings"
            )
        if (
            type(identifier) is not int
            or identifier < 0
            or identifier in identifiers
        ):
            raise ExtractionContractError(
                "tokenizer vocabulary IDs must be unique non-negative integers"
            )
        identifiers.add(identifier)
        pairs.append([token, identifier])
    pairs.sort(key=lambda pair: (pair[0], pair[1]))
    return pairs


def _fixed_integer(value: Any, *, field: str, expected: int) -> int:
    if type(value) is not int or value != expected:
        raise ExtractionContractError(
            f"{field} must be the fixed integer {expected}"
        )
    return value


def _fixed_text(value: Any, *, field: str, expected: str) -> str:
    if type(value) is not str or value != expected:
        raise ExtractionContractError(
            f"{field} must be the fixed value {expected!r}"
        )
    return value


def _threshold_values(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != len(
        FIXED_THRESHOLD_VALUES
    ):
        raise ExtractionContractError(
            "threshold_values must be the fixed three-item grid"
        )
    result: list[float] = []
    for actual, expected in zip(value, FIXED_THRESHOLD_VALUES):
        if (
            type(actual) is not float
            or not math.isfinite(actual)
            or actual != expected
        ):
            raise ExtractionContractError(
                "threshold_values must equal [-0.5, 0.0, 0.5]"
            )
        result.append(expected)
    return result


def _payload(
    *,
    entity_labels: Any,
    predicates: Any,
    schema_constraints: Any,
    max_length: Any,
    pointer_variant: Any,
    head_reduction: Any,
    threshold_values: Any,
    grid_size: Any,
) -> dict[str, Any]:
    labels = _nonempty_unique_strings(
        entity_labels,
        field="entity_labels",
    )
    predicate_table = _nonempty_unique_strings(
        predicates,
        field="predicates",
    )
    return {
        "schema_version": EXTRACTION_CONTRACT_SCHEMA_VERSION,
        "kind": EXTRACTION_CONTRACT_KIND,
        "entity_labels": labels,
        "predicates": predicate_table,
        "schema_constraints": _schema_constraints(
            schema_constraints,
            entity_labels=labels,
            predicates=predicate_table,
        ),
        "max_length": _fixed_integer(
            max_length,
            field="max_length",
            expected=FIXED_MAX_LENGTH,
        ),
        "pointer_variant": _fixed_text(
            pointer_variant,
            field="pointer_variant",
            expected=FIXED_POINTER_VARIANT,
        ),
        "head_reduction": _fixed_text(
            head_reduction,
            field="head_reduction",
            expected=FIXED_HEAD_REDUCTION,
        ),
        "threshold_values": _threshold_values(threshold_values),
        "grid_size": _fixed_integer(
            grid_size,
            field="grid_size",
            expected=FIXED_GRID_SIZE,
        ),
    }


def _validate_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _CONTRACT_KEYS:
        raise ExtractionContractError(
            "extraction contract has an invalid field set"
        )
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version")
        != EXTRACTION_CONTRACT_SCHEMA_VERSION
    ):
        raise ExtractionContractError(
            "extraction contract schema_version is invalid"
        )
    if (
        type(value.get("kind")) is not str
        or value.get("kind") != EXTRACTION_CONTRACT_KIND
    ):
        raise ExtractionContractError(
            "extraction contract kind is invalid"
        )
    normalised = _payload(
        entity_labels=value.get("entity_labels"),
        predicates=value.get("predicates"),
        schema_constraints=value.get("schema_constraints"),
        max_length=value.get("max_length"),
        pointer_variant=value.get("pointer_variant"),
        head_reduction=value.get("head_reduction"),
        threshold_values=value.get("threshold_values"),
        grid_size=value.get("grid_size"),
    )
    if value != normalised:
        raise ExtractionContractError(
            "extraction contract values are not canonical"
        )
    return normalised


def _semantic_hashes(
    payload: Mapping[str, Any],
    tokenizer_vocabulary: Any,
) -> dict[str, str]:
    vocabulary = _tokenizer_vocabulary(tokenizer_vocabulary)
    tokenizer_sha256 = _sha256(_compact_json_bytes(vocabulary))
    label_table = {
        "entity_labels": payload["entity_labels"],
        "predicates": payload["predicates"],
    }
    schema = payload["schema_constraints"]
    semantic_contract = {
        "contract": dict(payload),
        "tokenizer_vocabulary_sha256": tokenizer_sha256,
    }
    return {
        "tokenizer_vocabulary_sha256": tokenizer_sha256,
        "label_table_sha256": _sha256(
            _compact_json_bytes(label_table)
        ),
        "schema_constraints_sha256": _sha256(
            _compact_json_bytes(schema)
        ),
        "semantic_contract_sha256": _sha256(
            _compact_json_bytes(semantic_contract)
        ),
    }


def _binding(
    payload: Mapping[str, Any],
    raw: bytes,
    tokenizer_vocabulary: Any,
) -> dict[str, str]:
    return {
        "extraction_contract": EXTRACTION_CONTRACT_FILENAME,
        "extraction_contract_sha256": _sha256(raw),
        **_semantic_hashes(payload, tokenizer_vocabulary),
    }


def _validate_sha256(field: str, value: Any) -> str:
    if (
        type(value) is not str
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExtractionContractError(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def _expected_hashes(
    expected_binding: Mapping[str, Any] | None,
    explicit: Mapping[str, Any],
) -> dict[str, str]:
    if expected_binding is not None and not isinstance(
        expected_binding,
        Mapping,
    ):
        raise ExtractionContractError(
            "expected_binding must be a mapping"
        )
    if expected_binding is not None:
        expected_name = expected_binding.get("extraction_contract")
        if (
            expected_name is not None
            and expected_name != EXTRACTION_CONTRACT_FILENAME
        ):
            raise ExtractionContractError(
                "expected extraction contract filename is invalid"
            )
    result: dict[str, str] = {}
    for field in _HASH_KEYS:
        binding_value = (
            expected_binding.get(field)
            if expected_binding is not None
            else None
        )
        explicit_value = explicit.get(field)
        if (
            binding_value is not None
            and explicit_value is not None
            and binding_value != explicit_value
        ):
            raise ExtractionContractError(
                f"conflicting expected value for {field}"
            )
        selected = (
            explicit_value
            if explicit_value is not None
            else binding_value
        )
        if selected is None:
            raise ExtractionContractError(
                f"missing expected value for {field}"
            )
        result[field] = _validate_sha256(field, selected)
    return result


def publish_extraction_contract_once(
    checkpoint_dir: str | os.PathLike[str],
    *,
    tokenizer_vocabulary: Mapping[str, int],
    entity_labels: Sequence[str],
    predicates: Sequence[str],
    schema_constraints: Mapping[
        str,
        Sequence[Sequence[str]],
    ],
    max_length: int = FIXED_MAX_LENGTH,
    pointer_variant: str = FIXED_POINTER_VARIANT,
    head_reduction: str = FIXED_HEAD_REDUCTION,
    threshold_values: Sequence[float] = FIXED_THRESHOLD_VALUES,
    grid_size: int = FIXED_GRID_SIZE,
) -> dict[str, str]:
    """Atomically publish the create-once contract before bundle freezing."""

    root = _checkpoint_root(checkpoint_dir)
    manifest_path = root / CHECKPOINT_BUNDLE_MANIFEST
    if os.path.lexists(manifest_path):
        raise ExtractionContractError(
            "extraction contract must be published before the "
            "checkpoint bundle manifest"
        )
    target = root / EXTRACTION_CONTRACT_FILENAME
    if os.path.lexists(target):
        raise ExtractionContractError(
            f"extraction contract already exists: {target}"
        )

    payload = _payload(
        entity_labels=entity_labels,
        predicates=predicates,
        schema_constraints=schema_constraints,
        max_length=max_length,
        pointer_variant=pointer_variant,
        head_reduction=head_reduction,
        threshold_values=threshold_values,
        grid_size=grid_size,
    )
    raw = _contract_bytes(payload)
    binding = _binding(payload, raw, tokenizer_vocabulary)

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{EXTRACTION_CONTRACT_FILENAME}.",
            dir=root,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise ExtractionContractError(
                "extraction contract publication lost a create-once race"
            ) from exc
        except OSError as exc:
            if os.path.lexists(target):
                raise ExtractionContractError(
                    "extraction contract publication lost a "
                    "create-once race"
                ) from exc
            raise ExtractionContractError(
                "cannot atomically publish extraction contract"
            ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    return load_and_validate_extraction_contract(
        root,
        tokenizer_vocabulary=tokenizer_vocabulary,
        expected_binding=binding,
    )


def load_and_validate_extraction_contract(
    checkpoint_dir: str | os.PathLike[str],
    *,
    tokenizer_vocabulary: Mapping[str, int],
    expected_binding: Mapping[str, Any] | None = None,
    expected_extraction_contract_sha256: str | None = None,
    expected_tokenizer_vocabulary_sha256: str | None = None,
    expected_label_table_sha256: str | None = None,
    expected_schema_constraints_sha256: str | None = None,
    expected_semantic_contract_sha256: str | None = None,
) -> dict[str, str]:
    """Bound-read the contract and compare every independently stored digest.

    ``expected_binding`` may be the mapping returned by the publisher or a
    larger receipt containing those fields.  Individual ``expected_*``
    arguments are also accepted, but every hash is mandatory and a duplicate
    specification must agree.  This prevents a receipt from silently changing
    one semantic digest while retaining the others.
    """

    root = _checkpoint_root(checkpoint_dir)
    expected = _expected_hashes(
        expected_binding,
        {
            "extraction_contract_sha256": (
                expected_extraction_contract_sha256
            ),
            "tokenizer_vocabulary_sha256": (
                expected_tokenizer_vocabulary_sha256
            ),
            "label_table_sha256": expected_label_table_sha256,
            "schema_constraints_sha256": (
                expected_schema_constraints_sha256
            ),
            "semantic_contract_sha256": (
                expected_semantic_contract_sha256
            ),
        },
    )
    path = root / EXTRACTION_CONTRACT_FILENAME
    try:
        value, evidence = load_bound_json(
            path,
            "CMeIE extraction contract",
            reject_multiple_links=True,
            require_object=True,
        )
    except (OSError, ValueError) as exc:
        raise ExtractionContractError(
            "cannot load bound CMeIE extraction contract"
        ) from exc

    payload = _validate_payload(value)
    canonical = _contract_bytes(payload)
    if (
        evidence.get("size") != len(canonical)
        or evidence.get("sha256") != _sha256(canonical)
    ):
        raise ExtractionContractError(
            "extraction contract is not canonical JSON"
        )
    actual = _binding(payload, canonical, tokenizer_vocabulary)
    mismatches = [
        field
        for field in _HASH_KEYS
        if actual[field] != expected[field]
    ]
    if mismatches:
        raise ExtractionContractError(
            "extraction contract binding mismatch: "
            + ", ".join(mismatches)
        )
    return actual
