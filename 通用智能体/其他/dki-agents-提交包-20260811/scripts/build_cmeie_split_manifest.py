"""Create deterministic, family-disjoint physical CMeIE data shards.

The source corpus is opened only by this builder.  It publishes a manifest and
three fixed JSONL shards as one atomic, create-once directory so fit-only and
calibration consumers never need access to the original file or outer rows.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.nlp.cmeie_split_protocol import (
    BoundSourceSnapshot,
    FIXED_SEED,
    GROUPING_POLICY,
    MANIFEST_FILENAME,
    PREDICATE_COVERAGE_POLICY,
    SCHEMA_VERSION,
    SHARD_FILENAMES,
    SPLIT_RATIOS,
    build_shard_bytes,
    load_and_validate_role_shard,
    load_cmeie_source_bound,
    shard_evidence,
    validate_split_manifest,
)
from dki.nlp.cmeie_run_protocol import (
    RunBinding,
    canonical_run_paths,
    load_run_binding,
)


_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def normalise_sha256(value: str, *, field: str = "source_tree_sha256") -> str:
    """Validate a SHA-256 binding and return its canonical lowercase form."""
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be exactly 64 hexadecimal characters")
    return value.lower()


def _source_tree_sha256_argument(value: str) -> str:
    try:
        return normalise_sha256(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def source_tree_manifest_sha256(
    path: Path,
    expected_tree_sha256: str,
) -> str:
    """Validate a v1 included-tree manifest and hash its exact source bytes."""
    expected_tree_sha256 = normalise_sha256(expected_tree_sha256)
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("source tree manifest must be valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("source tree manifest root must be a JSON object")
    if (
        type(payload.get("schema_version")) is not int
        or payload["schema_version"] != 1
    ):
        raise ValueError("source tree manifest schema_version must be 1")
    if "included_tree_sha256" not in payload:
        raise ValueError("source tree manifest is missing included_tree_sha256")
    included_tree_sha256 = normalise_sha256(
        payload["included_tree_sha256"],
        field="included_tree_sha256",
    )
    if included_tree_sha256 != expected_tree_sha256:
        raise ValueError(
            "source tree manifest included_tree_sha256 does not match "
            "--source-tree-sha256"
        )
    return sha256_bytes(raw)


def normalise(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"[\s\W_]+", "", value, flags=re.UNICODE).lower()


def family_key(text: str) -> str:
    """Group an article title, or its stable leading text when no title exists."""
    title, separator, _ = str(text).partition("@")
    basis = title if separator else str(text)[:32]
    key = normalise(basis)
    return key or sha256_bytes(str(text).encode("utf-8"))[:32]


def _row_hash(row: dict[str, Any]) -> str:
    return sha256_bytes(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def family_assignments(rows: list[dict[str, Any]]) -> dict[int, str]:
    """Return the final coalesced family owner for every source row."""
    # Exact and normalized duplicates use the same family as their title/prefix
    # family; this makes text duplication unable to leak across a split.
    normalized_owner: dict[str, str] = {}
    near_owner: dict[str, str] = {}
    assignments: dict[int, str] = {}
    for index, row in enumerate(rows):
        text_key = normalise(row.get("text", ""))
        key = family_key(row.get("text", ""))
        if text_key in normalized_owner:
            key = normalized_owner[text_key]
        else:
            normalized_owner[text_key] = key
        # A stable leading fragment catches editorial near-duplicates that
        # differ after a title/boilerplate prefix while retaining the required
        # 32-character key as the primary family identity.
        near_key = text_key[:16]
        if near_key and near_key in near_owner:
            key = near_owner[near_key]
        elif near_key:
            near_owner[near_key] = key
        assignments[index] = key
    return assignments


def _ranked_families(rows: list[dict[str, Any]], seed: str) -> list[tuple[str, list[int]]]:
    families: dict[str, list[int]] = defaultdict(list)
    for index, key in family_assignments(rows).items():
        families[key].append(index)
    return sorted(
        families.items(),
        key=lambda item: sha256_bytes(f"{seed}\0{item[0]}".encode("utf-8")),
    )


def build_split(
    rows: list[dict[str, Any]],
    seed: str = FIXED_SEED,
) -> dict[str, list[int]]:
    """Greedily allocate family groups by size and predicate incidence.

    The deterministic ranking breaks all ties.  Large groups are placed first
    to prevent a late group from overshooting a small validation partition.
    """
    if len(rows) < 3:
        raise ValueError("CMeIE split requires at least three rows")
    groups = _ranked_families(rows, seed)
    predicate_total = Counter(
        str(spo.get("predicate"))
        for row in rows for spo in row.get("spo_list", []) if spo.get("predicate")
    )
    predicate_family_count: Counter[str] = Counter()
    for _, indices in groups:
        predicate_family_count.update(
            {
                str(spo.get("predicate"))
                for index in indices
                for spo in rows[index].get("spo_list", [])
                if spo.get("predicate")
            }
        )
    supported_predicates = {
        predicate
        for predicate, count in predicate_total.items()
        if count >= 10 and predicate_family_count[predicate] >= 3
    }
    target_ratios = dict(SPLIT_RATIOS)
    targets = {
        name: ratio * len(rows)
        for name, ratio in target_ratios.items()
    }
    assigned: dict[str, list[int]] = {name: [] for name in targets}
    sizes = Counter()
    predicate_counts: dict[str, Counter[str]] = {name: Counter() for name in targets}
    for family, indices in sorted(groups, key=lambda item: (-len(item[1]), item[0])):
        group_predicates = Counter(
            str(spo.get("predicate"))
            for index in indices for spo in rows[index].get("spo_list", []) if spo.get("predicate")
        )
        def cost(split: str) -> tuple[int, float, float, str]:
            # Compare *fill fractions*, not distance to the final target.  The
            # previous distance cost placed the first large family into a 10%
            # split because 200 rows is numerically closer to 1,400 than to
            # 11,500, which systematically produced an unintended 60/20/20
            # split.  Lowest normalized fill assigns work proportionally.
            size_fill = (
                sizes[split] + len(indices)
            ) / max(1.0, targets[split])
            if group_predicates:
                predicate_fill = sum(
                    (
                        predicate_counts[split][predicate] + count
                    )
                    / max(
                        1.0,
                        predicate_total[predicate] * target_ratios[split],
                    )
                    for predicate, count in group_predicates.items()
                ) / len(group_predicates)
            else:
                predicate_fill = size_fill
            blended_fill = 0.75 * size_fill + 0.25 * predicate_fill
            newly_covered = sum(
                predicate in supported_predicates
                and predicate_counts[split][predicate] == 0
                for predicate in group_predicates
            )
            return (-newly_covered, blended_fill, size_fill, split)
        split = min(targets, key=cost)
        assigned[split].extend(indices)
        sizes[split] += len(indices)
        predicate_counts[split].update(group_predicates)
    missing = {
        name: sorted(supported_predicates - set(predicate_counts[name]))
        for name in targets
    }
    if any(missing.values()):
        raise RuntimeError(
            "single family-disjoint split did not cover sufficiently supported "
            f"predicates; use the declared five-fold fallback: {missing}"
        )
    return {name: sorted(indices) for name, indices in assigned.items()}


def _hash_list(values: list[str]) -> str:
    return sha256_bytes("\n".join(sorted(values)).encode("utf-8"))


def make_manifest(
    rows: list[dict[str, Any]],
    source: Path,
    seed: str,
    source_tree_sha256: str,
    source_tree_manifest_sha256: str,
    run_binding: RunBinding | None = None,
    *,
    _bound_source: BoundSourceSnapshot | None = None,
) -> dict[str, Any]:
    source_tree_sha256 = normalise_sha256(source_tree_sha256)
    source_tree_manifest_sha256 = normalise_sha256(
        source_tree_manifest_sha256,
        field="source_tree_manifest_sha256",
    )
    source_snapshot = (
        load_cmeie_source_bound(source)
        if _bound_source is None
        else _bound_source
    )
    (
        bound_rows,
        source_row_bytes,
        source_raw,
        source_binding,
    ) = source_snapshot
    source_path_value = source_binding.get("canonical_path")
    source_sha256_value = source_binding.get("sha256")
    if not isinstance(source_path_value, str) or not isinstance(
        source_sha256_value,
        str,
    ):
        raise ValueError("bound CMeIE source lacks path or SHA-256")
    canonical_source = Path(source_path_value)
    if (
        source_binding.get("size") != len(source_raw)
        or source_sha256_value != sha256_bytes(source_raw)
    ):
        raise ValueError("bound CMeIE source bytes differ from their binding")
    if (
        len(rows) != len(bound_rows)
        or [_row_hash(row) for row in rows]
        != [_row_hash(row) for row in bound_rows]
    ):
        raise ValueError(
            "rows supplied to the split builder differ from the bound source"
        )
    rows = bound_rows
    if run_binding is not None:
        if type(run_binding) is not RunBinding:
            raise TypeError("run_binding must be a RunBinding")
        if source_tree_sha256 != run_binding.included_tree_sha256:
            raise ValueError(
                "source tree SHA-256 differs from the run binding"
            )
        if (
            source_tree_manifest_sha256
            != run_binding.source_manifest_sha256
        ):
            raise ValueError(
                "source tree manifest SHA-256 differs from the run binding"
            )
        run_identity = {
            "rid": run_binding.rid,
            "git_head": run_binding.git_head,
            "canonical_run_root": str(run_binding.run_root),
        }
    else:
        synthetic_rid = (
            "19700101T000000Z"
            f"-g{source_tree_sha256[:7]}"
            f"-t{source_tree_sha256[:12]}"
        )
        run_identity = {
            "rid": synthetic_rid,
            "git_head": source_tree_sha256[:40],
            "canonical_run_root": str(
                (
                    canonical_source.parent
                    / f"dki-agents-{synthetic_rid}"
                )
            ),
        }
    split = build_split(rows, seed)
    shards = build_shard_bytes(source_row_bytes, split)
    text_hashes = {index: sha256_bytes(normalise(row.get("text", "")).encode("utf-8")) for index, row in enumerate(rows)}
    families = dict(family_assignments(rows))
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        **run_identity,
        "seed": str(seed),
        "source_tree_sha256": source_tree_sha256,
        "source_tree_manifest_sha256": source_tree_manifest_sha256,
        "source_train": str(canonical_source),
        "source_train_sha256": source_sha256_value,
        "split_ratios": dict(SPLIT_RATIOS),
        "grouping": GROUPING_POLICY,
        "predicate_coverage_policy": dict(PREDICATE_COVERAGE_POLICY),
        "splits": {},
    }
    family_sets: dict[str, set[str]] = {}
    text_sets: dict[str, set[str]] = {}
    for name, indices in split.items():
        family_set = {families[index] for index in indices}
        text_set = {text_hashes[index] for index in indices}
        family_sets[name], text_sets[name] = family_set, text_set
        predicate_counts = Counter(
            str(spo.get("predicate"))
            for index in indices for spo in rows[index].get("spo_list", []) if spo.get("predicate")
        )
        payload["splits"][name] = {
            "indices": indices,
            "n_rows": len(indices),
            "target_ratio": payload["split_ratios"][name],
            "realized_ratio": len(indices) / max(1, len(rows)),
            "absolute_ratio_error": abs(
                len(indices) / max(1, len(rows))
                - payload["split_ratios"][name]
            ),
            "row_sha256": _hash_list([_row_hash(rows[index]) for index in indices]),
            "family_sha256": _hash_list(list(family_set)),
            "normalized_text_sha256": _hash_list(list(text_set)),
            "predicate_counts": dict(sorted(predicate_counts.items())),
            "shard": shard_evidence(
                name,
                shards[name],
                len(indices),
            ),
        }
    pairs = (("fit", "calibration"), ("fit", "outer"), ("calibration", "outer"))
    payload["intersections"] = {
        "families": {f"{left}:{right}": sorted(family_sets[left] & family_sets[right]) for left, right in pairs},
        "normalized_text_hashes": {f"{left}:{right}": sorted(text_sets[left] & text_sets[right]) for left, right in pairs},
    }
    if any(payload["intersections"][kind][pair] for kind in payload["intersections"] for pair in payload["intersections"][kind]):
        raise RuntimeError("family or normalized-text leakage while constructing CMeIE split")
    payload["generator_sha256"] = sha256_file(Path(__file__))
    validate_split_manifest(
        payload,
        rows,
        train_path=canonical_source,
        expected_source_tree_sha256=source_tree_sha256,
        expected_source_tree_manifest_sha256=(
            source_tree_manifest_sha256
        ),
        generator_path=Path(__file__),
        expected_run_binding=run_binding,
        _bound_source=source_snapshot,
    )
    return payload


def _publish_directory_create_once(
    staging: Path,
    destination: Path,
) -> None:
    """Atomically publish one same-parent directory without replacement."""
    staging = Path(staging).resolve(strict=True)
    destination = Path(destination)
    destination_parent = destination.parent.resolve(strict=True)
    destination = destination_parent / destination.name
    if staging.is_symlink() or not staging.is_dir():
        raise ValueError("publication staging path must be a real directory")
    if staging.parent != destination_parent:
        raise ValueError(
            "publication staging and destination must share one directory"
        )
    if os.name == "nt":
        move_file = ctypes.WinDLL(
            "kernel32",
            use_last_error=True,
        ).MoveFileExW
        move_file.argtypes = (
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_uint32,
        )
        move_file.restype = ctypes.c_int
        if move_file(str(staging), str(destination), 0):
            return
        error = ctypes.get_last_error()
        if os.path.lexists(destination) or error in {80, 183}:
            raise FileExistsError(
                errno.EEXIST,
                "split directory already exists",
                str(destination),
            )
        raise ctypes.WinError(error)
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            rename_no_replace = libc.renameat2
        except AttributeError as exc:
            raise OSError(
                errno.ENOSYS,
                "renameat2 is required for create-once publication",
                str(destination),
            ) from exc
        rename_no_replace.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename_no_replace.restype = ctypes.c_int
        ctypes.set_errno(0)
        if (
            rename_no_replace(
                _AT_FDCWD,
                os.fsencode(staging),
                _AT_FDCWD,
                os.fsencode(destination),
                _RENAME_NOREPLACE,
            )
            == 0
        ):
            return
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                error,
                "split directory already exists",
                str(destination),
            )
        raise OSError(error, os.strerror(error), str(destination))
    raise OSError(
        errno.ENOTSUP,
        "create-once directory publication is unsupported",
        str(destination),
    )


def _write_staging_file(path: Path, raw: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def build_split_directory(
    *,
    source: Path,
    output_dir: Path,
    seed: str,
    source_tree_sha256: str,
    source_tree_manifest_sha256: str,
    run_binding: RunBinding | None = None,
) -> dict[str, Any]:
    """Build, self-validate, and atomically publish one v6 split directory."""
    if os.path.lexists(output_dir):
        raise FileExistsError(
            f"split directory already exists; choose a new RID: {output_dir}"
        )
    if run_binding is not None:
        canonical_split_dir = canonical_run_paths(run_binding).split_dir
        if (
            not output_dir.is_absolute()
            or Path(os.path.abspath(output_dir)) != output_dir
            or output_dir != canonical_split_dir
        ):
            raise ValueError(
                "output_dir differs from the run binding canonical split_dir"
            )
        if (
            normalise_sha256(source_tree_sha256)
            != run_binding.included_tree_sha256
        ):
            raise ValueError(
                "source tree SHA-256 differs from the run binding"
            )
        if normalise_sha256(
            source_tree_manifest_sha256,
            field="source_tree_manifest_sha256",
        ) != run_binding.source_manifest_sha256:
            raise ValueError(
                "source tree manifest SHA-256 differs from the run binding"
            )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".staging",
            dir=output_dir.parent,
        )
    )
    source_snapshot = load_cmeie_source_bound(source)
    rows, source_row_bytes, _, _ = source_snapshot
    manifest = make_manifest(
        rows,
        source,
        seed,
        source_tree_sha256,
        source_tree_manifest_sha256,
        run_binding,
        _bound_source=source_snapshot,
    )
    split = {
        role: manifest["splits"][role]["indices"]
        for role in SHARD_FILENAMES
    }
    shards = build_shard_bytes(source_row_bytes, split)
    for role, filename in SHARD_FILENAMES.items():
        _write_staging_file(staging / filename, shards[role])
    manifest_raw = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    manifest_path = staging / MANIFEST_FILENAME
    _write_staging_file(manifest_path, manifest_raw)
    for role, filename in SHARD_FILENAMES.items():
        load_and_validate_role_shard(
            manifest_path,
            staging / filename,
            role=role,
            expected_source_tree_sha256=source_tree_sha256,
            expected_source_tree_manifest_sha256=(
                source_tree_manifest_sha256
            ),
            generator_path=Path(__file__),
            expected_run_binding=run_binding,
            allow_same_parent_staging=run_binding is not None,
        )
    _publish_directory_create_once(staging, output_dir)
    if run_binding is not None:
        for role, filename in SHARD_FILENAMES.items():
            load_and_validate_role_shard(
                output_dir / MANIFEST_FILENAME,
                output_dir / filename,
                role=role,
                expected_source_tree_sha256=source_tree_sha256,
                expected_source_tree_manifest_sha256=(
                    source_tree_manifest_sha256
                ),
                generator_path=Path(__file__),
                expected_run_binding=run_binding,
            )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", default=FIXED_SEED, choices=[FIXED_SEED])
    parser.add_argument(
        "--source-tree-sha256",
        type=_source_tree_sha256_argument,
        required=True,
    )
    parser.add_argument("--source-tree-manifest", type=Path, required=True)
    args = parser.parse_args()
    run_binding = load_run_binding(args.run_root)
    paths = canonical_run_paths(run_binding)
    if (
        not args.output_dir.is_absolute()
        or Path(os.path.abspath(args.output_dir)) != args.output_dir
        or args.output_dir != paths.split_dir
    ):
        raise SystemExit(
            "--output-dir must equal the run binding canonical split_dir"
        )
    if args.source_tree_sha256 != run_binding.included_tree_sha256:
        raise SystemExit(
            "--source-tree-sha256 differs from the run binding"
        )
    if (
        not args.source_tree_manifest.is_absolute()
        or Path(os.path.abspath(args.source_tree_manifest))
        != args.source_tree_manifest
        or args.source_tree_manifest != run_binding.source_manifest
    ):
        raise SystemExit(
            "--source-tree-manifest must equal the run binding source manifest"
        )
    tree_manifest_sha256 = source_tree_manifest_sha256(
        args.source_tree_manifest,
        args.source_tree_sha256,
    )
    manifest = build_split_directory(
        source=args.train,
        output_dir=args.output_dir,
        seed=args.seed,
        source_tree_sha256=args.source_tree_sha256,
        source_tree_manifest_sha256=tree_manifest_sha256,
        run_binding=run_binding,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "manifest": str(args.output_dir / MANIFEST_FILENAME),
                "n_rows": sum(
                    manifest["splits"][role]["n_rows"]
                    for role in SHARD_FILENAMES
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
