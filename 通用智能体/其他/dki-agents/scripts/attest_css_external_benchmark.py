"""Publish an aggregate-only attestation for a frozen private CSS benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"CSS public attestation already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def attest(
    manifest_path: Path,
    output: Path,
    *,
    source_tree_sha256: str,
) -> dict[str, Any]:
    if len(source_tree_sha256) != 64:
        raise ValueError("source tree identity must be a full SHA-256")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("kind")
        != "external_authoritative_nl2sql_benchmark_freeze"
        or manifest.get("benchmark") != "CSS"
    ):
        raise ValueError("not a supported private CSS freeze manifest")
    if manifest.get("four_table_blind_gate") is not False:
        raise ValueError("CSS may not be relabelled as the four-table blind gate")
    if manifest.get("holdout_consumed") is not False:
        raise ValueError("expected an unconsumed CSS holdout freeze")
    intersections = manifest.get("database_id_intersections")
    if not isinstance(intersections, dict) or any(intersections.values()):
        raise ValueError("CSS freeze does not prove schema-disjoint splits")

    base = manifest_path.resolve().parent
    verified_files = {}
    for relative, expected in manifest.get("files", {}).items():
        rel = PurePosixPath(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"unsafe private manifest path: {relative}")
        path = base.joinpath(*rel.parts)
        if not path.is_file():
            raise ValueError(f"private CSS file is absent: {relative}")
        actual_sha = _sha256(path)
        actual_size = path.stat().st_size
        if (
            actual_sha != expected.get("sha256")
            or actual_size != expected.get("size")
        ):
            raise ValueError(f"private CSS file differs from freeze: {relative}")
        verified_files[relative] = {
            "sha256": actual_sha,
            "size": actual_size,
        }

    payload = {
        "schema_version": 1,
        "kind": "public_aggregate_css_freeze_attestation",
        "benchmark": "CSS",
        "dataset_role": manifest["dataset_role"],
        "four_table_blind_gate": False,
        "holdout_consumed": False,
        "source_tree_sha256": source_tree_sha256,
        "freeze_manifest_sha256": _sha256(manifest_path),
        "source": manifest["source"],
        "license_evidence": manifest["license_evidence"],
        "official_split": manifest["official_split"],
        "counts": manifest["counts"],
        "database_id_intersections": intersections,
        "file_bindings": verified_files,
        "privacy": {
            "contains_questions": False,
            "contains_gold_sql": False,
            "contains_predictions": False,
            "contains_error_samples": False,
            "private_dataset_redistribution": False,
        },
        "attester_sha256": _sha256(Path(__file__)),
    }
    _publish_once(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    args = parser.parse_args()
    if "outputs/private/" not in args.manifest.resolve().as_posix():
        raise SystemExit("CSS freeze manifest must remain private")
    if "outputs/private/" in args.output.resolve().as_posix():
        raise SystemExit("aggregate attestation must be outside outputs/private")
    try:
        payload = attest(
            args.manifest,
            args.output,
            source_tree_sha256=args.source_tree_sha256,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "output": str(args.output),
                "benchmark": payload["benchmark"],
                "counts": payload["counts"],
                "holdout_consumed": False,
                "four_table_blind_gate": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
