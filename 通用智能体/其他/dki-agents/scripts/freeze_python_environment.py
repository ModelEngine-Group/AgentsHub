"""Create-once installed-distribution lock for sealed evaluations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import tempfile
from pathlib import Path


def _publish_once(path: Path, payload: dict) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"environment lock already exists: {path}")
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


def build_lock(source_tree_sha256: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{64}", source_tree_sha256):
        raise ValueError("source-tree-sha256 must be a lowercase SHA-256")
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = str(distribution.metadata.get("Name", "")).strip()
        version = str(distribution.version).strip()
        if name and version:
            packages[name.casefold()] = {"name": name, "version": version}
    ordered = [packages[name] for name in sorted(packages)]
    if "sqlglot" not in packages:
        raise RuntimeError("sqlglot is not installed in the sealing environment")
    canonical = "".join(
        f"{item['name'].casefold()}=={item['version']}\n" for item in ordered
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "kind": "python_environment_lock",
        "created_once": True,
        "source_tree_sha256": source_tree_sha256,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable_sha256": (
                hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
                if Path(sys.executable).is_file()
                else None
            ),
        },
        "platform": platform.platform(),
        "packages": ordered,
        "canonical_packages_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    args = parser.parse_args()
    try:
        payload = build_lock(args.source_tree_sha256.lower())
        _publish_once(args.output, payload)
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "kind": payload["kind"],
                "python": payload["python"]["version"],
                "n_packages": len(payload["packages"]),
                "canonical_packages_sha256": payload[
                    "canonical_packages_sha256"
                ],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
