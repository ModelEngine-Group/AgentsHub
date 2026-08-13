"""Create a create-once local baseline receipt for a dual-gate campaign.

The receipt intentionally records the dirty working tree rather than requiring
it to be clean.  It binds every historical artifact that must remain unchanged
while the CMeIE and NL2SQL gates are being worked on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8", errors="strict"
    )


def _publish_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"baseline receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--included-tree-manifest", type=Path, required=True)
    parser.add_argument("--history", action="append", type=Path, required=True)
    args = parser.parse_args()

    manifest = args.included_tree_manifest
    if not manifest.is_file():
        raise SystemExit(f"included-tree manifest is missing: {manifest}")
    if len({path.resolve() for path in args.history}) != len(args.history):
        raise SystemExit("historical artifact arguments must be unique")
    history: list[dict[str, object]] = []
    for path in args.history:
        if not path.is_file():
            raise SystemExit(f"historical artifact is missing: {path}")
        history.append(
            {
                "path": str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "git_head": _git("rev-parse", "HEAD").strip(),
        "git_status_porcelain_v1": _git(
            "status", "--porcelain=v1", "--untracked-files=all"
        ).splitlines(),
        "included_tree_manifest": {
            "path": str(manifest.resolve().relative_to(ROOT.resolve())).replace("\\", "/"),
            "bytes": manifest.stat().st_size,
            "sha256": _sha256(manifest),
        },
        "historical_hash_guard": history,
        "local_environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
        },
    }
    _publish_once(args.output, payload)
    print(json.dumps({"output": str(args.output), "history_count": len(history)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
