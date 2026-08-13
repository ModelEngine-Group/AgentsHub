"""Run one remote command and retain a complete create-once execution receipt.

The experiment commands in this project intentionally run on the Ascend host,
but the command itself should not need to reimplement evidence hygiene.  This
wrapper captures the exact argv, complete stdout/stderr, exit status, runtime
fingerprint and a before/after NPU snapshot in a directory that can only be
created once.  It never invokes a shell and fixes visibility to logical
``npu:0`` without touching other processes on the host.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"execution evidence artifact already exists: {path}")
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _json_once(path: Path, payload: dict[str, object]) -> None:
    _write_once(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _environment() -> dict[str, object]:
    packages: dict[str, str] = {}
    for name in ("torch", "torch_npu", "transformers", "sqlglot"):
        try:
            module = importlib.import_module(name)
            packages[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:  # Capture absence rather than hiding it.
            packages[name] = f"unavailable:{type(exc).__name__}"
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }


def _run(argv: list[str], environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=ROOT,
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("pass one command after --")
    if args.device != "npu:0":
        raise SystemExit("this receipt wrapper is restricted to the planned npu:0 device")

    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"execution evidence output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # mkdir is the create-once claim.  Every payload below is separately
    # opened with O_EXCL, so a retried RID cannot replace partial evidence.
    output.mkdir(exist_ok=False)

    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["ASCEND_RT_VISIBLE_DEVICES"] = "0"
    environment["DKI_NPU_DEVICE"] = "npu:0"
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _json_once(
        output / "started.json",
        {
            "schema_version": 1,
            "rid": args.rid,
            "started_at": started,
            "command": command,
            "cwd": str(ROOT),
            "device": "npu:0",
            "environment": _environment(),
        },
    )

    before = _run(["npu-smi", "info"], environment)
    _write_once(output / "npu-smi.before.stdout.log", before.stdout.encode("utf-8"))
    _write_once(output / "npu-smi.before.stderr.log", before.stderr.encode("utf-8"))

    result = _run(command, environment)
    _write_once(output / "command.stdout.log", result.stdout.encode("utf-8"))
    _write_once(output / "command.stderr.log", result.stderr.encode("utf-8"))

    after = _run(["npu-smi", "info"], environment)
    _write_once(output / "npu-smi.after.stdout.log", after.stdout.encode("utf-8"))
    _write_once(output / "npu-smi.after.stderr.log", after.stderr.encode("utf-8"))
    _json_once(
        output / "verification.json",
        {
            "schema_version": 1,
            "rid": args.rid,
            "started_at": started,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "command": command,
            "returncode": result.returncode,
            "npu_smi_before_returncode": before.returncode,
            "npu_smi_after_returncode": after.returncode,
            "environment": _environment(),
        },
    )
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
