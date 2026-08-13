"""Run a remote pytest selection and preserve complete create-once evidence.

The command is intentionally neutral about test selection: callers choose a
focused protocol suite or the eventual full suite, while this wrapper records
collect node IDs, full streams, JUnit XML, skip information, runtime versions,
and the exact invocation in one versioned output directory.
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
import xml.etree.ElementTree as xml_tree
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _write_once(path: Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"evidence artifact already exists: {path}")
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _json_once(path: Path, payload: dict) -> None:
    _write_once(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _environment() -> dict:
    packages = {}
    for name in ("torch", "torch_npu", "sqlglot", "pytest", "transformers"):
        try:
            module = importlib.import_module(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # report absence rather than hiding it
            packages[name] = f"unavailable:{type(exc).__name__}"
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }


def _run(command: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )


def _junit_skips(path: Path) -> list[dict[str, str]]:
    """Return deterministic skip node/reason evidence from pytest JUnit XML."""

    try:
        root = xml_tree.fromstring(path.read_bytes())
    except (OSError, xml_tree.ParseError) as exc:
        raise RuntimeError("pytest did not produce readable JUnit XML") from exc
    skipped: list[dict[str, str]] = []
    for case in root.iter("testcase"):
        marker = case.find("skipped")
        if marker is None:
            continue
        classname = str(case.get("classname", ""))
        name = str(case.get("name", ""))
        skipped.append(
            {
                "node_id": f"{classname}::{name}",
                "reason": str(marker.get("message", "")),
            }
        )
    return sorted(skipped, key=lambda item: (item["node_id"], item["reason"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("tests", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    tests = args.tests[1:] if args.tests[:1] == ["--"] else args.tests
    if not tests:
        raise SystemExit("pass one or more pytest paths/node IDs after --")
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"remote test evidence output exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["ASCEND_RT_VISIBLE_DEVICES"] = "0"
    environment["DKI_NPU_DEVICE"] = args.device
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    collect_command = [
        sys.executable,
        "-m",
        "pytest",
        "-o",
        "addopts=",
        "--collect-only",
        "-q",
        *tests,
    ]
    collect = _run(collect_command, env=environment)
    _write_once(output / "collect.stdout.log", collect.stdout.encode("utf-8"))
    _write_once(output / "collect.stderr.log", collect.stderr.encode("utf-8"))
    node_ids = [
        line.strip()
        for line in collect.stdout.splitlines()
        if "::" in line and not line.lstrip().startswith("<")
    ]
    _json_once(
        output / "collect.json",
        {"command": collect_command, "returncode": collect.returncode, "node_ids": node_ids},
    )
    npu_smi = _run(["npu-smi", "info"], env=environment)
    _write_once(output / "npu-smi.stdout.log", npu_smi.stdout.encode("utf-8"))
    _write_once(output / "npu-smi.stderr.log", npu_smi.stderr.encode("utf-8"))
    junit = output / "junit.xml"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-rs",
        f"--junitxml={junit}",
        *tests,
    ]
    result = _run(command, env=environment)
    _write_once(output / "pytest.stdout.log", result.stdout.encode("utf-8"))
    _write_once(output / "pytest.stderr.log", result.stderr.encode("utf-8"))
    skip_lines = [line for line in result.stdout.splitlines() if "SKIPPED" in line]
    skipped = _junit_skips(junit)
    _json_once(
        output / "verification.json",
        {
            "schema_version": 1,
            "rid": args.rid,
            "started_at": started,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "command": command,
            "returncode": result.returncode,
            "collect_returncode": collect.returncode,
            "node_ids": node_ids,
            "skip_lines": skip_lines,
            "skipped": skipped,
            "environment": _environment(),
            "npu_smi_returncode": npu_smi.returncode,
        },
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
