"""Fail-closed clean-room verification for a submission candidate.

Archive structure and every byte binding are checked before any member is
materialized.  The verifier never calls ``ZipFile.extractall`` and rejects
duplicate names, traversal, absolute paths, symbolic links, unrecorded
members, and external/internal manifest drift.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as xml_tree
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


class BundleVerificationError(ValueError):
    """A release archive violated a clean-room integrity contract."""


_SECRET_RE = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"vm-[A-Za-z0-9]{12,}"),
    re.compile(
        r"(?i)(password|passwd|token|api[_-]?key|secret)\s*[:=]\s*['\"][A-Za-z0-9]{8,}['\"]"
    ),
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"tp-[A-Za-z0-9_\-]{16,}"),
]
_SCAN_EXT = {
    ".py",
    ".md",
    ".txt",
    ".yml",
    ".yaml",
    ".json",
    ".jsonl",
    ".ipynb",
    ".ps1",
    ".sh",
    ".toml",
}
_FORBIDDEN_PREFIXES = (
    "outputs/private",
    "data/cblue",
    "data/ood_blind",
    "data/ood_train",
    "models",
    "outputs/models",
    "outputs/checkpoints",
)
_FORBIDDEN_BASENAMES = {
    ".env",
    "holdout.jsonl",
    "detailed_results.json",
    "gold_sql.jsonl",
}

_OPERATOR_ARCHIVE_NAMES = frozenset(
    {
        "medical_entity_extractor.zip",
        "medical_entity_extractor_om_shadow.zip",
        "medical_relation_extractor.zip",
        "medical_section_splitter.zip",
        "medical_text_cleaner.zip",
        "triple_builder_validator.zip",
    }
)
_REQUIRED_OPERATOR_MEMBERS = frozenset(
    {"__init__.py", "metadata.yml", "process.py"}
)

# These exercise the Nexent/smolagents integration and deliberately run in a
# separately provisioned agent environment.  Keeping the list explicit makes
# the non-agent clean-room suite reproducible on the minimal NPU runtime while
# still requiring every agent test to run in its own recorded pass.
AGENT_TEST_PATHS = (
    "tests/test_agents.py",
    "tests/test_insight_agent_nl2sql_e2e.py",
)

# A clean-room verifier must construct the Task 1 -> Task 2 -> Task 3 lineage
# before pytest.  Otherwise database-backed tests can pass by skipping merely
# because a fresh extraction does not yet have the runtime artifacts.  Runtime
# evaluation receipts intentionally live outside ``outputs/eval`` so this step
# cannot overwrite the promoted, claims-bound evaluation evidence in a release.
CLEANROOM_RUNTIME_COMMAND = ("-m", "dki", "demo-all")
CLEANROOM_RUNTIME_ARTIFACTS = (
    "outputs/processed/manifest.json",
    "outputs/kg/medical_kg.json",
    "outputs/kg/medical_kg.provenance.json",
    "outputs/insight.db",
    "outputs/insight.provenance.json",
    "outputs/runtime_eval/kg_extraction_eval.json",
    "outputs/runtime_eval/nl2sql_eval.json",
)
PROMOTED_CLAIM_RECEIPTS = (
    "outputs/eval/kg_extraction_eval.json",
    "outputs/eval/nl2sql_eval.json",
)


def non_agent_pytest_args() -> list[str]:
    """Return the complete non-agent suite with every agent test excluded."""

    return ["tests", *(f"--ignore={path}" for path in AGENT_TEST_PATHS)]


def cleanroom_runtime_command(python: Path) -> list[str]:
    """Return the fixed lineage-build command for a staged clean room."""

    return [str(python), *CLEANROOM_RUNTIME_COMMAND]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(name: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise BundleVerificationError("archive member path is not portable")
    parsed = PurePosixPath(name)
    if parsed.is_absolute() or ".." in parsed.parts or any(part in {"", "."} for part in parsed.parts):
        raise BundleVerificationError(f"unsafe archive member path: {name}")
    if ":" in parsed.parts[0]:
        raise BundleVerificationError(f"drive-qualified archive member path: {name}")
    if parsed.as_posix() != name:
        raise BundleVerificationError(f"non-canonical archive member path: {name}")
    return name


def _forbidden_path(rel: str) -> bool:
    normalized = rel.casefold()
    parsed = PurePosixPath(normalized)
    if parsed.name in {name.casefold() for name in _FORBIDDEN_BASENAMES}:
        return True
    if any(part in {".venv", ".git", "__pycache__", ".pytest_cache"} for part in parsed.parts):
        return True
    return any(
        normalized == prefix.casefold()
        or normalized.startswith(prefix.casefold() + "/")
        for prefix in _FORBIDDEN_PREFIXES
    )


def _member_is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF)


def _scan_bytes(rel: str, data: bytes) -> list[str]:
    if Path(rel).suffix.lower() not in _SCAN_EXT:
        return []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return []
    return [
        f"{rel}: {pattern.pattern[:30]}"
        for pattern in _SECRET_RE
        if pattern.search(text)
    ]


def _canonical_tree_hash(entries: Iterable[Mapping[str, Any]]) -> str:
    canonical = b"".join(
        f"{entry['path']}\0{entry['size']}\0{entry['sha256']}\n".encode("utf-8")
        for entry in sorted(entries, key=lambda item: str(item["path"]))
    )
    return _sha256(canonical)


def _validate_release_bindings(
    manifest: Mapping[str, Any],
    expected: Mapping[str, Mapping[str, Any]],
) -> None:
    """Validate the source and five aggregate-only evidence bindings.

    This verifies the full chain rather than trusting that a schema-2 manifest
    happens to list some files.  The source tree has to cover every payload
    except the source-manifest receipt itself and the five explicitly named
    public acceptance receipts.
    """

    source = manifest.get("source_included_tree")
    evidence = manifest.get("required_public_evidence")
    operators = manifest.get("required_operator_archives", {})
    if not isinstance(source, Mapping) or not isinstance(evidence, Mapping):
        raise BundleVerificationError("submission manifest lacks release bindings")
    if not isinstance(operators, Mapping):
        raise BundleVerificationError("submission operator bindings are malformed")
    required = {
        "candidate_seal",
        "public_freeze_attestation",
        "development_receipt",
        "blind_attestation",
        "consumption_receipt",
    }
    if set(evidence) != required:
        raise BundleVerificationError("submission manifest has incomplete public evidence bindings")

    source_path = source.get("path")
    source_sha = source.get("sha256")
    source_size = source.get("size")
    source_tree_sha = source.get("included_tree_sha256")
    source_entries = source.get("entries")
    if (
        not isinstance(source_path, str)
        or not source_path.startswith("outputs/remote_runs/")
        or not isinstance(source_sha, str)
        or not re.fullmatch(r"[0-9a-f]{64}", source_sha)
        or not isinstance(source_size, int)
        or source_size < 0
        or not isinstance(source_tree_sha, str)
        or not re.fullmatch(r"[0-9a-f]{64}", source_tree_sha)
        or not isinstance(source_entries, list)
    ):
        raise BundleVerificationError("submission source-tree binding is malformed")
    source_entry = expected.get(source_path)
    if source_entry is None or source_entry["sha256"] != source_sha or source_entry["size"] != source_size:
        raise BundleVerificationError("source-tree manifest is not bound to an archive member")

    public_paths: set[str] = set()
    for kind in required:
        item = evidence[kind]
        if not isinstance(item, Mapping):
            raise BundleVerificationError("public evidence binding is malformed")
        path = item.get("path")
        sha = item.get("sha256")
        size = item.get("size")
        if (
            not isinstance(path, str)
            or not path.startswith("outputs/remote_runs/")
            or not isinstance(sha, str)
            or not re.fullmatch(r"[0-9a-f]{64}", sha)
            or not isinstance(size, int)
            or size < 0
            or path in public_paths
        ):
            raise BundleVerificationError("public evidence binding is malformed")
        archive_entry = expected.get(path)
        if archive_entry is None or archive_entry["sha256"] != sha or archive_entry["size"] != size:
            raise BundleVerificationError("public evidence binding is not bound to an archive member")
        public_paths.add(path)

    operator_paths: set[str] = set()
    if operators:
        if set(operators) != _OPERATOR_ARCHIVE_NAMES:
            raise BundleVerificationError("submission has an incomplete operator archive set")
        for name in _OPERATOR_ARCHIVE_NAMES:
            item = operators[name]
            if not isinstance(item, Mapping):
                raise BundleVerificationError("operator archive binding is malformed")
            path = item.get("path")
            sha = item.get("sha256")
            size = item.get("size")
            if (
                path != f"dist/operators/{name}"
                or not isinstance(sha, str)
                or not re.fullmatch(r"[0-9a-f]{64}", sha)
                or not isinstance(size, int)
                or size < 0
                or path in operator_paths
            ):
                raise BundleVerificationError("operator archive binding is malformed")
            archive_entry = expected.get(path)
            if (
                archive_entry is None
                or archive_entry["sha256"] != sha
                or archive_entry["size"] != size
            ):
                raise BundleVerificationError(
                    "operator archive binding is not bound to an archive member"
                )
            operator_paths.add(path)

    source_index: dict[str, Mapping[str, Any]] = {}
    for entry in source_entries:
        if not isinstance(entry, Mapping):
            raise BundleVerificationError("source-tree binding has malformed entries")
        path = entry.get("path")
        sha = entry.get("sha256")
        size = entry.get("size")
        if (
            not isinstance(path, str)
            or not isinstance(sha, str)
            or not re.fullmatch(r"[0-9a-f]{64}", sha)
            or not isinstance(size, int)
            or size < 0
            or path in source_index
        ):
            raise BundleVerificationError("source-tree binding has malformed entries")
        _safe_relative(path)
        source_index[path] = {"path": path, "sha256": sha, "size": size}
    if list(source_index) != sorted(source_index):
        raise BundleVerificationError("source-tree binding entries are not canonically sorted")
    if _canonical_tree_hash(source_index.values()) != source_tree_sha:
        raise BundleVerificationError("source-tree binding aggregate hash is invalid")

    payload_source_paths = set(expected) - public_paths - operator_paths - {source_path}
    if set(source_index) != payload_source_paths:
        raise BundleVerificationError("source-tree binding does not cover the archive payload")
    for path, entry in source_index.items():
        archive_entry = expected[path]
        if archive_entry["sha256"] != entry["sha256"] or archive_entry["size"] != entry["size"]:
            raise BundleVerificationError("source-tree binding differs from archive payload")


def _validate_embedded_operator(rel: str, data: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise BundleVerificationError(
                    f"embedded operator has duplicate members: {rel}"
                )
            for info in infos:
                member = info.filename
                parsed = PurePosixPath(member)
                if (
                    not member
                    or parsed.is_absolute()
                    or ".." in parsed.parts
                    or "\\" in member
                    or parsed.as_posix() != member
                    or info.is_dir()
                    or _member_is_symlink(info)
                ):
                    raise BundleVerificationError(
                        f"embedded operator has an unsafe member: {rel}:{member}"
                    )
            if not _REQUIRED_OPERATOR_MEMBERS.issubset(names):
                raise BundleVerificationError(
                    f"embedded operator lacks required root members: {rel}"
                )
    except zipfile.BadZipFile as exc:
        raise BundleVerificationError(
            f"embedded operator is not a valid ZIP archive: {rel}"
        ) from exc


def validate_archive(
    bundle: Path,
    *,
    external_manifest: Path | None = None,
) -> dict[str, Any]:
    """Authenticate an archive without extracting an untrusted member."""

    bundle = Path(bundle).resolve()
    external_manifest = Path(
        external_manifest or bundle.with_suffix(".manifest.json")
    ).resolve()
    if not bundle.is_file() or not external_manifest.is_file():
        raise BundleVerificationError("bundle or external manifest is missing")
    external_bytes = external_manifest.read_bytes()
    try:
        with zipfile.ZipFile(bundle) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise BundleVerificationError("archive has duplicate member names")
            for info in infos:
                _safe_relative(info.filename)
                if info.is_dir():
                    raise BundleVerificationError("archive may not carry directory members")
                if _member_is_symlink(info):
                    raise BundleVerificationError("archive may not carry symbolic links")

            manifest_member = "dki-agents/SUBMISSION_MANIFEST.json"
            if manifest_member not in names:
                raise BundleVerificationError("archive is missing its internal manifest")
            internal_bytes = archive.read(manifest_member)
            if internal_bytes != external_bytes:
                raise BundleVerificationError("external and internal manifest bytes differ")
            try:
                manifest = json.loads(internal_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise BundleVerificationError("submission manifest is unreadable") from exc

            entries = manifest.get("files")
            if (
                manifest.get("schema_version") != 2
                or not isinstance(entries, list)
                or manifest.get("n_files") != len(entries)
                or not isinstance(manifest.get("included_tree_sha256"), str)
            ):
                raise BundleVerificationError("submission manifest has the wrong schema")

            paths: list[str] = []
            expected: dict[str, Mapping[str, Any]] = {}
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise BundleVerificationError("submission manifest contains a non-object file entry")
                rel = _safe_relative(entry.get("path"))
                if _forbidden_path(rel):
                    raise BundleVerificationError(f"manifest includes forbidden path: {rel}")
                if (
                    rel in expected
                    or not isinstance(entry.get("size"), int)
                    or entry["size"] < 0
                    or not isinstance(entry.get("sha256"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
                ):
                    raise BundleVerificationError("submission manifest has malformed or duplicate entries")
                paths.append(rel)
                expected[rel] = entry
            if paths != sorted(paths):
                raise BundleVerificationError("submission manifest file entries are not canonically sorted")
            if _canonical_tree_hash(entries) != manifest["included_tree_sha256"]:
                raise BundleVerificationError("submission manifest aggregate hash is invalid")
            _validate_release_bindings(manifest, expected)

            expected_members = {manifest_member, *(f"dki-agents/{rel}" for rel in paths)}
            actual_members = set(names)
            if actual_members != expected_members:
                missing = sorted(expected_members - actual_members)
                unexpected = sorted(actual_members - expected_members)
                raise BundleVerificationError(
                    "archive member set differs from manifest "
                    f"(missing={missing[:3]}, unexpected={unexpected[:3]})"
                )

            payloads: list[tuple[str, bytes]] = []
            secret_hits: list[str] = []
            for rel in paths:
                data = archive.read(f"dki-agents/{rel}")
                entry = expected[rel]
                if len(data) != entry["size"] or _sha256(data) != entry["sha256"]:
                    raise BundleVerificationError(f"archive member hash mismatch: {rel}")
                secret_hits.extend(_scan_bytes(rel, data))
                if rel.startswith("dist/operators/"):
                    _validate_embedded_operator(rel, data)
                payloads.append((rel, data))
            if secret_hits:
                raise BundleVerificationError("verified archive contains secret-like text")
    except zipfile.BadZipFile as exc:
        raise BundleVerificationError("bundle is not a valid ZIP archive") from exc

    return {
        "manifest": manifest,
        "manifest_bytes": internal_bytes,
        "external_manifest_sha256": _sha256(external_bytes),
        "bundle_sha256": _sha256_file(bundle),
        "payloads": payloads,
        "secret_scan_hits": [],
    }


def _write_once(path: Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"verification artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _json_once(path: Path, payload: Mapping[str, Any]) -> None:
    _write_once(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _stage_verified_payloads(payloads: Iterable[tuple[str, bytes]], target: Path) -> Path:
    target.mkdir(mode=0o700)
    project = target / "dki-agents"
    for rel, data in payloads:
        safe = _safe_relative(rel)
        destination = project.joinpath(*PurePosixPath(safe).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_once(destination, data)
    return project


def _node_ids(stdout: str) -> list[str]:
    return sorted(
        {
            line.strip()
            for line in stdout.splitlines()
            if line.strip().startswith("tests/") and "::" in line
        }
    )


def _junit_summary(path: Path) -> dict[str, Any]:
    root = xml_tree.fromstring(path.read_bytes())
    cases = list(root.iter("testcase"))
    failures = sum(1 for case in cases if case.find("failure") is not None)
    errors = sum(1 for case in cases if case.find("error") is not None)
    skipped = [
        f"{case.get('classname', '')}::{case.get('name', '')}"
        for case in cases
        if case.find("skipped") is not None
    ]
    return {
        "n_testcases": len(cases),
        "failures": failures,
        "errors": errors,
        "skipped_ids": sorted(skipped),
    }


def _run_pytest_evidence(
    *,
    python: Path,
    cwd: Path,
    label: str,
    pytest_args: list[str],
    evidence_dir: Path,
    pythonpath: Path | None,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath.resolve())
    base = [str(python), "-m", "pytest", "-q", "-rs", "-o", "addopts="]
    collect = subprocess.run(
        [*base, "--collect-only", *pytest_args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    _write_once(evidence_dir / f"{label}.collect.stdout.log", collect.stdout.encode("utf-8"))
    _write_once(evidence_dir / f"{label}.collect.stderr.log", collect.stderr.encode("utf-8"))
    junit = evidence_dir / f"{label}.junit.xml"
    proc = subprocess.run(
        [*base, f"--junitxml={junit}", *pytest_args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    _write_once(evidence_dir / f"{label}.pytest.stdout.log", proc.stdout.encode("utf-8"))
    _write_once(evidence_dir / f"{label}.pytest.stderr.log", proc.stderr.encode("utf-8"))
    if not junit.is_file():
        raise BundleVerificationError(f"pytest did not create JUnit XML for {label}")
    summary = _junit_summary(junit)
    return {
        "label": label,
        "collect_returncode": collect.returncode,
        "returncode": proc.returncode,
        "node_ids": _node_ids(collect.stdout),
        "junit": str(junit),
        "junit_sha256": _sha256_file(junit),
        "stdout": str(evidence_dir / f"{label}.pytest.stdout.log"),
        "stderr": str(evidence_dir / f"{label}.pytest.stderr.log"),
        "collect_stdout": str(evidence_dir / f"{label}.collect.stdout.log"),
        "collect_stderr": str(evidence_dir / f"{label}.collect.stderr.log"),
        "xpass_detected": "XPASS" in proc.stdout or "XPASS" in proc.stderr,
        **summary,
    }


def _run_cleanroom_runtime_evidence(
    *,
    python: Path,
    cwd: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    """Build and validate runtime lineage before clean-room test collection.

    The command is deliberately fixed rather than caller-supplied.  It records
    complete stdout/stderr, rejects missing lineage artifacts, and fail-closes
    if runtime evaluation changes either promoted claims receipt bundled in the
    candidate.
    """

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONUTF8"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"

    before_claims: dict[str, str] = {}
    for rel in PROMOTED_CLAIM_RECEIPTS:
        receipt = cwd / rel
        if not receipt.is_file():
            raise BundleVerificationError(
                f"clean-room candidate lacks promoted claim receipt: {rel}"
            )
        before_claims[rel] = _sha256_file(receipt)

    command = cleanroom_runtime_command(python)
    runtime = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    runtime_stdout = evidence_dir / "runtime.demo_all.stdout.log"
    runtime_stderr = evidence_dir / "runtime.demo_all.stderr.log"
    _write_once(runtime_stdout, runtime.stdout.encode("utf-8"))
    _write_once(runtime_stderr, runtime.stderr.encode("utf-8"))
    if runtime.returncode != 0:
        raise BundleVerificationError("clean-room demo-all failed")

    artifacts: list[dict[str, Any]] = []
    for rel in CLEANROOM_RUNTIME_ARTIFACTS:
        artifact = cwd / rel
        if not artifact.is_file():
            raise BundleVerificationError(f"clean-room runtime artifact is missing: {rel}")
        artifacts.append(
            {
                "path": rel,
                "size": artifact.stat().st_size,
                "sha256": _sha256_file(artifact),
            }
        )

    changed_claims = [
        rel
        for rel, before in before_claims.items()
        if _sha256_file(cwd / rel) != before
    ]
    if changed_claims:
        raise BundleVerificationError(
            "clean-room runtime changed promoted claim evidence: "
            + ", ".join(sorted(changed_claims))
        )

    lineage_code = "\n".join(
        (
            "from pathlib import Path",
            "from dki.analysis.database import database_matches_manifest",
            "from dki.pipeline.artifacts import manifest_path, validate_task2_artifact",
            "root = Path('outputs')",
            "kg = root / 'kg' / 'medical_kg.json'",
            "validate_task2_artifact(kg, task1_path=manifest_path())",
            "assert database_matches_manifest(root / 'insight.db', manifest_path(), kg)",
            "print('cleanroom-lineage-ok')",
        )
    )
    lineage = subprocess.run(
        [str(python), "-c", lineage_code],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    lineage_stdout = evidence_dir / "runtime.lineage.stdout.log"
    lineage_stderr = evidence_dir / "runtime.lineage.stderr.log"
    _write_once(lineage_stdout, lineage.stdout.encode("utf-8"))
    _write_once(lineage_stderr, lineage.stderr.encode("utf-8"))
    if lineage.returncode != 0 or "cleanroom-lineage-ok" not in lineage.stdout:
        raise BundleVerificationError("clean-room runtime lineage validation failed")

    return {
        "command": command,
        "returncode": runtime.returncode,
        "stdout": str(runtime_stdout),
        "stderr": str(runtime_stderr),
        "lineage_command": [str(python), "-c", lineage_code],
        "lineage_returncode": lineage.returncode,
        "lineage_stdout": str(lineage_stdout),
        "lineage_stderr": str(lineage_stderr),
        "artifacts": artifacts,
        "promoted_claim_receipts_unchanged": True,
    }


def _load_baseline_ids(path: Path, *, field: str) -> set[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, Mapping):
        values = payload.get(field, [])
    else:
        raise BundleVerificationError(f"baseline {field} has an invalid format")
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise BundleVerificationError(f"baseline {field} must be a string list")
    return set(values)


def main() -> int:
    parser = argparse.ArgumentParser(description="verify a clean-room release candidate")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--agent-python",
        type=Path,
        help="optional separately provisioned Python executable for agent dependency tests",
    )
    parser.add_argument("--agent-deps", type=Path)
    parser.add_argument("--baseline-nodes", type=Path, required=True)
    parser.add_argument("--baseline-skips", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    evidence_dir = output.with_suffix(".evidence")
    if output.exists() or evidence_dir.exists():
        raise SystemExit("verification receipt or evidence directory already exists")
    try:
        snapshot = validate_archive(args.bundle, external_manifest=args.external_manifest)
        baseline_nodes = _load_baseline_ids(args.baseline_nodes, field="node_ids")
        baseline_skips = _load_baseline_ids(args.baseline_skips, field="skipped_ids")
        evidence_dir.mkdir(parents=True)
        stage_root = Path(tempfile.mkdtemp(prefix="dki_cleanroom_"))
        try:
            extracted = _stage_verified_payloads(snapshot["payloads"], stage_root)
            runtime_lineage = _run_cleanroom_runtime_evidence(
                python=args.python,
                cwd=extracted,
                evidence_dir=evidence_dir,
            )
            runs: list[dict[str, Any]] = []
            if args.agent_deps is not None or args.agent_python is not None:
                runs.append(
                    _run_pytest_evidence(
                        python=args.python,
                        cwd=extracted,
                        label="non_agent",
                        pytest_args=non_agent_pytest_args(),
                        evidence_dir=evidence_dir,
                        pythonpath=None,
                    )
                )
                runs.append(
                    _run_pytest_evidence(
                        python=args.agent_python or args.python,
                        cwd=extracted,
                        label="agent",
                        pytest_args=list(AGENT_TEST_PATHS),
                        evidence_dir=evidence_dir,
                        pythonpath=args.agent_deps,
                    )
                )
            else:
                runs.append(
                    _run_pytest_evidence(
                        python=args.python,
                        cwd=extracted,
                        label="all",
                        pytest_args=[],
                        evidence_dir=evidence_dir,
                        pythonpath=None,
                    )
                )
        finally:
            shutil.rmtree(stage_root, ignore_errors=True)
    except (BundleVerificationError, FileNotFoundError, OSError, ValueError) as exc:
        if evidence_dir.exists() and not output.exists():
            failure = {
                "schema_version": 2,
                "status": "failed_preflight",
                "passed": False,
                "error_class": type(exc).__name__,
                "bundle": str(args.bundle),
                "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            _json_once(output, failure)
        raise SystemExit(str(exc)) from exc

    current_nodes = set().union(*(set(run["node_ids"]) for run in runs))
    current_skips = set().union(*(set(run["skipped_ids"]) for run in runs))
    missing_nodes = sorted(baseline_nodes - current_nodes)
    new_skips = sorted(current_skips - baseline_skips)
    failures = sum(int(run["failures"]) for run in runs)
    errors = sum(int(run["errors"]) for run in runs)
    xpasses = any(bool(run["xpass_detected"]) for run in runs)
    returncodes = [int(run["collect_returncode"]) for run in runs] + [int(run["returncode"]) for run in runs]
    receipt = {
        "schema_version": 2,
        "status": "verified" if not (failures or errors or xpasses or missing_nodes or new_skips or any(returncodes)) else "failed_tests",
        "passed": not (failures or errors or xpasses or missing_nodes or new_skips or any(returncodes)),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bundle": str(Path(args.bundle).resolve()),
        "bundle_sha256": snapshot["bundle_sha256"],
        "external_manifest_sha256": snapshot["external_manifest_sha256"],
        "manifest_files": snapshot["manifest"]["n_files"],
        "manifest_included_tree_sha256": snapshot["manifest"]["included_tree_sha256"],
        "secret_scan_hits": snapshot["secret_scan_hits"],
        "runtime_lineage": runtime_lineage,
        "pytest": runs,
        "missing_baseline_node_ids": missing_nodes,
        "new_skip_ids": new_skips,
        "failure_count": failures,
        "error_count": errors,
        "xpass_detected": xpasses,
        "evidence_dir": str(evidence_dir),
    }
    _json_once(output, receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "passed", "failure_count", "error_count", "xpass_detected", "missing_baseline_node_ids", "new_skip_ids")}, ensure_ascii=False, sort_keys=True))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
