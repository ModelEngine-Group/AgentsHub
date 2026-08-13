"""Build a create-once, clean-room submission candidate.

The builder is deliberately allowlist based.  A release build must name its
RID, bind an included-source-tree manifest, and explicitly supply the public
NL2SQL release attestations that it intends to carry. Private corpora,
checkpoints, CBLUE raw data, secrets, and remote transfer archives never enter
the archive.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parent.parent

INCLUDE = [
    "dki",
    "scripts",
    "configs",
    "tests",
    "docs",
    "notebooks/dki_end_to_end_demo.ipynb",
    "notebooks/README.md",
    "evidence/manifests",
    "data/raw",
    "data/gold",
    "data/nl2sql_eval.jsonl",
    "data/nl2sql_eval_hard.jsonl",
    "data/ood/ood_eval.jsonl",
    "outputs/claims",
    "outputs/eval",
    "outputs/ner_finetune",
    "outputs/verification",
    "README.md",
    "README_CN.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "CONTRACTS.md",
    "pyproject.toml",
    "run.ps1",
    "run.sh",
    ".env.example",
    ".gitignore",
]

INCLUDE_REMOTE_GLOBS = [
    "outputs/remote_acceptance_20260720_*.json",
    "outputs/remote_acceptance_20260721_*.json",
    "outputs/remote_acceptance_20260723_*.json",
    "outputs/remote_*_20260720.py",
]

EXCLUDE_GLOBS = [
    "*/__pycache__/*",
    "*.pyc",
    "*.pyo",
    "*/.pytest_cache/*",
    "*/.ipynb_checkpoints/*",
    "**/*.tgz",
    "**/*.tar",
    "**/*.tar.gz",
    "**/*.om",
    "**/*.onnx",
    "**/*.bin",
    "**/*.safetensors",
    "**/*.pt",
    "**/*.pth",
    "**/*.ckpt",
]

HARD_EXCLUDE = {
    ".env",
    "HANDOFF_NPU_GPT5.md",
    "_probe_npu_server.py",
    "_verify_datamate.py",
    "holdout.jsonl",
    "detailed_results.json",
    "gold_sql.jsonl",
}
HARD_EXCLUDE_DIR_PARTS = {
    ".venv",
    "models",
    "checkpoints",
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".ipynb_checkpoints",
}
HARD_EXCLUDE_PATH_PREFIXES = (
    "data/ood_blind",
    "data/ood_train",
    "data/cblue",
    "outputs/private",
    "outputs/demo_runs",
    "outputs/cblue",
    "outputs/models",
    "outputs/checkpoints",
)

REQUIRED_PUBLIC_EVIDENCE = (
    "candidate_seal",
    "public_freeze_attestation",
    "development_receipt",
    "blind_attestation",
    "consumption_receipt",
)

OPERATOR_ARCHIVE_NAMES = (
    "medical_entity_extractor.zip",
    "medical_entity_extractor_om_shadow.zip",
    "medical_relation_extractor.zip",
    "medical_section_splitter.zip",
    "medical_text_cleaner.zip",
    "triple_builder_validator.zip",
)

_REQUIRED_OPERATOR_MEMBERS = frozenset(
    {"__init__.py", "metadata.yml", "process.py"}
)

# Public release receipts are aggregate-only.  These are field names that
# would disclose a private blind item or a model answer if they occurred at
# any depth in one of the five explicit evidence documents.  Hashes and
# aggregate counters remain allowed.
_PRIVATE_EVIDENCE_KEYS = {
    "question",
    "questions",
    "gold_sql",
    "goldsql",
    "prediction_sql",
    "predicted_sql",
    "prediction",
    "predictions",
    "detailed_result",
    "detailed_results",
    "error_sample",
    "error_samples",
    "failed_examples",
}

# The typed public claims currently bind exactly these two small aggregate
# metric receipts. They contain no CBLUE examples, labels, model weights, or
# checkpoints. Keep the broad ``outputs/cblue`` exclusion below and make this
# exception explicit so a clean-room package can verify every bundled claim
# without redistributing the CBLUE corpus.
PUBLIC_AGGREGATE_CBLUE_RECEIPTS = frozenset(
    {
        "outputs/cblue/cmeee_metrics.json",
        "outputs/cblue/cmeie_metrics.json",
    }
)

_SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"vm-[A-Za-z0-9]{12,}"),
    re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
    re.compile(
        r"(?i)(password|passwd|secret|token|api[_-]?key)\s*[:=]\s*['\"][A-Za-z0-9]{8,}['\"]"
    ),
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"tp-[A-Za-z0-9_\-]{16,}"),
]
_IP_ALLOW = {"127.0.0.1", "0.0.0.0", "255.255.255.255"}
_SECRET_SCAN_EXT = {
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
    ".cfg",
    ".ini",
    "",
}
RID_RE = re.compile(r"[0-9]{8}T[0-9]{6}Z-g[0-9a-f]{7}-t[0-9a-f]{12}")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_rel(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"included path is outside the submission root: {path}") from exc
    rel = relative.as_posix()
    parsed = PurePosixPath(rel)
    if not rel or parsed.is_absolute() or ".." in parsed.parts or "\\" in rel:
        raise ValueError(f"unsafe submission path: {rel}")
    return rel


def _is_hard_excluded(rel: str) -> bool:
    normalized = rel.casefold()
    if normalized in {
        candidate.casefold() for candidate in PUBLIC_AGGREGATE_CBLUE_RECEIPTS
    }:
        return False
    parsed = PurePosixPath(normalized)
    if set(parsed.parts) & {part.casefold() for part in HARD_EXCLUDE_DIR_PARTS}:
        return True
    if parsed.name in {name.casefold() for name in HARD_EXCLUDE}:
        return True
    return any(
        normalized == prefix.casefold()
        or normalized.startswith(prefix.casefold() + "/")
        for prefix in HARD_EXCLUDE_PATH_PREFIXES
    )


def _is_glob_excluded(rel: str) -> bool:
    normalized = rel.casefold()
    return any(fnmatch.fnmatch(normalized, pattern.casefold()) for pattern in EXCLUDE_GLOBS)


def _scan_secrets(rel: str, data: bytes) -> list[str]:
    if Path(rel).suffix.lower() not in _SECRET_SCAN_EXT:
        return []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return []
    hits: list[str] = []
    for pattern in _SECRET_PATTERNS:
        for match in pattern.finditer(text):
            token = match.group(0)
            if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", token):
                if token in _IP_ALLOW or any(int(part) > 255 for part in token.split(".")):
                    continue
            hits.append(f"{rel}: {pattern.pattern[:32]} -> {token[:24]}")
    return hits


def _iter_tree_files(path: Path, root: Path) -> Iterable[Path]:
    if path.is_symlink():
        raise ValueError(f"submission input may not be a symbolic link: {_safe_rel(path, root)}")
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        return
    for candidate in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
        if candidate.is_symlink():
            raise ValueError(
                f"submission input may not contain a symbolic link: {_safe_rel(candidate, root)}"
            )
        if candidate.is_file():
            yield candidate


def _iter_include_files(
    root: Path,
    *,
    public_evidence: Iterable[Path] = (),
) -> list[Path]:
    root = root.resolve()
    files: list[Path] = []
    for entry in (*INCLUDE, *sorted(PUBLIC_AGGREGATE_CBLUE_RECEIPTS)):
        files.extend(_iter_tree_files(root / entry, root))
    for pattern in INCLUDE_REMOTE_GLOBS:
        for path in sorted(root.glob(pattern), key=lambda item: item.as_posix()):
            files.extend(_iter_tree_files(path, root))
    for path in public_evidence:
        files.extend(_iter_tree_files(path, root))

    selected: dict[str, Path] = {}
    selected_identities: set[str] = set()
    for path in files:
        rel = _safe_rel(path, root)
        if rel == "SUBMISSION_MANIFEST.json":
            raise ValueError("source tree may not predefine SUBMISSION_MANIFEST.json")
        if _is_hard_excluded(rel) or _is_glob_excluded(rel):
            continue
        identity = rel.casefold()
        if identity in selected_identities and rel not in selected:
            raise ValueError(f"case-alias submission path: {rel}")
        previous = selected.get(rel)
        if previous is not None and previous != path:
            raise ValueError(f"duplicate submission path: {rel}")
        selected[rel] = path
        selected_identities.add(identity)
    return [selected[rel] for rel in sorted(selected)]


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    head = completed.stdout.strip().lower()
    if completed.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("submission root must provide a full Git HEAD")
    return head


def _load_source_tree_binding(path: Path, root: Path) -> dict[str, object]:
    rel = _safe_rel(path, root)
    if not rel.startswith("outputs/remote_runs/"):
        raise ValueError("included-tree manifest must be public remote-run evidence")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("included-tree manifest is unreadable") from exc
    tree = payload.get("included_tree_sha256")
    head = payload.get("git_head")
    entries = payload.get("entries")
    if (
        not isinstance(tree, str)
        or not re.fullmatch(r"[0-9a-f]{64}", tree)
        or not isinstance(head, str)
        or not re.fullmatch(r"[0-9a-f]{40}", head)
        or payload.get("schema_version") != 1
        or not isinstance(entries, list)
    ):
        raise ValueError("included-tree manifest has an invalid contract")
    canonical_entries: list[dict[str, object]] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("included-tree manifest has a malformed entry")
        entry_path = entry.get("path")
        entry_size = entry.get("size")
        entry_sha = entry.get("sha256")
        if (
            not isinstance(entry_path, str)
            or not isinstance(entry_size, int)
            or entry_size < 0
            or not isinstance(entry_sha, str)
            or not re.fullmatch(r"[0-9a-f]{64}", entry_sha)
        ):
            raise ValueError("included-tree manifest has a malformed entry")
        parsed = PurePosixPath(entry_path)
        if (
            not entry_path
            or parsed.is_absolute()
            or ".." in parsed.parts
            or "\\" in entry_path
            or parsed.as_posix() != entry_path
            or entry_path in seen
        ):
            raise ValueError("included-tree manifest has unsafe or duplicate paths")
        seen.add(entry_path)
        canonical_entries.append(
            {"path": entry_path, "size": entry_size, "sha256": entry_sha}
        )
    if canonical_entries != sorted(canonical_entries, key=lambda entry: str(entry["path"])):
        raise ValueError("included-tree manifest entries are not canonically sorted")
    if _package_tree_sha(canonical_entries) != tree:
        raise ValueError("included-tree manifest aggregate hash is invalid")
    return {
        "path": rel,
        "sha256": _sha256_file(path),
        "size": path.stat().st_size,
        "included_tree_sha256": tree,
        "git_head": head,
        "entries": canonical_entries,
    }


def _validate_public_evidence_payload(kind: str, path: Path) -> None:
    """Ensure an explicit public receipt cannot smuggle private blind rows."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"required public evidence is not a UTF-8 JSON receipt: {kind}") from exc

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"required public evidence has a non-text field: {kind}")
                normalized = key.casefold().replace("-", "_").replace(" ", "_")
                if normalized in _PRIVATE_EVIDENCE_KEYS:
                    raise ValueError(
                        f"required public evidence exposes a private blind field: {kind}:{key}"
                    )
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    if not isinstance(payload, dict):
        raise ValueError(f"required public evidence must be a JSON object: {kind}")
    walk(payload)


def _validate_public_evidence(
    evidence: Mapping[str, Path], root: Path
) -> dict[str, dict[str, object]]:
    if set(evidence) != set(REQUIRED_PUBLIC_EVIDENCE):
        raise ValueError("release evidence names do not match the required contract")
    binding: dict[str, dict[str, object]] = {}
    seen: set[str] = set()
    for kind in REQUIRED_PUBLIC_EVIDENCE:
        path = Path(evidence[kind])
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"required public evidence is missing or unsafe: {kind}")
        rel = _safe_rel(path, root)
        if not rel.startswith("outputs/remote_runs/"):
            raise ValueError(f"required evidence must be public remote-run evidence: {kind}")
        if _is_hard_excluded(rel) or _is_glob_excluded(rel) or rel in seen:
            raise ValueError(f"required public evidence is excluded or duplicated: {kind}")
        _validate_public_evidence_payload(kind, path)
        seen.add(rel)
        binding[kind] = {
            "path": rel,
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
        }
    return binding


def _validate_operator_zip(name: str, data: bytes) -> None:
    """Authenticate one standalone DataMate archive before nesting it in the RC."""

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
            members = [info.filename for info in infos]
            if len(members) != len(set(members)):
                raise ValueError(f"operator archive has duplicate members: {name}")
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
                    or stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF)
                ):
                    raise ValueError(f"operator archive has an unsafe member: {name}:{member}")
            missing = sorted(_REQUIRED_OPERATOR_MEMBERS - set(members))
            if missing:
                raise ValueError(
                    f"operator archive lacks required root members: {name}:{missing}"
                )
    except zipfile.BadZipFile as exc:
        raise ValueError(f"operator archive is not a valid ZIP: {name}") from exc


def _load_operator_archives(
    operator_dir: Path | None,
) -> tuple[list[tuple[str, bytes]], dict[str, dict[str, object]]]:
    """Load exactly six create-once RID operator outputs as release inputs."""

    if operator_dir is None:
        return [], {}
    directory = Path(operator_dir).resolve()
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("operator archive directory is missing or unsafe")
    children = sorted(directory.iterdir(), key=lambda item: item.name)
    if any(child.is_symlink() or not child.is_file() for child in children):
        raise ValueError("operator archive directory may contain regular files only")
    observed = [child.name for child in children]
    if observed != list(OPERATOR_ARCHIVE_NAMES):
        raise ValueError(
            "operator archive set does not match the required six packages "
            f"(observed={observed})"
        )

    staged: list[tuple[str, bytes]] = []
    binding: dict[str, dict[str, object]] = {}
    for path in children:
        data = path.read_bytes()
        _validate_operator_zip(path.name, data)
        rel = f"dist/operators/{path.name}"
        digest = _sha256_bytes(data)
        staged.append((rel, data))
        binding[path.name] = {"path": rel, "sha256": digest, "size": len(data)}
    return staged, binding


def _validate_source_tree_coverage(
    tree_binding: Mapping[str, object] | None,
    files: Iterable[Path],
    root: Path,
    *,
    evidence_paths: Iterable[Path],
) -> None:
    """Bind every non-evidence bundle byte to the supplied source tree.

    A release may carry public receipts that were generated during acceptance,
    but all code, docs, fixtures, and tracked/untracked source payload must
    exactly match the immutable included-tree manifest.  A partial manifest is
    therefore not sufficient for a submission candidate.
    """

    if tree_binding is None:
        return
    evidence_rels = {_safe_rel(path.resolve(), root) for path in evidence_paths}
    manifest_rel = str(tree_binding["path"])
    source_entries = tree_binding["entries"]
    assert isinstance(source_entries, list)
    expected = {str(entry["path"]): entry for entry in source_entries}
    actual: dict[str, dict[str, object]] = {}
    for path in files:
        rel = _safe_rel(path, root)
        if rel in evidence_rels or rel == manifest_rel:
            continue
        data = path.read_bytes()
        actual[rel] = {"path": rel, "size": len(data), "sha256": _sha256_bytes(data)}
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(
            path
            for path in set(expected) & set(actual)
            if expected[path] != actual[path]
        )
        raise ValueError(
            "included-tree manifest does not exactly bind the release source payload "
            f"(missing={missing[:3]}, extra={extra[:3]}, changed={changed[:3]})"
        )


def _publish_bytes_once(path: Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"submission artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _package_tree_sha(entries: list[dict[str, object]]) -> str:
    canonical = b"".join(
        f"{entry['path']}\0{entry['size']}\0{entry['sha256']}\n".encode("utf-8")
        for entry in sorted(entries, key=lambda item: str(item["path"]))
    )
    return _sha256_bytes(canonical)


def build(
    output: Path,
    root: Path,
    *,
    included_tree_manifest: Path | None = None,
    public_evidence: Mapping[str, Path] | None = None,
    operator_dir: Path | None = None,
    git_head: str | None = None,
) -> dict[str, object]:
    """Build one immutable archive.

    ``included_tree_manifest`` and ``public_evidence`` are optional only for
    isolated unit tests.  The command-line release entry point requires them.
    """

    root = Path(root).resolve()
    output = Path(output).resolve()
    manifest_output = output.with_suffix(".manifest.json")
    sums_output = output.parent / "SHA256SUMS.txt"
    existing = [
        str(path)
        for path in (output, manifest_output, sums_output)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "submission artifacts already exist; choose a new RID directory:\n  "
            + "\n  ".join(existing)
        )

    tree_binding = None
    tree_path: Path | None = None
    evidence_binding: dict[str, dict[str, object]] = {}
    additional: list[Path] = []
    if included_tree_manifest is not None:
        tree_path = Path(included_tree_manifest).resolve()
        tree_binding = _load_source_tree_binding(tree_path, root)
        additional.append(tree_path)
    if public_evidence is not None:
        evidence_paths = {name: Path(path).resolve() for name, path in public_evidence.items()}
        evidence_binding = _validate_public_evidence(evidence_paths, root)
        if tree_path is not None and tree_path in evidence_paths.values():
            raise ValueError("included-tree manifest must not double as public release evidence")
        additional.extend(evidence_paths.values())

    operator_payloads, operator_binding = _load_operator_archives(operator_dir)
    files = _iter_include_files(root, public_evidence=additional)
    head = (git_head or _git_head(root)).lower()
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("git_head must be a full lowercase SHA-1")
    if tree_binding is not None and tree_binding["git_head"] != head:
        raise ValueError("included-tree manifest Git HEAD differs from the release source")
    _validate_source_tree_coverage(
        tree_binding,
        files,
        root,
        evidence_paths=additional,
    )
    manifest_entries: list[dict[str, object]] = []
    secret_hits: list[str] = []
    staged: list[tuple[str, bytes]] = []
    for path in files:
        rel = _safe_rel(path, root)
        data = path.read_bytes()
        secret_hits.extend(_scan_secrets(rel, data))
        manifest_entries.append(
            {"path": rel, "sha256": _sha256_bytes(data), "size": len(data)}
        )
        staged.append((rel, data))
    selected_rels = {rel for rel, _ in staged}
    for rel, data in operator_payloads:
        if rel in selected_rels:
            raise ValueError(f"operator release input collides with source payload: {rel}")
        manifest_entries.append(
            {"path": rel, "sha256": _sha256_bytes(data), "size": len(data)}
        )
        staged.append((rel, data))
    manifest_entries.sort(key=lambda entry: str(entry["path"]))
    staged.sort(key=lambda item: item[0])
    if secret_hits:
        raise ValueError(
            "secret scan failed; refusing to build bundle:\n  "
            + "\n  ".join(secret_hits[:50])
        )

    manifest = {
        "schema_version": 2,
        "bundle": output.name,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": head,
        "algorithm": "sha256 of sorted path NUL size NUL file_sha256 LF",
        "n_files": len(manifest_entries),
        "files": manifest_entries,
        "included_tree_sha256": _package_tree_sha(manifest_entries),
        "source_included_tree": tree_binding,
        "required_public_evidence": evidence_binding,
        "required_operator_archives": operator_binding,
        "excluded_note": (
            "private corpora, holdout/gold SQL/details, CBLUE, checkpoints, "
            "OM/ONNX, environment files, keys, and remote transfer archives are excluded"
        ),
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, suffix=".zip.tmp", delete=False
    ) as stream:
        temporary_zip = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            for rel, data in staged:
                archive.writestr(f"dki-agents/{rel}", data)
            archive.writestr("dki-agents/SUBMISSION_MANIFEST.json", manifest_bytes)
        with temporary_zip.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary_zip, output)
    finally:
        temporary_zip.unlink(missing_ok=True)
    _publish_bytes_once(manifest_output, manifest_bytes)
    sums_bytes = (
        f"{_sha256_file(output)}  {output.name}\n"
        f"{_sha256_file(manifest_output)}  {manifest_output.name}\n"
    ).encode("utf-8")
    _publish_bytes_once(sums_output, sums_bytes)
    return {
        "output": str(output),
        "n_files": len(manifest_entries),
        "manifest": str(manifest_output),
        "sha256s": str(sums_output),
        "bundle_sha256": _sha256_file(output),
        "included_tree_sha256": manifest["included_tree_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="build immutable clean-room release candidate")
    parser.add_argument("--rid", required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--included-tree-manifest", type=Path, required=True)
    parser.add_argument("--candidate-seal", type=Path, required=True)
    parser.add_argument("--public-freeze-attestation", type=Path, required=True)
    parser.add_argument("--development-receipt", type=Path, required=True)
    parser.add_argument("--blind-attestation", type=Path, required=True)
    parser.add_argument("--consumption-receipt", type=Path, required=True)
    parser.add_argument(
        "--operator-dir",
        type=Path,
        required=True,
        help="create-once RID directory containing exactly the six freshly built operator ZIPs",
    )
    args = parser.parse_args()

    if not RID_RE.fullmatch(args.rid):
        raise SystemExit("rid does not match the immutable release format")
    root = args.root.resolve()
    expected_parent = root / "dist" / "submission" / args.rid
    output = (args.output or (expected_parent / f"dki-agents_submission_{args.rid}.zip")).resolve()
    if output.parent != expected_parent.resolve():
        raise SystemExit("release output must be exactly below dist/submission/<RID>")
    evidence = {
        "candidate_seal": args.candidate_seal,
        "public_freeze_attestation": args.public_freeze_attestation,
        "development_receipt": args.development_receipt,
        "blind_attestation": args.blind_attestation,
        "consumption_receipt": args.consumption_receipt,
    }
    try:
        summary = build(
            output,
            root,
            included_tree_manifest=args.included_tree_manifest,
            public_evidence=evidence,
            operator_dir=args.operator_dir,
        )
    except (FileExistsError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
