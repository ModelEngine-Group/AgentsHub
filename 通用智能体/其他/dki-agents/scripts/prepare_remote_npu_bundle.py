"""Build a local, credential-free bundle and runbook for later NPU validation.

No SSH connection, environment-file read, NPU import, or remote side effect occurs
here. The source archive deliberately excludes data, models, outputs, .env files,
and every legacy remote launcher; optional normalized CBLUE data is a separately
opted-in local archive for a controlled target host.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ARCHIVE_ROOT = "dki-agents"
SOURCE_FILES = (
    "README.md",
    "CONTRACTS.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    ".gitignore",
    "configs/pipeline_medical_etl.yaml",
    "configs/pipeline_kg_build.yaml",
    "configs/pipeline_kg_build_cmeee.yaml",
    "configs/pipeline_kg_build_cmeee_om.yaml",
    "configs/pipeline_kg_build_shadow_om.yaml",
    "evidence/manifests/protected_baseline.json",
    "scripts/train_ner_cmeee.py",
    "scripts/train_re_cmeie.py",
    "scripts/download_cblue.py",
    "scripts/eval_ner_cmeee_checkpoint.py",
    "scripts/eval_re_cmeie_e2e.py",
    "scripts/eval_nl2sql_layered.py",
    "scripts/freeze_nl2sql_blind.py",
    "scripts/eval_nl2sql_blind.py",
    "scripts/export_cmeee_globalpointer_onnx.py",
    "scripts/compile_cmeee_om.py",
    "scripts/create_om_base_manifest.py",
    "scripts/bind_om_checkpoint.py",
    "scripts/remote_npu_preflight.py",
    "scripts/bench_medical_ner_adapter_batch.py",
    "scripts/sample_npu_power.py",
    "scripts/run_npu_energy_acceptance.py",
    "scripts/build_evidence_claims.py",
    "scripts/validate_evidence.py",
    "scripts/package_operators.py",
)
TEST_FILES = (
    "tests/test_agents.py",
    "tests/process_runtime_fixtures.py",
    "tests/test_cblue_normalization.py",
    "tests/test_cmeee_training_protocol.py",
    "tests/test_cmeie_training_protocol.py",
    "tests/test_medical_ner_om_adapter.py",
    "tests/test_om_manifest_gate.py",
    "tests/test_om_runtime.py",
    "tests/test_om_shadow.py",
    "tests/test_package_operators.py",
    "tests/test_nl2sql_sota.py",
    "tests/test_power_evidence.py",
    "tests/test_recovery.py",
    "tests/test_task1_artifacts.py",
)
CBLUE_FILES = (
    "data/cblue/CMeEE-V2/train.jsonl",
    "data/cblue/CMeEE-V2/dev.jsonl",
    "data/cblue/CMeIE-V2/train.jsonl",
    "data/cblue/CMeIE-V2/dev.jsonl",
)
EXCLUDED_PARTS = {".env", ".venv", ".git", ".claude", "outputs", "models", "dist", "__pycache__"}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    if relative.is_absolute() or any(part in EXCLUDED_PARTS for part in relative.parts):
        raise ValueError(f"forbidden bundle path: {relative.as_posix()}")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"bundle input must be a regular file: {relative.as_posix()}")
    return relative.as_posix()


def _dki_files(root: Path) -> Iterable[Path]:
    for path in sorted((root / "dki").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            yield path


def _source_paths(root: Path) -> list[Path]:
    paths = [root / relative for relative in SOURCE_FILES]
    paths.extend(root / relative for relative in TEST_FILES)
    paths.extend(_dki_files(root))
    missing = [str(path.relative_to(root)) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"remote source bundle inputs missing: {missing}")
    deduplicated = {path.resolve(): path for path in paths}
    return sorted(deduplicated.values(), key=lambda path: _safe_relative(path, root))


def _data_paths(root: Path) -> list[Path]:
    paths = [root / relative for relative in CBLUE_FILES]
    missing = [str(path.relative_to(root)) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "normalized CBLUE data is incomplete; run download_cblue.py locally first: "
            f"{missing}"
        )
    return paths


def _scan_secrets(root: Path, paths: Iterable[Path]) -> None:
    # Reuse the submission scanner so remote and submission bundles have the
    # same secret policy without placing a plaintext credential in either tool.
    from scripts.build_submission_bundle import _scan_secrets as scan

    hits = []
    for path in paths:
        relative = _safe_relative(path, root)
        hits.extend(scan(relative, path.read_bytes()))
    if hits:
        raise ValueError("secret scan rejected remote source bundle:\n  " + "\n  ".join(hits[:20]))


def _entries(root: Path, paths: Iterable[Path]) -> list[dict]:
    return [
        {
            "path": _safe_relative(path, root),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]


def _tree_digest(entries: list[dict]) -> str:
    canonical = "".join(f"{item['path']}\0{item['sha256']}\n" for item in entries).encode("utf-8")
    return _sha256_bytes(canonical)


def _runbook(source_tree_sha256: str) -> dict:
    return {
        "schema_version": 1,
        "source_tree_sha256": source_tree_sha256,
        "scope": "controlled_target_host_only",
        "hard_constraints": [
            "Do not use final3/protected blind data for training, selection, calibration, routing, profiling, or error analysis.",
            "CMeIE reports gold entity-pair plus gold type-signature predicate classification only; it is not end-to-end SPO.",
            "Do not claim formal DataMate cross-document microbatching until the platform calls a batch API in an observed target-host flow.",
            "Do not overwrite historical models or canonical metrics; every new run uses a versioned candidate path.",
        ],
        "required_variables": {
            "DKI_ROOT": "controlled target-host checkout directory",
            "RUN_OUTPUT": "versioned remote output directory, e.g. outputs/remote_runs/<run-id>",
            "DKI_SOC_VERSION": "exact ATC target, e.g. Ascend910B2C",
            "DKI_CANN_VERSION": "exact installed CANN version",
            "NPU_IDLE_WATTS": "separately measured idle watts for the selected NPU",
            "CMEEE_MODEL_DIR": "existing frozen CMeEE max_len=256 candidate directory",
            "CMEEE_ONNX": "create-once ONNX path outside CMEEE_MODEL_DIR",
            "CMEEE_EXPORT_RECEIPT": "create-once checkpoint-to-ONNX receipt",
            "OM_PATH": "candidate .om file",
            "OM_TOKENIZER_PATH": "exact tokenizer artifact hash-bound in OM_MANIFEST",
            "OM_LABELS_PATH": "exact label/config artifact hash-bound in OM_MANIFEST",
            "OM_ATC_LOG": "create-once JSON log emitted by compile_cmeee_om.py",
            "OM_BASE_MANIFEST": "schema-v1 manifest from real ACL IO inspection",
            "OM_MANIFEST": "schema-v3 checkpoint/export/OM hash-bound manifest",
        },
        "archive_verification": [
            "Compare sha256sum dki-agents_remote_npu_source.tar.gz with source_archive_sha256 in its uploaded manifest before extraction.",
            "If the separately prepared CBLUE archive is authorized, compare its sha256sum with cblue_archive_sha256 in its uploaded manifest before extraction.",
        ],
        "installation": [
            "mkdir -p $DKI_ROOT $RUN_OUTPUT",
            "tar xzf dki-agents_remote_npu_source.tar.gz --strip-components=1 -C $DKI_ROOT",
            "tar xzf dki-agents_cblue_normalized.tar.gz --strip-components=1 -C $DKI_ROOT  # only if the separately prepared controlled-host data archive is authorized",
            "source $ASCEND_ENV  # set this to the target host's verified CANN set_env.sh before an NPU step",
            "cd $DKI_ROOT && export PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false",
        ],
        "ordered_steps": [
            {
                "id": "host_preflight",
                "purpose": "verify host, normalized CBLUE rows, and checkpoint without ACL/model load",
                "command": (
                    "python3 scripts/remote_npu_preflight.py --root . --strict "
                    "--output $RUN_OUTPUT/preflight.json "
                    "--cmeee-model $CMEEE_MODEL_DIR"
                ),
            },
            {
                "id": "cmeee_onnx_export",
                "purpose": "export the frozen checkpoint and seal its tree/ONNX receipt",
                "command": (
                    "python3 scripts/export_cmeee_globalpointer_onnx.py "
                    "--checkpoint $CMEEE_MODEL_DIR --output $CMEEE_ONNX "
                    "--receipt $CMEEE_EXPORT_RECEIPT --batch-size 16"
                ),
            },
            {
                "id": "cmeee_atc_compile",
                "purpose": "compile the receipt-bound ONNX and preserve the complete ATC log",
                "command": (
                    "python3 scripts/compile_cmeee_om.py --onnx $CMEEE_ONNX "
                    "--export-receipt $CMEEE_EXPORT_RECEIPT --output $OM_PATH "
                    "--log $OM_ATC_LOG --soc-version $DKI_SOC_VERSION"
                ),
            },
            {
                "id": "inspect_real_om_io",
                "purpose": "load the OM once to record its real ACL tensor contract; no inference",
                "command": (
                    "python3 scripts/create_om_base_manifest.py --om $OM_PATH "
                    "--export-receipt $CMEEE_EXPORT_RECEIPT "
                    "--tokenizer $OM_TOKENIZER_PATH --labels $OM_LABELS_PATH "
                    "--soc-version $DKI_SOC_VERSION --cann-version $DKI_CANN_VERSION "
                    "--output $OM_BASE_MANIFEST"
                ),
            },
            {
                "id": "bind_checkpoint_to_om",
                "purpose": "promote only an exact checkpoint/export/OM/IO match to schema v3",
                "command": (
                    "python3 scripts/bind_om_checkpoint.py --manifest $OM_BASE_MANIFEST "
                    "--checkpoint $CMEEE_MODEL_DIR --export-receipt $CMEEE_EXPORT_RECEIPT "
                    "--onnx $CMEEE_ONNX --om $OM_PATH --tokenizer $OM_TOKENIZER_PATH "
                    "--labels $OM_LABELS_PATH --atc-log $OM_ATC_LOG --output $OM_MANIFEST"
                ),
            },
            {
                "id": "bound_artifact_preflight",
                "purpose": "verify the complete schema-v3 chain without loading the OM",
                "command": (
                    "python3 scripts/remote_npu_preflight.py --root . --strict "
                    "--output $RUN_OUTPUT/bound_artifact_preflight.json "
                    "--cmeee-model $CMEEE_MODEL_DIR --om-manifest $OM_MANIFEST "
                    "--om-path $OM_PATH --tokenizer-path $OM_TOKENIZER_PATH "
                    "--labels-path $OM_LABELS_PATH --checkpoint-path $CMEEE_MODEL_DIR "
                    "--export-receipt-path $CMEEE_EXPORT_RECEIPT "
                    "--atc-log-path $OM_ATC_LOG"
                ),
            },
            {
                "id": "cmeee_frozen_rescore",
                "purpose": "add macro/per-type/worst-type metrics to the frozen candidate; no retraining or threshold selection",
                "command": (
                    "python3 scripts/eval_ner_cmeee_checkpoint.py --model $CMEEE_MODEL_DIR "
                    "--device npu --batch-size 16 --output $RUN_OUTPUT/cmeee_frozen_rescore.json"
                ),
            },
            {
                "id": "cmeie_fit_only_seed42",
                "purpose": "rerun corrected multi-signature CMeIE with fit-partition-only type constraints",
                "command": (
                    "python3 scripts/train_re_cmeie.py --device npu --epochs 5 --batch-size 32 "
                    "--max-len 256 --neg-per-pos 1 --seed 42 "
                    "--output-model models/candidates/cmeie_multisig_fitonly_s42 "
                    "--output-metrics $RUN_OUTPUT/cmeie_multisig_fitonly_s42.json"
                ),
            },
            {
                "id": "cmeie_fit_only_seed43",
                "purpose": "independent-seed stability check; run only after seed 42 evidence is collected",
                "command": (
                    "python3 scripts/train_re_cmeie.py --device npu --epochs 5 --batch-size 32 "
                    "--max-len 256 --neg-per-pos 1 --seed 43 "
                    "--output-model models/candidates/cmeie_multisig_fitonly_s43 "
                    "--output-metrics $RUN_OUTPUT/cmeie_multisig_fitonly_s43.json"
                ),
            },
            {
                "id": "om_cpu_npu_energy_acceptance",
                "purpose": "three-run same-checkpoint CPU/NPU parity, latency and synchronized joules/document",
                "command": (
                    "python3 scripts/run_npu_energy_acceptance.py --root . --om $OM_PATH "
                    "--checkpoint $CMEEE_MODEL_DIR --manifest $OM_MANIFEST "
                    "--export-receipt $CMEEE_EXPORT_RECEIPT --atc-log $OM_ATC_LOG "
                    "--documents 200 --document-batch 64 --idle-watts $NPU_IDLE_WATTS "
                    "--soc-version $DKI_SOC_VERSION --cann-version $DKI_CANN_VERSION "
                    "--output-dir $RUN_OUTPUT/cmeee_npu_acceptance"
                ),
            },
        ],
        "collection": {
            "download_only": [
                "$RUN_OUTPUT/preflight.json",
                "$RUN_OUTPUT/bound_artifact_preflight.json",
                "$RUN_OUTPUT/cmeee_frozen_rescore.json",
                "$RUN_OUTPUT/cmeie_multisig_fitonly_s42.json",
                "$RUN_OUTPUT/cmeie_multisig_fitonly_s43.json",
                "$RUN_OUTPUT/cmeee_npu_acceptance/cmeee_cpu_npu_acceptance.json",
                "$RUN_OUTPUT/cmeee_npu_acceptance/cmeee_npu_power.csv",
                "$RUN_OUTPUT/cmeee_npu_acceptance/cmeee_npu_power.summary.json",
                "$RUN_OUTPUT/cmeee_npu_acceptance/cmeee_npu_release_receipt.json",
                "$CMEEE_EXPORT_RECEIPT",
                "$OM_ATC_LOG",
                "$OM_BASE_MANIFEST",
                "$OM_MANIFEST",
            ],
            "do_not_download": ["raw medical corpora", "model weights", "OM binaries", "environment files"],
        },
    }


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    tar.addfile(info, io.BytesIO(data))


def _write_archive(output: Path, root: Path, paths: list[Path], extra: dict[str, bytes]) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as tar:
            for path in paths:
                relative = _safe_relative(path, root)
                _add_bytes(tar, f"{ARCHIVE_ROOT}/{relative}", path.read_bytes())
            for relative, data in sorted(extra.items()):
                _add_bytes(tar, f"{ARCHIVE_ROOT}/{relative}", data)
    return _sha256(output)


def _ensure_absent_or_allowed(paths: Iterable[Path], overwrite: bool) -> None:
    present = [str(path) for path in paths if path.exists()]
    if present and not overwrite:
        raise FileExistsError("refusing to overwrite existing preparation artifact(s): " + ", ".join(present))


def build(output_dir: Path, root: Path, *, include_cblue: bool, overwrite: bool) -> dict:
    root = root.resolve()
    source_paths = _source_paths(root)
    _scan_secrets(root, source_paths)
    source_entries = _entries(root, source_paths)
    source_digest = _tree_digest(source_entries)
    runbook = _runbook(source_digest)
    source_manifest = {
        "schema_version": 1,
        "kind": "remote_npu_source_only",
        "source_tree_sha256": source_digest,
        "archive_root": ARCHIVE_ROOT,
        "forbidden_contents": [".env", "data", "models", "outputs", "dist", "legacy SSH launchers"],
        "files": source_entries,
    }
    output_dir = output_dir.resolve()
    source_archive = output_dir / "dki-agents_remote_npu_source.tar.gz"
    source_manifest_path = output_dir / "dki-agents_remote_npu_source.manifest.json"
    runbook_path = output_dir / "REMOTE_NPU_RUNBOOK.json"
    artifacts = [source_archive, source_manifest_path, runbook_path]
    if include_cblue:
        artifacts.extend([
            output_dir / "dki-agents_cblue_normalized.tar.gz",
            output_dir / "dki-agents_cblue_normalized.manifest.json",
        ])
    _ensure_absent_or_allowed(artifacts, overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_sha = _write_archive(
        source_archive, root, source_paths,
        {
            "REMOTE_NPU_RUNBOOK.json": json.dumps(runbook, ensure_ascii=False, indent=2).encode("utf-8"),
            "REMOTE_NPU_SOURCE_MANIFEST.json": json.dumps(source_manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        },
    )
    source_manifest["archive_sha256"] = archive_sha
    source_manifest_path.write_text(json.dumps(source_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    runbook_path.write_text(json.dumps(runbook, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {
        "source_archive": str(source_archive),
        "source_archive_sha256": archive_sha,
        "source_files": len(source_entries),
        "source_tree_sha256": source_digest,
        "runbook": str(runbook_path),
    }
    if include_cblue:
        data_paths = _data_paths(root)
        data_entries = _entries(root, data_paths)
        data_archive = output_dir / "dki-agents_cblue_normalized.tar.gz"
        data_manifest_path = output_dir / "dki-agents_cblue_normalized.manifest.json"
        data_sha = _write_archive(data_archive, root, data_paths, {})
        data_manifest = {
            "schema_version": 1,
            "kind": "controlled_host_normalized_cblue_data",
            "archive_sha256": data_sha,
            "archive_root": ARCHIVE_ROOT,
            "license_note": "CBLUE is third-party research data; transfer only to the controlled authorized target host, never publish or commit.",
            "files": data_entries,
        }
        data_manifest_path.write_text(json.dumps(data_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        result.update({
            "cblue_archive": str(data_archive),
            "cblue_archive_sha256": data_sha,
            "cblue_files": len(data_entries),
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a no-SSH remote NPU source bundle")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist" / "remote_npu")
    parser.add_argument("--include-cblue", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = build(
        args.output_dir, args.root,
        include_cblue=args.include_cblue, overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
