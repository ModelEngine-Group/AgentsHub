"""Versioned Task 1 artifact contract shared by Tasks 2 and 3.

The contest workflow must not silently return to ``data/raw`` after Task 1 has
completed.  A manifest gives every downstream consumer one immutable run
directory, source provenance, schemas, and content hashes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from dki.paths import PROCESSED_DIR, PROJECT_ROOT, TASK_DIR


MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1
TASK2_RECEIPT_NAME = "medical_kg.provenance.json"


def manifest_path() -> Path:
    return PROCESSED_DIR / MANIFEST_NAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    """Public hash helper used by downstream provenance receipts."""
    return _sha256(Path(path))


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _table_metadata(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        columns = next(reader, [])
        rows = sum(1 for _ in reader)
    return {
        "path": _relative(path),
        "sha256": _sha256(path),
        "columns": columns,
        "row_count": rows,
    }


def _file_metadata(path: Path) -> Dict[str, Any]:
    return {
        "path": _relative(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def write_task1_manifest(
    *,
    task_id: str,
    clean_dir: Path,
    source_text_dir: Path,
    structured_source_dir: Path,
    tables: Iterable[str] = ("patients.csv", "visits.csv"),
) -> Path:
    """Publish one Task 1 run without mutating its raw inputs.

    Structured inputs are copied into the run directory so downstream jobs read
    a Task 1-owned snapshot rather than a mutable raw-data location.
    """
    from dki.pipeline.state import _validate_task_id

    task_id = _validate_task_id(task_id)
    clean_dir = Path(clean_dir)
    if not clean_dir.is_dir():
        raise FileNotFoundError(f"任务一清洗目录不存在: {clean_dir}")
    try:
        clean_dir.resolve().relative_to(PROCESSED_DIR.resolve())
    except ValueError as exc:
        raise ValueError("任务一清洗目录必须位于 outputs/processed 下") from exc

    clean_files = sorted(clean_dir.glob("*.txt"))
    if not clean_files:
        raise ValueError("任务一没有可发布的清洗文本")

    table_sources = {
        filename: Path(structured_source_dir) / filename
        for filename in tables
    }
    missing_tables = [name for name, source in table_sources.items() if not source.is_file()]
    if missing_tables:
        raise FileNotFoundError(
            f"任务一结构化源表缺失: {', '.join(sorted(missing_tables))}"
        )

    run_dir = clean_dir.parent
    structured_dir = run_dir / "structured"
    structured_dir.mkdir(parents=True, exist_ok=True)

    copied_tables: Dict[str, Dict[str, Any]] = {}
    for filename, source in table_sources.items():
        target = structured_dir / filename
        shutil.copy2(source, target)
        metadata = _table_metadata(target)
        metadata["source_path"] = _relative(source)
        copied_tables[filename.removesuffix(".csv")] = metadata

    clean_entries = [_file_metadata(path) for path in clean_files]
    source_text_dir = Path(source_text_dir)
    source_files = (
        [_file_metadata(path) for path in sorted(source_text_dir.glob("*.txt"))]
        if source_text_dir.is_dir()
        else []
    )
    task_report = TASK_DIR / f"{task_id}.json"
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "task_id": task_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "raw_text_dir": _relative(source_text_dir),
            "raw_text_count": len(source_files),
            "raw_text_files": source_files,
            "structured_source_dir": _relative(Path(structured_source_dir)),
        },
        "clean_text": {
            "directory": _relative(clean_dir),
            "count": len(clean_entries),
            "files": clean_entries,
        },
        "structured_tables": copied_tables,
    }
    if task_report.is_file():
        payload["task_report"] = _file_metadata(task_report)
    path = manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{task_id}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def load_task1_manifest(path: Path | None = None, *, verify_hashes: bool = True) -> Mapping[str, Any]:
    """Load the current Task 1 snapshot and reject incomplete/tampered inputs."""
    path = Path(path or manifest_path())
    if not path.is_file():
        raise FileNotFoundError(f"缺少任务一产物清单: {path}；请先运行 task1")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"不支持的任务一 manifest 版本: {payload.get('schema_version')}")

    clean = payload.get("clean_text", {})
    clean_dir = _resolve(str(clean.get("directory", "")))
    if not clean_dir.is_dir() or not clean.get("files"):
        raise ValueError("任务一 manifest 未包含可用清洗文本")

    required_tables = {"patients", "visits"}
    tables = payload.get("structured_tables", {})
    missing = sorted(required_tables - set(tables))
    if missing:
        raise ValueError(f"任务一 manifest 缺少结构化表: {', '.join(missing)}")

    if verify_hashes:
        entries = [*clean.get("files", []), *tables.values()]
        task_report = payload.get("task_report")
        if task_report:
            entries.append(task_report)
        for entry in entries:
            artifact = _resolve(str(entry["path"]))
            if not artifact.is_file():
                raise FileNotFoundError(f"manifest 产物不存在: {artifact}")
            if _sha256(artifact) != entry.get("sha256"):
                raise ValueError(f"manifest 产物 hash 不匹配: {artifact}")
    return payload


def task1_clean_dir(manifest: Mapping[str, Any]) -> Path:
    return _resolve(str(manifest["clean_text"]["directory"]))


def task1_table_paths(manifest: Mapping[str, Any]) -> Dict[str, Path]:
    return {
        name: _resolve(str(entry["path"]))
        for name, entry in manifest["structured_tables"].items()
    }


def task2_receipt_path(kg_path: Path) -> Path:
    return Path(kg_path).with_name(TASK2_RECEIPT_NAME)


def write_task2_receipt(
    kg_path: Path,
    *,
    task1_path: Path | None = None,
    task_id: str = "",
) -> Path:
    """Bind a knowledge graph to the exact Task 1 snapshot that produced it."""
    kg_path = Path(kg_path)
    if not kg_path.is_file():
        raise FileNotFoundError(f"任务二知识图谱不存在: {kg_path}")
    task1_path = Path(task1_path or manifest_path())
    task1 = load_task1_manifest(task1_path)
    payload = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task_id": task_id,
        "task1_task_id": str(task1["task_id"]),
        "task1_manifest_path": _relative(task1_path),
        "task1_manifest_sha256": _sha256(task1_path),
        "kg_path": _relative(kg_path),
        "kg_sha256": _sha256(kg_path),
    }
    receipt = task2_receipt_path(kg_path)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt.with_name(f".{receipt.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(receipt)
    return receipt


def validate_task2_artifact(
    kg_path: Path,
    *,
    task1_path: Path | None = None,
) -> Mapping[str, Any]:
    """Reject a missing, modified, or stale Task 2 knowledge graph."""
    kg_path = Path(kg_path)
    task1_path = Path(task1_path or manifest_path())
    load_task1_manifest(task1_path)
    receipt = task2_receipt_path(kg_path)
    if not kg_path.is_file() or not receipt.is_file():
        raise FileNotFoundError("缺少任务二图谱或 provenance 回执；请重新运行 task2")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("不支持的任务二 provenance 版本")
    if payload.get("task1_manifest_sha256") != _sha256(task1_path):
        raise ValueError("任务二图谱不是由当前任务一 manifest 生成；请重新运行 task2")
    if payload.get("kg_sha256") != _sha256(kg_path):
        raise ValueError("任务二图谱 hash 与 provenance 不一致")
    return payload
