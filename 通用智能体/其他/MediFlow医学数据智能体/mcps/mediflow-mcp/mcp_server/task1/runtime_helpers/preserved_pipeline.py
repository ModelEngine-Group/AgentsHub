# -*- coding: utf-8 -*-
"""任务一源格式保留清洗流水线辅助函数。"""
import argparse
import difflib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Union

import os

import requests

from .datamate_ops import build_files, register_benchmark, register_dataset, run_sudo
from .quality_eval import evaluate_file, summarize
from .governance import register_governance
from .local_cleaning import CleanStats, clean_json, clean_jsonl


DM_BASE = os.environ["CCF_DATAMATE_BASE"]
DATASET_VOLUME = os.environ["CCF_DATASET_VOLUME"]


OperatorSpec = Union[str, Dict]


def op(spec: OperatorSpec) -> Dict:
    """把简写算子 ID 或完整算子配置标准化为 DataMate 任务实例。"""
    if isinstance(spec, str):
        return {"id": spec, "name": spec, "inputs": "text", "outputs": "text", "overrides": {}}
    operator_id = str(spec.get("id") or "").strip()
    if not operator_id:
        raise ValueError("operator spec requires id")
    return {
        "id": operator_id,
        "name": spec.get("name") or operator_id,
        "inputs": spec.get("inputs") or "text",
        "outputs": spec.get("outputs") or "text",
        "overrides": dict(spec.get("overrides") or {}),
    }


def run_pipeline(
    dataset: Dict,
    operators: List[OperatorSpec],
    label: str,
    timeout_seconds: int | None = None,
) -> Dict:
    created_at = int(time.time())
    payload = {
        "name": f"任务一-{label}-源格式清洗-{created_at}",
        "description": "任务一真实混合数据源格式清洗",
        "srcDatasetId": dataset["id"],
        "srcDatasetName": dataset["name"],
        "destDatasetName": f"任务一-{label}-清洗结果-源格式-{created_at}",
        "destDatasetType": "TEXT",
        "instance": [op(x) for x in operators],
    }
    cr = requests.post(f"{DM_BASE}/api/cleaning/tasks", json=payload, timeout=20)
    print(f"[{label}] create: {cr.status_code} {cr.text[:180]}")
    cr.raise_for_status()
    task_id = cr.json()["data"]["id"]
    # DataMate 会在创建清洗任务时自动启动执行。
    # 这里不重复调用 execute 接口，避免并发执行造成重复输出。
    print(f"[{label}] task auto-started by create endpoint: {task_id}")

    last = {}
    deadline = time.time() + timeout_seconds if timeout_seconds is not None else None
    tick = 0
    while deadline is None or time.time() < deadline:
        time.sleep(5)
        tick += 5
        try:
            info = requests.get(f"{DM_BASE}/api/cleaning/tasks/{task_id}", timeout=15)
        except requests.RequestException as exc:
            print(f"[{label}] status request failed; keep waiting: {type(exc).__name__}")
            continue
        if not info.ok:
            continue
        last = info.json().get("data", {})
        status = last.get("status")
        progress = last.get("progress", {})
        done = progress.get("succeedFileNum", 0) + progress.get("failedFileNum", 0)
        total = progress.get("totalFileNum", "?")
        print(f"[{label}] {tick}s status={status} progress={done}/{total}")
        if status in ("COMPLETED", "FAILED", "STOPPED", "PARTIAL_SUCCESS"):
            break
    if deadline is not None and last.get("status") != "COMPLETED":
        raise RuntimeError(f"{label} pipeline did not complete: {json.dumps(last, ensure_ascii=False)[:800]}")
    if last.get("status") != "COMPLETED":
        raise RuntimeError(f"{label} pipeline ended without a terminal completion state: {json.dumps(last, ensure_ascii=False)[:800]}")
    return {"task_id": task_id, "dest_dataset_id": last.get("destDatasetId"), "detail": last}


def _filter_duplicate_jsonl_records(content: str, seen_record_ids: set, source_name: str) -> str:
    lines = []
    parsed_any = False
    skipped = 0
    for raw in content.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
            parsed_any = isinstance(obj, dict)
        except Exception:
            lines.append(stripped)
            continue
        record_id = obj.get("record_id") if isinstance(obj, dict) else None
        if record_id and record_id in seen_record_ids:
            skipped += 1
            continue
        if record_id:
            seen_record_ids.add(record_id)
        lines.append(stripped)
    if skipped:
        print(f"[dedupe] skipped {skipped} duplicate records from {source_name}")
    if parsed_any:
        return "\n".join(lines) + ("\n" if lines else "")
    return content


def _query_dataset_files(dataset_id: str) -> List[Dict[str, str]]:
    dataset_id_sql = dataset_id.replace("'", "''")
    sql = (
        "select file_name, coalesce(file_path,''), coalesce(file_type,'') "
        "from t_dm_dataset_files "
        f"where dataset_id='{dataset_id_sql}' and status in ('ACTIVE','COMPLETED') "
        "order by file_name;"
    )
    result = subprocess.run(
        [
            "docker", "exec", "-i", "datamate-database", "psql",
            "-U", "postgres", "-d", "datamate", "-t", "-A", "-F", "\t",
        ],
        input=sql,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    files = []
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            files.append({"file_name": parts[0], "file_path": parts[1], "file_type": parts[2]})
    return files


def _dataset_host_path(dataset_id: str, file_name: str, file_path: str) -> str:
    normalized = (file_path or "").replace("\\", "/")
    marker = "/dataset/"
    if marker in normalized:
        rel = normalized.split(marker, 1)[1].lstrip("/")
        if rel:
            return f"{DATASET_VOLUME}/{rel}"
    return f"{DATASET_VOLUME}/{dataset_id}/{file_name}"


def collect_outputs(dest_dataset_id: str, output_dir: Path, label: str, seen_record_ids: set = None) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_record_ids = seen_record_ids if seen_record_ids is not None else set()
    paths = []
    for item in _query_dataset_files(dest_dataset_id):
        fname = item["file_name"]
        cat = run_sudo(["cat", _dataset_host_path(dest_dataset_id, fname, item["file_path"])])
        if cat.returncode != 0:
            raise RuntimeError(cat.stderr)
        content = _filter_duplicate_jsonl_records(cat.stdout, seen_record_ids, fname)
        if not content.strip():
            continue
        suffix = ".jsonl" if fname.endswith(".jsonl") else Path(fname).suffix or ".txt"
        local = output_dir / f"{label}_{Path(fname).stem}{suffix}"
        local.write_text(content, encoding="utf-8")
        paths.append(local)
    return paths


def postprocess_structured_outputs(paths: List[Path]) -> Dict:
    """对 DataMate 产出的结构化文件执行确定性收尾清理。

    某些 DataMate 版本在算子未生成新文件时会复用源文件路径。这里会在最终
    登记前补做字段级清理，保证任务一输出既保持源格式，又符合质量报告。
    """
    stats = CleanStats()
    changed = []
    skipped = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix not in {".json", ".jsonl"}:
            continue
        before = path.read_text(encoding="utf-8", errors="replace")
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            if suffix == ".jsonl":
                clean_jsonl(path, tmp, stats)
            else:
                clean_json(path, tmp, stats)
            after = tmp.read_text(encoding="utf-8", errors="replace")
            if after != before:
                changed.append(path.name)
            tmp.replace(path)
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            skipped.append({"file": path.name, "error": str(exc)[:240]})
    return {
        "text_fields": stats.text_fields,
        "changed_fields": stats.changed_fields,
        "change_counts": dict(sorted(stats.changes.items())),
        "changed_files": changed,
        "skipped": skipped,
    }


def restore_structured_output_suffix(paths: List[Path], group: str) -> List[Path]:
    """恢复结构化输出文件后缀。

    DataMate 数据集类型统一记录为 TEXT，部分导出路径会把 JSON/JSONL 文件
    命名成 txt。任务一最终数据集需要保留源格式，因此在质量检查和最终登记前
    恢复正确后缀。
    """
    expected_suffix = { "json": ".json", "jsonl": ".jsonl" }.get(group)
    if not expected_suffix:
        return paths

    restored: List[Path] = []
    for path in paths:
        if path.suffix.lower() == expected_suffix:
            restored.append(path)
            continue
        target = path.with_suffix(expected_suffix)
        if target.exists() and target != path:
            target.unlink()
        path.replace(target)
        restored.append(target)
    return restored


def merge_numbered_text_chunks(paths: List[Path]) -> List[Path]:
    """合并 DataMate 导出的编号文本块。

    长文本可能被导出为 `name.txt` 和 `name_1.txt`。如果多个文件是近似重复
    的完整文档，则保留一份；如果确实是连续分块，则按重叠部分合并。
    """
    text_paths = [path for path in paths if path.suffix.lower() == ".txt"]
    other_paths = [path for path in paths if path.suffix.lower() != ".txt"]
    stems = {path.stem for path in text_paths}
    groups: Dict[Path, List[Path]] = {}
    for path in text_paths:
        match = re.match(r"^(.*)_\d+$", path.stem)
        base_stem = match.group(1) if match and match.group(1) in stems else path.stem
        base_path = path.with_name(base_stem + path.suffix)
        groups.setdefault(base_path, []).append(path)

    merged_paths: List[Path] = []
    for base_path, members in groups.items():
        def sort_key(item: Path):
            if item.stem == base_path.stem:
                return (0, 0)
            suffix = item.stem[len(base_path.stem) + 1:]
            return (1, int(suffix) if suffix.isdigit() else 9999)

        ordered = sorted(members, key=sort_key)
        if len(ordered) > 1:
            chunks = [item.read_text(encoding="utf-8", errors="replace").strip() for item in ordered]
            merged = chunks[0]
            for chunk in chunks[1:]:
                if not chunk:
                    continue
                if not merged:
                    merged = chunk
                    continue
                if merged in chunk or chunk in merged:
                    continue
                overlap = _suffix_prefix_overlap(merged, chunk)
                if overlap >= min(200, len(chunk) // 5):
                    merged += chunk[overlap:]
                    continue
                similarity = difflib.SequenceMatcher(
                    None, merged, chunk, autojunk=False
                ).ratio()
                if similarity >= 0.90:
                    continue
                merged += "\n\n" + chunk
            base_path.write_text(merged.strip() + "\n", encoding="utf-8")
            for item in ordered:
                if item != base_path and item.exists():
                    item.unlink()
        merged_paths.append(base_path)
    return sorted(other_paths + merged_paths)


def _suffix_prefix_overlap(left: str, right: str) -> int:
    """返回两个文本块之间最长的后缀/前缀重叠长度。"""
    upper = min(len(left), len(right))
    for size in range(upper, 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def evaluate_outputs(paths: List[Path], min_records: int, label: str) -> Dict:
    results = [evaluate_file(path) for path in paths]
    report = summarize(results, min_records)
    print(json.dumps({"label": label, **report}, ensure_ascii=False, indent=2)[:4000])
    if not report["pass"]:
        raise AssertionError(f"{label} output quality failed")
    return report


def _file_type(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return suffix or "txt"


def register_final_delivery(outputs_dir: Path, name: str, description: str = "") -> Dict:
    files = []
    for path in sorted(outputs_dir.glob("*")):
        if path.is_file():
            files.append((path, _file_type(path)))
    file_types = {file_type for _path, file_type in files}
    dataset_format = next(iter(file_types)) if len(file_types) == 1 else "mixed"
    return register_dataset(
        name,
        files,
        dataset_format,
        description=description or "任务一最终交付：保留源数据格式，清理噪声与格式错误，并保留质量证据",
    )


def check_db_health() -> Dict:
    sql = (
        "SELECT "
        "(SELECT count(*) FROM t_dm_datasets WHERE dataset_type <> 'TEXT') AS bad_dataset_type,"
        "(SELECT count(*) FROM (SELECT dataset_id,file_path,count(*) FROM t_dm_dataset_files "
        "GROUP BY dataset_id,file_path HAVING count(*)>1) x) AS duplicate_file_paths;"
    )
    result = subprocess.run(
        ["docker", "exec", "-i", "datamate-database", "psql", "-U", "postgres", "-d", "datamate", "-t", "-A"],
        input=sql,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    parts = [x.strip() for x in result.stdout.strip().split("|")]
    health = {"bad_dataset_type": int(parts[0] or 0), "duplicate_file_paths": int(parts[1] or 0)}
    if health["bad_dataset_type"] or health["duplicate_file_paths"]:
        raise AssertionError(f"DataMate DB health check failed: {health}")
    return health


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--work-dir", default="data/task1_mixed_benchmark")
    parser.add_argument("--text-files", type=int, default=8)
    parser.add_argument("--csv-files", type=int, default=4)
    parser.add_argument("--json-files", type=int, default=4)
    parser.add_argument("--records-per-text-file", type=int, default=2)
    parser.add_argument("--rows-per-structured-file", type=int, default=3)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    manifest = build_files(
        work_dir,
        args.source_root,
        text_file_count=args.text_files,
        csv_file_count=args.csv_files,
        json_file_count=args.json_files,
        records_per_text_file=args.records_per_text_file,
        rows_per_structured_file=args.rows_per_structured_file,
    )
    manifest["datamate_datasets"] = register_benchmark(manifest)
    (work_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("BENCHMARK_MANIFEST")
    print(json.dumps(manifest, ensure_ascii=False, indent=2)[:4000])

    datasets = manifest["datamate_datasets"]
    outputs_dir = work_dir / "outputs"
    if outputs_dir.exists():
        shutil.rmtree(outputs_dir)
    reports = {}

    text_ops = [
        "EmojiCleaner", "UrlRemover", "GrableCharactersCleaner", "InvisibleCharactersCleaner",
        "FullWidthCharacterCleaner", "TraditionalChineseCleaner", "HtmlTagCleaner",
        "WhitespaceNormalizer", "MedicalTermNormalizer", "LLMNoiseFilter",
    ]
    csv_ops = ["TableColumnCleaner"]
    json_ops = ["JsonFieldCleaner"]

    text_result = run_pipeline(datasets["text"], text_ops, "text")
    text_paths = collect_outputs(text_result["dest_dataset_id"], outputs_dir, "text", set())
    text_paths = merge_numbered_text_chunks(text_paths)
    reports["text"] = evaluate_outputs(text_paths, max(1, len(text_paths)), "text")

    csv_result = run_pipeline(datasets["csv"], csv_ops, "csv")
    csv_paths = collect_outputs(csv_result["dest_dataset_id"], outputs_dir, "csv", set())
    reports["csv"] = evaluate_outputs(csv_paths, manifest["record_counts"]["csv_rows"], "csv")

    json_result = run_pipeline(datasets["json"], json_ops, "json")
    json_paths = collect_outputs(json_result["dest_dataset_id"], outputs_dir, "json", set())
    json_paths = restore_structured_output_suffix(json_paths, "json")
    reports["json"] = evaluate_outputs(json_paths, manifest["record_counts"]["json_records"], "json")

    delivery_dataset = register_final_delivery(
        outputs_dir,
        f"任务一_最终清洗结果_保持源格式_{int(time.time())}",
    )

    health = check_db_health()
    summary = {
        "pass": True,
        "db_health": health,
        "outputs_dir": str(outputs_dir),
        "source_mixed_dataset": datasets["mixed"],
        "delivery_dataset": delivery_dataset,
        "delivery_report": {"files": len(list(outputs_dir.glob('*')))},
        "reports": {k: v["totals"] for k, v in reports.items()},
    }
    report_path = work_dir / "test_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    governance_metadata = register_governance(report_path)
    summary["governance_metadata"] = governance_metadata
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TASK1_MIXED_BENCHMARK_PASSED")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
