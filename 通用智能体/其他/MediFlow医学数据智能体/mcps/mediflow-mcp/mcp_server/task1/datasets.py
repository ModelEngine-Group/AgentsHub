"""
任务一数据集文件工具模块。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


TASK1_FILE_GROUPS = ("text", "csv", "json", "jsonl", "pdf")


def classify_source_file(file_name: str, file_type: str = "") -> tuple[str | None, str | None]:
    """返回 DataMate 文件对应的任务一分组和输出类型。"""
    lower = (file_name or "").lower()
    declared = (file_type or "").lower()
    if lower.endswith((".csv", ".tsv")) or declared in {"csv", "tsv"}:
        return "csv", "csv"
    if lower.endswith(".jsonl") or declared == "jsonl":
        return "jsonl", "jsonl"
    if lower.endswith(".json") or declared == "json":
        return "json", "json"
    if lower.endswith(".pdf") or declared in {"pdf", "application/pdf"}:
        return "pdf", "pdf"
    if lower.endswith((".txt", ".md")) or declared in {"txt", "text", ""}:
        return "text", "txt"
    return None, None


def count_source_file_groups(rows: Iterable[Sequence[str]]) -> tuple[dict[str, int], list[str]]:
    """统计 DataMate 文件列表中任务一支持的文件分组。"""
    counts = {group: 0 for group in TASK1_FILE_GROUPS}
    unsupported: list[str] = []
    for row in rows:
        fname = row[0] if len(row) > 0 else ""
        ftype = row[2] if len(row) > 2 else ""
        group, _out_type = classify_source_file(fname, ftype)
        if group:
            counts[group] += 1
        else:
            unsupported.append(fname)
    return counts, unsupported


def datamate_dataset_host_path(
    dataset_volume: str,
    dataset_id: str,
    file_name: str,
    file_path: str,
) -> str:
    """解析 DataMate 数据集文件在宿主机挂载卷中的路径。"""
    marker = "/dataset/"
    normalized = (file_path or "").replace("\\", "/")
    if marker in normalized:
        rel = normalized.split(marker, 1)[1].lstrip("/")
        if rel:
            return f"{dataset_volume}/{rel}"
    return f"{dataset_volume}/{dataset_id}/{file_name}"
