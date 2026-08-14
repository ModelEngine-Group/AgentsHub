from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from runtime_common.common import ensure_directory, relative_to_project, resolve_path, write_json

DEFAULT_TRACE_DIR = "outputs/mcp_traces"
DEFAULT_TRACE_FILE = "outputs/mcp_traces/mcp_tool_calls.jsonl"
DEFAULT_SUMMARY_FILE = "outputs/mcp_traces/mcp_trace_summary.json"


def get_trace_dir() -> Path:
    return resolve_path(os.getenv("CHRONICCARE_TRACE_DIR", DEFAULT_TRACE_DIR))


def get_trace_file() -> Path:
    return resolve_path(os.getenv("CHRONICCARE_TRACE_FILE", DEFAULT_TRACE_FILE))


def get_trace_summary_file() -> Path:
    return resolve_path(os.getenv("CHRONICCARE_TRACE_SUMMARY_FILE", DEFAULT_SUMMARY_FILE))


def append_trace(record: Dict[str, Any]) -> None:
    trace_file = get_trace_file()
    ensure_directory(trace_file.parent)
    try:
        with trace_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_json(get_trace_summary_file(), summarize_traces())
    except Exception as exc:
        print(f"[trace_logger] warning: failed to write trace: {exc}")


def _load_trace_records() -> List[Dict[str, Any]]:
    trace_file = get_trace_file()
    if not trace_file.exists():
        return []

    records: List[Dict[str, Any]] = []
    try:
        with trace_file.open("r", encoding="utf-8") as file:
            for line in file:
                row = line.strip()
                if not row:
                    continue
                try:
                    doc = json.loads(row)
                except json.JSONDecodeError:
                    continue
                if isinstance(doc, dict):
                    records.append(doc)
    except Exception as exc:
        print(f"[trace_logger] warning: failed to read traces: {exc}")
        return []
    return records


def load_recent_traces(limit: int = 50) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    records = _load_trace_records()
    if not records:
        return []

    unique_records: List[Dict[str, Any]] = []
    seen_trace_ids = set()
    for item in reversed(records):
        trace_id = str(item.get("trace_id") or "")
        dedupe_key = trace_id or json.dumps(item, ensure_ascii=False, sort_keys=True)
        if dedupe_key in seen_trace_ids:
            continue
        seen_trace_ids.add(dedupe_key)
        unique_records.append(item)
    unique_records.reverse()
    return unique_records[-limit:]


def summarize_traces() -> Dict[str, Any]:
    trace_file = get_trace_file()
    traces = load_recent_traces(limit=1000000)
    tool_counts: Dict[str, int] = {}
    success_calls = 0
    error_calls = 0
    total_latency = 0.0
    for item in traces:
        tool_name = str(item.get("tool_name", "unknown"))
        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        latency = item.get("latency_ms", 0)
        if isinstance(latency, (int, float)):
            total_latency += float(latency)
        if item.get("status") == "success":
            success_calls += 1
        else:
            error_calls += 1
    total_calls = len(traces)
    summary = {
        "status": "success",
        "total_calls": total_calls,
        "success_calls": success_calls,
        "error_calls": error_calls,
        "success_rate": round(success_calls / total_calls, 4) if total_calls else 0.0,
        "tool_counts": tool_counts,
        "avg_latency_ms": round(total_latency / total_calls, 2) if total_calls else 0.0,
        "trace_file": relative_to_project(trace_file),
        "summary_file": relative_to_project(get_trace_summary_file()),
        "recent_trace": traces[-1] if traces else None,
    }
    return summary
