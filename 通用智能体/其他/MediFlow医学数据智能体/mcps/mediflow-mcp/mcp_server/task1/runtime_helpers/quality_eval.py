# -*- coding: utf-8 -*-
"""任务一源格式保留输出质量评估。"""
import argparse
from collections import Counter
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

RESIDUAL_PATTERNS = [
    ("html_tag", re.compile(r"</?[A-Za-z][A-Za-z0-9:_-]*(?:\s+[^<>]*)?/?>")),
    ("url", re.compile(r"https?://\S+|(?:www|w{2,3})\s*[.。．]\s*[^\s，。；;、]{1,80}", re.IGNORECASE)),
    ("emoji", re.compile(r"[\U0001F300-\U0001FFFF]")),
    ("control_char", re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")),
    ("his_export_banner", re.compile(r"(?:HIS|EMR|电子病历|医院信息).{0,8}(?:导出|生成)|系统自动(?:导出|生成)", re.IGNORECASE)),
    ("form_filler_meta", re.compile(r"填表时间|填表人|录入人|审核人")),
    ("social_contact", re.compile(r"@[A-Za-z0-9_\-\u4e00-\u9fff]{1,30}|微信|QQ[:：]?\d{4,}")),
    ("test_banner", re.compile(r"仅供(?:内部)?测试|测试数据请忽略")),
    ("controlled_noise_marker", re.compile(r"controlled_noise_\d{8}\??", re.IGNORECASE)),
    ("question_mark_mojibake", re.compile(r"(?<![\u4e00-\u9fffA-Za-z0-9])\?{2,}(?![\u4e00-\u9fffA-Za-z0-9])")),
    ("markup_noise", re.compile(r"###|@@[^@]{1,100}@@")),
    ("fullwidth_alphanumeric", re.compile(r"[\u3000\uff10-\uff19\uff21-\uff3a\uff41-\uff5a]")),
]


SUPPORTED_SUFFIXES = {".txt", ".csv", ".json", ".jsonl"}
DOCUMENT_HEADER_PATTERNS = (
    ("medical_case_header", re.compile(r"(?m)^\s*病例\s*(\d+)\s*$")),
    ("record_id_header", re.compile(r"(?m)^\s*record_id\s*[:：]\s*(\S+)\s*$")),
)


def iter_candidate_files(paths: Iterable[str]) -> Iterable[Path]:
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.suffix.lower() in SUPPORTED_SUFFIXES:
                    yield child
        elif path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def _scan_text(text: str, result: Dict, location: str) -> None:
    if not text.strip():
        result["empty_text"].append({"location": location})
        return
    for name, pattern in RESIDUAL_PATTERNS:
        match = pattern.search(text)
        if match:
            if name == "his_export_banner" and "免疫系统" in match.group(0):
                continue
            result["residual_noise"].append({
                "location": location,
                "issue": name,
                "match": match.group(0)[:80],
            })


def _scan_duplicate_document_headers(text: str, result: Dict, location: str) -> None:
    for name, pattern in DOCUMENT_HEADER_PATTERNS:
        counts = Counter(match.strip() for match in pattern.findall(text))
        duplicates = {value: count for value, count in counts.items() if count > 1}
        if duplicates:
            result["duplicate_content"].append({
                "location": location,
                "issue": name,
                "count": sum(duplicates.values()),
                "values": duplicates,
            })


def _iter_json_strings(value: Any, path: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _iter_json_strings(child, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            yield from _iter_json_strings(child, f"{path}[{idx}]")
    elif isinstance(value, str):
        yield path, value


def evaluate_file(path: Path) -> Dict:
    result = {
        "path": str(path),
        "format": path.suffix.lower().lstrip(".") or "txt",
        "records": 0,
        "parse_errors": [],
        "empty_text": [],
        "residual_noise": [],
        "duplicate_content": [],
    }
    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".txt":
        result["records"] = 1 if raw.strip() else 0
        _scan_text(raw, result, "text")
        _scan_duplicate_document_headers(raw, result, "text")
        return result

    if suffix == ".csv":
        try:
            rows = list(csv.DictReader(raw.splitlines()))
        except Exception as exc:
            result["parse_errors"].append({"location": "csv", "error": str(exc)[:160]})
            return result
        result["records"] = len(rows)
        if not rows:
            result["empty_text"].append({"location": "csv"})
        for row_idx, row in enumerate(rows, 1):
            for key, value in row.items():
                if isinstance(value, str) and value.strip():
                    _scan_text(value, result, f"row{row_idx}.{key}")
        return result

    if suffix == ".json":
        try:
            obj = json.loads(raw)
        except Exception as exc:
            result["parse_errors"].append({"location": "json", "error": str(exc)[:160]})
            return result
        strings = list(_iter_json_strings(obj))
        result["records"] = len(obj) if isinstance(obj, list) else 1
        if not strings:
            result["empty_text"].append({"location": "json"})
        for location, value in strings:
            if value.strip():
                _scan_text(value, result, location)
        return result

    if suffix == ".jsonl":
        for lineno, line in enumerate(raw.splitlines(), 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except Exception as exc:
                result["parse_errors"].append({"location": f"line{lineno}", "error": str(exc)[:160]})
                continue
            result["records"] += 1
            for location, value in _iter_json_strings(obj, f"line{lineno}"):
                if value.strip():
                    _scan_text(value, result, location)
        if result["records"] == 0:
            result["empty_text"].append({"location": "jsonl"})
        return result

    result["parse_errors"].append({"location": "file", "error": f"unsupported suffix: {suffix}"})
    return result


def summarize(results: List[Dict], min_records: int = 1) -> Dict:
    totals = {
        "files": len(results),
        "records": sum(int(r.get("records", 0) or 0) for r in results),
        "parse_errors": sum(len(r.get("parse_errors", [])) for r in results),
        "empty_text": sum(len(r.get("empty_text", [])) for r in results),
        "residual_noise": sum(len(r.get("residual_noise", [])) for r in results),
        "duplicate_content": sum(len(r.get("duplicate_content", [])) for r in results),
    }
    ok = (
        totals["files"] > 0
        and totals["records"] >= min_records
        and totals["parse_errors"] == 0
        and totals["empty_text"] == 0
        and totals["residual_noise"] == 0
        and totals["duplicate_content"] == 0
    )
    return {"pass": ok, "totals": totals, "files": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    files = list(iter_candidate_files(args.paths))
    report = summarize([evaluate_file(path) for path in files], args.min_records)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        totals = report["totals"]
        print(
            "TASK1_PRESERVED_OUTPUT_QUALITY "
            f"pass={report['pass']} files={totals['files']} records={totals['records']} "
            f"parse_errors={totals['parse_errors']} empty_text={totals['empty_text']} "
            f"residual_noise={totals['residual_noise']} "
            f"duplicate_content={totals['duplicate_content']}"
        )
        for item in report["files"]:
            issues = (
                len(item["parse_errors"])
                + len(item["empty_text"])
                + len(item["residual_noise"])
                + len(item["duplicate_content"])
            )
            print(f"- {item['path']}: format={item['format']} records={item['records']} issues={issues}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
