#!/usr/bin/env python3
"""CR-001..CR-008 确定性扫描器。

只读标准库。遍历 --root 下的 .py 文件，按行匹配规则，输出 JSON 发现项。
刻意不做路径豁免：测试目录里的 print / assert 也会进结果，是否豁免由
智能体查知识库《豁免清单》决定。

用法：
    python scan.py --root /path/to/repo --json
    python scan.py --root /path/to/repo
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SEVERITY = {
    "CR-001": "blocker",
    "CR-002": "blocker",
    "CR-003": "major",
    "CR-004": "major",
    "CR-005": "major",
    "CR-006": "minor",
    "CR-007": "minor",
    "CR-008": "minor",
}

MESSAGE = {
    "CR-001": "凭据被硬编码为字符串字面量，随代码分发即泄漏",
    "CR-002": "SQL 语句由字符串拼接或格式化生成，存在注入风险",
    "CR-003": "裸 except 或异常被静默吞掉，故障无法定位",
    "CR-004": "文件句柄未纳入 with 上下文管理，异常路径下可能泄漏",
    "CR-005": "日志或打印语句输出了敏感字段",
    "CR-006": "函数默认值是可变对象，跨调用会共享状态",
    "CR-007": "用 assert 校验外部输入，python -O 下校验会被移除",
    "CR-008": "HTTP 请求未设置超时，可能长期挂起",
}

RE_SECRET = re.compile(
    r"\b(api[_-]?key|access[_-]?key|private[_-]?key|password|passwd|secret|token)\b"
    r"\s*[:=]\s*['\"]([^'\"]{8,})['\"]",
    re.IGNORECASE,
)
RE_SQL = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|DROP)\b", re.IGNORECASE)
RE_SQL_BUILD = re.compile(r"f['\"]|\.format\(|%\s*\(|\+\s*['\"]|['\"]\s*\+")
RE_OPEN = re.compile(r"(?<![A-Za-z0-9_])open\s*\(")
RE_LOG = re.compile(r"\b(print|logger\.\w+|logging\.\w+|log\.\w+)\s*\(")
RE_SENSITIVE = re.compile(
    r"\b(password|passwd|token|secret|id_card|api[_-]?key)\b", re.IGNORECASE
)
RE_MUTABLE_DEFAULT = re.compile(
    r"\bdef\s+\w+\s*\([^)]*?\b\w+\s*=\s*(\[\]|\{\}|set\(\)|list\(\)|dict\(\))"
)
RE_HTTP = re.compile(r"\brequests\.(get|post|put|delete|patch)\s*\(")

PLACEHOLDER = re.compile(
    r"your|example|dummy|placeholder|xxxx|<[^>]*>|\$\{|os\.environ|getenv",
    re.IGNORECASE,
)

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".idea", ".vscode"}


def _clean(line: str) -> str:
    return line.strip()[:160]


def _noqa(line: str, rule_id: str) -> bool:
    return f"noqa: {rule_id}" in line


def scan_file(path: Path, rel: str) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (UnicodeDecodeError, OSError):
        return []

    found: list[dict] = []
    seen: set[tuple[str, int]] = set()

    def add(rule_id: str, lineno: int, snippet: str) -> None:
        key = (rule_id, lineno)
        if key in seen:
            return
        seen.add(key)
        found.append({
            "rule_id": rule_id,
            "severity": SEVERITY[rule_id],
            "file": rel,
            "line": lineno,
            "snippet": _clean(snippet),
            "message": MESSAGE[rule_id],
        })

    for i, raw in enumerate(lines):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lineno = i + 1

        # CR-001 硬编码凭据
        m = RE_SECRET.search(raw)
        if m and not PLACEHOLDER.search(m.group(2)) and not _noqa(raw, "CR-001"):
            add("CR-001", lineno, raw)

        # CR-002 SQL 字符串拼接
        if (
            "execute(" in raw
            and RE_SQL.search(raw)
            and RE_SQL_BUILD.search(raw)
            and not _noqa(raw, "CR-002")
        ):
            add("CR-002", lineno, raw)

        # CR-003 裸 except / 吞异常
        if re.match(r"^\s*except\s*:", raw) or re.search(r"^\s*except\b.*:\s*pass\s*$", raw):
            if not _noqa(raw, "CR-003"):
                add("CR-003", lineno, raw)
        elif re.match(r"^\s*except\b[^#]*:\s*(#.*)?$", raw):
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if nxt == "pass" and not _noqa(raw, "CR-003"):
                add("CR-003", lineno, raw)

        # CR-004 资源未纳入 with
        if (
            RE_OPEN.search(raw)
            and "with " not in raw
            and not _noqa(raw, "CR-004")
        ):
            add("CR-004", lineno, raw)

        # CR-005 日志输出敏感字段
        if RE_LOG.search(raw) and RE_SENSITIVE.search(raw) and not _noqa(raw, "CR-005"):
            add("CR-005", lineno, raw)

        # CR-006 可变默认参数
        if RE_MUTABLE_DEFAULT.search(raw) and not _noqa(raw, "CR-006"):
            add("CR-006", lineno, raw)

        # CR-007 assert 做校验（不做路径豁免）
        if re.match(r"^\s*assert\s+", raw) and not _noqa(raw, "CR-007"):
            add("CR-007", lineno, raw)

        # CR-008 HTTP 请求缺超时
        if RE_HTTP.search(raw) and "timeout" not in raw and not _noqa(raw, "CR-008"):
            add("CR-008", lineno, raw)

    return found


def scan_tree(root: Path) -> list[dict]:
    results: list[dict] = []
    if root.is_file():
        return scan_file(root, root.name)
    files = sorted(
        p for p in root.rglob("*.py")
        if not any(part in SKIP_DIRS for part in p.parts)
    )
    for p in files:
        rel = p.relative_to(root).as_posix()
        results.extend(scan_file(p, rel))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="CR-001..CR-008 确定性扫描器")
    parser.add_argument("--root", required=True, help="仓库根目录或单个 .py 文件")
    parser.add_argument("--json", action="store_true", help="输出 JSON 数组")
    parser.add_argument("--rule", help="只输出指定规则，如 CR-002")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"路径不存在：{root}", file=sys.stderr)
        return 2

    findings = scan_tree(root)
    if args.rule:
        findings = [f for f in findings if f["rule_id"] == args.rule.upper()]

    if args.json:
        print(json.dumps(findings, ensure_ascii=False, indent=2))
    else:
        if not findings:
            print("未发现符合规则的问题（扫描器不做豁免判断）。")
        for f in findings:
            print(f"{f['rule_id']} | {f['file']}:{f['line']} | {f['severity']} | {f['message']}")
        print(f"候选发现项合计：{len(findings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
