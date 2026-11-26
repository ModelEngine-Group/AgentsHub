#!/usr/bin/env python3
"""把候选发现项转成 SARIF 2.1.0（OASIS 静态分析结果交换格式）。

用途：让本作品的检出结果能被 GitHub Code Scanning、VS Code SARIF Viewer、
Azure DevOps 等通用工具直接消费，不必自己造一套 dashboard。

这是**离线导出工具**，不进入智能体运行时。输入可以是：
  - `scan.py --json` 输出的发现项数组
  - `scan_candidates` 返回的对象（取其 findings）

用法：
    python scripts/scan.py --root <dir> --json | python scripts/to_sarif.py > results.sarif
    python scripts/to_sarif.py --input findings.json --output results.sarif

只读标准库。字段映射见 references/output-contract.md 第 11 节。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SARIF_SCHEMA = (
    "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/"
    "sarif-2.1/schema/sarif-schema-2.1.0.json"
)
TOOL_NAME = "code-review-rules"
TOOL_VERSION = "0.4.0"

# SARIF 只有 error / warning / note 三级；内部严重度按此映射
LEVEL = {"blocker": "error", "major": "warning", "minor": "note"}
RULE_TITLE = {
    "CR-001": "硬编码凭据",
    "CR-002": "SQL 字符串拼接",
    "CR-003": "裸 except / 吞异常",
    "CR-004": "资源未纳入上下文管理",
    "CR-005": "日志输出敏感字段",
    "CR-006": "可变默认参数",
    "CR-007": "assert 做入参校验",
    "CR-008": "HTTP 请求缺少超时",
}


def _findings(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("findings"), list):
        return [x for x in payload["findings"] if isinstance(x, dict)]
    raise ValueError("输入必须是发现项数组，或含 findings 数组的对象")


def _fingerprint(item: dict) -> str:
    if item.get("fingerprint"):
        return str(item["fingerprint"])
    key = f"{item.get('rule_id')}|{item.get('file')}|{item.get('line')}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def to_sarif(findings: list[dict]) -> dict:
    used: dict[str, dict] = {}
    results: list[dict] = []

    for item in findings:
        rid = str(item.get("rule_id") or "UNKNOWN")
        severity = str(item.get("severity") or "minor")
        level = LEVEL.get(severity, "note")
        message = str(item.get("message") or rid)
        uri = str(item.get("file") or "unknown").replace("\\", "/")
        line = int(item.get("line") or 1)

        # 只声明真正产生了结果的规则，避免把 8 条全塞进 driver.rules
        used.setdefault(rid, {
            "level": level,
            "severity": severity,
            "title": RULE_TITLE.get(rid, rid),
            "message": message,
        })

        result: dict[str, Any] = {
            "ruleId": rid,
            "level": level,
            "message": {"text": message},
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": {"uri": uri},
                    "region": {"startLine": line},
                }
            }],
            # SARIF 的 partialFingerprints 用于跨运行去重与生命周期跟踪
            "partialFingerprints": {"codeReviewRuleLine": _fingerprint(item)},
            "properties": {"severity": severity},
        }
        if item.get("snippet"):
            result["properties"]["snippet"] = str(item["snippet"])
        if item.get("scope"):
            result["properties"]["scope"] = str(item["scope"])
        results.append(result)

    rules = [
        {
            "id": rid,
            "name": rid,
            "shortDescription": {"text": meta["title"]},
            "fullDescription": {"text": meta["message"]},
            "defaultConfiguration": {"level": meta["level"]},
            "properties": {"severity": meta["severity"]},
        }
        for rid, meta in sorted(used.items())
    ]

    return {
        "$schema": SARIF_SCHEMA,
        "version": "2.1.0",
        "runs": [{
            "tool": {
                "driver": {
                    "name": TOOL_NAME,
                    "version": TOOL_VERSION,
                    "rules": rules,
                }
            },
            "results": results,
            "properties": {"candidateCount": len(results)},
        }],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="候选发现项 → SARIF 2.1.0")
    parser.add_argument("--input", type=Path, help="输入 JSON 文件；省略则从 stdin 读")
    parser.add_argument("--output", type=Path, help="输出 .sarif 文件；省略则写 stdout")
    args = parser.parse_args()

    try:
        raw = args.input.read_text(encoding="utf-8") if args.input else sys.stdin.read()
        payload = json.loads(raw)
        doc = to_sarif(_findings(payload))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"转换失败：{exc}", file=sys.stderr)
        return 2

    text = json.dumps(doc, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"已写入 {args.output}（{len(doc['runs'][0]['results'])} 条结果）", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
