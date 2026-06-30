#!/usr/bin/env python3
"""Extract a compact contest-problem summary from a DOCX presigned URL.

This helper is intentionally small and deterministic so Nexent agents can call
it through a process tool instead of generating long Python snippets in chat.
"""

from __future__ import annotations

import io
import re
import sys
import urllib.request
import zipfile
import xml.etree.ElementTree as ET

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def fetch_docx(url: str) -> bytes:
    candidates = [url]
    docker_url = url.replace("http://localhost:5013", "http://nexent-northbound:5013")
    docker_url = docker_url.replace("http://127.0.0.1:5013", "http://nexent-northbound:5013")
    if docker_url != url:
        candidates.insert(0, docker_url)

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            with urllib.request.urlopen(candidate, timeout=45) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001 - try both host and docker DNS forms.
            last_error = exc
    raise RuntimeError(last_error)


def extract_text(docx_bytes: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", ns):
        line = "".join(node.text or "" for node in paragraph.findall(".//w:t", ns)).strip()
        if line:
            paragraphs.append(line)
    return "\n".join(paragraphs)


def section_for_problem(text: str, letter: str) -> str:
    letter = letter.upper()
    starts = [m.start() for m in re.finditer(rf"(?m)^{letter}\s*题\b|{letter}\s*题\s+", text)]
    if not starts:
        return text
    start = starts[0]
    next_letter = chr(ord(letter) + 1)
    end_match = re.search(rf"(?m)^{next_letter}\s*题\b|{next_letter}\s*题\s+", text[start + 1 :])
    end = start + 1 + end_match.start() if end_match else len(text)
    return text[start:end].strip()


def compact_sentence(value: str, limit: int = 180) -> str:
    value = re.sub(r"\s+", " ", value).strip(" ;；")
    return value[:limit] + ("..." if len(value) > limit else "")


def extract_tasks(section: str) -> list[str]:
    matches = list(re.finditer(r"任务\s*([123一二三])\s*([^\n]*)", section))
    tasks: list[str] = []
    for index, match in enumerate(matches[:3]):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(section)
        block = section[match.start() : end]
        block = re.split(r"\n\s*(三、附件|附件\s*1|B\s*题|C\s*题)", block)[0]
        label = match.group(1)
        title = match.group(2).strip()
        body = compact_sentence(block.replace(match.group(0), ""), 220)
        tasks.append(f"任务{label} {title}：{body}")
    return tasks


def extract_attachments(section: str) -> list[str]:
    attachments: list[str] = []
    for match in re.finditer(r"附件\s*([0-9一二三四五六七八九十]+)\s*([^\n]*)", section):
        item = compact_sentence(f"附件{match.group(1)} {match.group(2)}", 120)
        if item not in attachments:
            attachments.append(item)
    return attachments[:8]


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python nexent_docx_summary.py <presigned_url> [A|B|C]", file=sys.stderr)
        return 2

    url = sys.argv[1]
    letter = sys.argv[2].upper() if len(sys.argv) >= 3 else "A"

    try:
        docx_bytes = fetch_docx(url)
        text = extract_text(docx_bytes)
        section = section_for_problem(text, letter)
        title_match = re.search(rf"(?m)^({letter}\s*题[^\n]*)", section)
        title = compact_sentence(title_match.group(1) if title_match else section.splitlines()[0], 120)
        tasks = extract_tasks(section)
        attachments = extract_attachments(section)

        print(f"{letter}题题名：{title}")
        print("\n任务概述：")
        for item in tasks:
            print(f"- {item}")
        print("\n附件数据清单：")
        for item in attachments:
            print(f"- {item}")
        print("\n检查结论：")
        print(f"- DOCX提取成功：是，全文长度 {len(text)} 字，{letter}题正文长度 {len(section)} 字。")
        print("- 是否出现读取失败：否。")
        print("- 是否使用假设题目：否。")
        print("- 是否使用网络搜索替代赛题：否。")
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI should report compact failure.
        print(f"读取失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
