"""任务一 PDF 前置解析与质量证据。"""

from __future__ import annotations

import re
from pathlib import Path

from markdown_it import MarkdownIt

from mcp_server.task1.mineru_client import MineruAgentClient


PARSER_ID = "MinerUAgentRemoteParser"


def markdown_to_plain_text(markdown: str) -> str:
    """把 MinerU Markdown 转为适合 DataMate 文本清洗链的纯文本。"""

    tokens = MarkdownIt("commonmark").parse(markdown)
    blocks: list[str] = []
    for token in tokens:
        if token.type == "inline" and token.content.strip():
            blocks.append(token.content.strip())
        elif token.type in {"fence", "code_block"} and token.content.strip():
            blocks.append(token.content.strip())
    text = "\n".join(blocks)
    text = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\s*\|\s*", " | ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def inspect_pdf_capability(
    _datamate_base: str = "",
    *,
    client: MineruAgentClient | None = None,
) -> dict:
    """检查远程 MinerU Agent API 是否可达。"""

    active_client = client or MineruAgentClient.from_env()
    service_available, service_detail = active_client.probe()
    return {
        "available": service_available,
        "parser_id": PARSER_ID,
        "service_available": service_available,
        "service_detail": service_detail,
        "mineru_api": active_client.base_url,
        "output_format": "txt",
        "limits": {"max_file_mb": 10, "max_pages": 20, "single_file": True},
        "deployment_hint": "检查服务器到 mineru.net 的网络连接，或覆盖 CCF_MINERU_API",
    }


def parse_pdf_files(
    source_paths: list[Path],
    output_dir: Path,
    *,
    client: MineruAgentClient | None = None,
) -> tuple[list[tuple[Path, str]], list[dict]]:
    """将 PDF 解析为 TXT，并返回 DataMate 注册参数与解析证据。"""

    active_client = client or MineruAgentClient.from_env()
    output_dir.mkdir(parents=True, exist_ok=True)
    converted: list[tuple[Path, str]] = []
    reports: list[dict] = []
    for index, source in enumerate(source_paths, start=1):
        result = active_client.parse_file(source)
        plain_text = markdown_to_plain_text(result.markdown)
        if not plain_text:
            raise RuntimeError(f"MinerU 未从 {source.name} 提取到可清洗文本")
        target = output_dir / f"{source.stem}_MinerU解析_{index}.txt"
        target.write_text(plain_text, encoding="utf-8", newline="\n")
        report = result.as_dict()
        report.update({"output_file": target.name, "plain_text_chars": len(plain_text)})
        reports.append(report)
        converted.append((target, "txt"))
    return converted, reports


def summarize_pdf_evidence(
    source_paths: list[Path],
    output_paths: list[Path],
    parse_reports: list[dict] | None = None,
) -> dict:
    """形成 PDF 转文本和后续清洗的可核验证据。"""

    output_chars = sum(
        len(path.read_text(encoding="utf-8", errors="replace")) for path in output_paths
    )
    reports = parse_reports or []
    return {
        "source_pdf_files": len(source_paths),
        "output_text_files": len(output_paths),
        "output_text_chars": output_chars,
        "parser": PARSER_ID,
        "remote_tasks": reports,
        "output_format": "txt",
        "conversion_verified": bool(source_paths) and bool(output_paths) and output_chars > 0,
    }
