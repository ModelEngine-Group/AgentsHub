#!/usr/bin/env python3
"""代码审查助手 MCP 服务（只读）。

把「取变更范围 / 读 diff / 读带行号源码」暴露为 4 个 MCP 工具，
供 Nexent v2.7.0 平台接入。服务全程只读：不写文件、不执行被审查代码、
不访问网络、不使用 shell=True，并拒绝读取 .git 与凭据类文件。

启动（Nexent 平台接入用 SSE）::

    # 服务无鉴权，默认只监听回环地址
    python code_review_mcp_server.py --transport sse --host 127.0.0.1 --port 8766
    # Nexent 在 Docker 里时需绑定 0.0.0.0，并限定可读仓库根目录、只暴露给可信网络：
    python code_review_mcp_server.py --transport sse --host 0.0.0.0 --port 8766 --allowed-root DIR
    # MCP URL: http://<服务器地址>:8766/sse

stdio（MCP Inspector 等本地客户端）::

    python code_review_mcp_server.py

依赖：mcp>=1.2.0（见 requirements.txt）；逻辑层 fs_tools.py 只用标准库。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fs_tools import (  # noqa: E402
    ToolError,
    configure_allowed_roots,
    dumps,
    get_repo_meta as _get_repo_meta,
    list_changed_files as _list_changed_files,
    read_diff as _read_diff,
    read_file as _read_file,
    scan_candidates as _scan_candidates,
)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - 仅在未装依赖时触发
    raise SystemExit(
        "缺少依赖 mcp>=1.2.0，请先执行：pip install -r mcps/requirements.txt"
    ) from exc

# FastMCP 构造时若 host 是 localhost/127.0.0.1，会自动启用 DNS rebinding 保护，
# Host 白名单只含 localhost。Nexent 跑在 Docker 里，通过 host.docker.internal
# 访问本服务时会被 uvicorn 拒绝（421 Invalid Host header）。
# 因此构造时显式用 0.0.0.0，并在支持的 mcp 版本上补一条白名单，保留保护。
_ALLOWED_HOSTS = [
    "127.0.0.1:*",
    "localhost:*",
    "[::1]:*",
    "host.docker.internal:*",
]
_EXTRA_SETTINGS: dict = {}
try:
    import inspect

    from mcp.server.transport_security import TransportSecuritySettings

    if "transport_security" in inspect.signature(FastMCP.__init__).parameters:
        _EXTRA_SETTINGS["transport_security"] = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=_ALLOWED_HOSTS,
            allowed_origins=[f"http://{h}" for h in _ALLOWED_HOSTS],
        )
except ImportError:  # 旧版 mcp 无此模块；构造时用 0.0.0.0 时保护默认不启用，仍可访问
    pass

mcp = FastMCP(
    "code-review-fs",
    instructions=(
        "代码审查只读工具集：查看仓库元信息、列出相对 base_ref 的变更文件、"
        "读取统一 diff、读取带物理行号的源码。所有工具返回 JSON 字符串；"
        "出错时返回带 error 字段的 JSON。禁止用这些工具写入或修改任何文件。"
    ),
    host="0.0.0.0",
    **_EXTRA_SETTINGS,
)


def _guard(fn, *args, **kwargs) -> str:
    try:
        return dumps(fn(*args, **kwargs))
    except ToolError as exc:
        return dumps({"error": str(exc)})
    except Exception as exc:  # 不把堆栈泄漏给模型，只给可判定的错误类型
        return dumps({"error": f"工具内部错误（{type(exc).__name__}），请缩小范围后重试"})


@mcp.tool()
def get_repo_meta(repo_path: str) -> str:
    """查看仓库当前分支、HEAD 与未提交文件数。审查开始前先调一次确认路径可用。"""
    return _guard(_get_repo_meta, repo_path)


@mcp.tool()
def list_changed_files(repo_path: str, base_ref: str = "HEAD~1") -> str:
    """列出 base_ref..HEAD 之间的变更文件及状态（A/M/D）。"""
    return _guard(_list_changed_files, repo_path, base_ref)


@mcp.tool()
def read_diff(
    repo_path: str,
    base_ref: str = "HEAD~1",
    file_path: str = "",
    max_lines: int = 400,
) -> str:
    """读取 base_ref..HEAD 的统一 diff。file_path 为空表示全部变更，最多返回 400 行。"""
    return _guard(_read_diff, repo_path, base_ref, file_path or None, max_lines)


@mcp.tool()
def read_file(
    repo_path: str,
    file_path: str,
    start_line: int = 1,
    end_line: int = 200,
) -> str:
    """读取仓库内单个文件的指定行，返回带物理行号的原文。路径必须是相对路径。"""
    return _guard(_read_file, repo_path, file_path, start_line, end_line)


@mcp.tool()
def scan_candidates(
    repo_path: str,
    file_paths: str = "",
    diff_text: str = "",
) -> str:
    """按 CR-001~CR-008 规则做确定性扫描，返回候选发现项 JSON。

    diff_text 传入 read_diff 的结果时只保留 + 行上的候选；file_paths 是逗号分隔的
    相对路径；两者都不传则扫描仓库内全部 .py 文件全文。返回的每项带 fingerprint，
    findings 只是候选，是否上报还要查知识库《豁免清单》后决定。
    """
    paths = [p.strip() for p in file_paths.split(",") if p.strip()] or None
    return _guard(_scan_candidates, repo_path, paths, diff_text)


def main() -> int:
    parser = argparse.ArgumentParser(description="代码审查助手 MCP 服务（只读）")
    parser.add_argument(
        "--transport", default="stdio", choices=["stdio", "sse", "streamable-http"],
        help="传输方式；Nexent 平台接入请使用 sse",
    )
    parser.add_argument("--host", default="127.0.0.1", help="SSE/HTTP 监听地址")
    parser.add_argument("--port", type=int, default=8766, help="SSE/HTTP 监听端口")
    parser.add_argument(
        "--allowed-root", action="append", default=[], metavar="DIR",
        help="允许的仓库根目录，可重复指定；指定后所有工具只接受其下的 repo_path",
    )
    args = parser.parse_args()

    configure_allowed_roots(args.allowed_root)
    if args.allowed_root:
        print(
            "已限制可读仓库根目录：" + "、".join(args.allowed_root), file=sys.stderr
        )
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    if args.transport != "stdio":
        print(f"MCP URL: http://{args.host}:{args.port}/sse", file=sys.stderr)
    mcp.run(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
