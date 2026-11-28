"""代码审查 MCP 服务的纯逻辑层（只读、仅标准库）。

单独成文件是为了让路径校验和读取逻辑可以脱离 MCP 框架单独测试。
本模块不写文件、不执行被审查代码、不访问网络、不使用 shell=True。
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

MAX_DIFF_LINES = 400
MAX_FILE_LINES = 200
MAX_FILE_BYTES = 200_000
GIT_TIMEOUT = 15

FORBIDDEN_PARTS = {".git", ".svn", "node_modules", "__pycache__", ".venv", "venv"}
SENSITIVE_NAMES = {
    ".env", ".env.local", ".env.production", "id_rsa", "id_ed25519",
    "credentials.json", "service-account.json", "secrets.yaml", "secrets.yml",
}

# 允许的仓库根目录白名单。为空表示不限制（保留本地开发与既有验收流程）；
# 生产部署应通过 MCP 服务的 --allowed-root 参数显式限定可读范围。
ALLOWED_ROOTS: list[Path] = []


def configure_allowed_roots(roots: list[str] | None) -> None:
    """设置允许审查的仓库根目录；空列表表示不限制。"""
    global ALLOWED_ROOTS
    ALLOWED_ROOTS = [
        Path(p).expanduser().resolve() for p in (roots or []) if str(p).strip()
    ]


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


# 扫描范围排除：对齐开源审查工具（PR-Agent 等）的 scope paths 实践，
# 生成代码 / 第三方代码 / 迁移脚本不该产生噪声候选。
# 注意这是「扫描范围」而非「豁免」：豁免由知识库《豁免清单》在定稿时决定。
EXCLUDED_DIR_PARTS = frozenset({
    "vendor", "vendors", "third_party", "3rdparty", "node_modules",
    "generated", "_generated", "autogen", "migrations", "snapshots",
    "__snapshots__", "build", "dist", ".venv", "venv", ".tox",
    ".mypy_cache", ".pytest_cache",
})
EXCLUDED_FILE_SUFFIXES = ("_pb2.py", "_pb2_grpc.py", "_generated.py", ".generated.py")


class ToolError(Exception):
    """可直接展示给智能体的、不含堆栈的可预期错误。"""


def _root(repo_path: str) -> Path:
    if not repo_path or not isinstance(repo_path, str):
        raise ToolError("repo_path 不能为空")
    root = Path(repo_path).expanduser().resolve()
    if not root.is_dir():
        raise ToolError(f"仓库目录不存在或不是目录：{repo_path}")
    if ALLOWED_ROOTS and not any(_within(root, allowed) for allowed in ALLOWED_ROOTS):
        allowed = "、".join(str(p) for p in ALLOWED_ROOTS)
        raise ToolError(
            f"repo_path 不在允许的仓库根目录内：{root}；本次服务仅允许：{allowed}"
        )
    return root


def _safe_file(root: Path, file_path: str) -> Path:
    """把相对路径限制在仓库内，并挡掉密钥类文件与版本库元数据。"""
    if not file_path or not isinstance(file_path, str):
        raise ToolError("file_path 不能为空")
    rel = Path(file_path)
    if rel.is_absolute():
        raise ToolError("file_path 必须是相对仓库根的路径，不接受绝对路径")
    if ".." in rel.parts or any(p.startswith("\\") for p in rel.parts):
        raise ToolError("file_path 不允许包含 .. 或路径穿越片段")
    if set(rel.parts) & FORBIDDEN_PARTS:
        raise ToolError("该路径位于版本库元数据或依赖目录，按只读策略拒绝读取")
    if rel.name in SENSITIVE_NAMES or rel.name.startswith(".env"):
        raise ToolError("该路径疑似凭据文件，按只读策略拒绝读取")

    target = (root / rel).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise ToolError("解析后的路径越出仓库根目录，拒绝读取") from None
    if not target.is_file():
        raise ToolError(f"文件不存在：{file_path}")
    if target.stat().st_size > MAX_FILE_BYTES:
        raise ToolError(f"文件超过 {MAX_FILE_BYTES} 字节，请用 read_diff 分段查看")
    return target


def _git(root: Path, args: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(root),
            timeout=GIT_TIMEOUT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            shell=False,
        )
    except FileNotFoundError:
        raise ToolError("本机找不到 git，无法读取 diff") from None
    except subprocess.TimeoutExpired:
        raise ToolError("git 调用超时，请缩小变更范围") from None
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        raise ToolError("git 调用失败：" + (detail[0] if detail else "未知原因"))
    return proc.stdout


def get_repo_meta(repo_path: str) -> dict[str, Any]:
    root = _root(repo_path)
    is_git = (root / ".git").exists()
    payload: dict[str, Any] = {
        "repo_root": root.name,
        "is_git_repo": is_git,
        "branch": None,
        "head": None,
        "changed_file_count": None,
    }
    if not is_git:
        payload["note"] = "该目录不是 git 仓库，diff 类工具不可用，read_file 仍可用"
        return payload
    payload["branch"] = _git(root, ["rev-parse", "--abbrev-ref", "HEAD"]).strip()
    payload["head"] = _git(root, ["rev-parse", "--short", "HEAD"]).strip()
    payload["changed_file_count"] = len(
        [x for x in _git(root, ["status", "--porcelain"]).splitlines() if x.strip()]
    )
    return payload


def list_changed_files(repo_path: str, base_ref: str = "HEAD~1") -> dict[str, Any]:
    root = _root(repo_path)
    if not (root / ".git").exists():
        raise ToolError("该目录不是 git 仓库，无法列出变更文件；可用 read_file 逐个读取")
    out = _git(root, ["diff", "--name-status", base_ref, "HEAD"])
    files = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        files.append({"status": parts[0], "path": parts[1] if len(parts) > 1 else ""})
    return {"base_ref": base_ref, "count": len(files), "files": files[:MAX_DIFF_LINES]}


def read_diff(
    repo_path: str,
    base_ref: str = "HEAD~1",
    file_path: str | None = None,
    max_lines: int = MAX_DIFF_LINES,
) -> dict[str, Any]:
    root = _root(repo_path)
    if not (root / ".git").exists():
        raise ToolError("该目录不是 git 仓库，无法读取 diff")
    args = ["diff", "--no-color", "--unified=3", base_ref, "HEAD"]
    if file_path:
        _safe_file(root, file_path)
        args += ["--", file_path]
    out = _git(root, args)
    lines = out.splitlines()
    limit = max(1, min(int(max_lines), MAX_DIFF_LINES))
    return {
        "base_ref": base_ref,
        "file_path": file_path,
        "total_lines": len(lines),
        "returned_lines": min(len(lines), limit),
        "truncated": len(lines) > limit,
        "diff": "\n".join(lines[:limit]),
        "note": "以 + 开头的是新文件内容；报告里的行号按新文件计算，不要按 diff 自身的行号写",
    }


def read_file(
    repo_path: str,
    file_path: str,
    start_line: int = 1,
    end_line: int = MAX_FILE_LINES,
) -> dict[str, Any]:
    root = _root(repo_path)
    target = _safe_file(root, file_path)
    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    lo = max(1, int(start_line))
    hi = min(len(lines), max(lo, int(end_line)), lo + MAX_FILE_LINES - 1)
    return {
        "file": file_path,
        "total_lines": len(lines),
        "start_line": lo,
        "end_line": hi,
        "lines": [{"n": i, "text": lines[i - 1]} for i in range(lo, hi + 1)],
        "note": "n 是物理行号，报告里引用时写 文件:n",
    }


# ---------------------------------------------------------------------------
# 确定性候选扫描
# 复用 Skill 包内 scripts/scan.py 这唯一一份规则实现，不另写一套规则，
# 避免「文本规则」与「机器规则」随版本漂移。
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent


def _skill_scanner():
    """加载同一 agent 目录下 Skill 明文包里的 scan.py（纯标准库）。"""
    scripts = HERE.parent / "skills" / "code-review-rules" / "scripts"
    if not (scripts / "scan.py").is_file():
        raise ToolError(
            "找不到规则实现 skills/code-review-rules/scripts/scan.py；"
            "mcps/ 与 skills/ 必须同属一个 agent 目录"
        )
    key = str(scripts)
    if key not in sys.path:
        sys.path.insert(0, key)
    import scan as scanner  # noqa: PLC0415 - 运行时按路径加载，故意不用顶层导入
    return scanner


def _added_lines(diff_text: str) -> set[tuple[str, int]]:
    """解析 unified diff，返回 (新文件路径, 新文件物理行号)，只含 + 行。"""
    added: set[tuple[str, int]] = set()
    current: str | None = None
    new_line = 0
    for raw in diff_text.splitlines():
        if raw.startswith("+++ "):
            target = raw[4:].strip()
            if target == "/dev/null":
                current = None
            else:
                current = target[2:] if target.startswith("b/") else target
            continue
        if raw.startswith("@@"):
            m = re.search(r"\+(\d+)", raw)
            new_line = int(m.group(1)) if m else 0
            continue
        if current is None:
            continue
        # 文件头标记统一形如 "+++ b/path" / "--- a/path"（前缀后带空格），
        # 已在上面按 "+++ " 处理；源码行本身可能以 +++ / --- 开头，不能据此误判。
        if raw.startswith("+"):
            added.add((current, new_line))
            new_line += 1
        elif raw.startswith("-"):
            continue
        elif raw.startswith("\\ No newline"):
            continue
        elif raw.startswith(" ") or raw == "":
            new_line += 1
    return added


def _is_excluded(rel: str) -> str | None:
    """返回扫描范围排除原因；None 表示纳入扫描。

    这是「扫描范围」而不是「豁免」：生成代码、第三方目录本来就不该产生候选，
    而测试目录等场景是否豁免，仍由知识库《豁免清单》在定稿时决定。
    """
    parts = Path(rel).parts
    hit = set(parts[:-1]) & EXCLUDED_DIR_PARTS
    if hit:
        return "生成/第三方目录 " + ", ".join(sorted(hit))
    name = parts[-1]
    for suffix in EXCLUDED_FILE_SUFFIXES:
        if name.endswith(suffix):
            return "生成文件后缀 " + suffix
    return None


def scan_candidates(
    repo_path: str,
    file_paths: list[str] | None = None,
    diff_text: str = "",
) -> dict[str, Any]:
    """按规则表做确定性扫描，产出候选发现项。

    - 传 diff_text：只保留 diff 的 `+` 行上的候选，与「只审新增行」约束对齐。
    - 不传 diff_text：扫描 file_paths（或仓库内全部 .py）的全文。
    - 生成代码 / 第三方目录按扫描范围规则排除，写进 excluded_files，不静默跳过。
    - 所有路径都要过 _safe_file 的只读白名单，读不到的记入 skipped_files。
    """
    root = _root(repo_path)
    scanner = _skill_scanner()

    if diff_text:
        added = _added_lines(diff_text)
        files = sorted({f for f, _ in added})
        scope = "added_lines"
    else:
        added = set()
        if file_paths:
            files = list(file_paths)
        else:
            files = [
                p.relative_to(root).as_posix()
                for p in sorted(root.rglob("*.py"))
                if not (set(p.parts) & FORBIDDEN_PARTS)
            ]
        scope = "whole_files"

    excluded: list[str] = []
    kept: list[str] = []
    for rel in files:
        reason = _is_excluded(rel)
        if reason:
            excluded.append(f"{rel}: {reason}")
        else:
            kept.append(rel)
    files = kept

    findings: list[dict[str, Any]] = []
    skipped: list[str] = []
    for rel in files:
        try:
            target = _safe_file(root, rel)
        except ToolError as exc:
            skipped.append(f"{rel}: {exc}")
            continue
        findings.extend(scanner.scan_file(target, rel))

    if diff_text and added:
        findings = [f for f in findings if (f["file"], f["line"]) in added]

    for item in findings:
        key = f"{item['rule_id']}|{item['file']}|{item['line']}"
        item["fingerprint"] = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        item["scope"] = scope

    return {
        "scope": scope,
        "file_count": len(files),
        "candidate_count": len(findings),
        "findings": findings,
        "excluded_files": excluded,
        "skipped_files": skipped,
        "note": (
            "findings 只是候选，是否上报由知识库《豁免清单》决定；"
            "fingerprint 由 rule_id|file|line 生成，供跨次审查比对；"
            "excluded_files 是按扫描范围主动跳过的生成/第三方文件，"
            "skipped_files 是读不到或被只读白名单拒绝的文件，两者都不静默忽略"
        ),
    }


def dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
