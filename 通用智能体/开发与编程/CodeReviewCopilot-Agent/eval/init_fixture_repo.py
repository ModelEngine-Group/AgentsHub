#!/usr/bin/env python3
"""生成验收用的 git fixture 仓库（只用标准库 + 本机 git）。

为什么需要它：`eval/samples/` 只是散文件，不是 git 仓库，`read_diff` 会直接返回
error，导致 README 里 A/B/C 三组消融对照全部退化成「无法获取变更」。本脚本把样例
装进一个真实仓库并做两次提交，让 `HEAD~1..HEAD` 有可审的 diff。

提交 1（基线）：6 个文件只有一句 docstring，另有一个 `legacy.py` 含硬编码密钥。
提交 2（缺陷）：写入 5 个带缺陷的样例 + 1 个干净负样本，并删除 `legacy.py`。

于是 diff 里：
  - 10 条候选全部出现在 `+` 行上，与离线扫描器的数字对齐
  - `utils/clean.py` 是干净负样本，应 0 条，用来验证「不凑数」
  - `legacy.py` 整体是 `-` 行（含 CR-001），专门验证「删除行/删除文件不报」
  若智能体报出 `legacy.py` 或 `utils/clean.py`，说明第 4、9 条约束没生效。

用法：
    python 提交包/eval/init_fixture_repo.py
    python 提交包/eval/init_fixture_repo.py --target /path/to/empty_dir

生成的仓库是本地验收产物，**不要提交进 PR**（默认落在工作文档目录）。
目标目录必须不存在或为空；脚本不会删除任何已有内容。
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLES = HERE / "samples"
DEFAULT_TARGET = HERE.parents[1] / "工作文档" / "_fixture_repo"

# 基线：每个文件只有 docstring，保证第二次提交的所有内容都是新增行
BASELINE = {
    "services/payment.py": '"""支付服务（干净基线）。"""\n',
    "services/sync.py": '"""同步任务（干净基线）。"""\n',
    "utils/http.py": '"""HTTP 工具（干净基线）。"""\n',
    "utils/validate.py": '"""入参校验（干净基线）。"""\n',
    "tests/test_login.py": '"""登录测试（干净基线）。"""\n',
    "utils/clean.py": '"""合规样例（干净基线）。"""\n',
    # 故意在基线里留一个命中 CR-001 的文件，第二次提交删掉它
    "legacy.py": (
        '"""已下线的旧配置，本次提交删除。"""\n'
        'API_KEY = "sk-live-legacy0000000"\n'
    ),
    # 标记文件：只有带它的目录才允许 --recreate 重建
    ".code_review_fixture": "本目录由 init_fixture_repo.py 生成，可用 --recreate 重建。\n",
}

DEFECT_FILES = [
    "services/payment.py",
    "services/sync.py",
    "utils/http.py",
    "utils/validate.py",
    "utils/clean.py",
    "tests/test_login.py",
]


def git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(root), capture_output=True,
        text=True, encoding="utf-8", errors="replace", check=False, shell=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 失败：{proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout.strip()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _on_rm_error(func, path: str, exc_info) -> None:
    """Windows 上 git object 文件带只读属性，rmtree 需要先去掉只读再重试。"""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        raise exc_info[1].with_traceback(exc_info[2])


def main() -> int:
    parser = argparse.ArgumentParser(description="生成代码审查验收用 git fixture 仓库")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET,
                        help="目标目录，必须不存在或为空")
    parser.add_argument("--recreate", action="store_true",
                        help="仅当目标目录带本脚本生成的标记文件时才允许重建")
    args = parser.parse_args()
    target: Path = args.target.resolve()

    if target.exists() and any(target.iterdir()):
        marker = target / ".code_review_fixture"
        if args.recreate and marker.is_file():
            print(f"重建本脚本此前生成的 fixture：{target}")
            shutil.rmtree(target, onexc=_on_rm_error)
        else:
            print(f"目标目录已存在且非空，脚本不会删除任何内容：{target}", file=sys.stderr)
            print("请换一个 --target，或确认是本脚本生成的目录后加 --recreate。", file=sys.stderr)
            return 2
    target.mkdir(parents=True, exist_ok=True)

    try:
        # ---- 提交 1：干净基线 ----
        for rel, text in BASELINE.items():
            write(target / rel, text)
        git(target, "init", "-q")
        git(target, "config", "--local", "user.name", "Fixture Bot")
        git(target, "config", "--local", "user.email", "fixture@localhost")
        git(target, "add", "-A")
        git(target, "commit", "-q", "-m", "baseline: 干净代码与待下线的旧配置")

        # ---- 提交 2：植入缺陷、删除 legacy.py ----
        (target / "legacy.py").unlink()
        for rel in DEFECT_FILES:
            src = SAMPLES / rel
            write(target / rel, src.read_text(encoding="utf-8"))
        git(target, "add", "-A")
        git(target, "commit", "-q", "-m", "feat: 引入 5 个评审样例缺陷")
    except Exception as exc:
        print(f"生成失败：{exc}", file=sys.stderr)
        print(f"已生成的部分留在 {target}，请检查后手动处理。", file=sys.stderr)
        return 1

    changed = git(target, "diff", "--name-status", "HEAD~1", "HEAD")
    print("fixture 仓库已生成：", target)
    print("\nHEAD~1..HEAD 变更：")
    print(changed)
    print("\n验收表里的输入请写成：")
    print(f'  帮我审查 {target.as_posix()} 最近一次提交的改动')
    print("\n提示：legacy.py 状态为 D，diff 全是 - 行且含硬编码密钥；")
    print("      报告里出现 legacy.py 就是「只审 + 行」约束失效。")
    print("\n这是本地验收产物，不要提交进 PR。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
