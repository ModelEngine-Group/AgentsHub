# -*- coding: utf-8 -*-
"""
部署脚本环境变量读取模块。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def load_runtime_env(
    root: Path | None = None,
    filename: str = ".env.runtime",
    keys: Iterable[str] | None = None,
) -> dict[str, str]:
    """读取 .env.runtime 中的配置值，且不覆盖已有进程环境变量。

    The file is a deployment-time runtime config. This helper keeps scripts
    reproducible while avoiding hard-coded credentials in source code.
    """

    repo_root = Path(root or Path(__file__).resolve().parents[1])
    env_path = repo_root / filename
    allowed = set(keys or [])
    loaded: dict[str, str] = {}

    if not env_path.exists():
        return loaded

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if allowed and key not in allowed:
            continue
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def get_runtime_secret(name: str, *fallback_names: str) -> str:
    """从环境变量或 .env.runtime 读取密钥，且不打印密钥内容。"""

    names = (name, *fallback_names)
    for item in names:
        value = os.environ.get(item)
        if value:
            return value
    load_runtime_env(keys=names)
    for item in names:
        value = os.environ.get(item)
        if value:
            return value
    return ""
