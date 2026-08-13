"""内容哈希级 final3/blind 防泄漏门（供所有下游消费入口调用）。

历史盲测集（final3 等）只允许做一次性评测与历史复现，**禁止**进入训练、校准、
选模、路由、profiling、阈值搜索、量化校准、误差分析等会“看到答案”的环节。此前
只有 `profile_ner_workload.py` 做**文件名**检查——一旦文件被重命名即绕过。

本模块用受保护清单里的 **SHA-256 内容哈希**判定，重命名也拒绝（fail closed）。
纯标准库实现，任何脚本都可在重依赖导入前先调用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from dki.evidence import reject_forbidden_file

_DEFAULT_MANIFEST = Path(__file__).resolve().parent.parent.parent / "evidence" / "manifests" / "protected_baseline.json"

# 会“看到答案”的消费用途；这些用途一律拒绝 final3/blind 内容。
CONSUMER_PURPOSES = (
    "training",
    "calibration",
    "selection",
    "model_selection",
    "routing",
    "profiling",
    "threshold_search",
    "quantization",
    "error_analysis",
)


def _load_manifest(manifest_path: Path | None) -> dict:
    path = manifest_path or _DEFAULT_MANIFEST
    if not path.is_file():
        # 缺清单不放行：宁可报错也不静默跳过内容门。
        raise FileNotFoundError(f"protected manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def guard_path(path: str | Path, purpose: str, *, manifest_path: Path | None = None) -> None:
    """若 ``path`` 的内容哈希被清单标记为该 ``purpose`` 禁用则抛错。"""
    if purpose not in CONSUMER_PURPOSES:
        raise ValueError(f"unknown consumer purpose: {purpose}")
    target = Path(path)
    if not target.is_file():
        return
    reject_forbidden_file(target, _load_manifest(manifest_path), purpose)


def guard_paths(paths: Iterable[str | Path], purpose: str, *, manifest_path: Path | None = None) -> None:
    manifest = _load_manifest(manifest_path)
    for path in paths:
        target = Path(path)
        if target.is_file():
            reject_forbidden_file(target, manifest, purpose)


def guard_dir(directory: str | Path, purpose: str, *, pattern: str = "*.jsonl",
              manifest_path: Path | None = None) -> None:
    """对目录内所有匹配文件做内容门（用于 --ood-data 之类的目录入参）。"""
    root = Path(directory)
    if not root.is_dir():
        return
    guard_paths(sorted(root.rglob(pattern)), purpose, manifest_path=manifest_path)
