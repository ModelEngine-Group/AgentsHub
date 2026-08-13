"""流水线配置文件（YAML/JSON）加载：兼容 DataMate process.yaml 的字段。"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def load_pipeline_config(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """读取流水线配置，返回 (process 列表, 其余元数据)。

    支持两种形态：
    1. DataMate process.yaml 全量结构（dataset_path/export_path/process/...）
    2. 精简结构：直接就是 process 列表
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    data = json.loads(raw) if p.suffix == ".json" else yaml.safe_load(raw)
    if isinstance(data, list):
        return _normalize(data), {}
    if isinstance(data, dict) and "process" in data:
        meta = {k: v for k, v in data.items() if k != "process"}
        return _normalize(data["process"]), meta
    raise ValueError(f"无法识别的流水线配置结构: {p}")


def _normalize(process: List[Any]) -> List[Dict[str, Any]]:
    out = []
    for item in process:
        if isinstance(item, str):
            out.append({item: {}})
        elif isinstance(item, dict) and len(item) == 1:
            k, v = list(item.items())[0]
            out.append({k: dict(v) if v else {}})
        elif isinstance(item, dict) and "op" in item:
            # 智能体常用的宽松结构 {"op": name, "params": {...}}
            out.append({item["op"]: dict(item.get("params") or {})})
        else:
            raise ValueError(f"process 项结构非法: {item!r}")
    return out
