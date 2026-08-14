from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set

SAFETY_SUFFIX = "本结果仅用于慢性病随访数据分析和知识组织，不构成临床诊断或治疗建议。"


def dedupe_preserve(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def limit_paths(paths: Iterable[str], limit: int = 20) -> List[str]:
    result = dedupe_preserve(paths)
    return result[:limit]


def answer_with_safety(text: str) -> str:
    return f"{text} {SAFETY_SUFFIX}"


def entity_label(node: Dict[str, Any]) -> str:
    return str(node.get("display_name") or node.get("name") or node.get("id"))

