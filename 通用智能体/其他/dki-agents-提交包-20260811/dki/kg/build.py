"""从流水线产物聚合构建知识图谱。"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

from dki.kg.store import KnowledgeGraph

logger = logging.getLogger(__name__)


def extract_triples_from_sample(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从单个样本取三元组：优先 ext_params，其次解析 text JSON。"""
    ext = sample.get("ext_params") or {}
    if isinstance(ext, dict) and ext.get("triples"):
        return list(ext["triples"])
    text = sample.get("text") or ""
    if text.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, dict) and data.get("triples"):
                return list(data["triples"])
        except json.JSONDecodeError:
            pass
    return []


def build_kg_from_samples(samples: Iterable[Dict[str, Any]], validate: bool = True) -> KnowledgeGraph:
    """把任务二流水线的输出样本聚合为知识图谱（跨文档去重合并）。"""
    kg = KnowledgeGraph()
    n_docs = 0
    for s in samples:
        if s.get("execute_result") is False:
            continue
        triples = extract_triples_from_sample(s)
        if not triples:
            continue
        n_docs += 1
        source_default = str(s.get("fileId", ""))
        for t in triples:
            kg.add_triple(
                head=t["head"],
                head_type=t.get("head_type", ""),
                relation=t["relation"],
                tail=t["tail"],
                tail_type=t.get("tail_type", ""),
                confidence=float(t.get("confidence", 1.0)),
                source=t.get("source") or source_default,
                evidence=t.get("evidence", ""),
                validate=validate,
            )
    logger.info("KG 聚合完成：%d 文档 → %d 实体 / %d 三元组",
                n_docs, len(kg.entities), len(kg.triples))
    return kg


def build_kg_from_jsonl_dir(triples_dir: Path, validate: bool = True) -> KnowledgeGraph:
    """从目录下的 *.json / *.jsonl 三元组文件聚合构建图谱。"""
    kg = KnowledgeGraph()
    files = sorted(Path(triples_dir).glob("*.json*"))
    for f in files:
        content = f.read_text(encoding="utf-8").strip()
        rows: List[Dict[str, Any]] = []
        if not content:
            continue
        if content.startswith("["):
            rows = json.loads(content)
        elif content.startswith("{"):
            data = json.loads(content)
            rows = data.get("triples", [])
        else:  # jsonl
            for line in content.splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        for t in rows:
            kg.add_triple(
                head=t["head"], head_type=t.get("head_type", ""),
                relation=t["relation"],
                tail=t["tail"], tail_type=t.get("tail_type", ""),
                confidence=float(t.get("confidence", 1.0)),
                source=t.get("source", f.stem), evidence=t.get("evidence", ""),
                validate=validate,
            )
    return kg
