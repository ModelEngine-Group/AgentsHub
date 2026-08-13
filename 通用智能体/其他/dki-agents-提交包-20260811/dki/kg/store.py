"""知识图谱三元组存储与查询（纯 Python 实现，无额外依赖）。

存储模型：
- entities: {实体名: {type, aliases, mention_count}}
- triples:  [{head, head_type, relation, tail, tail_type, confidence,
              count, sources, evidence:[{doc, sent}]}]

支持：三元组聚合去重、置信度合并、邻居/子图/路径查询、统计、
ECharts 关系图导出、JSON 持久化。
"""

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from dki.kg.schema import RELATION_TYPES, relation_zh, validate_triple

MAX_EVIDENCE_PER_TRIPLE = 5


class KnowledgeGraph:
    def __init__(self):
        self.entities: Dict[str, Dict[str, Any]] = {}
        self._triples: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        # 邻接表：entity -> [(relation, neighbor, direction)]
        self._adj: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)

    # ------------------------------------------------------------- 写入
    def add_entity(self, name: str, etype: str, aliases: Optional[List[str]] = None):
        ent = self.entities.setdefault(
            name, {"type": etype, "aliases": [], "mention_count": 0}
        )
        ent["mention_count"] += 1
        for a in aliases or []:
            if a and a != name and a not in ent["aliases"]:
                ent["aliases"].append(a)

    def add_triple(
        self,
        head: str,
        head_type: str,
        relation: str,
        tail: str,
        tail_type: str,
        confidence: float = 1.0,
        source: str = "",
        evidence: str = "",
        validate: bool = True,
    ) -> Tuple[bool, str]:
        """新增三元组（自动 Schema 校验 + 聚合去重 + 置信度合并）。"""
        if validate:
            ok, reason = validate_triple(head, head_type, relation, tail, tail_type)
            if not ok:
                return False, reason
        self.add_entity(head, head_type)
        self.add_entity(tail, tail_type)
        key = (head, relation, tail)
        if key in self._triples:
            t = self._triples[key]
            t["count"] += 1
            # 置信度按 noisy-or 合并：多次独立观察提升置信度
            t["confidence"] = round(1 - (1 - t["confidence"]) * (1 - confidence), 4)
            if source and source not in t["sources"]:
                t["sources"].append(source)
            if evidence and len(t["evidence"]) < MAX_EVIDENCE_PER_TRIPLE:
                t["evidence"].append({"doc": source, "sent": evidence[:200]})
        else:
            self._triples[key] = {
                "head": head,
                "head_type": head_type,
                "relation": relation,
                "tail": tail,
                "tail_type": tail_type,
                "confidence": round(confidence, 4),
                "count": 1,
                "sources": [source] if source else [],
                "evidence": [{"doc": source, "sent": evidence[:200]}] if evidence else [],
            }
            self._adj[head].append((relation, tail, "out"))
            self._adj[tail].append((relation, head, "in"))
        return True, "ok"

    # ------------------------------------------------------------- 查询
    @property
    def triples(self) -> List[Dict[str, Any]]:
        return list(self._triples.values())

    def get_entity(self, name: str) -> Optional[Dict[str, Any]]:
        if name in self.entities:
            return {"name": name, **self.entities[name]}
        # 别名归一
        for ent_name, ent in self.entities.items():
            if name in ent.get("aliases", []):
                return {"name": ent_name, **ent}
        return None

    def resolve_name(self, name: str) -> Optional[str]:
        ent = self.get_entity(name)
        return ent["name"] if ent else None

    def neighbors(
        self,
        name: str,
        relation: Optional[str] = None,
        direction: str = "both",
    ) -> List[Dict[str, Any]]:
        """查询实体的一跳邻居。direction: out/in/both。"""
        canonical = self.resolve_name(name)
        if canonical is None:
            return []
        out = []
        for rel, other, dirn in self._adj.get(canonical, []):
            if relation and rel != relation:
                continue
            if direction != "both" and dirn != direction:
                continue
            key = (canonical, rel, other) if dirn == "out" else (other, rel, canonical)
            t = self._triples.get(key, {})
            out.append(
                {
                    "entity": other,
                    "entity_type": self.entities.get(other, {}).get("type", ""),
                    "relation": rel,
                    "relation_zh": relation_zh(rel),
                    "direction": dirn,
                    "confidence": t.get("confidence", 0),
                    "count": t.get("count", 0),
                }
            )
        out.sort(key=lambda x: (-x["confidence"], -x["count"]))
        return out

    def triples_by_relation(self, relation: str) -> List[Dict[str, Any]]:
        return [t for t in self._triples.values() if t["relation"] == relation]

    def subgraph(self, names: Iterable[str], hops: int = 1) -> Dict[str, Any]:
        """以给定实体为中心的 N 跳子图（BFS）。"""
        seeds = [self.resolve_name(n) for n in names]
        seeds = [s for s in seeds if s]
        visited: Set[str] = set(seeds)
        frontier = deque([(s, 0) for s in seeds])
        while frontier:
            node, depth = frontier.popleft()
            if depth >= hops:
                continue
            for _, other, _ in self._adj.get(node, []):
                if other not in visited:
                    visited.add(other)
                    frontier.append((other, depth + 1))
        sub_triples = [
            t for (h, _, tl), t in self._triples.items() if h in visited and tl in visited
        ]
        return {
            "entities": {
                n: self.entities[n] for n in visited if n in self.entities
            },
            "triples": sub_triples,
        }

    def find_path(self, src: str, dst: str, max_hops: int = 3) -> List[Dict[str, Any]]:
        """BFS 最短路径（返回路径上的三元组序列）。"""
        s, d = self.resolve_name(src), self.resolve_name(dst)
        if not s or not d:
            return []
        prev: Dict[str, Tuple[str, str, str]] = {}
        visited = {s}
        q = deque([(s, 0)])
        while q:
            node, depth = q.popleft()
            if node == d:
                break
            if depth >= max_hops:
                continue
            for rel, other, dirn in self._adj.get(node, []):
                if other not in visited:
                    visited.add(other)
                    prev[other] = (node, rel, dirn)
                    q.append((other, depth + 1))
        if d not in visited or s == d:
            return []
        path = []
        cur = d
        while cur != s:
            node, rel, dirn = prev[cur]
            if dirn == "out":
                key = (node, rel, cur)
            else:
                key = (cur, rel, node)
            t = self._triples.get(key)
            if t:
                path.append(t)
            cur = node
        return list(reversed(path))

    def stats(self) -> Dict[str, Any]:
        by_etype: Dict[str, int] = defaultdict(int)
        for ent in self.entities.values():
            by_etype[ent["type"]] += 1
        by_rel: Dict[str, int] = defaultdict(int)
        for t in self._triples.values():
            by_rel[t["relation"]] += 1
        return {
            "n_entities": len(self.entities),
            "n_triples": len(self._triples),
            "entities_by_type": dict(by_etype),
            "triples_by_relation": dict(by_rel),
            "avg_confidence": round(
                sum(t["confidence"] for t in self._triples.values())
                / max(1, len(self._triples)),
                4,
            ),
        }

    def top_entities(self, etype: Optional[str] = None, k: int = 10) -> List[Dict[str, Any]]:
        """按度数排序的核心实体。"""
        items = []
        for name, ent in self.entities.items():
            if etype and ent["type"] != etype:
                continue
            items.append(
                {
                    "name": name,
                    "type": ent["type"],
                    "degree": len(self._adj.get(name, [])),
                    "mention_count": ent["mention_count"],
                }
            )
        items.sort(key=lambda x: (-x["degree"], -x["mention_count"]))
        return items[:k]

    # ------------------------------------------------------------- 导出
    def to_echarts_graph(self, max_nodes: int = 120) -> Dict[str, Any]:
        """导出 ECharts graph series 需要的 nodes/links/categories。"""
        from dki.kg.schema import ENTITY_TYPES

        categories = [{"name": zh} for zh in ENTITY_TYPES.values()]
        cat_index = {et: i for i, et in enumerate(ENTITY_TYPES)}
        top = {e["name"] for e in self.top_entities(k=max_nodes)}
        nodes = []
        for name in top:
            ent = self.entities[name]
            nodes.append(
                {
                    "name": name,
                    "category": cat_index.get(ent["type"], 0),
                    "symbolSize": min(60, 10 + 4 * len(self._adj.get(name, []))),
                    "value": ent["mention_count"],
                }
            )
        links = [
            {
                "source": t["head"],
                "target": t["tail"],
                "value": relation_zh(t["relation"]),
            }
            for (h, _, tl), t in self._triples.items()
            if h in top and tl in top
        ]
        return {"nodes": nodes, "links": links, "categories": categories}

    def to_dict(self) -> Dict[str, Any]:
        return {"entities": self.entities, "triples": self.triples}

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "KnowledgeGraph":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        kg = cls()
        kg.entities = data.get("entities", {})
        for t in data.get("triples", []):
            key = (t["head"], t["relation"], t["tail"])
            kg._triples[key] = t
            kg._adj[t["head"]].append((t["relation"], t["tail"], "out"))
            kg._adj[t["tail"]].append((t["relation"], t["head"], "in"))
        return kg

    @classmethod
    def from_triples(cls, triples: Iterable[Dict[str, Any]], validate: bool = True) -> "KnowledgeGraph":
        """从算子输出的三元组列表构建图谱（含校验与聚合）。"""
        kg = cls()
        for t in triples:
            kg.add_triple(
                head=t["head"],
                head_type=t.get("head_type", ""),
                relation=t["relation"],
                tail=t["tail"],
                tail_type=t.get("tail_type", ""),
                confidence=float(t.get("confidence", 1.0)),
                source=t.get("source", ""),
                evidence=t.get("evidence", ""),
                validate=validate,
            )
        return kg
