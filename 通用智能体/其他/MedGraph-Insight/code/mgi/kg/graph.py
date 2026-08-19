"""知识图谱存储：networkx 内存图 + JSON 持久化 + SQLite 导出（供 NL2SQL）。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import networkx as nx

from .schema import ENTITY_TYPES, RELATION_TYPES, Triple


class MedicalKG:
    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()

    # ---- 构建 -----------------------------------------------------------

    def add_triples(self, triples: list[Triple]) -> None:
        for t in triples:
            self.g.add_node(t.head, etype=t.head_type)
            self.g.add_node(t.tail, etype=t.tail_type)
            self.g.add_edge(
                t.head, t.tail, key=t.relation,
                relation=t.relation, confidence=t.confidence,
                source=t.source, evidence=t.evidence,
            )

    # ---- 查询 -----------------------------------------------------------

    def neighbors(self, entity: str, relation: str | None = None) -> list[tuple[str, str, float]]:
        out = []
        if entity in self.g:
            for _, tail, data in self.g.out_edges(entity, data=True):
                if relation is None or data["relation"] == relation:
                    out.append((data["relation"], tail, data["confidence"]))
        return out

    def reverse_neighbors(self, entity: str, relation: str | None = None) -> list[tuple[str, str, float]]:
        out = []
        if entity in self.g:
            for head, _, data in self.g.in_edges(entity, data=True):
                if relation is None or data["relation"] == relation:
                    out.append((data["relation"], head, data["confidence"]))
        return out

    def entities(self, etype: str | None = None) -> list[str]:
        return [n for n, d in self.g.nodes(data=True) if etype is None or d.get("etype") == etype]

    def entity_type(self, entity: str) -> str:
        return self.g.nodes[entity].get("etype", "") if entity in self.g else ""

    def stats(self) -> dict:
        rel_counts: dict[str, int] = {}
        for _, _, d in self.g.edges(data=True):
            rel_counts[d["relation"]] = rel_counts.get(d["relation"], 0) + 1
        type_counts: dict[str, int] = {}
        for _, d in self.g.nodes(data=True):
            type_counts[d.get("etype", "?")] = type_counts.get(d.get("etype", "?"), 0) + 1
        return {
            "num_entities": self.g.number_of_nodes(),
            "num_relations": self.g.number_of_edges(),
            "entity_types": type_counts,
            "relation_types": rel_counts,
        }

    # ---- 持久化 ---------------------------------------------------------

    def save(self, path: str | Path) -> None:
        data = {
            "nodes": [{"id": n, "etype": d.get("etype")} for n, d in self.g.nodes(data=True)],
            "edges": [
                {"head": u, "tail": v, "relation": d["relation"], "confidence": d["confidence"],
                 "source": d.get("source", ""), "evidence": d.get("evidence", "")}
                for u, v, d in self.g.edges(data=True)
            ],
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MedicalKG":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        kg = cls()
        for n in data["nodes"]:
            kg.g.add_node(n["id"], etype=n["etype"])
        for e in data["edges"]:
            kg.g.add_edge(e["head"], e["tail"], key=e["relation"], relation=e["relation"],
                          confidence=e["confidence"], source=e.get("source", ""),
                          evidence=e.get("evidence", ""))
        return kg

    def to_sqlite(self, db_path: str | Path) -> None:
        """导出为关系表，供任务三 NL2SQL 使用。"""
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.executescript(
            """
            DROP TABLE IF EXISTS entities;
            DROP TABLE IF EXISTS relations;
            CREATE TABLE entities (name TEXT PRIMARY KEY, etype TEXT, etype_zh TEXT);
            CREATE TABLE relations (
                head TEXT, relation TEXT, relation_zh TEXT, tail TEXT,
                head_type TEXT, tail_type TEXT, confidence REAL, evidence TEXT
            );
            """
        )
        for n, d in self.g.nodes(data=True):
            et = d.get("etype", "")
            cur.execute("INSERT OR REPLACE INTO entities VALUES (?,?,?)",
                        (n, et, ENTITY_TYPES.get(et, et)))
        for u, v, d in self.g.edges(data=True):
            rel = d["relation"]
            rel_zh = RELATION_TYPES.get(rel, ("", "", rel))[2]
            hd = self.g.nodes[u].get("etype", "")
            td = self.g.nodes[v].get("etype", "")
            cur.execute("INSERT INTO relations VALUES (?,?,?,?,?,?,?,?)",
                        (u, rel, rel_zh, v, hd, td, d["confidence"], d.get("evidence", "")))
        conn.commit()
        conn.close()
