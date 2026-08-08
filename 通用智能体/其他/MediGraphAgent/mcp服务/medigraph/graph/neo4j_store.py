"""Optional Neo4j knowledge-graph backend.

Activate by setting GRAPH_BACKEND=neo4j and NEO4J_* in .env, with a running
Neo4j (e.g. `docker run -p 7687:7687 -p 7474:7474 neo4j:5`). Same interface as
LocalGraphStore so agents need no changes.
"""
from __future__ import annotations

from medigraph.graph.base import GraphStore
from medigraph.schema.ontology import RELATION_TYPES


class Neo4jGraphStore(GraphStore):
    def __init__(self, uri: str, user: str, password: str):
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def upsert_entity(self, name: str, etype: str, **props) -> None:
        name = name.strip()
        if not name:
            return
        with self.driver.session() as s:
            s.run(
                "MERGE (e:Entity {name:$name}) "
                "ON CREATE SET e.type=$etype "
                "SET e.type = coalesce(e.type, $etype)",
                name=name,
                etype=etype,
            )

    def upsert_triple(
        self,
        head: str,
        head_type: str,
        relation: str,
        tail: str,
        tail_type: str,
        confidence: float = 1.0,
        source: str = "",
        operator_version: str = "",
        **props,
    ) -> None:
        self.upsert_entity(head, head_type)
        self.upsert_entity(tail, tail_type)
        with self.driver.session() as s:
            s.run(
                "MATCH (h:Entity {name:$head}), (t:Entity {name:$tail}) "
                "MERGE (h)-[r:REL {relation:$relation}]->(t) "
                "ON CREATE SET r.confidence=$confidence, r.source=$source, "
                "  r.operator_version=$opv, r.timestamp=timestamp() "
                "ON MATCH SET r.confidence = CASE WHEN $confidence > r.confidence "
                "  THEN $confidence ELSE r.confidence END",
                head=head, tail=tail, relation=relation,
                confidence=float(confidence), source=source, opv=operator_version,
            )

    def find_entities(self, names: list[str]) -> list[str]:
        out: list[str] = []
        with self.driver.session() as s:
            for q in names:
                q = q.strip()
                if not q:
                    continue
                rec = s.run(
                    "MATCH (e:Entity) WHERE toLower(e.name) = toLower($q) "
                    "OR toLower(e.name) CONTAINS toLower($q) RETURN e.name AS name LIMIT 3",
                    q=q,
                )
                out.extend(r["name"] for r in rec)
        seen, res = set(), []
        for n in out:
            if n not in seen:
                seen.add(n)
                res.append(n)
        return res

    def neighbors(self, name: str, hops: int = 1) -> list[dict]:
        hops = max(1, hops)
        with self.driver.session() as s:
            rec = s.run(
                f"MATCH p=(a:Entity {{name:$name}})-[*1..{hops}]-(b:Entity) "
                "UNWIND relationships(p) AS r "
                "WITH startNode(r) AS h, endNode(r) AS t, r "
                "RETURN DISTINCT h.name AS head, h.type AS head_type, r.relation AS relation, "
                "  t.name AS tail, t.type AS tail_type, r.confidence AS confidence, r.source AS source",
                name=name,
            )
            triples = []
            for r in rec:
                d = dict(r)
                d["relation_zh"] = RELATION_TYPES.get(d.get("relation", ""), d.get("relation", ""))
                triples.append(d)
            return triples

    def stats(self) -> dict:
        with self.driver.session() as s:
            n = s.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
            m = s.run("MATCH ()-[r:REL]->() RETURN count(r) AS c").single()["c"]
            tc = {
                r["t"]: r["c"]
                for r in s.run("MATCH (e:Entity) RETURN e.type AS t, count(*) AS c")
            }
            rc = {
                r["rel"]: r["c"]
                for r in s.run("MATCH ()-[r:REL]->() RETURN r.relation AS rel, count(*) AS c")
            }
        return {"num_entities": n, "num_triples": m, "entity_type_counts": tc, "relation_counts": rc}
