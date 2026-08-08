"""Default knowledge-graph backend: NetworkX MultiDiGraph + JSON persistence.

Zero external services -- runs immediately. Edges carry relation, confidence,
source document and operator version for explainability / provenance.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import networkx as nx

from medigraph.graph.base import GraphStore
from medigraph.extraction.cascade import load_entity_linker
from medigraph.schema.ontology import ENTITY_TYPES, RELATION_TYPES, check_relation_constraint
from medigraph.schema.normalize import canonical_key, canonical_name, is_valid_entity_name
from medigraph.utils.io import write_json


class LocalGraphStore(GraphStore):
    def __init__(self, linker=None):
        self.g = nx.MultiDiGraph()
        # canonical_key -> display node name, so case/space variants merge.
        self._canon: dict[str, str] = {}
        self.linker = linker if linker is not None else load_entity_linker()
        self._revision_snapshot: dict | None = None

    @classmethod
    def load_json(cls, path: str | Path) -> "LocalGraphStore":
        """Rebuild a store from a graph.json produced by export_json()."""
        import json

        store = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for node in data.get("nodes", []):
            nid = node.get("id")
            if nid:
                props = {k: v for k, v in node.items() if k not in ("id", "type")}
                store.upsert_entity(nid, node.get("type", "Unknown"), **props)
        for e in data.get("edges", []):
            store.upsert_triple(
                head=e["head"], head_type=e.get("head_type", ""), relation=e["relation"],
                tail=e["tail"], tail_type=e.get("tail_type", ""),
                confidence=e.get("confidence", 1.0), source=e.get("source", ""),
                operator_version=e.get("operator_version", ""),
                timestamp=e.get("timestamp", ""),
            )
        return store

    # ------------------------------------------------------------------ #
    def _link(self, name: str, etype: str = "") -> dict:
        if self.linker is None:
            cleaned = canonical_name(name)
            return {
                "surface_form": cleaned,
                "canonical_name": cleaned,
                "canonical_id": "",
                "link_score": 0.0,
                "match_method": "normalization_only",
                "kb_source": "",
            }
        linked = self.linker.link(name, etype)
        # Fuzzy matches are useful as candidates but are too risky to silently
        # rename graph nodes. Exact/alias links are safe for merging.
        if linked["match_method"] == "fuzzy":
            linked["canonical_name"] = canonical_name(name)
        return linked

    def _resolve(self, name: str, etype: str = "") -> str:
        """Map a surface name to its canonical node name, merging variants."""
        name = self._link(name, etype)["canonical_name"]
        key = canonical_key(name)
        if key in self._canon:
            return self._canon[key]
        self._canon[key] = name
        return name

    def upsert_entity(self, name: str, etype: str, **props) -> None:
        linked = self._link(name, etype)
        surface = canonical_name(name)
        name = self._resolve(name, etype)
        if not name or not is_valid_entity_name(name):
            return
        node_props = {
            "canonical_id": linked.get("canonical_id", ""),
            "canonical_name": linked.get("canonical_name", name),
            "surface_forms": [surface],
            "link_score": linked.get("link_score", 0.0),
            "link_method": linked.get("match_method", ""),
            "kb_source": linked.get("kb_source", ""),
            **props,
        }
        if self.g.has_node(name):
            self.g.nodes[name].setdefault("type", etype)
            surfaces = self.g.nodes[name].setdefault("surface_forms", [])
            if surface and surface not in surfaces:
                surfaces.append(surface)
            if float(node_props.get("confidence", 0.0) or 0.0) > float(
                self.g.nodes[name].get("confidence", 0.0) or 0.0
            ):
                self.g.nodes[name]["confidence"] = node_props["confidence"]
            for key, value in node_props.items():
                if value not in ("", None, [], {}):
                    self.g.nodes[name].setdefault(key, value)
        else:
            self.g.add_node(name, type=etype, **node_props)

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
        timestamp: str = "",
        **props,
    ) -> None:
        if not is_valid_entity_name(head) or not is_valid_entity_name(tail):
            return
        self.upsert_entity(head, head_type, confidence=props.get("head_confidence", 0.0))
        self.upsert_entity(tail, tail_type, confidence=props.get("tail_confidence", 0.0))
        head = self._resolve(head, head_type)
        tail = self._resolve(tail, tail_type)
        event_time = timestamp or datetime.now(timezone.utc).isoformat()
        provenance = props.get("provenance")
        if isinstance(provenance, dict):
            provenance = [provenance]
        if not isinstance(provenance, list):
            provenance = []
        event = {
            "source_doc": props.get("source_doc") or source,
            "source_hash": props.get("source_hash", ""),
            "chunk_id": props.get("chunk_id", ""),
            "sentence_id": props.get("sentence_id", ""),
            "extractor": props.get("extractor") or operator_version,
            "model_version": props.get("model_version", ""),
            "evidence": props.get("evidence", ""),
            "confidence": round(float(confidence), 3),
            "timestamp": event_time,
        }
        if any(value not in ("", None) for key, value in event.items() if key not in {"confidence", "timestamp"}):
            provenance.append(event)
        # dedup edges by (relation) between same pair; keep max confidence
        existing = [
            k
            for k in self.g.get_edge_data(head, tail, default={}).keys()
            if self.g[head][tail][k].get("relation") == relation
        ] if self.g.has_edge(head, tail) else []
        if existing:
            k = existing[0]
            if confidence > self.g[head][tail][k].get("confidence", 0):
                self.g[head][tail][k]["confidence"] = round(confidence, 3)
            if source:
                sources = self.g[head][tail][k].setdefault(
                    "sources",
                    [self.g[head][tail][k].get("source", "")] if self.g[head][tail][k].get("source") else [],
                )
                if source not in sources:
                    sources.append(source)
                self.g[head][tail][k]["source"] = sources[0]
            if operator_version:
                self.g[head][tail][k]["operator_version"] = operator_version
            existing_provenance = self.g[head][tail][k].setdefault("provenance", [])
            for item in provenance:
                marker = json.dumps(item, ensure_ascii=False, sort_keys=True)
                if all(json.dumps(old, ensure_ascii=False, sort_keys=True) != marker for old in existing_provenance):
                    existing_provenance.append(item)
            return
        self.g.add_edge(
            head,
            tail,
            relation=relation,
            confidence=round(confidence, 3),
            source=source,
            sources=[source] if source else [],
            operator_version=operator_version,
            timestamp=event_time,
            provenance=provenance,
        )

    # ------------------------------------------------------------------ #
    def find_entities(self, names: list[str]) -> list[str]:
        resolved: list[str] = []
        nodes = list(self.g.nodes)
        for q in names:
            ql = q.strip().lower()
            if not ql:
                continue
            # exact (case-insensitive) then substring match
            exact = [n for n in nodes if n.lower() == ql]
            if exact:
                resolved.extend(exact)
                continue
            sub = [n for n in nodes if ql in n.lower() or n.lower() in ql]
            resolved.extend(sub[:3])
        # dedup preserve order
        seen, out = set(), []
        for n in resolved:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    def neighbors(self, name: str, hops: int = 1) -> list[dict]:
        if not self.g.has_node(name):
            return []
        triples: list[dict] = []
        frontier = {name}
        visited = set()
        for _ in range(max(1, hops)):
            nxt = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                for _, tail, data in self.g.out_edges(node, data=True):
                    triples.append(self._edge_dict(node, tail, data))
                    nxt.add(tail)
                for head, _, data in self.g.in_edges(node, data=True):
                    triples.append(self._edge_dict(head, node, data))
                    nxt.add(head)
            frontier = nxt - visited
        # dedup
        seen, out = set(), []
        for t in triples:
            key = (t["head"], t["relation"], t["tail"])
            if key not in seen:
                seen.add(key)
                out.append(t)
        return out

    def traverse_paths(self, name: str, hops: int = 3, max_paths: int = 60) -> list[dict]:
        """Return cycle-free 1..N hop reasoning paths over the undirected view."""
        if not self.g.has_node(name):
            return []
        paths: list[dict] = []
        queue: list[tuple[str, list[str], list[dict]]] = [(name, [name], [])]
        while queue and len(paths) < max_paths:
            node, visited_nodes, steps = queue.pop(0)
            if len(steps) >= max(1, hops):
                continue
            adjacent = []
            for _, tail, data in self.g.out_edges(node, data=True):
                adjacent.append((tail, self._edge_dict(node, tail, data)))
            for head, _, data in self.g.in_edges(node, data=True):
                adjacent.append((head, self._edge_dict(head, node, data)))
            for next_node, step in adjacent:
                if next_node in visited_nodes:
                    continue
                next_steps = [*steps, step]
                confidences = [float(item.get("confidence", 0.0) or 0.0) for item in next_steps]
                path = {
                    "path_id": f"path-{len(paths) + 1}",
                    "anchor": name,
                    "hops": len(next_steps),
                    "nodes": [*visited_nodes, next_node],
                    "relations": [item["relation"] for item in next_steps],
                    "confidence": round(
                        math.prod(confidences) ** (1 / len(confidences)),
                        4,
                    ),
                    "steps": next_steps,
                    "sources": sorted(
                        {
                            source
                            for item in next_steps
                            for source in item.get("sources", [item.get("source", "")])
                            if source
                        }
                    ),
                }
                paths.append(path)
                queue.append((next_node, [*visited_nodes, next_node], next_steps))
                if len(paths) >= max_paths:
                    break
        return paths

    def _edge_dict(self, head: str, tail: str, data: dict) -> dict:
        return {
            "head": head,
            "head_type": self.g.nodes[head].get("type", ""),
            "relation": data.get("relation", ""),
            "relation_zh": RELATION_TYPES.get(data.get("relation", ""), data.get("relation", "")),
            "tail": tail,
            "tail_type": self.g.nodes[tail].get("type", ""),
            "confidence": data.get("confidence", 1.0),
            "source": data.get("source", ""),
            "sources": data.get("sources", [data.get("source", "")] if data.get("source") else []),
            "operator_version": data.get("operator_version", ""),
            "timestamp": data.get("timestamp", ""),
            "provenance": data.get("provenance", []),
        }

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        type_counts: dict[str, int] = {}
        for _, d in self.g.nodes(data=True):
            t = d.get("type", "Unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        rel_counts: dict[str, int] = {}
        for _, _, d in self.g.edges(data=True):
            r = d.get("relation", "Unknown")
            rel_counts[r] = rel_counts.get(r, 0) + 1
        top = sorted(self.g.degree, key=lambda x: x[1], reverse=True)[:10]
        return {
            "num_entities": self.g.number_of_nodes(),
            "num_triples": self.g.number_of_edges(),
            "entity_type_counts": type_counts,
            "relation_counts": rel_counts,
            "top_entities": [{"name": n, "degree": d} for n, d in top],
        }

    # ------------------------------------------------------------------ #
    def export_json(self, path: str | Path) -> str:
        nodes = [{"id": n, **d} for n, d in self.g.nodes(data=True)]
        edges = [self._edge_dict(h, t, d) for h, t, d in self.g.edges(data=True)]
        return write_json(
            {
                "metadata": {
                    "format": "medigraph_graph",
                    "schema_version": "2.0.0",
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "stats": self.stats(),
                },
                "nodes": nodes,
                "edges": edges,
            },
            path,
        )

    # ------------------------------------------------------------------ #
    def begin_revision(self, label: str = "") -> None:
        """Start an incremental graph revision for delta reporting."""
        self._revision_snapshot = {
            "label": label,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "nodes": set(self.g.nodes),
            "edges": {
                (head, data.get("relation", ""), tail)
                for head, tail, data in self.g.edges(data=True)
            },
        }

    def commit_revision(self) -> dict:
        before = self._revision_snapshot or {
            "label": "",
            "started_at": "",
            "nodes": set(),
            "edges": set(),
        }
        current_edges = {
            (head, data.get("relation", ""), tail)
            for head, tail, data in self.g.edges(data=True)
        }
        added_nodes = sorted(set(self.g.nodes) - before["nodes"])
        added_edge_keys = current_edges - before["edges"]
        added_edges = [
            self._edge_dict(head, tail, data)
            for head, tail, data in self.g.edges(data=True)
            if (head, data.get("relation", ""), tail) in added_edge_keys
        ]
        delta = {
            "label": before["label"],
            "started_at": before["started_at"],
            "committed_at": datetime.now(timezone.utc).isoformat(),
            "added_entity_count": len(added_nodes),
            "added_triple_count": len(added_edges),
            "added_entities": [
                {"id": name, **self.g.nodes[name]}
                for name in added_nodes
            ],
            "added_triples": added_edges,
            "graph_stats": self.stats(),
        }
        self._revision_snapshot = None
        return delta

    def audit(self) -> dict:
        """Measure schema legality, canonical coverage and edge provenance."""
        illegal = []
        provenance_count = 0
        for head, tail, data in self.g.edges(data=True):
            relation = data.get("relation", "")
            head_type = self.g.nodes[head].get("type", "")
            tail_type = self.g.nodes[tail].get("type", "")
            if not check_relation_constraint(relation, head_type, tail_type):
                illegal.append(
                    {
                        "head": head,
                        "head_type": head_type,
                        "relation": relation,
                        "tail": tail,
                        "tail_type": tail_type,
                    }
                )
            if data.get("source") or data.get("provenance"):
                provenance_count += 1
        total_edges = self.g.number_of_edges()
        linked_nodes = sum(1 for _, data in self.g.nodes(data=True) if data.get("canonical_id"))
        total_nodes = self.g.number_of_nodes()
        return {
            "entities": total_nodes,
            "triples": total_edges,
            "canonical_id_coverage": round(linked_nodes / total_nodes, 4) if total_nodes else 1.0,
            "provenance_coverage": round(provenance_count / total_edges, 4) if total_edges else 1.0,
            "illegal_triples": len(illegal),
            "illegal_examples": illegal[:20],
        }

    def export_html(self, path: str | Path, height: str = "800px") -> str:
        """Interactive visualization via pyvis."""
        from pyvis.network import Network

        # color per entity type
        palette = [
            "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
            "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990", "#9A6324",
        ]
        types = list(ENTITY_TYPES.keys())
        color_of = {t: palette[i % len(palette)] for i, t in enumerate(types)}

        net = Network(
            height=height,
            width="100%",
            directed=True,
            notebook=False,
            cdn_resources="in_line",
            bgcolor="#ffffff",
            font_color="#1b2a2e",
        )
        # pyvis defaults render node labels at ~14px and let barnes_hut fling nodes
        # far apart, so the auto-fit zoom shrinks everything into unreadable dots.
        # Scale node/edge typography up and pull the layout tighter instead. Node
        # size adapts to graph size: small demo graphs get big, readable nodes;
        # large graphs stay legible without overlapping into a blob.
        n_nodes = self.g.number_of_nodes()
        node_size = 34 if n_nodes <= 30 else (26 if n_nodes <= 120 else 18)
        node_font = 26 if n_nodes <= 30 else (20 if n_nodes <= 120 else 15)
        edge_font = 18 if n_nodes <= 30 else (15 if n_nodes <= 120 else 12)
        spring_len = 220 if n_nodes <= 30 else (160 if n_nodes <= 120 else 110)
        net.set_options(
            json.dumps(
                {
                    "nodes": {
                        "size": node_size,
                        "borderWidth": 2,
                        "borderWidthSelected": 4,
                        "font": {
                            "size": node_font,
                            "face": "sans-serif",
                            "color": "#1b2a2e",
                            "strokeWidth": 4,
                            "strokeColor": "#ffffff",
                        },
                    },
                    "edges": {
                        "color": {"color": "#9aa7ad", "highlight": "#e6194B"},
                        "width": 2,
                        "font": {
                            "size": edge_font,
                            "face": "sans-serif",
                            "color": "#5b6a6c",
                            "strokeWidth": 5,
                            "strokeColor": "#ffffff",
                            "align": "middle",
                        },
                        "arrows": {"to": {"enabled": True, "scaleFactor": 0.9}},
                        "smooth": {"type": "continuous"},
                    },
                    "physics": {
                        "barnesHut": {
                            "gravitationalConstant": -12000,
                            "centralGravity": 0.55,
                            "springLength": spring_len,
                            "springConstant": 0.05,
                            "damping": 0.4,
                            "avoidOverlap": 0.35,
                        },
                        "stabilization": {"enabled": True, "iterations": 220},
                    },
                    "interaction": {"hover": True, "navigationButtons": True, "tooltipDelay": 120},
                },
                ensure_ascii=False,
            )
        )
        for n, d in self.g.nodes(data=True):
            t = d.get("type", "Unknown")
            label_zh = ENTITY_TYPES.get(t, t)
            net.add_node(n, label=n, title=f"{n} ({label_zh})", color=color_of.get(t, "#cccccc"))
        for h, t, d in self.g.edges(data=True):
            rel = d.get("relation", "")
            net.add_edge(h, t, label=RELATION_TYPES.get(rel, rel), title=f"{rel} (conf={d.get('confidence')})")

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Write UTF-8 ourselves; pyvis.save_graph uses the locale codec (GBK on
        # Windows) and crashes on chars like the copyright sign.
        html = net.generate_html(notebook=False)
        p.write_text(html, encoding="utf-8")
        return str(p)
