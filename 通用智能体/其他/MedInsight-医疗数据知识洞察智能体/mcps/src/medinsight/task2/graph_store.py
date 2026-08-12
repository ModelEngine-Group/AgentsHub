from __future__ import annotations

import argparse
import html
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from medinsight.common.knowledge_graph import stable_id
from medinsight.common.utils import ensure_dir, write_json


DEFAULT_PENDING_GRAPH = Path("data/shared/knowledge_graph/task2_pending_medical_kg.json")
DEFAULT_LIBRARY_GRAPH = Path("data/shared/knowledge_graph/task2_user_medical_kg_library.json")
IMPORT_SOURCE = "user_confirmed_graph"


def load_graph_or_empty(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "metadata": {"name": "task2_user_medical_kg_library", "source": "empty_user_graph_library"},
        "schema": {"entity_types": [], "relation_schemas": [], "predicate_count": 0},
        "nodes": [],
        "edges": [],
    }


def custom_import_id(graph: dict[str, Any]) -> str:
    metadata = graph.get("metadata") or {}
    value = str(metadata.get("custom_import_id") or "").strip()
    if value:
        return value
    doc_ids = sorted(graph_doc_ids(graph))
    return "custom:" + ",".join(doc_ids) if doc_ids else "custom:unknown"


def graph_doc_ids(graph: dict[str, Any]) -> set[str]:
    doc_ids: set[str] = set()
    for node in graph.get("nodes") or []:
        for doc_id in node.get("source_docs") or []:
            if doc_id:
                doc_ids.add(str(doc_id))
    for edge in graph.get("edges") or []:
        for evidence in edge.get("evidence") or []:
            doc_id = evidence.get("doc_id") if isinstance(evidence, dict) else None
            if doc_id:
                doc_ids.add(str(doc_id))
    return doc_ids


def unique_values(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        if value in (None, ""):
            continue
        key = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        if key not in seen:
            result.append(value)
            seen.add(key)
    return result


def as_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    return [value]


def canonical_node_id(node: dict[str, Any]) -> str:
    node_id = str(node.get("node_id") or "").strip()
    if node_id:
        return node_id
    return stable_id("node", str(node.get("type") or "未知"), str(node.get("name") or ""))


def canonical_edge_id(edge: dict[str, Any]) -> str:
    source = str(edge.get("source") or "")
    predicate = str(edge.get("predicate") or "")
    target = str(edge.get("target") or "")
    return stable_id("edge", source, predicate, target)


def edge_key(edge: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(edge.get("source") or ""),
        str(edge.get("predicate") or ""),
        str(edge.get("target") or ""),
    )


def tagged_evidence(edge: dict[str, Any], import_id: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in edge.get("evidence") or []:
        evidence = deepcopy(item) if isinstance(item, dict) else {"text": str(item)}
        evidence.setdefault("custom_import_id", import_id)
        evidence.setdefault("custom_import_source", IMPORT_SOURCE)
        result.append(evidence)
    return result


def merge_properties(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(existing)
    for key, value in incoming.items():
        if value in (None, "", []):
            continue
        if key not in merged or merged[key] in (None, "", []):
            merged[key] = deepcopy(value)
        elif merged[key] != value:
            merged[f"{key}_values"] = unique_values(as_list(merged.get(f"{key}_values")) + [merged[key], value])
    return merged


def merge_node(existing: dict[str, Any], incoming: dict[str, Any], import_id: str) -> None:
    existing["node_id"] = canonical_node_id(existing)
    existing.setdefault("name", incoming.get("name"))
    existing.setdefault("type", incoming.get("type"))
    existing["aliases"] = unique_values(as_list(existing.get("aliases")) + as_list(incoming.get("aliases")))
    existing["source_docs"] = unique_values(as_list(existing.get("source_docs")) + as_list(incoming.get("source_docs")))
    properties = merge_properties(existing.get("properties") or {}, incoming.get("properties") or {})
    properties["custom_import_source"] = IMPORT_SOURCE
    properties["custom_import_ids"] = unique_values(
        as_list(properties.get("custom_import_ids")) + as_list(properties.get("custom_import_id")) + [import_id]
    )
    properties.pop("custom_import_id", None)
    existing["properties"] = properties


def merge_existing_node(existing: dict[str, Any], incoming: dict[str, Any]) -> None:
    existing["node_id"] = canonical_node_id(existing)
    existing.setdefault("name", incoming.get("name"))
    existing.setdefault("type", incoming.get("type"))
    existing["aliases"] = unique_values(as_list(existing.get("aliases")) + as_list(incoming.get("aliases")))
    existing["source_docs"] = unique_values(as_list(existing.get("source_docs")) + as_list(incoming.get("source_docs")))
    properties = merge_properties(existing.get("properties") or {}, incoming.get("properties") or {})
    import_ids = unique_values(as_list(properties.get("custom_import_ids")) + as_list(properties.get("custom_import_id")))
    if import_ids:
        properties["custom_import_ids"] = import_ids
    properties.pop("custom_import_id", None)
    existing["properties"] = properties


def merge_edge(existing: dict[str, Any], incoming: dict[str, Any], import_id: str) -> bool:
    before_evidence_count = len(existing.get("evidence") or [])
    existing["edge_id"] = canonical_edge_id(existing)
    existing.setdefault("source", incoming.get("source"))
    existing.setdefault("target", incoming.get("target"))
    existing.setdefault("predicate", incoming.get("predicate"))

    properties = merge_properties(existing.get("properties") or {}, incoming.get("properties") or {})
    original_edge_ids = as_list(properties.get("original_edge_ids")) + as_list(properties.get("original_edge_id"))
    original_edge_ids.append(incoming.get("edge_id"))
    properties["original_edge_ids"] = unique_values(original_edge_ids)
    properties["custom_import_source"] = IMPORT_SOURCE
    properties["custom_import_ids"] = unique_values(
        as_list(properties.get("custom_import_ids")) + as_list(properties.get("custom_import_id")) + [import_id]
    )
    properties.pop("custom_import_id", None)
    properties.pop("original_edge_id", None)
    existing["properties"] = properties

    evidence_items = as_list(existing.get("evidence")) + tagged_evidence(incoming, import_id)
    existing["evidence"] = unique_values(evidence_items)[:12]
    return len(existing["evidence"]) > before_evidence_count


def merge_existing_edge(existing: dict[str, Any], incoming: dict[str, Any]) -> None:
    existing["edge_id"] = canonical_edge_id(existing)
    existing.setdefault("source", incoming.get("source"))
    existing.setdefault("target", incoming.get("target"))
    existing.setdefault("predicate", incoming.get("predicate"))
    properties = merge_properties(existing.get("properties") or {}, incoming.get("properties") or {})
    import_ids = unique_values(as_list(properties.get("custom_import_ids")) + as_list(properties.get("custom_import_id")))
    if import_ids:
        properties["custom_import_ids"] = import_ids
    properties.pop("custom_import_id", None)
    original_edge_ids = as_list(properties.get("original_edge_ids")) + as_list(properties.get("original_edge_id"))
    original_edge_ids.extend([existing.get("edge_id"), incoming.get("edge_id")])
    properties["original_edge_ids"] = unique_values(original_edge_ids)
    properties.pop("original_edge_id", None)
    existing["properties"] = properties
    existing["evidence"] = unique_values(as_list(existing.get("evidence")) + as_list(incoming.get("evidence")))[:12]


def add_pending_graph_to_library(
    pending_path: Path = DEFAULT_PENDING_GRAPH,
    library_path: Path = DEFAULT_LIBRARY_GRAPH,
) -> dict[str, Any]:
    pending_graph = load_graph_or_empty(pending_path)
    library_graph = load_graph_or_empty(library_path)
    import_id = custom_import_id(pending_graph)

    purge_custom_import(library_graph, import_id)
    nodes_by_id: dict[str, dict[str, Any]] = {}
    for node in library_graph.get("nodes") or []:
        node_id = canonical_node_id(node)
        imported = deepcopy(node)
        imported["node_id"] = node_id
        if node_id in nodes_by_id:
            merge_existing_node(nodes_by_id[node_id], imported)
        else:
            nodes_by_id[node_id] = imported

    edges_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for edge in library_graph.get("edges") or []:
        key = edge_key(edge)
        if not all(key):
            continue
        imported = deepcopy(edge)
        imported["edge_id"] = canonical_edge_id(imported)
        if key in edges_by_key:
            merge_existing_edge(edges_by_key[key], imported)
        else:
            edges_by_key[key] = imported
    added_nodes = 0
    added_edges = 0
    merged_nodes = 0
    merged_edges = 0
    merged_evidence = 0

    for node in pending_graph.get("nodes") or []:
        node_id = canonical_node_id(node)
        if not node_id:
            continue
        imported = deepcopy(node)
        imported["node_id"] = node_id
        if node_id in nodes_by_id:
            merge_node(nodes_by_id[node_id], imported, import_id)
            merged_nodes += 1
        else:
            imported.setdefault("properties", {})
            merge_node(imported, imported, import_id)
            nodes_by_id[node_id] = imported
            added_nodes += 1

    for edge in pending_graph.get("edges") or []:
        key = edge_key(edge)
        if not all(key):
            continue
        imported_edge = deepcopy(edge)
        imported_edge["edge_id"] = canonical_edge_id(imported_edge)
        if key in edges_by_key:
            added_evidence = merge_edge(edges_by_key[key], imported_edge, import_id)
            merged_edges += 1
            if added_evidence:
                merged_evidence += 1
        else:
            new_edge = deepcopy(imported_edge)
            new_edge["evidence"] = []
            merge_edge(new_edge, imported_edge, import_id)
            edges_by_key[key] = new_edge
            added_edges += 1

    library_graph["nodes"] = list(nodes_by_id.values())
    library_graph["edges"] = list(edges_by_key.values())
    refresh_graph_metadata(library_graph)
    library_graph.setdefault("metadata", {})
    library_graph["metadata"].update(
        {
            "name": "task2_user_medical_kg_library",
            "source": "user_confirmed_medical_text_graphs",
            "description": "Only graphs explicitly confirmed by the user are stored here.",
        }
    )
    imports = library_graph["metadata"].setdefault("user_graphs", [])
    imports = [item for item in imports if item.get("custom_import_id") != import_id]
    imports.append(
        {
            "custom_import_id": import_id,
            "pending_graph_path": str(pending_path),
            "source": IMPORT_SOURCE,
            "added_nodes": added_nodes,
            "added_edges": added_edges,
            "merged_nodes": merged_nodes,
            "merged_edges": merged_edges,
            "merged_evidence": merged_evidence,
            "doc_ids": sorted(graph_doc_ids(pending_graph)),
        }
    )
    library_graph["metadata"]["user_graphs"] = imports
    library_graph["metadata"]["user_graph_count"] = len(imports)
    write_json(library_path, library_graph)
    return {
        "status": "ok",
        "library_graph_path": str(library_path),
        "custom_import_id": import_id,
        "added_nodes": added_nodes,
        "added_edges": added_edges,
        "merged_nodes": merged_nodes,
        "merged_edges": merged_edges,
        "merged_evidence": merged_evidence,
        "library_graph_count": len(imports),
        "library_nodes": len(library_graph["nodes"]),
        "library_edges": len(library_graph["edges"]),
    }


def purge_custom_import(project_graph: dict[str, Any], import_id: str) -> dict[str, int]:
    removed_edges = 0
    removed_nodes = 0
    kept_edges = []
    for edge in project_graph.get("edges") or []:
        properties = edge.get("properties") or {}
        import_ids = unique_values(as_list(properties.get("custom_import_ids")) + as_list(properties.get("custom_import_id")))
        if import_id not in import_ids:
            kept_edges.append(edge)
            continue

        remaining_import_ids = [item for item in import_ids if item != import_id]
        if not remaining_import_ids:
            removed_edges += 1
            continue
        cleaned_edge = deepcopy(edge)
        cleaned_properties = cleaned_edge.setdefault("properties", {})
        cleaned_properties["custom_import_ids"] = remaining_import_ids
        cleaned_properties.pop("custom_import_id", None)
        cleaned_edge["evidence"] = [
            item
            for item in cleaned_edge.get("evidence") or []
            if not (isinstance(item, dict) and item.get("custom_import_id") == import_id)
        ]
        kept_edges.append(cleaned_edge)
    project_graph["edges"] = kept_edges
    connected = {str(edge.get("source")) for edge in kept_edges} | {str(edge.get("target")) for edge in kept_edges}
    kept_nodes = []
    for node in project_graph.get("nodes") or []:
        properties = node.get("properties") or {}
        import_ids = unique_values(as_list(properties.get("custom_import_ids")) + as_list(properties.get("custom_import_id")))
        imported_by_this_run = import_id in import_ids
        remaining_import_ids = [item for item in import_ids if item != import_id]
        if imported_by_this_run and not remaining_import_ids and str(node.get("node_id")) not in connected:
            removed_nodes += 1
            continue
        if imported_by_this_run:
            cleaned_node = deepcopy(node)
            cleaned_properties = cleaned_node.setdefault("properties", {})
            cleaned_properties["custom_import_ids"] = remaining_import_ids
            cleaned_properties.pop("custom_import_id", None)
            kept_nodes.append(cleaned_node)
        else:
            kept_nodes.append(node)
    project_graph["nodes"] = kept_nodes
    metadata = project_graph.setdefault("metadata", {})
    metadata["user_graphs"] = [
        item for item in metadata.get("user_graphs", []) if item.get("custom_import_id") != import_id
    ]
    metadata["user_graph_count"] = len(metadata["user_graphs"])
    return {"removed_edges": removed_edges, "removed_nodes": removed_nodes}


def delete_pending_graph(pending_path: Path = DEFAULT_PENDING_GRAPH) -> dict[str, Any]:
    deleted_files: list[str] = []
    for path in related_custom_paths(pending_path):
        if path.exists():
            path.unlink()
            deleted_files.append(str(path))
    return {"status": "ok", "deleted_files": deleted_files}


def clear_library_graph(library_path: Path = DEFAULT_LIBRARY_GRAPH) -> dict[str, Any]:
    deleted_files: list[str] = []
    if library_path.exists():
        library_path.unlink()
        deleted_files.append(str(library_path))
    return {
        "status": "ok",
        "library_graph_path": str(library_path),
        "deleted_files": deleted_files,
        "graph_count": 0,
    }


def related_custom_paths(custom_graph_path: Path) -> list[Path]:
    folder = custom_graph_path.parent
    summary = folder / "task2_pending_medical_kg_summary.json"
    triples = folder / "task2_pending_medical_triples.jsonl"
    return [
        custom_graph_path,
        summary,
        triples,
    ]


def library_stats(library_path: Path = DEFAULT_LIBRARY_GRAPH) -> dict[str, Any]:
    graph = load_graph_or_empty(library_path)
    metadata = graph.get("metadata") or {}
    user_graphs = metadata.get("user_graphs") or []
    graph_count = int(metadata.get("user_graph_count") or len(user_graphs))
    return {
        "status": "ok",
        "library_graph_path": str(library_path),
        "graph_count": graph_count,
        "node_count": len(graph.get("nodes") or []),
        "edge_count": len(graph.get("edges") or []),
        "has_graphs": graph_count > 0 and bool(graph.get("edges")),
    }


def supported_diseases(library_path: Path = DEFAULT_LIBRARY_GRAPH, limit: int = 80) -> dict[str, Any]:
    graph = load_graph_or_empty(library_path)
    edges = graph.get("edges") or []
    edge_count_by_node: dict[str, int] = {}
    for edge in edges:
        source = str(edge.get("source") or "")
        if source:
            edge_count_by_node[source] = edge_count_by_node.get(source, 0) + 1

    disease_names: list[str] = []
    for node in graph.get("nodes") or []:
        if node.get("type") != "疾病":
            continue
        name = str(node.get("name") or "").strip()
        if name and edge_count_by_node.get(str(node.get("node_id") or ""), 0) > 0:
            disease_names.append(name)

    disease_names = sorted(set(disease_names), key=lambda item: (len(item), item))
    shown = disease_names[: max(limit, 0)]
    return {
        "status": "ok",
        "library_graph_path": str(library_path),
        "disease_count": len(disease_names),
        "shown_count": len(shown),
        "diseases": shown,
        "has_more": len(disease_names) > len(shown),
    }


def refresh_graph_metadata(graph: dict[str, Any]) -> None:
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    node_types: dict[str, int] = {}
    predicates: dict[str, int] = {}
    for node in nodes:
        node_type = str(node.get("type") or "未知")
        node_types[node_type] = node_types.get(node_type, 0) + 1
    for edge in edges:
        predicate = str(edge.get("predicate") or "")
        predicates[predicate] = predicates.get(predicate, 0) + 1
    graph.setdefault("metadata", {})
    graph["metadata"].update(
        {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "node_type_counts": node_types,
            "predicate_counts": predicates,
        }
    )
    graph.setdefault("schema", {})
    graph["schema"]["entity_types"] = sorted(node_types)
    graph["schema"]["predicate_count"] = len(predicates)


def generate_graph_html(graph_path: Path, output_path: Path, title: str = "医疗知识图谱可视化") -> dict[str, Any]:
    graph = load_graph_or_empty(graph_path)
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    node_index = {str(node.get("node_id")): index for index, node in enumerate(nodes)}
    graph_payload = {
        "nodes": [
            {
                "id": str(node.get("node_id")),
                "name": str(node.get("name") or ""),
                "type": str(node.get("type") or ""),
            }
            for node in nodes[:160]
        ],
        "edges": [
            {
                "source": str(edge.get("source")),
                "target": str(edge.get("target")),
                "predicate": str(edge.get("predicate") or ""),
            }
            for edge in edges
            if str(edge.get("source")) in node_index and str(edge.get("target")) in node_index
        ][:240],
    }
    ensure_dir(output_path.parent)
    output_path.write_text(render_graph_html(graph_payload, title), encoding="utf-8")
    return {
        "status": "ok",
        "visualization_path": str(output_path),
        "nodes": len(graph_payload["nodes"]),
        "edges": len(graph_payload["edges"]),
    }


def render_graph_html(graph: dict[str, Any], title: str) -> str:
    payload = json.dumps(graph, ensure_ascii=False)
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>{safe_title}</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172033; background: #f6f8fb; }}
    header {{ padding: 20px 28px; background: #ffffff; border-bottom: 1px solid #dbe3ef; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .meta {{ color: #5b677a; font-size: 14px; }}
    #graph {{ width: 100vw; height: calc(100vh - 86px); display: block; }}
    .node text {{ font-size: 12px; fill: #172033; paint-order: stroke; stroke: #fff; stroke-width: 4px; }}
    .edge-label {{ font-size: 11px; fill: #596579; paint-order: stroke; stroke: #fff; stroke-width: 3px; }}
  </style>
</head>
<body>
  <header>
    <h1>{safe_title}</h1>
    <div class="meta">节点 <span id="nodeCount"></span> 个，关系 <span id="edgeCount"></span> 条。该视图来自本次生成的知识图谱 JSON。</div>
  </header>
  <svg id="graph"></svg>
  <script>
    const graph = {payload};
    document.getElementById("nodeCount").textContent = graph.nodes.length;
    document.getElementById("edgeCount").textContent = graph.edges.length;
    const svg = document.getElementById("graph");
    const width = svg.clientWidth;
    const height = svg.clientHeight;
    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.max(160, Math.min(width, height) * 0.34);
    const byId = new Map();
    graph.nodes.forEach((node, index) => {{
      const angle = (Math.PI * 2 * index) / Math.max(graph.nodes.length, 1);
      node.x = cx + Math.cos(angle) * radius;
      node.y = cy + Math.sin(angle) * radius;
      byId.set(node.id, node);
    }});
    function el(name, attrs = {{}}) {{
      const item = document.createElementNS("http://www.w3.org/2000/svg", name);
      for (const [key, value] of Object.entries(attrs)) item.setAttribute(key, value);
      return item;
    }}
    const edgeLayer = el("g");
    const nodeLayer = el("g");
    svg.append(edgeLayer, nodeLayer);
    graph.edges.forEach(edge => {{
      const s = byId.get(edge.source);
      const t = byId.get(edge.target);
      if (!s || !t) return;
      edgeLayer.appendChild(el("line", {{ x1: s.x, y1: s.y, x2: t.x, y2: t.y, stroke: "#9fb0c7", "stroke-width": 1.4 }}));
      const label = el("text", {{ x: (s.x + t.x) / 2, y: (s.y + t.y) / 2, class: "edge-label", "text-anchor": "middle" }});
      label.textContent = edge.predicate;
      edgeLayer.appendChild(label);
    }});
    graph.nodes.forEach(node => {{
      const group = el("g", {{ class: "node" }});
      const color = node.type === "疾病" ? "#2f6fed" : node.type === "症状" ? "#df6b3f" : node.type === "药物" ? "#2d9d78" : "#7d8aa0";
      group.appendChild(el("circle", {{ cx: node.x, cy: node.y, r: 18, fill: color, stroke: "#ffffff", "stroke-width": 3 }}));
      const label = el("text", {{ x: node.x, y: node.y + 34, "text-anchor": "middle" }});
      label.textContent = node.name + " · " + node.type;
      group.appendChild(label);
      nodeLayer.appendChild(group);
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Task2 file-backed knowledge graph stores.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    merge = subparsers.add_parser("add-pending")
    merge.add_argument("--pending", default=str(DEFAULT_PENDING_GRAPH))
    merge.add_argument("--library", default=str(DEFAULT_LIBRARY_GRAPH))
    delete = subparsers.add_parser("delete-pending")
    delete.add_argument("--pending", default=str(DEFAULT_PENDING_GRAPH))
    clear = subparsers.add_parser("clear-library")
    clear.add_argument("--library", default=str(DEFAULT_LIBRARY_GRAPH))
    stats = subparsers.add_parser("stats")
    stats.add_argument("--library", default=str(DEFAULT_LIBRARY_GRAPH))
    stats.add_argument("--format", choices=["json", "count"], default="json")
    diseases = subparsers.add_parser("diseases")
    diseases.add_argument("--library", default=str(DEFAULT_LIBRARY_GRAPH))
    diseases.add_argument("--limit", type=int, default=80)
    diseases.add_argument("--format", choices=["json", "text"], default="json")
    visualize = subparsers.add_parser("visualize")
    visualize.add_argument("--graph", default=str(DEFAULT_PENDING_GRAPH))
    visualize.add_argument("--output", default="artifacts/task2/text_kg_demo/graph_visualization.html")
    visualize.add_argument("--title", default="医疗知识图谱可视化")
    args = parser.parse_args()
    if args.command == "add-pending":
        print(json.dumps(add_pending_graph_to_library(Path(args.pending), Path(args.library)), ensure_ascii=False, indent=2))
    elif args.command == "delete-pending":
        print(json.dumps(delete_pending_graph(Path(args.pending)), ensure_ascii=False, indent=2))
    elif args.command == "clear-library":
        print(json.dumps(clear_library_graph(Path(args.library)), ensure_ascii=False, indent=2))
    elif args.command == "stats":
        result = library_stats(Path(args.library))
        if args.format == "count":
            print(result["graph_count"])
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "diseases":
        result = supported_diseases(Path(args.library), args.limit)
        if args.format == "text":
            diseases_text = "、".join(result["diseases"])
            if result["has_more"]:
                diseases_text += f"、... 共 {result['disease_count']} 个"
            print(diseases_text)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "visualize":
        print(json.dumps(generate_graph_html(Path(args.graph), Path(args.output), args.title), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
