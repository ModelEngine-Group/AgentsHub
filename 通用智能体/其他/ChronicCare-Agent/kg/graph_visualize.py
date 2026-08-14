from __future__ import annotations

import html
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import networkx as nx

from runtime_common.common import ensure_directory

FOCUS_NODE_IDS = [
    "Disease::hypertension",
    "Disease::diabetes",
    "Disease::hyperlipidemia",
    "Drug::metformin",
    "Indicator::fasting_glucose",
    "Patient::P0001",
]

TYPE_COLORS = {
    "Patient": "#2f69bf",
    "Disease": "#d1495b",
    "Indicator": "#2e933c",
    "RiskEvent": "#e67e22",
    "Drug": "#7b5cb8",
    "DrugCategory": "#9b7ed4",
    "Visit": "#00838f",
    "LabResult": "#4e79a7",
    "FollowupPlan": "#00a6a6",
    "LifestyleRecord": "#607d8b",
    "DoctorAdvice": "#ad3b75",
    "RiskScore": "#8d6e63",
    "RiskFactor": "#5d4037",
}

TYPE_DISPLAY_NAMES = {
    "Patient": "患者",
    "Disease": "疾病",
    "Indicator": "指标",
    "RiskEvent": "风险事件",
    "Drug": "药物",
    "DrugCategory": "药物类别",
    "Visit": "就诊",
    "LabResult": "检验结果",
    "FollowupPlan": "随访计划",
    "LifestyleRecord": "生活方式记录",
    "DoctorAdvice": "医生建议",
    "RiskScore": "风险评分",
    "RiskFactor": "风险因素",
}

LEGEND_TYPES = [
    "Patient",
    "Disease",
    "Indicator",
    "RiskEvent",
    "Drug",
    "FollowupPlan",
    "LifestyleRecord",
    "DoctorAdvice",
    "RiskScore",
    "Visit",
    "LabResult",
]


def choose_subgraph_nodes(graph: nx.MultiDiGraph, max_nodes: int = 120) -> Set[str]:
    selected: Set[str] = set()
    for node_id in FOCUS_NODE_IDS:
        if node_id not in graph:
            continue
        selected.add(node_id)
        selected.update(graph.successors(node_id))
        selected.update(graph.predecessors(node_id))
        if len(selected) >= max_nodes:
            break
    if not selected:
        ranked = sorted(graph.degree, key=lambda item: item[1], reverse=True)[:max_nodes]
        selected.update(node_id for node_id, _ in ranked)
    if len(selected) > max_nodes:
        ranked_selected = sorted(selected, key=lambda item: graph.degree(item), reverse=True)[:max_nodes]
        selected = set(ranked_selected)
    return selected


def _label_for(node_id: str, node_data: Dict[str, Any]) -> str:
    display_name = str(node_data.get("display_name") or node_data.get("name") or "").strip()
    if display_name:
        return display_name[:22]
    if "::" in node_id:
        node_type, raw_value = node_id.split("::", 1)
        if node_type == "Patient":
            return f"患者 {raw_value}"
        return raw_value[:22]
    return node_id[:22]


def _iter_table_rows(items: Iterable[Tuple[str, int]]) -> str:
    return "".join(
        f"<tr><td>{html.escape(name)}</td><td>{count}</td></tr>"
        for name, count in items
    )


def _render_graph_overview_html(
    path: Path,
    *,
    title: str,
    intro: str,
    summary: Dict[str, Any],
    entity_type_count: Dict[str, int],
    relation_type_count: Dict[str, int],
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    note: str,
) -> Tuple[int, int]:
    ensure_directory(path.parent)
    width = 2560
    height = 1540
    positions: Dict[str, Tuple[float, float]] = {}
    nodes_by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    node_lookup = {str(node["id"]): node for node in nodes}

    for node in nodes:
        node_type = str(node.get("type") or "Unknown")
        nodes_by_type[node_type].append(node)

    patient_nodes = nodes_by_type.get("Patient", [])
    disease_nodes = nodes_by_type.get("Disease", [])
    indicator_nodes = nodes_by_type.get("Indicator", [])
    risk_nodes = nodes_by_type.get("RiskEvent", [])
    drug_nodes = nodes_by_type.get("Drug", [])
    followup_nodes = nodes_by_type.get("FollowupPlan", [])
    visit_nodes = nodes_by_type.get("Visit", [])
    lab_nodes = nodes_by_type.get("LabResult", [])
    lifestyle_nodes = nodes_by_type.get("LifestyleRecord", [])
    advice_nodes = nodes_by_type.get("DoctorAdvice", [])
    score_nodes = nodes_by_type.get("RiskScore", [])
    other_nodes = [
        node
        for node_type, bucket in nodes_by_type.items()
        if node_type
        not in {
            "Patient",
            "Disease",
            "Indicator",
            "RiskEvent",
            "Drug",
            "FollowupPlan",
            "Visit",
            "LabResult",
            "LifestyleRecord",
            "DoctorAdvice",
            "RiskScore",
        }
        for node in bucket
    ]

    def place_vertical(bucket: List[Dict[str, Any]], x: float, y_start: float, y_end: float) -> None:
        if not bucket:
            return
        step = (y_end - y_start) / max(1, len(bucket) - 1) if len(bucket) > 1 else 0
        for index, node in enumerate(bucket):
            y = y_start + index * step if len(bucket) > 1 else (y_start + y_end) / 2
            positions[str(node["id"])] = (x, y)

    def place_grid(
        bucket: List[Dict[str, Any]],
        x_start: float,
        x_end: float,
        y_start: float,
        y_end: float,
        columns: int,
    ) -> None:
        if not bucket:
            return
        columns = max(1, columns)
        rows = math.ceil(len(bucket) / columns)
        x_step = (x_end - x_start) / max(1, columns - 1) if columns > 1 else 0
        y_step = (y_end - y_start) / max(1, rows - 1) if rows > 1 else 0
        for index, node in enumerate(bucket):
            row = index // columns
            col = index % columns
            x = x_start + col * x_step if columns > 1 else (x_start + x_end) / 2
            y = y_start + row * y_step if rows > 1 else (y_start + y_end) / 2
            positions[str(node["id"])] = (x, y)

    place_grid(patient_nodes, 100, 940, 180, 1360, columns=6)
    place_vertical(disease_nodes, 1180, 360, 1180)
    place_vertical(score_nodes, 1350, 480, 1040)
    place_grid(indicator_nodes, 1560, 1930, 170, 680, columns=2)
    place_grid(lab_nodes, 1540, 1950, 700, 1030, columns=2)
    place_grid(risk_nodes, 1560, 1930, 1060, 1380, columns=2)
    place_vertical(visit_nodes, 2080, 220, 700)
    place_vertical(followup_nodes, 2080, 760, 980)
    place_vertical(lifestyle_nodes, 2080, 1040, 1180)
    place_vertical(advice_nodes, 2080, 1220, 1360)
    place_vertical(drug_nodes, 2290, 240, 1240)
    place_vertical(other_nodes, 1450, 1140, 1400)

    edge_svg: List[str] = []
    display_edges = 0
    for edge in edges:
        source = str(edge["source"])
        target = str(edge["target"])
        if source not in positions or target not in positions:
            continue
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        edge_svg.append(
            f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='#c7d0d9' stroke-width='1.05' />"
        )
        display_edges += 1

    node_svg: List[str] = []
    for node in nodes:
        node_id = str(node["id"])
        if node_id not in positions:
            continue
        x, y = positions[node_id]
        node_type = str(node.get("type") or "Unknown")
        fill = TYPE_COLORS.get(node_type, "#455a64")
        label = html.escape(_label_for(node_id, node_lookup[node_id]))
        title_text = html.escape(node_id)
        label_y = y + 35
        font_size = 11 if node_type == "Patient" else 10
        radius = 17 if node_type in {"Disease", "RiskScore"} else 15
        node_svg.append(
            f"<g><title>{title_text}</title>"
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{radius}' fill='{fill}' />"
            f"<text x='{x:.1f}' y='{label_y:.1f}' text-anchor='middle' font-size='{font_size}' fill='#102a43'>{label}</text></g>"
        )

    legend_html = "".join(
        f"<div style='display:flex;align-items:center;gap:8px;'>"
        f"<span style='display:inline-block;width:12px;height:12px;background:{TYPE_COLORS.get(item, '#455a64')};border-radius:50%;'></span>"
        f"{TYPE_DISPLAY_NAMES.get(item, item)}</div>"
        for item in LEGEND_TYPES
        if nodes_by_type.get(item)
    )

    overview_cards = "".join(
        [
            f"<div class='card'><div class='k'>总节点数</div><div class='v'>{int(summary.get('node_count', 0) or 0):,}</div></div>",
            f"<div class='card'><div class='k'>总边数</div><div class='v'>{int(summary.get('edge_count', 0) or 0):,}</div></div>",
            f"<div class='card'><div class='k'>当前展示节点</div><div class='v'>{len(positions):,}</div></div>",
            f"<div class='card'><div class='k'>当前展示边</div><div class='v'>{display_edges:,}</div></div>",
        ]
    )

    entity_rows = _iter_table_rows(sorted(entity_type_count.items(), key=lambda item: (-item[1], item[0]))[:20])
    relation_rows = _iter_table_rows(sorted(relation_type_count.items(), key=lambda item: (-item[1], item[0]))[:20])

    node_rows = "".join(
        f"<tr><td>{html.escape(str(node['id']))}</td><td>{html.escape(TYPE_DISPLAY_NAMES.get(str(node.get('type') or ''), str(node.get('type') or '')))}</td><td>{html.escape(_label_for(str(node['id']), node))}</td></tr>"
        for node in nodes[:160]
    )
    edge_rows = "".join(
        f"<tr><td>{html.escape(str(edge['source']))}</td><td>{html.escape(str(edge['relation']))}</td><td>{html.escape(str(edge['target']))}</td></tr>"
        for edge in edges[:220]
    )

    html_content = f"""<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f4f7fb; color: #102a43; }}
    .panel {{ background: white; border-radius: 18px; padding: 22px; margin-bottom: 20px; box-shadow: 0 14px 36px rgba(16,42,67,0.08); }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
    .card {{ border: 1px solid #e6edf3; border-radius: 16px; padding: 18px; background: linear-gradient(180deg, #fbfdff 0%, #f3f8ff 100%); }}
    .card .k {{ font-size: 13px; color: #486581; margin-bottom: 8px; }}
    .card .v {{ font-size: 30px; font-weight: 700; color: #102a43; }}
    .legend {{ display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }}
    .graph-wrap {{ overflow-x: auto; overflow-y: hidden; padding-bottom: 10px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid #e6edf3; text-align: left; padding: 8px 10px; font-size: 13px; vertical-align: top; }}
    .split {{ display:grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .note {{ color: #486581; line-height: 1.8; }}
  </style>
</head>
<body>
  <div class="panel">
    <h1>{html.escape(title)}</h1>
    <p class="note">{html.escape(intro)}</p>
    <p class="note">{html.escape(note)}</p>
    <p class="note">默认图谱页不会直接铺满全部 {int(summary.get('node_count', 0) or 0):,} 个节点，而是基于最新图谱 JSON 自动生成一张可读的局部总览图，方便在浏览器中稳定查看。</p>
  </div>
  <div class="panel">
    <h2>核心概览</h2>
    <div class="cards">{overview_cards}</div>
  </div>
  <div class="panel">
    <h2>图例</h2>
    <div class="legend">{legend_html}</div>
  </div>
  <div class="panel">
    <h2>图谱总览视图</h2>
    <div class="graph-wrap">
      <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="max-width:none;height:auto;background:#fcfdff;border:1px solid #e6edf3;border-radius:14px;">
        {''.join(edge_svg)}
        {''.join(node_svg)}
      </svg>
    </div>
  </div>
  <div class="panel split">
    <div>
      <h2>实体类型分布</h2>
      <table><thead><tr><th>实体类型</th><th>数量</th></tr></thead><tbody>{entity_rows}</tbody></table>
    </div>
    <div>
      <h2>关系类型分布</h2>
      <table><thead><tr><th>关系类型</th><th>数量</th></tr></thead><tbody>{relation_rows}</tbody></table>
    </div>
  </div>
  <div class="panel">
    <h2>当前展示节点说明</h2>
    <table><thead><tr><th>Node ID</th><th>Type</th><th>Label</th></tr></thead><tbody>{node_rows}</tbody></table>
  </div>
  <div class="panel">
    <h2>当前展示关系说明</h2>
    <table><thead><tr><th>Source</th><th>Relation</th><th>Target</th></tr></thead><tbody>{edge_rows}</tbody></table>
  </div>
</body>
</html>"""
    path.write_text(html_content, encoding="utf-8")
    return len(positions), display_edges


def build_fallback_html(
    path: Path,
    *,
    summary: Dict[str, Any],
    entity_type_count: Dict[str, int],
    relation_type_count: Dict[str, int],
    top_degree_nodes: List[Dict[str, Any]],
    note: str,
) -> None:
    nodes = []
    for index, item in enumerate(top_degree_nodes[:24]):
        node_id = str(item.get("id") or item.get("node_id") or f"Node::{index}")
        node_type = str(item.get("type") or "Unknown")
        nodes.append(
            {
                "id": node_id,
                "type": node_type,
                "display_name": str(item.get("label") or item.get("display_name") or node_id),
            }
        )
    edges: List[Dict[str, Any]] = []
    _render_graph_overview_html(
        path,
        title="ChronicCare Knowledge Graph",
        intro="此页面基于最新知识图谱摘要自动生成，用于兜底展示图谱总览信息。",
        summary=summary,
        entity_type_count=entity_type_count,
        relation_type_count=relation_type_count,
        nodes=nodes,
        edges=edges,
        note=note,
    )


def render_graph_html(
    graph: nx.MultiDiGraph,
    path: Path,
    *,
    total_node_count: int,
    total_edge_count: int,
) -> Tuple[bool, int, int]:
    selected = choose_subgraph_nodes(graph, max_nodes=140)
    subgraph = graph.subgraph(selected).copy()
    nodes = []
    for node_id, node_data in subgraph.nodes(data=True):
        node_type = str(node_data.get("type") or "Unknown")
        nodes.append(
            {
                "id": str(node_id),
                "type": node_type,
                "display_name": str(node_data.get("display_name") or node_data.get("name") or node_id),
            }
        )
    edges = []
    for source, target, _key, edge_data in subgraph.edges(keys=True, data=True):
        edges.append(
            {
                "source": str(source),
                "target": str(target),
                "relation": str(edge_data.get("relation") or edge_data.get("relation_type") or edge_data.get("label") or "related_to"),
            }
        )

    entity_type_count: Dict[str, int] = defaultdict(int)
    for _node_id, node_data in graph.nodes(data=True):
        entity_type_count[str(node_data.get("type") or "Unknown")] += 1

    relation_type_count: Dict[str, int] = defaultdict(int)
    for _source, _target, _key, edge_data in graph.edges(keys=True, data=True):
        relation_type_count[str(edge_data.get("relation") or edge_data.get("relation_type") or edge_data.get("label") or "related_to")] += 1

    display_node_count, display_edge_count = _render_graph_overview_html(
        path,
        title="ChronicCare Knowledge Graph",
        intro="此页面展示的是基于最新图谱数据生成的可读总览子图，不是把全部节点直接铺满。",
        summary={"node_count": total_node_count, "edge_count": total_edge_count},
        entity_type_count=dict(entity_type_count),
        relation_type_count=dict(relation_type_count),
        nodes=nodes,
        edges=edges,
        note="当前图谱页由最新 graph.json 实时渲染生成。",
    )
    return True, display_node_count, display_edge_count
