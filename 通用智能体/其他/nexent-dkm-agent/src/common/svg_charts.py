"""Low-level SVG chart builders for offline defense figures."""

from __future__ import annotations

import math
from typing import Any


def svg_header(width: int, height: int, background: str = "#fbfdff") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="{background}"/>'
    )


def esc(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def short_text(value: Any, limit: int = 24) -> str:
    text = str(value)
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def bar_chart_svg(chart: dict[str, Any]) -> str:
    rows = chart.get("data", [])
    if not rows:
        return ""
    width = int(chart.get("width", 900))
    row_height = 34
    height = 54 + row_height * len(rows)
    label_width = 220
    max_value = max(float(row.get("value", 0) or 0) for row in rows) or 1.0
    parts = [
        svg_header(width, height),
        f'<text x="20" y="28" font-size="18" font-weight="700">{esc(chart.get("title", ""))}</text>',
    ]
    for index, row in enumerate(rows):
        y = 48 + index * row_height
        value = float(row.get("value", 0) or 0)
        bar_width = max(4.0, (width - label_width - 100) * value / max_value)
        parts.append(f'<text x="20" y="{y + 18}" font-size="13">{esc(short_text(row.get("category", "")))}</text>')
        parts.append(f'<rect x="{label_width}" y="{y}" width="{bar_width:.1f}" height="20" rx="5" fill="#246bfe"/>')
        parts.append(f'<text x="{label_width + bar_width + 10:.1f}" y="{y + 16}" font-size="13">{esc(row.get("value", 0))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def grouped_bar_chart_svg(chart: dict[str, Any]) -> str:
    groups = chart.get("groups", [])
    if not groups:
        return ""
    width = int(chart.get("width", 900))
    height = int(chart.get("height", 360))
    pad_x = 80
    pad_y = 60
    group_width = (width - pad_x * 2) / max(1, len(groups))
    max_value = max(float(item.get("value", 0) or 0) for group in groups for item in group.get("series", [])) or 1.0
    parts = [
        svg_header(width, height),
        f'<text x="20" y="28" font-size="18" font-weight="700">{esc(chart.get("title", ""))}</text>',
    ]
    colors = ("#94a3b8", "#246bfe")
    for group_index, group in enumerate(groups):
        base_x = pad_x + group_index * group_width + group_width * 0.15
        parts.append(
            f'<text x="{base_x + group_width * 0.2:.1f}" y="{height - 18}" font-size="12" text-anchor="middle">{esc(group.get("label", ""))}</text>'
        )
        series = group.get("series", [])
        bar_slot = group_width * 0.7 / max(1, len(series))
        for series_index, item in enumerate(series):
            value = float(item.get("value", 0) or 0)
            bar_height = max(4.0, (height - pad_y * 2) * value / max_value)
            x = base_x + series_index * bar_slot
            y = height - pad_y - bar_height
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_slot * 0.7:.1f}" height="{bar_height:.1f}" rx="4" fill="{colors[series_index % len(colors)]}"/>'
            )
            parts.append(f'<text x="{x + bar_slot * 0.35:.1f}" y="{y - 6:.1f}" font-size="11" text-anchor="middle">{esc(value)}</text>')
            parts.append(
                f'<text x="{x + bar_slot * 0.35:.1f}" y="{height - pad_y + 14:.1f}" font-size="10" text-anchor="middle">{esc(item.get("label", ""))}</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def line_chart_svg(chart: dict[str, Any]) -> str:
    rows = chart.get("data", [])
    if not rows:
        return ""
    y_key = chart.get("encoding", {}).get("y", "value")
    values = [float(row.get(y_key, 0) or 0) for row in rows]
    entity_values = [float(row.get("entity_count", 0) or 0) for row in rows]
    has_entity_series = any(entity_values) and "entity_count" in (rows[0] or {})
    width = int(chart.get("width", 900))
    height = int(chart.get("height", 360))
    pad = 50
    max_value = max(values + (entity_values if has_entity_series else [])) or 1.0
    x_step = (width - pad * 2) / max(1, len(rows) - 1)

    def _series_points(series_values: list[float], marker_color: str) -> tuple[str, str]:
        points: list[str] = []
        markers: list[str] = []
        for index, value in enumerate(series_values):
            x = pad + x_step * index
            y = height - pad - (value / max_value) * (height - pad * 2)
            points.append(f"{x:.1f},{y:.1f}")
            markers.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{marker_color}"/>')
        return " ".join(points), "".join(markers)

    edge_points, edge_markers = _series_points(values, "#246bfe")
    parts = [
        svg_header(width, height),
        f'<text x="20" y="28" font-size="18" font-weight="700">{esc(chart.get("title", ""))}</text>',
        f'<polyline points="{edge_points}" fill="none" stroke="#246bfe" stroke-width="4"/>',
        edge_markers,
    ]
    if has_entity_series:
        entity_points, entity_markers = _series_points(entity_values, "#91cc75")
        parts.extend([
            f'<polyline points="{entity_points}" fill="none" stroke="#91cc75" stroke-width="4" stroke-dasharray="8 5"/>',
            entity_markers,
            f'<rect x="{width - 210}" y="12" width="12" height="12" fill="#246bfe"/>'
            f'<text x="{width - 192}" y="22" font-size="12">边数</text>'
            f'<rect x="{width - 120}" y="12" width="12" height="12" fill="#91cc75"/>'
            f'<text x="{width - 102}" y="22" font-size="12">实体数</text>',
        ])
    parts.append("</svg>")
    return "".join(parts)


def network_chart_svg(chart: dict[str, Any], *, max_nodes: int = 12) -> str:
    nodes = chart.get("nodes", [])[:max_nodes]
    if not nodes:
        return ""
    width = int(chart.get("width", 900))
    height = int(chart.get("height", 520))
    center_x = width / 2
    center_y = height / 2 + 20
    positions: dict[str, tuple[float, float]] = {}
    positions[str(nodes[0].get("id", "root"))] = (center_x, center_y)
    for index, node in enumerate(nodes[1:]):
        angle = 2 * math.pi * index / max(1, len(nodes) - 1)
        positions[str(node.get("id", index))] = (
            center_x + 310 * math.cos(angle),
            center_y + 170 * math.sin(angle),
        )
    parts = [
        svg_header(width, height),
        f'<text x="20" y="28" font-size="18" font-weight="700">{esc(chart.get("title", ""))}</text>',
    ]
    for edge in chart.get("edges", [])[:24]:
        source = positions.get(str(edge.get("source")))
        target = positions.get(str(edge.get("target")))
        if source and target:
            parts.append(
                f'<line x1="{source[0]:.1f}" y1="{source[1]:.1f}" x2="{target[0]:.1f}" y2="{target[1]:.1f}" stroke="#b9c6d4" stroke-width="1.5"/>'
            )
    for node in nodes:
        node_id = str(node.get("id", ""))
        x, y = positions.get(node_id, (center_x, center_y))
        fill = "#246bfe" if node.get("type") == "Disease" else "#0f8f70"
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="21" fill="{fill}" opacity="0.92"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{y + 38:.1f}" text-anchor="middle" font-size="12">{esc(short_text(node.get("label", node_id), 12))}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def chart_spec_to_svg(chart: dict[str, Any]) -> str:
    chart_type = chart.get("type")
    if chart_type == "bar":
        return bar_chart_svg(chart)
    if chart_type == "grouped_bar":
        return grouped_bar_chart_svg(chart)
    if chart_type == "line":
        return line_chart_svg(chart)
    if chart_type == "network":
        return network_chart_svg(chart)
    return ""
