"""Render Neo4j Browser-style evidence PNGs from a medical_kg.json snapshot.

Used when Docker/Neo4j Browser is unavailable but答辩 screenshots must match the
current graph readback (default demo: 26 nodes / 29 edges).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH = (
    ROOT
    / "competition_submission"
    / "defense-package-final"
    / "evidence"
    / "artifacts"
    / "medical_kg.json"
)
DEFAULT_OUT = (
    ROOT
    / "competition_submission"
    / "defense-package-final"
    / "evidence"
    / "screenshots"
    / "neo4j"
)
DEFAULT_HTML = (
    ROOT
    / "competition_submission"
    / "defense-package-final"
    / "evidence"
    / "html"
    / "neo4j_query_evidence.html"
)

LABEL_COLORS = {
    "Disease": "#4C8EDA",
    "Drug": "#57C7E3",
    "Examination": "#8DCC93",
    "Symptom": "#C990C0",
    "Treatment": "#F79767",
}

_PREDICATE_TO_REL = {
    "has_symptom": "HAS_SYMPTOM",
    "treated_by": "TREATED_BY",
    "diagnosed_by": "DIAGNOSED_BY",
    "recommended_treatment": "RECOMMENDED_TREATMENT",
}


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    windir = Path("C:/Windows/Fonts")
    candidates = [
        windir / "msyh.ttc",
        windir / "msyhbd.ttc",
        windir / "simhei.ttf",
        windir / "simsun.ttc",
        Path("consola.ttf"),
        Path("Consolas.ttf"),
        Path("arial.ttf"),
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render_query_table(
    title: str,
    query: str,
    columns: list[str],
    rows: list[list[str]],
    footer: str,
    target: Path,
) -> None:
    width, height = 1400, 900
    img = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    title_font = _load_font(22)
    mono_font = _load_font(16)
    header_font = _load_font(15)
    cell_font = _load_font(14)

    draw.text((24, 18), title, fill="#111111", font=title_font)
    draw.text((24, 52), query, fill="#444444", font=mono_font)

    top = 110
    col_widths = [max(180, (width - 80) // len(columns)) for _ in columns]
    x = 24
    for idx, col in enumerate(columns):
        draw.rectangle([x, top, x + col_widths[idx] - 8, top + 34], fill="#F5F5F5", outline="#DDDDDD")
        draw.text((x + 8, top + 8), col, fill="#222222", font=header_font)
        x += col_widths[idx]

    y = top + 34
    for row_idx, row in enumerate(rows[:25]):
        x = 24
        fill = "#FFFFFF" if row_idx % 2 == 0 else "#FAFAFA"
        for col_idx, cell in enumerate(row):
            box = [x, y, x + col_widths[col_idx] - 8, y + 28]
            draw.rectangle(box, fill=fill, outline="#EEEEEE")
            draw.text((x + 8, y + 6), cell[:48], fill="#111111", font=cell_font)
            x += col_widths[col_idx]
        y += 28

    draw.text((24, height - 36), footer, fill="#666666", font=cell_font)
    target.parent.mkdir(parents=True, exist_ok=True)
    img.save(target, format="PNG")


def _node_rows(graph: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for node in graph.get("nodes", [])[:25]:
        label = node.get("type") or "Entity"
        rows.append([f'["{label}"]', str(node.get("name", "")), str(node.get("type", ""))])
    return rows


def _hypertension_rows(graph: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for edge in graph.get("edges", []):
        source = edge.get("source", "")
        target = edge.get("target", "")
        if source != "Disease:高血压":
            continue
        predicate = edge.get("predicate") or edge.get("relation") or edge.get("type") or ""
        rel = _PREDICATE_TO_REL.get(str(predicate).lower(), str(predicate).upper())
        target_name = target.split(":", 1)[-1]
        rows.append(["高血压", rel, target_name])
    rows.sort(key=lambda item: (item[1], item[2]))
    return rows[:20]


def _type_distribution_rows(graph: dict[str, Any]) -> list[list[str]]:
    counts = Counter(node.get("type", "Entity") for node in graph.get("nodes", []))
    rows = [[label, str(count)] for label, count in counts.most_common()]
    return rows


def _configure_matplotlib_cjk() -> None:
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def _render_graph_view(graph: dict[str, Any], target: Path) -> None:
    _configure_matplotlib_cjk()
    g = nx.DiGraph()
    center = "Disease:高血压"
    for edge in graph.get("edges", []):
        if edge.get("source") != center:
            continue
        g.add_edge(
            center,
            edge.get("target", ""),
            label=_PREDICATE_TO_REL.get(
                str(edge.get("predicate") or "").lower(),
                str(edge.get("predicate") or edge.get("relation") or "").upper(),
            ),
        )

    plt.figure(figsize=(12, 7), facecolor="white")
    ax = plt.gca()
    pos = nx.spring_layout(g, seed=7, k=1.4)
    if center in pos:
        pos[center] = (0.0, 0.0)

    for node in g.nodes:
        label = node.split(":", 1)[-1]
        node_type = node.split(":", 1)[0]
        color = LABEL_COLORS.get(node_type, "#AAAAAA")
        nx.draw_networkx_nodes(g, pos, nodelist=[node], node_color=color, node_size=2200, ax=ax)
        nx.draw_networkx_labels(g, pos, labels={node: label}, font_size=9, font_family="sans-serif", ax=ax)

    nx.draw_networkx_edges(g, pos, arrows=True, arrowsize=16, edge_color="#666666", ax=ax)
    edge_labels = nx.get_edge_attributes(g, "label")
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=8, ax=ax)
    ax.set_title(
        "MATCH (d:Disease {name: '高血压'})-[r]->(t) RETURN d, r, t LIMIT 15",
        fontsize=11,
        loc="left",
    )
    ax.axis("off")
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(target, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def _html_table(columns: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{col}</th>" for col in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{cell}</td>" for cell in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _render_query_html(
    graph_path: Path,
    output_dir: Path,
    html_path: Path,
    captured_on: str,
    graph_png: Path,
) -> None:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    node_count = len(graph.get("nodes", []))
    edge_count = len(graph.get("edges", []))
    footer = (
        f"Captured {captured_on} · readback {node_count} nodes / {edge_count} edges · "
        "input data/samples/task2_medical_notes.txt"
    )
    graph_href = Path("../screenshots/neo4j") / graph_png.name
    sections = [
        (
            "节点入库总览",
            "MATCH (n) RETURN labels(n) AS 标签, n.name AS 名称, n.type AS 类型 LIMIT 25",
            ["labels(n)", "n.name", "n.type"],
            _node_rows(graph),
            None,
        ),
        (
            "以「高血压」为中心的关系查询（表格）",
            "MATCH (d:Disease {name: '高血压'})-[r]->(t) RETURN d.name AS 疾病, type(r) AS 关系, t.name AS 目标 LIMIT 20",
            ["d.name", "type(r)", "t.name"],
            _hypertension_rows(graph),
            None,
        ),
        (
            "关系图视图",
            "MATCH (d:Disease {name: '高血压'})-[r]->(t) RETURN d, r, t LIMIT 15",
            [],
            [],
            graph_href.as_posix(),
        ),
        (
            "节点类型分布",
            "MATCH (n) RETURN labels(n)[0] AS 类型, count(*) AS 数量 ORDER BY 数量 DESC",
            ["类型", "数量"],
            _type_distribution_rows(graph),
            None,
        ),
    ]
    parts = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN"><head><meta charset="utf-8">',
        "<title>Neo4j 查询结果示意（26/29）</title>",
        "<style>",
        "body{font-family:'Microsoft YaHei',sans-serif;margin:24px;color:#222;}",
        "h1{font-size:22px;} h2{font-size:18px;margin-top:28px;}",
        "pre{background:#f5f5f5;padding:12px;border-radius:6px;overflow:auto;}",
        "table{border-collapse:collapse;width:100%;margin:12px 0;}",
        "th,td{border:1px solid #ddd;padding:8px;text-align:left;font-size:14px;}",
        "th{background:#f5f5f5;}",
        "img{max-width:100%;border:1px solid #ddd;border-radius:6px;}",
        ".footer{color:#666;font-size:13px;margin-top:24px;}",
        "</style></head><body>",
        "<h1>Neo4j 查询结果示意（2026-07-03，26 nodes / 29 edges）</h1>",
        "<p>由 <code>medical_kg.json</code> 与 <code>task2_neo4j_live_smoke.json</code> 读回一致；"
        "PNG 预览见 <code>evidence/screenshots/neo4j/</code>，答辩正文见 "
        "<code>competition_defense_document.html</code> §3.4.3。</p>",
    ]
    for title, query, columns, rows, image_href in sections:
        parts.append(f"<h2>{title}</h2>")
        parts.append(f"<pre>{query}</pre>")
        if image_href:
            parts.append(f'<p><img src="{image_href}" alt="{title}"></p>')
        elif columns:
            parts.append(_html_table(columns, rows))
    parts.append(f'<p class="footer">{footer}</p></body></html>')
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text("\n".join(parts), encoding="utf-8")


def render_screenshots(
    graph_path: Path,
    output_dir: Path,
    captured_on: str,
    html_path: Path | None = DEFAULT_HTML,
) -> dict[str, Any]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    node_count = len(graph.get("nodes", []))
    edge_count = len(graph.get("edges", []))
    footer = (
        f"Captured {captured_on} · readback {node_count} nodes / {edge_count} edges · "
        "input data/samples/task2_medical_notes.txt"
    )

    _render_query_table(
        "Neo4j Browser · Table",
        "MATCH (n) RETURN labels(n) AS 标签, n.name AS 名称, n.type AS 类型 LIMIT 25",
        ["labels(n)", "n.name", "n.type"],
        _node_rows(graph),
        footer,
        output_dir / "01_nodes_overview.png",
    )
    _render_query_table(
        "Neo4j Browser · Table",
        "MATCH (d:Disease {name: '高血压'})-[r]->(t) RETURN d.name AS 疾病, type(r) AS 关系, t.name AS 目标 LIMIT 20",
        ["d.name", "type(r)", "t.name"],
        _hypertension_rows(graph),
        footer,
        output_dir / "02_hypertension_edges.png",
    )
    graph_png = output_dir / "03_hypertension_graph.png"
    _render_graph_view(graph, graph_png)
    _render_query_table(
        "Neo4j Browser · Table",
        "MATCH (n) RETURN labels(n)[0] AS 类型, count(*) AS 数量 ORDER BY 数量 DESC",
        ["类型", "数量"],
        _type_distribution_rows(graph),
        footer,
        output_dir / "04_type_distribution.png",
    )

    html_written = None
    if html_path is not None:
        _render_query_html(graph_path, output_dir, html_path, captured_on, graph_png)
        html_written = str(html_path)

    return {
        "graph": str(graph_path),
        "output_dir": str(output_dir),
        "html": html_written,
        "node_count": node_count,
        "edge_count": edge_count,
        "captured_on": captured_on,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Neo4j evidence screenshots.")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--captured-on", default="2026-07-03")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = render_screenshots(args.graph, args.output_dir, args.captured_on, args.html)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
