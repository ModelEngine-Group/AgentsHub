"""BI 与图文可视化：柱状图/饼图/关系子图/热力图/关系网络 + 图文结合的 HTML 分析报告。"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from ..kg import MedicalKG, RELATION_TYPES
from ..kg.schema import ENTITY_TYPES

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def bar_chart(labels: list[str], values: list[float], title: str, xlabel: str = "", ylabel: str = "数量") -> str:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(range(len(labels)), values, color="#4C78A8")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    return _fig_to_b64(fig)


def pie_chart(labels: list[str], values: list[float], title: str) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
    ax.set_title(title)
    return _fig_to_b64(fig)


def subgraph_chart(kg: MedicalKG, center: str, title: str | None = None) -> str:
    """以某实体为中心的一跳关系子图。"""
    nodes = {center}
    edges = []
    for rel, tail, _ in kg.neighbors(center):
        nodes.add(tail)
        edges.append((center, tail, rel))
    for rel, head, _ in kg.reverse_neighbors(center):
        nodes.add(head)
        edges.append((head, center, rel))
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    for u, v, rel in edges:
        g.add_edge(u, v, label=RELATION_TYPES.get(rel, ("", "", rel))[2])
    pos = nx.spring_layout(g, k=1.4, seed=42)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#E45756" if n == center else "#72B7B2" for n in g.nodes]
    nx.draw_networkx_nodes(g, pos, node_color=colors, node_size=1600, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=9, ax=ax,
                            font_family=plt.rcParams["font.sans-serif"][0])
    nx.draw_networkx_edges(g, pos, arrows=True, arrowsize=14, edge_color="#888", ax=ax)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=nx.get_edge_attributes(g, "label"),
                                 font_size=8, ax=ax,
                                 font_family=plt.rcParams["font.sans-serif"][0])
    ax.set_title(title or f"「{center}」知识子图")
    ax.axis("off")
    return _fig_to_b64(fig)


def heatmap_chart(matrix: list[list[float]], row_labels: list[str],
                  col_labels: list[str], title: str) -> str:
    """实体类型 × 关系类型 覆盖热力图，直观展示图谱各知识维度的覆盖密度。"""
    arr = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(1.1 * len(col_labels) + 3, 0.6 * len(row_labels) + 2))
    im = ax.imshow(arr, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = int(arr[i, j])
            if v:
                ax.text(j, i, str(v), ha="center", va="center",
                        color="#222" if arr[i, j] < arr.max() * 0.6 else "#fff", fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    return _fig_to_b64(fig)


def network_chart(kg: MedicalKG, relation: str, title: str | None = None,
                  max_edges: int = 25) -> str:
    """绘制某一类关系构成的多节点关系网络（如并发症网络）。"""
    g = nx.DiGraph()
    label = RELATION_TYPES.get(relation, ("", "", relation))[2]
    for head in kg.entities():
        for rel, tail, _ in kg.neighbors(head):
            if rel == relation:
                g.add_edge(head, tail)
                if g.number_of_edges() >= max_edges:
                    break
        if g.number_of_edges() >= max_edges:
            break
    fig, ax = plt.subplots(figsize=(10, 7.5))
    if g.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "无该类关系", ha="center", va="center")
        ax.axis("off")
        return _fig_to_b64(fig)
    pos = nx.spring_layout(g, k=1.2, seed=7)
    deg = dict(g.degree())
    sizes = [400 + 260 * deg[n] for n in g.nodes]
    nx.draw_networkx_nodes(g, pos, node_color="#F58518", node_size=sizes, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=8, ax=ax,
                            font_family=plt.rcParams["font.sans-serif"][0])
    nx.draw_networkx_edges(g, pos, arrows=True, arrowsize=13, edge_color="#999", ax=ax)
    ax.set_title(title or f"「{label}」关系网络")
    ax.axis("off")
    return _fig_to_b64(fig)


def coverage_matrix(kg: MedicalKG) -> tuple[list[list[int]], list[str], list[str]]:
    """统计 head 实体类型 × 关系类型 的边数矩阵。"""
    rels = list(RELATION_TYPES)
    etypes = list(ENTITY_TYPES)
    idx_r = {r: j for j, r in enumerate(rels)}
    idx_e = {e: i for i, e in enumerate(etypes)}
    mat = [[0] * len(rels) for _ in etypes]
    for head in kg.entities():
        htype = kg.entity_type(head)
        if htype not in idx_e:
            continue
        for rel, _tail, _ in kg.neighbors(head):
            if rel in idx_r:
                mat[idx_e[htype]][idx_r[rel]] += 1
    row_labels = [ENTITY_TYPES[e] for e in etypes]
    col_labels = [RELATION_TYPES[r][2] for r in rels]
    # 去掉全零行，减少视觉噪声
    keep = [i for i, row in enumerate(mat) if any(row)]
    mat = [mat[i] for i in keep]
    row_labels = [row_labels[i] for i in keep]
    return mat, row_labels, col_labels


def render_report(title: str, sections: list[dict[str, Any]], out_path: str | Path) -> Path:
    """sections: [{"heading", "text", "image_b64"(可选), "table"(可选 {columns, rows})}]"""
    parts = [
        "<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>",
        f"<title>{title}</title>",
        "<style>body{font-family:'Noto Sans CJK SC',sans-serif;max-width:960px;margin:32px auto;"
        "padding:0 16px;color:#222;line-height:1.7}h1{border-bottom:2px solid #4C78A8;padding-bottom:8px}"
        "h2{color:#2b5d8a}img{max-width:100%;border:1px solid #eee;border-radius:6px}"
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:6px 10px;font-size:14px}"
        "th{background:#f0f4f8}.note{background:#f6f8fa;border-left:4px solid #4C78A8;padding:10px 14px}</style>",
        f"</head><body><h1>{title}</h1>",
    ]
    for sec in sections:
        parts.append(f"<h2>{sec['heading']}</h2>")
        if sec.get("text"):
            parts.append(f"<div class='note'>{sec['text']}</div>")
        if sec.get("table"):
            t = sec["table"]
            parts.append("<table><tr>" + "".join(f"<th>{c}</th>" for c in t["columns"]) + "</tr>")
            for row in t["rows"]:
                parts.append("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>")
            parts.append("</table>")
        if sec.get("image_b64"):
            parts.append(f"<p><img src='data:image/png;base64,{sec['image_b64']}'/></p>")
    parts.append("</body></html>")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")
    return out
