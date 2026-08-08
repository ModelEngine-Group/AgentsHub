#!/usr/bin/env python3
"""Render a medical knowledge graph (graph.json) into a self-contained HTML report.

Pure standard library (no third-party deps) so it runs inside the Nexent skill
runtime. Reads MediGraph's exported graph.json ({nodes:[{id,type}], edges:[...]}).
"""
import argparse
import html
import json
import os
from collections import Counter
from pathlib import Path

DEFAULT_OUTPUT_DIR = "/mnt/nexent"
ENTITY_ZH = {
    "Disease": "疾病", "Symptom": "症状", "Drug": "药物", "Examination": "检查",
    "Procedure": "手术", "Body": "身体部位", "Department": "科室", "Tumor": "肿瘤",
    "Biomarker": "标志物", "Gene": "基因", "Morphology": "形态学",
}


def build_html(graph: dict) -> tuple[str, int, int]:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    type_counts = Counter(n.get("type", "Unknown") for n in nodes)
    rel_counts = Counter(e.get("relation_zh") or e.get("relation", "") for e in edges)
    deg = Counter()
    for e in edges:
        deg[e.get("head", "")] += 1
        deg[e.get("tail", "")] += 1
    top = deg.most_common(10)

    def rows(counter):
        return "".join(f"<tr><td>{html.escape(str(k))}</td><td>{v}</td></tr>" for k, v in counter.items())

    triple_rows = "".join(
        f"<tr><td>{html.escape(str(e.get('head','')))}</td>"
        f"<td>{html.escape(str(e.get('relation_zh') or e.get('relation','')))}</td>"
        f"<td>{html.escape(str(e.get('tail','')))}</td>"
        f"<td>{html.escape(str(e.get('source','')))}</td></tr>"
        for e in edges
    )
    top_rows = "".join(f"<tr><td>{html.escape(str(n))}</td><td>{d}</td></tr>" for n, d in top)

    doc = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>医疗知识图谱报告</title>
<style>body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#222}}
h2{{color:#2c3e50}} table{{border-collapse:collapse;margin:10px 0;width:100%}}
th,td{{border:1px solid #ddd;padding:6px 10px;text-align:left}} th{{background:#f5f7fa}}
.kpi{{display:inline-block;background:#eef6ff;border-left:4px solid #3498db;padding:10px 16px;margin:6px}}</style>
</head><body>
<h1>📊 医疗知识图谱报告</h1>
<div class="kpi"><b>实体总数</b><br>{len(nodes)}</div>
<div class="kpi"><b>三元组总数</b><br>{len(edges)}</div>
<div class="kpi"><b>实体类型数</b><br>{len(type_counts)}</div>
<div class="kpi"><b>关系类型数</b><br>{len(rel_counts)}</div>
<h2>实体类型分布</h2><table><tr><th>类型</th><th>数量</th></tr>{rows(type_counts)}</table>
<h2>关系类型分布</h2><table><tr><th>关系</th><th>数量</th></tr>{rows(rel_counts)}</table>
<h2>核心实体 (Top 10, 按度数)</h2><table><tr><th>实体</th><th>度数</th></tr>{top_rows}</table>
<h2>全部三元组</h2><table><tr><th>头实体</th><th>关系</th><th>尾实体</th><th>来源</th></tr>{triple_rows}</table>
</body></html>"""
    return doc, len(nodes), len(edges)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, help="path to graph.json")
    ap.add_argument("--output", default="medical_kg_report.html")
    ap.add_argument("--working-dir", default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()

    graph = json.loads(Path(args.graph).read_text(encoding="utf-8"))
    doc, n_ent, n_tri = build_html(graph)

    out = args.output if os.path.isabs(args.output) else os.path.join(args.working_dir, args.output)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(doc, encoding="utf-8")
    print(json.dumps({
        "status": "success", "file_path": args.output, "absolute_path": os.path.abspath(out),
        "file_name": os.path.basename(out), "mime_type": "text/html",
        "num_entities": n_ent, "num_triples": n_tri,
        "message": "Medical KG report generated.",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
