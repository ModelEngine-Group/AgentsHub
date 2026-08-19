"""Competition-facing SVG figure export for defense and evidence bundles."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.common.svg_charts import bar_chart_svg, chart_spec_to_svg, esc, grouped_bar_chart_svg, network_chart_svg, svg_header


def export_architecture_diagram(target: str | Path) -> dict[str, Any]:
    """Export the DKM layered architecture diagram."""

    width, height = 980, 430
    top_boxes = [
        (40, 72, 180, 88, "Demo / API", "#e8f1ff"),
        (250, 72, 180, 88, "Agent 层", "#dbeafe"),
        (460, 72, 180, 88, "Operator 层", "#dcfce7"),
        (670, 72, 180, 88, "Pipeline 层", "#fef3c7"),
    ]
    bottom_boxes = [
        (40, 228, 180, 88, ["CLI · REST", "入口"], "#f8fafc"),
        (250, 228, 180, 88, ["Nexent Adapter", "Agent Spec"], "#ede9fe"),
        (460, 228, 180, 88, ["DataMate · Neo4j", "LLM 可选增强"], "#ffe4e6"),
        (670, 228, 180, 88, ["Benchmark / pytest", "End-to-End"], "#f1f5f9"),
    ]
    parts = [
        svg_header(width, height),
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
        '<path d="M0,0 L6,3 L0,6 Z" fill="#64748b"/></marker></defs>',
        '<text x="24" y="34" font-size="22" font-weight="700">DKM Agent Architecture</text>',
        '<text x="24" y="56" font-size="13" fill="#475569">'
        "Demo 入口 → Agent 规划 → Operator 执行 → Pipeline 编排；底部为各层外部集成与验证"
        "</text>",
    ]
    for x, y, w, h, label, fill in top_boxes:
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="#94a3b8"/>')
        parts.append(
            f'<text x="{x + w / 2:.1f}" y="{y + h / 2 + 5:.1f}" text-anchor="middle" '
            f'font-size="14" font-weight="600">{esc(label)}</text>'
        )
    for x, y, w, h, lines, fill in bottom_boxes:
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="#94a3b8"/>')
        if len(lines) == 1:
            parts.append(
                f'<text x="{x + w / 2:.1f}" y="{y + h / 2 + 5:.1f}" text-anchor="middle" '
                f'font-size="13" font-weight="600">{esc(lines[0])}</text>'
            )
        else:
            parts.append(
                f'<text x="{x + w / 2:.1f}" y="{y + 34:.1f}" text-anchor="middle" '
                f'font-size="13" font-weight="600">{esc(lines[0])}</text>'
            )
            parts.append(
                f'<text x="{x + w / 2:.1f}" y="{y + 54:.1f}" text-anchor="middle" '
                f'font-size="12" fill="#475569">{esc(lines[1])}</text>'
            )
    for x, y, w, _h, _label, _fill in top_boxes:
        cx = x + w / 2
        parts.append(
            f'<line x1="{cx:.1f}" y1="{y + 88:.1f}" x2="{cx:.1f}" y2="228" '
            f'stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>'
        )
    for index in range(len(top_boxes) - 1):
        x1 = top_boxes[index][0] + top_boxes[index][2]
        x2 = top_boxes[index + 1][0]
        y = top_boxes[index][1] + top_boxes[index][3] / 2
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" '
            f'stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>'
        )
    parts.append(
        '<text x="24" y="360" font-size="13" fill="#334155">'
        "业务闭环：Task1 数据清洗 → Task2 医疗 KG → Task3 NL2SQL / BI 洞察"
        "</text>"
    )
    parts.append(
        '<text x="24" y="382" font-size="12" fill="#64748b">'
        "跨任务串联由 Pipeline 层 end_to_end_pipeline 与 DKMOrchestrator 负责，Agent 之间不直接耦合"
        "</text>"
    )
    parts.append("</svg>")
    return _write_svg(target, "".join(parts), name="architecture_diagram")


def export_dkm_workflow_diagram(target: str | Path) -> dict[str, Any]:
    """Export the end-to-end DKM workflow diagram."""

    width, height = 980, 260
    stages = [
        (60, 110, "Task 1\nData Processing", "#dbeafe"),
        (300, 110, "Task 2\nMedical KG", "#dcfce7"),
        (540, 110, "Task 3\nGraph Analysis", "#fde68a"),
        (780, 110, "Insight\nReports", "#ede9fe"),
    ]
    parts = [
        svg_header(width, height),
        '<text x="24" y="34" font-size="22" font-weight="700">Data -> Knowledge -> Insight</text>',
    ]
    for x, y, label, fill in stages:
        parts.append(f'<rect x="{x}" y="{y}" width="150" height="84" rx="14" fill="{fill}" stroke="#64748b"/>')
        for line_index, line in enumerate(label.split("\n")):
            parts.append(
                f'<text x="{x + 75:.1f}" y="{y + 34 + line_index * 18:.1f}" text-anchor="middle" font-size="13" font-weight="600">{esc(line)}</text>'
            )
    for x1, x2 in ((210, 300), (450, 540), (690, 780)):
        parts.append(f'<line x1="{x1}" y1="152" x2="{x2}" y2="152" stroke="#64748b" stroke-width="3"/>')
        parts.append(f'<polygon points="{x2 - 8},147 {x2},152 {x2 - 8},157" fill="#64748b"/>')
    parts.append(
        '<text x="24" y="230" font-size="12" fill="#475569">DKMOrchestrator plans and chains stages; Nexent suite registers all three tools.</text>'
    )
    parts.append("</svg>")
    return _write_svg(target, "".join(parts), name="dkm_workflow")


def export_task1_quality_figure(report_path: str | Path, target: str | Path) -> dict[str, Any]:
    """Export before/after data quality metrics from a task-1 benchmark report."""

    if not Path(report_path).exists():
        return {"status": "skipped", "name": "task1_quality_improvement", "reason": "missing report"}
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    metrics = report.get("quality_metrics", {})
    chart = {
        "type": "grouped_bar",
        "title": "Task 1 data quality improvement",
        "groups": [
            {
                "label": "Quality score",
                "series": [
                    {"label": "before", "value": metrics.get("quality_score_before", 0)},
                    {"label": "after", "value": metrics.get("quality_score_after", 0)},
                ],
            },
            {
                "label": "Duplicate rows",
                "series": [
                    {"label": "before", "value": metrics.get("duplicate_rows_before", 0)},
                    {"label": "after", "value": metrics.get("duplicate_rows_after", 0)},
                ],
            },
            {
                "label": "Missing values",
                "series": [
                    {"label": "before", "value": metrics.get("missing_values_before", 0)},
                    {"label": "after", "value": metrics.get("missing_values_after", 0)},
                ],
            },
        ],
    }
    return _write_svg(target, grouped_bar_chart_svg(chart), name="task1_quality_improvement")


def export_oov_extraction_figure(report_path: str | Path, target: str | Path) -> dict[str, Any]:
    """Export closed-vocabulary vs OOV extraction F1/recall comparison."""

    if not Path(report_path).exists():
        return {"status": "skipped", "name": "task2_oov_extraction", "reason": "missing report"}
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    comparison = report.get("comparison", {})
    closed = report.get("closed_corpus", {}).get("overall", {})
    oov_corpus = report.get("oov_corpus", {}).get("overall", {})
    chart = {
        "type": "grouped_bar",
        "title": "Task 2 extraction: closed vocabulary vs OOV hold-out",
        "groups": [
            {
                "label": "F1",
                "series": [
                    {"label": "closed (30 records)", "value": comparison.get("closed_overall_f1", closed.get("f1", 0))},
                    {"label": "OOV (8 records)", "value": comparison.get("oov_overall_f1", oov_corpus.get("f1", 0))},
                ],
            },
            {
                "label": "Recall",
                "series": [
                    {"label": "closed", "value": closed.get("recall", 0)},
                    {"label": "OOV", "value": comparison.get("oov_recall", 0)},
                ],
            },
        ],
    }
    return _write_svg(target, grouped_bar_chart_svg(chart), name="task2_oov_extraction")


def export_task2_pipeline_latency_figure(report_path: str | Path, target: str | Path) -> dict[str, Any]:
    """Export end-to-end task-2 pipeline latency (rule vs tensor CPU)."""

    if not Path(report_path).exists():
        return {"status": "skipped", "name": "task2_pipeline_latency", "reason": "missing report"}
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    rule = report.get("rule_backend", {})
    tensor = report.get("tensor_cpu_backend", {})
    chart = {
        "type": "bar",
        "title": "Task 2 end-to-end pipeline latency (ms avg, lower is better)",
        "data": [
            {"category": "Rule pipeline", "value": rule.get("latency_ms_avg", 0)},
            {"category": "Tensor CPU pipeline", "value": tensor.get("latency_ms_avg", 0)},
        ],
    }
    return _write_svg(target, bar_chart_svg(chart), name="task2_pipeline_latency")


def export_planner_operator_figure(report_path: str | Path, target: str | Path) -> dict[str, Any]:
    """Export rule vs enhanced operator counts per task from planner evidence."""

    if not Path(report_path).exists():
        return {"status": "skipped", "name": "planner_operator_comparison", "reason": "missing report"}
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    diff = report.get("diff_summary") or []
    groups = []
    for item in diff:
        if item.get("task") == "orchestrator":
            continue
        groups.append({
            "label": item["task"],
            "series": [
                {"label": "rule", "value": item.get("rule_operator_count", 0)},
                {"label": "enhanced", "value": item.get("enhanced_operator_count", 0)},
            ],
        })
    if not groups:
        return {"status": "skipped", "name": "planner_operator_comparison", "reason": "empty diff"}
    chart = {
        "type": "grouped_bar",
        "title": "Planner operator counts: rule vs LLM-enhanced",
        "groups": groups,
    }
    return _write_svg(target, grouped_bar_chart_svg(chart), name="planner_operator_comparison")


def export_nl2sql_accuracy_figure(report_path: str | Path, target: str | Path) -> dict[str, Any]:
    """Export NL2SQL intent / execution / holdout accuracy bars."""

    if not Path(report_path).exists():
        return {"status": "skipped", "name": "task3_nl2sql_accuracy", "reason": "missing report"}
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    intent = report.get("intent_classification", report)
    execution = report.get("execution", {})
    holdout = report.get("holdout_generalization", {})
    chart = {
        "type": "bar",
        "title": "任务三 NL2SQL 评测准确率",
        "data": [
            {
                "category": "意图分类",
                "value": round(intent.get("accuracy", report.get("accuracy", 0)) * 100, 2),
            },
            {
                "category": "执行级匹配",
                "value": round(execution.get("accuracy", 0) * 100, 2),
            },
            {
                "category": "改写回归",
                "value": round(holdout.get("accuracy", 0) * 100, 2),
            },
        ],
    }
    return _write_svg(target, bar_chart_svg(chart), name="task3_nl2sql_accuracy")


def export_kg_overview_figure(graph_path: str | Path, target: str | Path) -> dict[str, Any]:
    """Export a task-2 KG overview network figure."""

    if not Path(graph_path).exists():
        return {"status": "skipped", "name": "task2_kg_overview", "reason": "missing graph"}
    graph = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    nodes = [
        {
            "id": node.get("id", node.get("name", index)),
            "label": node.get("name", node.get("id", "")),
            "type": node.get("type", "Entity"),
        }
        for index, node in enumerate(graph.get("nodes", []))
    ]
    edges = []
    for edge in graph.get("edges", []):
        source = edge.get("source") or edge.get("from")
        edge_target = edge.get("target") or edge.get("to")
        if source and edge_target:
            edges.append({"source": source, "target": edge_target, "relation": edge.get("type", "related")})
    chart = {
        "type": "network",
        "title": "Task 2 medical knowledge graph overview",
        "nodes": nodes,
        "edges": edges,
    }
    return _write_svg(target, network_chart_svg(chart), name="task2_kg_overview")


def export_kg_type_distribution_figure(graph_path: str | Path, target: str | Path) -> dict[str, Any]:
    """Export entity type distribution for task-2 graph."""

    if not Path(graph_path).exists():
        return {"status": "skipped", "name": "task2_entity_types", "reason": "missing graph"}
    graph = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    counts = Counter(str(node.get("type", "Unknown")) for node in graph.get("nodes", []))
    chart = {
        "type": "bar",
        "title": "Task 2 entity type distribution",
        "data": [{"category": key, "value": value} for key, value in sorted(counts.items())],
    }
    return _write_svg(target, bar_chart_svg(chart), name="task2_entity_types")


def export_task3_figures_from_report(report_path: Path, figure_dir: Path) -> list[dict[str, str]]:
    """Export task-3 chart specs as standalone SVG files."""

    if not report_path.exists():
        return []
    report = json.loads(report_path.read_text(encoding="utf-8"))
    charts = report.get("visualizations", {}).get("charts", {})
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures: list[dict[str, str]] = []
    for name, chart in charts.items():
        svg = chart_spec_to_svg(chart)
        if not svg:
            continue
        target = figure_dir / f"{name}.svg"
        target.write_text(svg, encoding="utf-8")
        figures.append({"name": name, "chart": name, "path": str(target), "type": chart.get("type", "unknown")})
    return figures


def export_npu_mode_speedup_figure(
    report_path: str | Path,
    target: str | Path,
    name: str,
    title: str,
) -> dict[str, Any]:
    """Export a tensor-benchmark mode speedup bar chart (vs CPU baseline).

    Reads a relation/graph tensor report's ``mode_benchmarks`` and plots each
    mode's ``speedup_vs_cpu`` alongside the CPU baseline (1.0x).
    """

    if not Path(report_path).exists():
        return {"status": "skipped", "name": name, "reason": "missing report"}
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    modes = report.get("mode_benchmarks", [])
    rows = [{"category": "cpu_baseline", "value": 1.0}]
    for mode in modes:
        speedup = mode.get("speedup_vs_cpu")
        if speedup is not None:
            rows.append({"category": mode.get("name", "mode"), "value": round(float(speedup), 2)})
    if len(rows) <= 1:
        return {"status": "skipped", "name": name, "reason": "no mode speedups"}
    chart = {"type": "bar", "title": title, "data": rows}
    return _write_svg(target, bar_chart_svg(chart), name=name)


def export_npu_utilization_figure(
    report_paths: dict[str, str | Path],
    target: str | Path,
) -> dict[str, Any]:
    """Export an NPU utilization/power figure from benchmark ``npu_utilization``.

    ``report_paths`` maps a workload label to a benchmark report path that
    contains an ``npu_utilization`` block (from ``benchmarks/npu_monitor.py``).
    Plots peak NPU utilization (%) and peak power (W) per workload.
    """

    groups = []
    for label, path in report_paths.items():
        if not Path(path).exists():
            continue
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        util = report.get("npu_utilization", {})
        if not util.get("available"):
            continue
        npu_util = util.get("npu_utilization_pct", {})
        power = util.get("power_w", {})
        groups.append({
            "label": label,
            "series": [
                {"label": "NPU util max (%)", "value": npu_util.get("max", 0)},
                {"label": "Power max (W)", "value": power.get("max", 0)},
            ],
        })
    if not groups:
        return {"status": "skipped", "name": "npu_utilization", "reason": "no utilization data"}
    chart = {"type": "grouped_bar", "title": "NPU utilization & power (peak)", "groups": groups}
    return _write_svg(target, grouped_bar_chart_svg(chart), name="npu_utilization")


def export_all_defense_figures(
    output_dir: str | Path,
    task1_quality_report: str | Path | None = None,
    kg_graph_file: str | Path | None = None,
    task3_report_file: str | Path | None = None,
    oov_extraction_report: str | Path | None = None,
    nl2sql_report: str | Path | None = None,
    planner_llm_report: str | Path | None = None,
    pipeline_latency_report: str | Path | None = None,
    task2_tensor_report: str | Path | None = None,
    task3_tensor_report: str | Path | None = None,
    npu_utilization_reports: dict[str, str | Path] | None = None,
) -> list[dict[str, Any]]:
    """Generate the full defense figure set into one directory."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = [
        export_architecture_diagram(output / "architecture_diagram.svg"),
        export_dkm_workflow_diagram(output / "dkm_workflow.svg"),
    ]
    if task1_quality_report:
        manifest.append(export_task1_quality_figure(task1_quality_report, output / "task1_quality_improvement.svg"))
    if kg_graph_file:
        manifest.append(export_kg_overview_figure(kg_graph_file, output / "task2_kg_overview.svg"))
        manifest.append(export_kg_type_distribution_figure(kg_graph_file, output / "task2_entity_types.svg"))
    if task3_report_file:
        manifest.extend(
            {
                "status": "completed",
                "name": item["name"],
                "path": item["path"],
                "type": item.get("type", "unknown"),
            }
            for item in export_task3_figures_from_report(Path(task3_report_file), output)
        )
    if oov_extraction_report:
        manifest.append(export_oov_extraction_figure(
            oov_extraction_report, output / "task2_oov_extraction.svg",
        ))
    if nl2sql_report:
        manifest.append(export_nl2sql_accuracy_figure(
            nl2sql_report, output / "task3_nl2sql_accuracy.svg",
        ))
    if planner_llm_report:
        manifest.append(export_planner_operator_figure(
            planner_llm_report, output / "planner_operator_comparison.svg",
        ))
    if pipeline_latency_report:
        manifest.append(export_task2_pipeline_latency_figure(
            pipeline_latency_report, output / "task2_pipeline_latency.svg",
        ))
    if task2_tensor_report:
        manifest.append(export_npu_mode_speedup_figure(
            task2_tensor_report, output / "npu_task2_mode_speedup.svg",
            name="npu_task2_mode_speedup", title="Task 2 relation tensor: NPU mode speedup vs CPU",
        ))
    if task3_tensor_report:
        manifest.append(export_npu_mode_speedup_figure(
            task3_tensor_report, output / "npu_task3_mode_speedup.svg",
            name="npu_task3_mode_speedup", title="Task 3 graph tensor: NPU mode speedup vs CPU",
        ))
    if npu_utilization_reports:
        manifest.append(export_npu_utilization_figure(
            npu_utilization_reports, output / "npu_utilization.svg",
        ))
    return [item for item in manifest if item.get("status") != "skipped"]


def _write_svg(target: str | Path, svg: str, name: str) -> dict[str, Any]:
    path = Path(target)
    if not svg:
        return {"status": "skipped", "name": name, "reason": "empty svg"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    return {"status": "completed", "name": name, "path": str(path)}
