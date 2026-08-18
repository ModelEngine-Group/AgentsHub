from __future__ import annotations

import re
from typing import Any, Dict, List

from runtime_common.common import read_json, relative_to_project, resolve_path
from tool_server.utils import (
    artifact_route_path,
    latest_subgraph_public_url,
    latest_subgraph_service_url,
    load_current_metrics,
    load_server_config,
    public_artifact_url,
    safety_note,
    service_artifact_url,
)

STATIC_GRAPH_DRIVEN_PUBLIC_ITEMS = [
    {
        "title": "未来 30 天高风险随访患者疾病分布",
        "path": "outputs/runtime_generated/graph_driven_analysis/analysis_future_30d_high_risk_followup_disease_distribution_chart.html",
        "url_path": "/artifacts/graph-driven/analysis_future_30d_high_risk_followup_disease_distribution_chart",
        "kind": "html",
        "description": "针对高风险未来随访群体的疾病分布分析页。",
    },
    {
        "title": "高盐饮食与血压异常",
        "path": "outputs/runtime_generated/graph_driven_analysis/analysis_high_salt_bp_abnormal_rate_chart.html",
        "url_path": "/artifacts/graph-driven/analysis_high_salt_bp_abnormal_rate_chart",
        "kind": "html",
        "description": "高盐饮食患者血压异常比例的图谱驱动分析页。",
    },
    {
        "title": "高血压合并糖尿病多指标分析",
        "path": "outputs/runtime_generated/graph_driven_analysis/analysis_hypertension_diabetes_multi_indicator_chart.html",
        "url_path": "/artifacts/graph-driven/analysis_hypertension_diabetes_multi_indicator_chart",
        "kind": "html",
        "description": "高血压合并糖尿病患者 HbA1c、血压、LDL-C 的图谱驱动分析页。",
    },
]

LEGACY_CHART_PREFIXES = ("NLQ", "D170", "A")
FOLLOWUP_BUNDLE_PATTERN = re.compile(r"analysis_future_followup_chart_bundle_(\d+)d(?:_chart)?\.html$")


def _entry_button(label: str, href: str, *, primary: bool = False) -> str:
    if not href:
        return ""
    background = "linear-gradient(135deg,#1f5fbf,#3478f6)" if primary else "#f8fbff"
    color = "#ffffff" if primary else "#1f5fbf"
    border = "none" if primary else "1px solid #d9e2ec"
    return (
        f"<a href='{href}' target='_blank' style='display:inline-flex;align-items:center;justify-content:center;"
        f"padding:10px 16px;border-radius:12px;text-decoration:none;font-size:14px;font-weight:700;"
        f"background:{background};color:{color};border:{border};'>{label}</a>"
    )


def _entry_meta(label: str, value: str) -> str:
    if not value:
        return ""
    return (
        "<div style='margin-top:8px;padding:10px 12px;background:#f8fbff;border-radius:10px;"
        "font-size:12px;line-height:1.6;color:#52606d;word-break:break-all;'>"
        f"<strong style='color:#334e68'>{label}：</strong>{value}</div>"
    )


def _artifact_entry_card(
    *,
    title: str,
    description: str,
    browser_href: str,
    service_href: str | None = None,
    browser_label: str = "浏览器入口",
    service_label: str = "服务入口",
) -> str:
    actions = _entry_button(browser_label, browser_href, primary=True)
    if service_href and service_href != browser_href:
        actions += "&nbsp;" + _entry_button(service_label, service_href, primary=False)
    return (
        "<section style='background:white;border-radius:18px;padding:18px 20px;"
        "box-shadow:0 12px 30px rgba(16,42,67,0.08);margin-bottom:18px;'>"
        f"<h2 style='margin:0 0 10px 0;color:#102a43'>{title}</h2>"
        f"<p style='margin:0 0 12px 0;color:#486581'>{description}</p>"
        f"<div style='display:flex;flex-wrap:wrap;gap:10px;'>{actions}</div>"
        f"{_entry_meta('当前页内相对路径', browser_href)}"
        f"{_entry_meta('服务访问地址', service_href or '')}"
        "</section>"
    )


def _entry_table_row(label: str, description: str, browser_href: str, service_href: str | None = None) -> str:
    service_cell = (
        _entry_button("服务入口", service_href, primary=False)
        if service_href and service_href != browser_href
        else "<span style='color:#9fb3c8;font-size:13px;'>同浏览器入口</span>"
    )
    return (
        "<tr>"
        f"<td style='padding:12px 14px;border-bottom:1px solid #e6edf3;font-weight:700;color:#102a43;'>{label}</td>"
        f"<td style='padding:12px 14px;border-bottom:1px solid #e6edf3;color:#486581;'>{description}</td>"
        f"<td style='padding:12px 14px;border-bottom:1px solid #e6edf3;'>{_entry_button('浏览器入口', browser_href, primary=True)}</td>"
        f"<td style='padding:12px 14px;border-bottom:1px solid #e6edf3;'>{service_cell}</td>"
        "</tr>"
    )


def _dynamic_followup_items(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    latest_by_days: Dict[int, float] = {}
    for base_str in ["outputs/runtime_generated/graph_driven_analysis", "outputs/graph_driven_analysis"]:
        analysis_dir = resolve_path(base_str)
        for path in analysis_dir.glob("analysis_future_followup_chart_bundle_*d.html"):
            match = FOLLOWUP_BUNDLE_PATTERN.match(path.name)
            if not match or path.name.endswith("_chart.html"):
                continue
            days = int(match.group(1))
            latest_by_days[days] = max(latest_by_days.get(days, 0.0), path.stat().st_mtime)
    if not latest_by_days:
        return []
    latest_days = max(latest_by_days.items(), key=lambda item: item[1])[0]
    items: List[Dict[str, Any]] = []
    for days in [latest_days]:
        chart_path = None
        for candidate_str in [
            f"outputs/runtime_generated/graph_driven_analysis/analysis_future_followup_chart_bundle_{days}d_chart.html",
            f"outputs/graph_driven_analysis/analysis_future_followup_chart_bundle_{days}d_chart.html",
        ]:
            candidate = resolve_path(candidate_str)
            if candidate.exists():
                chart_path = candidate
                break
        line_path = None
        for candidate_str in [
            f"outputs/runtime_generated/charts/line_followup_trend_{days}d.svg",
            f"outputs/charts/line_followup_trend_{days}d.svg",
        ]:
            candidate = resolve_path(candidate_str)
            if candidate.exists():
                line_path = candidate
                break
        pie_path = None
        for candidate_str in [
            f"outputs/runtime_generated/charts/pie_risk_distribution_{days}d.svg",
            f"outputs/charts/pie_risk_distribution_{days}d.svg",
        ]:
            candidate = resolve_path(candidate_str)
            if candidate.exists():
                pie_path = candidate
                break
        if chart_path and chart_path.exists():
            items.append(
                {
                    "title": f"未来 {days} 天随访图表总览",
                    "kind": "html",
                    "description": f"基于真实 future follow-up 数据实时生成的未来 {days} 天总览页面，包含趋势折线图和风险分布饼图。",
                    "path": relative_to_project(chart_path),
                    "url_path": f"/artifacts/graph-driven/analysis_future_followup_chart_bundle_{days}d_chart",
                    "url": public_artifact_url(cfg, f"/artifacts/graph-driven/analysis_future_followup_chart_bundle_{days}d_chart"),
                    "service_url": service_artifact_url(cfg, f"/artifacts/graph-driven/analysis_future_followup_chart_bundle_{days}d_chart"),
                    "size_bytes": chart_path.stat().st_size,
                }
            )
        if pie_path and pie_path.exists():
            items.append(
                {
                    "title": f"未来 {days} 天随访患者风险等级分布",
                    "kind": "image",
                    "description": f"实时生成的未来 {days} 天 pending/scheduled 去重患者风险等级分布图。",
                    "path": relative_to_project(pie_path),
                    "url_path": f"/artifacts/charts/pie_risk_distribution_{days}d.svg",
                    "url": public_artifact_url(cfg, f"/artifacts/charts/pie_risk_distribution_{days}d.svg"),
                    "service_url": service_artifact_url(cfg, f"/artifacts/charts/pie_risk_distribution_{days}d.svg"),
                    "size_bytes": pie_path.stat().st_size,
                }
            )
        if line_path and line_path.exists():
            items.append(
                {
                    "title": f"未来 {days} 天随访人数趋势",
                    "kind": "image",
                    "description": f"实时生成的未来 {days} 天 pending/scheduled 去重患者每日趋势图。",
                    "path": relative_to_project(line_path),
                    "url_path": f"/artifacts/charts/line_followup_trend_{days}d.svg",
                    "url": public_artifact_url(cfg, f"/artifacts/charts/line_followup_trend_{days}d.svg"),
                    "service_url": service_artifact_url(cfg, f"/artifacts/charts/line_followup_trend_{days}d.svg"),
                    "size_bytes": line_path.stat().st_size,
                }
            )
    return items


def _graph_driven_items(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = _dynamic_followup_items(cfg)
    for item in STATIC_GRAPH_DRIVEN_PUBLIC_ITEMS:
        path = resolve_path(item["path"])
        if not path.exists():
            continue
        items.append(
            {
                "title": item["title"],
                "kind": item["kind"],
                "description": item["description"],
                "path": relative_to_project(path),
                "url_path": item["url_path"],
                "url": public_artifact_url(cfg, item["url_path"]),
                "service_url": service_artifact_url(cfg, item["url_path"]),
                "size_bytes": path.stat().st_size,
            }
        )
    return items


def _latest_graph_driven_report(cfg: Dict[str, Any]) -> Dict[str, Any] | None:
    json_paths = []
    for base_str in ["outputs/runtime_generated/graph_driven_analysis", "outputs/graph_driven_analysis"]:
        analysis_dir = resolve_path(base_str)
        json_paths.extend(analysis_dir.glob("analysis_*.json"))
    json_paths = sorted(json_paths, key=lambda item: item.stat().st_mtime)
    if not json_paths:
        return None
    latest = read_json(json_paths[-1])
    analysis_id = latest.get("analysis_id") or json_paths[-1].stem
    chart_path = json_paths[-1].parent / f"{analysis_id}_chart.html"
    inferred_chart_url = public_artifact_url(cfg, f"/artifacts/graph-driven/{analysis_id}_chart") if chart_path.exists() else None
    inferred_chart_service_url = service_artifact_url(cfg, f"/artifacts/graph-driven/{analysis_id}_chart") if chart_path.exists() else None
    return {
        "analysis_id": analysis_id,
        "title": latest.get("title") or latest.get("question") or analysis_id,
        "report_url": public_artifact_url(cfg, f"/artifacts/graph-driven/{analysis_id}"),
        "report_service_url": service_artifact_url(cfg, f"/artifacts/graph-driven/{analysis_id}"),
        "graph_url": latest.get("graph_url"),
        "chart_url": latest.get("chart_url") or inferred_chart_url,
        "chart_service_url": latest.get("chart_service_url") or inferred_chart_service_url,
        "cohort_table_url": latest.get("cohort_table_url"),
    }


def chart_gallery_html() -> str:
    cfg = load_server_config()
    items = _graph_driven_items(cfg)
    hidden_legacy = [
        path.name
        for path in sorted(resolve_path("outputs/runtime_generated/charts").glob("*.html"))
        if path.name.startswith(LEGACY_CHART_PREFIXES)
    ]
    cards: List[str] = []
    for item in items:
        preview = ""
        if item["kind"] == "image":
            preview = f"<img src='{artifact_route_path(item.get('url_path') or item['url'])}' alt='{item['title']}' style='width:100%;border:1px solid #d9e2ec;border-radius:12px;background:#fff;' />"
        else:
            preview = f"<a href='{artifact_route_path(item.get('url_path') or item['url'])}' target='_blank' style='display:inline-block;margin-top:8px;color:#1f5fbf;'>打开图表页面</a>"
        browser_href = artifact_route_path(item.get("url_path") or item["url"])
        service_href = item.get("service_url")
        cards.append(
            "<section style='background:white;border-radius:18px;padding:18px 20px;"
            "box-shadow:0 12px 30px rgba(16,42,67,0.08);margin-bottom:18px;'>"
            f"<h2 style='margin:0 0 10px 0;color:#102a43'>{item['title']}</h2>"
            f"<p style='margin:0 0 12px 0;color:#486581'>{item['description']}</p>"
            f"{preview}"
            "<div style='display:flex;flex-wrap:wrap;gap:10px;margin-top:14px;'>"
            f"{_entry_button('直接打开', browser_href, primary=True)}"
            f"{_entry_button('服务入口', service_href, primary=False) if service_href and service_href != browser_href else ''}"
            "</div>"
            f"{_entry_meta('当前页内相对路径', browser_href)}"
            f"{_entry_meta('服务访问地址', service_href or '')}"
            "</section>"
        )
    if not cards:
        cards.append("<p>当前还没有可公开展示的实时图表，请先触发图谱驱动分析生成图表。</p>")
    hidden_text = (
        f"<p style='margin-top:20px;color:#7b8794'>已隐藏 {len(hidden_legacy)} 个历史中间图表产物，避免继续暴露旧链路。</p>"
        if hidden_legacy
        else ""
    )
    return (
        "<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'><title>ChronicCare 图表画廊</title></head>"
        "<body style='font-family:Arial;padding:24px;background:#f7f9fb;color:#102a43;'>"
        "<h1>ChronicCare 图表画廊</h1>"
        "<p>这里优先展示当前系统实时生成的图谱驱动图表。</p>"
        f"{''.join(cards)}"
        f"{hidden_text}"
        "<p style='margin-top:24px;'>本项目仅用于慢病随访数据处理、知识组织和辅助分析，不提供临床诊断，不替代医生决策。</p>"
        "</body></html>"
    )


def report_overview_markdown() -> str:
    cfg = load_server_config()
    current_metrics = load_current_metrics()
    items = _graph_driven_items(cfg)
    latest = _latest_graph_driven_report(cfg)
    hidden_legacy_count = sum(
        1 for path in resolve_path("outputs/runtime_generated/charts").glob("*.html") if path.name.startswith(LEGACY_CHART_PREFIXES)
    )
    lines = [
        "# ChronicCare 当前交付报告",
        "",
        "## 当前数据概览",
        f"- 数据版本: {current_metrics.get('data_version', 'unknown')}",
        f"- 浏览器报告入口: {public_artifact_url(cfg, '/artifacts/report')}",
        f"- 图表画廊: {public_artifact_url(cfg, '/artifacts/charts')}",
        f"- 默认图谱入口（优先打开最近一次问题驱动子图）: {public_artifact_url(cfg, '/artifacts/graph.html')}",
    ]
    if latest:
        lines.append(f"- 最新图谱驱动分析: {latest.get('title')} -> {latest.get('report_url')}")
        if latest.get("chart_url"):
            lines.append(f"- 最新分析图表页: {latest.get('chart_url')}")
        if latest.get("graph_url"):
            lines.append(f"- 最新分析专属子图: {latest.get('graph_url')}")
        if latest.get("cohort_table_url"):
            lines.append(f"- 最新分析全量患者列表: {latest.get('cohort_table_url')}")
    lines.extend(
        [
            "",
            "## 当前公开图表",
        ]
    )
    if items:
        lines.extend([f"- {item['title']}: {item['url']}" for item in items])
    else:
        lines.append("- 当前暂无公开图表，请先触发图谱驱动分析。")
    lines.extend(
        [
            "",
            "## 交付说明",
            f"- 已隐藏历史中间图表: {hidden_legacy_count} 个",
            "- 当前入口只保留对外交付所需的稳定 report / charts / graph 页面，图谱入口会优先打开最近一次真实生成的问题驱动子图。",
            f"- 安全声明: {safety_note(cfg)}",
        ]
    )
    return "\n".join(lines)


def report_overview_html() -> str:
    cfg = load_server_config()
    current_metrics = load_current_metrics()
    items = _graph_driven_items(cfg)
    latest = _latest_graph_driven_report(cfg)
    hidden_legacy_count = sum(
        1 for path in resolve_path("outputs/runtime_generated/charts").glob("*.html") if path.name.startswith(LEGACY_CHART_PREFIXES)
    )
    latest_block = ""
    if latest:
        analysis_id = latest.get("analysis_id")
        extra_links = ""
        if latest.get("chart_url"):
            extra_links += f"<p style='margin:8px 0 0 0;'><a href='{artifact_route_path(f'/artifacts/graph-driven/{analysis_id}_chart')}' target='_blank'>查看该分析对应图表页</a></p>"
        if latest.get("graph_url"):
            extra_links += f"<p style='margin:8px 0 0 0;'><a href='{artifact_route_path('/artifacts/graph.html')}' target='_blank'>查看该分析对应图谱子图</a></p>"
        if latest.get("cohort_table_url"):
            extra_links += f"<p style='margin:8px 0 0 0;'><a href='{artifact_route_path(f'/artifacts/graph-driven/{analysis_id}_patients')}' target='_blank'>查看该分析对应全量患者列表</a></p>"
        latest_block = (
            "<section style='background:white;border-radius:18px;padding:18px 20px;"
            "box-shadow:0 12px 30px rgba(16,42,67,0.08);margin:18px 0;'>"
            "<h2 style='margin:0 0 10px 0;color:#102a43'>最新图谱驱动分析</h2>"
            f"<p style='margin:0;color:#486581'>{latest.get('title')}</p>"
            f"<p style='margin:12px 0 0 0;'><a href='{artifact_route_path(f'/artifacts/graph-driven/{analysis_id}')}' target='_blank'>当前页内直接打开</a></p>"
            f"<p style='margin:8px 0 0 0;'><a href='{latest.get('report_url')}' target='_blank'>{latest.get('report_url')}</a></p>"
            f"{extra_links}"
            "</section>"
        )
    cards = []
    for item in items:
        cards.append(
            _entry_table_row(
                item["title"],
                item["description"],
                artifact_route_path(item.get("url_path") or item["url"]),
                item.get("service_url"),
            )
        )
    if not cards:
        cards.append("<tr><td colspan='4' style='padding:10px 12px;'>当前暂无公开图表，请先触发图谱驱动分析。</td></tr>")
    return (
        "<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'><title>ChronicCare 当前交付报告</title></head>"
        "<body style='font-family:Arial;padding:24px;background:#f7f9fb;color:#102a43;'>"
        "<h1>ChronicCare 当前交付报告</h1>"
        "<p>这里展示的是当前对外公开的稳定入口。</p>"
        "<section style='background:white;border-radius:18px;padding:18px 20px;box-shadow:0 12px 30px rgba(16,42,67,0.08);'>"
        "<h2 style='margin:0 0 12px 0;color:#102a43'>当前数据概览</h2>"
        f"<p style='margin:0 0 8px 0;color:#486581'>数据版本：{current_metrics.get('data_version', 'unknown')}</p>"
        "<div style='display:flex;flex-wrap:wrap;gap:10px;margin:12px 0 0 0;'>"
        f"{_entry_button('打开分析报告', artifact_route_path('/artifacts/report'), primary=True)}"
        f"{_entry_button('打开图表画廊', artifact_route_path('/artifacts/charts'), primary=False)}"
        f"{_entry_button('打开默认图谱', artifact_route_path('/artifacts/graph.html'), primary=False)}"
        "</div>"
        "</section>"
        f"{latest_block}"
        "<section style='background:white;border-radius:18px;padding:18px 20px;box-shadow:0 12px 30px rgba(16,42,67,0.08);margin:18px 0;'>"
        "<h2 style='margin:0 0 12px 0;color:#102a43'>当前公开图表</h2>"
        "<table style='width:100%;border-collapse:collapse;border:1px solid #d9e2ec;border-radius:14px;overflow:hidden;'>"
        "<thead><tr><th style='text-align:left;padding:10px 12px;border-bottom:1px solid #d9e2ec;'>名称</th><th style='text-align:left;padding:10px 12px;border-bottom:1px solid #d9e2ec;'>说明</th><th style='text-align:left;padding:10px 12px;border-bottom:1px solid #d9e2ec;'>浏览器入口</th><th style='text-align:left;padding:10px 12px;border-bottom:1px solid #d9e2ec;'>服务入口</th></tr></thead>"
        f"<tbody>{''.join(cards)}</tbody></table>"
        "</section>"
        "<section style='background:white;border-radius:18px;padding:18px 20px;box-shadow:0 12px 30px rgba(16,42,67,0.08);'>"
        "<h2 style='margin:0 0 12px 0;color:#102a43'>交付说明</h2>"
        f"<p style='margin:0 0 8px 0;color:#486581'>已隐藏历史中间图表：{hidden_legacy_count} 个。</p>"
        "<p style='margin:0 0 8px 0;color:#486581'>当前入口仅保留对外交付所需的稳定 report / charts / graph 页面，图谱入口会优先打开最近一次真实生成的问题驱动子图。</p>"
        f"<p style='margin:0;color:#486581'>安全声明：{safety_note(cfg)}</p>"
        "</section>"
        "</body></html>"
    )


def reports_summary() -> Dict[str, Any]:
    cfg = load_server_config()
    current_metrics = load_current_metrics()
    latest = _latest_graph_driven_report(cfg)
    latest_graph_url = latest.get("graph_url") if latest and latest.get("graph_url") else latest_subgraph_public_url(cfg)
    latest_graph_service_url = latest.get("graph_service_url") if latest and latest.get("graph_service_url") else latest_subgraph_service_url(cfg)
    global_graph_url = public_artifact_url(cfg, "/artifacts/graph.html")
    global_graph_service_url = service_artifact_url(cfg, "/artifacts/graph.html")
    return {
        "status": "success",
        "data_version": current_metrics.get("data_version"),
        "analysis_report_html": public_artifact_url(cfg, "/artifacts/report"),
        "analysis_report_html_path": cfg["paths"]["analysis_report_html"],
        "analysis_report_md": public_artifact_url(cfg, "/artifacts/report.md"),
        "analysis_report_md_path": cfg["paths"]["analysis_report_md"],
        "chart_index": public_artifact_url(cfg, "/artifacts/charts"),
        "chart_index_path": cfg["paths"]["chart_index"],
        "graph_html": public_artifact_url(cfg, "/artifacts/graph.html"),
        "graph_html_path": cfg["paths"]["graph_html"],
        "graph_url": global_graph_url,
        "graph_service_url": global_graph_service_url,
        "global_graph_url": global_graph_url,
        "global_graph_service_url": global_graph_service_url,
        "latest_subgraph_url": latest_graph_url,
        "latest_subgraph_service_url": latest_graph_service_url,
        "chart_index_url": public_artifact_url(cfg, "/artifacts/charts"),
        "chart_index_service_url": service_artifact_url(cfg, "/artifacts/charts"),
        "report_url": public_artifact_url(cfg, "/artifacts/report"),
        "report_service_url": service_artifact_url(cfg, "/artifacts/report"),
        "latest_graph_driven_analysis": latest,
        "summary_text": "当前已整理出稳定的分析报告入口、图表画廊入口和图谱入口。优先查看最新问题驱动分析，其次查看全局图表与报告总览。",
        "entry_guide": [
            {
                "name": "分析报告入口",
                "url": public_artifact_url(cfg, "/artifacts/report"),
                "route_path": artifact_route_path("/artifacts/report"),
                "service_url": service_artifact_url(cfg, "/artifacts/report"),
            },
            {
                "name": "图表总览入口",
                "url": public_artifact_url(cfg, "/artifacts/charts"),
                "route_path": artifact_route_path("/artifacts/charts"),
                "service_url": service_artifact_url(cfg, "/artifacts/charts"),
            },
            {
                "name": "图谱入口",
                "url": global_graph_url,
                "route_path": artifact_route_path("/artifacts/graph.html"),
                "service_url": global_graph_service_url,
            },
            {
                "name": "最新问题驱动子图入口",
                "url": latest_graph_url,
                "route_path": artifact_route_path(latest_graph_url or "/artifacts/graph.html"),
                "service_url": latest_graph_service_url,
            },
        ],
        "demo_manifest": cfg["paths"]["demo_manifest"],
        "safety_note": safety_note(cfg),
    }


def charts_list() -> Dict[str, Any]:
    cfg = load_server_config()
    chart_dir = resolve_path("outputs/runtime_generated/charts")
    public_items = _graph_driven_items(cfg)
    hidden_legacy_count = sum(1 for path in chart_dir.glob("*.html") if path.name.startswith(LEGACY_CHART_PREFIXES))
    return {
        "status": "success",
        "gallery_url": public_artifact_url(cfg, "/artifacts/charts"),
        "charts": public_items,
        "public_chart_count": len(public_items),
        "hidden_legacy_chart_count": hidden_legacy_count,
        "safety_note": safety_note(cfg),
    }
