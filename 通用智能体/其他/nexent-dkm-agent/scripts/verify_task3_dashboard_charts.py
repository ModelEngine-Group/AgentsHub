"""Verify ECharts dashboard charts render with visible series data."""
from __future__ import annotations

import json
import re
from pathlib import Path

from playwright.sync_api import sync_playwright

HTML = Path(
    "competition_submission/defense-package-final/evidence/html/task3_interactive_dashboard.html"
).resolve()
PNG = Path(
    "competition_submission/defense-package-final/evidence/screenshots/task3/01_interactive_dashboard_overview.png"
)
OUT = Path("outputs/task3_verify")
OUT.mkdir(parents=True, exist_ok=True)


def _count_series_pixels(canvas) -> dict[str, int]:
    ctx = canvas.get_context("2d")
    img = ctx.get_image_data(0, 0, canvas.width, canvas.height)
    stats = {"blue": 0, "green": 0, "red": 0, "nonwhite": 0}
    for i in range(0, len(img["data"]), 4):
        r, g, b, a = img["data"][i : i + 4]
        if a < 10:
            continue
        if r > 240 and g > 240 and b > 240:
            continue
        stats["nonwhite"] += 1
        if b > 150 and r < 120:
            stats["blue"] += 1
        if g > 120 and r < 120 and b < 120:
            stats["green"] += 1
        if r > 180 and g < 120 and b < 120:
            stats["red"] += 1
    return stats


def check_html_series() -> dict:
    content = HTML.read_text(encoding="utf-8")
    entity = re.search(r'"name": "实体数".*?"data": (\[[^\]]+\])', content, re.S)
    edge = re.search(r'"name": "边数".*?"data": (\[[^\]]+\])', content, re.S)
    relation = re.search(
        r'"type": "bar", "data": (\[[^\]]+\]).*?"关系类型分布"|"关系类型分布".*?"type": "bar", "data": (\[[^\]]+\])',
        content,
        re.S,
    )
    rel_data = None
    if relation:
        rel_data = relation.group(1) or relation.group(2)
    hub = re.search(r'"type": "bar", "data": (\[[^\]]+\])', content)
    return {
        "html_exists": HTML.exists(),
        "entity_series": json.loads(entity.group(1)) if entity else None,
        "edge_series": json.loads(edge.group(1)) if edge else None,
        "relation_bar_data": json.loads(rel_data) if rel_data else None,
        "hub_bar_data": json.loads(hub.group(1)) if hub else None,
        "has_rAF_init": "requestAnimationFrame" in content,
        "has_resize": "chart.resize()" in content,
        "no_zero_entity_series": entity and all(v > 0 for v in json.loads(entity.group(1))),
        "no_empty_data_msg": "暂无数据" not in content,
    }


def check_browser_render() -> dict:
    results: dict = {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(HTML.as_uri(), wait_until="networkidle")
        page.wait_for_function(
            """() => {
                if (typeof echarts === 'undefined') return false;
                const el = document.getElementById('relation-chart');
                const inst = el && echarts.getInstanceByDom(el);
                return !!inst && !!el.querySelector('canvas');
            }""",
            timeout=15000,
        )
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "browser_overview_900.png"), full_page=False)

        for chart_id, key in [
            ("entity-chart", "entity_pie"),
            ("relation-chart", "relation_bar"),
            ("trend-chart", "trend_line"),
            ("hub-chart", "hub_bar"),
        ]:
            stats = page.evaluate(
                """(chartId) => {
                    const el = document.getElementById(chartId);
                    const canvas = el.querySelector('canvas');
                    if (!canvas) return {ok: false, reason: 'no canvas'};
                    const ctx = canvas.getContext('2d');
                    const img = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
                    let blue = 0, green = 0, red = 0, nonwhite = 0;
                    for (let i = 0; i < img.length; i += 4) {
                        const r = img[i], g = img[i+1], b = img[i+2], a = img[i+3];
                        if (a < 10) continue;
                        if (r > 240 && g > 240 && b > 240) continue;
                        nonwhite++;
                        if (b > 150 && r < 120) blue++;
                        if (g > 120 && r < 120 && b < 120) green++;
                        if (r > 180 && g < 120 && b < 120) red++;
                    }
                    const inst = echarts.getInstanceByDom(el);
                    return {
                        ok: nonwhite > 100,
                        canvas: [canvas.width, canvas.height],
                        pixels: {blue, green, red, nonwhite},
                        seriesData: inst ? inst.getOption().series.map(s => s.data) : null,
                    };
                }""",
                chart_id,
            )
            results[key] = stats

        browser.close()
    return results


def main() -> int:
    html_check = check_html_series()
    render_check = check_browser_render()
    report = {
        "html_check": html_check,
        "render_check": render_check,
        "png_exists": PNG.exists(),
        "png_size_bytes": PNG.stat().st_size if PNG.exists() else 0,
    }

    passed = (
        html_check["entity_series"] == [8, 8, 7, 7]
        and html_check["edge_series"] == [7, 7, 6, 9]
        and html_check["no_zero_entity_series"]
        and html_check["no_empty_data_msg"]
        and html_check["has_rAF_init"]
        and render_check["relation_bar"]["ok"]
        and render_check["trend_line"]["ok"]
        and render_check["hub_bar"]["ok"]
        and render_check["entity_pie"]["ok"]
    )
    report["verification_passed"] = passed
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
