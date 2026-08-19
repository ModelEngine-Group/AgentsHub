"""Render Task 3 dashboard PNG previews from self-contained HTML evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HTML = (
    ROOT
    / "competition_submission"
    / "defense-package-final"
    / "evidence"
    / "html"
    / "task3_interactive_dashboard.html"
)
DEFAULT_OUT = (
    ROOT
    / "competition_submission"
    / "defense-package-final"
    / "evidence"
    / "screenshots"
    / "task3"
)


def _sync_html_sources(source_dir: Path, evidence_html_dir: Path) -> list[str]:
    copied: list[str] = []
    evidence_html_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "task3_interactive_dashboard.html",
        "task3_analysis_dashboard.html",
        "task3_insight_report.html",
    ):
        src = source_dir / name
        if not src.exists():
            continue
        dst = evidence_html_dir / name
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def render_screenshots(html_path: Path, output_dir: Path) -> dict[str, str]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "playwright is required: pip install playwright && python -m playwright install chromium"
        ) from exc

    if not html_path.exists():
        raise SystemExit(f"HTML not found: {html_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    file_url = html_path.resolve().as_uri()
    overview = output_dir / "01_interactive_dashboard_overview.png"
    nl2sql = output_dir / "02_interactive_dashboard_graph_nl2sql.png"

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(file_url, wait_until="networkidle")
        page.wait_for_function(
            """() => {
                if (typeof echarts === 'undefined') return false;
                const el = document.getElementById('relation-chart');
                if (!el) return false;
                const inst = echarts.getInstanceByDom(el);
                if (!inst) return false;
                const canvas = el.querySelector('canvas');
                if (!canvas || canvas.width === 0 || canvas.height === 0) return false;
                const ctx = canvas.getContext('2d');
                const img = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
                for (let i = 0; i < img.length; i += 4) {
                    if (img[i + 2] > 180 && img[i] < 120) return true;
                }
                return false;
            }""",
            timeout=15000,
        )
        page.wait_for_timeout(300)
        page.screenshot(path=str(overview), full_page=False)

        nl2sql_panel = page.locator("text=NL2SQL Evidence").first
        nl2sql_panel.scroll_into_view_if_needed()
        page.wait_for_timeout(300)
        box = nl2sql_panel.locator("xpath=ancestor::div[contains(@class,'panel')]").first
        box.screenshot(path=str(nl2sql))
        browser.close()

    return {
        "overview": str(overview),
        "nl2sql": str(nl2sql),
        "html": str(html_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Task 3 dashboard PNG previews.")
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--sync-from",
        type=Path,
        default=None,
        help="Copy task3 HTML outputs into evidence/html before rendering.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    html_path = args.html
    if args.sync_from is not None:
        evidence_html = html_path.parent
        copied = _sync_html_sources(args.sync_from, evidence_html)
        print(json.dumps({"synced_html": copied}, ensure_ascii=False, indent=2))
    summary = render_screenshots(html_path, args.output_dir)
    print(json.dumps({"status": "completed", **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
