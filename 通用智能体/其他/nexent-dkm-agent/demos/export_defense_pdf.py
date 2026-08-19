"""Build a polished, self-contained defense HTML report with embedded figures.

Usage:
    python demos/export_defense_pdf.py
    python demos/export_defense_pdf.py --source outputs/competition_evidence/defense-package-final
    python demos/export_defense_pdf.py --source competition_submission/defense-package-final --sync-from docs/competition_defense_document.md
"""

from __future__ import annotations

import argparse
import base64
import html
import mimetypes
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SOURCE = ROOT / "outputs" / "competition_evidence" / "defense-package-final"
IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
BOLD_PATTERN = re.compile(r"\*\*(.+?)\*\*")
CODE_PATTERN = re.compile(r"`([^`]+)`")
SUBSECTION_LEAD = re.compile(r"^\*\*\d+(?:\.\d+)+\s+.+?\*\*$")
TABLE_SEPARATOR_CELL = re.compile(r"^:?-{3,}:?$")
MATERIAL_DATE_PATTERN = re.compile(r"^\*\*材料日期：\*\*\s*(.+?)\s*$", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export defense Markdown to a styled HTML report.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML path. Defaults to <source>/competition_defense_document.html",
    )
    parser.add_argument(
        "--sync-from",
        default=None,
        help="Rewrite docs/competition_defense_document.md into <source>/competition_defense_document.md before export",
    )
    return parser.parse_args()


def _extract_material_date(text: str) -> str:
    match = MATERIAL_DATE_PATTERN.search(text)
    return match.group(1) if match else "未标注"


def _inline_format(text: str) -> str:
    escaped = html.escape(text)
    escaped = CODE_PATTERN.sub(r"<code>\1</code>", escaped)
    return BOLD_PATTERN.sub(r"<strong>\1</strong>", escaped)


def _figure_class(rel_path: str) -> str:
    lowered = rel_path.lower().replace("\\", "/")
    if "/screenshots/" in lowered:
        return "figure-screenshot"
    if lowered.endswith(".svg"):
        return "figure-diagram"
    return "figure-chart"


def _resolve_image_asset(source_dir: Path, rel_path: str) -> Path | None:
    """Resolve image path relative to package dir, repo root, or submission evidence."""
    candidates = [
        source_dir / rel_path,
        ROOT / rel_path,
    ]
    prefix = "competition_submission/defense-package-final/"
    normalized = rel_path.replace("\\", "/")
    if normalized.startswith(prefix):
        candidates.append(source_dir / normalized[len(prefix) :])
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    return None


def _embed_image(source_dir: Path, caption: str, rel_path: str) -> str:
    asset = _resolve_image_asset(source_dir, rel_path)
    if asset is None:
        return (
            f'<figure class="figure-missing"><p>图片缺失：{html.escape(rel_path)}</p></figure>'
        )
    mime, _ = mimetypes.guess_type(str(asset))
    if not mime:
        mime = "application/octet-stream"
    encoded = base64.b64encode(asset.read_bytes()).decode("ascii")
    figure_class = _figure_class(rel_path)
    label = html.escape(caption or asset.stem)
    return (
        f'<figure class="figure-card {figure_class}">'
        f'<div class="figure-frame">'
        f'<img src="data:{mime};base64,{encoded}" alt="{label}" '
        f'data-fullscreen="true" loading="lazy" />'
        f'<span class="zoom-hint">点击放大查看原图</span>'
        f"</div>"
        f'<figcaption>{label}</figcaption>'
        f"</figure>"
    )


def _markdown_to_html_body(text: str, source_dir: Path) -> str:
    lines = text.splitlines()
    out: list[str] = []
    in_code = False
    in_table = False
    in_list = False
    table_rows: list[str] = []
    chapter_open = False

    def close_chapter() -> None:
        nonlocal chapter_open
        if chapter_open:
            out.append("</section>")
            chapter_open = False

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    def flush_table() -> None:
        nonlocal in_table, table_rows
        if not in_table:
            return
        close_list()
        rows_html = []
        for index, row in enumerate(table_rows):
            cells = [cell.strip() for cell in row.strip("|").split("|")]
            if all(TABLE_SEPARATOR_CELL.fullmatch(cell) for cell in cells):
                continue
            if index == 0 and not rows_html:
                rows_html.append(
                    "<tr>"
                    + "".join(f"<th>{_inline_format(cell)}</th>" for cell in cells)
                    + "</tr>"
                )
            else:
                rows_html.append(
                    "<tr>"
                    + "".join(f"<td>{_inline_format(cell)}</td>" for cell in cells)
                    + "</tr>"
                )
        if rows_html:
            out.append('<div class="table-wrap"><table>' + "".join(rows_html) + "</table></div>")
        in_table = False
        table_rows = []

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("```"):
            flush_table()
            close_list()
            if in_code:
                out.append("</pre></div>")
                in_code = False
            else:
                out.append('<div class="code-card"><pre>')
                in_code = True
            continue
        if in_code:
            out.append(html.escape(line))
            continue

        image_match = IMAGE_PATTERN.fullmatch(line.strip())
        if image_match:
            flush_table()
            close_list()
            out.append(_embed_image(source_dir, image_match.group(1), image_match.group(2)))
            continue

        if line.startswith("|"):
            in_table = True
            table_rows.append(line)
            continue
        flush_table()

        if not line.strip():
            close_list()
            continue
        if line.strip() == "---":
            close_list()
            out.append('<hr class="section-divider"/>')
            continue

        if SUBSECTION_LEAD.match(line.strip()):
            close_list()
            out.append(f'<p class="subsection-lead">{_inline_format(line.strip())}</p>')
            continue
        if line.startswith("## "):
            close_list()
            close_chapter()
            chapter_open = True
            anchor = re.sub(r"[^\w\u4e00-\u9fff]+", "-", line[3:]).strip("-").lower()
            out.append(
                f'<section class="content-section" id="{html.escape(anchor)}">'
                f"<h2>{_inline_format(line[3:])}</h2>"
            )
            continue
        if line.startswith("# "):
            close_list()
            out.append(
                f'<header class="doc-hero">'
                f"<h1>{_inline_format(line[2:])}</h1>"
                f"</header>"
            )
            continue
        if line.startswith("- "):
            if not in_list:
                out.append('<ul class="bullet-list">')
                in_list = True
            out.append(f"<li>{_inline_format(line[2:])}</li>")
            continue
        if line.startswith("> "):
            close_list()
            out.append(f'<blockquote class="callout">{_inline_format(line[2:])}</blockquote>')
            continue

        close_list()
        out.append(f"<p>{_inline_format(line)}</p>")

    flush_table()
    close_list()
    if in_code:
        out.append("</pre></div>")
    close_chapter()
    return "\n".join(out)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>nexent-dkm-agent 技术答辩材料</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --paper: #ffffff;
      --ink: #0f172a;
      --muted: #64748b;
      --line: #dbe4f0;
      --brand: #246bfe;
      --brand-soft: #e8f0ff;
      --accent: #0f8f70;
      --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
      --radius: 16px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(36,107,254,0.08), transparent 28%),
        radial-gradient(circle at top right, rgba(15,143,112,0.06), transparent 24%),
        var(--bg);
      line-height: 1.75;
    }}
    .page-shell {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 32px 20px 64px;
    }}
    .doc-hero {{
      background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 45%, #0f8f70 100%);
      color: #fff;
      border-radius: calc(var(--radius) + 4px);
      padding: 36px 40px;
      margin-bottom: 28px;
      box-shadow: var(--shadow);
    }}
    .doc-hero h1 {{
      margin: 0;
      font-size: 2.2rem;
      letter-spacing: 0.02em;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-bottom: 28px;
    }}
    .meta-card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 16px 18px;
      box-shadow: var(--shadow);
    }}
    .meta-card .label {{
      color: var(--muted);
      font-size: 0.82rem;
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .meta-card .value {{
      font-weight: 600;
      font-size: 1rem;
    }}
    .content-section {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-top: 4px solid var(--brand);
      border-radius: var(--radius);
      padding: 28px 30px;
      margin: 22px 0;
      box-shadow: var(--shadow);
      overflow: visible;
    }}
    h2, h3 {{ color: #1e293b; }}
    h2 {{ margin: 0 0 0.8rem; font-size: 1.55rem; }}
    p.subsection-lead {{
      margin: 1.5rem 0 0.5rem;
      padding-top: 0.9rem;
      border-top: 1px dashed var(--line);
      font-size: 1.08rem;
      font-weight: 600;
      color: #1e293b;
    }}
    section.content-section > p.subsection-lead:first-of-type {{
      margin-top: 0.2rem;
      padding-top: 0;
      border-top: none;
    }}
    p {{ margin: 0.8rem 0; }}
    .section-divider {{
      border: none;
      height: 1px;
      background: linear-gradient(90deg, transparent, var(--line), transparent);
      margin: 28px 0;
    }}
    .bullet-list {{
      margin: 0.6rem 0 0.8rem 1.2rem;
      padding: 0;
    }}
    .bullet-list li {{ margin: 0.35rem 0; }}
    .callout {{
      margin: 14px 0;
      padding: 14px 16px;
      border-left: 4px solid var(--brand);
      background: var(--brand-soft);
      border-radius: 12px;
      color: #334155;
    }}
    .table-wrap {{
      overflow-x: auto;
      margin: 16px 0;
      border: 1px solid var(--line);
      border-radius: 12px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.96rem;
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      text-align: left;
    }}
    th {{
      background: #f8fafc;
      color: #334155;
    }}
    tr:last-child td {{ border-bottom: none; }}
    .code-card {{
      margin: 16px 0;
      border-radius: 12px;
      overflow: hidden;
      border: 1px solid #cbd5e1;
      background: #0f172a;
    }}
    .code-card pre {{
      margin: 0;
      padding: 16px 18px;
      color: #e2e8f0;
      overflow: auto;
      font-size: 0.92rem;
      line-height: 1.6;
    }}
    .figure-card {{
      display: block;
      clear: both;
      margin: 22px 0;
      width: 100%;
    }}
    .figure-frame {{
      position: relative;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
    }}
    .figure-card img {{
      display: block;
      width: 100%;
      max-width: 100%;
      height: auto;
      border-radius: 10px;
      cursor: zoom-in;
      background: #fff;
      object-fit: contain;
    }}
    .figure-screenshot img {{
      image-rendering: -webkit-optimize-contrast;
      image-rendering: crisp-edges;
    }}
    .figure-diagram .figure-frame {{
      background: linear-gradient(180deg, #ffffff, #f8fbff);
    }}
    .zoom-hint {{
      position: absolute;
      right: 18px;
      bottom: 18px;
      background: rgba(15, 23, 42, 0.72);
      color: #fff;
      font-size: 12px;
      padding: 6px 10px;
      border-radius: 999px;
      pointer-events: none;
    }}
    figcaption {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.92rem;
      text-align: center;
    }}
    .figure-missing {{
      color: #b91c1c;
      background: #fef2f2;
      border: 1px dashed #fecaca;
      border-radius: 12px;
      padding: 16px;
    }}
    code {{
      background: #eff6ff;
      color: #1d4ed8;
      padding: 0.1rem 0.35rem;
      border-radius: 6px;
      font-size: 0.92em;
    }}
    #lightbox {{
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(15, 23, 42, 0.88);
      z-index: 9999;
      padding: 24px;
      align-items: center;
      justify-content: center;
    }}
    #lightbox.open {{ display: flex; }}
    #lightbox img {{
      max-width: min(96vw, 1600px);
      max-height: 92vh;
      width: auto;
      height: auto;
      border-radius: 12px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.35);
      image-rendering: -webkit-optimize-contrast;
    }}
    #lightbox .close {{
      position: fixed;
      top: 18px;
      right: 22px;
      color: #fff;
      font-size: 28px;
      cursor: pointer;
      user-select: none;
    }}
    @media (max-width: 720px) {{
      .page-shell {{ padding: 18px 12px 40px; }}
      .doc-hero {{ padding: 24px 20px; }}
      .doc-hero h1 {{ font-size: 1.7rem; }}
      .content-section {{ padding: 20px 16px; }}
    }}
  </style>
</head>
<body>
  <div class="page-shell">
    <div class="meta-grid">
      <div class="meta-card"><div class="label">赛题</div><div class="value">数据—知识—洞察智能体与算子</div></div>
      <div class="meta-card"><div class="label">工程</div><div class="value">nexent-dkm-agent</div></div>
      <div class="meta-card"><div class="label">材料日期</div><div class="value">{material_date}</div></div>
      <div class="meta-card"><div class="label">验证环境</div><div class="value">Windows · Python 3.12 · RTX 5070</div></div>
    </div>
    {body}
  </div>
  <div id="lightbox" aria-hidden="true">
    <span class="close" title="关闭">×</span>
    <img alt="放大预览"/>
  </div>
  <script>
    const lightbox = document.getElementById('lightbox');
    const lightboxImg = lightbox.querySelector('img');
    document.querySelectorAll('img[data-fullscreen="true"]').forEach((img) => {{
      img.addEventListener('click', () => {{
        lightboxImg.src = img.src;
        lightbox.classList.add('open');
      }});
    }});
    lightbox.addEventListener('click', () => lightbox.classList.remove('open'));
  </script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    md_path = source / "competition_defense_document.md"
    if args.sync_from:
        from demos.build_defense_pdf_package import _package_markdown

        sync_source = Path(args.sync_from)
        if not sync_source.is_file():
            print(f"Missing sync source: {sync_source}")
            return 1
        md_path.write_text(
            _package_markdown(sync_source.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
    if not md_path.exists():
        print(f"Missing markdown file: {md_path}")
        return 1

    output = Path(args.output) if args.output else source / "competition_defense_document.html"
    markdown = md_path.read_text(encoding="utf-8")
    body = _markdown_to_html_body(markdown, source)
    output.write_text(
        HTML_TEMPLATE.format(
            body=body,
            material_date=html.escape(_extract_material_date(markdown)),
        ),
        encoding="utf-8",
    )
    print({"status": "completed", "html": str(output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
