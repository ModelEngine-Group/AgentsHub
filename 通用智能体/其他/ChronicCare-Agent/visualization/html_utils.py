from __future__ import annotations

from html import escape
from typing import Any, Iterable


def html_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape(title)}</title>
  <style>
    body {{
      margin: 0;
      font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
      background: linear-gradient(180deg, #f6f3eb 0%, #ffffff 100%);
      color: #1f2937;
    }}
    .page {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    .card {{
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
      padding: 24px;
      margin-bottom: 20px;
    }}
    h1, h2, h3 {{ color: #102a43; }}
    .muted {{ color: #52606d; }}
    .metric-value {{
      font-size: 56px;
      line-height: 1;
      font-weight: 700;
      color: #0b7285;
      margin: 12px 0;
    }}
    .badge {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: #e6fffa;
      color: #0f766e;
      font-size: 13px;
      margin-right: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid #e5e7eb;
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f8fafc;
    }}
    ul {{ padding-left: 20px; }}
    a {{ color: #0b7285; }}
    .safety {{
      border-left: 4px solid #f59f00;
      background: #fff9db;
      padding: 12px 14px;
      margin-top: 16px;
      border-radius: 10px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
    }}
  </style>
</head>
<body>
  <div class="page">
    {body}
  </div>
</body>
</html>"""


def render_table(columns: list[str], rows: list[dict[str, Any]], max_rows: int = 100) -> str:
    header = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
    body_rows = []
    for row in rows[:max_rows]:
        cells = "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def render_list(items: Iterable[str]) -> str:
    return "<ul>" + "".join(f"<li>{escape(str(item))}</li>" for item in items) + "</ul>"
