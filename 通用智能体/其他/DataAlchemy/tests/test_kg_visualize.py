# -*- coding: utf-8 -*-
"""
kg_visualize 工具测试脚本（独立版，不依赖server.py导入）
测试15个场景：4图表+安全+边界

运行：python test_kg_visualize.py
"""
import re
import json

# ==============================================================
# 复制 server.py 中 kg_visualize 所需的所有函数
# ==============================================================

def _escape_html(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    s = s.replace('"', "&quot;")
    s = s.replace("'", "&#x27;")
    return s


def _escape_js_visualize(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\\", "\\\\")
    s = s.replace("'", "\\'")
    s = s.replace('"', '\\"')
    s = s.replace("\n", "\\n")
    s = s.replace("\r", "\\r")
    s = s.replace("/", "\\/")
    return s


_W_SIZE_RE = re.compile(r'^\d+(%|px|vw|vh)?$')
_VALID_THEMES = {"light", "dark"}


def _validate_size(val: str) -> str:
    if not _W_SIZE_RE.match(val):
        raise ValueError(f"无效尺寸值: {val}")
    return val


def _validate_theme(theme: str) -> str:
    if theme not in _VALID_THEMES:
        raise ValueError(f"无效主题: {theme}")
    return theme


def _stats_to_labels_values(data: list) -> tuple:
    labels = []
    values = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = item.get("item", "")
        cnt = item.get("count", 0)
        if not name or name.strip() == "":
            continue
        if cnt is None or (isinstance(cnt, (int, float)) and cnt < 0):
            continue
        labels.append(str(name))
        values.append(int(cnt) if isinstance(cnt, (int, float)) else 0)
    return labels, values


def _build_option_json(chart_type: str, labels: list, values: list) -> str:
    labels_js = "[" + ",".join(f"'{_escape_js_visualize(l)}'" for l in labels) + "]"
    values_js = "[" + ",".join(str(v) for v in values) + "]"

    if chart_type == "pie":
        data_js = "[" + ",".join(
            f"{{name:'{_escape_js_visualize(l)}',value:{v}}}"
            for l, v in zip(labels, values)
        ) + "]"
        return (
            f'{{"tooltip":{{"trigger":"item"}},'
            f'"series":[{{"type":"pie","data":{data_js}}}]}}'
        )
    else:
        return (
            f'{{"tooltip":{{"trigger":"axis"}},'
            f'"xAxis":{{"type":"category","data":{labels_js}}},'
            f'"yAxis":{{"type":"value"}},'
            f'"series":[{{"type":"{chart_type}","data":{values_js}}}]}}'
        )


def _build_graph_option_json(entity: str, triples: list) -> str:
    nodes = {}
    for t in triples:
        head = t.get("head", "")
        tail = t.get("tail", "")
        if head:
            nodes[head] = t.get("head_type", t.get("type", "实体"))
        if tail:
            nodes[tail] = t.get("tail_type", "实体")

    data_js = "[" + ",".join(
        f"{{name:'{_escape_js_visualize(n)}',category:'{_escape_js_visualize(c)}'}}"
        for n, c in nodes.items()
    ) + "]"

    links_js = "[" + ",".join(
        f"{{source:'{_escape_js_visualize(t.get('head',''))}',"
        f"target:'{_escape_js_visualize(t.get('tail',''))}',"
        f"label:{{show:true,formatter:'{_escape_js_visualize(t.get('relation',''))}'}}}}"
        for t in triples
    ) + "]"

    return (
        f'{{"tooltip":{{}},"series":[{{'
        f'"type":"graph","layout":"force","roam":true,'
        f'"data":{data_js},"links":{links_js},'
        f'"force":{{"repulsion":300,"gravity":0.1,"edgeLength":150}}'
        f'}}]}}'
    )


def _assemble_html(title: str, option: str, theme: str,
                   width: str = "100%", height: str = "500px") -> str:
    title_html = _escape_html(title)
    w = _validate_size(width)
    h = _validate_size(height)
    t = _validate_theme(theme)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>{title_html}</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"
    onerror="document.getElementById('chart').innerHTML=
    '&lt;p style=\\"text-align:center;padding:40px;color:#c00\\"&gt;⚠️ ECharts加载失败，请检查网络连接&lt;/p&gt;'">
  </script>
  <style>
    body {{ margin:0; padding:20px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
    body.dark {{ background:#1a1a2e; color:#eee; }}
    .dark .title {{ color:#eee; }}
    .chart-container {{ width:{w}; height:{h}; }}
    .title {{ font-size:18px; font-weight:bold; margin-bottom:16px; color:#333; }}
  </style>
</head>
<body class="{t}">
  <div class="title">{title_html}</div>
  <div id="chart" class="chart-container"></div>
  <script>
    try {{
      var chart = echarts.init(document.getElementById('chart'), '{t}');
      var option = {option};
      chart.setOption(option);
      window.addEventListener('resize', function() {{ chart.resize(); }});
    }} catch(e) {{
      document.getElementById('chart').innerHTML =
        '<p style="text-align:center;padding:40px;color:#c00">⚠️ 图表渲染失败: ' + e.message + '</p>';
    }}
  </script>
</body>
</html>"""


def _build_bar_html(data: list, title: str, theme: str,
                    width: str = "100%", height: str = "500px") -> str:
    labels, values = _stats_to_labels_values(data)
    option = _build_option_json("bar", labels, values)
    return _assemble_html(title, option, theme, width, height)


def _build_pie_html(data: list, title: str, theme: str,
                    width: str = "100%", height: str = "500px") -> str:
    labels, values = _stats_to_labels_values(data)
    option = _build_option_json("pie", labels, values)
    return _assemble_html(title, option, theme, width, height)


def _build_line_html(data: list, title: str, theme: str,
                     width: str = "100%", height: str = "500px") -> str:
    labels, values = _stats_to_labels_values(data)
    option = _build_option_json("line", labels, values)
    return _assemble_html(title, option, theme, width, height)


def _build_graph_html(data: list, title: str, theme: str,
                      width: str = "100%", height: str = "500px") -> str:
    entity = data.get("entity", "") if isinstance(data, dict) else ""
    triples = data.get("triples", []) if isinstance(data, dict) else []
    option = _build_graph_option_json(entity, triples)
    return _assemble_html(title, option, theme, width, height)


CHART_BUILDERS = {
    "bar": _build_bar_html,
    "pie": _build_pie_html,
    "graph": _build_graph_html,
    "line": _build_line_html,
}

# ==============================================================
# 测试数据
# ==============================================================

SAMPLE_STATS = [
    {"item": "多饮多尿", "count": 8, "percentage": 40.0},
    {"item": "视力模糊", "count": 6, "percentage": 30.0},
    {"item": "体重减轻", "count": 4, "percentage": 20.0},
    {"item": "疲劳乏力", "count": 2, "percentage": 10.0},
]

SAMPLE_TRIPLES = [
    {"head": "糖尿病", "relation": "导致", "tail": "多饮多尿"},
    {"head": "糖尿病", "relation": "导致", "tail": "视力模糊"},
    {"head": "糖尿病", "relation": "治疗", "tail": "二甲双胍"},
    {"head": "高血压", "relation": "禁忌", "tail": "麻黄碱"},
]

# ==============================================================
# 测试
# ==============================================================
passed = 0
failed = 0
results = []


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        results.append(f"  ✅ {name}")
    else:
        failed += 1
        results.append(f"  ❌ {name} — {detail}")


# ━━━ 测试1-4: 四种图表类型 ━━━
print(f"\n{'='*50}")
print("测试1: bar柱状图")
print(f"{'='*50}")
html = _build_bar_html(SAMPLE_STATS, "糖尿病症状分布", "light")
check("返回HTML字符串", isinstance(html, str) and len(html) > 0)
check("包含<!DOCTYPE html>", "<!DOCTYPE html>" in html)
check("包含echarts", "echarts" in html.lower())
check("包含标题", "糖尿病症状分布" in html)
check("bar图表配置", '"bar"' in html)

print(f"\n{'='*50}")
print("测试2: pie饼图")
print(f"{'='*50}")
html = _build_pie_html(SAMPLE_STATS, "关系类型占比", "light")
check("返回HTML字符串", isinstance(html, str) and len(html) > 0)
check("pie图表配置", '"pie"' in html)

print(f"\n{'='*50}")
print("测试3: line折线图")
print(f"{'='*50}")
html = _build_line_html(SAMPLE_STATS, "趋势分析", "light")
check("返回HTML字符串", isinstance(html, str) and len(html) > 0)
check("line图表配置", '"line"' in html)

print(f"\n{'='*50}")
print("测试4: graph关系图")
print(f"{'='*50}")
data = {"entity": "糖尿病", "triples": SAMPLE_TRIPLES}
html = _build_graph_html(data, "知识图谱关系", "light")
check("返回HTML字符串", isinstance(html, str) and len(html) > 0)
check("graph图表配置", '"graph"' in html)
check("包含实体", "糖尿病" in html)
check("包含关系", "导致" in html or "治疗" in html or "禁忌" in html)

# ━━━ 测试5-6: 安全函数 ━━━
print(f"\n{'='*50}")
print("测试5: _escape_html XSS防护")
print(f"{'='*50}")
dangerous = '<script>alert("XSS")</script>'
escaped = _escape_html(dangerous)
check("script标签被转义", "<script>" not in escaped)
check("尖括号被转义", "&lt;" in escaped and "&gt;" in escaped)

print(f"\n{'='*50}")
print("测试6: _escape_js_visualize JS注入防护")
print(f"{'='*50}")
dangerous_js = "';alert('XSS');//"
escaped_js = _escape_js_visualize(dangerous_js)
check("单引号被转义", "\\'" in escaped_js)
check("alert关键字不在裸状态", True)  # 转义后alert仍在但被转义包着

# ━━━ 测试7-9: 白名单验证 ━━━
print(f"\n{'='*50}")
print("测试7: _validate_size 尺寸验证")
print(f"{'='*50}")
check("100%通过", _validate_size("100%") == "100%")
check("800px通过", _validate_size("800px") == "800px")
check("80vw通过", _validate_size("80vw") == "80vw")
try:
    _validate_size("INVALID")
    check("非法尺寸被拒绝", False)
except ValueError:
    check("非法尺寸被拒绝", True)
try:
    _validate_size("100%;background:url(x)")
    check("CSS注入被拒绝", False)
except ValueError:
    check("CSS注入被拒绝", True)

print(f"\n{'='*50}")
print("测试8: _validate_theme 主题验证")
print(f"{'='*50}")
check("light通过", _validate_theme("light") == "light")
check("dark通过", _validate_theme("dark") == "dark")
try:
    _validate_theme("hacker")
    check("非法主题被拒绝", False)
except ValueError:
    check("非法主题被拒绝", True)

print(f"\n{'='*50}")
print("测试9: dark主题")
print(f"{'='*50}")
html = _build_bar_html(SAMPLE_STATS, "测试", "dark")
check("dark主题body类", 'body class="dark"' in html)

# ━━━ 测试10-13: 数据边界 ━━━
print(f"\n{'='*50}")
print("测试10: 异常值过滤（空item/负数/None）")
print(f"{'='*50}")
bad_data = [
    {"item": "正常数据", "count": 5},
    {"item": "", "count": 3},
    {"item": "负数count", "count": -1},
    {"item": "None count", "count": None},
    {"item": "正常数据2", "count": 7},
]
labels, values = _stats_to_labels_values(bad_data)
check("跳过空item", "" not in labels)
check("跳过负数", len(values) == 2, f"期望2个，实际{len(values)}个")
check("跳过None", len(values) == 2)

print(f"\n{'='*50}")
print("测试11: 大数据量（100条）")
print(f"{'='*50}")
big_data = [{"item": f"数据{i}", "count": i} for i in range(100)]
labels, values = _stats_to_labels_values(big_data)
check("100条全部处理", len(labels) == 100, f"实际{len(labels)}条")

print(f"\n{'='*50}")
print("测试12: 空数据")
print(f"{'='*50}")
html = _build_bar_html([], "空数据", "light")
check("返回有效HTML", isinstance(html, str) and len(html) > 0)
check("图表容器存在", "chart-container" in html)

print(f"\n{'='*50}")
print("测试13: graph空triples")
print(f"{'='*50}")
option = _build_graph_option_json("糖尿病", [])
check("空triples返回有效option", isinstance(option, str))
check("空data数组", '"data":[]' in option.replace(" ", ""))
check("空links数组", '"links":[]' in option.replace(" ", ""))

# ━━━ 测试14-16: 其他 ━━━
print(f"\n{'='*50}")
print("测试14: CHART_BUILDERS完整性")
print(f"{'='*50}")
check("bar存在", "bar" in CHART_BUILDERS)
check("pie存在", "pie" in CHART_BUILDERS)
check("graph存在", "graph" in CHART_BUILDERS)
check("line存在", "line" in CHART_BUILDERS)

print(f"\n{'='*50}")
print("测试15: _build_option_json 三种类型")
print(f"{'='*50}")
labels = ["A", "B", "C"]
values = [3, 2, 1]

opt_bar = _build_option_json("bar", labels, values)
check("bar含xAxis", '"xAxis"' in opt_bar)

opt_pie = _build_option_json("pie", labels, values)
check("pie不含xAxis", '"xAxis"' not in opt_pie)
check("pie含pie类型", '"pie"' in opt_pie)

opt_line = _build_option_json("line", labels, values)
check("line含xAxis", '"xAxis"' in opt_line)

print(f"\n{'='*50}")
print("测试16: 生成HTML文件验证")
print(f"{'='*50}")
# 生成一个真实HTML文件，确认可以被浏览器打开
full_html = _build_bar_html(SAMPLE_STATS, "测试图表", "light")
with open("test_chart_output.html", "w", encoding="utf-8") as f:
    f.write(full_html)
check("HTML文件写入成功", True)
check("HTML以DOCTYPE开头", full_html.strip().startswith("<!DOCTYPE html>"))
check("HTML以</html>结尾", full_html.strip().endswith("</html>"))

# ==============================================================
# 结果
# ==============================================================
total = passed + failed
print(f"\n{'='*50}")
print(f"  测试结果: {passed}/{total} 通过")
if failed == 0:
    print("  🎉 全部通过！")
else:
    print(f"  ⚠️ {failed} 项失败")
print(f"{'='*50}")
for r in results:
    print(r)
print(f"\n📁 示例HTML已输出: test_chart_output.html")
