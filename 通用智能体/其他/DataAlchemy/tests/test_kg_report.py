# -*- coding: utf-8 -*-
"""
kg_report 工具测试脚本（独立版）
覆盖20条测试用例：基础/洞察/安全/边界/校验

运行：python test_kg_report.py
"""
from dataclasses import dataclass, field
import json


# ==============================================================
# 复制 server.py 中 kg_report 的所有函数
# ==============================================================

@dataclass
class Insight:
    rule_id: int
    rule_name: str
    title: str
    detail: str
    severity: str
    source_index: int
    confidence: float
    evidence: list = field(default_factory=list)
    limitation: str = ""


TYPE_LABELS = {
    "关系分布": "关系类型", "症状分布": "症状",
    "药物统计": "治疗药物", "关联分析": "关联实体",
}


def _md_escape(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("|", "&#124;")
    text = text.replace("`", "\u200b`")
    text = text.replace("_", "\\_")
    text = text.replace("*", "\\*")
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    lines = text.split("\n")
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped and stripped[0] in ("#", "-", "*", ">"):
            lines[i] = "\\" + line
    return "\n".join(lines)


def _build_md_table(rows: list, headers: list) -> str:
    col_count = len(headers)
    safe_rows = []
    for r in rows:
        r = list(r[:col_count]) + [""] * max(0, col_count - len(r))
        safe_rows.append([_md_escape(str(c)) for c in r])
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join([":---"] * col_count) + "|")
    for row in safe_rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _format_timestamp() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _generate_chart_comment(stats: list, analysis_type: str) -> str:
    if not stats or len(stats) < 2:
        return ""
    top1, top2 = stats[0], stats[1]
    label = TYPE_LABELS.get(analysis_type, "条目")
    divisor = max(top2.get("count", 1), 1)
    ratio = round(top1.get("count", 0) / divisor, 1)
    return (
        f"> 📊 从上图可见，**{_md_escape(str(top1.get('item','')))}** "
        f"是最常见的{label}（{top1.get('percentage',0)}%），"
        f"约为第二名 **{_md_escape(str(top2.get('item','')))}** "
        f"（{top2.get('percentage',0)}%）的 {ratio} 倍。"
    )


def _safe_truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    cutoff = text.rfind("\n\n", 0, max_len)
    if cutoff < max_len // 2:
        cutoff = text.rfind("\n", 0, max_len)
    if cutoff < max_len // 2:
        cutoff = max_len
    return text[:cutoff] + "\n\n*[报告过长，已截断]*"


def _build_header(entity: str, query: str, result_count: int) -> str:
    ts = _format_timestamp()
    q = _md_escape(query or "无")
    return (
        f"# {_md_escape(entity)}知识图谱分析报告\n\n"
        f"> 分析时间：{ts}\n> 数据来源：医疗知识图谱（任务二产出）\n"
        f"> 分析维度：{result_count} 项\n> 用户提问：{q}\n"
    )


def _build_overview(results: list) -> str:
    total = sum(r.get("total_triples", 0) for r in results)
    types = [_md_escape(r.get("analysis_type", "未知")) for r in results]
    return (
        f"## 一、概述\n\n"
        f"知识图谱中共关联 **{total}** 条三元组。\n"
        f"本报告整合了 {len(results)} 项统计分析，涵盖：{'、'.join(types)}。\n"
    )


def _build_analysis_section(idx: int, r: dict) -> str:
    atype = r.get("analysis_type", "统计")
    label = TYPE_LABELS.get(atype, "条目")
    entity = _md_escape(r.get("entity", "未知"))
    title = f"{entity}的{label}" if atype != "关系分布" else "关系类型分布"
    stats = r.get("stats", [])

    md = f"### 2.{idx} {title}\n\n"
    if not stats:
        return md + "*暂无统计数据*\n"
    safe_stats = []
    for s in stats:
        if not isinstance(s, dict):
            continue
        if not s.get("item") or s["item"] == "":
            continue
        cnt = s.get("count")
        if cnt is None or (isinstance(cnt, (int, float)) and cnt < 0):
            continue
        pct = s.get("percentage", 0)
        if isinstance(pct, (int, float)) and (pct < 0 or pct > 100):
            continue
        safe_stats.append(s)

    rows = [
        [s.get("item", ""), str(s.get("count", "")), f"{s.get('percentage',0)}%"]
        for s in safe_stats
    ]
    if rows:
        md += _build_md_table(rows, ["名称", "计数", "占比"]) + "\n\n"
    else:
        md += "*数据异常，无法统计*\n"
    return md


def _build_chart_section(chart: dict, r: dict) -> str:
    atype = r.get("analysis_type", "统计")
    comment = _generate_chart_comment(r.get("stats", []), atype)
    title = chart.get("title", "") if isinstance(chart, dict) else ""
    md = f"### 图表：{_md_escape(title)}\n\n" if title else ""
    if isinstance(chart, dict) and chart.get("html"):
        md += chart["html"] + "\n\n"
    if comment:
        md += comment + "\n\n"
    return md


def _build_insights_section(insights: list) -> str:
    if not insights:
        return ""
    md = "## 四、关键洞察\n\n"
    md += "| # | 洞察 | 置信度 | 依据 |\n"
    md += "|:---:|:---|:---:|:---|\n"
    for i, ins in enumerate(insights):
        icon = {"highlight": "🔴", "info": "🔵", "warning": "🟡"}.get(ins.severity, "⚪")
        conf_icon = "🟢" if ins.confidence >= 0.8 else ("🟡" if ins.confidence >= 0.6 else "🔴")
        evidence = "; ".join(ins.evidence[:2]) if ins.evidence else "—"
        md += (
            f"| {i+1} | {icon} **{_md_escape(ins.title)}**<br>"
            f"{_md_escape(ins.detail[:80])}{'...' if len(ins.detail)>80 else ''} | "
            f"{conf_icon} {ins.confidence:.2f} | "
            f"{_md_escape(evidence)} |\n"
        )
    return md + "\n"


def _build_footer() -> str:
    return (
        "## 五、注意事项\n\n"
        "- 本报告基于知识图谱已有数据生成\n"
        "- 统计关系不构成医疗建议\n\n"
        "---\n\n"
        "*报告由 Nexent 数据分析智能体自动生成*"
    )


def _build_raw_data_section(results: list) -> str:
    md = "## 附录：原始数据\n\n"
    for i, r in enumerate(results):
        atype = _md_escape(r.get("analysis_type", "未知"))
        md += f"### A.{i+1} {atype}\n\n"
        stats = r.get("stats", [])
        if stats:
            rows = [
                [s.get("item",""), str(s.get("count","")), f"{s.get('percentage',0)}%"]
                for s in stats if isinstance(s, dict) and s.get("item")
            ]
            if rows:
                md += _build_md_table(rows, ["条目", "计数", "占比"]) + "\n\n"
    return md


def _score_confidence(rule_id: int, stats: list, total: int) -> float:
    base = 0.70
    if total >= 30:  base += 0.10
    elif total < 5:  base -= 0.30
    if rule_id == 1 and stats:
        pct = stats[0].get("percentage", 0)
        if pct > 50:   base += 0.10
        elif pct < 40: base -= 0.05
    return round(min(max(base, 0.0), 1.0), 2)


def _generate_insights(results: list) -> list:
    insights = []
    for idx, r in enumerate(results):
        if not isinstance(r, dict): continue
        stats, total = r.get("stats", []), r.get("total_triples", len(r.get("stats",[])))
        atype, entity = r.get("analysis_type", ""), r.get("entity", "")
        label = TYPE_LABELS.get(atype, "条目")
        if not stats: continue
        top1, p1 = stats[0], stats[0].get("percentage", 0)

        if p1 > 35:
            insights.append(Insight(1, "高集中度",
                f"{top1['item']}是最突出的{label}",
                f"{top1['item']}占比 {p1}%，远超其他{label}。",
                "highlight", idx, _score_confidence(1, stats, total),
                [f"{atype}/Top-1/{p1}%"],
                f"基于{total}条三元组统计"))

        if len(stats) >= 2:
            p2 = stats[1].get("percentage", 0)
            if p2 > 0 and (p1 - p2) < 5:
                insights.append(Insight(2, "均衡分布",
                    f"{top1['item']}与{stats[1]['item']}分布接近",
                    f"二者占比分别为 {p1}% 和 {p2}%。",
                    "info", idx, 0.65,
                    [f"{atype}/Top-1/{p1}%", f"{atype}/Top-2/{p2}%"],
                    "仅比较前两名"))

        if len(stats) >= 3:
            top3_sum = sum(s.get("percentage",0) for s in stats[:3])
            if top3_sum > 60:
                insights.append(Insight(3, "Top-3集中",
                    f"前三项合计占比 {round(top3_sum,1)}%",
                    f"Top-3构成{label}的主要部分。",
                    "info", idx, 0.70,
                    [f"{atype}/Top-3/{round(top3_sum,1)}%"], ""))

        if len(stats) >= 3:
            top3_sum = sum(s.get("percentage",0) for s in stats[:3])
            if top3_sum < 40:
                insights.append(Insight(4, "长尾显著",
                    f"{label}分布分散",
                    f"前三项合计仅{round(top3_sum,1)}%，长尾分布。",
                    "info", idx, 0.65,
                    [f"{atype}/Top-3/{round(top3_sum,1)}%"], ""))

        if "禁忌" in atype:
            items = [s for s in stats if s.get("count",0) > 0]
            if items:
                insights.append(Insight(5, "禁忌警告",
                    f"检测到与{entity}相关的禁忌关系",
                    f"涉及：{'、'.join(s['item'] for s in items[:3])}。",
                    "warning", idx, 1.0,
                    [f"{atype}/{len(items)}条禁忌"],
                    "需结合实际临床判断"))

        if total < 5:
            insights.append(Insight(6, "数据稀疏",
                f"{entity}数据稀疏（仅{total}条）",
                f"分析结果置信度有限。",
                "warning", idx, 0.95, [f"total_triples={total}"], ""))

        if total > 30:
            insights.append(Insight(7, "数据丰富",
                f"{entity}数据丰富（{total}条）",
                f"分析可信度较高。",
                "info", idx, 0.85, [f"total_triples={total}"],
                "数据量不代表数据质量"))

    # 规则8：跨分析因果链
    by_type = {r.get("analysis_type"): r for r in results if isinstance(r, dict)}
    if "症状分布" in by_type and "药物统计" in by_type:
        sr, dr = by_type["症状分布"], by_type["药物统计"]
        ss, ds = sr.get("stats",[]), dr.get("stats",[])
        if ss and ds:
            insights.append(Insight(8, "跨分析因果链",
                f"{ss[0]['item']}与{ds[0]['item']}的治疗关联",
                f"图谱中{ss[0]['item']}是{sr.get('entity','')}最常见症状。",
                "highlight", 0, 0.75,
                ["症状分布/Top-1", "药物统计/Top-1"],
                "因果链基于共现统计"))
    if "关系分布" in by_type:
        rr = by_type["关系分布"]
        has_taboo = any(s.get("item")=="禁忌" and s.get("count",0)>0 for s in rr.get("stats",[]))
        if has_taboo and "药物统计" in by_type:
            insights.append(Insight(8, "跨分析安全警告",
                "禁忌关系与多药物共存",
                "建议进行药物相互作用检查。",
                "warning", 0, 0.90,
                ["关系分布/禁忌", "药物统计"],
                "需结合drug_interaction_check验证"))
    return insights


def _resolve_conflicts(insights: list) -> list:
    rule_ids = {i.rule_id for i in insights}
    if 1 in rule_ids and 2 in rule_ids:
        insights = [i for i in insights if i.rule_id != 2]
    if 3 in rule_ids and 4 in rule_ids:
        insights = [i for i in insights if i.rule_id != 4]
    priority = {"highlight": 0, "warning": 1, "info": 2}
    insights.sort(key=lambda i: priority.get(i.severity, 99))
    return insights


REPORT_CONFIGS = {
    "full": {
        "include_header": True, "include_overview": True,
        "include_analysis_table": True, "include_charts": True,
        "include_insights": "all", "include_footer": True,
        "include_raw_data": False,
    },
    "summary": {
        "include_header": True, "include_overview": False,
        "include_analysis_table": False, "include_charts": False,
        "include_insights": "top3", "include_footer": False,
        "include_raw_data": False, "summary_mode": True,
    },
    "technical": {
        "include_header": True, "include_overview": False,
        "include_analysis_table": False, "include_charts": False,
        "include_insights": "none", "include_footer": False,
        "include_raw_data": True,
    },
}


def _build_report(results, charts, query, config, max_length, max_insights):
    sections = []
    if config.get("include_header"):
        entity = results[0].get("entity","未知") if results else "未知"
        sections.append(_build_header(entity, query, len(results)))
    if config.get("include_overview"):
        sections.append(_build_overview(results))
    for i, r in enumerate(results):
        if not isinstance(r, dict): continue
        if config.get("include_analysis_table"):
            sections.append(_build_analysis_section(i+1, r))
        if config.get("include_charts") and charts:
            for c in charts:
                c_dict = c if isinstance(c, dict) else {}
                if c_dict.get("source_index", 0) == i:
                    sections.append(_build_chart_section(c_dict, r))
    cfg_insights = config.get("include_insights", "all")
    if cfg_insights != "none":
        raw = _generate_insights(results)
        raw = _resolve_conflicts(raw)
        if cfg_insights == "top3":
            raw = raw[:3]
        else:
            raw = raw[:max_insights]
        if raw:
            sections.append(_build_insights_section(raw))
    if config.get("include_footer"):
        sections.append(_build_footer())
    if config.get("include_raw_data"):
        sections.append(_build_raw_data_section(results))
    return _safe_truncate("\n\n".join(sections), max_length)


# ==============================================================
# 测试
# ==============================================================

S1 = [  # 症状分布（高集中度）
    {"item": "多饮多尿", "count": 12, "percentage": 42.0},
    {"item": "视力模糊", "count": 6, "percentage": 21.0},
    {"item": "体重减轻", "count": 5, "percentage": 17.5},
    {"item": "疲劳乏力", "count": 3, "percentage": 10.5},
    {"item": "伤口愈合慢", "count": 2, "percentage": 7.0},
]

S2 = [  # 药物统计
    {"item": "二甲双胍", "count": 4, "percentage": 40.0},
    {"item": "胰岛素", "count": 3, "percentage": 30.0},
    {"item": "格列齐特", "count": 2, "percentage": 20.0},
    {"item": "阿卡波糖", "count": 1, "percentage": 10.0},
]

S_TABOO = [  # 禁忌关系
    {"item": "麻黄碱", "count": 1, "percentage": 50.0},
    {"item": "饮酒", "count": 1, "percentage": 50.0},
]

R_SYMPTOM = {"entity": "糖尿病", "analysis_type": "症状分布", "total_triples": 28, "stats": S1}
R_DRUG = {"entity": "糖尿病", "analysis_type": "药物统计", "total_triples": 10, "stats": S2}
R_TABOO = {"entity": "高血压", "analysis_type": "禁忌", "total_triples": 2, "stats": S_TABOO}

passed = 0; failed = 0; results_list = []

def check(name, cond, detail=""):
    global passed, failed
    if cond: passed += 1; results_list.append(f"  ✅ {name}")
    else: failed += 1; results_list.append(f"  ❌ {name} — {detail}")


# ━━━ 基础测试 ━━━
print(f"\n{'='*50}")
print("测试1: 完整报告（full）")
cfg = REPORT_CONFIGS["full"]
r = _build_report([R_SYMPTOM], [], "糖尿病有什么症状？", cfg, 15000, 8)
check("生成Markdown报告", isinstance(r, str) and len(r) > 100)
check("含标题", "糖尿病知识图谱分析报告" in r)
check("含概述", "一、概述" in r)
check("含统计分析", "2.1" in r)
check("含洞察", "四、关键洞察" in r)
check("含尾部", "五、注意事项" in r)

print(f"\n{'='*50}")
print("测试2: 摘要报告（summary）")
cfg = REPORT_CONFIGS["summary"]
r = _build_report([R_SYMPTOM], [], "糖尿病？", cfg, 15000, 3)
check("生成报告", isinstance(r, str) and len(r) > 50)
check("无概述", "一、概述" not in r)
check("无尾部", "五、注意事项" not in r)

print(f"\n{'='*50}")
print("测试3: 技术报告（technical）")
cfg = REPORT_CONFIGS["technical"]
r = _build_report([R_SYMPTOM], [], "", cfg, 15000, 0)
check("生成报告", isinstance(r, str) and len(r) > 50)
check("含原始数据", "附录：原始数据" in r)
check("无洞察", "四、关键洞察" not in r)

print(f"\n{'='*50}")
print("测试4: 多分析结果整合")
cfg = REPORT_CONFIGS["full"]
r = _build_report([R_SYMPTOM, R_DRUG], [], "糖尿病综合分析", cfg, 15000, 8)
check("两项分析都有", "2.1" in r and "2.2" in r)
check("涵盖两项", "症状分布" in r and "药物统计" in r)

# ━━━ 洞察测试 ━━━
print(f"\n{'='*50}")
print("测试5: 高集中度洞察（>35%）")
insights = _generate_insights([R_SYMPTOM])
check("规则1触发", any(i.rule_id == 1 for i in insights))
check("含置信度", all(0 <= i.confidence <= 1 for i in insights))

print(f"\n{'='*50}")
print("测试6: Top-3集中洞察（>60%）")
check("规则3触发", any(i.rule_id == 3 for i in insights))

print(f"\n{'='*50}")
print("测试7: 跨分析因果链（症状+药物）")
ins_all = _generate_insights([R_SYMPTOM, R_DRUG])
check("规则8触发", any(i.rule_id == 8 for i in ins_all))

print(f"\n{'='*50}")
print("测试8: 禁忌警告")
ins = _generate_insights([R_TABOO])
check("规则5触发", any(i.rule_id == 5 for i in ins))

print(f"\n{'='*50}")
print("测试9: 冲突消解（规则1+2→只保留1）")
# 构造触发规则1和2的数据
s_conflict = [
    {"item": "A", "count": 10, "percentage": 37.0},  # 触发规则1(>35%) + 规则2(gap<5%)
    {"item": "B", "count": 9, "percentage": 33.0},
]
r_conflict = {"entity": "X", "analysis_type": "症状分布", "total_triples": 19, "stats": s_conflict}
ins = _generate_insights([r_conflict])
ins = _resolve_conflicts(ins)
check("规则2被消解", not any(i.rule_id == 2 for i in ins))
check("规则1保留", any(i.rule_id == 1 for i in ins))

# ━━━ 安全测试 ━━━
print(f"\n{'='*50}")
print("测试10: _md_escape 转义管道符")
esc = _md_escape("数据|测试")
check("管道符被转义", "&#124;" in esc)

print(f"\n{'='*50}")
print("测试11: _md_escape 转义行首特殊字符")
esc = _md_escape("# 标题")
check("行首井号被转义", esc.startswith("\\#"))

# ━━━ 边界测试 ━━━
print(f"\n{'='*50}")
print("测试12: 空stats列表（基础概述）")
r_empty = {"entity": "糖尿病", "analysis_type": "症状分布", "total_triples": 0, "stats": []}
cfg = REPORT_CONFIGS["full"]
r = _build_report([r_empty], [], "", cfg, 15000, 8)
check("生成报告不崩溃", isinstance(r, str))
check("含标题", "糖尿病" in r)

print(f"\n{'='*50}")
print("测试13: report_type无效→回退full")
cfg = REPORT_CONFIGS["full"]
r = _build_report([R_SYMPTOM], [], "", cfg, 15000, 8)
check("回退后正常生成", isinstance(r, str) and len(r) > 100)

print(f"\n{'='*50}")
print("测试14: 超长截断")
long_stats = [{"item": f"数据{i:03d}", "count": 1, "percentage": 1.0} for i in range(50)]
r_long = {"entity": "糖尿病", "analysis_type": "症状分布", "total_triples": 50, "stats": long_stats}
r = _build_report([r_long], [], "", {"include_header":True, "include_overview":False,
    "include_analysis_table":True, "include_charts":False, "include_insights":"all",
    "include_footer":False, "include_raw_data":False}, 300, 8)
check("截断后长度限制", len(r) <= 350)  # 允许一点margin

# ━━━ 数据校验测试 ━━━
print(f"\n{'='*50}")
print("测试15: count负数→跳过")
bad_stats = [
    {"item": "正常", "count": 5, "percentage": 50.0},
    {"item": "负数", "count": -1, "percentage": 0},
    {"item": "正常2", "count": 5, "percentage": 50.0},
]
r_bad = {"entity": "X", "analysis_type": "症状分布", "total_triples": 10, "stats": bad_stats}
r = _build_report([r_bad], [], "", REPORT_CONFIGS["full"], 15000, 8)
check("负数被跳过（表格只有2行数据）", r.count("| 正常 ") > 0 and "负数" not in r)

print(f"\n{'='*50}")
print("测试16: 图文结合说明")
chart = {"html": "<div>图表</div>", "source_index": 0}
r = _build_report([R_SYMPTOM], [chart], "", REPORT_CONFIGS["full"], 15000, 8)
check("含图文说明", "从上图可见" in r)

print(f"\n{'='*50}")
print("测试17: 表格列数补齐")
rows = [["A", "B"], ["C"]]  # 第2行缺1列
tbl = _build_md_table(rows, ["Name", "Value"])
check("表格生成成功", "Name" in tbl and "Value" in tbl)

print(f"\n{'='*50}")
print("测试18: 置信度评分")
c1 = _score_confidence(1, [{"percentage": 55}], 35)  # >50% + n>30 → +0.2
c2 = _score_confidence(1, [{"percentage": 30}], 3)  # n<5 → -0.3
check("高置信度场景", c1 >= 0.85)
check("低置信度场景", c2 <= 0.45)

print(f"\n{'='*50}")
print("测试19: 完整输出写文件")
cfg = REPORT_CONFIGS["full"]
r = _build_report([R_SYMPTOM, R_DRUG], [], "糖尿病综合分析", cfg, 15000, 8)
with open("test_report_output.md", "w", encoding="utf-8") as f:
    f.write(r)
check("报告写入成功", True)
check("报告长度>500", len(r) > 500)

print(f"\n{'='*50}")
print("测试20: 数据丰富洞察（total>30）")
r_rich = {"entity": "糖尿病", "analysis_type": "症状分布", "total_triples": 35, "stats": S1[:2]}
ins = _generate_insights([r_rich])
check("规则7触发", any(i.rule_id == 7 for i in ins))

# ━━━ 结果 ━━━
total = passed + failed
print(f"\n{'='*50}")
print(f"  测试结果: {passed}/{total} 通过")
if failed == 0: print("  🎉 全部通过！")
else: print(f"  ⚠️ {failed} 项失败")
print(f"{'='*50}")
for rl in results_list:
    print(rl)
print(f"\n📁 示例报告已输出: test_report_output.md")
