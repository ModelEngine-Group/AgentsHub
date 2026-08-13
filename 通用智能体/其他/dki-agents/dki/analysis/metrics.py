"""标准分析指标集（BI 报告的默认数据源）。"""

from pathlib import Path
from typing import Any, Dict, List

from dki.analysis.database import run_sql
from dki.analysis.nl2sql_sqlite import connect_readonly


def standard_metrics(db_path: Path) -> List[Dict[str, Any]]:
    """返回一组标准指标：[{key,title,columns,rows,chart_type}]。"""
    specs = [
        {
            "key": "dept_visits",
            "title": "各科室就诊量",
            "chart_type": "bar",
            "sql": "SELECT department, COUNT(*) AS n FROM visits GROUP BY department ORDER BY n DESC",
        },
        {
            "key": "top_diseases",
            "title": "就诊量 Top10 疾病",
            "chart_type": "bar",
            "sql": "SELECT disease, COUNT(*) AS n FROM visits GROUP BY disease ORDER BY n DESC LIMIT 10",
        },
        {
            "key": "monthly_trend",
            "title": "月度就诊量趋势",
            "chart_type": "line",
            "sql": ("SELECT strftime('%Y-%m', visit_date) AS m, COUNT(*) AS n "
                    "FROM visits GROUP BY m ORDER BY m"),
        },
        {
            "key": "dept_avg_cost",
            "title": "各科室平均费用（元）",
            "chart_type": "bar",
            "sql": ("SELECT department, ROUND(AVG(cost),2) AS v FROM visits "
                    "GROUP BY department ORDER BY v DESC"),
        },
        {
            "key": "age_group",
            "title": "年龄段就诊分布",
            "chart_type": "pie",
            "sql": ("SELECT CASE WHEN p.age<=18 THEN '0-18岁' WHEN p.age<=40 THEN '19-40岁' "
                    "WHEN p.age<=60 THEN '41-60岁' ELSE '60岁以上' END AS g, COUNT(*) AS n "
                    "FROM visits v JOIN patients p ON v.patient_id=p.patient_id "
                    "GROUP BY g ORDER BY n DESC"),
        },
        {
            "key": "gender_dist",
            "title": "性别就诊分布",
            "chart_type": "pie",
            "sql": ("SELECT p.gender, COUNT(*) AS n FROM visits v "
                    "JOIN patients p ON v.patient_id=p.patient_id GROUP BY p.gender"),
        },
        {
            "key": "top_drugs",
            "title": "处方 Top10 药物",
            "chart_type": "bar",
            "sql": ("SELECT drug_prescribed, COUNT(*) AS n FROM visits "
                    "WHERE drug_prescribed != '' GROUP BY drug_prescribed ORDER BY n DESC LIMIT 10"),
        },
        {
            "key": "kg_core_entities",
            "title": "图谱核心实体（提及次数 Top10）",
            "chart_type": "bar",
            "sql": ("SELECT name, mention_count FROM kg_entities "
                    "ORDER BY mention_count DESC LIMIT 10"),
        },
    ]
    out = []
    for spec in specs:
        try:
            columns, rows = run_sql(spec["sql"], db_path)
            out.append({**{k: spec[k] for k in ("key", "title", "chart_type", "sql")},
                        "columns": columns, "rows": [list(r) for r in rows]})
        except Exception:
            continue
    return out


def age_disease_heatmap(db_path: Path, top_k: int = 8) -> Dict[str, Any]:
    """年龄段 × Top-K 疾病 就诊量热力图数据（ECharts heatmap 形状）。"""
    age_expr = ("CASE WHEN p.age<=18 THEN '0-18' WHEN p.age<=40 THEN '19-40' "
                "WHEN p.age<=60 THEN '41-60' ELSE '60+' END")
    ycats = ["0-18", "19-40", "41-60", "60+"]
    top_sql = (
        "SELECT disease, COUNT(*) n FROM visits GROUP BY disease "
        f"ORDER BY n DESC LIMIT {int(top_k)}"
    )
    try:
        _, drows = run_sql(top_sql, db_path)
        xcats = [r[0] for r in drows]
        cell_sql = (
            f"SELECT {age_expr} ag, v.disease, COUNT(*) n "
            f"FROM visits v JOIN patients p ON v.patient_id=p.patient_id "
            f"WHERE v.disease IN ({','.join(chr(39)+d.replace(chr(39),chr(39)*2)+chr(39) for d in xcats)}) "
            f"GROUP BY ag, v.disease"
        )
        _, rows = run_sql(cell_sql, db_path)
    except Exception:
        return {}
    xi = {d: i for i, d in enumerate(xcats)}
    yi = {a: i for i, a in enumerate(ycats)}
    cell = {(xi[d], yi[a]): 0 for d in xcats for a in ycats}
    for ag, dis, n in rows:
        if dis in xi and ag in yi:
            cell[(xi[dis], yi[ag])] = n
    data = [[x, y, v] for (x, y), v in cell.items()]
    return {
        "xcats": xcats,
        "ycats": ycats,
        "data": data,
        "sql": cell_sql,
        "selection_sql": top_sql,
        "result_rows": len(rows),
    }


def disease_department_sankey(db_path: Path, top_k: int = 12) -> Dict[str, Any]:
    """疾病 → 科室 就诊流向 桑基图数据（Top-K 疾病）。"""
    top_sql = (
        "SELECT disease, COUNT(*) n FROM visits GROUP BY disease "
        f"ORDER BY n DESC LIMIT {int(top_k)}"
    )
    try:
        _, drows = run_sql(top_sql, db_path)
        top = [r[0] for r in drows]
        if not top:
            return {}
        flow_sql = (
            f"SELECT disease, department, COUNT(*) n FROM visits "
            f"WHERE disease IN ({','.join(chr(39)+d.replace(chr(39),chr(39)*2)+chr(39) for d in top)}) "
            f"GROUP BY disease, department HAVING n > 0 ORDER BY n DESC"
        )
        _, rows = run_sql(flow_sql, db_path)
    except Exception:
        return {}
    names = set()
    links = []
    for dis, dept, n in rows:
        src, dst = f"疾病:{dis}", f"科室:{dept}"
        names.add(src); names.add(dst)
        links.append({"source": src, "target": dst, "value": n})
    return {
        "nodes": [{"name": x} for x in sorted(names)],
        "links": links,
        "sql": flow_sql,
        "selection_sql": top_sql,
        "result_rows": len(rows),
    }


def graph_associations(db_path: Path, limit: int = 8) -> List[Dict[str, Any]]:
    """图谱驱动的"疾病—症状/药物/科室"关联洞察（跨表 JOIN 图谱三元组）。"""
    out = []
    queries = {
        "高频疾病的典型症状": (
            "SELECT head disease, GROUP_CONCAT(tail, '、') syms FROM ("
            "  SELECT head, tail FROM kg_triples WHERE relation='HAS_SYMPTOM' "
            "  ORDER BY confidence DESC) GROUP BY head LIMIT ?"),
        "疾病的常用治疗药物": (
            "SELECT head disease, GROUP_CONCAT(tail, '、') drugs FROM ("
            "  SELECT head, tail FROM kg_triples WHERE relation='TREATED_BY_DRUG' "
            "  ORDER BY confidence DESC) GROUP BY head LIMIT ?"),
    }
    conn = connect_readonly(db_path)
    try:
        for title, sql in queries.items():
            try:
                rows = conn.execute(sql, (limit,)).fetchall()
                out.append({
                    "title": title,
                    "rows": [list(r) for r in rows],
                    "sql": sql.replace("?", str(int(limit))),
                    "result_rows": len(rows),
                })
            except Exception:
                continue
    finally:
        conn.close()
    return out


def key_insights(db_path: Path, metrics: List[Dict[str, Any]], overview: Dict[str, Any]) -> List[str]:
    """从指标自动萃取 Top-N 关键洞察句（数据驱动，供报告置顶）。"""
    insights: List[str] = []
    by_key = {m["key"]: m for m in metrics}
    total = overview.get("就诊记录", 0) or 1

    def add(text: str, key: str) -> None:
        metric = by_key[key]
        insights.append(
            f"{text}【证据：Task 1 manifest → insight.db；SQL `{metric['sql']}`；"
            f"结果行数 {len(metric['rows'])}；相关不等于因果。】"
        )

    if "top_diseases" in by_key and by_key["top_diseases"]["rows"]:
        r = by_key["top_diseases"]["rows"]
        top3 = "、".join(f"{x[0]}({x[1]})" for x in r[:3])
        add(
            f"就诊量最高的三种疾病为 {top3}，合计占总就诊 "
            f"{round(100*sum(x[1] for x in r[:3])/total,1)}%。",
            "top_diseases",
        )
    if "dept_visits" in by_key and by_key["dept_visits"]["rows"]:
        r = by_key["dept_visits"]["rows"]
        add(
            f"{r[0][0]} 是就诊量最大的科室（{r[0][1]} 次），"
            f"最小的是 {r[-1][0]}（{r[-1][1]} 次）。",
            "dept_visits",
        )
    if "dept_avg_cost" in by_key and by_key["dept_avg_cost"]["rows"]:
        r = by_key["dept_avg_cost"]["rows"]
        add(
            f"平均费用最高的科室为 {r[0][0]}（{r[0][1]} 元），"
            f"约为最低科室 {r[-1][0]}（{r[-1][1]} 元）的 "
            f"{round(r[0][1]/max(1,r[-1][1]),1)} 倍。",
            "dept_avg_cost",
        )
    if "age_group" in by_key and by_key["age_group"]["rows"]:
        r = by_key["age_group"]["rows"]
        add(
            f"就诊次数最多的年龄组为 {r[0][0]}（{r[0][1]} 次）。",
            "age_group",
        )
    if overview.get("图谱三元组"):
        insights.append(
            f"知识图谱包含 {overview.get('图谱实体')} 个实体、"
            f"{overview.get('图谱三元组')} 条三元组。"
            "【证据：Task 2 knowledge graph → insight.db；"
            "SQL `SELECT COUNT(*) FROM kg_entities` 与 "
            "`SELECT COUNT(*) FROM kg_triples`；结果各 1 行；"
            "抽取关联不构成医学因果或临床建议。】"
        )
    # 可行动洞察（模板化建议，绑定已有统计）
    if "dept_avg_cost" in by_key and by_key["dept_avg_cost"]["rows"]:
        r = by_key["dept_avg_cost"]["rows"]
        add(
            f"【人工复核线索】可比较高费用科室「{r[0][0]}」与"
            f"低费用科室「{r[-1][0]}」的统计口径和路径差异，不据此推断原因。",
            "dept_avg_cost",
        )
    if "top_diseases" in by_key and by_key["top_diseases"]["rows"]:
        top = by_key["top_diseases"]["rows"][0][0]
        add(
            f"【人工复核线索】可对高频疾病「{top}」继续复核复诊与共病分布。",
            "top_diseases",
        )
    if "dept_visits" in by_key and len(by_key["dept_visits"]["rows"]) >= 2:
        r = by_key["dept_visits"]["rows"]
        add(
            f"【人工复核线索】就诊量头部「{r[0][0]}」与尾部"
            f"「{r[-1][0]}」存在描述性差异，可进一步检查样本构成。",
            "dept_visits",
        )
    return insights


def overview_stats(db_path: Path) -> Dict[str, Any]:
    """概览卡片数据。"""
    stats = {}
    for key, sql in {
        "就诊记录": "SELECT COUNT(*) FROM visits",
        "患者数": "SELECT COUNT(*) FROM patients",
        "科室数": "SELECT COUNT(DISTINCT department) FROM visits",
        "疾病种类": "SELECT COUNT(DISTINCT disease) FROM visits",
        "图谱实体": "SELECT COUNT(*) FROM kg_entities",
        "图谱三元组": "SELECT COUNT(*) FROM kg_triples",
    }.items():
        try:
            _, rows = run_sql(sql, db_path)
            stats[key] = rows[0][0] if rows else 0
        except Exception:
            stats[key] = 0
    return stats
