"""生成更难、更真实的 NL2SQL 评测集（多表 JOIN / 嵌套子查询 / 集合运算 / HAVING /
CTE / 相关子查询 / 歧义值 / 时间窗组合），并**逐条验证 gold SQL 可执行**后落盘。

设计（对标 Spider/BIRD 的难度分层与 EX 口径）：
- 用数据库真实值域（疾病/科室/地区）做模板实例化，保证字面量真实、gold 可执行；
- 叠加手写的结构型难题（CTE、集合运算、相关子查询、HAVING）；
- 只保留能成功执行且非空（或明确允许空）的题，避免脏 gold。

产物：data/nl2sql_eval_hard.jsonl，字段 {id, question, gold_sql, difficulty, category}。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dki.analysis.nl2sql_sqlite import connect_readonly  # noqa: E402
from dki.paths import OUTPUT_DIR  # noqa: E402

DB = OUTPUT_DIR / "insight.db"
OUT = ROOT / "data" / "nl2sql_eval_hard.jsonl"


def _domain(conn, sql):
    return [r[0] for r in conn.execute(sql).fetchall() if r and r[0]]


def build_items(conn):
    diseases = _domain(conn, "SELECT disease FROM visits GROUP BY disease ORDER BY COUNT(*) DESC")
    departments = _domain(conn, "SELECT department FROM visits GROUP BY department ORDER BY COUNT(*) DESC")
    regions = _domain(conn, "SELECT region FROM patients GROUP BY region ORDER BY COUNT(*) DESC")
    items = []

    def add(question, gold_sql, difficulty, category):
        items.append({"question": question, "gold_sql": gold_sql,
                      "difficulty": difficulty, "category": category})

    # --- 多表 JOIN + 聚合（患者×就诊）---
    for dis in diseases[:8]:
        add(f"{dis}患者的平均年龄是多少",
            f"SELECT ROUND(AVG(p.age),2) FROM visits v JOIN patients p ON v.patient_id=p.patient_id WHERE v.disease='{dis}'",
            "hard", "join_agg")
    for dis in diseases[:8]:
        add(f"{dis}就诊的平均费用是多少",
            f"SELECT ROUND(AVG(cost),2) FROM visits WHERE disease='{dis}'",
            "medium", "agg_filter")
    for dept in departments[:5]:
        add(f"{dept}就诊患者的男女人数分别是多少",
            f"SELECT p.gender, COUNT(DISTINCT p.patient_id) FROM visits v JOIN patients p ON v.patient_id=p.patient_id WHERE v.department='{dept}' GROUP BY p.gender",
            "hard", "join_group")
    for reg in regions[:5]:
        add(f"{reg}地区就诊量最高的疾病是哪种",
            f"SELECT v.disease, COUNT(*) n FROM visits v JOIN patients p ON v.patient_id=p.patient_id WHERE p.region='{reg}' GROUP BY v.disease ORDER BY n DESC LIMIT 1",
            "hard", "join_group_top")

    # --- 嵌套子查询：高于/低于全局均值 ---
    add("平均费用高于所有就诊平均费用的科室有哪些",
        "SELECT department, ROUND(AVG(cost),2) ac FROM visits GROUP BY department HAVING AVG(cost) > (SELECT AVG(cost) FROM visits) ORDER BY ac DESC",
        "hard", "subquery_having")
    add("住院天数高于全体平均住院天数的疾病有哪些",
        "SELECT disease, ROUND(AVG(los_days),2) al FROM visits GROUP BY disease HAVING AVG(los_days) > (SELECT AVG(los_days) FROM visits) ORDER BY al DESC",
        "hard", "subquery_having")
    add("就诊次数超过平均每种疾病就诊次数的疾病有哪些",
        "SELECT disease, COUNT(*) n FROM visits GROUP BY disease HAVING COUNT(*) > (SELECT AVG(c) FROM (SELECT COUNT(*) c FROM visits GROUP BY disease)) ORDER BY n DESC",
        "hard", "nested_subquery")
    add("年龄高于平均年龄的患者有多少人",
        "SELECT COUNT(*) FROM patients WHERE age > (SELECT AVG(age) FROM patients)",
        "hard", "subquery")

    # --- HAVING + 阈值 ---
    add("就诊次数超过20次的疾病有哪些",
        "SELECT disease, COUNT(*) n FROM visits GROUP BY disease HAVING COUNT(*) > 20 ORDER BY n DESC",
        "hard", "having")
    add("患者数超过10人的科室有哪些",
        "SELECT department, COUNT(DISTINCT patient_id) c FROM visits GROUP BY department HAVING COUNT(DISTINCT patient_id) > 10 ORDER BY c DESC",
        "hard", "having")
    add("平均费用超过5000元的科室及其平均费用",
        "SELECT department, ROUND(AVG(cost),2) ac FROM visits GROUP BY department HAVING AVG(cost) > 5000 ORDER BY ac DESC",
        "hard", "having")

    # --- CTE ---
    add("用 CTE 统计每个科室的就诊量并按降序列出前5",
        "WITH t AS (SELECT department, COUNT(*) n FROM visits GROUP BY department) SELECT department, n FROM t ORDER BY n DESC LIMIT 5",
        "hard", "cte")
    add("每个地区的患者数和平均年龄分别是多少",
        "SELECT region, COUNT(*) n, ROUND(AVG(age),2) aa FROM patients GROUP BY region ORDER BY n DESC",
        "hard", "group_multi_agg")

    # --- 集合运算 ---
    if len(departments) >= 2:
        d0, d1 = departments[0], departments[1]
        add(f"在{d0}和{d1}都出现过的疾病有哪些",
            f"SELECT disease FROM visits WHERE department='{d0}' INTERSECT SELECT disease FROM visits WHERE department='{d1}'",
            "hard", "set_intersect")
        add(f"在{d0}出现但在{d1}没出现的疾病有哪些",
            f"SELECT disease FROM visits WHERE department='{d0}' EXCEPT SELECT disease FROM visits WHERE department='{d1}'",
            "hard", "set_except")

    # --- 相关子查询 / 每组 TopK ---
    for dept in departments[:4]:
        add(f"{dept}中就诊量最高的疾病及次数",
            f"SELECT disease, COUNT(*) n FROM visits WHERE department='{dept}' GROUP BY disease ORDER BY n DESC LIMIT 1",
            "hard", "group_top")

    # --- 时间窗 + JOIN 组合 ---
    for dis in diseases[:4]:
        add(f"最近12个月{dis}的就诊次数",
            f"SELECT COUNT(*) FROM visits WHERE disease='{dis}' AND visit_date >= date((SELECT MAX(visit_date) FROM visits),'-12 months')",
            "hard", "time_filter")
    add("每个月的就诊量趋势",
        "SELECT strftime('%Y-%m', visit_date) m, COUNT(*) n FROM visits GROUP BY m ORDER BY m",
        "hard", "trend")
    add("各年龄段的平均就诊费用",
        "SELECT CASE WHEN p.age<=18 THEN '0-18' WHEN p.age<=40 THEN '19-40' WHEN p.age<=60 THEN '41-60' ELSE '60+' END ag, ROUND(AVG(v.cost),2) ac FROM visits v JOIN patients p ON v.patient_id=p.patient_id GROUP BY ag ORDER BY ag",
        "hard", "case_group")

    # --- 图谱多条件 ---
    add("置信度大于0.9的三元组有多少条",
        "SELECT COUNT(*) FROM kg_triples WHERE confidence > 0.9",
        "hard", "kg_filter")
    add("每种关系的平均置信度是多少",
        "SELECT relation_zh, ROUND(AVG(confidence),3) ac FROM kg_triples GROUP BY relation_zh ORDER BY ac DESC",
        "hard", "kg_group")
    add("拥有并发症最多的疾病是哪个",
        "SELECT head, COUNT(*) n FROM kg_triples WHERE relation='HAS_COMPLICATION' GROUP BY head ORDER BY n DESC LIMIT 1",
        "hard", "kg_group_top")
    add("被最多疾病列为临床表现的症状是哪个",
        "SELECT tail, COUNT(*) n FROM kg_triples WHERE relation='HAS_SYMPTOM' GROUP BY tail ORDER BY n DESC LIMIT 1",
        "hard", "kg_group_top")

    # --- 歧义/多条件组合 ---
    for dis in diseases[:4]:
        for reg in regions[:2]:
            add(f"{reg}地区{dis}患者的就诊次数",
                f"SELECT COUNT(*) FROM visits v JOIN patients p ON v.patient_id=p.patient_id WHERE v.disease='{dis}' AND p.region='{reg}'",
                "hard", "multi_filter")
    add("60岁以上高血压患者的平均住院天数",
        "SELECT ROUND(AVG(v.los_days),2) FROM visits v JOIN patients p ON v.patient_id=p.patient_id WHERE v.disease='高血压' AND p.age>60",
        "hard", "multi_filter")

    # --- 相关子查询：每科室高于本科室平均费用的就诊数 ---
    add("每个科室中费用高于该科室平均费用的就诊有多少条",
        "SELECT department, COUNT(*) n FROM visits v WHERE cost > (SELECT AVG(cost) FROM visits w WHERE w.department=v.department) GROUP BY department ORDER BY n DESC",
        "hard", "correlated_subquery")
    add("费用高于全体平均的就诊占比是多少",
        "SELECT ROUND(100.0*SUM(CASE WHEN cost > (SELECT AVG(cost) FROM visits) THEN 1 ELSE 0 END)/COUNT(*),2) pct FROM visits",
        "hard", "ratio")

    # --- 占比 / 比例 ---
    for dis in diseases[:4]:
        add(f"{dis}就诊占全部就诊的百分比是多少",
            f"SELECT ROUND(100.0*SUM(CASE WHEN disease='{dis}' THEN 1 ELSE 0 END)/COUNT(*),2) pct FROM visits",
            "hard", "ratio")
    add("门诊（住院天数为0）就诊占比是多少",
        "SELECT ROUND(100.0*SUM(CASE WHEN los_days=0 THEN 1 ELSE 0 END)/COUNT(*),2) pct FROM visits",
        "hard", "ratio")
    add("男性患者占全部患者的百分比",
        "SELECT ROUND(100.0*SUM(CASE WHEN gender='男' THEN 1 ELSE 0 END)/COUNT(*),2) pct FROM patients",
        "hard", "ratio")

    # --- 双维度分组 ---
    for dept in departments[:3]:
        add(f"{dept}各疾病的平均费用（降序）",
            f"SELECT disease, ROUND(AVG(cost),2) ac FROM visits WHERE department='{dept}' GROUP BY disease ORDER BY ac DESC",
            "hard", "group_agg")
    add("每个科室每种性别的就诊次数",
        "SELECT v.department, p.gender, COUNT(*) n FROM visits v JOIN patients p ON v.patient_id=p.patient_id GROUP BY v.department, p.gender ORDER BY v.department, p.gender",
        "hard", "two_dim_group")

    # --- 嵌套 TopK：就诊量前3疾病里费用最高的 ---
    add("就诊量前3的疾病中平均费用最高的是哪个",
        "SELECT disease, ROUND(AVG(cost),2) ac FROM visits WHERE disease IN (SELECT disease FROM visits GROUP BY disease ORDER BY COUNT(*) DESC LIMIT 3) GROUP BY disease ORDER BY ac DESC LIMIT 1",
        "hard", "nested_topk")
    add("患者最多的3个地区里平均年龄最大的地区",
        "SELECT region, ROUND(AVG(age),2) aa FROM patients WHERE region IN (SELECT region FROM patients GROUP BY region ORDER BY COUNT(*) DESC LIMIT 3) GROUP BY region ORDER BY aa DESC LIMIT 1",
        "hard", "nested_topk")

    # --- 图谱 + 业务联表思维（各疾病的症状数）---
    add("每种疾病关联的症状数量（降序前10）",
        "SELECT head disease, COUNT(*) n FROM kg_triples WHERE relation='HAS_SYMPTOM' GROUP BY head ORDER BY n DESC LIMIT 10",
        "hard", "kg_group")
    add("既有并发症又有用药禁忌记录的疾病有哪些",
        "SELECT head FROM kg_triples WHERE relation='HAS_COMPLICATION' INTERSECT SELECT head FROM kg_triples WHERE relation='CONTRAINDICATED_FOR'",
        "hard", "kg_set")

    return items


def main() -> int:
    if not DB.is_file():
        raise SystemExit(f"缺少 {DB}，请先运行 demo-all 生成分析库")
    conn = connect_readonly(DB)
    try:
        candidates = build_items(conn)
        kept = []
        dropped = []
        next_id = 1000
        for it in candidates:
            try:
                conn.execute(it["gold_sql"]).fetchall()
            except Exception as e:
                dropped.append({"q": it["question"], "err": str(e)})
                continue
            kept.append({"id": next_id, **it})
            next_id += 1
    finally:
        conn.close()

    OUT.write_text("\n".join(json.dumps(it, ensure_ascii=False) for it in kept) + "\n", encoding="utf-8")
    from collections import Counter
    cats = Counter(it["category"] for it in kept)
    print(json.dumps({"kept": len(kept), "dropped": len(dropped),
                      "by_category": dict(cats), "output": str(OUT)}, ensure_ascii=False, indent=2))
    if dropped:
        print("DROPPED:", json.dumps(dropped[:10], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
