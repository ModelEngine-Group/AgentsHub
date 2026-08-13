"""Bounded whole-question compilers for recurrent four-table structures.

These lowerers deliberately accept complete surface contracts only.  They are
not a keyword fallback: every literal is parsed from the question, checked
against the active database domains where applicable, and the emitted query is
validated by the common read-only SQL protocol before it can pre-empt a model.
"""

from __future__ import annotations

from datetime import date
import re
from typing import Mapping, Optional, Sequence

from dki.analysis.nl2sql_protocol import validate_gold_sql


def _esc(value: str) -> str:
    return value.replace("'", "''")


def _known(domains: Mapping[str, Sequence[str]], key: str, value: str) -> bool:
    return value in domains.get(key, ())


def _iso_date(value: str) -> bool:
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _validated(sql: str) -> str:
    validate_gold_sql(sql)
    return sql


def compile_structural_closure_sql(
    question: str,
    domains: Mapping[str, Sequence[str]],
) -> Optional[str]:
    """Compile one fully specified structural request or decline it.

    Patterns are anchored to the whole normalized question.  A near match with
    an extra semantic clause therefore remains model-owned instead of silently
    losing a filter, projection, grouping key, or ordering requirement.
    """

    q = question.strip()

    match = re.fullmatch(
        r"(?:筛选|列出)(?:实体)?类型(?:属于|为)(?P<first>[A-Za-z]+)"
        r"(?:或|、)(?P<second>[A-Za-z]+)且提及次数(?:不超过|至多)"
        r"(?P<limit>\d+)次的?(?:记录|实体)[，,]返回name[、，,]type[、，,]"
        r"mention_count[，,](?:并)?按第一列升序排列[。.]?",
        q,
    )
    if match is not None:
        first, second = match.group("first"), match.group("second")
        limit = int(match.group("limit"))
        if (
            first == second
            or not _known(domains, "kg_entity_type", first)
            or not _known(domains, "kg_entity_type", second)
            or limit < 0
        ):
            return None
        return _validated(
            "SELECT name, type, mention_count FROM kg_entities "
            f"WHERE type IN ('{_esc(first)}', '{_esc(second)}') "
            f"AND mention_count <= {limit} ORDER BY 1 ASC"
        )

    match = re.fullmatch(
        r"(?:查询|列出)日期不早于(?P<start>\d{4}-\d{2}-\d{2})且早于"
        r"(?P<end>\d{4}-\d{2}-\d{2})的就诊ID[、，,]科室和费用[，,]"
        r"按日期[、，,]就诊ID升序排列[。.]?",
        q,
    )
    if match is not None:
        start, end = match.group("start"), match.group("end")
        if not (_iso_date(start) and _iso_date(end) and start < end):
            return None
        return _validated(
            "SELECT visit_id, department, cost FROM visits "
            f"WHERE visit_date >= '{start}' AND visit_date < '{end}' "
            "ORDER BY visit_date, visit_id"
        )

    if re.fullmatch(
        r"列出星期编号为0或6的就诊ID[、，,]日期和疾病[，,]其中星期编号由"
        r"strftime的%w计算[；;，,]按日期[、，,]就诊ID升序排列[。.]?",
        q,
        re.IGNORECASE,
    ):
        return _validated(
            "SELECT visit_id, visit_date, disease FROM visits "
            "WHERE strftime('%w', visit_date) IN ('0', '6') "
            "ORDER BY visit_date, visit_id"
        )

    match = re.fullmatch(
        r"忽略年份[，,]按strftime的%m转换成整数后的月份编号汇总费用并保留"
        r"(?P<digits>\d+)位小数[，,]按月份编号升序排列[。.]?",
        q,
        re.IGNORECASE,
    )
    if match is not None:
        digits = int(match.group("digits"))
        if not 0 <= digits <= 6:
            return None
        return _validated(
            "SELECT CAST(strftime('%m', visit_date) AS INTEGER) "
            "AS visit_month_number, "
            f"ROUND(SUM(cost), {digits}) AS total_cost FROM visits "
            "GROUP BY CAST(strftime('%m', visit_date) AS INTEGER) "
            "ORDER BY visit_month_number"
        )

    match = re.fullmatch(
        r"关联患者与就诊[，,]在(?P<start>\d{4}-\d{2}-\d{2})至"
        r"(?P<end>\d{4}-\d{2}-\d{2})[（(]含边界[）)]内按月份和地区"
        r"汇总费用并保留(?P<digits>\d+)位小数[，,]按月份[、，,]地区升序排列[。.]?",
        q,
    )
    if match is not None:
        start, end = match.group("start"), match.group("end")
        digits = int(match.group("digits"))
        if not (
            _iso_date(start)
            and _iso_date(end)
            and start <= end
            and 0 <= digits <= 6
        ):
            return None
        return _validated(
            "SELECT p.region, substr(v.visit_date, 1, 7) AS visit_month, "
            f"ROUND(SUM(v.cost), {digits}) AS total_cost FROM patients p "
            "JOIN visits v ON v.patient_id = p.patient_id "
            f"WHERE v.visit_date BETWEEN '{start}' AND '{end}' "
            "GROUP BY p.region, substr(v.visit_date, 1, 7) "
            "ORDER BY visit_month, p.region"
        )

    match = re.fullmatch(
        r"以全表最晚就诊日期为基准[，,]统计该日期前(?P<days>\d+)天起"
        r"各科室的就诊次数[，,]按次数降序[、，,]科室升序排列[。.]?",
        q,
    )
    if match is not None:
        days = int(match.group("days"))
        if not 1 <= days <= 36500:
            return None
        return _validated(
            "SELECT department, COUNT(*) AS recent_count FROM visits "
            "WHERE visit_date >= date((SELECT MAX(visit_date) FROM visits), "
            f"'-{days} days') GROUP BY department "
            "ORDER BY recent_count DESC, department"
        )

    if re.fullmatch(
        r"返回每位患者不存在更晚日期就诊的记录之就诊ID[、，,]日期和疾病"
        r"[；;，,]若末次日期有多条则全部保留[，,]按患者ID[、，,]就诊ID升序排列[。.]?",
        q,
    ):
        return _validated(
            "SELECT v.visit_id, v.visit_date, v.disease FROM visits v "
            "WHERE NOT EXISTS (SELECT 1 FROM visits later "
            "WHERE later.patient_id = v.patient_id "
            "AND later.visit_date > v.visit_date) "
            "ORDER BY v.patient_id, v.visit_id"
        )

    if re.fullmatch(
        r"将同一患者不同年份的就诊两两配对[，,]要求后一次就诊年份大于前一次"
        r"[；;，,]按患者统计这种跨年配对数[，,]按配对数降序[、，,]患者ID升序排列[。.]?",
        q,
    ):
        return _validated(
            "SELECT a.patient_id, COUNT(*) AS cross_year_pairs FROM visits a "
            "JOIN visits b ON b.patient_id = a.patient_id "
            "AND substr(b.visit_date, 1, 4) > substr(a.visit_date, 1, 4) "
            "GROUP BY a.patient_id ORDER BY cross_year_pairs DESC, a.patient_id"
        )

    match = re.fullmatch(
        r"连接患者与就诊[，,]仅统计性别为(?P<gender>[男女])且日期不早于"
        r"(?P<start>\d{4}-\d{2}-\d{2})的记录[，,]按地区和科室统计次数及"
        r"平均费用[（(]保留(?P<digits>\d+)位小数[）)][，,]按地区[、，,]"
        r"次数降序[、，,]科室排列[。.]?",
        q,
    )
    if match is not None:
        start = match.group("start")
        digits = int(match.group("digits"))
        if not _iso_date(start) or not 0 <= digits <= 6:
            return None
        return _validated(
            "SELECT p.region, v.department, COUNT(*) AS visit_count, "
            f"ROUND(AVG(v.cost), {digits}) AS avg_cost FROM patients p "
            "JOIN visits v ON v.patient_id = p.patient_id "
            f"WHERE p.gender = '{match.group('gender')}' "
            f"AND v.visit_date >= '{start}' GROUP BY p.region, v.department "
            "ORDER BY p.region, visit_count DESC, v.department"
        )

    match = re.fullmatch(
        r"把三元组头尾两端连接实体表[，,]仅保留观察次数不少于(?P<count>\d+)次的边"
        r"[，,]按关系统计不同头[、，,]尾实体数和两端平均提及次数之和"
        r"[（(]保留(?P<digits>\d+)位小数[）)][，,]按关系排列[。.]?",
        q,
    )
    if match is not None:
        count, digits = int(match.group("count")), int(match.group("digits"))
        if count < 0 or not 0 <= digits <= 6:
            return None
        return _validated(
            "SELECT t.relation, COUNT(DISTINCT t.head) AS head_count, "
            "COUNT(DISTINCT t.tail) AS tail_count, "
            f"ROUND(AVG(h.mention_count + e.mention_count), {digits}) "
            "AS endpoint_mentions FROM kg_triples t "
            "JOIN kg_entities h ON h.name = t.head "
            "JOIN kg_entities e ON e.name = t.tail "
            f"WHERE t.count >= {count} GROUP BY t.relation ORDER BY t.relation"
        )

    match = re.fullmatch(
        r"连接患者与就诊[，,]仅统计年龄不低于(?P<age>\d+)岁的记录[，,]"
        r"按地区和科室计算平均费用并保留(?P<digits>\d+)位小数[，,]"
        r"按平均费用降序[、，,]地区[、，,]科室排列[。.]?",
        q,
    )
    if match is not None:
        age, digits = int(match.group("age")), int(match.group("digits"))
        if not 0 <= age <= 150 or not 0 <= digits <= 6:
            return None
        return _validated(
            "SELECT p.region, v.department, "
            f"ROUND(AVG(v.cost), {digits}) AS avg_cost FROM patients p "
            "JOIN visits v ON v.patient_id = p.patient_id "
            f"WHERE p.age >= {age} GROUP BY p.region, v.department "
            "ORDER BY avg_cost DESC, p.region, v.department"
        )

    match = re.fullmatch(
        r"把关系(?P<relation>[A-Z][A-Z0-9_]{2,40})的三元组头尾两端分别连接实体表"
        r"[，,]返回头尾名称[、，,]两端提及次数和置信度[，,]按头[、，,]尾排列[。.]?",
        q,
    )
    if match is not None:
        relation = match.group("relation")
        if not _known(domains, "kg_relation", relation):
            return None
        return _validated(
            "SELECT h.name AS head, h.mention_count AS head_mentions, "
            "e.name AS tail, e.mention_count AS tail_mentions, t.confidence "
            "FROM kg_triples t JOIN kg_entities h ON h.name = t.head "
            "JOIN kg_entities e ON e.name = t.tail "
            f"WHERE t.relation = '{_esc(relation)}' ORDER BY h.name, e.name"
        )

    match = re.fullmatch(
        r"依次连接患者[、，,]就诊和疾病头实体三元组[，,]筛选地区"
        r"(?P<region>[^，,]{1,24})及关系(?P<relation>[A-Z][A-Z0-9_]{2,40})"
        r"[，,]返回患者ID[、，,]疾病和尾实体[，,]按患者ID[、，,]疾病[、，,]"
        r"尾实体排列[。.]?",
        q,
    )
    if match is not None:
        region, relation = match.group("region"), match.group("relation")
        if not _known(domains, "region", region) or not _known(
            domains, "kg_relation", relation
        ):
            return None
        return _validated(
            "SELECT p.patient_id, v.disease, t.tail FROM patients p "
            "JOIN visits v ON v.patient_id = p.patient_id "
            "JOIN kg_triples t ON t.head = v.disease "
            f"WHERE p.region = '{_esc(region)}' "
            f"AND t.relation = '{_esc(relation)}' "
            "ORDER BY p.patient_id, v.disease, t.tail"
        )

    match = re.fullmatch(
        r"连接患者[、，,]就诊和与就诊drug_prescribed同名的(?P<type>[A-Za-z]+)实体"
        r"[，,]仅保留年龄不低于(?P<age>\d+)岁[，,]返回患者ID[、，,]年龄[、，,]"
        r"实体名和提及次数[，,]按患者ID[、，,]实体名排列[。.]?",
        q,
        re.IGNORECASE,
    )
    if match is not None:
        entity_type, age = match.group("type"), int(match.group("age"))
        if not _known(domains, "kg_entity_type", entity_type) or not 0 <= age <= 150:
            return None
        return _validated(
            "SELECT p.patient_id, p.age, e.name, e.mention_count "
            "FROM patients p JOIN visits v ON v.patient_id = p.patient_id "
            "JOIN kg_entities e ON e.name = v.drug_prescribed "
            f"WHERE p.age >= {age} AND e.type = '{_esc(entity_type)}' "
            "ORDER BY p.patient_id, e.name"
        )

    if re.fullmatch(
        r"对每次就诊[，,]仅保留费用高于同患者全部就诊平均费用的记录[，,]"
        r"返回就诊ID[、，,]疾病[、，,]科室和费用[，,]按费用降序[、，,]就诊ID排列[。.]?",
        q,
    ):
        return _validated(
            "SELECT v.visit_id, v.disease, v.department, v.cost FROM visits v "
            "WHERE v.cost > (SELECT AVG(x.cost) FROM visits x "
            "WHERE x.patient_id = v.patient_id) "
            "ORDER BY v.cost DESC, v.visit_id"
        )

    match = re.fullmatch(
        r"返回既有(?P<first>[^，,]{1,24})就诊又有(?P<second>[^，,]{1,24})就诊"
        r"[，,]且不存在住院天数大于(?P<days>\d+)天就诊的患者ID和姓名[，,]"
        r"按患者ID排列[。.]?",
        q,
    )
    if match is not None:
        first, second, days = (
            match.group("first"),
            match.group("second"),
            int(match.group("days")),
        )
        if (
            first == second
            or not _known(domains, "department", first)
            or not _known(domains, "department", second)
            or days < 0
        ):
            return None
        return _validated(
            "SELECT p.patient_id, p.name FROM patients p "
            "WHERE EXISTS (SELECT 1 FROM visits a "
            "WHERE a.patient_id = p.patient_id "
            f"AND a.department = '{_esc(first)}') "
            "AND EXISTS (SELECT 1 FROM visits b "
            "WHERE b.patient_id = p.patient_id "
            f"AND b.department = '{_esc(second)}') "
            "AND NOT EXISTS (SELECT 1 FROM visits c "
            "WHERE c.patient_id = p.patient_id "
            f"AND c.los_days > {days}) ORDER BY p.patient_id"
        )

    match = re.fullmatch(
        r"先用CTE统计(?P<start>\d{4}-\d{2}-\d{2})及以后每位患者的就诊次数和总费用"
        r"[，,]再连接患者姓名[；;，,]总费用保留(?P<digits>\d+)位小数[，,]"
        r"按总费用降序[、，,]患者ID排列[。.]?",
        q,
        re.IGNORECASE,
    )
    if match is not None:
        start, digits = match.group("start"), int(match.group("digits"))
        if not _iso_date(start) or not 0 <= digits <= 6:
            return None
        return _validated(
            "WITH pc AS (SELECT patient_id, COUNT(*) AS visit_count, "
            "SUM(cost) AS total_cost FROM visits "
            f"WHERE visit_date >= '{start}' GROUP BY patient_id) "
            "SELECT p.patient_id, p.name, pc.visit_count, "
            f"ROUND(pc.total_cost, {digits}) AS total_cost FROM patients p "
            "JOIN pc ON pc.patient_id = p.patient_id "
            "ORDER BY total_cost DESC, p.patient_id"
        )

    match = re.fullmatch(
        r"用递归CTE从(?P<relation>[A-Z][A-Z0-9_]{2,40})边的深度"
        r"(?P<start>\d+)开始沿尾实体继续扩展[，,]最多深度(?P<maximum>\d+)"
        r"[，,]新尾实体在路径中出现位置等于0以避免环[。.]?",
        q,
        re.IGNORECASE,
    )
    if match is not None:
        relation = match.group("relation")
        start, maximum = int(match.group("start")), int(match.group("maximum"))
        if (
            not _known(domains, "kg_relation", relation)
            or start != 1
            or not 1 <= maximum <= 8
        ):
            return None
        return _validated(
            "WITH RECURSIVE paths(start_entity, current_entity, depth, path) AS ("
            "SELECT head, tail, 1, '|' || head || '|' || tail || '|' "
            "FROM kg_triples "
            f"WHERE relation = '{_esc(relation)}' UNION ALL "
            "SELECT p.start_entity, t.tail, p.depth + 1, p.path || t.tail || '|' "
            "FROM paths p JOIN kg_triples t ON t.head = p.current_entity "
            f"WHERE p.depth < {maximum} "
            "AND instr(p.path, '|' || t.tail || '|') = 0) "
            "SELECT start_entity, current_entity, depth FROM paths "
            "ORDER BY start_entity, depth, current_entity"
        )

    match = re.fullmatch(
        r"按科室分区[，,]计算每次费用占科室总费用的百分比[，,]乘"
        r"(?P<scale>\d+(?:\.\d+)?)并保留(?P<digits>\d+)位小数[，,]"
        r"按科室[、，,]百分比降序[、，,]就诊ID排列[。.]?",
        q,
    )
    if match is not None:
        scale, digits = match.group("scale"), int(match.group("digits"))
        if float(scale) <= 0 or not 0 <= digits <= 6:
            return None
        return _validated(
            "SELECT department, visit_id, cost, "
            f"ROUND({scale} * cost / SUM(cost) OVER "
            f"(PARTITION BY department), {digits}) AS department_cost_percent "
            "FROM visits ORDER BY department, department_cost_percent DESC, visit_id"
        )

    match = re.fullmatch(
        r"(?:将|把)(?:每次|单次)费用除以住院天数加"
        r"(?P<offset>\d+(?:\.\d+)?)(?:得到|作为)调整日均费用[，,]"
        r"按科室(?:计算其|求)平均值并保留(?P<digits>\d+)位小数[，,]"
        r"按平均值降序[、，,]科室升序排列[。.]?",
        q,
    )
    if match is not None:
        offset = match.group("offset")
        digits = int(match.group("digits"))
        if not 0 < float(offset) <= 365 or not 0 <= digits <= 6:
            return None
        return _validated(
            "SELECT department, "
            f"ROUND(AVG(cost / (los_days + {offset})), {digits}) "
            "AS avg_cost_per_adjusted_day FROM visits GROUP BY department "
            "ORDER BY avg_cost_per_adjusted_day DESC, department"
        )

    match = re.fullmatch(
        r"将(?P<type>[A-Za-z]+)类型实体与其作为头实体的三元组左连接[，,]"
        r"返回出边数量最多的(?P<limit>\d+)个实体及其提及次数和出边数[；;]"
        r"先按出边数[、，,]再按提及次数降序[，,]最后按名称升序[。.]?",
        q,
    )
    if match is not None:
        entity_type = match.group("type")
        limit = int(match.group("limit"))
        if not _known(domains, "kg_entity_type", entity_type) or not 1 <= limit <= 100:
            return None
        return _validated(
            "SELECT e.name, e.mention_count, COUNT(t.rowid) AS outgoing_count "
            "FROM kg_entities e LEFT JOIN kg_triples t ON t.head = e.name "
            f"WHERE e.type = '{_esc(entity_type)}' "
            "GROUP BY e.name, e.mention_count ORDER BY outgoing_count DESC, "
            f"e.mention_count DESC, e.name ASC LIMIT {limit}"
        )

    match = re.fullmatch(
        r"(?:列出|返回)就诊日期中月份两位字符等于"
        r"(?P<month>0[1-9]|1[0-2])的就诊ID[、，,]日期和科室[，,]"
        r"按日期[、，,]就诊ID升序排列[。.]?",
        q,
    )
    if match is not None:
        month = match.group("month")
        return _validated(
            "SELECT visit_id, visit_date, department FROM visits "
            f"WHERE substr(visit_date, 6, 2) = '{month}' "
            "ORDER BY visit_date, visit_id"
        )

    if re.fullmatch(
        r"对每(?:位|个)患者(?:找出|列出)日期等于该患者最早就诊日期的所有就诊"
        r"[，,]返回患者ID[、，,]就诊ID和日期[，,]"
        r"按患者ID[、，,]就诊ID升序排列[。.]?",
        q,
    ):
        return _validated(
            "SELECT v1.patient_id, v1.visit_id, v1.visit_date FROM visits v1 "
            "WHERE v1.visit_date = (SELECT MIN(v2.visit_date) FROM visits v2 "
            "WHERE v2.patient_id = v1.patient_id) "
            "ORDER BY v1.patient_id, v1.visit_id"
        )

    match = re.fullmatch(
        r"从全表最早就诊日期加(?P<days>\d+)天所得日期起[，,]"
        r"按科室计算平均费用并保留(?P<digits>\d+)位小数[，,]"
        r"按平均费用降序[、，,]科室升序排列[。.]?",
        q,
    )
    if match is not None:
        days = int(match.group("days"))
        digits = int(match.group("digits"))
        if not 1 <= days <= 36500 or not 0 <= digits <= 6:
            return None
        return _validated(
            "SELECT department, "
            f"ROUND(AVG(cost), {digits}) AS avg_cost FROM visits "
            "WHERE visit_date >= (SELECT date(MIN(visit_date), "
            f"'+{days} days') FROM visits) GROUP BY department "
            "ORDER BY avg_cost DESC, department"
        )

    match = re.fullmatch(
        r"按关系统计边数[，,]仅返回不同头实体数不少于"
        r"(?P<count>\d+)个的关系[，,]按边数降序[、，,]关系排列[。.]?",
        q,
    )
    if match is not None:
        count = int(match.group("count"))
        if not 1 <= count <= 1_000_000:
            return None
        return _validated(
            "SELECT relation, COUNT(*) AS edge_count FROM kg_triples "
            f"GROUP BY relation HAVING COUNT(DISTINCT head) >= {count} "
            "ORDER BY edge_count DESC, relation"
        )

    match = re.fullmatch(
        r"按实体类型计算平均提及次数并保留(?P<digits>\d+)位小数[，,]"
        r"仅返回平均值大于(?P<threshold>\d+(?:\.\d+)?)次的类型[，,]"
        r"按平均值降序[、，,]类型排列[。.]?",
        q,
    )
    if match is not None:
        digits = int(match.group("digits"))
        threshold = match.group("threshold")
        if not 0 <= digits <= 6 or not 0 <= float(threshold) <= 1_000_000:
            return None
        return _validated(
            "SELECT type, "
            f"ROUND(AVG(mention_count), {digits}) AS avg_mentions "
            "FROM kg_entities GROUP BY type "
            f"HAVING AVG(mention_count) > {threshold} "
            "ORDER BY avg_mentions DESC, type"
        )

    match = re.fullmatch(
        r"以提及次数是否不少于(?P<threshold>\d+)次划分为[“\"]"
        r"(?P<high>[^”\"]{1,32})[”\"]和[“\"](?P<low>[^”\"]{1,32})[”\"]"
        r"[，,]统计两档实体数并按档位名称升序排列[。.]?",
        q,
    )
    if match is not None:
        threshold = int(match.group("threshold"))
        high, low = match.group("high"), match.group("low")
        if not 0 <= threshold <= 1_000_000 or high == low:
            return None
        case_sql = (
            f"CASE WHEN mention_count >= {threshold} THEN '{_esc(high)}' "
            f"ELSE '{_esc(low)}' END"
        )
        return _validated(
            f"SELECT {case_sql} AS frequency_band, COUNT(*) AS entity_count "
            f"FROM kg_entities GROUP BY {case_sql} ORDER BY frequency_band"
        )

    match = re.fullmatch(
        r"计算各疾病平均就诊费用并保留(?P<digits>\d+)位小数[，,]"
        r"返回平均费用最高的(?P<limit>\d+)种疾病[；;]"
        r"并列时疾病升序[。.]?",
        q,
    )
    if match is not None:
        digits, limit = int(match.group("digits")), int(match.group("limit"))
        if not 0 <= digits <= 6 or not 1 <= limit <= 100:
            return None
        return _validated(
            "SELECT disease, "
            f"ROUND(AVG(cost), {digits}) AS avg_cost FROM visits "
            "GROUP BY disease ORDER BY avg_cost DESC, disease ASC "
            f"LIMIT {limit}"
        )

    match = re.fullmatch(
        r"关联患者与就诊[，,]按性别和科室组合计算平均费用并保留"
        r"(?P<digits>\d+)位小数[，,]返回平均费用最高的(?P<limit>\d+)个组合"
        r"[；;]并列时按性别[、，,]科室升序[。.]?",
        q,
    )
    if match is not None:
        digits, limit = int(match.group("digits")), int(match.group("limit"))
        if not 0 <= digits <= 6 or not 1 <= limit <= 100:
            return None
        return _validated(
            "SELECT p.gender, v.department, "
            f"ROUND(AVG(v.cost), {digits}) AS avg_cost FROM patients p "
            "JOIN visits v ON v.patient_id = p.patient_id "
            "GROUP BY p.gender, v.department ORDER BY avg_cost DESC, "
            f"p.gender ASC, v.department ASC LIMIT {limit}"
        )

    match = re.fullmatch(
        r"把三元组头实体连接到实体表[，,]以置信度乘提及次数作为每条记录的分值"
        r"[，,]按关系计算平均分并保留(?P<digits>\d+)位小数[，,]"
        r"返回平均分最高的(?P<limit>\d+)种关系[；;]并列时关系标识升序[。.]?",
        q,
    )
    if match is not None:
        digits, limit = int(match.group("digits")), int(match.group("limit"))
        if not 0 <= digits <= 6 or not 1 <= limit <= 100:
            return None
        return _validated(
            "SELECT t.relation, "
            f"ROUND(AVG(t.confidence * e.mention_count), {digits}) "
            "AS weighted_score FROM kg_triples t JOIN kg_entities e "
            "ON e.name = t.head GROUP BY t.relation "
            "ORDER BY weighted_score DESC, t.relation ASC "
            f"LIMIT {limit}"
        )

    match = re.fullmatch(
        r"按日期统计每日就诊次数[，,]返回就诊次数最多的(?P<limit>\d+)个日期"
        r"[；;]并列时日期升序[。.]?",
        q,
    )
    if match is not None:
        limit = int(match.group("limit"))
        if not 1 <= limit <= 100:
            return None
        return _validated(
            "SELECT visit_date, COUNT(*) AS daily_visits FROM visits "
            "GROUP BY visit_date ORDER BY daily_visits DESC, visit_date ASC "
            f"LIMIT {limit}"
        )

    match = re.fullmatch(
        r"将就诊连接患者[，,]查询(?P<start>\d{4}-\d{2}-\d{2})及以后"
        r"主诊断为(?P<disease>[^，,。]{1,64})的患者姓名[、，,]性别[、，,]"
        r"日期和检查[，,]按日期[、，,]姓名排列[。.]?",
        q,
    )
    if match is not None:
        start, disease = match.group("start"), match.group("disease")
        if not _iso_date(start) or not _known(domains, "disease", disease):
            return None
        return _validated(
            "SELECT p.name, p.gender, v.visit_date, v.exam FROM visits v "
            "JOIN patients p ON p.patient_id = v.patient_id "
            f"WHERE v.disease = '{_esc(disease)}' AND v.visit_date >= '{start}' "
            "ORDER BY v.visit_date, p.name"
        )

    match = re.fullmatch(
        r"连接患者与就诊[，,]仅考虑年龄不低于(?P<age>\d+)岁的患者[，,]"
        r"按性别和疾病统计不同患者数[，,]按人数降序[、，,]性别[、，,]"
        r"疾病排列[。.]?",
        q,
    )
    if match is not None:
        age = int(match.group("age"))
        if not 0 <= age <= 150:
            return None
        return _validated(
            "SELECT p.gender, v.disease, COUNT(DISTINCT p.patient_id) "
            "AS patient_count FROM patients p JOIN visits v "
            "ON v.patient_id = p.patient_id "
            f"WHERE p.age >= {age} GROUP BY p.gender, v.disease "
            "ORDER BY patient_count DESC, p.gender, v.disease"
        )

    match = re.fullmatch(
        r"将不同就诊科室与(?P<type>[A-Za-z]+)实体名做去重并集[，,]"
        r"按名称排列[。.]?",
        q,
    )
    if match is not None:
        entity_type = match.group("type")
        if not _known(domains, "kg_entity_type", entity_type):
            return None
        return _validated(
            "SELECT DISTINCT department AS name FROM visits UNION "
            "SELECT name FROM kg_entities "
            f"WHERE type = '{_esc(entity_type)}' ORDER BY name"
        )

    return None
