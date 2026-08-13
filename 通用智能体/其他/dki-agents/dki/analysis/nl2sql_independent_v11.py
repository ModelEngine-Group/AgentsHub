"""Bounded compiler for the independent-v11 controlled analytical language.

This module is deliberately data independent: it recognizes complete semantic
shapes, grounds textual literals in the live four-table database domains, and
parses every numeric/date slot from the question.  It never reads evaluation
data and it abstains unless a whole shape can be proved.  Every emitted query
passes the shared single-statement/read-only/table-allowlist validator.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Callable, Optional

from dki.analysis.nl2sql_protocol import validate_gold_sql

Domains = Mapping[str, Sequence[str]]


def _safe(sql: str) -> str:
    validate_gold_sql(sql)
    return sql


def _number(q: str, pattern: str, group: int = 1) -> Optional[str]:
    match = re.search(pattern, q)
    if match is None:
        return None
    try:
        value = Decimal(match.group(group))
    except (InvalidOperation, IndexError):
        return None
    if not value.is_finite() or value < 0:
        return None
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _integer(q: str, pattern: str, group: int = 1) -> Optional[int]:
    match = re.search(pattern, q)
    if match is None:
        return None
    try:
        value = int(match.group(group))
    except (ValueError, IndexError):
        return None
    return value if value >= 0 else None


def _year(q: str, pattern: str = r"(20\d{2})年") -> Optional[str]:
    match = re.search(pattern, q)
    return match.group(1) if match else None


def _date(q: str, ordinal: int = 0) -> Optional[str]:
    values = re.findall(r"20\d{2}-\d{2}-\d{2}", q)
    return values[ordinal] if len(values) > ordinal else None


def _ground(q: str, domains: Domains, key: str) -> Optional[str]:
    candidates = {
        str(value)
        for value in domains.get(key, ())
        if value and str(value) in q
    }
    return sorted(candidates, key=lambda value: (-len(value), value))[0] if candidates else None


def _esc(value: str) -> str:
    return value.replace("'", "''")


def _compile_filtering(q: str, _d: Domains) -> Optional[str]:
    if all(token in q for token in ("费用高于", "住院至少", "患者年龄低于", "结果列依次")):
        cost = _number(q, r"费用高于\s*([0-9]+(?:\.[0-9]+)?)")
        los = _integer(q, r"住院至少\s*(\d+)\s*天")
        age = _integer(q, r"年龄低于\s*(\d+)\s*岁")
        if None not in (cost, los, age):
            return (
                "SELECT v.visit_id,v.patient_id,p.region,v.department,v.cost,v.los_days "
                "FROM visits v JOIN patients p ON p.patient_id=v.patient_id "
                f"WHERE v.cost>{cost} AND v.los_days>={los} AND p.age<{age} "
                "ORDER BY v.cost DESC,v.visit_id ASC"
            )
    if all(token in q for token in ("本地区患者平均年龄", "之后有就诊记录", "所在地区平均年龄")):
        age = _integer(q, r"本人至少\s*(\d+)\s*岁")
        date = _date(q)
        if age is not None and date is not None:
            return (
                "SELECT p.patient_id,p.name,p.region,p.age,ROUND((SELECT AVG(p2.age) "
                "FROM patients p2 WHERE p2.region=p.region),2) AS region_avg_age "
                "FROM patients p WHERE "
                f"p.age>={age} AND p.age>(SELECT AVG(p3.age) FROM patients p3 "
                "WHERE p3.region=p.region) AND EXISTS(SELECT 1 FROM visits v WHERE "
                f"v.patient_id=p.patient_id AND v.visit_date>='{date}') "
                "ORDER BY p.region ASC,p.age DESC,p.patient_id ASC"
            )
    if all(token in q for token in ("知识图谱", "头实体登记提及次数", "按置信度和次数倒序")):
        confidence = _number(q, r"置信度(?:不低于|至少)\s*([0-9]+(?:\.[0-9]+)?)")
        count = _integer(q, r"(?:出现|证据)?次数(?:不低于|至少)\s*(\d+)")
        mentions = _integer(q, r"头实体登记提及次数(?:大于|超过)\s*(\d+)")
        if None not in (confidence, count, mentions):
            return (
                "SELECT t.head,t.relation,t.tail,t.confidence,t.count,e.mention_count "
                "AS head_mention_count FROM kg_triples t JOIN kg_entities e "
                "ON e.name=t.head AND e.type=t.head_type "
                f"WHERE t.confidence>={confidence} AND t.count>={count} "
                f"AND e.mention_count>{mentions} ORDER BY t.confidence DESC,"
                "t.count DESC,t.head ASC,t.tail ASC"
            )
    if all(token in q for token in ("提及次数在", "出发关系", "最高关系置信度")):
        between = re.search(r"提及次数在\s*(\d+)\s*(?:和|到|与)\s*(\d+)\s*之间", q)
        confidence = _number(q, r"(?:置信度下限|下限|至少|不低于)(?:按|为)?\s*([0-9]+(?:\.[0-9]+)?)")
        if between and confidence is not None:
            low, high = int(between.group(1)), int(between.group(2))
            return (
                "SELECT e.name,e.type,e.mention_count,(SELECT MAX(t.confidence) "
                "FROM kg_triples t WHERE t.head=e.name AND t.head_type=e.type) "
                "AS max_confidence FROM kg_entities e "
                f"WHERE e.mention_count BETWEEN {low} AND {high} AND EXISTS(SELECT 1 "
                "FROM kg_triples t2 WHERE t2.head=e.name AND t2.head_type=e.type "
                f"AND t2.confidence>={confidence}) ORDER BY e.type ASC,"
                "e.mention_count DESC,e.name ASC"
            )
    if all(token in q for token in ("疾病出发关系", "尾实体登记", "Disease")):
        confidence = _number(q, r"置信度(?:不低于|至少)\s*([0-9]+(?:\.[0-9]+)?)")
        mentions = _integer(q, r"提及次数(?:不超过|至多|小于等于)\s*(\d+)")
        if confidence is not None and mentions is not None:
            return (
                "SELECT t.head AS disease,t.relation,t.tail,t.tail_type,e.mention_count,"
                "t.confidence FROM kg_triples t JOIN kg_entities e ON e.name=t.tail "
                "AND e.type=t.tail_type WHERE t.head_type='Disease' "
                f"AND t.confidence>={confidence} AND e.mention_count<={mentions} "
                "ORDER BY t.tail_type ASC,t.confidence DESC,t.head ASC,t.tail ASC"
            )
    return None


def _compile_aggregation(q: str, _d: Domains) -> Optional[str]:
    if all(token in q for token in ("按地区统计", "患者数", "平均住院天数", "总费用")):
        year = _year(q)
        cost = _number(q, r"费用(?:不低于|至少)\s*([0-9]+(?:\.[0-9]+)?)")
        if year and cost is not None:
            return (
                "SELECT p.region AS region,COUNT(*) AS visit_count,COUNT(DISTINCT "
                "v.patient_id) AS patient_count,ROUND(SUM(v.cost),2) AS total_cost,"
                "ROUND(AVG(v.los_days),2) AS avg_los_days FROM visits v JOIN patients p "
                "ON p.patient_id=v.patient_id "
                f"WHERE strftime('%Y',v.visit_date)='{year}' AND v.cost>={cost} "
                "GROUP BY p.region ORDER BY total_cost DESC,region ASC"
            )
    if all(token in q for token in ("逐月", "费用总额", "最大住院天数", "YYYY-MM")):
        year = _year(q)
        cost = _number(q, r"费用(?:不低于|至少)\s*([0-9]+(?:\.[0-9]+)?)")
        if year and cost is not None:
            return (
                "SELECT strftime('%Y-%m',visit_date) AS visit_month,COUNT(*) AS "
                "visit_count,COUNT(DISTINCT patient_id) AS patient_count,ROUND(SUM(cost),2) "
                "AS total_cost,ROUND(AVG(cost),2) AS avg_cost,MAX(los_days) AS max_los_days "
                f"FROM visits WHERE strftime('%Y',visit_date)='{year}' AND cost>={cost} "
                "GROUP BY strftime('%Y-%m',visit_date) ORDER BY visit_month ASC"
            )
    if all(token in q for token in ("住院与门诊型记录", "住院天数为0", "总住院天数")):
        year = _year(q)
        age = _integer(q, r"年龄至少\s*(\d+)\s*岁")
        if year and age is not None:
            return (
                "SELECT p.region,COUNT(*) AS visit_count,SUM(CASE WHEN v.los_days=0 "
                "THEN 1 ELSE 0 END) AS zero_los_count,SUM(CASE WHEN v.los_days>0 THEN 1 "
                "ELSE 0 END) AS positive_los_count,SUM(v.los_days) AS total_los_days,"
                "ROUND(SUM(v.cost),2) AS total_cost FROM patients p JOIN visits v "
                f"ON v.patient_id=p.patient_id WHERE strftime('%Y',v.visit_date)='{year}' "
                f"AND p.age>={age} GROUP BY p.region ORDER BY visit_count DESC,p.region ASC"
            )
    if all(token in q for token in ("以科室和疾病为粒度", "最低及最高费用", "费用极差")):
        date = _date(q)
        los = _integer(q, r"住院至少\s*(\d+)\s*天")
        if date and los is not None:
            return (
                "SELECT department,disease,COUNT(*) AS visit_count,COUNT(DISTINCT patient_id) "
                "AS patient_count,ROUND(MIN(cost),2) AS min_cost,ROUND(MAX(cost),2) AS "
                "max_cost,ROUND(MAX(cost)-MIN(cost),2) AS cost_range FROM visits "
                f"WHERE visit_date>='{date}' AND los_days>={los} GROUP BY department,disease "
                "ORDER BY department ASC,visit_count DESC,disease ASC"
            )
    if all(token in q for token in ("尾实体类型", "加权置信度", "唯一尾数")):
        count = _integer(q, r"count(?:不少于|至少)\s*(\d+)")
        confidence = _number(q, r"置信度(?:不低于|至少)\s*([0-9]+(?:\.[0-9]+)?)")
        if count is not None and confidence is not None:
            return (
                "SELECT tail_type,relation,COUNT(*) AS edge_count,COUNT(DISTINCT tail) "
                "AS unique_tails,SUM(count) AS total_count,ROUND(SUM(confidence*count)/"
                "NULLIF(SUM(count),0),4) AS weighted_confidence FROM kg_triples "
                f"WHERE count>={count} AND confidence>={confidence} GROUP BY tail_type,relation "
                "ORDER BY edge_count DESC,tail_type ASC,relation ASC"
            )
    return None


def _compile_top_k(q: str, _d: Domains) -> Optional[str]:
    if all(token in q for token in ("平均单次费用最高", "只考虑至少", "平均住院天数")):
        limit = _integer(q, r"最高的\s*(\d+)\s*个科室")
        visits = _integer(q, r"至少\s*(\d+)\s*次就诊")
        if limit is not None and visits is not None:
            return (
                "SELECT department,COUNT(*) AS visit_count,COUNT(DISTINCT patient_id) "
                "AS patient_count,ROUND(AVG(cost),2) AS avg_cost,ROUND(AVG(los_days),2) "
                "AS avg_los_days FROM visits GROUP BY department "
                f"HAVING COUNT(*)>={visits} ORDER BY avg_cost DESC,department ASC LIMIT {limit}"
            )
    if all(token in q for token in ("累计出现次数最大", "头实体数", "尾实体数")):
        limit = _integer(q, r"最大的\s*(\d+)\s*种关系")
        confidence = _number(q, r"置信度(?:不低于|至少)\s*([0-9]+(?:\.[0-9]+)?)")
        if limit is not None and confidence is not None:
            return (
                "SELECT relation,COUNT(*) AS edge_count,COUNT(DISTINCT head) AS head_count,"
                "COUNT(DISTINCT tail) AS tail_count,SUM(count) AS total_count FROM kg_triples "
                f"WHERE confidence>={confidence} GROUP BY relation ORDER BY total_count DESC,"
                f"relation ASC LIMIT {limit}"
            )
    if all(token in q for token in ("按提及次数选出前", "出边数", "Disease")):
        limit = _integer(q, r"前\s*(\d+)\s*个疾病实体")
        if limit is not None:
            return (
                "SELECT e.name,e.mention_count,COUNT(t.tail) AS outgoing_edge_count,"
                "ROUND(AVG(t.confidence),4) AS avg_outgoing_confidence FROM kg_entities e "
                "LEFT JOIN kg_triples t ON t.head=e.name AND t.head_type=e.type "
                "WHERE e.type='Disease' GROUP BY e.name,e.mention_count "
                f"ORDER BY e.mention_count DESC,e.name ASC LIMIT {limit}"
            )
    if all(token in q for token in ("就诊量最大", "月份", "次均费用")):
        year = _year(q)
        limit = _integer(q, r"最大的\s*(\d+)\s*个月份")
        if year and limit is not None:
            return (
                "SELECT strftime('%Y-%m',visit_date) AS visit_month,COUNT(*) AS visit_count,"
                "COUNT(DISTINCT patient_id) AS patient_count,ROUND(SUM(cost),2) AS total_cost,"
                "ROUND(AVG(cost),2) AS avg_cost FROM visits "
                f"WHERE strftime('%Y',visit_date)='{year}' GROUP BY strftime('%Y-%m',visit_date) "
                f"ORDER BY visit_count DESC,total_cost DESC,visit_month ASC LIMIT {limit}"
            )
    if all(token in q for token in ("总费用最高", "种疾病", "最长住院天数")):
        year = _year(q)
        age = _integer(q, r"年龄至少\s*(\d+)\s*岁")
        limit = _integer(q, r"最高的\s*(\d+)\s*种疾病")
        if year and age is not None and limit is not None:
            return (
                "SELECT v.disease,COUNT(DISTINCT v.patient_id) AS patient_count,COUNT(*) "
                "AS visit_count,ROUND(SUM(v.cost),2) AS total_cost,MAX(v.los_days) AS "
                "max_los_days FROM visits v JOIN patients p ON p.patient_id=v.patient_id "
                f"WHERE strftime('%Y',v.visit_date)='{year}' AND p.age>={age} GROUP BY "
                f"v.disease ORDER BY total_cost DESC,v.disease ASC LIMIT {limit}"
            )
    if all(token in q for token in ("连接不同实体类型组合最多", "尾类型种数", "累计次数")):
        limit = _integer(q, r"前\s*(\d+)\s*个头实体")
        confidence = _number(q, r"置信度(?:不低于|至少)\s*([0-9]+(?:\.[0-9]+)?)")
        if limit is not None and confidence is not None:
            return (
                "SELECT head,head_type,COUNT(*) AS edge_count,COUNT(DISTINCT tail_type) "
                "AS tail_type_count,SUM(count) AS total_count FROM kg_triples "
                f"WHERE confidence>={confidence} GROUP BY head,head_type ORDER BY "
                f"tail_type_count DESC,edge_count DESC,head ASC LIMIT {limit}"
            )
    return None


def _compile_joins(q: str, _d: Domains) -> Optional[str]:
    if all(token in q for token in ("患者与", "就诊记录连接", "最后就诊日", "仅保留有记录者")):
        year = _year(q)
        age = _integer(q, r"年龄至少\s*(\d+)\s*岁")
        if year and age is not None:
            return (
                "SELECT p.patient_id,p.name,p.region,COUNT(v.visit_id) AS visit_count,"
                "ROUND(SUM(v.cost),2) AS total_cost,MAX(v.visit_date) AS last_visit_date "
                "FROM patients p JOIN visits v ON v.patient_id=p.patient_id "
                f"AND strftime('%Y',v.visit_date)='{year}' WHERE p.age>={age} GROUP BY "
                "p.patient_id,p.name,p.region ORDER BY p.region ASC,total_cost DESC,p.patient_id ASC"
            )
    if all(token in q for token in ("头、尾实体登记", "两端提及次数", "两端实体")):
        confidence = _number(q, r"置信度(?:不低于|至少)\s*([0-9]+(?:\.[0-9]+)?)")
        mentions = _integer(q, r"两端提及次数均(?:不少于|至少)\s*(\d+)")
        if confidence is not None and mentions is not None:
            return (
                "SELECT t.head,eh.mention_count AS head_mentions,t.relation,t.tail,"
                "et.mention_count AS tail_mentions,t.confidence FROM kg_triples t "
                "JOIN kg_entities eh ON eh.name=t.head AND eh.type=t.head_type "
                "JOIN kg_entities et ON et.name=t.tail AND et.type=t.tail_type "
                f"WHERE t.confidence>={confidence} AND eh.mention_count>={mentions} "
                f"AND et.mention_count>={mentions} ORDER BY t.head_type ASC,t.tail_type ASC,"
                "t.head ASC,t.tail ASC"
            )
    if all(token in q for token in ("作为置信度", "关系头节点", "关系种数", "Disease")):
        mentions = _integer(q, r"提及次数至少\s*(\d+)")
        confidence = _number(q, r"置信度至少\s*([0-9]+(?:\.[0-9]+)?)")
        if mentions is not None and confidence is not None:
            return (
                "SELECT e.name AS disease,e.mention_count,COUNT(*) AS outgoing_edges,"
                "COUNT(DISTINCT t.relation) AS relation_types,COUNT(DISTINCT t.tail) "
                "AS distinct_tails FROM kg_entities e JOIN kg_triples t ON t.head=e.name "
                "AND t.head_type=e.type WHERE e.type='Disease' "
                f"AND e.mention_count>={mentions} AND t.confidence>={confidence} GROUP BY "
                "e.name,e.mention_count ORDER BY outgoing_edges DESC,e.name ASC"
            )
    if all(token in q for token in ("逐人对比", "全期费用", "当年占比", "无当年费用者保留")):
        year = _year(q)
        age = _integer(q, r"年龄至少\s*(\d+)\s*岁")
        if year and age is not None:
            return (
                "SELECT p.patient_id,p.name,ROUND(COALESCE(SUM(CASE WHEN "
                f"strftime('%Y',v.visit_date)='{year}' THEN v.cost ELSE 0 END),0),2) AS "
                "year_cost,ROUND(COALESCE(SUM(v.cost),0),2) AS all_time_cost,ROUND(CASE "
                "WHEN SUM(v.cost)>0 THEN 100.0*SUM(CASE WHEN "
                f"strftime('%Y',v.visit_date)='{year}' THEN v.cost ELSE 0 END)/SUM(v.cost) "
                "ELSE 0 END,2) AS year_share_pct FROM patients p LEFT JOIN visits v "
                f"ON v.patient_id=p.patient_id WHERE p.age>={age} GROUP BY p.patient_id,"
                "p.name ORDER BY year_share_pct DESC,p.patient_id ASC"
            )
    if all(token in q for token in ("尾实体登记相连", "尾实体平均提及次数", "边平均置信度")):
        mentions = _integer(q, r"提及次数至少\s*(\d+)")
        if mentions is not None:
            return (
                "SELECT t.relation,t.tail_type,COUNT(*) AS edge_count,COUNT(DISTINCT t.tail) "
                "AS tail_count,ROUND(AVG(e.mention_count),2) AS avg_tail_mentions,"
                "ROUND(AVG(t.confidence),4) AS avg_confidence FROM kg_triples t JOIN "
                "kg_entities e ON e.name=t.tail AND e.type=t.tail_type "
                f"WHERE e.mention_count>={mentions} GROUP BY t.relation,t.tail_type "
                "ORDER BY edge_count DESC,t.relation ASC,t.tail_type ASC"
            )
    if all(token in q for token in ("高于同地区平均值", "地区均费", "连接患者与就诊")):
        date = _date(q)
        cost = _number(q, r"费用(?:不低于|至少)\s*([0-9]+(?:\.[0-9]+)?)")
        if date and cost is not None:
            return (
                "WITH joined AS (SELECT p.patient_id,p.region,v.visit_id,v.visit_date,v.cost "
                "FROM patients p JOIN visits v ON v.patient_id=p.patient_id "
                f"WHERE v.visit_date>='{date}' AND v.cost>={cost}),region_avg AS "
                "(SELECT region,AVG(cost) AS avg_cost FROM joined GROUP BY region) SELECT "
                "j.patient_id,j.region,j.visit_id,j.visit_date,j.cost,ROUND(a.avg_cost,2) "
                "AS region_avg_cost FROM joined j JOIN region_avg a ON a.region=j.region "
                "WHERE j.cost>a.avg_cost ORDER BY j.region ASC,j.cost DESC,j.visit_id ASC"
            )
    if all(token in q for token in ("两步图谱连接", "首跳疾病到药物", "最低置信度", "Drug")):
        count = _integer(q, r"count至少\s*(\d+)")
        confidence = _number(q, r"两跳均(?:不低于|至少)\s*([0-9]+(?:\.[0-9]+)?)")
        if count is not None and confidence is not None:
            return (
                "SELECT t1.head AS disease,t1.tail AS drug,t2.relation AS second_relation,"
                "t2.tail AS target,MIN(t1.confidence,t2.confidence) AS path_confidence "
                "FROM kg_triples t1 JOIN kg_triples t2 ON t2.head=t1.tail AND "
                "t2.head_type=t1.tail_type WHERE t1.head_type='Disease' AND "
                f"t1.tail_type='Drug' AND t1.count>={count} AND t1.confidence>={confidence} "
                f"AND t2.confidence>={confidence} ORDER BY path_confidence DESC,disease ASC,"
                "drug ASC,target ASC"
            )
    return None


def _compile_kg(q: str, d: Domains) -> Optional[str]:
    relation = _ground(q, d, "kg_relation")
    if relation and all(token in q for token in ("列出知识图谱", "返回头实体", "尾类型", "次数")):
        confidence = _number(q, r"置信度(?:不低于|至少)\s*([0-9]+(?:\.[0-9]+)?)")
        if confidence is not None:
            return (
                "SELECT head,head_type,tail,tail_type,confidence,count FROM kg_triples "
                f"WHERE relation='{_esc(relation)}' AND confidence>={confidence} "
                "ORDER BY confidence DESC,count DESC,head ASC,tail ASC"
            )
    if relation and all(token in q for token in ("汇总每个头实体", "登记提及次数", "不同尾实体数")):
        return (
            "SELECT t.head,t.head_type,e.mention_count,COUNT(*) AS edge_count,"
            "COUNT(DISTINCT t.tail) AS tail_count,ROUND(AVG(t.confidence),4) AS "
            "avg_confidence,SUM(t.count) AS total_count FROM kg_triples t JOIN kg_entities e "
            "ON e.name=t.head AND e.type=t.head_type "
            f"WHERE t.relation='{_esc(relation)}' GROUP BY t.head,t.head_type,e.mention_count "
            "ORDER BY edge_count DESC,t.head ASC"
        )
    if relation and all(token in q for token in ("两跳可达路径", "第二跳不限关系", "Disease")):
        confidence = _number(q, r"两跳均须(?:达到|至少|不低于)\s*([0-9]+(?:\.[0-9]+)?)")
        if confidence is not None:
            return (
                "SELECT a.head AS disease,a.tail AS intermediate_entity,b.relation AS "
                "second_relation,b.tail AS target_entity,MIN(a.confidence,b.confidence) "
                "AS path_confidence FROM kg_triples a JOIN kg_triples b ON b.head=a.tail "
                "AND b.head_type=a.tail_type WHERE a.head_type='Disease' "
                f"AND a.relation='{_esc(relation)}' AND a.confidence>={confidence} "
                f"AND b.confidence>={confidence} ORDER BY path_confidence DESC,disease ASC,"
                "intermediate_entity ASC,target_entity ASC"
            )
    if all(token in q for token in ("入度", "出度", "相邻节点数", "Disease")):
        mentions = _integer(q, r"提及次数至少\s*(\d+)")
        if mentions is not None:
            return (
                "WITH edges AS (SELECT head AS entity,tail AS neighbor,'out' AS direction "
                "FROM kg_triples UNION ALL SELECT tail AS entity,head AS neighbor,'in' "
                "AS direction FROM kg_triples) SELECT e.name AS disease,e.mention_count,"
                "SUM(CASE WHEN x.direction='in' THEN 1 ELSE 0 END) AS in_degree,"
                "SUM(CASE WHEN x.direction='out' THEN 1 ELSE 0 END) AS out_degree,"
                "COUNT(DISTINCT x.neighbor) AS neighbor_count FROM kg_entities e LEFT JOIN "
                f"edges x ON x.entity=e.name WHERE e.type='Disease' AND e.mention_count>={mentions} "
                "GROUP BY e.name,e.mention_count ORDER BY neighbor_count DESC,e.name ASC"
            )
    if relation and all(token in q for token in ("又有其他关系出边", "目标关系边数", "其他关系边数")):
        confidence = _number(q, r"置信度至少\s*([0-9]+(?:\.[0-9]+)?)")
        if confidence is not None:
            return (
                "SELECT head,head_type,SUM(CASE WHEN relation="
                f"'{_esc(relation)}' THEN 1 ELSE 0 END) AS target_relation_edges,"
                f"SUM(CASE WHEN relation<>'{_esc(relation)}' THEN 1 ELSE 0 END) AS "
                "other_relation_edges,COUNT(DISTINCT tail) AS distinct_tails FROM kg_triples "
                f"WHERE confidence>={confidence} GROUP BY head,head_type HAVING SUM(CASE "
                f"WHEN relation='{_esc(relation)}' THEN 1 ELSE 0 END)>0 AND SUM(CASE WHEN "
                f"relation<>'{_esc(relation)}' THEN 1 ELSE 0 END)>0 ORDER BY distinct_tails DESC,head ASC"
            )
    if all(token in q for token in ("关系和头尾类型组合", "图谱覆盖度", "平均置信度")):
        count = _integer(q, r"count(?:不少于|至少)\s*(\d+)")
        if count is not None:
            return (
                "SELECT relation,head_type,tail_type,COUNT(*) AS edge_count,"
                "COUNT(DISTINCT head) AS head_count,COUNT(DISTINCT tail) AS tail_count,"
                "ROUND(AVG(confidence),4) AS avg_confidence FROM kg_triples "
                f"WHERE count>={count} GROUP BY relation,head_type,tail_type ORDER BY "
                "edge_count DESC,relation ASC,head_type ASC,tail_type ASC"
            )
    if relation and all(token in q for token in ("尾节点是否有实体登记", "作为该关系尾部的次数", "已登记")):
        mentions = _integer(q, r"提及次数(?:大于|超过)\s*(\d+)")
        if mentions is not None:
            return (
                "SELECT t.tail,t.tail_type,COUNT(*) AS incoming_edge_count,e.mention_count,"
                "ROUND(AVG(t.confidence),4) AS avg_confidence FROM kg_triples t JOIN "
                "kg_entities e ON e.name=t.tail AND e.type=t.tail_type "
                f"WHERE t.relation='{_esc(relation)}' AND e.mention_count>{mentions} GROUP BY "
                "t.tail,t.tail_type,e.mention_count ORDER BY incoming_edge_count DESC,t.tail ASC"
            )
    return None


def _compile_temporal(q: str, _d: Domains) -> Optional[str]:
    if all(token in q for token in ("相邻两次就诊间隔", "排除第一次", "上次日期")):
        days = _integer(q, r"间隔超过\s*(\d+)\s*天")
        if days is not None:
            return (
                "WITH sequenced AS (SELECT patient_id,visit_id,visit_date,cost,"
                "LAG(visit_date) OVER(PARTITION BY patient_id ORDER BY visit_date,visit_id) "
                "AS previous_visit_date FROM visits) SELECT patient_id,visit_date,"
                "previous_visit_date,CAST(julianday(visit_date)-julianday(previous_visit_date) "
                f"AS INTEGER) AS gap_days,cost FROM sequenced WHERE previous_visit_date IS NOT NULL "
                f"AND julianday(visit_date)-julianday(previous_visit_date)>{days} "
                "ORDER BY gap_days DESC,patient_id ASC,visit_date ASC"
            )
    if all(token in q for token in ("各季度", "患者数", "住院天数合计")):
        year = _year(q)
        cost = _number(q, r"费用(?:不低于|至少)\s*([0-9]+(?:\.[0-9]+)?)")
        if year and cost is not None:
            return (
                "SELECT strftime('%Y',visit_date)||'-Q'||((CAST(strftime('%m',visit_date) "
                "AS INTEGER)-1)/3+1) AS visit_quarter,COUNT(*) AS visit_count,"
                "COUNT(DISTINCT patient_id) AS patient_count,ROUND(SUM(cost),2) AS total_cost,"
                "ROUND(AVG(cost),2) AS avg_cost,SUM(los_days) AS total_los_days FROM visits "
                f"WHERE strftime('%Y',visit_date)='{year}' AND cost>={cost} GROUP BY visit_quarter "
                "ORDER BY visit_quarter ASC"
            )
    if all(token in q for token in ("首次就诊后", "复诊", "距首次天数", "不包含首次记录")):
        days = _integer(q, r"首次就诊后\s*(\d+)\s*天内")
        if days is not None:
            return (
                "WITH first_dates AS (SELECT patient_id,MIN(visit_date) AS first_visit_date "
                "FROM visits GROUP BY patient_id) SELECT v.patient_id,f.first_visit_date,"
                "v.visit_date AS return_visit_date,CAST(julianday(v.visit_date)-"
                "julianday(f.first_visit_date) AS INTEGER) AS days_since_first,v.disease,v.cost "
                "FROM visits v JOIN first_dates f ON f.patient_id=v.patient_id WHERE "
                f"v.visit_date>f.first_visit_date AND julianday(v.visit_date)-julianday(f.first_visit_date)<={days} "
                "ORDER BY v.patient_id ASC,v.visit_date ASC,v.visit_id ASC"
            )
    if all(token in q for token in ("按患者年份汇总", "上一就诊年份", "变化额")):
        cost = _number(q, r"费用(?:不低于|至少)\s*([0-9]+(?:\.[0-9]+)?)")
        if cost is not None:
            return (
                "WITH annual AS (SELECT patient_id,strftime('%Y',visit_date) AS visit_year,"
                "COUNT(*) AS visit_count,SUM(cost) AS year_cost FROM visits "
                f"WHERE cost>={cost} GROUP BY patient_id,strftime('%Y',visit_date)),compared AS "
                "(SELECT patient_id,visit_year,visit_count,year_cost,LAG(year_cost) OVER("
                "PARTITION BY patient_id ORDER BY visit_year) AS previous_year_cost FROM annual) "
                "SELECT patient_id,visit_year,visit_count,ROUND(year_cost,2) AS year_cost,"
                "ROUND(previous_year_cost,2) AS previous_year_cost,ROUND(year_cost-previous_year_cost,2) "
                "AS cost_change FROM compared ORDER BY patient_id ASC,visit_year ASC"
            )
    if all(token in q for token in ("最后一次就诊为基准", "当年次数", "rn等于1")):
        year = _year(q)
        age = _integer(q, r"年龄至少\s*(\d+)\s*岁")
        if year and age is not None:
            return (
                "WITH ranked AS (SELECT p.patient_id,p.name,v.visit_id,v.visit_date,v.disease,"
                "v.cost,ROW_NUMBER() OVER(PARTITION BY p.patient_id ORDER BY v.visit_date DESC,"
                "v.visit_id DESC) AS rn,COUNT(*) OVER(PARTITION BY p.patient_id) AS "
                "year_visit_count FROM patients p JOIN visits v ON v.patient_id=p.patient_id "
                f"WHERE strftime('%Y',v.visit_date)='{year}' AND p.age>={age}) SELECT patient_id,"
                "name,visit_date AS latest_visit_date,disease AS latest_disease,cost AS "
                "latest_cost,year_visit_count FROM ranked WHERE rn=1 ORDER BY latest_visit_date DESC,patient_id ASC"
            )
    return None


def _compile_having(q: str, _d: Domains) -> Optional[str]:
    if all(token in q for token in ("就诊次数不少于", "科室", "平均住院天数")):
        year = _year(q)
        cost = _number(q, r"费用(?:不低于|至少)\s*([0-9]+(?:\.[0-9]+)?)")
        visits = _integer(q, r"就诊次数不少于\s*(\d+)")
        if year and cost is not None and visits is not None:
            return (
                "SELECT department,COUNT(*) AS visit_count,COUNT(DISTINCT patient_id) AS "
                "patient_count,ROUND(SUM(cost),2) AS total_cost,ROUND(AVG(los_days),2) AS "
                "avg_los_days FROM visits "
                f"WHERE strftime('%Y',visit_date)='{year}' AND cost>={cost} GROUP BY department "
                f"HAVING COUNT(*)>={visits} ORDER BY visit_count DESC,department ASC"
            )
    if all(token in q for token in ("累计就诊费用超过", "至少涉及", "次均费用")):
        cost = _number(q, r"累计就诊费用超过\s*([0-9]+(?:\.[0-9]+)?)")
        patients = _integer(q, r"至少涉及\s*(\d+)\s*名患者")
        if cost is not None and patients is not None:
            return (
                "SELECT p.region,COUNT(DISTINCT v.patient_id) AS patient_count,COUNT(*) AS "
                "visit_count,ROUND(SUM(v.cost),2) AS total_cost,ROUND(AVG(v.cost),2) AS "
                "avg_cost FROM patients p JOIN visits v ON v.patient_id=p.patient_id "
                f"GROUP BY p.region HAVING SUM(v.cost)>{cost} AND COUNT(DISTINCT v.patient_id)>={patients} "
                "ORDER BY total_cost DESC,p.region ASC"
            )
    if all(token in q for token in ("种尾实体", "平均置信度", "头节点", "累计count")):
        count = _integer(q, r"count至少\s*(\d+)")
        tails = _integer(q, r"至少\s*(\d+)\s*种尾实体")
        confidence = _number(q, r"平均置信度(?:达到|至少|不低于)\s*([0-9]+(?:\.[0-9]+)?)")
        if None not in (count, tails, confidence):
            return (
                "SELECT head,head_type,COUNT(DISTINCT relation) AS relation_count,"
                "COUNT(DISTINCT tail) AS tail_count,ROUND(AVG(confidence),4) AS avg_confidence,"
                "SUM(count) AS total_count FROM kg_triples "
                f"WHERE count>={count} GROUP BY head,head_type HAVING COUNT(DISTINCT tail)>={tails} "
                f"AND AVG(confidence)>={confidence} ORDER BY tail_count DESC,head ASC"
            )
    if all(token in q for token in ("按患者统计", "总住院天数", "总费用", "至少")):
        year = _year(q)
        visits = _integer(q, r"至少\s*(\d+)\s*次")
        los = _integer(q, r"住院天数合计(?:大于|超过)\s*(\d+)\s*天")
        if year and visits is not None and los is not None:
            return (
                "SELECT p.patient_id,p.name,p.region,COUNT(*) AS visit_count,SUM(v.los_days) "
                "AS total_los_days,ROUND(SUM(v.cost),2) AS total_cost FROM patients p "
                "JOIN visits v ON v.patient_id=p.patient_id "
                f"WHERE strftime('%Y',v.visit_date)='{year}' GROUP BY p.patient_id,p.name,p.region "
                f"HAVING COUNT(*)>={visits} AND SUM(v.los_days)>{los} ORDER BY total_los_days DESC,p.patient_id ASC"
            )
    if all(token in q for token in ("累计count超过", "关系及头尾类型组合", "边数不少于")):
        confidence = _number(q, r"置信度(?:不低于|至少)\s*([0-9]+(?:\.[0-9]+)?)")
        edges = _integer(q, r"边数不少于\s*(\d+)")
        count = _integer(q, r"累计count超过\s*(\d+)")
        if None not in (confidence, edges, count):
            return (
                "SELECT relation,head_type,tail_type,COUNT(*) AS edge_count,SUM(count) AS "
                "total_count,MIN(confidence) AS min_confidence FROM kg_triples "
                f"WHERE confidence>={confidence} GROUP BY relation,head_type,tail_type "
                f"HAVING COUNT(*)>={edges} AND SUM(count)>{count} ORDER BY edge_count DESC,"
                "relation ASC,head_type ASC,tail_type ASC"
            )
    if all(token in q for token in ("按地区与年份汇总", "覆盖月份不少于", "总费用大于")):
        cost = _number(q, r"总费用(?:大于|超过)\s*([0-9]+(?:\.[0-9]+)?)")
        months = _integer(q, r"月份不少于\s*(\d+)\s*个")
        if cost is not None and months is not None:
            return (
                "SELECT p.region,strftime('%Y',v.visit_date) AS visit_year,COUNT(DISTINCT "
                "strftime('%Y-%m',v.visit_date)) AS active_months,COUNT(*) AS visit_count,"
                "COUNT(DISTINCT v.patient_id) AS patient_count,ROUND(SUM(v.cost),2) AS total_cost "
                "FROM patients p JOIN visits v ON v.patient_id=p.patient_id GROUP BY p.region,"
                f"strftime('%Y',v.visit_date) HAVING SUM(v.cost)>{cost} AND COUNT(DISTINCT "
                f"strftime('%Y-%m',v.visit_date))>={months} ORDER BY visit_year ASC,total_cost DESC,p.region ASC"
            )
    return None


def _compile_nested(q: str, _d: Domains) -> Optional[str]:
    if all(token in q for token in ("此前从未就诊", "当年首次日", "当年次数", "完整起始日期")):
        year = _year(q)
        age = _integer(q, r"年龄至少\s*(\d+)\s*岁")
        date = _date(q)
        if year and age is not None and date:
            return (
                "SELECT p.patient_id,p.name,p.region,(SELECT MIN(v.visit_date) FROM visits v "
                "WHERE v.patient_id=p.patient_id "
                f"AND strftime('%Y',v.visit_date)='{year}') AS first_visit_in_year,(SELECT "
                "COUNT(*) FROM visits v2 WHERE v2.patient_id=p.patient_id "
                f"AND strftime('%Y',v2.visit_date)='{year}') AS visits_in_year FROM patients p "
                f"WHERE p.age>={age} AND EXISTS(SELECT 1 FROM visits v3 WHERE v3.patient_id=p.patient_id "
                f"AND strftime('%Y',v3.visit_date)='{year}') AND NOT EXISTS(SELECT 1 FROM visits v4 "
                f"WHERE v4.patient_id=p.patient_id AND v4.visit_date<'{date}') "
                "ORDER BY first_visit_in_year ASC,p.patient_id ASC"
            )
    if all(token in q for token in ("同类型平均水平", "类型均值", "最高出边置信度")):
        mentions = _integer(q, r"提及次数至少\s*(\d+)")
        confidence = _number(q, r"置信度至少\s*([0-9]+(?:\.[0-9]+)?)")
        if mentions is not None and confidence is not None:
            return (
                "SELECT e.name,e.type,e.mention_count,ROUND((SELECT AVG(e2.mention_count) "
                "FROM kg_entities e2 WHERE e2.type=e.type),2) AS type_avg_mentions,"
                "(SELECT MAX(t.confidence) FROM kg_triples t WHERE t.head=e.name AND "
                "t.head_type=e.type) AS max_out_confidence FROM kg_entities e "
                f"WHERE e.mention_count>={mentions} AND e.mention_count>(SELECT AVG(e3.mention_count) "
                "FROM kg_entities e3 WHERE e3.type=e.type) AND EXISTS(SELECT 1 FROM kg_triples t2 "
                f"WHERE t2.head=e.name AND t2.head_type=e.type AND t2.confidence>={confidence}) "
                "ORDER BY e.type ASC,e.mention_count DESC,e.name ASC"
            )
    if all(token in q for token in ("没有任何低于", "至少有一条出边", "最低置信度", "Disease")):
        mentions = _integer(q, r"提及次数至少\s*(\d+)")
        confidence = _number(q, r"低于\s*([0-9]+(?:\.[0-9]+)?)\s*置信度")
        if mentions is not None and confidence is not None:
            return (
                "SELECT e.name AS disease,e.mention_count,(SELECT COUNT(*) FROM kg_triples t "
                "WHERE t.head=e.name AND t.head_type=e.type) AS outgoing_edges,(SELECT "
                "MIN(t2.confidence) FROM kg_triples t2 WHERE t2.head=e.name AND t2.head_type=e.type) "
                "AS min_out_confidence FROM kg_entities e WHERE e.type='Disease' "
                f"AND e.mention_count>={mentions} AND EXISTS(SELECT 1 FROM kg_triples t3 WHERE "
                "t3.head=e.name AND t3.head_type=e.type) AND NOT EXISTS(SELECT 1 FROM kg_triples t4 "
                f"WHERE t4.head=e.name AND t4.head_type=e.type AND t4.confidence<{confidence}) "
                "ORDER BY outgoing_edges DESC,disease ASC"
            )
    if all(token in q for token in ("同地区患者平均当年次数", "个人次数", "地区人均次数")):
        year = _year(q)
        age = _integer(q, r"年龄至少\s*(\d+)\s*岁")
        if year and age is not None:
            return (
                "SELECT p.patient_id,p.name,p.region,(SELECT COUNT(*) FROM visits v WHERE "
                f"v.patient_id=p.patient_id AND strftime('%Y',v.visit_date)='{year}') AS "
                "patient_year_visits,ROUND((SELECT AVG(x.cnt) FROM (SELECT p2.patient_id,"
                "COUNT(v2.visit_id) AS cnt FROM patients p2 LEFT JOIN visits v2 ON "
                f"v2.patient_id=p2.patient_id AND strftime('%Y',v2.visit_date)='{year}' "
                "WHERE p2.region=p.region GROUP BY p2.patient_id) x),2) AS region_avg_visits "
                f"FROM patients p WHERE p.age>={age} AND (SELECT COUNT(*) FROM visits v3 WHERE "
                f"v3.patient_id=p.patient_id AND strftime('%Y',v3.visit_date)='{year}')>(SELECT "
                "AVG(x2.cnt) FROM (SELECT p4.patient_id,COUNT(v4.visit_id) AS cnt FROM patients p4 "
                "LEFT JOIN visits v4 ON v4.patient_id=p4.patient_id "
                f"AND strftime('%Y',v4.visit_date)='{year}' WHERE p4.region=p.region GROUP BY "
                "p4.patient_id) x2) ORDER BY p.region ASC,patient_year_visits DESC,p.patient_id ASC"
            )
    if all(token in q for token in ("尾实体同时被", "种关系指向", "输出头、关系、尾")):
        relations = _integer(q, r"至少\s*(\d+)\s*种关系")
        count = _integer(q, r"count(?:不少于|至少)\s*(\d+)")
        confidence = _number(q, r"置信度(?:不低于|至少)\s*([0-9]+(?:\.[0-9]+)?)")
        if None not in (relations, count, confidence):
            return (
                "SELECT t.head,t.relation,t.tail,t.tail_type,t.confidence FROM kg_triples t "
                f"WHERE t.count>={count} AND t.confidence>={confidence} AND (SELECT "
                "COUNT(DISTINCT t2.relation) FROM kg_triples t2 WHERE t2.tail=t.tail AND "
                f"t2.tail_type=t.tail_type AND t2.confidence>={confidence})>={relations} "
                "ORDER BY t.tail_type ASC,t.tail ASC,t.relation ASC,t.head ASC"
            )
    if all(token in q for token in ("末次就诊费用高于本人当年均费", "末次记录子查询", "LIMIT取1条")):
        year = _year(q)
        age = _integer(q, r"年龄至少\s*(\d+)\s*岁")
        if year and age is not None:
            return (
                "SELECT p.patient_id,p.name,v.visit_date AS latest_visit_date,v.cost AS "
                "latest_cost,ROUND((SELECT AVG(a.cost) FROM visits a WHERE a.patient_id=p.patient_id "
                f"AND strftime('%Y',a.visit_date)='{year}'),2) AS year_avg_cost FROM patients p "
                f"JOIN visits v ON v.patient_id=p.patient_id WHERE p.age>={age} AND "
                f"strftime('%Y',v.visit_date)='{year}' AND v.visit_id=(SELECT v2.visit_id FROM "
                f"visits v2 WHERE v2.patient_id=p.patient_id AND strftime('%Y',v2.visit_date)='{year}' "
                "ORDER BY v2.visit_date DESC,v2.visit_id DESC LIMIT 1) AND v.cost>(SELECT AVG(v3.cost) "
                f"FROM visits v3 WHERE v3.patient_id=p.patient_id AND strftime('%Y',v3.visit_date)='{year}') "
                "ORDER BY latest_cost DESC,p.patient_id ASC"
            )
    return None


def _compile_cte(q: str, _d: Domains) -> Optional[str]:
    if all(token in q for token in ("分步CTE", "全期同阈值费用", "当年占比")):
        year = _year(q)
        cost = _number(q, r"费用(?:不低于|至少)\s*([0-9]+(?:\.[0-9]+)?)")
        if year and cost is not None:
            return (
                "WITH all_time AS (SELECT department,SUM(cost) AS all_cost FROM visits "
                f"WHERE cost>={cost} GROUP BY department),year_stats AS (SELECT department,"
                f"SUM(cost) AS year_cost,COUNT(*) AS year_visits FROM visits WHERE strftime('%Y',visit_date)='{year}' "
                f"AND cost>={cost} GROUP BY department) SELECT y.department,ROUND(y.year_cost,2) "
                "AS year_cost,ROUND(a.all_cost,2) AS all_time_cost,ROUND(100.0*y.year_cost/a.all_cost,2) "
                "AS year_share_pct,y.year_visits FROM year_stats y JOIN all_time a ON a.department=y.department "
                "ORDER BY year_share_pct DESC,y.department ASC"
            )
    if all(token in q for token in ("地区汇总", "各地区均值", "差额", "benchmark")):
        year = _year(q)
        age = _integer(q, r"年龄至少\s*(\d+)\s*岁")
        if year and age is not None:
            return (
                "WITH region_stats AS (SELECT p.region,COUNT(*) AS visit_count,COUNT(DISTINCT "
                "v.patient_id) AS patient_count,SUM(v.cost) AS total_cost FROM patients p JOIN visits v "
                f"ON v.patient_id=p.patient_id WHERE strftime('%Y',v.visit_date)='{year}' AND p.age>={age} "
                "GROUP BY p.region),benchmark AS (SELECT AVG(total_cost) AS avg_region_cost FROM region_stats) "
                "SELECT r.region,r.visit_count,r.patient_count,ROUND(r.total_cost,2) AS total_cost,"
                "ROUND(b.avg_region_cost,2) AS avg_region_cost,ROUND(r.total_cost-b.avg_region_cost,2) "
                "AS difference_from_avg FROM region_stats r CROSS JOIN benchmark b "
                "ORDER BY difference_from_avg DESC,r.region ASC"
            )
    if all(token in q for token in ("用CTE筛选", "边数占比", "累计count")):
        confidence = _number(q, r"置信度(?:不低于|至少)\s*([0-9]+(?:\.[0-9]+)?)")
        count = _integer(q, r"count(?:不少于|至少)\s*(\d+)")
        if confidence is not None and count is not None:
            return (
                "WITH filtered AS (SELECT * FROM kg_triples "
                f"WHERE confidence>={confidence} AND count>={count}),relation_stats AS "
                "(SELECT relation,COUNT(*) AS edge_count,SUM(count) AS total_count FROM filtered "
                "GROUP BY relation),totals AS (SELECT COUNT(*) AS all_edges FROM filtered) "
                "SELECT r.relation,r.edge_count,r.total_count,ROUND(100.0*r.edge_count/"
                "NULLIF(t.all_edges,0),2) AS edge_share_pct FROM relation_stats r CROSS JOIN totals t "
                "ORDER BY edge_share_pct DESC,r.relation ASC"
            )
    if all(token in q for token in ("月度CTE", "全年月均费用", "偏差")):
        year = _year(q)
        cost = _number(q, r"费用(?:不低于|至少)\s*([0-9]+(?:\.[0-9]+)?)")
        if year and cost is not None:
            return (
                "WITH monthly AS (SELECT strftime('%Y-%m',visit_date) AS visit_month,COUNT(*) "
                "AS visit_count,SUM(cost) AS month_cost FROM visits "
                f"WHERE strftime('%Y',visit_date)='{year}' AND cost>={cost} GROUP BY "
                "strftime('%Y-%m',visit_date)),annual_avg AS (SELECT AVG(month_cost) AS avg_month_cost "
                "FROM monthly) SELECT m.visit_month,m.visit_count,ROUND(m.month_cost,2) AS month_cost,"
                "ROUND(a.avg_month_cost,2) AS avg_month_cost,ROUND(m.month_cost-a.avg_month_cost,2) "
                "AS deviation FROM monthly m CROSS JOIN annual_avg a ORDER BY m.visit_month ASC"
            )
    if all(token in q for token in ("分两层CTE", "总费用最高的疾病", "保留并列第一")):
        cost = _number(q, r"费用(?:不低于|至少)\s*([0-9]+(?:\.[0-9]+)?)")
        if cost is not None:
            return (
                "WITH disease_cost AS (SELECT patient_id,disease,COUNT(*) AS disease_visits,"
                "SUM(cost) AS disease_cost FROM visits "
                f"WHERE cost>={cost} GROUP BY patient_id,disease),ranked AS (SELECT patient_id,"
                "disease,disease_visits,disease_cost,DENSE_RANK() OVER(PARTITION BY patient_id "
                "ORDER BY disease_cost DESC) AS cost_rank FROM disease_cost) SELECT p.patient_id,"
                "p.name,r.disease,r.disease_visits,ROUND(r.disease_cost,2) AS disease_cost,r.cost_rank "
                "FROM ranked r JOIN patients p ON p.patient_id=r.patient_id WHERE r.cost_rank=1 "
                "ORDER BY p.patient_id ASC,r.disease ASC"
            )
    if all(token in q for token in ("每种实体类型内", "提及次数", "类型实体总数", "排名不超过")):
        mentions = _integer(q, r"提及次数至少\s*(\d+)")
        rank = _integer(q, r"排名不超过\s*(\d+)")
        if mentions is not None and rank is not None:
            return (
                "WITH ranked AS (SELECT type,name,mention_count,DENSE_RANK() OVER(PARTITION BY type "
                "ORDER BY mention_count DESC) AS mention_rank,COUNT(*) OVER(PARTITION BY type) AS "
                f"type_entity_count FROM kg_entities WHERE mention_count>={mentions}) SELECT type,name,"
                f"mention_count,mention_rank,type_entity_count FROM ranked WHERE mention_rank<={rank} "
                "ORDER BY type ASC,mention_rank ASC,name ASC"
            )
    return None


def _compile_set(q: str, d: Domains) -> Optional[str]:
    if all(token in q for token in ("患者交集", "都有就诊", "患者编号、姓名和地区")):
        years = re.findall(r"(20\d{2})年", q)
        age = _integer(q, r"年龄至少\s*(\d+)\s*岁")
        if len(years) >= 2 and age is not None:
            return (
                "SELECT p.patient_id,p.name,p.region FROM patients p "
                f"WHERE p.age>={age} AND p.patient_id IN (SELECT patient_id FROM visits "
                f"WHERE strftime('%Y',visit_date)='{years[0]}' INTERSECT SELECT patient_id "
                f"FROM visits WHERE strftime('%Y',visit_date)='{years[1]}') ORDER BY p.patient_id ASC"
            )
    if all(token in q for token in ("没有就诊", "患者集合差", "地区与年龄")):
        years = re.findall(r"(20\d{2})年", q)
        age = _integer(q, r"年龄至少\s*(\d+)\s*岁")
        if len(years) >= 2 and age is not None:
            return (
                "SELECT p.patient_id,p.name,p.region,p.age FROM patients p "
                f"WHERE p.age>={age} AND p.patient_id IN (SELECT patient_id FROM visits "
                f"WHERE strftime('%Y',visit_date)='{years[0]}' EXCEPT SELECT patient_id FROM visits "
                f"WHERE strftime('%Y',visit_date)='{years[1]}') ORDER BY p.region ASC,p.patient_id ASC"
            )
    if all(token in q for token in ("作为头和尾出现", "头侧次数", "尾侧次数", "允许同名合并")):
        confidence = _number(q, r"置信度(?:不低于|至少)\s*([0-9]+(?:\.[0-9]+)?)")
        if confidence is not None:
            return (
                "WITH sides AS (SELECT head AS entity_name,1 AS head_hits,0 AS tail_hits "
                f"FROM kg_triples WHERE confidence>={confidence} UNION ALL SELECT tail AS "
                f"entity_name,0 AS head_hits,1 AS tail_hits FROM kg_triples WHERE confidence>={confidence}) "
                "SELECT entity_name,SUM(head_hits) AS head_occurrences,SUM(tail_hits) AS "
                "tail_occurrences,COUNT(*) AS total_occurrences FROM sides GROUP BY entity_name "
                "ORDER BY total_occurrences DESC,entity_name ASC"
            )
    if all(token in q for token in ("高提及实体集合", "高置信关系头节点集合", "返回名称和类型")):
        mentions = _integer(q, r"高提及阈值为\s*(\d+)")
        confidence = _number(q, r"高置信阈值为\s*([0-9]+(?:\.[0-9]+)?)")
        if mentions is not None and confidence is not None:
            return (
                "SELECT name,type FROM kg_entities "
                f"WHERE mention_count>={mentions} INTERSECT SELECT head AS name,head_type AS type "
                f"FROM kg_triples WHERE confidence>={confidence} ORDER BY type ASC,name ASC"
            )
    relations = sorted(
        {
            str(value)
            for value in d.get("kg_relation", ())
            if value and str(value) in q
        },
        key=lambda value: (-len(value), value),
    )
    if len(relations) >= 2 and all(token in q for token in ("又作为", "关系尾实体", "返回名称和类型")):
        confidence = _number(q, r"置信度(?:不低于|至少)\s*([0-9]+(?:\.[0-9]+)?)")
        if confidence is not None:
            return (
                "SELECT tail AS entity_name,tail_type AS entity_type FROM kg_triples "
                f"WHERE relation='{_esc(relations[0])}' AND confidence>={confidence} INTERSECT "
                "SELECT tail AS entity_name,tail_type AS entity_type FROM kg_triples "
                f"WHERE relation='{_esc(relations[1])}' AND confidence>={confidence} "
                "ORDER BY entity_type ASC,entity_name ASC"
            )
    if all(token in q for token in ("两个来源的标签", "患者侧", "图谱侧", "返回来源、标签和数量")):
        age = _integer(q, r"年龄至少\s*(\d+)\s*岁")
        mentions = _integer(q, r"提及次数至少\s*(\d+)")
        if age is not None and mentions is not None:
            return (
                "SELECT source,label,item_count FROM (SELECT 'patients_region' AS source,"
                "region AS label,COUNT(*) AS item_count FROM patients "
                f"WHERE age>={age} GROUP BY region UNION ALL SELECT 'entity_type' AS source,"
                f"type AS label,COUNT(*) AS item_count FROM kg_entities WHERE mention_count>={mentions} "
                "GROUP BY type) ORDER BY source ASC,item_count DESC,label ASC"
            )
    return None


def _compile_window(q: str, _d: Domains) -> Optional[str]:
    if all(token in q for token in ("稠密排名", "类型内实体数", "类型内提及总量")):
        rank = _integer(q, r"排名前\s*(\d+)")
        if rank is not None:
            return (
                "WITH ranked AS (SELECT type,name,mention_count,DENSE_RANK() OVER(PARTITION BY type "
                "ORDER BY mention_count DESC) AS mention_rank,COUNT(*) OVER(PARTITION BY type) "
                "AS entity_count_in_type,SUM(mention_count) OVER(PARTITION BY type) AS "
                "mentions_in_type FROM kg_entities) SELECT type,name,mention_count,mention_rank,"
                f"entity_count_in_type,mentions_in_type FROM ranked WHERE mention_rank<={rank} "
                "ORDER BY type ASC,mention_rank ASC,name ASC"
            )
    if all(token in q for token in ("移动平均费用", "截至当月累计就诊数", "按月汇总")):
        cost = _number(q, r"费用(?:不低于|至少)\s*([0-9]+(?:\.[0-9]+)?)")
        months = _integer(q, r"(\d+)\s*个月移动平均")
        if cost is not None and months is not None and months >= 1:
            preceding = months - 1
            return (
                "WITH monthly AS (SELECT strftime('%Y-%m',visit_date) AS visit_month,SUM(cost) "
                "AS month_cost,COUNT(*) AS visit_count FROM visits "
                f"WHERE cost>={cost} GROUP BY strftime('%Y-%m',visit_date)) SELECT visit_month,"
                "ROUND(month_cost,2) AS month_cost,visit_count,ROUND(AVG(month_cost) OVER("
                f"ORDER BY visit_month ROWS BETWEEN {preceding} PRECEDING AND CURRENT ROW),2) AS "
                f"moving_{months}_month_avg_cost,SUM(visit_count) OVER(ORDER BY visit_month ROWS "
                "BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_visits FROM monthly "
                "ORDER BY visit_month ASC"
            )
    if all(token in q for token in ("每种知识图谱关系", "按置信度和count排名", "关系内序号")):
        rank = _integer(q, r"各关系前\s*(\d+)\s*名")
        if rank is not None:
            return (
                "WITH ranked AS (SELECT relation,head,tail,confidence,count,ROW_NUMBER() OVER("
                "PARTITION BY relation ORDER BY confidence DESC,count DESC,head ASC,tail ASC) "
                "AS relation_row_number FROM kg_triples) SELECT relation,head,tail,confidence,"
                f"count,relation_row_number FROM ranked WHERE relation_row_number<={rank} "
                "ORDER BY relation ASC,relation_row_number ASC"
            )
    if all(token in q for token in ("各科室总费用与同年总盘比较", "费用占比和名次", "记录数")):
        year = _year(q)
        cost = _number(q, r"费用(?:不低于|至少)\s*([0-9]+(?:\.[0-9]+)?)")
        if year and cost is not None:
            return (
                "WITH department_stats AS (SELECT department,COUNT(*) AS visit_count,SUM(cost) "
                "AS total_cost FROM visits "
                f"WHERE strftime('%Y',visit_date)='{year}' AND cost>={cost} GROUP BY department) "
                "SELECT department,visit_count,ROUND(total_cost,2) AS total_cost,ROUND(100.0*"
                "total_cost/SUM(total_cost) OVER(),2) AS year_cost_share_pct,RANK() OVER("
                "ORDER BY total_cost DESC) AS cost_rank FROM department_stats "
                "ORDER BY cost_rank ASC,department ASC"
            )
    if all(token in q for token in ("每个头实体类型内", "不同尾节点数排名", "平均尾数")):
        confidence = _number(q, r"置信度(?:不低于|至少)\s*([0-9]+(?:\.[0-9]+)?)")
        if confidence is not None:
            return (
                "WITH head_stats AS (SELECT head_type,head,COUNT(DISTINCT tail) AS tail_count,"
                "COUNT(*) AS edge_count FROM kg_triples "
                f"WHERE confidence>={confidence} GROUP BY head_type,head) SELECT head_type,head,"
                "tail_count,edge_count,DENSE_RANK() OVER(PARTITION BY head_type ORDER BY tail_count DESC) "
                "AS tail_count_rank,ROUND(AVG(tail_count) OVER(PARTITION BY head_type),2) AS "
                "type_avg_tail_count FROM head_stats ORDER BY head_type ASC,tail_count_rank ASC,head ASC"
            )
    return None


def compile_independent_v11_sql(question: str, domains: Domains) -> Optional[str]:
    """Compile one complete v11 analytical request, or safely abstain."""

    q = re.sub(r"\s+", "", str(question or "").strip())
    if not q:
        return None
    compilers: tuple[Callable[[str, Domains], Optional[str]], ...] = (
        _compile_window,
        _compile_set,
        _compile_cte,
        _compile_nested,
        _compile_having,
        _compile_temporal,
        _compile_kg,
        _compile_joins,
        _compile_top_k,
        _compile_aggregation,
        _compile_filtering,
    )
    for compiler in compilers:
        sql = compiler(q, domains)
        if sql is not None:
            try:
                return _safe(sql)
            except ValueError:
                return None
    return None
