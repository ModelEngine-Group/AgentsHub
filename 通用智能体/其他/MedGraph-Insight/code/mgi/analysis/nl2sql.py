"""NL2SQL：自然语言 → SQL（图谱导出的 entities / relations 关系表）。

混合策略：
1. LLM 直接生成 SQL（配置 MGI_LLM_API_KEY 时，携带 Schema 提示）；
2. 语义模板解析器（确定性回退）：意图分类 + 槽位填充 → SQL。

评测：eval() 在内置评测集上执行「结果集等价」比对，报告准确率。
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA_PROMPT = """SQLite 表结构:
entities(name TEXT PRIMARY KEY, etype TEXT, etype_zh TEXT)
  -- etype ∈ Disease/Symptom/Drug/Treatment/Examination/BodyPart/Population
relations(head TEXT, relation TEXT, relation_zh TEXT, tail TEXT, head_type TEXT, tail_type TEXT, confidence REAL, evidence TEXT)
  -- relation ∈ HAS_SYMPTOM/TREATED_BY_DRUG/TREATED_BY/DIAGNOSED_BY/COMPLICATION_OF/AFFECTS/HIGH_RISK_GROUP/CONTRAINDICATED
"""

REL_KEYWORDS: list[tuple[list[str], str]] = [
    (["症状", "表现"], "HAS_SYMPTOM"),
    (["药物", "用药", "什么药"], "TREATED_BY_DRUG"),
    (["治疗方式", "疗法", "怎么治"], "TREATED_BY"),
    (["检查", "确诊", "诊断"], "DIAGNOSED_BY"),
    (["并发症", "并发"], "COMPLICATION_OF"),
    (["人群", "高发", "好发", "哪些人"], "HIGH_RISK_GROUP"),
    (["部位", "累及"], "AFFECTS"),
    (["禁忌", "禁用", "慎用"], "CONTRAINDICATED"),
]

ETYPE_KEYWORDS: list[tuple[list[str], str]] = [
    (["疾病"], "Disease"), (["症状"], "Symptom"), (["药物", "药品"], "Drug"),
    (["治疗方式"], "Treatment"), (["检查"], "Examination"),
    (["部位"], "BodyPart"), (["人群"], "Population"),
]


class NL2SQL:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        conn = sqlite3.connect(self.db_path)
        self.entity_names = [r[0] for r in conn.execute("SELECT name FROM entities")]
        conn.close()
        # 别名/英文名/缩写 → 规范名，用于把问题里的口语化实体链接到库中规范实体
        from ..kg.normalize import EntityNormalizer

        self._nz = EntityNormalizer()
        self._name_set = set(self.entity_names)

    # ---- 入口 -----------------------------------------------------------

    def to_sql(self, question: str) -> str:
        if os.getenv("MGI_LLM_API_KEY"):
            try:
                sql = self._llm_sql(question)
                if sql:
                    return sql
            except Exception:
                pass
        return self._rule_sql(question)

    def query(self, question: str) -> dict[str, Any]:
        sql = self.to_sql(question)
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        finally:
            conn.close()
        return {"question": question, "sql": sql, "columns": cols, "rows": rows}

    # ---- LLM ------------------------------------------------------------

    def _llm_sql(self, question: str) -> str:
        import requests

        r = requests.post(
            f"{os.getenv('MGI_LLM_BASE_URL', 'https://api.openai.com/v1')}/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['MGI_LLM_API_KEY']}"},
            json={
                "model": os.getenv("MGI_LLM_MODEL", "gpt-4o-mini"),
                "temperature": 0,
                "messages": [{"role": "user", "content":
                              SCHEMA_PROMPT + f"\n将问题转成一条 SQLite SQL，只输出 SQL:\n{question}"}],
            },
            timeout=60,
        )
        r.raise_for_status()
        sql = r.json()["choices"][0]["message"]["content"].strip()
        sql = sql.removeprefix("```sql").removeprefix("```").removesuffix("```").strip()
        if not sql.lower().startswith("select"):
            return ""
        return sql

    # ---- 规则语义解析 -----------------------------------------------------

    def _find_entity(self, question: str) -> str | None:
        cands = [e for e in self.entity_names if e in question]
        if cands:
            return max(cands, key=len)
        # 别名归一化：问题含"高血压"但库中规范名为"原发性高血压"
        hits = []
        for alias, canon in self._nz.canonical.items():
            if len(alias) >= 2 and alias in question and canon in self._name_set:
                hits.append(canon)
        return max(hits, key=len) if hits else None

    def _find_entities(self, question: str) -> list[str]:
        """多条件查询：找出问题中出现的全部实体（如"胸痛且心悸"）。"""
        found = {e for e in self.entity_names if e in question}
        for alias, canon in self._nz.canonical.items():
            if len(alias) >= 2 and alias in question and canon in self._name_set:
                found.add(canon)
        return sorted(found, key=len, reverse=True)

    _TAILTYPE_REL = {"Symptom": "HAS_SYMPTOM", "Drug": "TREATED_BY_DRUG",
                     "Treatment": "TREATED_BY", "Examination": "DIAGNOSED_BY",
                     "Population": "HIGH_RISK_GROUP", "BodyPart": "AFFECTS"}

    def _etype_of(self, name: str) -> str | None:
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT etype FROM entities WHERE name=?", (name,)).fetchone()
        finally:
            conn.close()
        return row[0] if row else None

    def _infer_relation_by_tail(self, name: str) -> str | None:
        et = self._etype_of(name)
        return self._TAILTYPE_REL.get(et) if et else None

    def _is_tail(self, name: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        try:
            hit = conn.execute("SELECT 1 FROM relations WHERE tail=? LIMIT 1", (name,)).fetchone()
        finally:
            conn.close()
        return hit is not None

    def _find_relation(self, question: str) -> str | None:
        for kws, rel in REL_KEYWORDS:
            if any(k in question for k in kws):
                return rel
        return None

    def _find_etype(self, question: str) -> str | None:
        for kws, et in ETYPE_KEYWORDS:
            if any(k in question for k in kws):
                return et
        return None

    def _rule_sql(self, question: str) -> str:
        q = question
        entity = self._find_entity(q)
        rel = self._find_relation(q)
        etype = self._find_etype(q)
        counting = any(k in q for k in ["多少", "几个", "几种", "数量", "统计"])
        count_strict = any(k in q for k in ["多少", "几个", "几种"])
        topn = re.search(r"(?:前|top\s*)(\d+)", q, re.IGNORECASE)
        most = any(k in q for k in ["最多", "最常", "最高"])

        # (A) 多条件交集：同时具有多个尾实体的疾病（"既有胸痛又有心悸的疾病"）
        both = any(k in q for k in ["同时", "既有", "又有", "并且", "而且", "都有"])
        if both:
            found = self._find_entities(q)
            head = next((e for e in found if self._etype_of(e) == "Disease"), None)
            tails = [e for e in found if e != head and self._is_tail(e)]
            if len(tails) >= 2:
                mrel = rel or self._infer_relation_by_tail(tails[0])
                cond = f"relation='{mrel}' AND " if mrel else ""
                sub = " INTERSECT ".join(
                    f"SELECT head FROM relations WHERE {cond}tail='{t}'" for t in tails)
                return f"SELECT head FROM ({sub}) ORDER BY head"

        # (C) 置信度过滤：("置信度大于0.8的关系") —— 需在聚合阈值之前，避免小数被误判
        confm = re.search(r"置信度\s*(?:大于|超过|高于|不低于|>=?|≥)?\s*(0?\.\d+|1(?:\.0+)?)", q)
        if confm:
            thv = float(confm.group(1))
            base = (f"SELECT head, relation_zh, tail, confidence FROM relations "
                    f"WHERE confidence >= {thv}")
            if rel:
                base += f" AND relation='{rel}'"
            if entity:
                base += f" AND head='{entity}'"
            return base + " ORDER BY confidence DESC, head"

        # (B) 聚合 + HAVING：某关系尾实体数超过阈值的疾病（"症状超过3个的疾病"）
        thr = re.search(r"(?:超过|大于|多于|至少|不少于|≥|>=?)\s*(\d+)(?!\s*\.)", q)
        if rel and thr and "置信度" not in q:
            op = ">=" if any(k in q for k in ["至少", "≥", "不少于"]) else ">"
            return (f"SELECT head, COUNT(DISTINCT tail) AS cnt FROM relations "
                    f"WHERE relation='{rel}' GROUP BY head HAVING cnt {op} {thr.group(1)} "
                    f"ORDER BY cnt DESC, head")

        # (E) 反向查询：哪些疾病有某症状/用某药/会并发某病（须先于"实体+关系"正向分支）
        if entity and any(k in q for k in ["哪些疾病", "什么病", "什么疾病", "哪种疾病", "哪些病"]):
            return f"SELECT DISTINCT head FROM relations WHERE tail = '{entity}' ORDER BY head"

        # (D) JOIN：实体及其尾实体的类型信息（"疾病及其药物属于什么类型"）
        if entity and rel and any(k in q for k in ["类型", "类别", "归类", "属于"]):
            return (f"SELECT r.tail, e.etype_zh FROM relations r "
                    f"JOIN entities e ON r.tail = e.name "
                    f"WHERE r.head='{entity}' AND r.relation='{rel}' ORDER BY r.tail")

        # 某类型实体的数量（"多少种药物"）
        if count_strict and etype and entity is None:
            return f"SELECT COUNT(*) AS cnt FROM entities WHERE etype = '{etype}'"
        # 分布统计（"统计各类实体/关系的数量"）
        if counting and entity is None and topn is None and not most:
            if "关系" in q:
                return "SELECT relation_zh, COUNT(*) AS cnt FROM relations GROUP BY relation_zh ORDER BY cnt DESC"
            if etype is None and rel is None:
                return "SELECT etype_zh, COUNT(*) AS cnt FROM entities GROUP BY etype_zh ORDER BY cnt DESC"
        # 某类型实体列表
        if etype and entity is None and rel is None:
            return f"SELECT name FROM entities WHERE etype = '{etype}' ORDER BY name"
        # 哪个疾病的 X 最多 / TopN 排行
        if rel and (most or topn) and entity is None:
            limit = int(topn.group(1)) if topn else (1 if most else 10)
            return (f"SELECT head, COUNT(*) AS cnt FROM relations WHERE relation = '{rel}' "
                    f"GROUP BY head ORDER BY cnt DESC LIMIT {limit}")
        # 实体 + 关系：查尾实体
        if entity and rel:
            if counting:
                return (f"SELECT COUNT(DISTINCT tail) AS cnt FROM relations "
                        f"WHERE head = '{entity}' AND relation = '{rel}'")
            return (f"SELECT DISTINCT tail FROM relations "
                    f"WHERE head = '{entity}' AND relation = '{rel}' ORDER BY tail")
        # 实体的全部关系
        if entity:
            return (f"SELECT relation_zh, tail, confidence FROM relations "
                    f"WHERE head = '{entity}' ORDER BY relation, tail")
        # 关系分布
        if rel:
            return (f"SELECT head, tail FROM relations WHERE relation = '{rel}' ORDER BY head, tail")
        return "SELECT relation_zh, COUNT(*) AS cnt FROM relations GROUP BY relation_zh ORDER BY cnt DESC"


# ---- 评测 -----------------------------------------------------------------

def eval_accuracy(nl2sql: NL2SQL, eval_path: str | Path) -> dict[str, Any]:
    """结果集等价（execution accuracy）评测。"""
    cases = [json.loads(ln) for ln in Path(eval_path).read_text(encoding="utf-8").strip().splitlines()]
    conn = sqlite3.connect(nl2sql.db_path)
    correct = 0
    failures: list[dict[str, Any]] = []
    for case in cases:
        pred_sql = nl2sql.to_sql(case["question"])
        try:
            pred = sorted(map(tuple, conn.execute(pred_sql).fetchall()))
        except Exception as exc:  # SQL 执行失败视为错误
            failures.append({"question": case["question"], "pred_sql": pred_sql, "error": str(exc)})
            continue
        gold = sorted(map(tuple, conn.execute(case["gold_sql"]).fetchall()))
        if pred == gold:
            correct += 1
        else:
            failures.append({"question": case["question"], "pred_sql": pred_sql,
                             "gold_sql": case["gold_sql"]})
    conn.close()
    return {
        "total": len(cases),
        "correct": correct,
        "accuracy": round(correct / len(cases), 4) if cases else 0.0,
        "failures": failures,
    }
