"""NL2SQL engine targeting the SQLite analytics DB.

Pipeline (each step is a known accuracy lever; they stack):
  1. Schema injection + dynamic few-shot (most-similar (question, SQL) examples).
  2. LLM generates a single SELECT (Qwen3.5).
  3. Read-only guard: parse to an AST and require a single projection statement with
     no DDL/DML anywhere in the tree (see `medigraph.analysis.sql_guard`). The
     executor then adds a read-only connection, `PRAGMA query_only` and a SQLite
     authorizer, so a parser gap still cannot become a write.
  4. Execution on SQLite; on error, feed the DB error back and re-generate
     (execution-guided self-correction), up to `max_correction` rounds.

Returns the SQL, the executed columns/rows, and a trace of attempts.
"""
from __future__ import annotations

import re
import sqlite3
import time
from typing import Any

from medigraph.analysis.relational import schema_text
from medigraph.analysis.schema_linking import MedicalSchemaLinker
from medigraph.analysis.sql_guard import ensure_read_only, is_read_only

_SYSTEM = "你是资深数据分析师，只输出可在 SQLite 上执行的单条 SELECT 语句，不要解释、不要 markdown。"

# Curated few-shot pool (question -> gold SQL). Dynamic selection picks the most
# lexically similar ones to inject, DAIL-SQL style.
_FEWSHOT: list[tuple[str, str]] = [
    ("每个科室的就诊量是多少",
     "SELECT department, COUNT(*) AS visits FROM patient_visits GROUP BY department ORDER BY visits DESC"),
    ("高血压患者的平均年龄是多少",
     "SELECT AVG(age) AS avg_age FROM patient_visits WHERE disease = '高血压'"),
    ("各疾病的就诊人次按从多到少排序",
     "SELECT disease, COUNT(*) AS cnt FROM patient_visits GROUP BY disease ORDER BY cnt DESC"),
    ("2024年每个月的就诊量趋势",
     "SELECT substr(visit_date,1,7) AS month, COUNT(*) AS visits FROM patient_visits GROUP BY month ORDER BY month"),
    ("开具次数最多的5种药物",
     "SELECT drug, COUNT(*) AS cnt FROM prescriptions GROUP BY drug ORDER BY cnt DESC LIMIT 5"),
    ("男性和女性的就诊比例",
     "SELECT gender, COUNT(*) AS cnt FROM patient_visits GROUP BY gender"),
    ("检查结果异常率最高的检查项目",
     "SELECT test_name, AVG(abnormal) AS abnormal_rate FROM lab_tests GROUP BY test_name ORDER BY abnormal_rate DESC"),
    ("内科的平均就诊费用",
     "SELECT AVG(cost) AS avg_cost FROM patient_visits WHERE department = '内科'"),
]


class NL2SQL:
    def __init__(
        self,
        db_path: str,
        llm: Any | None = None,
        max_correction: int = 2,
        num_shots: int = 4,
        backend: str | None = None,
    ):
        """`backend`: 'sqlite' (default) or 'postgres'.

        Generation is engine-agnostic (prompt + templates emit SQLite-flavoured
        SQL); on the postgres backend the vetted statement is transpiled via
        sqlglot AST and executed on the pooled server. Schema linking always
        reads the SQLite file -- it is the vocabulary source and exists offline.
        """
        self.db_path = db_path
        if llm is None:
            from medigraph.llm.client import LLMClient
            llm = LLMClient()
        self.llm = llm
        self.max_correction = max_correction
        self.num_shots = num_shots
        self.schema_linker = MedicalSchemaLinker(db_path)
        if backend is None:
            from config.settings import get_analytics_config
            backend = get_analytics_config().backend
        self.backend = backend if backend in {"sqlite", "postgres"} else "sqlite"

    # ------------------------------------------------------------------ #
    def _select_shots(self, question: str) -> list[tuple[str, str]]:
        """Lexical-overlap ranking of few-shot examples (lightweight schema/intent linking)."""
        q = set(question)
        scored = sorted(_FEWSHOT, key=lambda ex: len(q & set(ex[0])), reverse=True)
        return scored[: self.num_shots]

    def _build_prompt(self, question: str, error: str = "", prev_sql: str = "") -> str:
        links = self.schema_linker.link(question)
        shots = "\n".join(f"问题：{q}\nSQL：{s}" for q, s in self._select_shots(question))
        fix = ""
        if error:
            fix = (
                f"\n上一次生成的 SQL 执行报错，请修正：\n上次SQL：{prev_sql}\n报错：{error}\n"
                "请输出修正后的 SQL。"
            )
        return (
            f"数据库 schema：\n{schema_text()}\n\n"
            f"Schema-Linking 结果（已用知识图谱词表消歧）：\n{links}\n\n"
            f"参考示例：\n{shots}\n\n"
            f"请为下面的问题生成一条 SQLite SELECT 语句（只输出 SQL）：\n问题：{question}{fix}"
        )

    @staticmethod
    def _extract_sql(text: str) -> str:
        if not text:
            return ""
        m = re.search(r"```(?:sql)?\s*(.+?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1)
        text = text.strip().rstrip(";").strip()
        # keep from the first SELECT/WITH
        m2 = re.search(r"\b(SELECT|WITH)\b", text, flags=re.IGNORECASE)
        return text[m2.start():].strip() if m2 else text

    @staticmethod
    def _superlative_order(q: str, explicit_n: int | None) -> tuple[str, str]:
        """(direction, limit_clause) for a ranking ORDER BY.

        Handles the asymmetry between "最高/最多/最大" (DESC) and "最低/最少/
        最小" (ASC) -- the three ranking branches below used to hard-code DESC
        unconditionally, so "平均就诊费用最低的科室是哪个" silently returned the
        *highest*-cost department first with no LIMIT at all (the full ranked
        list), which happened to still contain the right answer buried inside
        it but is not what "是哪个" (asking for one) was asking for.

        limit_clause is empty for a plain "各X的..." listing (no superlative,
        no explicit top-N -- return every group), "LIMIT 1" for a bare
        superlative with no explicit count ("最低的是哪个"), or "LIMIT N" when
        the question gave one ("前3个").
        """
        ascending = any(w in q for w in ("最低", "最少", "最小"))
        descending = any(w in q for w in ("最高", "最多", "最大", "top", "前"))
        direction = "ASC" if ascending else "DESC"
        if explicit_n is not None:
            return direction, f" LIMIT {explicit_n}"
        if ascending or descending:
            return direction, " LIMIT 1"
        return direction, ""

    @staticmethod
    def _is_readonly(sql: str) -> bool:
        """Structural read-only check (see `medigraph.analysis.sql_guard`).

        Kept as a method for backwards compatibility; the decision itself is an AST
        walk rather than a keyword blacklist.
        """
        return is_read_only(sql)

    def _execute(self, sql: str, timeout_seconds: float = 5.0, max_rows: int = 5000) -> tuple[list[str], list[tuple], str]:
        if self.backend == "postgres":
            from medigraph.analysis.pg_relational import execute_readonly_pg
            from medigraph.analysis.sql_guard import transpile

            try:
                pg_sql = transpile(sql, write="postgres")
            except Exception as exc:  # noqa: BLE001 - feed back into self-correction
                return [], [], f"dialect transpile failed: {exc}"
            return execute_readonly_pg(pg_sql, timeout_seconds=timeout_seconds, max_rows=max_rows)
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            conn.execute("PRAGMA query_only=ON")
            denied = {
                sqlite3.SQLITE_INSERT,
                sqlite3.SQLITE_UPDATE,
                sqlite3.SQLITE_DELETE,
                sqlite3.SQLITE_CREATE_INDEX,
                sqlite3.SQLITE_CREATE_TABLE,
                sqlite3.SQLITE_CREATE_TEMP_INDEX,
                sqlite3.SQLITE_CREATE_TEMP_TABLE,
                sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
                sqlite3.SQLITE_CREATE_TEMP_VIEW,
                sqlite3.SQLITE_CREATE_TRIGGER,
                sqlite3.SQLITE_CREATE_VIEW,
                sqlite3.SQLITE_DROP_INDEX,
                sqlite3.SQLITE_DROP_TABLE,
                sqlite3.SQLITE_DROP_TEMP_INDEX,
                sqlite3.SQLITE_DROP_TEMP_TABLE,
                sqlite3.SQLITE_DROP_TEMP_TRIGGER,
                sqlite3.SQLITE_DROP_TEMP_VIEW,
                sqlite3.SQLITE_DROP_TRIGGER,
                sqlite3.SQLITE_DROP_VIEW,
                sqlite3.SQLITE_ALTER_TABLE,
                sqlite3.SQLITE_ATTACH,
                sqlite3.SQLITE_DETACH,
            }
            conn.set_authorizer(
                lambda action, arg1, arg2, db, source: (
                    sqlite3.SQLITE_DENY if action in denied else sqlite3.SQLITE_OK
                )
            )
            started = time.monotonic()
            conn.set_progress_handler(
                lambda: 1 if time.monotonic() - started > timeout_seconds else 0,
                10_000,
            )
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchmany(max_rows + 1)
            if len(rows) > max_rows:
                conn.close()
                return [], [], f"result exceeds safety limit ({max_rows} rows)"
            cols = [d[0] for d in cur.description] if cur.description else []
            conn.close()
            return cols, rows, ""
        except Exception as exc:  # noqa: BLE001 - surface SQL errors for self-correction
            return [], [], str(exc)

    @staticmethod
    def _router_should_defer(q: str) -> bool:
        """Conservative gate: the deterministic router only expresses single-table
        group/filter/aggregate SQL. Questions needing subqueries, cross-table
        JOINs, HAVING or a column-specific SUM are handed to the LLM instead of
        being answered with structurally-wrong template SQL.
        """
        # 1) comparison against an aggregate -> needs a subquery
        if re.search(r"(高于|超过|大于|低于|小于|多于|少于).{0,5}(平均|均值|全院|平均值|中位)", q):
            return True
        # 2) HAVING / distinct-count combinations
        if "及以上" in q or re.search(r"(两|二|三|四|五|多)\s*种", q):
            return True
        if re.search(r"不同.{0,4}(检验|检查|药|项目|科室|病人|患者)", q) or re.search(r"(病人|患者).{0,3}不同", q):
            return True
        # distinct COUNT the router emits as COUNT(*)
        if "不同" in q and any(t in q for t in ("多少", "几", "数量", "个数")):
            return True
        # attribute-of-the-extreme-row (e.g. "花费最高的那次就诊是什么疾病") -> the
        # router wrongly groups; this needs ORDER BY <num> LIMIT 1 then project attr
        if re.search(r"(最高|最贵|最低|最少|最大|最小|最多).{0,6}(那次|哪次|这次|一次|一位|那位|哪位)", q):
            return True
        # 3) column-specific SUM the router does not emit
        if re.search(r"总(天数|时长|用药天数|住院天数)", q):
            return True
        # 4) cross-table: a drug/prescription term together with a patient-visit
        #    attribute or a disease filter -> needs prescriptions JOIN patient_visits
        drug_ctx = any(t in q for t in ("药", "处方", "开具", "用药", "服用"))
        attr_ctx = any(t in q for t in ("年龄", "岁", "费用", "花费", "住院费"))
        disease_ctx = bool(re.search(r"[一-鿿]{2,8}(病|症|癌|炎)", q))
        if drug_ctx and (attr_ctx or disease_ctx):
            return True
        # 5) lab abnormality asking for a disease -> needs lab_tests JOIN patient_visits
        if any(t in q for t in ("异常", "检验", "检查")) and ("疾病" in q or "最常见" in q):
            return True
        # 6) Nth-rank ("第二多", "倒数第三", "排第二") -> needs LIMIT/OFFSET the
        #    router has no template for; a naive match would silently return the
        #    unranked full list instead of the Nth row.
        if re.search(r"第\s*[一二三四五六七八九十\d]+\s*(多|少|高|低|大|小|名|位)", q) or "倒数第" in q:
            return True
        # 7) exclusion/negation ("除了X以外", "不包括X", "排除X") -> the router's
        #    group/filter templates have no NOT-filter support; matching one
        #    anyway would silently drop the exclusion and answer the un-excluded
        #    question instead.
        if re.search(r"除了.{0,10}(以外|之外)", q) or any(t in q for t in ("不包括", "排除", "不含")):
            return True
        # 8) half-year/quarter date phrases -- the router's date handling only
        #    covers an exact "YYYY-MM" equality or an unfiltered month-trend
        #    GROUP BY; a range like "2024年上半年" combined with another filter
        #    (e.g. a disease) would otherwise silently drop the date qualifier.
        if any(t in q for t in ("上半年", "下半年", "季度")):
            return True
        # 9) HAVING-style count threshold asking for a *list*, not a scalar
        #    ("超过50次的科室有哪些") -- distinct from the "及以上"/quantifier
        #    check above, which catches a different phrasing.
        if re.search(r"(超过|多于|达到|不少于|至少)\s*\d+\s*(次|人次)", q) and any(
            t in q for t in ("哪些", "哪几个", "有哪")
        ):
            return True
        return False

    def _deterministic_sql(self, question: str) -> tuple[str, dict]:
        """Cover common analytics intents without an LLM call."""
        links = self.schema_linker.link(question)
        q = links["normalized_question"].lower()
        if self._router_should_defer(q):
            return "", links
        values = {item["field"]: item["value"] for item in links["values"]}
        department_match = re.search(r"([\u4e00-\u9fff]{0,6}(?:内科|外科|全科))", q)
        if department_match and not any(token in department_match.group(1) for token in ("每个", "各个")):
            values["department"] = department_match.group(1)
        # Regex fallback for a drug name not already resolved via the (case-exact)
        # vocabulary literal match above. Only fills the gap -- it must not
        # overwrite a real match with its own capture, because `q` is lower-cased
        # and Python's str.lower() *does* remap Unicode Roman numerals (Ⅰ -> ⅰ,
        # U+2160 -> U+2170), a suffix pattern common in Chinese drug names
        # ("硝苯地平缓释片Ⅰ"); overwriting would silently query for a drug that
        # cannot exist in the database, without the query claiming to be exact.
        if "drug" not in values:
            drug_match = re.search(r"(.{2,20}?)(?:一共|总共)?被开具了?多少次", q)
            if drug_match:
                values["drug"] = drug_match.group(1).strip()
        graph_value = values.get("kg_entity")
        if graph_value:
            if any(term in q for term in ("药", "处方", "开具")):
                values.setdefault("drug", graph_value)
            elif any(term in q for term in ("检查", "检验")):
                values.setdefault("test_name", graph_value)
            elif any(term in q for term in ("科室", "挂号", "内科", "外科")):
                values.setdefault("department", graph_value)
            else:
                values.setdefault("disease", graph_value)
        top_n = self.schema_linker.extract_top_n(q)

        # Knowledge-graph aggregate analysis.  ``kg_triples`` is a read-only
        # relational mirror of the Task-2 graph and is the right source for
        # rankings/counts across many graph entities.  Keep this deterministic
        # so the recording prompt works without LLM retries or prompt
        # reformulation.
        kg_context = any(
            term in q
            for term in ("知识图谱", "图谱", "kg_triples", "has_symptom", "关联症状")
        )
        aggregate_context = any(
            term in q
            for term in ("统计", "数量", "计数", "最多", "最少", "排名", "排序", "top", "前")
        )
        symptom_context = any(term in q for term in ("症状", "has_symptom"))
        if kg_context and aggregate_context and symptom_context:
            return (
                "SELECT head, COUNT(DISTINCT tail) AS symptom_count "
                "FROM kg_triples "
                "WHERE head_type = 'Disease' AND relation = 'has_symptom' "
                "GROUP BY head ORDER BY symptom_count DESC "
                f"LIMIT {top_n}",
                links,
            )

        # Highest-cost patient/visit ranking used by the recording script.
        # Keep it deterministic so this core NL2SQL demo remains available
        # even when the optional external LLM quota is exhausted.
        #
        # Guarded against "平均"/"科室" so this individual-*visit* ranking does
        # not also swallow a department-*aggregate* question: "平均就诊费用最高
        # 的科室是哪个" contains 费用+最高+就诊 (matching the three checks above)
        # but is asking which department has the highest *average*, which is the
        # grouped branch further below -- this branch used to win the race and
        # silently return a list of (patient_id, cost) instead.
        if (
            any(term in q for term in ("费用", "住院费", "花费", "cost"))
            and any(term in q for term in ("最高", "最多", "top", "前"))
            and any(term in q for term in ("患者", "病人", "就诊"))
            and "平均" not in q
            and "科室" not in q
        ):
            return (
                "SELECT patient_id, cost FROM patient_visits "
                f"ORDER BY cost DESC LIMIT {top_n}",
                links,
            )

        # Time series.
        if (
            any(term in q for term in ("每月", "每个月", "月度", "月份", "各月"))
            and any(term in q for term in ("就诊", "人次", "患者"))
        ):
            return (
                "SELECT substr(visit_date,1,7) AS month, COUNT(*) AS visits "
                "FROM patient_visits GROUP BY month ORDER BY month",
                links,
            )
        # Lab analysis. A specific test_name filter must be checked *before* the
        # generic per-test_name GROUP BY below -- otherwise "血压检查的异常次数
        # 是多少" (asking about one named test) matches the same "异常"+"检查"
        # keywords as "每个检查项目的异常次数" (asking for all of them grouped)
        # and silently answers the wrong, unfiltered question.
        if "test_name" in values and "异常" in q:
            return (
                f"SELECT SUM(abnormal) AS abnormal_count FROM lab_tests "
                f"WHERE test_name={self.schema_linker.quoted(values['test_name'])}",
                links,
            )
        if "异常" in q and ("率" in q or "比例" in q):
            return (
                "SELECT test_name, AVG(abnormal) AS abnormal_rate FROM lab_tests "
                "GROUP BY test_name ORDER BY abnormal_rate DESC",
                links,
            )
        if "异常" in q and ("检查" in q or "项目" in q):
            return (
                "SELECT test_name, SUM(abnormal) AS abnormal_count FROM lab_tests "
                "GROUP BY test_name ORDER BY abnormal_count DESC",
                links,
            )
        # Prescription analysis.
        if ("药物" in q or "开具" in q or "处方" in q) and any(term in q for term in ("最多", "最高", "top", "前")):
            return (
                f"SELECT drug, COUNT(*) AS cnt FROM prescriptions GROUP BY drug "
                f"ORDER BY cnt DESC LIMIT {top_n}",
                links,
            )
        if "drug" in values and any(term in q for term in ("多少", "次数", "几次", "开具")):
            return (
                f"SELECT COUNT(*) AS cnt FROM prescriptions WHERE drug={self.schema_linker.quoted(values['drug'])}",
                links,
            )
        # Grouped visit metrics. These templates express exactly one dimension
        # (GROUP BY {group}); if the question also carries a resolved filter
        # value for a *different* field (e.g. "接诊高血压病人最多的前三个科室"
        # groups by department but also names a specific disease), the template
        # cannot express both and silently answering with the filter dropped is
        # worse than deferring -- so `group` is cleared and the LLM path handles
        # the compound question instead.
        group = ""
        if "科室" in q:
            group = "department"
        elif "疾病" in q or "病种" in q:
            group = "disease"
        elif "性别" in q or ("男性" in q and "女性" in q) or "男女" in q:
            group = "gender"
        if group and any(field in values for field in ("disease", "department", "drug") if field != group):
            group = ""
        explicit_n = self.schema_linker.explicit_top_n(q)
        if group and any(term in q for term in ("平均费用", "平均就诊费用")):
            direction, limit = self._superlative_order(q, explicit_n)
            return (
                f"SELECT {group}, AVG(cost) AS avg_cost FROM patient_visits "
                f"GROUP BY {group} ORDER BY avg_cost {direction}{limit}",
                links,
            )
        # "平均就诊年龄" is the same request as "平均年龄" but the inserted 就诊
        # used to miss this check, fall through to the visit-count branch below
        # (which matches on the bare "多少") and silently answer with a COUNT(*)
        # ranking instead of an average age -- a different metric, not a
        # degraded one. The evaluation sets never caught it because their one
        # such question spelled "平均年龄" out in a trailing clause.
        if group and any(term in q for term in ("平均年龄", "平均就诊年龄", "平均患者年龄")):
            direction, limit = self._superlative_order(q, explicit_n)
            return (
                f"SELECT {group}, AVG(age) AS avg_age FROM patient_visits "
                f"GROUP BY {group} ORDER BY avg_age {direction}{limit}",
                links,
            )
        if group and any(
            term in q for term in ("就诊量", "人次", "多少", "数量", "次数", "排名", "最高", "最多", "最少")
        ):
            direction, limit = self._superlative_order(q, explicit_n)
            return (
                f"SELECT {group}, COUNT(*) AS cnt FROM patient_visits "
                f"GROUP BY {group} ORDER BY cnt {direction}{limit}",
                links,
            )
        # Filtered scalar visit metrics.
        filters = []
        if "disease" in values:
            filters.append(f"disease={self.schema_linker.quoted(values['disease'])}")
        if "department" in values:
            filters.append(f"department={self.schema_linker.quoted(values['department'])}")
        if "女性" in q or re.search(r"(?<!男)女(?:性|患者)", q):
            filters.append("gender='女'")
        elif "男性" in q or re.search(r"男(?:性|患者)", q):
            filters.append("gender='男'")
        # Two-sided range ("30到50岁之间", "30-50岁") must be checked before the
        # one-sided pattern below: e.g. "50岁以下" alone matches the one-sided
        # regex on its own, but "30到50岁之间" has no 以上/大于/超过/以下/小于/
        # 不满 keyword at all, so without this branch the filter is silently
        # dropped and the query answers "how many visits total" instead.
        range_match = re.search(r"(\d+)\s*(?:到|至|[-~])\s*(\d+)\s*岁", q)
        age_match = re.search(r"(\d+)\s*岁\s*(以上|大于|超过|以下|小于|不满)", q)
        if range_match:
            low, high = sorted((int(range_match.group(1)), int(range_match.group(2))))
            filters.append(f"age BETWEEN {low} AND {high}")
        elif age_match:
            operator = ">" if age_match.group(2) in {"以上", "大于", "超过"} else "<"
            filters.append(f"age{operator}{int(age_match.group(1))}")
        where = " WHERE " + " AND ".join(filters) if filters else ""
        if any(term in q for term in ("平均年龄", "平均就诊年龄", "平均患者年龄")):
            return f"SELECT AVG(age) AS avg_age FROM patient_visits{where}", links
        if "平均" in q and ("费用" in q or "花费" in q):
            return f"SELECT AVG(cost) AS avg_cost FROM patient_visits{where}", links
        if ("总费用" in q or "费用总计" in q) and not filters:
            return "SELECT SUM(cost) AS total_cost FROM patient_visits", links
        if any(term in q for term in ("多少次就诊", "多少人次", "人次是多少", "就诊总数", "总共有多少", "就诊次数")):
            return f"SELECT COUNT(*) AS cnt FROM patient_visits{where}", links
        return "", links

    # ------------------------------------------------------------------ #
    def query(self, question: str) -> dict:
        """NL question -> {sql, columns, rows, error, attempts}."""
        attempts: list[dict] = []
        deterministic_sql, links = self._deterministic_sql(question)
        if deterministic_sql:
            cols, rows, error = self._execute(deterministic_sql)
            attempts.append(
                {
                    "sql": deterministic_sql,
                    "error": error,
                    "source": "deterministic_template",
                }
            )
            if not error:
                return {
                    "question": question,
                    "normalized_question": links["normalized_question"],
                    "schema_links": links,
                    "sql": deterministic_sql,
                    "columns": cols,
                    "rows": rows,
                    "error": "",
                    "attempts": attempts,
                    "generation_mode": "deterministic_template",
                }
        error = prev_sql = ""
        for _ in range(self.max_correction + 1):
            raw = self.llm.chat(self._build_prompt(question, error, prev_sql), system=_SYSTEM, temperature=0.0)
            sql = self._extract_sql(raw)
            readonly, reason = ensure_read_only(sql)
            if not readonly:
                # Feed the *specific* structural reason back rather than a generic
                # refusal: "forbidden construct: Delete" is actionable for the model,
                # "not allowed" is not.
                error, prev_sql = f"rejected by read-only guard: {reason}", sql
                attempts.append({"sql": sql, "error": error, "source": "llm"})
                continue
            cols, rows, err = self._execute(sql)
            attempts.append({"sql": sql, "error": err, "source": "llm"})
            if not err:
                return {"question": question, "sql": sql, "columns": cols,
                        "rows": rows, "error": "", "attempts": attempts,
                        "schema_links": links, "generation_mode": "llm"}
            error, prev_sql = err, sql
        return {"question": question, "sql": prev_sql, "columns": [], "rows": [],
                "error": error, "attempts": attempts, "schema_links": links,
                "generation_mode": "llm_failed"}
