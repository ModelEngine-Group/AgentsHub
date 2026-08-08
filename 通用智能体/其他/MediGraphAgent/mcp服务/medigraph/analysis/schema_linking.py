"""Graph-aware schema/value linking for medical NL2SQL."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from medigraph.analysis.numerals import normalize, normalize_numerals, numeral_bearing

_TABLE_TERMS = {
    "patient_visits": ("就诊", "患者", "疾病", "科室", "年龄", "性别", "费用", "日期", "人次"),
    "prescriptions": ("处方", "药物", "用药", "开具", "天数"),
    "lab_tests": ("检查", "检验", "异常", "项目"),
    "kg_entities": ("实体", "类型", "知识图谱"),
    "kg_triples": ("关系", "三元组", "置信度", "来源", "图谱"),
}

_COLUMN_TERMS = {
    "age": ("年龄", "岁"),
    "gender": ("性别", "男性", "女性", "男", "女"),
    "disease": ("疾病", "病种", "诊断"),
    "department": ("科室", "挂号"),
    "visit_date": ("日期", "时间", "月份", "季度", "年度"),
    "cost": ("费用", "花费", "金额"),
    "drug": ("药物", "用药", "处方", "开具"),
    "days": ("天数", "疗程"),
    "test_name": ("检查", "检验", "项目"),
    "abnormal": ("异常", "阳性"),
}

_MEDICAL_ALIASES = {
    "糖网": "糖尿病视网膜病变",
    "糖肾": "糖尿病肾病",
    "高血压病": "高血压",
    "二甲": "二甲双胍",
    "心梗": "急性心肌梗死",
}


class MedicalSchemaLinker:
    def __init__(self, db_path: str):
        self.db_path = str(Path(db_path))
        self.values = self._load_values()
        # Terms a Chinese-numeral rewrite could corrupt ("血液生化六项检查",
        # "二十五味松石丸", "百日咳" ...). Kept as a mask for normalize_numerals so
        # quantity counters stay enabled without damaging vocabulary matches.
        self._numeral_protected = numeral_bearing(
            {value for values in self.values.values() for value in values}
        )

    def _load_values(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        queries = {
            "disease": "SELECT DISTINCT disease FROM patient_visits LIMIT 1000",
            "department": "SELECT DISTINCT department FROM patient_visits LIMIT 1000",
            "drug": "SELECT DISTINCT drug FROM prescriptions LIMIT 1000",
            "test_name": "SELECT DISTINCT test_name FROM lab_tests LIMIT 1000",
            "kg_entity": "SELECT DISTINCT name FROM kg_entities LIMIT 3000",
        }
        try:
            connection = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            for field, sql in queries.items():
                try:
                    result[field] = [str(row[0]) for row in connection.execute(sql) if row[0]]
                except sqlite3.Error:
                    result[field] = []
            connection.close()
        except sqlite3.Error:
            return {field: [] for field in queries}
        return result

    def link(self, question: str) -> dict:
        # Chinese numerals first, so "二型糖尿病" can still match the "2型糖尿病"
        # value literal below and the downstream ASCII-digit regexes fire.
        normalized, numerals = normalize_numerals(question, self._numeral_protected)
        aliases = {}
        for alias, canonical in _MEDICAL_ALIASES.items():
            if alias in normalized:
                normalized = normalized.replace(alias, canonical)
                aliases[alias] = canonical
        table_scores = {
            table: sum(term in normalized for term in terms)
            for table, terms in _TABLE_TERMS.items()
        }
        tables = [
            table
            for table, score in sorted(table_scores.items(), key=lambda item: (-item[1], item[0]))
            if score > 0
        ]
        columns = [
            column
            for column, terms in _COLUMN_TERMS.items()
            if any(term in normalized for term in terms)
        ]
        value_matches = []
        for field, values in self.values.items():
            for value in values:
                if len(value) >= 2 and value in normalized:
                    value_matches.append({"field": field, "value": value, "match": "literal"})
        # Longest values first avoids selecting both 糖尿病 and 2型糖尿病.
        value_matches.sort(key=lambda item: (-len(item["value"]), item["field"], item["value"]))
        selected = []
        occupied: list[tuple[int, int]] = []
        for match in value_matches:
            start = normalized.find(match["value"])
            span = (start, start + len(match["value"]))
            if start >= 0 and not any(max(span[0], old[0]) < min(span[1], old[1]) for old in occupied):
                selected.append(match)
                occupied.append(span)
        result = {
            "normalized_question": normalized,
            "aliases": aliases,
            "tables": tables[:3],
            "columns": columns,
            "values": selected[:12],
        }
        # NL2SQL._build_prompt embeds this dict verbatim in the LLM prompt, so an
        # always-present key would perturb every prompt (and re-roll the model's
        # output) even for questions containing no numerals. Report the rewrite log
        # only when a rewrite actually happened.
        if numerals:
            result["numerals"] = numerals
        return result

    @staticmethod
    def quoted(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    # Threshold phrasings put a number in front of a counter without asking for
    # a top-N list ("就诊人次超过50次的科室"). Reading that 50 as a LIMIT would
    # answer a different question, so a question carrying one of these words is
    # treated as having no explicit N.
    _THRESHOLD_WORDS = ("超过", "多于", "达到", "不少于", "至少", "大于",
                        "小于", "低于", "少于", "以上", "以下", "不满")

    @staticmethod
    def explicit_top_n(question: str) -> int | None:
        """Parsed "前N"/"topN"/"N个"/"N种"/"N次"/"N位" count, or None if the
        question has no explicit number. Split out from `extract_top_n` so a
        caller can tell "no number was given" apart from "the number happened to
        be the default" -- needed to pick LIMIT 1 for a bare superlative
        ("最高的是哪个") vs the requested N for an explicit list ("前3个").

        个/种 alone did not cover ranking questions counted in visits or people
        ("费用最高的3次就诊"), which silently fell back to the default 5 and
        returned two rows more than were asked for.
        """
        question = normalize(question)
        counters = "个|种|次|位|名|条"
        if any(word in question for word in MedicalSchemaLinker._THRESHOLD_WORDS):
            counters = "个|种"
        match = re.search(rf"(?:前|top\s*)(\d+)|(\d+)\s*(?:{counters})",
                          question, flags=re.IGNORECASE)
        if not match:
            return None
        value = int(next(group for group in match.groups() if group))
        return max(1, min(100, value))

    @staticmethod
    def extract_top_n(question: str, default: int = 5) -> int:
        # Idempotent on already-normalized text; also makes the helper correct for
        # direct callers that pass a raw question ("前三个科室" -> 3, not `default`).
        value = MedicalSchemaLinker.explicit_top_n(question)
        return value if value is not None else default
