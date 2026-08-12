#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
带来源追踪的医学知识图谱构建脚本。

该脚本写入实体、关系、三元组和来源信息，供任务二和任务三复用。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS kg_build_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS kg_sources (
    source_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT NOT NULL UNIQUE,
    source_path TEXT,
    source_type TEXT,
    source_url TEXT,
    record_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kg_entities (
    entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    description TEXT,
    source_id INTEGER,
    external_id TEXT,
    confidence REAL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    UNIQUE(canonical_name, entity_type),
    FOREIGN KEY (source_id) REFERENCES kg_sources(source_id)
);

CREATE TABLE IF NOT EXISTS kg_aliases (
    alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
    alias TEXT NOT NULL,
    entity_id INTEGER NOT NULL,
    source_id INTEGER,
    confidence REAL DEFAULT 1.0,
    UNIQUE(alias, entity_id),
    FOREIGN KEY (entity_id) REFERENCES kg_entities(entity_id),
    FOREIGN KEY (source_id) REFERENCES kg_sources(source_id)
);

CREATE TABLE IF NOT EXISTS kg_relations (
    relation_code TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    subject_type TEXT,
    object_type TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS kg_triples (
    triple_id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER NOT NULL,
    relation_code TEXT NOT NULL,
    object_id INTEGER NOT NULL,
    source_id INTEGER,
    evidence TEXT,
    confidence REAL DEFAULT 1.0,
    extractor TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(subject_id, relation_code, object_id, source_id, evidence),
    FOREIGN KEY (subject_id) REFERENCES kg_entities(entity_id),
    FOREIGN KEY (object_id) REFERENCES kg_entities(entity_id),
    FOREIGN KEY (relation_code) REFERENCES kg_relations(relation_code),
    FOREIGN KEY (source_id) REFERENCES kg_sources(source_id)
);

CREATE TABLE IF NOT EXISTS kg_quality_issues (
    issue_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER,
    field_name TEXT,
    value TEXT,
    issue_type TEXT,
    evidence TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES kg_sources(source_id)
);

CREATE INDEX IF NOT EXISTS idx_kg_entities_name ON kg_entities(canonical_name);
CREATE INDEX IF NOT EXISTS idx_kg_entities_type ON kg_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_kg_triples_relation ON kg_triples(relation_code);
CREATE INDEX IF NOT EXISTS idx_kg_triples_subject ON kg_triples(subject_id);
CREATE INDEX IF NOT EXISTS idx_kg_triples_object ON kg_triples(object_id);
CREATE INDEX IF NOT EXISTS idx_kg_quality_issue_type ON kg_quality_issues(issue_type);

CREATE VIEW IF NOT EXISTS v_entity_stats AS
SELECT entity_type, COUNT(*) AS entity_count
FROM kg_entities
GROUP BY entity_type;

CREATE VIEW IF NOT EXISTS v_relation_stats AS
SELECT r.relation_code, r.display_name, COUNT(t.triple_id) AS triple_count
FROM kg_relations r
LEFT JOIN kg_triples t ON r.relation_code = t.relation_code
GROUP BY r.relation_code, r.display_name;

CREATE VIEW IF NOT EXISTS v_disease_facts AS
SELECT
    s.canonical_name AS disease,
    s.entity_type AS subject_type,
    r.relation_code,
    r.display_name AS relation_name,
    o.canonical_name AS object,
    o.entity_type AS object_type,
    t.evidence,
    t.confidence,
    src.source_name
FROM kg_triples t
JOIN kg_entities s ON t.subject_id = s.entity_id
JOIN kg_entities o ON t.object_id = o.entity_id
JOIN kg_relations r ON t.relation_code = r.relation_code
LEFT JOIN kg_sources src ON t.source_id = src.source_id
WHERE s.entity_type = 'disease';
"""


BASE_RELATIONS = {
    "has_symptom": ("临床表现", "disease", "symptom"),
    "treated_by_drug": ("药物治疗", "disease", "drug"),
    "treated_by_procedure": ("治疗方式", "disease", "procedure"),
    "requires_test": ("检查", "disease", "test"),
    "visit_department": ("就诊科室", "disease", "department"),
    "has_complication": ("并发症", "disease", "disease"),
    "has_cause": ("病因", "disease", "cause"),
    "has_prevention": ("预防", "disease", "prevention"),
    "susceptible_population": ("易感人群", "disease", "population"),
    "transmission_way": ("传播途径", "disease", "other"),
    "affects_body_part": ("发病部位", "disease", "body_part"),
    "belongs_to_category": ("疾病分类", "disease", "category"),
    "differential_diagnosis": ("鉴别诊断", "disease", "disease"),
    "alias_of": ("别名", "other", "other"),
    "related_to": ("相关", "other", "other"),
}


CMEIE_ENTITY_TYPE_MAP = {
    "疾病": "disease",
    "症状": "symptom",
    "药物": "drug",
    "药品": "drug",
    "检查": "test",
    "检查项目": "test",
    "科室": "department",
    "治疗": "procedure",
    "治疗方法": "procedure",
    "手术治疗": "procedure",
    "其他治疗": "procedure",
    "部位": "body_part",
    "身体部位": "body_part",
    "人群": "population",
    "社会学": "population",
}


CMEIE_RELATION_MAP = {
    "临床表现": "has_symptom",
    "药物治疗": "treated_by_drug",
    "治疗后症状": "has_symptom",
    "检查": "requires_test",
    "辅助检查": "requires_test",
    "实验室检查": "requires_test",
    "影像学检查": "requires_test",
    "手术治疗": "treated_by_procedure",
    "放射治疗": "treated_by_procedure",
    "化疗": "treated_by_procedure",
    "所属科室": "visit_department",
    "并发症": "has_complication",
    "病因": "has_cause",
    "发病原因": "has_cause",
    "预防": "has_prevention",
    "多发群体": "susceptible_population",
    "传播途径": "transmission_way",
    "发病部位": "affects_body_part",
    "鉴别诊断": "differential_diagnosis",
    "同义词": "alias_of",
}


SYMPTOM_CUES = (
    "痛",
    "疼",
    "热",
    "烧",
    "咳",
    "痰",
    "喘",
    "泻",
    "吐",
    "恶心",
    "乏力",
    "无力",
    "困难",
    "出血",
    "血尿",
    "便血",
    "肿",
    "水肿",
    "皮疹",
    "瘙痒",
    "麻",
    "晕",
    "昏迷",
    "抽搐",
    "惊厥",
    "黄疸",
    "紫绀",
    "发绀",
    "心悸",
    "胸闷",
    "气短",
    "腹胀",
    "便秘",
    "腹泻",
    "尿频",
    "尿急",
    "尿痛",
    "消瘦",
    "肥胖",
    "畏寒",
    "寒战",
    "溃疡",
    "瘫痪",
    "失眠",
    "嗜睡",
)


COMMON_SHORT_SYMPTOMS = {
    "休克",
    "心慌",
    "窒息",
    "奇脉",
    "猝死",
    "代脉",
    "创伤",
    "梗噎",
    "目赤",
    "眼花",
    "脸红",
}


KNOWN_BAD_SYMPTOM_TOKENS = {
    "毓卓",
    "测试",
    "驻站医",
    "衣玉品",
}


COMMON_SURNAME_CHARS = set(
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
    "戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐"
    "费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平"
    "黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋庞熊纪舒屈项祝董"
    "梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田胡凌"
    "霍虞万支柯昝管卢莫经房裘缪干解应宗丁宣贲邓郁单杭洪包诸左石崔吉"
    "龚程嵇邢滑裴陆荣翁荀羊於惠甄曲家封芮羿储靳汲邴糜松井段富巫乌焦"
    "巴弓牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘斜厉戎祖武符刘景"
    "詹束龙叶幸司韶郜黎蓟薄印宿白怀蒲邰从鄂索咸籍赖卓蔺屠蒙池乔阴胥"
    "能苍双闻莘党翟谭贡劳逄姬申扶堵冉宰郦雍郤璩桑桂濮牛寿通边扈燕冀"
    "郏浦尚农温别庄晏柴瞿阎闫充慕连茹习宦艾鱼容向古易慎戈廖庾终暨居"
    "衡步都耿满弘匡国文寇广禄阙东殴殳沃利蔚越夔隆师巩厍聂晁勾敖融冷"
    "訾辛阚那简饶空曾毋沙乜养鞠须丰巢关蒯相查後荆红游竺权逯盖益桓公"
    "代岳栗兰"
)


def looks_like_person_name(value: str) -> bool:
    return len(value) in {2, 3} and value[0] in COMMON_SURNAME_CHARS


class QualityContext:
    def __init__(self, symptom_vocab: set[str] | None = None) -> None:
        self.symptom_vocab = symptom_vocab or set()

    def is_bad_text(self, value: str) -> bool:
        if not value:
            return True
        if "\ufffd" in value:
            return True
        return any(unicodedata.category(ch) in {"Cc", "Cs", "Co"} for ch in value)

    def is_plausible_symptom(self, value: str) -> bool:
        value = compact_value(value)
        if self.is_bad_text(value):
            return False
        if value in KNOWN_BAD_SYMPTOM_TOKENS:
            return False
        if value in self.symptom_vocab:
            return True
        if value in COMMON_SHORT_SYMPTOMS:
            return True
        if any(cue in value for cue in SYMPTOM_CUES):
            return True
        if looks_like_person_name(value):
            return False
        return True


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def compact_value(value: Any, max_len: int = 240) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        text = "；".join(str(v).strip() for v in value if str(v).strip())
    else:
        text = str(value).strip()
    text = " ".join(text.replace("\r", "\n").split())
    if len(text) > max_len:
        return text[: max_len - 1] + "…"
    return text


def load_cmeee_symptom_vocab(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return set()
    if not isinstance(data, list):
        return set()
    vocab: set[str] = set()
    for item in data:
        for ent in item.get("entities") or []:
            if ent.get("type") == "sym":
                value = compact_value(ent.get("entity"))
                if value:
                    vocab.add(value)
    return vocab


def iter_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [compact_value(v) for v in value if compact_value(v)]
    text = compact_value(value)
    return [text] if text else []


def object_value(value: Any) -> str:
    if isinstance(value, dict):
        if "@value" in value:
            return compact_value(value.get("@value"))
        for item in value.values():
            text = compact_value(item)
            if text:
                return text
    return compact_value(value)


def type_value(value: Any) -> str:
    if isinstance(value, dict):
        if "@value" in value:
            return compact_value(value.get("@value"))
        for item in value.values():
            text = compact_value(item)
            if text:
                return text
    return compact_value(value)


def fallback_relation_code(predicate: str) -> str:
    digest = hashlib.sha1(predicate.encode("utf-8")).hexdigest()[:10]
    return f"cmeie_{digest}"


class KgWriter:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(DDL)
        self._seed_relations()
        self.run_id = self.conn.execute(
            "INSERT INTO kg_build_runs (started_at, notes) VALUES (?, ?)",
            (utc_now(), "Task 2 local KG v2 build"),
        ).lastrowid

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def _seed_relations(self) -> None:
        for code, (display, subject_type, object_type) in BASE_RELATIONS.items():
            self.ensure_relation(code, display, subject_type, object_type)

    def ensure_source(
        self,
        source_name: str,
        source_path: Path,
        source_type: str,
        source_url: str,
    ) -> int:
        row = self.conn.execute(
            "SELECT source_id FROM kg_sources WHERE source_name=?", (source_name,)
        ).fetchone()
        if row:
            return int(row[0])
        return int(
            self.conn.execute(
                """
                INSERT INTO kg_sources
                    (source_name, source_path, source_type, source_url, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (source_name, str(source_path), source_type, source_url, utc_now()),
            ).lastrowid
        )

    def update_source_count(self, source_id: int, record_count: int) -> None:
        self.conn.execute(
            "UPDATE kg_sources SET record_count=? WHERE source_id=?",
            (record_count, source_id),
        )

    def ensure_relation(
        self,
        code: str,
        display_name: str,
        subject_type: str = "",
        object_type: str = "",
        description: str = "",
    ) -> str:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO kg_relations
                (relation_code, display_name, subject_type, object_type, description)
            VALUES (?, ?, ?, ?, ?)
            """,
            (code, display_name, subject_type, object_type, description),
        )
        return code

    def ensure_entity(
        self,
        name: str,
        entity_type: str,
        source_id: int | None = None,
        description: str = "",
        external_id: str = "",
        confidence: float = 1.0,
    ) -> int | None:
        name = compact_value(name)
        if not name:
            return None
        now = utc_now()
        self.conn.execute(
            """
            INSERT OR IGNORE INTO kg_entities
                (canonical_name, entity_type, description, source_id, external_id,
                 confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (name, entity_type, description, source_id, external_id, confidence, now),
        )
        row = self.conn.execute(
            "SELECT entity_id FROM kg_entities WHERE canonical_name=? AND entity_type=?",
            (name, entity_type),
        ).fetchone()
        return int(row[0]) if row else None

    def add_alias(
        self,
        alias: str,
        entity_id: int | None,
        source_id: int,
        confidence: float = 0.9,
    ) -> None:
        alias = compact_value(alias)
        if not alias or entity_id is None:
            return
        self.conn.execute(
            """
            INSERT OR IGNORE INTO kg_aliases
                (alias, entity_id, source_id, confidence)
            VALUES (?, ?, ?, ?)
            """,
            (alias, entity_id, source_id, confidence),
        )

    def add_triple(
        self,
        subject_id: int | None,
        relation_code: str,
        object_id: int | None,
        source_id: int,
        evidence: str = "",
        confidence: float = 1.0,
        extractor: str = "rule",
    ) -> None:
        if subject_id is None or object_id is None:
            return
        self.conn.execute(
            """
            INSERT OR IGNORE INTO kg_triples
                (subject_id, relation_code, object_id, source_id, evidence,
                 confidence, extractor, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                subject_id,
                relation_code,
                object_id,
                source_id,
                compact_value(evidence, 500),
                confidence,
                extractor,
                utc_now(),
            ),
        )

    def add_quality_issue(
        self,
        source_id: int,
        field_name: str,
        value: str,
        issue_type: str,
        evidence: str = "",
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO kg_quality_issues
                (source_id, field_name, value, issue_type, evidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                source_id,
                field_name,
                compact_value(value),
                issue_type,
                compact_value(evidence, 500),
                utc_now(),
            ),
        )


def load_medical_json(
    writer: KgWriter,
    path: Path,
    limit: int | None = None,
    quality: QualityContext | None = None,
) -> int:
    quality = quality or QualityContext()
    source_id = writer.ensure_source(
        "QASystemOnMedicalKG:medical.json",
        path,
        "jsonl",
        "https://github.com/liuhuanyong/QASystemOnMedicalKG",
    )
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if limit is not None and count >= limit:
                break
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = compact_value(item.get("name"))
            if not name:
                continue
            external_id = compact_value((item.get("_id") or {}).get("$oid", ""))
            disease_id = writer.ensure_entity(
                name,
                "disease",
                source_id=source_id,
                description=compact_value(item.get("desc"), 500),
                external_id=external_id,
                confidence=1.0,
            )
            writer.add_alias(name, disease_id, source_id)

            list_mappings = [
                ("symptom", "has_symptom", "symptom"),
                ("acompany", "has_complication", "disease"),
                ("cure_department", "visit_department", "department"),
                ("cure_way", "treated_by_procedure", "procedure"),
                ("check", "requires_test", "test"),
                ("recommand_drug", "treated_by_drug", "drug"),
                ("common_drug", "treated_by_drug", "drug"),
                ("drug_detail", "treated_by_drug", "drug"),
                ("category", "belongs_to_category", "category"),
                ("easy_get", "susceptible_population", "population"),
                ("get_way", "transmission_way", "other"),
            ]
            for field, relation_code, object_type in list_mappings:
                for value in iter_values(item.get(field)):
                    if object_type == "symptom" and not quality.is_plausible_symptom(value):
                        writer.add_quality_issue(
                            source_id,
                            field,
                            value,
                            "suspicious_symptom",
                            evidence=f"{name}: {value}",
                        )
                        continue
                    if quality.is_bad_text(value):
                        writer.add_quality_issue(
                            source_id,
                            field,
                            value,
                            "invalid_text",
                            evidence=f"{name}: {value}",
                        )
                        continue
                    object_id = writer.ensure_entity(
                        value, object_type, source_id=source_id, confidence=0.95
                    )
                    writer.add_triple(
                        disease_id,
                        relation_code,
                        object_id,
                        source_id,
                        evidence=f"{field}: {value}",
                        confidence=0.95,
                        extractor="qasystem_medical_json",
                    )

            text_mappings = [
                ("cause", "has_cause", "cause", 0.85),
                ("prevent", "has_prevention", "prevention", 0.85),
            ]
            for field, relation_code, object_type, confidence in text_mappings:
                for value in iter_values(item.get(field)):
                    object_id = writer.ensure_entity(
                        value, object_type, source_id=source_id, confidence=confidence
                    )
                    writer.add_triple(
                        disease_id,
                        relation_code,
                        object_id,
                        source_id,
                        evidence=f"{field}: {value}",
                        confidence=confidence,
                        extractor="qasystem_medical_json",
                    )
            count += 1
    writer.update_source_count(source_id, count)
    return count


def normalize_cmeie_type(raw_type: Any) -> str:
    raw = type_value(raw_type)
    return CMEIE_ENTITY_TYPE_MAP.get(raw, "other")


def load_cmeie_symptom_vocab(path: Path, limit: int | None = None) -> set[str]:
    if not path.exists():
        return set()
    vocab: set[str] = set()
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if limit is not None and count >= limit:
                break
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            for spo in item.get("spo_list") or []:
                pairs = [
                    (spo.get("subject"), spo.get("subject_type")),
                    (spo.get("object"), spo.get("object_type")),
                ]
                for value, raw_type in pairs:
                    if normalize_cmeie_type(raw_type) == "symptom":
                        text = object_value(value)
                        if text:
                            vocab.add(text)
            count += 1
    return vocab


def load_cmeie_jsonl(writer: KgWriter, path: Path, limit: int | None = None) -> int:
    source_id = writer.ensure_source(
        f"CBLUE:CMeIE:{path.name}",
        path,
        "jsonl",
        "https://github.com/CBLUEbenchmark/CBLUE",
    )
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if limit is not None and count >= limit:
                break
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            evidence = compact_value(item.get("text"), 500)
            for spo in item.get("spo_list") or []:
                subject = compact_value(spo.get("subject"))
                object_text = object_value(spo.get("object"))
                predicate = compact_value(spo.get("predicate"))
                if not subject or not object_text or not predicate:
                    continue
                subject_type = normalize_cmeie_type(spo.get("subject_type"))
                object_type = normalize_cmeie_type(spo.get("object_type"))
                relation_code = CMEIE_RELATION_MAP.get(predicate)
                if relation_code is None:
                    continue
                subject_id = writer.ensure_entity(
                    subject, subject_type, source_id=source_id, confidence=0.9
                )
                object_id = writer.ensure_entity(
                    object_text, object_type, source_id=source_id, confidence=0.9
                )
                writer.add_triple(
                    subject_id,
                    relation_code,
                    object_id,
                    source_id,
                    evidence=evidence,
                    confidence=0.9,
                    extractor="cblue_cmeie_annotation",
                )
            count += 1
    writer.update_source_count(source_id, count)
    return count


def summarize(db_path: Path) -> dict[str, int]:
    conn = sqlite3.connect(str(db_path))
    tables = [
        "kg_sources",
        "kg_entities",
        "kg_aliases",
        "kg_relations",
        "kg_triples",
        "kg_quality_issues",
    ]
    stats = {
        table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in tables
    }
    conn.close()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/task2_medical_kg.db")
    parser.add_argument("--medical-json", default=None)
    parser.add_argument("--cmeie-jsonl", action="append", default=[])
    parser.add_argument("--cmeee-json", default=None)
    parser.add_argument("--medical-limit", type=int, default=None)
    parser.add_argument("--cmeie-limit", type=int, default=None)
    args = parser.parse_args()

    db_path = Path(args.db)
    symptom_vocab = (
        load_cmeee_symptom_vocab(Path(args.cmeee_json)) if args.cmeee_json else set()
    )
    for cmeie_path in args.cmeie_jsonl:
        symptom_vocab.update(load_cmeie_symptom_vocab(Path(cmeie_path)))
    quality = QualityContext(symptom_vocab=symptom_vocab)
    writer = KgWriter(db_path)
    try:
        if args.medical_json:
            load_medical_json(
                writer,
                Path(args.medical_json),
                args.medical_limit,
                quality=quality,
            )
        for cmeie_path in args.cmeie_jsonl:
            load_cmeie_jsonl(writer, Path(cmeie_path), args.cmeie_limit)
    finally:
        writer.close()

    stats = summarize(db_path)
    print(f"db={db_path}")
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
