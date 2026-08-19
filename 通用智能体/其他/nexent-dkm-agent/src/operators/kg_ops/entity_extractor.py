"""Medical entity extraction operators for task 2."""

from __future__ import annotations

import json
import re
from collections import OrderedDict, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.operators.kg_ops.pattern_extractor import ENTITY_SURFACE_REJECT, find_pattern_entities

_BUILTIN_ENTITY_DICTIONARY: dict[str, list[str]] = {
    "Disease": [
        "2型糖尿病",
        "支气管哮喘急性发作",
        "高血压",
        "糖尿病",
        "哮喘",
        "冠心病",
        "心力衰竭",
        "胃炎",
        "肺炎",
        "反流性食管炎",
        "甲亢",
        "甲状腺功能减退",
        "系统性红斑狼疮",
        "脑卒中",
    ],
    "Symptom": [
        "头晕",
        "头痛",
        "口渴",
        "多尿",
        "喘息",
        "呼吸困难",
        "胸闷",
        "气短",
        "上腹痛",
        "反酸",
        "发热",
        "咳嗽",
        "恶心",
        "呕吐",
        "腹泻",
        "乏力",
        "心悸",
        "消瘦",
        "手抖",
        "怕冷",
        "关节痛",
    ],
    "Drug": [
        "氨氯地平",
        "阿司匹林",
        "二甲双胍",
        "辛伐他汀",
        "布洛芬",
        "奥美拉唑",
        "阿莫西林",
        "甲巯咪唑",
        "左甲状腺素",
        "泼尼松",
    ],
    "Examination": [
        "血常规",
        "肝功能",
        "尿常规",
        "血糖",
        "血糖监测",
        "肺功能",
        "心电图",
        "胃镜",
        "肠镜",
        "胸片",
        "CT",
        "MRI",
        "CTA",
        "冠脉造影",
        "甲状腺超声",
        "抗核抗体",
    ],
    "Treatment": [
        "调节血脂",
        "对症治疗",
        "抗感染",
        "继续服用",
        "低盐饮食",
        "低脂饮食",
        "戒烟",
        "碘131治疗",
        "免疫抑制",
    ],
}

_TERMINOLOGY_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "samples" / "medical_terminology.json"
)


@lru_cache(maxsize=1)
def load_entity_dictionary() -> dict[str, list[str]]:
    """Load entity dictionary from external terminology file with in-memory cache."""

    if _TERMINOLOGY_PATH.is_file():
        payload = json.loads(_TERMINOLOGY_PATH.read_text(encoding="utf-8"))
        merged: dict[str, list[str]] = {}
        for entity_type, builtin_terms in _BUILTIN_ENTITY_DICTIONARY.items():
            external_terms = payload.get(entity_type, [])
            seen: set[str] = set()
            ordered: list[str] = []
            for term in list(builtin_terms) + list(external_terms):
                if term not in seen:
                    seen.add(term)
                    ordered.append(term)
            merged[entity_type] = ordered
        return merged
    return {key: list(value) for key, value in _BUILTIN_ENTITY_DICTIONARY.items()}


def reload_entity_dictionary() -> dict[str, list[str]]:
    """Clear the terminology cache (used in tests)."""

    load_entity_dictionary.cache_clear()
    dictionary = load_entity_dictionary()
    globals()["ENTITY_DICTIONARY"] = dictionary
    return dictionary


ENTITY_DICTIONARY: dict[str, list[str]] = load_entity_dictionary()

ENTITY_ALIASES: dict[str, dict[str, str]] = {
    "Disease": {
        "糖尿病": "2型糖尿病",
        "哮喘": "支气管哮喘急性发作",
        "Graves病": "甲亢",
        "hyperthyroidism": "甲亢",
        "hypothyroidism": "甲状腺功能减退",
        "hypertension": "高血压",
        "type 2 diabetes": "2型糖尿病",
        "CHD": "冠心病",
        "coronary heart disease": "冠心病",
        "pneumonia": "肺炎",
        "asthma exacerbation": "支气管哮喘急性发作",
    },
    "Symptom": {
        "dizziness": "头晕",
        "chest tightness": "胸闷",
        "weight loss": "消瘦",
        "tremor": "手抖",
        "fever": "发热",
        "dyspnea": "呼吸困难",
        "wheezing": "喘息",
    },
    "Drug": {
        "amlodipine": "氨氯地平",
        "metformin": "二甲双胍",
        "aspirin": "阿司匹林",
        "statin": "辛伐他汀",
        "methimazole": "甲巯咪唑",
        "amoxicillin": "阿莫西林",
    },
    "Examination": {
        "血糖监测": "血糖",
        "ECG": "心电图",
        "CBC": "血常规",
        "CTA": "CTA",
        "coronary CTA": "CTA",
        "thyroid ultrasound": "甲状腺超声",
        "pulmonary function test": "肺功能",
        "肺功能检查": "肺功能",
    },
    "Treatment": {
        "symptomatic care": "对症治疗",
        "调脂": "调节血脂",
    },
}


def extract_medical_entities(text: str) -> dict[str, Any]:
    """Extract and normalize medical entities from raw medical text records."""

    records = []
    global_aliases: dict[str, list[str]] = defaultdict(list)
    for index, record_text in enumerate(_split_records(text), start=1):
        entities: dict[str, list[str]] = {entity_type: [] for entity_type in ENTITY_DICTIONARY}
        record_aliases: dict[str, list[str]] = {}
        mentions: dict[str, list[dict[str, Any]]] = {}

        matched_by_type: dict[str, list[str]] = defaultdict(list)
        for entity_type, surface_term in _find_all_terms_global(record_text):
            matched_by_type[entity_type].append(surface_term)

        for entity_type, found_terms in matched_by_type.items():
            normalized, aliases = _normalize_terms(entity_type, found_terms)
            entities[entity_type] = normalized
            if aliases:
                record_aliases.update(aliases)
                for canonical, alias_values in aliases.items():
                    for alias in alias_values:
                        if alias not in global_aliases[canonical]:
                            global_aliases[canonical].append(alias)
            for term in normalized:
                mentions[term] = _mention_spans(record_text, term)
            for alias_values in aliases.values():
                for alias in alias_values:
                    mentions[alias] = _mention_spans(record_text, alias)

        records.append(
            {
                "record_id": f"record_{index}",
                "text": record_text,
                "entities": entities,
                "normalization": {"aliases": record_aliases},
                "mentions": mentions,
            }
        )

    entity_counts = {
        entity_type: sum(len(record["entities"][entity_type]) for record in records)
        for entity_type in ENTITY_DICTIONARY
    }
    return {
        "status": "completed",
        "record_count": len(records),
        "records": records,
        "entity_counts": entity_counts,
        "normalization": {"aliases": dict(global_aliases)},
    }


def infer_entity_type(name: str) -> str | None:
    """Infer an entity type from the task-2 dictionary or alias map."""

    for entity_type, terms in ENTITY_DICTIONARY.items():
        if name in terms:
            return entity_type
        if name in ENTITY_ALIASES.get(entity_type, {}).values():
            return entity_type
    return None


def _split_records(text: str) -> list[str]:
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*---\s*\n?", text) if chunk.strip()]
    if chunks:
        return chunks
    return [text.strip()] if text.strip() else []


def _searchable_terms(entity_type: str) -> list[str]:
    terms = list(ENTITY_DICTIONARY.get(entity_type, []))
    for alias in ENTITY_ALIASES.get(entity_type, {}):
        if alias not in terms:
            terms.append(alias)
    return terms


def _dictionary_candidates() -> list[tuple[str, str, int, int]]:
    candidates: list[tuple[str, str, int, int]] = []
    for entity_type in ENTITY_DICTIONARY:
        for term in _searchable_terms(entity_type):
            candidates.append((entity_type, term, len(term), 1))
    return candidates


def _entity_in_vocabulary(name: str, entity_type: str) -> bool:
    terms = set(ENTITY_DICTIONARY.get(entity_type, []))
    aliases = ENTITY_ALIASES.get(entity_type, {})
    canonical = aliases.get(name, name)
    return canonical in terms or name in terms


def _pattern_supplement_allowed(entity_type: str, term: str) -> bool:
    """Allow suffix-pattern hits only when they extend open-domain coverage."""

    if _entity_in_vocabulary(term, entity_type):
        return False
    if entity_type == "Disease" and term.endswith("病"):
        stem = term[:-1]
        if _entity_in_vocabulary(stem, "Disease"):
            return False
        if len(stem) < 2 or stem[0] in "多各两几种类其合主诊断疑似混否":
            return False
    if entity_type == "Examination" and term.endswith("抗体") and term.startswith("查"):
        return False
    if entity_type == "Examination" and term.startswith("血") and len(term) > 4:
        return False
    if entity_type == "Examination" and term.startswith("血") and term.endswith(
        ("监测", "未控", "控制", "示高", "小板", "板")
    ):
        return False
    return True


def _find_all_terms_global(text: str) -> list[tuple[str, str]]:
    """Match dictionary, alias, and suffix-pattern terms with global longest-first spans."""

    candidates: list[tuple[str, str, int, int]] = _dictionary_candidates()
    for entity_type, term in find_pattern_entities(text):
        if _pattern_supplement_allowed(entity_type, term):
            candidates.append((entity_type, term, len(term), 0))
    candidates.sort(key=lambda item: (item[2], item[3]), reverse=True)

    occupied: list[tuple[int, int]] = []
    found: list[tuple[str, str]] = []
    lower_text = text.lower()
    for entity_type, term, _length, _priority in candidates:
        if term in ENTITY_SURFACE_REJECT:
            continue
        haystacks = [(text, term)]
        if term.isascii():
            haystacks = [(lower_text, term.lower())]
        matched = False
        for haystack, needle in haystacks:
            start = 0
            while True:
                index = haystack.find(needle, start)
                if index < 0:
                    break
                end = index + len(needle)
                if not _span_overlaps(index, end, occupied):
                    found.append((entity_type, term))
                    occupied.append((index, end))
                    matched = True
                    break
                start = index + 1
            if matched:
                break
    return found


def _find_terms_global(text: str) -> list[tuple[str, str]]:
    """Match dictionary and alias terms with global non-overlapping spans."""

    return _find_all_terms_global(text)


def _find_terms(text: str, terms: list[str]) -> list[str]:
    found: OrderedDict[str, None] = OrderedDict()
    occupied: list[tuple[int, int]] = []
    for term in sorted(terms, key=len, reverse=True):
        start = 0
        while True:
            index = text.find(term, start)
            if index < 0:
                break
            end = index + len(term)
            if not _span_overlaps(index, end, occupied):
                found[term] = None
                occupied.append((index, end))
            start = index + 1
    return list(found.keys())


def _span_overlaps(start: int, end: int, occupied: list[tuple[int, int]]) -> bool:
    return any(not (end <= o_start or start >= o_end) for o_start, o_end in occupied)


def _normalize_terms(entity_type: str, terms: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    normalized: OrderedDict[str, None] = OrderedDict()
    aliases: dict[str, list[str]] = defaultdict(list)
    alias_map = ENTITY_ALIASES.get(entity_type, {})
    for term in terms:
        canonical = alias_map.get(term, term)
        normalized[canonical] = None
        if canonical != term and term not in aliases[canonical]:
            aliases[canonical].append(term)
    return list(normalized.keys()), dict(aliases)


def _mention_spans(text: str, term: str) -> list[dict[str, Any]]:
    return [
        {
            "start": match.start(),
            "end": match.end(),
            "text": term,
        }
        for match in re.finditer(re.escape(term), text)
    ]
