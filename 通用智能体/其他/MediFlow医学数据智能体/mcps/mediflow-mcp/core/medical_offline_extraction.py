# -*- coding: utf-8 -*-
"""
本地医学实体和关系抽取模块。

该模块基于词典、规则和文本模式完成基础抽取，降低任务二对外部模型接口的依赖。
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from .schemas import Entity, Relation, Triple
from .medical_extraction_validation import relations_to_triples
from .medical_lexicon import load_benchmark_terms, load_known_relation_pairs, load_relation_terms
from .medical_reliability import reliability_for


KG_TO_ENTITY_TYPE = {
    "disease": "dis",
    "symptom": "sym",
    "drug": "dru",
    "test": "ite",
    "procedure": "pro",
    "department": "dep",
    "body_part": "bod",
    "microorganism": "mic",
}

KG_TO_RELATION_TYPE = {
    "has_symptom": "临床表现",
    "treated_by_drug": "药物治疗",
    "treated_by_procedure": "辅助治疗",
    "requires_test": "辅助检查",
    "visit_department": "就诊科室",
    "has_complication": "并发症",
    "has_cause": "病因",
    "has_prevention": "预防",
    "affects_body_part": "发病部位",
    "alias_of": "同义词",
    "related_to": "相关（导致）",
}

TYPE_TO_DEFAULT_RELATION = {
    "sym": "临床表现",
    "dru": "药物治疗",
    "ite": "辅助检查",
    "pro": "辅助治疗",
    "dep": "就诊科室",
    "bod": "发病部位",
    "mic": "病因",
}

_OFFLINE_RELATION_OBJECT_TYPES = {
    "\u4e34\u5e8a\u8868\u73b0": {"sym", "bod"},
    "\u836f\u7269\u6cbb\u7597": {"dru"},
    "\u8f85\u52a9\u68c0\u67e5": {"ite"},
    "\u5b9e\u9a8c\u5ba4\u68c0\u67e5": {"ite"},
    "\u5f71\u50cf\u5b66\u68c0\u67e5": {"ite"},
    "\u5185\u7aa5\u955c\u68c0\u67e5": {"ite"},
    "\u7ec4\u7ec7\u5b66\u68c0\u67e5": {"ite"},
    "\u53d1\u75c5\u90e8\u4f4d": {"bod"},
    "\u5916\u4fb5\u90e8\u4f4d": {"bod"},
    "\u8f6c\u79fb\u90e8\u4f4d": {"bod"},
    "\u5e76\u53d1\u75c7": {"dis", "sym"},
    "\u9274\u522b\u8bca\u65ad": {"dis"},
}


def _offline_relation_object_type_is_compatible(
    predicate: str, object_type: str
) -> bool:
    allowed = _OFFLINE_RELATION_OBJECT_TYPES.get(predicate)
    return not allowed or object_type in allowed


def _offline_relation_context_is_supported(
    predicate: str, disease: Entity, obj: Entity, sentence: str
) -> bool:
    """Reject a known pair when the sentence explicitly scopes it elsewhere."""

    if predicate != "\u836f\u7269\u6cbb\u7597":
        return True
    object_start = sentence.find(obj.text)
    if object_start < 0:
        return False
    local_left = max(0, object_start - 36)
    local_right = min(len(sentence), object_start + len(obj.text) + 36)
    local_text = sentence[local_left:local_right]
    if any(
        cue in local_text
        for cue in ("\u65e0\u53cd\u5e94", "\u4e0d\u9700\u8981", "\u65e0\u9700", "\u672a\u4f7f\u7528")
    ):
        return False
    disease_start = sentence.find(disease.text)
    if disease_start >= 0:
        before = sentence[max(0, disease_start - 12) : disease_start]
        if any(cue in before for cue in ("\u65e2\u5f80\u6709", "\u6709\u75c5\u53f2", "\u6709\u2026\u53f2")):
            disease_end = disease_start + len(disease.text)
            between = sentence[disease_end : object_start + len(obj.text)]
            if not any(
                cue in between
                for cue in ("口服", "服用", "使用", "应用", "注射", "给予")
            ):
                return False
    return True


SEED_TERMS = {
    "dis": ["糖尿病", "2型糖尿病", "高血压", "胃溃疡", "心力衰竭", "幽门螺杆菌感染"],
    "sym": ["多饮", "多尿", "口干", "胸闷", "气促", "水肿", "上腹部疼痛", "反酸", "嗳气"],
    "dru": ["二甲双胍", "胰岛素", "硝苯地平", "奥美拉唑", "阿莫西林", "克拉霉素"],
    "ite": ["血糖", "空腹血糖", "糖化血红蛋白", "HbA1c", "血压", "胃镜", "尿糖"],
    "pro": ["饮食控制", "血糖监测", "利尿", "降压"],
    "dep": ["内分泌科", "心内科", "消化内科"],
    "mic": ["幽门螺杆菌", "Hp"],
}

STOP_TERMS = {
    "患者", "医生", "治疗", "检查", "诊断", "病史", "阳性", "阴性",
    "糖尿", "内科", "外科", "儿童", "老人", "男性", "女性",
    "升高", "口服", "长期口服", "控制血糖",
}
TYPE_PRIORITY = {
    "dis": 0,
    "dru": 1,
    "sym": 2,
    "ite": 3,
    "pro": 4,
    "dep": 5,
    "mic": 6,
    "bod": 7,
}
SENTENCE_RE = re.compile(r"[^。！？!?；;@\n]+")
NEGATION_RE = re.compile(r"(?:无|未见|未发现|否认|排除|不考虑|不需要|未出现)$")
NEGATION_CONTEXT_RE = re.compile(
    r"(?:无|未见|未发现|否认|排除|不考虑|不需要|无需|没有|未曾)"
    r"[^。！？!?；;\n，,]{0,14}$"
)

RELATION_CUES = {
    "同义词": ("又称", "别名", "简称", "即"),
    "鉴别诊断": ("鉴别", "区分"),
    "并发症": ("并发", "合并"),
    "病因": ("病因", "由于", "引起"),
    "预防": ("预防", "避免"),
    "相关（导致）": ("导致", "引起", "造成"),
}

TYPE_RELATION_CUES = {
    "sym": ("表现为", "症状", "伴有", "伴随", "出现", "可见", "主诉"),
    "dru": ("治疗", "用药", "给予", "予", "服用", "口服", "注射"),
    "ite": ("检查", "监测", "复查", "提示", "检测", "测定"),
    "pro": ("治疗", "手术", "处理", "干预", "切除", "移植", "放疗", "化疗"),
    "dep": ("就诊", "转诊", "科室"),
    "bod": ("发生于", "位于", "累及", "侵犯", "转移至", "部位"),
    "mic": ("感染", "病因", "由于", "引起"),
}


_CAUSAL_CONTEXT_CUES = (
    "\u6f5c\u5728\u7684\u75c5\u56e0",
    "\u75c5\u56e0",
    "\u539f\u56e0",
    "\u7531\u4e8e",
    "\u5f15\u8d77",
    "\u5bfc\u81f4",
    "\u8bf1\u53d1",
    "\u75c5\u539f\u4f53",
)
_BODY_SITE_CONTEXT_CUES = (
    "\u53d1\u75c5\u90e8\u4f4d",
    "\u597d\u53d1\u4e8e",
    "\u53d1\u751f\u4e8e",
    "\u4f4d\u4e8e",
    "\u7d2f\u53ca",
    "\u4fb5\u72af",
    "\u8f6c\u79fb\u81f3",
    "\u5728",
)
_RELATION_CLAUSE_BOUNDARIES = (
    "\u3002",
    "\uff1b",
    ";",
    "!",
    "\uff01",
    "?",
    "\uff1f",
    "@",
    "\n",
)


def _valid_term(term: str) -> bool:
    term = (term or "").strip()
    if not term or term in STOP_TERMS:
        return False
    if len(term) < 2:
        return False
    if len(term) > 32:
        return False
    if re.fullmatch(r"[\d.]+", term):
        return False
    return True


@lru_cache(maxsize=8)
def load_entity_dictionary(db_path: str = "") -> tuple[tuple[str, str], ...]:
    """加载实体词典，返回术语和 CMeEE 类型。"""
    term_types: dict[str, set[str]] = {}
    benchmark_terms = load_benchmark_terms()
    benchmark_type_by_term = {
        value: entity_type for value, entity_type in benchmark_terms
    }
    relation_terms = load_relation_terms()
    relation_type_by_term = {value: entity_type for value, entity_type in relation_terms}
    path = Path(db_path) if db_path else None
    if path and path.exists():
        conn = sqlite3.connect(str(path))
        try:
            rows = conn.execute(
                """
                SELECT canonical_name, entity_type
                FROM kg_entities
                WHERE canonical_name IS NOT NULL AND canonical_name != ''
                """
            )
            for name, kg_type in rows:
                entity_type = KG_TO_ENTITY_TYPE.get(str(kg_type or "").strip())
                value = str(name or "").strip()
                if entity_type and _valid_term(value):
                    term_types.setdefault(value, set()).add(entity_type)
        finally:
            conn.close()

    for entity_type, values in SEED_TERMS.items():
        for value in values:
            if _valid_term(value):
                term_types.setdefault(value, set()).add(entity_type)

    for value, entity_type in benchmark_terms:
        if _valid_term(value):
            term_types.setdefault(value, set()).add(entity_type)

    terms = [
        (
            value,
            benchmark_type_by_term.get(value)
            or relation_type_by_term.get(value)
            or sorted(types, key=lambda item: TYPE_PRIORITY.get(item, 99))[0],
        )
        for value, types in term_types.items()
    ]
    return tuple(sorted(terms, key=lambda item: (-len(item[0]), item[0], item[1])))


@lru_cache(maxsize=8)
def _dictionary_index(db_path: str = "") -> dict[str, tuple[tuple[str, str], ...]]:
    buckets: dict[str, list[tuple[str, str]]] = {}
    for term, entity_type in load_entity_dictionary(db_path):
        buckets.setdefault(term[0], []).append((term, entity_type))
    return {key: tuple(values) for key, values in buckets.items()}


def _select_non_overlapping_matches(
    text: str,
    index: dict[str, tuple[tuple[str, str], ...]],
) -> list[tuple[int, int, str, str]]:
    """Select maximal dictionary spans while retaining separate mentions.

    The old matcher only blocked an identical span.  That allowed ``关节`` to
    survive inside ``踝关节`` and ``骨性关节炎``.  Candidate spans are now
    ordered by length first, so the longest mention owns its character range;
    a later mention at a separate position is still retained.
    """

    candidates: list[tuple[int, int, str, str]] = []
    seen: set[tuple[int, int, str]] = set()

    def is_cjk(value: str) -> bool:
        return bool(value) and "\u4e00" <= value <= "\u9fff"

    for start, first_char in enumerate(text):
        for term, entity_type in index.get(first_char, ()):
            if not text.startswith(term, start):
                continue
            end = start + len(term) - 1
            if (
                len(term) <= 2
                and start > 0
                and end + 1 < len(text)
                and is_cjk(text[start - 1])
                and is_cjk(text[end + 1])
            ):
                continue
            key = (start, end, entity_type)
            if key in seen:
                continue
            seen.add(key)
            candidates.append((start, end, term, entity_type))

    candidates.sort(
        key=lambda item: (
            -(item[1] - item[0] + 1),
            item[0],
            TYPE_PRIORITY.get(item[3], 99),
            item[2],
        )
    )
    selected: list[tuple[int, int, str, str]] = []
    for candidate in candidates:
        start, end, _, _ = candidate
        overlaps = [
            other
            for other in selected
            if start <= other[1] and end >= other[0]
        ]
        if not overlaps:
            selected.append(candidate)
            continue
        # CMeEE marks symptom spans with nested body-part, test, procedure
        # and disease mentions. Same-type lexical fragments remain longest
        # match only; cross-type symptom nesting is retained.
        can_keep_nested = any(
            candidate[3] != other[3]
            and (
                (other[3] == "sym" and other[0] <= start and end <= other[1])
                or (candidate[3] == "sym" and start <= other[0] and other[1] <= end)
            )
            for other in overlaps
        )
        if not can_keep_nested:
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda item: (item[0], -(item[1] - item[0] + 1), item[2]))


def extract_entities_offline(text: str, db_path: str = "") -> list[Entity]:
    """使用本地知识图谱词典匹配抽取实体。"""
    if not text or not text.strip():
        return []

    entities: list[Entity] = []
    for start, end, term, entity_type in _select_non_overlapping_matches(
        text, _dictionary_index(db_path)
    ):
        left = max(0, start - 20)
        right = min(len(text), end + 21)
        reliability = reliability_for("entity", "dictionary_exact", entity_type)
        entities.append(
            Entity(
                text=term,
                type=entity_type,
                start_idx=start,
                end_idx=end,
                confidence=reliability.score,
                evidence=text[left:right],
                extraction_method="dictionary_exact",
                reliability_level=reliability.level,
            )
        )

    return sorted(entities, key=lambda item: (item.start_idx or 0, -len(item.text)))


@lru_cache(maxsize=1)
def _relation_term_index() -> dict[str, tuple[tuple[str, str], ...]]:
    buckets: dict[str, list[tuple[str, str]]] = {}
    for term, entity_type in load_relation_terms():
        buckets.setdefault(term[0], []).append((term, entity_type))
    return {key: tuple(values) for key, values in buckets.items()}


def _augment_relation_entities(text: str, entities: list[Entity]) -> list[Entity]:
    """补充关系训练集中的主客体术语，仅用于关系抽取。"""
    result = list(entities)
    for start, end, term, entity_type in _select_non_overlapping_matches(
        text, _relation_term_index()
    ):
        if any(
            item.start_idx is not None
            and item.end_idx is not None
            and start <= item.end_idx
            and end >= item.start_idx
            for item in result
        ):
            continue
        result.append(Entity(text=term, type=entity_type, start_idx=start, end_idx=end))
    return sorted(result, key=lambda item: (item.start_idx or 0, -len(item.text)))


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in SENTENCE_RE.finditer(text or "")]


def _section_marker_position(text: str, entity: Entity) -> int:
    """Return the nearest CMeIE section marker associated with an entity."""

    start = entity.start_idx
    if start is None:
        return -1
    end = entity.end_idx if entity.end_idx is not None else start + len(entity.text) - 1
    positions = [
        text.rfind(marker, 0, start + 1)
        for marker in ("@", "###")
    ]
    # In the source format a disease heading is commonly written as
    # ``疾病@正文``; associate the heading with the marker immediately after it.
    following = text.find("@", end + 1, min(len(text), end + 4))
    if following >= 0:
        positions.append(following)
    return max(positions, default=-1)


def _offline_relation_section_is_supported(
    text: str,
    disease: Entity,
    obj: Entity,
    predicate: str,
) -> bool:
    """Avoid assigning facts from a later ``###`` section to an earlier disease."""

    disease_marker = _section_marker_position(text, disease)
    object_marker = _section_marker_position(text, obj)
    if disease_marker == object_marker:
        return True
    # Disease-to-disease links are often intentionally expressed across
    # section headings (for example differential diagnosis); keep only those
    # explicit cross-section predicates and constrain the rest.
    return predicate in {
        "鉴别诊断",
        "相关（转化）",
        "相关（导致）",
    }


def _owning_section_disease(
    text: str,
    sentence_start: int,
    diseases: list[Entity],
) -> Entity | None:
    """Return the disease heading that owns a later ``@`` section sentence."""

    marker = text.rfind("@", max(0, sentence_start - 240), sentence_start)
    if marker < 0:
        return None
    return max(
        (
            entity
            for entity in diseases
            if entity.start_idx is not None
            and entity.end_idx is not None
            and entity.end_idx < marker
            and marker - entity.end_idx <= 240
        ),
        key=lambda item: item.end_idx or -1,
        default=None,
    )


_DIAGNOSIS_HEADER_RE = re.compile(
    r"(?:初步诊断|入院诊断|出院诊断|主要诊断|临床诊断|诊断)\s*[:：]"
)


def _preceding_diagnosis_context(
    text: str,
    sentence_start: int,
    sentence_spans: list[tuple[int, int, str]],
    diseases: list[Entity],
) -> tuple[Entity | None, str]:
    """Find a nearby explicit diagnosis that owns following clinical steps."""

    checked = 0
    for prior_start, prior_end, prior_sentence in reversed(sentence_spans):
        if prior_end > sentence_start:
            continue
        if sentence_start - prior_end > 360 or checked >= 3:
            break
        if "\n\n" in text[prior_end:sentence_start]:
            break
        checked += 1
        header = _DIAGNOSIS_HEADER_RE.search(prior_sentence)
        if header is None:
            continue
        diagnosis_start = prior_start + header.end()
        candidates = [
            entity
            for entity in diseases
            if entity.start_idx is not None
            and diagnosis_start <= entity.start_idx < prior_end
        ]
        if candidates:
            primary = min(candidates, key=lambda item: item.start_idx or 0)
            return primary, prior_sentence
    return None, ""


def _is_complication_frame(sentence: str) -> bool:
    return "并发症" in sentence and any(
        cue in sentence for cue in ("并发", "监测", "警惕", "预防", "出现")
    )


def _context_relation_for(
    sentence: str,
    sentence_start: int,
    disease: Entity,
    obj: Entity,
    db_path: str,
) -> tuple[str, str]:
    """Resolve a relation owned by a diagnosis in a preceding sentence."""

    if _is_complication_frame(sentence) and obj.type in {"dis", "sym"}:
        return "并发症", "clinical_context_rule"
    object_pos = _entity_position(sentence, obj, sentence_start)
    if obj.type == "pro" and object_pos >= 0:
        prefix = sentence[max(0, object_pos - 12) : object_pos]
        if any(cue in prefix for cue in ("行", "接受", "实施", "进行")):
            predicate = (
                "手术治疗"
                if any(cue in obj.text for cue in ("手术", "介入", "切除", "移植"))
                else "辅助治疗"
            )
            return predicate, "clinical_context_rule"
    if object_pos < 0:
        return "", ""

    prefix = disease.text + "@"
    local_object_start = object_pos
    local_object_end = local_object_start + len(obj.text) - 1
    contextual_object = replace(
        obj,
        start_idx=len(prefix) + local_object_start,
        end_idx=len(prefix) + local_object_end,
    )
    contextual_disease = replace(
        disease,
        start_idx=0,
        end_idx=len(disease.text) - 1,
    )
    return _relation_for(
        prefix + sentence,
        contextual_disease,
        contextual_object,
        db_path,
        sentence_start=0,
    )


def _primary_diseases(text: str, entities: list[Entity]) -> list[Entity]:
    diseases = [entity for entity in entities if entity.type == "dis"]
    if not diseases:
        return []
    diagnosis_pos = text.find("诊断")
    if diagnosis_pos >= 0:
        near = [entity for entity in diseases if (entity.start_idx or 0) >= diagnosis_pos]
        if near:
            return near[:3]
    return diseases[:3]


def _known_relation(db_path: str, subject: str, obj: str) -> str:
    trained = load_known_relation_pairs().get((subject, obj), "")
    if trained:
        return trained
    path = Path(db_path) if db_path else None
    if not path or not path.exists():
        return ""
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute(
            """
            SELECT t.relation_code
            FROM kg_triples t
            JOIN kg_entities s ON s.entity_id = t.subject_id
            JOIN kg_entities o ON o.entity_id = t.object_id
            WHERE s.canonical_name = ? AND o.canonical_name = ?
            GROUP BY t.relation_code
            ORDER BY COUNT(*) DESC
            LIMIT 1
            """,
            (subject, obj),
        ).fetchone()
    finally:
        conn.close()
    return KG_TO_RELATION_TYPE.get(row[0], "") if row else ""


def _entity_position(
    sentence: str,
    entity: Entity,
    sentence_start: int = 0,
) -> int:
    """Resolve an entity mention to its actual occurrence in this sentence."""

    if entity.start_idx is not None:
        relative = entity.start_idx - sentence_start
        if (
            0 <= relative < len(sentence)
            and sentence.startswith(entity.text, relative)
        ):
            return relative
    return sentence.find(entity.text)


def _is_negated(
    sentence: str,
    entity: Entity,
    relative_position: int | None = None,
) -> bool:
    relative = (
        relative_position
        if relative_position is not None
        else sentence.find(entity.text)
    )
    if relative < 0:
        return False
    prefix = sentence[max(0, relative - 24):relative].strip()
    return bool(NEGATION_RE.search(prefix) or NEGATION_CONTEXT_RE.search(prefix))


def _relation_for(
    sentence: str,
    disease: Entity,
    obj: Entity,
    db_path: str = "",
    sentence_start: int = 0,
) -> tuple[str, str]:
    disease_pos = _entity_position(sentence, disease, sentence_start)
    object_pos = _entity_position(sentence, obj, sentence_start)
    if disease_pos < 0 or object_pos < 0:
        return "", ""
    if _is_negated(sentence, obj, object_pos):
        return "", ""

    # CMeIE 中主体通常先于客体。禁止同句反向枚举，避免两个疾病、
    # 检查或药物因为共享一个触发词而互相生成关系。
    if object_pos <= disease_pos:
        return "", ""

    prefix = sentence[:object_pos]
    clause_start = max(
        (prefix.rfind(boundary) for boundary in _RELATION_CLAUSE_BOUNDARIES),
        default=-1,
    ) + 1
    clause_prefix = prefix[clause_start:]
    disease_end = disease_pos + len(disease.text)

    context_cues = [
        ("\u75c5\u56e0", _CAUSAL_CONTEXT_CUES),
        (
            "\u53d1\u75c5\u90e8\u4f4d",
            tuple(
                cue
                for cue in _BODY_SITE_CONTEXT_CUES
                if cue not in ("\u5728", "\u8f6c\u79fb\u81f3", "\u4fb5\u72af")
            ),
        ),
        ("\u5916\u4fb5\u90e8\u4f4d", ("\u4fb5\u72af",)),
        ("\u8f6c\u79fb\u90e8\u4f4d", ("\u8f6c\u79fb\u81f3",)),
    ]
    cue_positions = [
        (predicate, max((clause_prefix.rfind(cue) for cue in cues), default=-1))
        for predicate, cues in context_cues
    ]
    for predicate, cue_position in sorted(
        (item for item in cue_positions if item[1] >= 0),
        key=lambda item: item[1],
        reverse=True,
    ):
        absolute_cue = clause_start + cue_position
        if (
            absolute_cue >= disease_end
            and 0 <= object_pos - absolute_cue <= 96
        ):
            return predicate, "context_rule"

    # Require a local body-part phrase before treating ? as a location cue.
    in_position = clause_prefix.rfind("\u5728")
    if obj.type == "bod" and in_position >= 0:
        absolute_in = clause_start + in_position
        between = sentence[absolute_in:object_pos + len(obj.text)]
        if (
            absolute_in >= disease_end
            and 0 <= object_pos - absolute_in <= 48
            and (
                object_pos - absolute_in <= 16
                or any(cue in between for cue in ("\u5982", "\u5305\u62ec", "\u4f8b\u5982"))
            )
        ):
            return "\u53d1\u75c5\u90e8\u4f4d", "context_rule"

    # An explicit local relation frame is stronger than a corpus-level pair
    # mapping. This prevents a globally frequent pair from overriding the
    # relation stated in the current sentence.
    known = _known_relation(db_path, disease.text, obj.text)
    if known and abs(disease_pos - object_pos) <= 96:
        return known, "known_pair"

    left = max(0, min(disease_pos, object_pos) - 8)
    right = min(
        len(sentence),
        max(disease_pos + len(disease.text), object_pos + len(obj.text)) + 8,
    )
    local_context = sentence[left:right]
    between = sentence[disease_end:object_pos]
    if obj.type == "dis":
        for predicate, cues in RELATION_CUES.items():
            if any(cue in between for cue in cues):
                return predicate, "sentence_rule"
        return "", ""

    distance = abs(disease_pos - object_pos)
    if distance > (28 if obj.type == "sym" else 48):
        return "", ""
    if not any(cue in between for cue in TYPE_RELATION_CUES.get(obj.type, ())):
        return "", ""
    if obj.type == "ite":
        if any(cue in obj.text for cue in ("\u0043\u0054", "\u004d\u0052\u0049", "\u8d85\u58f0", "\u5f71\u50cf", "\u0058\u7ebf", "\u80f8\u7247")):
            return "\u5f71\u50cf\u5b66\u68c0\u67e5", "sentence_rule"
        if any(cue in obj.text for cue in ("\u955c", "\u5185\u7aa5")):
            return "\u5185\u7aa5\u955c\u68c0\u67e5", "sentence_rule"
        if any(cue in obj.text for cue in ("\u75c5\u7406", "\u6d3b\u68c0", "\u7ec4\u7ec7\u5b66")):
            return "\u7ec4\u7ec7\u5b66\u68c0\u67e5", "sentence_rule"
        return "\u5b9e\u9a8c\u5ba4\u68c0\u67e5", "sentence_rule"
    if obj.type == "pro" and any(
        cue in obj.text
        for cue in ("\u624b\u672f", "\u5207\u9664", "\u79fb\u690d", "\u543b\u5408")
    ):
        return "\u624b\u672f\u6cbb\u7597", "sentence_rule"
    method = "sentence_rule"
    if obj.type == "dru" and any(
        cue in between for cue in ("给予", "口服", "服用", "注射", "应用")
    ):
        method = "explicit_medication_frame"
    return TYPE_TO_DEFAULT_RELATION.get(obj.type, ""), method


def _make_relation(disease: Entity, obj: Entity, predicate: str, method: str, evidence: str) -> Relation:
    reliability = reliability_for("relation", method, predicate)
    return Relation(
        subject=disease.text,
        subject_type=disease.type,
        predicate=predicate,
        object=obj.text,
        object_type=obj.type,
        confidence=reliability.score,
        evidence=evidence.strip()[:500],
        extraction_method=method,
        reliability_level=reliability.level,
    )


_STRUCTURED_LIST_RE = re.compile(r"（(?:如|包括|例如)([^）]{1,96})）")


def _extract_explicit_section_relations(
    text: str,
    entities: list[Entity],
) -> list[Relation]:
    """Extract narrow, explicit list frames after a disease section marker.

    Medical source text and CMeIE both serialize headings as ``疾病@正文``.
    Parenthetical examples attached directly to a body-site or cause frame are
    substantially stronger than generic sentence co-occurrence.  This rule is
    deliberately limited to entities already found in the exact source span;
    unknown terms remain the LLM gap path's responsibility.
    """

    relations: list[Relation] = []
    seen: set[tuple[str, str, str]] = set()
    for marker_match in re.finditer(r"@", text):
        marker = marker_match.start()
        subject = max(
            (
                entity
                for entity in entities
                if entity.type == "dis"
                and entity.start_idx is not None
                and entity.end_idx is not None
                and entity.end_idx < marker
                and marker - entity.end_idx <= 240
            ),
            key=lambda item: item.end_idx or -1,
            default=None,
        )
        if subject is None:
            continue
        section_end_candidates = [
            position
            for position in (
                text.find("@", marker + 1),
                text.find("\n", marker + 1),
            )
            if position >= 0
        ]
        section_end = min(section_end_candidates, default=len(text))
        section_start = marker + 1
        section = text[section_start:section_end]
        for list_match in _STRUCTURED_LIST_RE.finditer(section):
            prefix = section[max(0, list_match.start() - 32) : list_match.start()]
            group_start = section_start + list_match.start(1)
            group_end = section_start + list_match.end(1)
            specs: list[tuple[str, int, int]] = []
            if "病因" in prefix:
                specs.append(("病因", group_start, group_end))
            in_position = prefix.rfind("在")
            if in_position >= 0:
                frame_start = section_start + max(0, list_match.start() - 32) + in_position
                frame_end = section_start + list_match.end()
                if any(
                    entity.type == "bod"
                    and entity.start_idx is not None
                    and frame_start <= entity.start_idx < frame_end
                    for entity in entities
                ):
                    specs.append(("发病部位", frame_start, frame_end))

            for predicate, frame_start, frame_end in specs:
                for obj in entities:
                    if obj.start_idx is None or not frame_start <= obj.start_idx < frame_end:
                        continue
                    if obj.text == subject.text or obj.type == "dis" and predicate == "发病部位":
                        continue
                    if not _offline_relation_object_type_is_compatible(predicate, obj.type):
                        continue
                    key = (subject.text, predicate, obj.text)
                    if key in seen:
                        continue
                    seen.add(key)
                    relations.append(
                        _make_relation(
                            subject,
                            obj,
                            predicate,
                            "explicit_section_frame",
                            section,
                        )
                    )
    return relations


def extract_relations_offline(
    text: str,
    entities: list[Entity] | None = None,
    db_path: str = "",
) -> list[Relation]:
    """基于本地规则和知识图谱实体对抽取疾病中心关系。"""
    if not text or not text.strip():
        return []
    entities = entities if entities is not None else extract_entities_offline(text, db_path)
    entities = _augment_relation_entities(text, entities)
    diseases = [entity for entity in entities if entity.type == "dis"]
    if not diseases:
        return []

    relations: list[Relation] = []
    seen: set[tuple[str, str, str]] = set()

    sentence_spans = _sentence_spans(text)
    for sent_start, sent_end, sentence in sentence_spans:
        sent_diseases = [
            entity
            for entity in diseases
            if entity.start_idx is not None and sent_start <= entity.start_idx < sent_end
        ]
        section_disease = None
        if not sent_diseases:
            section_disease = _owning_section_disease(text, sent_start, diseases)
        diagnosis_disease, diagnosis_evidence = _preceding_diagnosis_context(
            text,
            sent_start,
            sentence_spans,
            diseases,
        )
        use_diagnosis_context = diagnosis_disease is not None and (
            not sent_diseases or _is_complication_frame(sentence)
        )
        if not sent_diseases and section_disease is None and not use_diagnosis_context:
            continue
        sent_entities = [
            entity
            for entity in entities
            if entity.start_idx is not None and sent_start <= entity.start_idx < sent_end
        ]
        active_diseases = (
            [diagnosis_disease]
            if use_diagnosis_context
            else (sent_diseases[:2] or [section_disease])
        )
        for disease in active_diseases:
            if disease is None:
                continue
            for obj in sent_entities:
                if obj.text == disease.text:
                    continue
                relation_sentence = sentence
                relation_disease = disease
                relation_object = obj
                relation_sentence_start = sent_start
                inherited_diagnosis = (
                    use_diagnosis_context and disease is diagnosis_disease
                )
                if section_disease is disease and disease not in sent_diseases:
                    prefix = disease.text + "@"
                    object_end = (
                        obj.end_idx
                        if obj.end_idx is not None
                        else obj.start_idx + len(obj.text) - 1
                    )
                    relation_sentence = prefix + sentence
                    relation_disease = replace(
                        disease,
                        start_idx=0,
                        end_idx=len(disease.text) - 1,
                    )
                    relation_object = replace(
                        obj,
                        start_idx=len(prefix) + (obj.start_idx - sent_start),
                        end_idx=len(prefix) + (object_end - sent_start),
                    )
                    relation_sentence_start = 0
                if inherited_diagnosis:
                    predicate, method = _context_relation_for(
                        sentence,
                        sent_start,
                        disease,
                        obj,
                        db_path,
                    )
                else:
                    predicate, method = _relation_for(
                        relation_sentence,
                        relation_disease,
                        relation_object,
                        db_path,
                        sentence_start=relation_sentence_start,
                    )
                if not predicate:
                    continue
                if not _offline_relation_object_type_is_compatible(
                    predicate, obj.type
                ):
                    continue
                if not _offline_relation_context_is_supported(
                    predicate, relation_disease, relation_object, relation_sentence
                ):
                    continue
                if not _offline_relation_section_is_supported(
                    text, disease, obj, predicate
                ):
                    continue
                key = (disease.text, predicate, obj.text)
                if key in seen:
                    continue
                seen.add(key)
                evidence = (
                    f"{diagnosis_evidence}；{sentence}"
                    if inherited_diagnosis
                    else (
                        relation_sentence
                        if section_disease is disease and disease not in sent_diseases
                        else sentence
                    )
                )
                relations.append(_make_relation(disease, obj, predicate, method, evidence))

    for relation in _extract_explicit_section_relations(text, entities):
        key = (relation.subject, relation.predicate, relation.object)
        if key in seen:
            for index, existing in enumerate(relations):
                if (
                    existing.subject,
                    existing.predicate,
                    existing.object,
                ) == key and existing.reliability_level == "low":
                    relations[index] = relation
                    break
            continue
        seen.add(key)
        relations.append(relation)

    return relations


def generate_triples_offline(
    text: str,
    entities: list[Entity] | None = None,
    relations: list[Relation] | None = None,
    db_path: str = "",
) -> list[Triple]:
    """根据本地关系抽取结果生成三元组。"""
    entities = entities if entities is not None else extract_entities_offline(text, db_path)
    relations = relations if relations is not None else extract_relations_offline(text, entities, db_path)
    return relations_to_triples(relations, min_confidence=0.0)
