# -*- coding: utf-8 -*-
"""
医学抽取结果校验模块。

该模块校验实体、关系和三元组结构，过滤缺字段、类型不合法或明显冲突的结果。
"""

from __future__ import annotations

from typing import Any, Iterable

from .schemas import Entity, Relation, Triple
from .medical_reliability import reliability_for


ENTITY_TYPES = {
    "dis",
    "sym",
    "dru",
    "equ",
    "pro",
    "bod",
    "ite",
    "mic",
    "dep",
}

ENTITY_TYPE_ALIASES = {
    "疾病": "dis",
    "disease": "dis",
    "症状": "sym",
    "症状体征": "sym",
    "symptom": "sym",
    "药物": "dru",
    "药品": "dru",
    "drug": "dru",
    "医疗设备": "equ",
    "设备": "equ",
    "equipment": "equ",
    "医疗程序": "pro",
    "治疗": "pro",
    "治疗方法": "pro",
    "手术": "pro",
    "操作": "pro",
    "procedure": "pro",
    "身体部位": "bod",
    "部位": "bod",
    "body": "bod",
    "检验项目": "ite",
    "检查项目": "ite",
    "检查": "ite",
    "检验": "ite",
    "test": "ite",
    "微生物": "mic",
    "microorganism": "mic",
    "科室": "dep",
    "department": "dep",
}

CMEIE_RELATION_TYPES = {
    "临床表现",
    "传播途径",
    "侵及周围组织转移的症状",
    "内窥镜检查",
    "化疗",
    "发病年龄",
    "发病性别倾向",
    "发病机制",
    "发病率",
    "发病部位",
    "同义词",
    "外侵部位",
    "多发地区",
    "多发季节",
    "多发群体",
    "实验室检查",
    "就诊科室",
    "并发症",
    "影像学检查",
    "手术治疗",
    "放射治疗",
    "死亡率",
    "治疗后症状",
    "病史",
    "病因",
    "病理分型",
    "病理生理",
    "相关（导致）",
    "相关（症状）",
    "相关（转化）",
    "筛查",
    "组织学检查",
    "药物治疗",
    "转移部位",
    "辅助检查",
    "辅助治疗",
    "遗传因素",
    "鉴别诊断",
    "阶段",
    "预后状况",
    "预后生存率",
    "预防",
    "风险评估因素",
    "高危因素",
}

RELATION_ALIASES = {
    "症状": "临床表现",
    "临床症状": "临床表现",
    "治疗": "辅助治疗",
    "治疗方式": "辅助治疗",
    "检查": "辅助检查",
    "诊断": "辅助检查",
    "风险因素": "高危因素",
    "危险因素": "高危因素",
    "所属科室": "就诊科室",
    "科室": "就诊科室",
    "预后": "预后状况",
    "转移": "转移部位",
}

RELATION_EVIDENCE_CUES = {
    "临床表现": ("症状", "表现", "伴有", "伴随", "出现", "可见", "主诉", "有"),
    "传播途径": ("传播", "途径", "飞沫", "接触", "经", "通过"),
    "内窥镜检查": ("内窥镜", "内镜", "胃镜", "肠镜"),
    "发病年龄": ("年龄", "岁", "儿童", "老年", "青年"),
    "发病性别倾向": ("男性", "女性", "性别"),
    "发病率": ("发病率", "发生率", "患病率", "比例", "%"),
    "发病部位": ("部位", "发生于", "好发于", "位于", "累及", "侵犯", "在"),
    "同义词": ("又称", "别名", "简称", "即", "英文名", "英文"),
    "外侵部位": ("侵犯", "侵及", "浸润"),
    "多发地区": ("地区", "地域", "流行", "多发"),
    "多发季节": ("季节", "春", "夏", "秋", "冬"),
    "多发群体": ("人群", "男性", "女性", "儿童", "老人", "多发", "好发"),
    "实验室检查": ("实验室", "血液", "血清", "检测", "检查", "测定", "化验"),
    "并发症": ("并发", "合并", "可致", "有", "出现", "可见"),
    "影像学检查": ("影像", "CT", "MRI", "超声", "X线", "扫描", "检查"),
    "手术治疗": ("手术", "介入", "切除", "移植", "引流", "吻合", "实施"),
    "放射治疗": ("放疗", "放射", "照射"),
    "死亡率": ("死亡率", "病死率", "死亡"),
    "治疗后症状": ("治疗后", "术后", "治疗后出现"),
    "病因": ("病因", "原因", "由于", "引起", "导致", "病原体", "感染"),
    "病理分型": ("病理分型", "分型", "分为", "分类", "类型", "亚型"),
    "相关（导致）": ("导致", "引起", "造成", "相关", "转化"),
    "相关（症状）": ("相关症状", "症状", "表现", "伴随"),
    "相关（转化）": ("转化", "转变", "演变", "发展为"),
    "筛查": ("筛查", "筛选"),
    "组织学检查": ("组织学", "病理", "活组织", "活检"),
    "药物治疗": ("治疗", "用药", "服用", "口服", "给予", "注射", "使用", "首选"),
    "转移部位": ("转移", "转移至", "转移到"),
    "辅助检查": ("检查", "检测", "监测", "复查", "测定", "诊断"),
    "辅助治疗": ("治疗", "手术", "处理", "干预", "切除", "移植", "放疗", "化疗"),
    "鉴别诊断": ("鉴别", "排除", "区分", "诊断"),
    "预后状况": ("预后", "结局", "恢复"),
    "预后生存率": ("生存率", "存活率", "预后"),
    "预防": ("预防", "避免", "防止", "接种"),
    "风险评估因素": ("风险", "危险因素", "相关因素", "相关", "因素"),
    "高危因素": ("高危", "危险因素", "风险因素", "风险"),
}


def normalize_entity_type(value: Any) -> str:
    raw = str(value or "").strip()
    normalized = ENTITY_TYPE_ALIASES.get(raw, raw.lower())
    return normalized if normalized in ENTITY_TYPES else ""


def normalize_relation_type(value: Any) -> str:
    raw = str(value or "").strip()
    normalized = RELATION_ALIASES.get(raw, raw)
    return normalized if normalized in CMEIE_RELATION_TYPES else ""


def relation_evidence_supports_predicate(text: str, predicate: str) -> bool:
    """Reject LLM relation labels with no predicate-specific textual cue.

    Entity and endpoint existence alone cannot distinguish, for example,
    ``相关（导致）`` from a merely adjacent medical mention.  The cue check
    is intentionally permissive and only applies to predicates with a known
    lexical frame; unknown future labels remain governed by the normal schema
    and LLM review gates.
    """

    if not any("\u4e00" <= char <= "\u9fff" for char in text):
        return True
    cues = RELATION_EVIDENCE_CUES.get(predicate)
    return not cues or any(cue in text for cue in cues)


_STRICT_RELATION_CUES = {
    "\u4e34\u5e8a\u8868\u73b0": ("\u75c7\u72b6", "\u8868\u73b0", "\u4f34\u6709", "\u51fa\u73b0"),
    "\u53d1\u75c5\u90e8\u4f4d": (
        "\u53d1\u75c5\u90e8\u4f4d",
        "\u53d1\u751f\u4e8e",
        "\u597d\u53d1\u4e8e",
        "\u4f4d\u4e8e",
        "\u7d2f\u53ca",
        "\u4fb5\u72af",
        "\u5728",
    ),
    "\u5916\u4fb5\u90e8\u4f4d": ("\u4fb5\u72af", "\u4fb5\u53ca", "\u6d78\u6da6"),
    "\u8f6c\u79fb\u90e8\u4f4d": ("\u8f6c\u79fb",),
    "\u75c5\u56e0": (
        "\u75c5\u56e0",
        "\u539f\u56e0",
        "\u7531\u4e8e",
        "\u5f15\u8d77",
        "\u5bfc\u81f4",
        "\u8bf1\u53d1",
        "\u75c5\u539f\u4f53",
        "\u611f\u67d3",
    ),
    "\u5e76\u53d1\u75c7": ("\u5e76\u53d1", "\u5408\u5e76", "\u51fa\u73b0"),
    "\u836f\u7269\u6cbb\u7597": ("\u6cbb\u7597", "\u7528\u836f", "\u670d\u7528", "\u7ed9\u4e88", "\u4f7f\u7528", "\u5e94\u7528", "\u53e3\u670d", "\u6ce8\u5c04"),
    "\u8f85\u52a9\u6cbb\u7597": ("\u6cbb\u7597", "\u5904\u7406", "\u5e72\u9884", "\u624b\u672f"),
    "\u624b\u672f\u6cbb\u7597": (
        "\u624b\u672f", "\u4ecb\u5165", "\u5207\u9664", "\u79fb\u690d", "\u5f15\u6d41", "\u543b\u5408", "\u5b9e\u65bd", "\u884c"
    ),
    "\u8f85\u52a9\u68c0\u67e5": ("\u68c0\u67e5", "\u68c0\u6d4b", "\u76d1\u6d4b", "\u590d\u67e5"),
    "\u5b9e\u9a8c\u5ba4\u68c0\u67e5": ("\u5b9e\u9a8c\u5ba4", "\u8840\u6db2", "\u68c0\u6d4b", "\u68c0\u67e5"),
    "\u5f71\u50cf\u5b66\u68c0\u67e5": ("\u5f71\u50cf", "CT", "MRI", "\u8d85\u58f0", "X\u7ebf", "\u626b\u63cf", "\u68c0\u67e5"),
}

_STRICT_RELATION_OBJECT_TYPES = {
    "\u53d1\u75c5\u90e8\u4f4d": {"bod"},
    "\u5916\u4fb5\u90e8\u4f4d": {"bod"},
    "\u8f6c\u79fb\u90e8\u4f4d": {"bod"},
    "\u836f\u7269\u6cbb\u7597": {"dru"},
    "\u624b\u672f\u6cbb\u7597": {"pro"},
    "\u8f85\u52a9\u68c0\u67e5": {"ite"},
    "\u5b9e\u9a8c\u5ba4\u68c0\u67e5": {"ite"},
    "\u5f71\u50cf\u5b66\u68c0\u67e5": {"ite"},
    "\u4e34\u5e8a\u8868\u73b0": {"sym", "bod"},
    "\u5e76\u53d1\u75c7": {"dis", "sym"},
}

_STRICT_FORWARD_RELATIONS = {
    "\u4e34\u5e8a\u8868\u73b0",
    "\u53d1\u75c5\u90e8\u4f4d",
    "\u5916\u4fb5\u90e8\u4f4d",
    "\u8f6c\u79fb\u90e8\u4f4d",
    "\u75c5\u56e0",
    "\u5e76\u53d1\u75c7",
    "\u75c5\u7406\u5206\u578b",
}


def relation_evidence_supports_pair(
    text: str,
    predicate: str,
    subject: str,
    object_value: str,
    *,
    subject_type: str = "",
    object_type: str = "",
    require_disease_subject: bool = False,
    max_distance: int = 128,
) -> bool:
    """Require a predicate cue in the local span connecting both endpoints.

    The previous validation checked the whole segment.  That allowed a cue
    belonging to an unrelated fact to authorize any other pair in the same
    sentence.  Cascade gap relations use this stricter local check.
    """

    if not subject or not object_value or subject not in text or object_value not in text:
        return False
    if subject == object_value or object_value == predicate:
        return False
    if require_disease_subject and subject_type != "dis":
        return False
    allowed_types = _STRICT_RELATION_OBJECT_TYPES.get(predicate)
    if allowed_types and object_type and object_type not in allowed_types:
        return False
    if not any("\u4e00" <= char <= "\u9fff" for char in text):
        return True

    cues = _STRICT_RELATION_CUES.get(predicate)
    if not cues:
        return False
    subject_spans = list(_all_occurrences(text, subject))
    object_spans = list(_all_occurrences(text, object_value))
    for subject_start, subject_end in subject_spans:
        for object_start, object_end in object_spans:
            if predicate in _STRICT_FORWARD_RELATIONS and subject_start > object_start:
                continue
            start = min(subject_start, object_start)
            end = max(subject_end, object_end)
            if end - start + 1 > max_distance:
                continue
            local_text = text[start : end + 1]
            if any(cue in local_text for cue in cues):
                return True
            if predicate == "并发症":
                # A shared trailing label is common in clinical notes, for
                # example “心力衰竭、心律失常等并发症”.
                trailing = text[end + 1 : min(len(text), end + 25)]
                if any(cue in trailing for cue in cues):
                    return True
    return False


def _all_occurrences(text: str, value: str) -> Iterable[tuple[int, int]]:
    start = text.find(value)
    while start >= 0:
        yield start, start + len(value) - 1
        start = text.find(value, start + 1)


def _resolve_entity_overlaps(entities: list[Entity]) -> list[Entity]:
    """Remove duplicate spans while retaining CMeEE nested symptom mentions."""

    positioned = [item for item in entities if item.start_idx is not None]
    unpositioned = [item for item in entities if item.start_idx is None]
    selected: list[Entity] = []
    for entity in sorted(
        positioned,
        key=lambda item: (
            -(len(item.text)),
            item.start_idx or 0,
            -float(item.confidence),
        ),
    ):
        entity_end = entity.end_idx if entity.end_idx is not None else entity.start_idx
        if entity_end is None:
            selected.append(entity)
            continue
        overlaps = [
            other
            for other in selected
            if other.start_idx is not None
            and entity.start_idx <= (other.end_idx if other.end_idx is not None else other.start_idx)
            and entity_end >= other.start_idx
        ]
        if not overlaps:
            selected.append(entity)
            continue
        can_keep_nested = any(
            entity.type != other.type
            and (
                (
                    other.type == "sym"
                    and other.start_idx <= entity.start_idx
                    and entity_end <= (other.end_idx if other.end_idx is not None else other.start_idx)
                )
                or (
                    entity.type == "sym"
                    and entity.start_idx <= other.start_idx
                    and (other.end_idx if other.end_idx is not None else other.start_idx) <= entity_end
                )
            )
            for other in overlaps
        )
        if not can_keep_nested:
            continue
        selected.append(entity)
    return sorted(
        selected + unpositioned,
        key=lambda item: (item.start_idx is None, item.start_idx or 0, -len(item.text)),
    )


def validate_entities(text: str, raw_items: Any) -> list[Entity]:
    if not isinstance(raw_items, list):
        return []

    entities: list[Entity] = []
    seen: set[tuple[int, int, str]] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        value = str(item.get("text", item.get("entity", "")) or "").strip()
        entity_type = normalize_entity_type(item.get("type"))
        if not value or not entity_type or value not in text:
            continue

        positions: list[tuple[int, int]] = []
        try:
            start = int(item.get("start_idx", item.get("start")))
            end = int(item.get("end_idx", item.get("end")))
        except (TypeError, ValueError):
            start = end = -1
        if 0 <= start <= end < len(text) and text[start : end + 1] == value:
            positions.append((start, end))
        else:
            positions.extend(_all_occurrences(text, value))

        reliability = reliability_for("entity", "llm", entity_type)
        try:
            confidence = min(1.0, max(0.0, float(item.get("confidence", reliability.score))))
        except (TypeError, ValueError):
            confidence = reliability.score

        for start, end in positions:
            key = (start, end, entity_type)
            if key in seen:
                continue
            seen.add(key)
            left = max(0, start - 20)
            right = min(len(text), end + 21)
            entities.append(
                Entity(
                    text=value,
                    type=entity_type,
                    start_idx=start,
                    end_idx=end,
                    confidence=confidence,
                    evidence=text[left:right],
                    extraction_method="llm",
                    reliability_level=reliability.level,
                )
            )
    return _resolve_entity_overlaps(entities)


def validate_relations(
    text: str,
    raw_items: Any,
    entities: list[Entity] | None = None,
) -> list[Relation]:
    if not isinstance(raw_items, list):
        return []

    entity_types: dict[str, str] = {}
    for entity in entities or []:
        entity_types.setdefault(entity.text, entity.type)

    relations: list[Relation] = []
    seen: set[tuple[str, str, str]] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject", "") or "").strip()
        object_value = item.get("object", "")
        if isinstance(object_value, dict):
            object_value = object_value.get("@value", object_value.get("value", ""))
        obj = str(object_value or "").strip()
        predicate = normalize_relation_type(item.get("predicate"))
        if not subject or not obj or not predicate:
            continue
        if subject not in text or obj not in text:
            continue
        if entities and (subject not in entity_types or obj not in entity_types):
            continue
        if not relation_evidence_supports_predicate(text, predicate):
            continue

        key = (subject, predicate, obj)
        if key in seen:
            continue
        seen.add(key)
        reliability = reliability_for("relation", "llm", predicate)
        try:
            confidence = min(1.0, max(0.0, float(item.get("confidence", reliability.score))))
        except (TypeError, ValueError):
            confidence = reliability.score
        subject_type = entity_types.get(
            subject, normalize_entity_type(item.get("subject_type"))
        )
        object_type = entity_types.get(
            obj, normalize_entity_type(item.get("object_type"))
        )
        relations.append(
            Relation(
                subject=subject,
                predicate=predicate,
                object=obj,
                subject_type=subject_type,
                object_type=object_type,
                confidence=confidence,
                evidence=text[:500],
                extraction_method="llm",
                reliability_level=reliability.level,
            )
        )
    return relations


def relations_to_triples(
    relations: Iterable[Relation],
    min_confidence: float = 0.7,
) -> list[Triple]:
    triples: list[Triple] = []
    seen: set[tuple[str, str, str]] = set()
    for relation in relations:
        key = (relation.subject, relation.predicate, relation.object)
        if relation.confidence < min_confidence or key in seen:
            continue
        seen.add(key)
        triples.append(
            Triple(
                subject=relation.subject,
                predicate=relation.predicate,
                object=relation.object,
                confidence=relation.confidence,
                subject_type=relation.subject_type,
                object_type=relation.object_type,
                evidence=relation.evidence,
                extraction_method=relation.extraction_method,
                reliability_level=relation.reliability_level,
            )
        )
    return triples
