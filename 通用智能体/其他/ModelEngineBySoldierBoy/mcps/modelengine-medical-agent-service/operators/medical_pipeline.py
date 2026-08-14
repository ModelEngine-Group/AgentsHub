"""Rule-based medical demo pipeline.

The project currently has no packaged model backend in this repository, so this
module provides a deterministic baseline that can run locally and generate the
artifact set described in the docs:

- processed documents
- entities
- triples
- graph snapshot
- analysis summary
- QA traces
"""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Protocol

from .data_processing import clean_text, load_records, normalize_fields, split_sentences
from .llm_client import chat_completion, get_openai_compat_config


DATA_SUFFIXES = (
    "病例摘要",
    "门诊记录",
    "随访摘要",
    "随访",
    "急诊摘要",
    "记录",
    "病例",
)


def _sorted_unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def strip_title_suffix(title: str) -> str:
    stripped = title.strip()
    for suffix in DATA_SUFFIXES:
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)]
            break
    return stripped.strip()


@dataclass(frozen=True)
class MedicalDocument:
    doc_id: str
    title: str
    source: str
    text: str


@dataclass
class EntityRecord:
    name: str
    type: str
    aliases: set[str] = field(default_factory=set)
    source_doc_ids: set[str] = field(default_factory=set)
    confidence: float = 0.0

    def add_alias(self, alias: str) -> None:
        if alias:
            self.aliases.add(alias)

    def add_source(self, doc_id: str) -> None:
        if doc_id:
            self.source_doc_ids.add(doc_id)


@dataclass(frozen=True)
class TripleRecord:
    subject: str
    subject_type: str
    predicate: str
    object: str
    object_type: str
    source_doc_id: str
    evidence: str
    confidence: float

    def dedup_key(self) -> tuple[str, str, str, str]:
        return (self.subject, self.predicate, self.object, self.source_doc_id)


@dataclass(frozen=True)
class ExtractionBackendResult:
    backend: str
    version: str
    entities: list[dict[str, Any]]
    triples: list[dict[str, Any]]
    main_disease: str
    disease_alias: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class KGBackendSelection:
    requested_backend: str
    selected_backend: str
    llm_configured: bool
    available_backends: list[str]
    fallback_reason: str | None = None


class KGExtractionBackend(Protocol):
    name: str
    version: str

    def run(self, document: MedicalDocument) -> ExtractionBackendResult:
        ...


@dataclass
class MedicalGraph:
    documents: list[dict[str, Any]]
    entities: list[dict[str, Any]]
    triples: list[dict[str, Any]]
    graph_snapshot: dict[str, Any]
    qa_traces: list[dict[str, Any]]
    analysis_report: str
    charts: dict[str, Any]


DISEASE_PATTERNS: list[tuple[str, str]] = sorted(
    [
        ("原发性高血压", "原发性高血压"),
        ("高血压", "原发性高血压"),
        ("2型糖尿病", "2型糖尿病"),
        ("糖尿病", "2型糖尿病"),
        ("社区获得性肺炎", "社区获得性肺炎"),
        ("慢性阻塞性肺疾病", "慢性阻塞性肺疾病"),
        ("支气管哮喘", "支气管哮喘"),
        ("胃食管反流病", "胃食管反流病"),
        ("消化性溃疡", "消化性溃疡"),
        ("缺铁性贫血", "缺铁性贫血"),
        ("甲状腺功能亢进", "甲状腺功能亢进"),
        ("甲亢", "甲状腺功能亢进"),
        ("甲状腺功能减退", "甲状腺功能减退"),
        ("甲减", "甲状腺功能减退"),
        ("痛风急性发作", "痛风急性发作"),
        ("痛风", "痛风急性发作"),
        ("慢性肾脏病", "慢性肾脏病"),
        ("心力衰竭", "心力衰竭"),
        ("急性胆囊炎", "急性胆囊炎"),
        ("泌尿系感染", "泌尿系感染"),
        ("尿路感染", "泌尿系感染"),
        ("骨质疏松", "骨质疏松"),
        ("抑郁障碍", "抑郁障碍"),
        ("抑郁症", "抑郁障碍"),
        ("冠心病", "冠心病"),
        ("脑卒中", "脑卒中"),
        ("糖尿病肾病", "糖尿病肾病"),
        ("严重肾功能不全", "严重肾功能不全"),
    ],
    key=lambda item: len(item[0]),
    reverse=True,
)

DISEASE_ALIAS_MAP = defaultdict(list)
for alias, canonical in DISEASE_PATTERNS:
    DISEASE_ALIAS_MAP[canonical].append(alias)

SYMPTOMS = sorted(
    [
        "口渴",
        "多饮",
        "多尿",
        "体重下降",
        "头痛",
        "头晕",
        "胸痛",
        "胸闷",
        "喘息",
        "气促",
        "咳嗽",
        "咳痰",
        "发热",
        "鼻塞",
        "流涕",
        "咽痛",
        "低热",
        "反酸",
        "烧心",
        "上腹痛",
        "言语不清",
        "肢体无力",
        "面色苍白",
        "乏力",
        "怕热",
        "多汗",
        "怕冷",
        "便秘",
        "红肿热痛",
        "尿频",
        "尿急",
        "尿痛",
        "腰背痛",
        "情绪低落",
        "兴趣减退",
        "睡眠障碍",
        "恶心",
        "下肢水肿",
        "夜间阵发性呼吸困难",
        "活动后气短",
        "活动后胸痛",
        "活动后胸闷",
        "夜间及运动后加重",
        "墨菲征阳性",
    ],
    key=len,
    reverse=True,
)

TESTS = sorted(
    [
        "空腹血糖",
        "糖化血红蛋白",
        "心电图",
        "冠脉CTA",
        "肺功能检查",
        "胸片",
        "血常规",
        "C反应蛋白",
        "痰培养",
        "胃镜",
        "头颅CT",
        "头颅MRI",
        "血红蛋白",
        "铁蛋白",
        "甲状腺功能检查",
        "FT3",
        "FT4",
        "TSH",
        "血尿酸",
        "血肌酐",
        "估算肾小球滤过率",
        "心脏超声",
        "腹部超声",
        "尿常规",
        "尿培养",
        "骨密度检查",
        "T值",
        "尿蛋白",
        "肾小球滤过率",
        "血压",
    ],
    key=len,
    reverse=True,
)

DRUGS = sorted(
    [
        "二甲双胍",
        "胰岛素",
        "钙通道阻滞剂",
        "ACEI",
        "抗血小板药物",
        "他汀类药物",
        "吸入糖皮质激素",
        "支气管舒张剂",
        "质子泵抑制剂",
        "非甾体抗炎药",
        "秋水仙碱",
        "糖皮质激素",
        "利尿剂",
        "ARNI",
        "β受体阻滞剂",
        "抗甲状腺药物",
        "左甲状腺素",
        "补铁",
        "维生素D",
        "抗骨吸收药物",
        "碘131",
    ],
    key=len,
    reverse=True,
)

TREATMENTS = sorted(
    [
        "饮食控制",
        "运动治疗",
        "生活方式干预",
        "戒烟",
        "肺康复",
        "抗感染治疗",
        "退热",
        "补液",
        "休息",
        "对症治疗",
        "避免过敏原",
        "避免饱餐",
        "减少咖啡和酒精摄入",
        "根除幽门螺杆菌方案",
        "控制血压",
        "控制血糖",
        "低盐饮食",
        "肾内科随访",
        "禁食",
        "手术",
        "溶栓",
        "取栓",
        "查找失血原因",
        "营养指导",
        "定期复查甲状腺功能",
        "防跌倒干预",
        "限制钠盐摄入",
        "心理治疗",
        "定期随访",
        "内分泌科随访",
        "抗抑郁药物",
        "复诊",
        "及时干预",
    ],
    key=len,
    reverse=True,
)

RISK_FACTORS = sorted(
    [
        "吸烟",
        "病毒感染",
        "幽门螺杆菌",
        "泌尿系统结构异常",
        "失血原因",
    ],
    key=len,
    reverse=True,
)

DEPARTMENTS = sorted(["内分泌科", "心内科", "呼吸科", "肾内科"], key=len, reverse=True)

CONTRAINDICATION_HINTS = [
    ("二甲双胍", "严重肾功能不全"),
    ("非甾体抗炎药", "消化性溃疡"),
]

QUESTION_RELATION_HINTS = {
    "has_symptom": ["症状", "表现", "体征"],
    "treated_by": ["治疗", "用药", "方案"],
    "diagnosed_by": ["检查", "诊断", "检验"],
    "has_complication": ["并发症", "并发", "风险"],
    "belongs_to_department": ["科室", "就诊"],
    "contraindicated_for": ["禁忌", "慎用", "避免"],
    "has_risk_factor": ["危险因素", "风险因素", "病因", "原因", "相关因素"],
    "prevented_by": ["预防", "管理", "干预"],
    "causes": ["因果", "导致", "造成", "引起", "诱发"],
    "located_in": ["部位", "位置", "发生在哪里"],
    "has_stage": ["分期", "分型", "阶段"],
    "has_patient_group": ["年龄", "人群", "群体"],
    "has_history": ["病史", "既往史"],
    "spreads_via": ["传播", "传播途径"],
    "has_epidemiology": ["流行病学", "发病率", "死亡率", "季节"],
}

QUESTION_RELATION_WEIGHTS = {
    "has_symptom": 1.15,
    "treated_by": 1.15,
    "diagnosed_by": 1.05,
    "has_complication": 1.3,
    "belongs_to_department": 1.0,
    "contraindicated_for": 1.2,
    "has_risk_factor": 1.25,
    "prevented_by": 1.1,
    "causes": 1.35,
    "located_in": 0.95,
    "has_stage": 0.95,
    "has_patient_group": 0.95,
    "has_history": 0.9,
    "spreads_via": 1.05,
    "has_epidemiology": 0.95,
}

QUESTION_OBJECT_TYPE_WEIGHTS = {
    "Symptom": 1.12,
    "Disease": 1.1,
    "RiskFactor": 1.08,
    "Treatment": 1.06,
    "Drug": 1.04,
    "Test": 1.02,
}

RELATION_EVIDENCE_HINTS = {
    "has_symptom": ("症状", "表现", "出现", "伴", "反复", "突发", "持续"),
    "treated_by": ("治疗", "使用", "用药", "方案", "建议", "可选择", "包括"),
    "diagnosed_by": ("检查", "提示", "检测", "评估", "诊断", "用于", "完善", "监测"),
    "has_complication": ("风险", "并发", "增加", "合并", "可导致", "相关"),
    "belongs_to_department": ("科", "随访", "就诊"),
    "contraindicated_for": ("慎用", "避免", "禁忌", "不宜"),
    "has_risk_factor": ("相关", "危险因素", "风险因素", "病因", "原因", "异常"),
    "prevented_by": ("预防", "管理", "干预", "控制", "避免", "戒", "随访", "接种", "康复"),
    "causes": ("导致", "造成", "引起", "诱发", "增加", "相关"),
    "located_in": ("发生于", "位于", "累及", "侵犯", "受累", "部位", "病变"),
    "has_stage": ("分为", "分型", "分期", "类型", "处于", "属于", "分级"),
    "has_patient_group": ("好发于", "多见于", "常见于", "高发于", "影响", "人群"),
    "has_history": ("既往史", "病史", "家族史", "过敏史", "手术史"),
    "spreads_via": ("传播", "传播途径", "传播方式", "通过", "经由", "感染"),
    "has_epidemiology": ("流行病学", "流行特点", "发病特点", "多发", "高发", "发病率", "死亡率"),
}

NEGATION_PREFIXES = ("无", "未见", "未发现", "否认", "不伴", "不考虑", "排除", "无需")
NEGATION_SUFFIXES = ("阴性", "不明显", "未提示")


def load_documents(path: Path) -> list[MedicalDocument]:
    docs: list[MedicalDocument] = []
    for payload in load_records(path):
        docs.append(
            MedicalDocument(
                doc_id=str(payload.get("id") or payload.get("doc_id") or payload.get("document_id") or "unknown"),
                title=str(payload.get("title") or payload.get("name") or path.stem),
                source=str(payload.get("source") or payload.get("dataset") or path.stem),
                text=clean_text(str(payload.get("text") or payload.get("content") or "")),
            )
        )
    return docs


def _contains(text: str, term: str) -> bool:
    return term in text


def _match_many(text: str, phrases: Iterable[str]) -> list[str]:
    matched: list[str] = []
    for phrase in phrases:
        if phrase in text:
            matched.append(phrase)
    return matched


def _select_canonical_disease_from_title(title: str, text: str) -> tuple[str, str]:
    title_core = strip_title_suffix(title)
    if "@" in text:
        prefix = clean_text(text.split("@", 1)[0])
        if _looks_like_medical_entity(prefix, "Disease"):
            return prefix, prefix
    for alias, canonical in DISEASE_PATTERNS:
        if alias in title_core or alias in text:
            return canonical, alias
    text_disease = _first_medical_entity_like_phrase(text, "Disease")
    if text_disease:
        return text_disease, text_disease
    if title_core and not title_core.startswith("CMeIE "):
        return title_core, title_core
    return "未识别疾病", ""


def _strip_open_phrase(value: str) -> str:
    text = clean_text(value)
    text = re.split(r"(?:通常|一般|以及|并且|同时|如果|其中|例如|如|使得|导致|可将之)", text, maxsplit=1)[0]
    text = text.strip(" *#-—:：，。；;、,.()（）[]【】")
    text = re.sub(r"^[0-9一二三四五六七八九十]+[.、．]\s*", "", text)
    text = re.sub(r"（[^）]{1,40}）", "", text)
    text = re.sub(r"\([^)]{1,40}\)", "", text)
    return text.strip(" *#-—:：，。；;、,.()（）[]【】")


def _looks_like_medical_entity(value: str, entity_type: str) -> bool:
    text = _strip_open_phrase(value)
    if not text or len(text) > 32:
        return False
    if len(text) == 1 or any(marker in text for marker in ("@", "查看全部", "体征/症状", "###")):
        return False
    if any(text.startswith(prefix) for prefix in ("伴", "有", "为", "是", "与", "可", "导", "由于", "根据", "可将", "通常")):
        return False
    if any(prefix in text for prefix in ("亦称", "是指", "由于", "根据", "可将", "患儿为", "）")):
        return False
    if any(ch in text for ch in "\n\r\t"):
        return False
    if entity_type == "Disease":
        return any(
            suffix in text
            for suffix in (
                "病",
                "症",
                "炎",
                "癌",
                "瘤",
                "溃疡",
                "综合征",
                "贫血",
                "感染",
                "早熟",
                "疏松",
                "不良",
                "衰竭",
                "畸形",
                "障碍",
            )
        )
    if entity_type == "Symptom":
        return any(
            suffix in text
            for suffix in ("痛", "热", "咳", "嗽", "喘", "吐", "泻", "血", "肿", "晕", "乏力", "困难", "青紫", "苍白", "气短")
        )
    if entity_type == "Drug":
        return any(suffix in text for suffix in ("素", "肽", "霉素", "沙星", "普利", "洛尔", "药", "剂"))
    if entity_type == "Treatment":
        return any(
            suffix in text
            for suffix in ("治疗", "手术", "给氧", "补液", "锻炼", "训练", "置换", "接种", "喂养", "饮食", "管理", "随访", "干预", "控制", "康复")
        )
    if entity_type == "Test":
        return any(suffix in text for suffix in ("检查", "试验", "分析", "CT", "MRI", "X线", "内镜", "超声", "筛查"))
    if entity_type == "Anatomy":
        return any(
            suffix in text
            for suffix in ("部", "道", "管", "腔", "膜", "脏", "肺", "肾", "心", "脑", "胃", "肠", "胆囊", "关节", "皮肤", "骨")
        )
    if entity_type == "ClinicalStage":
        return any(suffix in text for suffix in ("期", "型", "级", "阶段", "急性发作", "缓解期", "早期", "晚期", "轻症", "重症", "中症"))
    if entity_type == "PatientGroup":
        return any(
            suffix in text
            for suffix in ("人群", "患者", "儿童", "成人", "老年人", "孕妇", "女性", "男性", "青少年", "婴幼儿", "新生儿")
        )
    if entity_type == "History":
        return any(suffix in text for suffix in ("史", "病史", "既往史", "家族史", "手术史", "过敏史"))
    if entity_type == "TransmissionRoute":
        return any(suffix in text for suffix in ("传播", "飞沫", "接触", "血液", "母婴", "消化道", "呼吸道", "途径"))
    if entity_type == "Epidemiology":
        return any(suffix in text for suffix in ("多发", "流行", "发病率", "死亡率", "季节", "高发", "散发", "暴发"))
    return True


def _first_medical_entity_like_phrase(text: str, entity_type: str) -> str:
    candidates: list[str] = []
    if "@" in text:
        prefix = _strip_open_phrase(text.split("@", 1)[0])
        if _looks_like_medical_entity(prefix, entity_type):
            return prefix
    for pattern in (
        r"([\u4e00-\u9fffA-Za-z0-9\-]{2,18}(?:病|症|炎|癌|瘤|溃疡|综合征|贫血|感染|早熟|衰竭|障碍|疏松))(?:患儿|患者|新生儿|儿童|成人)",
        r"(?:^|[。；;，,\s])([\u4e00-\u9fffA-Za-z0-9\-]{2,18}(?:病|症|炎|癌|瘤|溃疡|综合征|贫血|感染|早熟|衰竭|障碍|疏松))",
    ):
        candidates.extend(re.findall(pattern, text[:160]))
    for candidate in candidates:
        phrase = _strip_open_phrase(candidate)
        if _looks_like_medical_entity(phrase, entity_type):
            return phrase
    return ""


def _split_open_items(raw: str, object_type: str) -> list[str]:
    text = raw
    for separator in ("以及", "及", "和", "或", "、", "，", ",", "；", ";"):
        text = text.replace(separator, "、")
    items = []
    for part in text.split("、"):
        phrase = _strip_open_phrase(part)
        max_len = 18 if object_type in {"Symptom", "Test"} else 32 if object_type in {"Epidemiology", "TransmissionRoute", "PatientGroup"} else 24
        if len(phrase) <= max_len and _looks_like_medical_entity(phrase, object_type):
            items.append(phrase)
    return _sorted_unique(items)


def _split_guideline_items(raw: str, object_type: str) -> list[str]:
    text = clean_text(raw)
    text = re.split(r"(?:等(?:表现|症状|检查|措施|治疗|管理)?|均可|时应|以便|从而|因此|必要时)", text, maxsplit=1)[0]
    text = re.sub(r"^(?:症状|临床表现|表现|评估项目|监测指标|辅助检查|诊断依据|检查|治疗方案|管理方案|预防措施|危险因素|风险因素)(?:包括|包含|为|有)?", "", text)
    text = re.sub(r"^(?:可|应|需|建议|推荐|主要|常见|包括|包含|进行|给予|采取|完善|监测|控制|避免|加强|鼓励|接种|使用)", "", text)
    items = _split_open_items(text, object_type)
    if items:
        return items
    phrase = _strip_open_phrase(text)
    return [phrase] if _looks_like_medical_entity(phrase, object_type) else []


GUIDELINE_ENUM_RELATION_PATTERNS: tuple[tuple[str, str, str], ...] = (
    (
        "has_symptom",
        "Symptom",
        r"(?:临床表现(?:包括|有|为)?|症状(?:包括|有|为)?|常见症状(?:包括|有|为)?|主要症状(?:包括|有|为)?|主要表现(?:包括|有|为)?|典型表现(?:包括|有|为)?|可出现|表现为)([^。；;]{2,90})",
    ),
    (
        "diagnosed_by",
        "Test",
        r"(?:建议|应|需|可)?(?:完善|进行|监测|复查|评估)([^。；;]{2,80}(?:检查|筛查|评估|监测|CT|MRI|X线|超声|培养|血糖|血常规|尿常规|功能))",
    ),
    (
        "diagnosed_by",
        "Test",
        r"(?:辅助检查|诊断依据|评估项目|监测指标)(?:包括|为|有)?([^。；;]{2,90})",
    ),
    (
        "treated_by",
        "Treatment",
        r"(?:治疗上|治疗时|治疗方案|处理原则|管理方案)(?:建议|推荐|包括|为|可采用|应)?([^。；;]{2,100})",
    ),
    (
        "prevented_by",
        "Treatment",
        r"(?:预防|长期管理|健康管理|随访管理|二级预防)(?:建议|包括|需|应|强调)?([^。；;]{2,100})",
    ),
    (
        "has_risk_factor",
        "RiskFactor",
        r"(?:危险因素|风险因素|诱因|相关因素)(?:包括|有|为|常见为)?([^。；;]{2,90})",
    ),
)


def _guideline_relation_candidates(sentence: str, subject: str) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for predicate, object_type, pattern in GUIDELINE_ENUM_RELATION_PATTERNS:
        for match in re.finditer(pattern, sentence):
            for item in _split_guideline_items(match.group(1), object_type):
                relation_object_type = object_type
                if predicate == "treated_by" and _looks_like_medical_entity(item, "Drug"):
                    relation_object_type = "Drug"
                candidates.append(
                    {
                        "subject": subject,
                        "predicate": predicate,
                        "object": item,
                        "object_type": relation_object_type,
                    }
                )
    return candidates


def _infer_open_subject_type(subject: str, predicate: str) -> str:
    if _looks_like_medical_entity(subject, "Disease"):
        return "Disease"
    if predicate == "causes":
        if _looks_like_medical_entity(subject, "Treatment"):
            return "Treatment"
        return "RiskFactor"
    return "Disease"


def _open_relation_base_confidence(predicate: str) -> float:
    return {
        "has_symptom": 0.78,
        "treated_by": 0.78,
        "diagnosed_by": 0.78,
        "has_complication": 0.76,
        "has_risk_factor": 0.76,
        "causes": 0.76,
        "located_in": 0.74,
        "has_stage": 0.74,
        "has_patient_group": 0.73,
        "has_history": 0.73,
        "spreads_via": 0.73,
        "has_epidemiology": 0.72,
    }.get(predicate, 0.74)


def _accumulate_entity(
    entity_index: OrderedDict[tuple[str, str], EntityRecord],
    name: str,
    entity_type: str,
    doc_id: str,
    alias: str,
    confidence: float,
) -> None:
    key = (name, entity_type)
    record = entity_index.get(key)
    if record is None:
        record = EntityRecord(name=name, type=entity_type, confidence=confidence)
        entity_index[key] = record
    record.add_source(doc_id)
    record.add_alias(alias)
    record.confidence = max(record.confidence, confidence)


def _scan_category(
    entity_index: OrderedDict[tuple[str, str], EntityRecord],
    text: str,
    doc_id: str,
    phrases: Iterable[str],
    entity_type: str,
    confidence: float,
) -> None:
    for phrase in phrases:
        if phrase in text:
            _accumulate_entity(entity_index, phrase, entity_type, doc_id, phrase, confidence)


def _open_relation_candidates(text: str, main_disease: str) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []

    def add(predicate: str, object_name: str, object_type: str, subject: str | None = None) -> None:
        phrase = _strip_open_phrase(object_name)
        if not phrase or phrase == (subject or main_disease):
            return
        if not _looks_like_medical_entity(phrase, object_type):
            return
        candidates.append(
            {
                "subject": subject or main_disease,
                "predicate": predicate,
                "object": phrase,
                "object_type": object_type,
            }
        )

    for sentence in split_sentences(text):
        sentence = clean_text(sentence)
        if not sentence:
            continue
        subject = main_disease
        if "@" in sentence:
            left, right = sentence.split("@", 1)
            left = _strip_open_phrase(left)
            if _looks_like_medical_entity(left, "Disease"):
                subject = left
                sentence = right

        for candidate in _guideline_relation_candidates(sentence, subject):
            add(candidate["predicate"], candidate["object"], candidate["object_type"], candidate["subject"])

        for match in re.finditer(r"(?:可分为|分为|分型为|分期为|类型为)([^。；;]{2,90})", sentence):
            for item in _split_open_items(match.group(1), "ClinicalStage"):
                add("has_stage", item, "ClinicalStage", subject)
        for match in re.finditer(r"(?:处于|属于|诊断为|分级为)([^。；;，,]{2,40}(?:期|型|级|阶段|急性发作|缓解期|早期|晚期))", sentence):
            for item in _split_open_items(match.group(1), "ClinicalStage"):
                add("has_stage", item, "ClinicalStage", subject)
        for match in re.finditer(r"(?:病因|发病机制|危险因素|风险因素)[^。；;，,]{0,12}(?:为|是|包括|包含|与)?([^。；;]{2,80})", sentence):
            for item in _split_open_items(match.group(1), "RiskFactor"):
                add("has_risk_factor", item, "RiskFactor", subject)
        for match in re.finditer(r"(?:临床症状|临床表现|主要症状|常见症状)[^。；;，,]{0,8}(?:为|包括|有|出现)?([^。；;]{2,60})", sentence):
            for item in _split_open_items(match.group(1), "Symptom"):
                add("has_symptom", item, "Symptom", subject)
        for match in re.finditer(r"(?:治疗方法|治疗方案|药物治疗|手术治疗|治疗药物)[^。；;，,]{0,12}(?:为|包括|使用|选用|可用|开始)?([^。；;]{2,70})", sentence):
            for item in _split_open_items(match.group(1), "Treatment"):
                object_type = "Drug" if _looks_like_medical_entity(item, "Drug") else "Treatment"
                add("treated_by", item, object_type, subject)
        for match in re.finditer(r"(?:实验室检查|影像学检查|辅助检查|诊断检查|筛查项目)[^。；;，,]{0,12}(?:为|包括|使用|通过)?([^。；;]{2,60})", sentence):
            for item in _split_open_items(match.group(1), "Test"):
                add("diagnosed_by", item, "Test", subject)
        for match in re.finditer(r"(?:并发|合并|导致|造成|引起|诱发)([^。；;，,]{2,50})", sentence):
            for item in _split_open_items(match.group(1), "Disease"):
                add("has_complication", item, "Disease", subject)
        for match in re.finditer(r"(?:发生于|位于|累及|侵犯|受累|病变部位(?:为|包括)?)([^。；;]{2,60})", sentence):
            for item in _split_open_items(match.group(1), "Anatomy"):
                add("located_in", item, "Anatomy", subject)
        for match in re.finditer(r"(?:好发于|多见于|常见于|高发于|主要影响)([^。；;]{2,70})", sentence):
            for item in _split_open_items(match.group(1), "PatientGroup"):
                add("has_patient_group", item, "PatientGroup", subject)
        for match in re.finditer(r"(?:既往史|病史|家族史|过敏史|手术史)[^为包有。；;，,]{0,10}(?:为|包括|有)([^。；;，,]{2,70})", sentence):
            for item in _split_open_items(match.group(1), "History"):
                add("has_history", item, "History", subject)
        for match in re.finditer(r"(?:传播途径|传播方式)[^为包是。；;，,]{0,10}(?:为|包括|是)([^。；;，,]{2,70})", sentence):
            for item in _split_open_items(match.group(1), "TransmissionRoute"):
                add("spreads_via", item, "TransmissionRoute", subject)
        for match in re.finditer(r"(?:通过|经由)([^。；;，,]{2,40})(?:传播|扩散|感染)", sentence):
            for item in _split_open_items(match.group(1), "TransmissionRoute"):
                add("spreads_via", item, "TransmissionRoute", subject)
        for match in re.finditer(r"(?:流行病学|流行特点|发病特点)[^为包表。；;，,]{0,10}(?:为|包括|表现为)([^。；;，,]{2,90})", sentence):
            for item in _split_open_items(match.group(1), "Epidemiology"):
                add("has_epidemiology", item, "Epidemiology", subject)
        for match in re.finditer(r"([^。；;，,]{2,36}(?:多发|高发|流行|发病率升高|死亡率升高))", sentence):
            if any(prefix in match.group(1) for prefix in ("流行病学", "流行特点", "发病特点")):
                continue
            for item in _split_open_items(match.group(1), "Epidemiology"):
                add("has_epidemiology", item, "Epidemiology", subject)
        for match in re.finditer(r"([\u4e00-\u9fffA-Za-z0-9\-]{2,32})(?:导致|造成|引起|诱发)([\u4e00-\u9fffA-Za-z0-9\-]{2,32})", sentence):
            cause = _strip_open_phrase(match.group(1))
            effect = _strip_open_phrase(match.group(2))
            if _looks_like_medical_entity(effect, "Disease") or _looks_like_medical_entity(effect, "Symptom"):
                candidates.append(
                    {
                        "subject": cause,
                        "predicate": "causes",
                        "object": effect,
                        "object_type": "Disease" if _looks_like_medical_entity(effect, "Disease") else "Symptom",
                    }
                )
    seen: set[tuple[str, str, str, str]] = set()
    deduped: list[dict[str, str]] = []
    for item in candidates:
        key = (item["subject"], item["predicate"], item["object"], item["object_type"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def extract_entities(document: MedicalDocument) -> tuple[list[dict[str, Any]], str, str]:
    """Return entity payloads for a single document and the canonical disease."""
    entity_index: OrderedDict[tuple[str, str], EntityRecord] = OrderedDict()
    disease_name, disease_alias = _select_canonical_disease_from_title(document.title, document.text)
    if disease_name != "未识别疾病":
        _accumulate_entity(entity_index, disease_name, "Disease", document.doc_id, disease_alias or disease_name, 0.96)

    _scan_category(entity_index, document.text, document.doc_id, SYMPTOMS, "Symptom", 0.88)
    _scan_category(entity_index, document.text, document.doc_id, TESTS, "Test", 0.90)
    _scan_category(entity_index, document.text, document.doc_id, DRUGS, "Drug", 0.92)
    _scan_category(entity_index, document.text, document.doc_id, TREATMENTS, "Treatment", 0.84)
    _scan_category(entity_index, document.text, document.doc_id, RISK_FACTORS, "RiskFactor", 0.84)
    _scan_category(entity_index, document.text, document.doc_id, DEPARTMENTS, "Department", 0.95)
    for candidate in _open_relation_candidates(document.text, disease_name):
        subject = candidate["subject"]
        subject_type = _infer_open_subject_type(subject, candidate["predicate"])
        if _looks_like_medical_entity(subject, subject_type):
            _accumulate_entity(entity_index, subject, subject_type, document.doc_id, subject, 0.86)
        _accumulate_entity(
            entity_index,
            candidate["object"],
            candidate["object_type"],
            document.doc_id,
            candidate["object"],
            0.82,
        )

    # Pull in any other disease mentions for downstream complication reasoning.
    for alias, canonical in DISEASE_PATTERNS:
        if canonical == disease_name and alias in (disease_alias, canonical):
            continue
        if alias in document.text:
            _accumulate_entity(entity_index, canonical, "Disease", document.doc_id, alias, 0.91)

    entities = []
    for idx, record in enumerate(entity_index.values(), start=1):
        entities.append(
            {
                "entity_id": f"E{idx:04d}",
                "name": record.name,
                "type": record.type,
                "aliases": sorted(record.aliases),
                "source_doc_ids": sorted(record.source_doc_ids),
                "confidence": round(record.confidence, 2),
            }
        )
    return entities, disease_name, disease_alias


def _sentence_for_evidence(text: str, subject: str, object_name: str) -> str:
    for sentence in split_sentences(text):
        if subject in sentence and object_name in sentence:
            return sentence
    return text[:240]


def _has_negated_mention(sentence: str, term: str) -> bool:
    if not sentence or not term or term not in sentence:
        return False
    for match_start in [index for index in range(len(sentence)) if sentence.startswith(term, index)]:
        prefix_window = sentence[max(0, match_start - 8):match_start]
        suffix_window = sentence[match_start + len(term):match_start + len(term) + 6]
        if any(prefix in prefix_window for prefix in NEGATION_PREFIXES):
            return True
        if any(suffix in suffix_window for suffix in NEGATION_SUFFIXES):
            return True
    return False


def _relation_evidence_sentences(
    *,
    sentences: list[str],
    title: str,
    subject: str,
    object_name: str,
    predicate: str,
) -> list[dict[str, Any]]:
    subject_context = subject in title
    hints = RELATION_EVIDENCE_HINTS.get(predicate, ())
    candidates: list[dict[str, Any]] = []
    for index, sentence in enumerate(sentences):
        if object_name not in sentence:
            continue
        negated = _has_negated_mention(sentence, object_name)
        subject_hit = subject in sentence or subject_context
        hint_count = sum(1 for hint in hints if hint in sentence)
        score = 0.34
        if subject_hit:
            score += 0.26
        if hint_count:
            score += min(0.24, 0.08 * hint_count)
        if len(sentence) <= 120:
            score += 0.06
        if negated:
            score -= 0.45
        candidates.append(
            {
                "sentence_index": index,
                "sentence": sentence,
                "subject_hit": subject_hit,
                "relation_hint_count": hint_count,
                "negated": negated,
                "score": round(max(0.0, min(score, 1.0)), 4),
            }
        )
    candidates.sort(
        key=lambda item: (
            item["negated"],
            -float(item["score"]),
            -int(item["subject_hit"]),
            -int(item["relation_hint_count"]),
            int(item["sentence_index"]),
        )
    )
    return candidates


def _select_relation_evidence(
    *,
    text: str,
    sentences: list[str],
    title: str,
    subject: str,
    object_name: str,
    predicate: str,
) -> tuple[str, dict[str, Any] | None]:
    candidates = _relation_evidence_sentences(
        sentences=sentences,
        title=title,
        subject=subject,
        object_name=object_name,
        predicate=predicate,
    )
    for candidate in candidates:
        if not candidate["negated"]:
            return str(candidate["sentence"]), candidate
    return text[:240], None


def extract_triples(
    document: MedicalDocument,
    entities: list[dict[str, Any]],
    main_disease: str,
) -> list[dict[str, Any]]:
    entity_by_name_type = {(item["name"], item["type"]): item for item in entities}
    triples: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    text = document.text
    sentences = split_sentences(text)
    evidence_cache: dict[tuple[str, str, str], tuple[str, dict[str, Any] | None]] = {}
    symptom_matches = _match_many(text, SYMPTOMS)
    test_matches = _match_many(text, TESTS)
    drug_matches = _match_many(text, DRUGS)
    treatment_matches = _match_many(text, TREATMENTS)
    department_matches = _match_many(text, DEPARTMENTS)
    risk_factor_matches = _match_many(text, RISK_FACTORS)
    prevention_matches = _match_many(
        text,
        [
            "戒烟",
            "低盐饮食",
            "控制血压",
            "控制血糖",
            "饮食控制",
            "运动治疗",
            "防跌倒干预",
            "避免过敏原",
            "避免饱餐",
            "定期随访",
            "复诊",
            "及时干预",
            "生活方式干预",
            "限制钠盐摄入",
        ],
    )
    contraindicated_drugs = {
        subject
        for subject, target in CONTRAINDICATION_HINTS
        if subject in text and (target in text or any(keyword in text for keyword in ("慎用", "避免", "禁忌", "不宜")))
    }
    open_candidates = _open_relation_candidates(text, main_disease)

    def add_triple(
        subject: str,
        subject_type: str,
        predicate: str,
        object_name: str,
        object_type: str,
        confidence: float,
    ) -> None:
        key = (subject, predicate, object_name, document.doc_id)
        if key in seen:
            return
        evidence_key = (subject, predicate, object_name)
        evidence_payload = evidence_cache.get(evidence_key)
        if evidence_payload is None:
            evidence_payload = _select_relation_evidence(
                text=text,
                sentences=sentences,
                title=document.title,
                subject=subject,
                object_name=object_name,
                predicate=predicate,
            )
            evidence_cache[evidence_key] = evidence_payload
        evidence, evidence_meta = evidence_payload
        if evidence_meta is None:
            if predicate in {"has_symptom", "has_complication", "has_risk_factor"}:
                return
            evidence_meta = {
                "sentence_index": None,
                "subject_hit": subject in evidence or subject in document.title,
                "relation_hint_count": 0,
                "negated": False,
                "score": 0.28,
            }
        if evidence_meta.get("negated"):
            return
        calibrated_confidence = confidence
        calibrated_confidence += 0.04 * float(evidence_meta.get("score", 0))
        if evidence_meta.get("subject_hit"):
            calibrated_confidence += 0.02
        if int(evidence_meta.get("relation_hint_count", 0) or 0) > 0:
            calibrated_confidence += 0.02
        seen.add(key)
        triples.append(
            {
                "subject": subject,
                "subject_type": subject_type,
                "predicate": predicate,
                "object": object_name,
                "object_type": object_type,
                "source_doc_id": document.doc_id,
                "evidence": evidence,
                "confidence": round(min(calibrated_confidence, 0.97), 2),
                "evidence_quality": {
                    "sentence_index": evidence_meta.get("sentence_index"),
                    "score": evidence_meta.get("score"),
                    "subject_hit": evidence_meta.get("subject_hit"),
                    "relation_hint_count": evidence_meta.get("relation_hint_count"),
                    "negation_checked": True,
                },
            }
        )

    if main_disease != "未识别疾病" and (main_disease, "Disease") in entity_by_name_type:
        for symptom in symptom_matches:
            add_triple(main_disease, "Disease", "has_symptom", symptom, "Symptom", 0.9)

        for test in test_matches:
            add_triple(main_disease, "Disease", "diagnosed_by", test, "Test", 0.88)

        for item in drug_matches:
            if item in contraindicated_drugs:
                continue
            add_triple(main_disease, "Disease", "treated_by", item, "Drug", 0.91)

        for item in treatment_matches:
            add_triple(main_disease, "Disease", "treated_by", item, "Treatment", 0.84)

        for department in department_matches:
            add_triple(main_disease, "Disease", "belongs_to_department", department, "Department", 0.95)

        if risk_factor_matches or "查找失血原因" in text:
            for item in risk_factor_matches + (["查找失血原因"] if "查找失血原因" in text else []):
                risk_name = "失血原因" if item == "查找失血原因" else item
                add_triple(main_disease, "Disease", "has_risk_factor", risk_name, "RiskFactor", 0.78)

        if prevention_matches:
            for item in prevention_matches:
                add_triple(main_disease, "Disease", "prevented_by", item, "Treatment", 0.8)

        for candidate in open_candidates:
            subject = candidate["subject"]
            subject_type = _infer_open_subject_type(subject, candidate["predicate"])
            add_triple(
                subject,
                subject_type,
                candidate["predicate"],
                candidate["object"],
                candidate["object_type"],
                _open_relation_base_confidence(candidate["predicate"]),
            )

    disease_mentions = [
        item["name"]
        for item in entities
        if item["type"] == "Disease" and item["name"] != main_disease
    ]

    if any(keyword in text for keyword in ("风险", "并发", "增加", "可增加")):
        for disease in disease_mentions:
            add_triple(main_disease, "Disease", "has_complication", disease, "Disease", 0.75)

    if any(keyword in text for keyword in ("慎用", "避免", "禁忌", "不宜")):
        for subject, target in CONTRAINDICATION_HINTS:
            if subject not in text:
                continue
            if target in text:
                add_triple(subject, "Drug", "contraindicated_for", target, "Disease", 0.82)
            elif main_disease != "未识别疾病":
                add_triple(subject, "Drug", "contraindicated_for", main_disease, "Disease", 0.78)

    return [
        {
            "triple_id": f"T{idx:04d}",
            **item,
        }
        for idx, item in enumerate(triples, start=1)
    ]


def _parse_llm_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("LLM extraction response must be a JSON object")
    return payload


def _canonicalize_llm_entities(
    document: MedicalDocument,
    payload: dict[str, Any],
    fallback_entities: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, str]:
    allowed_types = {
        "Disease",
        "Symptom",
        "Drug",
        "Treatment",
        "Test",
        "Department",
        "RiskFactor",
        "Anatomy",
        "ClinicalStage",
        "PatientGroup",
        "History",
        "TransmissionRoute",
        "Epidemiology",
        "Complication",
    }
    entity_index: OrderedDict[tuple[str, str], EntityRecord] = OrderedDict()
    main_disease = str(payload.get("main_disease") or "").strip()
    disease_alias = str(payload.get("disease_alias") or main_disease).strip()

    for item in payload.get("entities", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        entity_type = str(item.get("type") or "").strip()
        if not name or entity_type not in allowed_types:
            continue
        aliases = item.get("aliases")
        alias_values = aliases if isinstance(aliases, list) else [name]
        try:
            confidence = float(item.get("confidence", 0.82))
        except (TypeError, ValueError):
            confidence = 0.82
        _accumulate_entity(
            entity_index,
            name,
            entity_type,
            document.doc_id,
            name,
            min(max(confidence, 0.0), 1.0),
        )
        for alias in alias_values:
            _accumulate_entity(
                entity_index,
                name,
                entity_type,
                document.doc_id,
                str(alias).strip(),
                min(max(confidence, 0.0), 1.0),
            )

    if not entity_index:
        return fallback_entities, main_disease or "未识别疾病", disease_alias

    if not main_disease:
        for record in entity_index.values():
            if record.type == "Disease":
                main_disease = record.name
                disease_alias = next(iter(record.aliases), record.name)
                break
    if not main_disease:
        main_disease = "未识别疾病"

    entities: list[dict[str, Any]] = []
    for idx, record in enumerate(entity_index.values(), start=1):
        entities.append(
            {
                "entity_id": f"E{idx:04d}",
                "name": record.name,
                "type": record.type,
                "aliases": sorted(record.aliases),
                "source_doc_ids": sorted(record.source_doc_ids),
                "confidence": round(record.confidence, 2),
            }
        )
    return entities, main_disease, disease_alias


def _canonicalize_llm_triples(
    document: MedicalDocument,
    payload: dict[str, Any],
    entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    entity_types = {item["name"]: item["type"] for item in entities}
    allowed_predicates = {
        "has_symptom",
        "treated_by",
        "diagnosed_by",
        "has_complication",
        "has_risk_factor",
        "contraindicated_for",
        "prevented_by",
        "belongs_to_department",
        "causes",
        "located_in",
        "has_stage",
        "has_patient_group",
        "has_history",
        "spreads_via",
        "has_epidemiology",
    }
    triples: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in payload.get("triples", []):
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject") or "").strip()
        predicate = str(item.get("predicate") or "").strip()
        object_name = str(item.get("object") or "").strip()
        if not subject or not object_name or predicate not in allowed_predicates:
            continue
        subject_type = str(item.get("subject_type") or entity_types.get(subject) or "").strip()
        object_type = str(item.get("object_type") or entity_types.get(object_name) or "").strip()
        if not subject_type or not object_type:
            continue
        key = (subject, predicate, object_name, document.doc_id)
        if key in seen:
            continue
        seen.add(key)
        evidence = str(item.get("evidence") or "").strip()
        if not evidence:
            evidence = _sentence_for_evidence(document.text, subject, object_name)
        try:
            confidence = float(item.get("confidence", 0.78))
        except (TypeError, ValueError):
            confidence = 0.78
        triples.append(
            {
                "triple_id": f"T{len(triples) + 1:04d}",
                "subject": subject,
                "subject_type": subject_type,
                "predicate": predicate,
                "object": object_name,
                "object_type": object_type,
                "source_doc_id": document.doc_id,
                "evidence": evidence,
                "confidence": round(min(max(confidence, 0.0), 1.0), 2),
            }
        )
    return triples


class RuleKGExtractionBackend:
    name = "rule_medical_kg_backend"
    version = "2026-07-01"

    def run(self, document: MedicalDocument) -> ExtractionBackendResult:
        entities, main_disease, disease_alias = extract_entities(document)
        triples = extract_triples(document, entities, main_disease)
        return ExtractionBackendResult(
            backend=self.name,
            version=self.version,
            entities=entities,
            triples=triples,
            main_disease=main_disease,
            disease_alias=disease_alias,
            metadata={
                "entity_count": len(entities),
                "triple_candidate_count": len(triples),
                "supports_llm_replacement": True,
                "fallback": "deterministic_dictionary_and_pattern_rules",
            },
        )


class LLMKGExtractionBackend:
    name = "llm_medical_kg_backend"
    version = "2026-07-01"

    def run(self, document: MedicalDocument) -> ExtractionBackendResult:
        config = get_openai_compat_config()
        if config is None:
            raise RuntimeError("llm_not_configured")

        rule_result = RuleKGExtractionBackend().run(document)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是医疗知识图谱抽取后端。只输出 JSON 对象，不要输出解释。"
                    "实体类型只能使用 Disease、Symptom、Drug、Treatment、Test、Department、RiskFactor、"
                    "Anatomy、ClinicalStage、PatientGroup、History、TransmissionRoute、Epidemiology、Complication。"
                    "关系只能使用 has_symptom、treated_by、diagnosed_by、has_complication、has_risk_factor、"
                    "contraindicated_for、prevented_by、belongs_to_department、causes、located_in、has_stage、"
                    "has_patient_group、has_history、spreads_via、has_epidemiology。"
                    "每个三元组必须包含 evidence，且 evidence 必须来自原文。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "从以下医疗文本抽取知识图谱。返回字段：main_disease、disease_alias、entities、triples。"
                    "entities 中包含 name、type、aliases、confidence；triples 中包含 subject、subject_type、"
                    "predicate、object、object_type、evidence、confidence。\n"
                    f"doc_id: {document.doc_id}\n"
                    f"title: {document.title}\n"
                    f"text: {document.text}"
                ),
            },
        ]
        raw_text = chat_completion(
            messages,
            model=config.model,
            temperature=0.0,
            max_tokens=min(config.max_tokens, 1600),
            timeout_seconds=config.timeout_seconds,
        )
        payload = _parse_llm_json_object(raw_text)
        entities, main_disease, disease_alias = _canonicalize_llm_entities(
            document,
            payload,
            rule_result.entities,
        )
        triples = _canonicalize_llm_triples(document, payload, entities)
        if not triples:
            triples = rule_result.triples
        return ExtractionBackendResult(
            backend=self.name,
            version=self.version,
            entities=entities,
            triples=triples,
            main_disease=main_disease,
            disease_alias=disease_alias,
            metadata={
                "entity_count": len(entities),
                "triple_candidate_count": len(triples),
                "llm_model": config.model,
                "llm_base_url": config.base_url,
                "rule_fallback_entity_count": len(rule_result.entities),
                "rule_fallback_triple_count": len(rule_result.triples),
                "fallback": None,
            },
        )


def _tensor_float_scores(scores: Any) -> list[float]:
    values = scores.detach().to("cpu").flatten().tolist()
    return [float(item) for item in values]


def _tensor_score_rows(scores: Any) -> list[list[float]]:
    rows = scores.detach().to("cpu").tolist()
    return [[float(value) for value in row] for row in rows]


def _normalize_tensor_score(score: float) -> float:
    return max(0.0, min((score + 1.0) / 2.0, 1.0))


def _build_relation_catalog() -> dict[str, tuple[str, ...]]:
    return {name: tuple(hints) for name, hints in RELATION_EVIDENCE_HINTS.items()}


def _tensor_kg_config() -> tuple[str, int]:
    requested_device = os.getenv("MEDICAL_KG_TENSOR_DEVICE", "auto").strip().lower() or "auto"
    vector_dim = int(os.getenv("MEDICAL_KG_TENSOR_VECTOR_DIM", "512"))
    return requested_device, vector_dim


def _build_tensor_descriptor(
    *,
    requested_device: str,
    selected_device: str,
    vector_dim: int,
    entity_candidate_count: int,
    relation_candidate_count: int,
    batch_size: int,
) -> dict[str, Any]:
    from .npu_tensor_kg_ops import build_operator_descriptor

    descriptor = build_operator_descriptor()
    descriptor.update(
        {
            "requested_device": requested_device,
            "selected_device": selected_device,
            "vector_dim": vector_dim,
            "entity_candidate_count": entity_candidate_count,
            "relation_candidate_count": relation_candidate_count,
            "batch_size": batch_size,
            "batched": batch_size > 1,
        }
    )
    return descriptor


def _apply_tensor_scores_to_rule_result(
    rule_result: ExtractionBackendResult,
    *,
    entity_score_by_name_type: dict[tuple[str, str], float],
    relation_score_by_name: dict[str, float],
    descriptor: dict[str, Any],
) -> ExtractionBackendResult:
    ranked_entities = sorted(
        (
            {
                **entity,
                "tensor_score": round(
                    entity_score_by_name_type.get((str(entity["name"]), str(entity["type"])), 0.0),
                    4,
                ),
            }
            for entity in rule_result.entities
        ),
        key=lambda item: (-float(item["tensor_score"]), str(item["type"]), str(item["name"])),
    )
    entity_rank_by_key = {
        (str(item["name"]), str(item["type"])): index
        for index, item in enumerate(ranked_entities, start=1)
    }
    entities: list[dict[str, Any]] = []
    for entity in rule_result.entities:
        key = (str(entity["name"]), str(entity["type"]))
        tensor_score = entity_score_by_name_type.get(key, 0.0)
        entity_payload = dict(entity)
        entity_payload["tensor_rank"] = entity_rank_by_key.get(key)
        entity_payload["tensor_score"] = round(tensor_score, 4)
        entity_payload["confidence"] = round(
            min(
                0.99,
                max(
                    float(entity.get("confidence", 0.0)),
                    0.92 * float(entity.get("confidence", 0.0)) + 0.08 * tensor_score,
                ),
            ),
            2,
        )
        entities.append(entity_payload)

    triples: list[dict[str, Any]] = []
    for triple in rule_result.triples:
        predicate = str(triple.get("predicate", ""))
        subject_key = (str(triple.get("subject", "")), str(triple.get("subject_type", "")))
        object_key = (str(triple.get("object", "")), str(triple.get("object_type", "")))
        relation_score = relation_score_by_name.get(predicate, 0.0)
        endpoint_scores = [
            entity_score_by_name_type[item]
            for item in (subject_key, object_key)
            if item in entity_score_by_name_type
        ]
        endpoint_score = sum(endpoint_scores) / len(endpoint_scores) if endpoint_scores else 0.0
        tensor_score = round(0.65 * relation_score + 0.35 * endpoint_score, 4)
        triple_payload = dict(triple)
        triple_payload["tensor_relation_score"] = round(relation_score, 4)
        triple_payload["tensor_endpoint_score"] = round(endpoint_score, 4)
        triple_payload["tensor_score"] = tensor_score
        triple_payload["confidence"] = round(
            min(
                0.99,
                max(
                    float(triple.get("confidence", 0.0)),
                    0.9 * float(triple.get("confidence", 0.0)) + 0.1 * tensor_score,
                ),
            ),
            2,
        )
        triples.append(triple_payload)

    return ExtractionBackendResult(
        backend=TensorKGExtractionBackend.name,
        version=TensorKGExtractionBackend.version,
        entities=entities,
        triples=triples,
        main_disease=rule_result.main_disease,
        disease_alias=rule_result.disease_alias,
        metadata={
            "entity_count": len(entities),
            "triple_candidate_count": len(triples),
            "supports_llm_replacement": True,
            "rule_fallback_entity_count": len(rule_result.entities),
            "rule_fallback_triple_count": len(rule_result.triples),
            "tensor_operator": descriptor,
            "tensor_ranker_enabled": True,
            "tensor_device": descriptor["selected_device"],
            "fallback": None,
        },
    )


class TensorKGExtractionBackend:
    name = "tensor_medical_kg_backend"
    version = "2026-07-05"

    def run(self, document: MedicalDocument) -> ExtractionBackendResult:
        from .npu_tensor_kg_ops import (
            hashed_vector,
            rank_entity_candidates,
            score_relation_intents,
            select_tensor_device,
        )

        requested_device, vector_dim = _tensor_kg_config()
        selected_device = select_tensor_device(requested_device)

        rule_result = RuleKGExtractionBackend().run(document)
        document_vector = hashed_vector(f"{document.title}\n{document.text}", dim=vector_dim)
        entity_candidates = [(str(item["name"]), str(item["type"])) for item in rule_result.entities]
        entity_score_by_name_type: dict[tuple[str, str], float] = {}
        if entity_candidates:
            raw_entity_scores = rank_entity_candidates(
                [document_vector],
                entity_candidates,
                vector_dim=vector_dim,
                device=selected_device,
            )
            entity_scores = _tensor_float_scores(raw_entity_scores)
            entity_score_by_name_type = {
                candidate: _normalize_tensor_score(score)
                for candidate, score in zip(entity_candidates, entity_scores)
            }

        relation_catalog = _build_relation_catalog()
        relation_names = list(relation_catalog)
        raw_relation_scores = score_relation_intents(
            [document_vector],
            relation_catalog,
            vector_dim=vector_dim,
            device=selected_device,
        )
        relation_scores = _tensor_float_scores(raw_relation_scores)
        relation_score_by_name = {
            name: _normalize_tensor_score(score)
            for name, score in zip(relation_names, relation_scores)
        }

        descriptor = _build_tensor_descriptor(
            requested_device=requested_device,
            selected_device=selected_device,
            vector_dim=vector_dim,
            entity_candidate_count=len(entity_candidates),
            relation_candidate_count=len(relation_catalog),
            batch_size=1,
        )
        return _apply_tensor_scores_to_rule_result(
            rule_result,
            entity_score_by_name_type=entity_score_by_name_type,
            relation_score_by_name=relation_score_by_name,
            descriptor=descriptor,
        )


def run_tensor_kg_backend_batch(documents: list[MedicalDocument]) -> list[ExtractionBackendResult]:
    from .npu_tensor_kg_ops import hashed_vector, rank_entity_candidates, score_relation_intents, select_tensor_device

    if not documents:
        return []
    requested_device, vector_dim = _tensor_kg_config()
    selected_device = select_tensor_device(requested_device)
    rule_results = [RuleKGExtractionBackend().run(document) for document in documents]
    document_vectors = [hashed_vector(f"{document.title}\n{document.text}", dim=vector_dim) for document in documents]
    entity_candidates = list(
        OrderedDict.fromkeys(
            (str(entity["name"]), str(entity["type"]))
            for result in rule_results
            for entity in result.entities
        )
    )
    if entity_candidates:
        raw_entity_scores = rank_entity_candidates(
            document_vectors,
            entity_candidates,
            vector_dim=vector_dim,
            device=selected_device,
        )
        entity_score_rows = _tensor_score_rows(raw_entity_scores)
    else:
        entity_score_rows = [[] for _ in documents]
    relation_catalog = _build_relation_catalog()
    relation_names = list(relation_catalog)
    raw_relation_scores = score_relation_intents(
        document_vectors,
        relation_catalog,
        vector_dim=vector_dim,
        device=selected_device,
    )
    relation_score_rows = _tensor_score_rows(raw_relation_scores)
    descriptor = _build_tensor_descriptor(
        requested_device=requested_device,
        selected_device=selected_device,
        vector_dim=vector_dim,
        entity_candidate_count=len(entity_candidates),
        relation_candidate_count=len(relation_catalog),
        batch_size=len(documents),
    )
    results: list[ExtractionBackendResult] = []
    for index, rule_result in enumerate(rule_results):
        entity_score_by_name_type = {
            candidate: _normalize_tensor_score(score)
            for candidate, score in zip(entity_candidates, entity_score_rows[index])
        }
        relation_score_by_name = {
            name: _normalize_tensor_score(score)
            for name, score in zip(relation_names, relation_score_rows[index])
        }
        results.append(
            _apply_tensor_scores_to_rule_result(
                rule_result,
                entity_score_by_name_type=entity_score_by_name_type,
                relation_score_by_name=relation_score_by_name,
                descriptor=descriptor,
            )
        )
    return results


KG_BACKENDS: dict[str, KGExtractionBackend] = {
    "rule": RuleKGExtractionBackend(),
    "tensor": TensorKGExtractionBackend(),
    "llm": LLMKGExtractionBackend(),
}


def _normalize_requested_backend(backend: str | None, use_llm: bool | None = None) -> str:
    if backend:
        normalized = backend.strip().lower()
    elif use_llm is True:
        normalized = "llm"
    elif use_llm is False:
        normalized = "rule"
    else:
        normalized = os.getenv("MEDICAL_KG_BACKEND", "rule").strip().lower()
    aliases = {
        "rules": "rule",
        "rule_medical_kg_backend": "rule",
        "tensor_medical_kg_backend": "tensor",
        "npu": "tensor",
        "npu_tensor": "tensor",
        "tensor_ranker": "tensor",
        "llm_medical_kg_backend": "llm",
        "model": "llm",
        "ner": "llm",
        "re": "llm",
    }
    return aliases.get(normalized, normalized)


def select_kg_backend(backend: str | None = None, use_llm: bool | None = None) -> KGBackendSelection:
    requested = _normalize_requested_backend(backend, use_llm)
    llm_configured = get_openai_compat_config() is not None
    available = sorted(KG_BACKENDS)
    if requested == "llm" and not llm_configured:
        return KGBackendSelection(
            requested_backend=requested,
            selected_backend="rule",
            llm_configured=False,
            available_backends=available,
            fallback_reason="llm_not_configured",
        )
    if requested not in KG_BACKENDS:
        return KGBackendSelection(
            requested_backend=requested,
            selected_backend="rule",
            llm_configured=llm_configured,
            available_backends=available,
            fallback_reason=f"unknown_backend:{requested}",
        )
    return KGBackendSelection(
        requested_backend=requested,
        selected_backend=requested,
        llm_configured=llm_configured,
        available_backends=available,
    )


def run_kg_backend(
    document: MedicalDocument,
    *,
    backend: str | None = None,
    use_llm: bool | None = None,
) -> ExtractionBackendResult:
    selection = select_kg_backend(backend=backend, use_llm=use_llm)
    backend_impl = KG_BACKENDS[selection.selected_backend]
    try:
        result = backend_impl.run(document)
    except Exception as exc:
        if selection.selected_backend != "llm":
            raise
        fallback = KG_BACKENDS["rule"].run(document)
        metadata = dict(fallback.metadata)
        metadata.update(
            {
                "requested_backend": selection.requested_backend,
                "selected_backend": "rule",
                "fallback_reason": f"{selection.selected_backend}_failed:{exc.__class__.__name__}",
                "fallback_error": str(exc)[:500],
                "available_backends": selection.available_backends,
                "llm_configured": selection.llm_configured,
            }
        )
        return ExtractionBackendResult(
            backend=fallback.backend,
            version=fallback.version,
            entities=fallback.entities,
            triples=fallback.triples,
            main_disease=fallback.main_disease,
            disease_alias=fallback.disease_alias,
            metadata=metadata,
        )

    metadata = dict(result.metadata)
    metadata.update(
        {
            "requested_backend": selection.requested_backend,
            "selected_backend": selection.selected_backend,
            "fallback_reason": selection.fallback_reason,
            "available_backends": selection.available_backends,
            "llm_configured": selection.llm_configured,
        }
    )
    return ExtractionBackendResult(
        backend=result.backend,
        version=result.version,
        entities=result.entities,
        triples=result.triples,
        main_disease=result.main_disease,
        disease_alias=result.disease_alias,
        metadata=metadata,
    )


def run_rule_kg_backend(document: MedicalDocument) -> ExtractionBackendResult:
    """Run the deterministic KG backend and expose backend metadata.

    This keeps the local pipeline reproducible while making the extraction
    backend replaceable by future LLM/NER/RE implementations.
    """
    return RuleKGExtractionBackend().run(document)


def build_processed_document(document: MedicalDocument) -> dict[str, Any]:
    return normalize_fields(
        {
            "id": document.doc_id,
            "title": document.title,
            "source": document.source,
            "text": document.text,
        },
        1,
    )


def build_graph_snapshot(entities: list[dict[str, Any]], triples: list[dict[str, Any]]) -> dict[str, Any]:
    entity_lookup = {
        (entity["name"], entity["type"]): entity["entity_id"]
        for entity in entities
    }
    nodes = [
        {
            "id": entity["entity_id"],
            "name": entity["name"],
            "type": entity["type"],
            "aliases": entity["aliases"],
            "source_doc_ids": entity["source_doc_ids"],
        }
        for entity in entities
    ]
    edges = [
        {
            "id": triple["triple_id"],
            "source": entity_lookup[(triple["subject"], triple["subject_type"])],
            "target": entity_lookup[(triple["object"], triple["object_type"])],
            "predicate": triple["predicate"],
            "source_doc_id": triple["source_doc_id"],
            "confidence": triple["confidence"],
        }
        for triple in triples
        if (triple["subject"], triple["subject_type"]) in entity_lookup
        and (triple["object"], triple["object_type"]) in entity_lookup
    ]
    entity_type_counts = Counter(entity["type"] for entity in entities)
    relation_type_counts = Counter(triple["predicate"] for triple in triples)
    return {
        "nodes": nodes,
        "edges": edges,
        "statistics": {
            "entity_type_counts": dict(sorted(entity_type_counts.items())),
            "relation_type_counts": dict(sorted(relation_type_counts.items())),
            "entity_count": len(nodes),
            "triple_count": len(edges),
        },
    }


def build_subgraph(
    question: str,
    triples: list[dict[str, Any]],
    *,
    limit: int = 12,
) -> dict[str, Any]:
    disease = detect_question_disease(question)
    relations = _detect_question_relations(question)
    if disease:
        subject_triples = [item for item in triples if item["subject"] == disease]
    else:
        subject_triples = triples[:]
    if relations:
        filtered = [item for item in subject_triples if item["predicate"] in relations]
        if filtered:
            subject_triples = filtered

    selected = rank_supporting_triples(
        question,
        subject_triples,
        disease=disease,
        relations=relations,
        limit=limit,
    )
    entities: dict[tuple[str, str], dict[str, Any]] = {}
    for triple in selected:
        entities[(triple["subject"], triple["subject_type"])] = {
            "name": triple["subject"],
            "type": triple["subject_type"],
        }
        entities[(triple["object"], triple["object_type"])] = {
            "name": triple["object"],
            "type": triple["object_type"],
        }

    nodes = [
        {"id": f"N{index:03d}", "name": entity["name"], "type": entity["type"]}
        for index, entity in enumerate(sorted(entities.values(), key=lambda item: (item["type"], item["name"])), start=1)
    ]
    node_lookup = {(node["name"], node["type"]): node["id"] for node in nodes}
    edges = [
        {
            "id": triple["triple_id"],
            "source": node_lookup[(triple["subject"], triple["subject_type"])],
            "target": node_lookup[(triple["object"], triple["object_type"])],
            "predicate": triple["predicate"],
            "evidence": triple.get("evidence", ""),
            "source_doc_id": triple["source_doc_id"],
            "confidence": triple["confidence"],
        }
        for triple in selected
        if (triple["subject"], triple["subject_type"]) in node_lookup
        and (triple["object"], triple["object_type"]) in node_lookup
    ]
    return {
        "question": question,
        "focus_disease": disease,
        "relations": relations,
        "nodes": nodes,
        "edges": edges,
        "source_triples": selected,
    }


def build_charts(entities: list[dict[str, Any]], triples: list[dict[str, Any]]) -> dict[str, Any]:
    entity_type_counts = Counter(entity["type"] for entity in entities)
    relation_type_counts = Counter(triple["predicate"] for triple in triples)
    return {
        "entity_types": {
            "labels": list(entity_type_counts.keys()),
            "values": list(entity_type_counts.values()),
        },
        "relation_types": {
            "labels": list(relation_type_counts.keys()),
            "values": list(relation_type_counts.values()),
        },
    }


def _format_support_lines(triples: list[dict[str, Any]]) -> list[str]:
    lines = []
    for idx, triple in enumerate(triples, start=1):
        lines.append(
            f"{idx}. {triple['subject']} -{triple['predicate']}-> {triple['object']}，来源 {triple['source_doc_id']}"
        )
    return lines


def _detect_question_relations(question: str) -> list[str]:
    relations: list[str] = []
    for relation, keywords in QUESTION_RELATION_HINTS.items():
        if any(keyword in question for keyword in keywords):
            relations.append(relation)
    return relations


def _question_keywords(question: str, disease: str | None) -> list[str]:
    keywords = [disease] if disease else []
    for relation, hints in QUESTION_RELATION_HINTS.items():
        if any(hint in question for hint in hints):
            keywords.extend(hints)
            keywords.append(relation)
    for token in ("为什么", "如何", "哪些", "需要", "风险", "治疗", "检查", "预防", "导致", "原因"):
        if token in question:
            keywords.append(token)
    return [item for item in dict.fromkeys(keywords) if item]


def _triple_keyword_overlap(triple: dict[str, Any], keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    text = " ".join(
        str(triple.get(key, ""))
        for key in ("subject", "predicate", "object", "subject_type", "object_type", "evidence")
    )
    matched = sum(1 for keyword in keywords if keyword and keyword in text)
    return matched / len(keywords)


def rank_supporting_triples(
    question: str,
    triples: list[dict[str, Any]],
    *,
    disease: str | None = None,
    relations: list[str] | None = None,
    limit: int = 12,
) -> list[dict[str, Any]]:
    disease = disease or detect_question_disease(question)
    relations = relations if relations is not None else _detect_question_relations(question)
    relation_set = set(relations)
    keywords = _question_keywords(question, disease)
    ranked: list[dict[str, Any]] = []
    for triple in triples:
        try:
            confidence = float(triple.get("confidence", 0.75))
        except (TypeError, ValueError):
            confidence = 0.75
        relation = str(triple.get("predicate", ""))
        subject_match = 1.0 if disease and triple.get("subject") == disease else 0.0
        relation_match = 1.0 if relation_set and relation in relation_set else 0.0
        if not relation_set:
            relation_match = 0.35
        evidence_score = 1.0 if str(triple.get("evidence", "")).strip() else 0.0
        relation_weight = QUESTION_RELATION_WEIGHTS.get(relation, 0.85)
        object_type_weight = QUESTION_OBJECT_TYPE_WEIGHTS.get(str(triple.get("object_type", "")), 1.0)
        overlap = _triple_keyword_overlap(triple, keywords)
        evidence_quality = triple.get("evidence_quality")
        evidence_quality_score = 0.0
        if isinstance(evidence_quality, dict):
            try:
                evidence_quality_score = float(evidence_quality.get("score", 0))
            except (TypeError, ValueError):
                evidence_quality_score = 0.0
        score = (
            0.30 * confidence
            + 0.24 * subject_match
            + 0.18 * relation_match
            + 0.13 * overlap
            + 0.08 * evidence_score
            + 0.07 * evidence_quality_score
        ) * relation_weight * object_type_weight
        item = dict(triple)
        item["qa_score"] = round(min(score, 1.5), 4)
        item["qa_score_features"] = {
            "confidence": round(confidence, 4),
            "subject_match": subject_match,
            "relation_match": relation_match,
            "keyword_overlap": round(overlap, 4),
            "evidence_present": evidence_score,
            "evidence_quality_score": round(evidence_quality_score, 4),
            "relation_weight": relation_weight,
            "object_type_weight": object_type_weight,
        }
        ranked.append(item)
    ranked.sort(
        key=lambda item: (
            -float(item.get("qa_score", 0)),
            str(item.get("predicate", "")),
            str(item.get("object", "")),
        )
    )
    return ranked[:limit]


def find_reasoning_paths(
    question: str,
    triples: list[dict[str, Any]],
    *,
    disease: str | None = None,
    max_depth: int = 2,
    limit: int = 6,
) -> list[dict[str, Any]]:
    disease = disease or detect_question_disease(question)
    if not disease:
        return []
    adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for triple in triples:
        subject = str(triple.get("subject", ""))
        if not subject:
            continue
        adjacency[subject].append(triple)

    relation_bias = {
        "causes": 1.18,
        "has_complication": 1.12,
        "has_risk_factor": 1.1,
        "treated_by": 1.04,
        "diagnosed_by": 1.02,
    }
    paths: list[dict[str, Any]] = []
    queue: list[tuple[str, list[dict[str, Any]], set[str]]] = [(disease, [], {disease})]
    while queue and len(paths) < limit * 5:
        current, path, visited = queue.pop(0)
        if len(path) >= max_depth:
            continue
        ranked_edges = rank_supporting_triples(
            question,
            adjacency.get(current, []),
            disease=current,
            relations=None,
            limit=10,
        )
        for edge in ranked_edges:
            target = str(edge.get("object", ""))
            if not target or target in visited:
                continue
            next_path = path + [edge]
            if next_path:
                confidences = []
                score = 1.0
                for item in next_path:
                    confidence = float(item.get("confidence", 0.75))
                    confidences.append(confidence)
                    score *= confidence * relation_bias.get(str(item.get("predicate", "")), 1.0)
                paths.append(
                    {
                        "start": disease,
                        "end": target,
                        "length": len(next_path),
                        "nodes": [disease] + [str(item.get("object", "")) for item in next_path],
                        "relations": [str(item.get("predicate", "")) for item in next_path],
                        "score": round(min(score, 1.0), 4),
                        "mean_confidence": round(sum(confidences) / len(confidences), 4),
                        "triples": [
                            {
                                "subject": item.get("subject"),
                                "predicate": item.get("predicate"),
                                "object": item.get("object"),
                                "source_doc_id": item.get("source_doc_id"),
                                "confidence": item.get("confidence"),
                            }
                            for item in next_path
                        ],
                    }
                )
            queue.append((target, next_path, visited | {target}))
    paths.sort(key=lambda item: (-float(item["score"]), -int(item["length"]), item["end"]))
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for path in paths:
        key = (tuple(path["nodes"]), tuple(path["relations"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
        if len(deduped) >= limit:
            break
    return deduped


def detect_question_disease(question: str) -> str | None:
    for alias, canonical in DISEASE_PATTERNS:
        if alias in question:
            return canonical
    return None


def answer_question(question: str, triples: list[dict[str, Any]]) -> dict[str, Any]:
    disease = detect_question_disease(question)
    relations = _detect_question_relations(question)
    if not disease:
        return {
            "answer": "未能识别问题中的疾病名称，请换一种说法。",
            "supporting_triples": [],
            "ranked_supporting_triples": [],
            "reasoning_paths": [],
            "subgraph": build_subgraph(question, triples),
        }

    subject_triples = [item for item in triples if item["subject"] == disease]
    if relations:
        filtered = [item for item in subject_triples if item["predicate"] in relations]
        if filtered:
            subject_triples = filtered

    ranked_supporting_triples = rank_supporting_triples(
        question,
        subject_triples,
        disease=disease,
        relations=relations,
        limit=12,
    )
    reasoning_paths = find_reasoning_paths(question, triples, disease=disease, max_depth=2, limit=6)

    buckets: dict[str, list[str]] = defaultdict(list)
    for item in ranked_supporting_triples:
        buckets[item["predicate"]].append(item["object"])

    answer_parts: list[str] = []
    if buckets.get("has_symptom"):
        answer_parts.append(f"{disease}常见症状包括{_join_zh(buckets['has_symptom'])}。")
    if buckets.get("treated_by"):
        answer_parts.append(f"治疗方式包括{_join_zh(buckets['treated_by'])}。")
    if buckets.get("diagnosed_by"):
        answer_parts.append(f"辅助检查包括{_join_zh(buckets['diagnosed_by'])}。")
    if buckets.get("has_complication"):
        answer_parts.append(f"相关并发症或风险包括{_join_zh(buckets['has_complication'])}。")
    if buckets.get("belongs_to_department"):
        answer_parts.append(f"建议就诊科室包括{_join_zh(buckets['belongs_to_department'])}。")
    if buckets.get("contraindicated_for"):
        answer_parts.append(f"相关禁忌或慎用对象包括{_join_zh(buckets['contraindicated_for'])}。")
    if buckets.get("has_risk_factor"):
        answer_parts.append(f"相关危险因素包括{_join_zh(buckets['has_risk_factor'])}。")
    if buckets.get("prevented_by"):
        answer_parts.append(f"预防或管理措施包括{_join_zh(buckets['prevented_by'])}。")
    if reasoning_paths and any(keyword in question for keyword in ("为什么", "导致", "原因", "因果", "风险", "相关")):
        top_path = reasoning_paths[0]
        answer_parts.append(
            "图谱推理路径显示："
            f"{' -> '.join(top_path['nodes'])}"
            f"（关系：{' -> '.join(top_path['relations'])}）。"
        )

    if not answer_parts:
        answer_parts.append(f"未找到与{disease}直接相关的结构化结论。")

    return {
        "answer": "".join(answer_parts),
        "supporting_triples": ranked_supporting_triples[:10],
        "ranked_supporting_triples": ranked_supporting_triples,
        "reasoning_paths": reasoning_paths,
        "subgraph": build_subgraph(question, triples),
    }


def _join_zh(items: list[str]) -> str:
    items = _sorted_unique(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return "和".join(items)
    return "、".join(items[:-1]) + f"和{items[-1]}"
