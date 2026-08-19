from __future__ import annotations

import re
from typing import Any

from medinsight.common.knowledge_graph import stable_id


TEXT_KG_RELATION_SCHEMAS = [
    {"subject_type": "疾病", "predicate": "临床表现", "object_type": "症状", "source_dataset": "generic_medical_text"},
    {"subject_type": "疾病", "predicate": "药物治疗", "object_type": "药物", "source_dataset": "generic_medical_text"},
    {"subject_type": "疾病", "predicate": "辅助治疗", "object_type": "治疗方法", "source_dataset": "generic_medical_text"},
    {"subject_type": "疾病", "predicate": "辅助检查", "object_type": "医学检查项目", "source_dataset": "generic_medical_text"},
    {"subject_type": "疾病", "predicate": "并发症", "object_type": "疾病", "source_dataset": "generic_medical_text"},
    {"subject_type": "疾病", "predicate": "病因", "object_type": "病因", "source_dataset": "generic_medical_text"},
    {"subject_type": "疾病", "predicate": "病理分型", "object_type": "疾病", "source_dataset": "generic_medical_text"},
    {"subject_type": "疾病", "predicate": "就诊科室", "object_type": "科室", "source_dataset": "generic_medical_text"},
]


COMMON_DISEASES = {
    "甲型流感",
    "急性肠胃炎",
    "新型冠状病毒肺炎",
    "急性上呼吸道感染",
    "上呼吸道感染",
    "细菌性咽炎",
    "扁桃体炎",
    "普通感冒",
    "流行性感冒",
    "流感",
    "感冒",
    "风寒感冒",
    "肺炎",
    "细菌性肺炎",
    "病毒性肺炎",
    "支原体肺炎",
    "衣原体肺炎",
    "支气管炎",
    "嗜酸粒细胞性支气管炎",
    "哮喘",
    "咳嗽变异性哮喘",
    "新冠感染",
    "慢性胃炎",
    "浅表性胃炎",
    "萎缩性胃炎",
    "结肠炎",
    "慢性咽炎",
    "过敏性咽炎",
    "急性咽喉炎",
    "便秘",
    "痛风",
    "急性痛风关节炎",
    "慢性痛风关节炎",
    "痛风性肾病",
    "肺结核",
    "急性粟粒性肺结核",
    "纤维空洞型肺结核",
    "气管、支气管结核",
    "缺铁性贫血",
    "原发性高血压",
    "继发性高血压",
    "胃食管反流",
    "胃食管反流性咳嗽",
    "小脑梗塞",
    "脑干梗塞",
    "脑桥梗塞",
    "糖尿病",
    "高血压",
    "冠心病",
    "脑梗死",
    "胃炎",
    "胃溃疡",
    "肝炎",
    "贫血",
    "溶血性贫血",
    "霍乱",
    "手足口病",
    "急性阑尾炎",
    "无菌性脑膜炎",
}

COMMON_SYMPTOMS = {
    "发热",
    "发烧",
    "咳嗽",
    "干咳",
    "咳痰",
    "咽痛",
    "鼻塞",
    "流涕",
    "头痛",
    "头晕",
    "乏力",
    "胸痛",
    "胸闷",
    "气促",
    "呼吸困难",
    "恶心",
    "呕吐",
    "腹痛",
    "腹泻",
    "黄疸",
    "皮疹",
    "肌肉酸痛",
    "肌痛",
    "周身关节疼痛",
    "高热",
    "午后低热",
    "消瘦",
    "白细胞计数升高",
    "中性粒细胞计数升高",
    "皮肤黏膜苍白",
    "眼前发黑",
    "耳鸣",
    "食欲减退",
    "胃痛",
    "胃部不适",
    "反酸",
    "烧心",
    "口干",
    "口苦",
    "腹胀",
    "厌食",
    "消化不良",
    "慢性腹痛",
    "便血",
    "黑便",
    "大便不成形",
    "大便带黏液",
    "嗓子干痒",
    "咽腔黏膜痒感",
    "异物感",
    "视物不清",
    "眼前闪光点",
    "肚子胀气",
    "眩晕",
    "共济失调",
    "复视",
    "偏瘫",
    "声音嘶哑",
    "嗳气",
    "吞咽困难",
    "气短",
    "四肢无力",
    "心前区不适",
    "心口堵",
    "胸口疼痛",
    "压迫感",
    "冒汗",
    "出汗",
    "心绞痛",
    "心慌",
    "手抖",
    "饥饿感",
    "低血糖发作",
    "流鼻涕",
    "打喷嚏",
    "咽喉疼痛",
    "咽喉肿痛",
    "颈项强直",
    "频繁呕吐",
    "后头部疼痛",
    "交叉性瘫痪",
    "四肢瘫",
    "红肿热痛",
}

COMMON_DRUGS = {
    "奥司他韦",
    "对乙酰氨基酚",
    "布洛芬",
    "阿莫西林",
    "头孢",
    "青霉素",
    "二甲双胍",
    "胰岛素",
    "硝酸甘油",
    "阿司匹林",
    "他汀类",
    "氯吡格雷",
    "蒙脱石散",
    "奥斯他韦",
    "阿比多尔",
    "抗病毒药物",
    "退热药物",
    "扑热息痛",
    "芬必得",
    "抗生素",
    "抗菌素",
    "消炎药",
    "糖皮质激素类药物",
    "抗过敏药物",
    "抗结核药物",
    "降压药物",
    "布地奈德",
    "色甘酸钠",
    "孟鲁司特钠",
    "头孢地尼",
    "左氧氟沙星",
    "氯雷他定片",
    "金喉健喷雾剂",
    "苏黄止咳胶囊",
    "莫沙必利",
    "伊托必利",
    "吗丁啉",
    "小儿氨酚黄那敏颗粒",
    "感冒清热颗粒",
}

COMMON_CHECKS = {
    "血常规",
    "白细胞计数",
    "中性粒细胞计数",
    "尿常规",
    "便常规",
    "肝功能",
    "肾功能",
    "胸片",
    "胸部CT",
    "胸部ct",
    "CT",
    "核酸检测",
    "新冠病毒核酸检测",
    "病原学检测",
    "心电图",
    "动态心电图",
    "冠状动脉CT",
    "冠状动脉ct",
    "头颅CT",
    "头颅ct",
    "头颅磁共振",
    "肺部x线片",
    "肺部的x线片",
    "结核菌素试验",
    "抽血化验",
    "胃镜",
    "胃镜检查",
    "超声",
    "胸部x线",
    "肺CT",
    "肺部CT",
    "肺部ct",
    "C反应蛋白",
    "c-反应蛋白",
    "降钙素原",
    "甲流核酸检测",
    "气道激发实验",
    "气道可逆试验",
    "痰的嗜酸粒细胞计数",
    "食管测压",
    "24小时反流监测",
    "血清电解质",
    "甲状腺功能",
    "血压监测",
}

COMMON_DEPARTMENTS = {"呼吸内科", "消化内科", "感染科", "心内科", "神经内科", "儿科", "急诊科", "血液科"}

COMMON_TREATMENTS = {
    "抗生素治疗",
    "止咳化痰",
    "抗感染治疗",
    "止咳",
    "化痰",
    "平喘",
    "对症治疗",
    "物理降温",
    "抗炎治疗",
    "抗结核药物治疗",
    "抗心绞痛治疗",
    "降压治疗",
    "低钠饮食",
    "控制食盐摄入",
    "适当锻炼",
    "戒烟",
    "限制饮酒",
    "避免情绪激动",
    "清淡饮食",
    "低盐、低脂肪饮食",
    "脱离过敏原",
    "减轻气道炎症",
    "隔离",
    "多喝水",
    "休息",
    "饮食调整",
    "消化专科诊治",
}

COMMON_CAUSES = {
    "多种病原菌感染",
    "病原菌感染",
    "细菌感染",
    "病毒感染",
    "甲型流感病毒感染",
    "甲型流感病毒",
    "鼻病毒",
    "柯萨奇病毒",
    "幽门螺杆菌感染",
    "饮食结构",
    "饮食习惯",
    "肠道菌群失调",
    "肠道黏膜炎症刺激",
    "肠道蠕动亢进",
    "长期血尿酸增高",
    "血尿酸增高",
    "肾动脉狭窄",
    "肾脏病",
    "甲状腺功能亢进",
    "遗传因素",
    "花粉",
    "冷空气",
    "咖啡因",
    "交感神经兴奋",
    "胃液反流",
    "胃酸刺激",
    "缺铁",
}

ENTITY_LEXICONS = {
    "疾病": COMMON_DISEASES,
    "症状": COMMON_SYMPTOMS,
    "药物": COMMON_DRUGS,
    "医学检查项目": COMMON_CHECKS,
    "科室": COMMON_DEPARTMENTS,
    "治疗方法": COMMON_TREATMENTS,
    "病因": COMMON_CAUSES,
}

SORTED_COMMON_DISEASES = sorted(COMMON_DISEASES, key=len, reverse=True)
SORTED_COMMON_SYMPTOMS = sorted(COMMON_SYMPTOMS, key=len, reverse=True)
SORTED_COMMON_DRUGS = sorted(COMMON_DRUGS, key=len, reverse=True)
SORTED_COMMON_CHECKS = sorted(COMMON_CHECKS, key=len, reverse=True)
SORTED_COMMON_TREATMENTS = sorted(COMMON_TREATMENTS, key=len, reverse=True)
SORTED_COMMON_CAUSES = sorted(COMMON_CAUSES, key=len, reverse=True)
SORTED_ENTITY_LEXICONS = {
    entity_type: sorted(terms, key=len, reverse=True)
    for entity_type, terms in ENTITY_LEXICONS.items()
}


def extract_text_relations(text: str, doc_id: str) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    current_disease = ""
    for sentence_index, sentence in enumerate(split_sentences(text)):
        detected_disease = detect_disease(sentence)
        if current_disease and current_disease in sentence:
            if detected_disease and detected_disease != current_disease:
                detected_before_current = sentence.find(detected_disease) <= sentence.find(current_disease)
                detected_is_prefixed_specific = sentence.lstrip().startswith(detected_disease)
                detected_is_subtype = detected_disease.endswith(current_disease)
                if detected_is_subtype and not detected_is_prefixed_specific:
                    disease = current_disease
                elif detected_before_current:
                    disease = detected_disease
                else:
                    disease = current_disease
            else:
                disease = current_disease
        else:
            disease = detected_disease or current_disease
        if not disease:
            continue
        current_disease = disease
        relation_specs = [
            (
                "临床表现",
                "症状",
                [
                    "主要临床表现是",
                    "主要临床表现为",
                    "主要症状是",
                    "主要症状为",
                    "症状主要以",
                    "临床症状无特异性,主要表现为",
                    "临床症状无特异性，主要表现为",
                    "主要有",
                    "主要以",
                    "主要表现为",
                    "表现为",
                    "是以",
                    "最典型的症状是",
                    "典型症状是",
                    "最常见的是",
                    "常见的症状有以下几种",
                    "常见的症状有",
                    "常见的临床表现包括",
                    "症状包括",
                    "症状是",
                    "症状有",
                    "症状,如",
                    "症状，如",
                    "症状为",
                    "临床表现包括",
                    "临床表现为",
                    "通常包括",
                    "可出现",
                    "会出现",
                    "可以出现",
                    "伴随有",
                    "伴有",
                    "常伴有",
                    "可伴有",
                    "患者可伴有",
                    "表现出",
                ],
            ),
            (
                "药物治疗",
                "药物",
                [
                    "药物治疗包括",
                    "药物治疗,像",
                    "药物治疗，像",
                    "药物可用",
                    "可用",
                    "可使用",
                    "可以使用",
                    "可应用",
                    "给予",
                    "服用",
                    "使用",
                    "可以口服",
                    "口服",
                ],
            ),
            ("辅助治疗", "治疗方法", ["治疗包括", "治疗方式包括", "可采用", "综合治疗", "对症治疗"]),
            (
                "辅助检查",
                "医学检查项目",
                [
                    "检查包括",
                    "检查可进行",
                    "可进行",
                    "建议进行",
                    "需要进行",
                    "需要完善",
                    "需要行",
                    "需要做",
                    "诊断依靠",
                    "辅助检查",
                    "辅助检查包括",
                ],
            ),
            ("并发症", "疾病", ["并发症包括", "可并发", "并发"]),
            ("病因", "病因", ["病因包括", "原因如下", "主要是由于", "可能是因为", "由于", "是由", "由"]),
            ("病理分型", "疾病", ["分为两种", "分为"]),
            ("就诊科室", "科室", ["就诊科室为", "建议就诊", "可就诊"]),
        ]
        for predicate, object_type, triggers in relation_specs:
            if predicate == "病因" and disease not in sentence:
                continue
            for trigger in triggers:
                span = extract_after_trigger(sentence, trigger)
                if not span:
                    continue
                for object_value in split_terms(span):
                    object_value = normalize_object_value(object_value, object_type)
                    if not object_value or object_value == disease:
                        continue
                    if not should_keep_object_value(object_value, object_type):
                        continue
                    if predicate == "药物治疗" and object_value not in COMMON_DRUGS:
                        continue
                    predicate_for_value = predicate
                    object_type_for_value = object_type
                    key = (disease, predicate_for_value, object_value)
                    if key in seen:
                        continue
                    seen.add(key)
                    relations.append(
                        make_relation(
                            doc_id=doc_id,
                            sentence_index=sentence_index,
                            subject=disease,
                            predicate=predicate_for_value,
                            object_value=object_value,
                            object_type=object_type_for_value,
                            evidence=sentence,
                        )
                    )
        for relation in extract_context_symptom_relations(
            doc_id=doc_id,
            sentence_index=sentence_index,
            disease=disease,
            sentence=sentence,
            seen=seen,
        ):
            relations.append(relation)
        for relation in extract_context_check_relations(
            doc_id=doc_id,
            sentence_index=sentence_index,
            disease=disease,
            sentence=sentence,
            seen=seen,
        ):
            relations.append(relation)
        for relation in extract_context_treatment_relations(
            doc_id=doc_id,
            sentence_index=sentence_index,
            disease=disease,
            sentence=sentence,
            seen=seen,
        ):
            relations.append(relation)
        for relation in extract_context_subtype_relations(
            doc_id=doc_id,
            sentence_index=sentence_index,
            disease=disease,
            sentence=sentence,
            seen=seen,
        ):
            relations.append(relation)
        for relation in extract_context_cause_relations(
            doc_id=doc_id,
            sentence_index=sentence_index,
            disease=disease,
            sentence=sentence,
            seen=seen,
        ):
            relations.append(relation)
    return relations


def make_relation(
    *,
    doc_id: str,
    sentence_index: int,
    subject: str,
    predicate: str,
    object_value: str,
    object_type: str,
    evidence: str,
) -> dict[str, Any]:
    return {
        "relation_id": stable_id("rel", "custom-text", doc_id, sentence_index, subject, predicate, object_value),
        "doc_id": doc_id,
        "subject": subject,
        "subject_type": "疾病",
        "predicate": predicate,
        "object": object_value,
        "object_type": object_type,
        "evidence": evidence,
        "source_dataset": "generic_medical_text",
        "source_split": "user_input",
        "confidence": 0.76,
        "properties": {
            "extraction_method": "rule_and_lexicon",
            "note": "Facts are extracted only from the provided text.",
        },
    }


def extract_text_entities(text: str, doc_id: str, relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entities_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    def add_entity(name: str, entity_type: str, method: str, confidence: float = 0.72) -> None:
        cleaned = clean_term(name)
        if not cleaned:
            return
        key = (cleaned, entity_type)
        if key in entities_by_key:
            return
        start_idx = text.find(cleaned)
        entities_by_key[key] = {
            "mention_id": stable_id("mention", "custom-text", doc_id, cleaned, entity_type),
            "doc_id": doc_id,
            "name": cleaned,
            "type": entity_type,
            "type_code": entity_type,
            "start_idx": start_idx if start_idx >= 0 else None,
            "end_idx": start_idx + len(cleaned) - 1 if start_idx >= 0 else None,
            "source_dataset": "generic_medical_text",
            "source_split": "user_input",
            "extraction_method": method,
            "confidence": confidence,
        }

    for relation in relations:
        add_entity(str(relation.get("subject") or ""), "疾病", "relation_subject", 0.8)
        add_entity(str(relation.get("object") or ""), str(relation.get("object_type") or "未知"), "relation_object", 0.78)
    for entity_type, terms in SORTED_ENTITY_LEXICONS.items():
        for term in terms:
            if term in text:
                add_entity(term, entity_type, "lexicon_match", 0.82)
    return list(entities_by_key.values())


def detect_disease(sentence: str) -> str:
    matches = [(sentence.find(disease), -len(disease), disease) for disease in SORTED_COMMON_DISEASES if disease in sentence]
    if matches:
        matches.sort()
        return matches[0][2]
    match = re.match(r"^\s*([^，,。；;：:]{2,20}?)(?:主要表现为|表现为|症状包括|临床表现|可出现|会出现|伴有|治疗|检查|可并发|并发症)", sentence)
    if not match:
        return ""
    candidate = clean_term(match.group(1))
    candidate = re.sub(r"^(患者|儿童|成人|该病|疾病|本病)", "", candidate)
    candidate = re.sub(r"(可|会|常)$", "", candidate)
    if re.search(r"(实验室|检查|诊断|临床|辅助|血常规|胸片|胸部|肺部|胃镜)", candidate):
        return ""
    if re.search(r"(上述|这些|通过|进行|相关|相应|最好|方面|比如|例如|可以|需要)", candidate):
        return ""
    return candidate if 2 <= len(candidate) <= 20 else ""


def extract_after_trigger(sentence: str, trigger: str) -> str:
    if trigger == "由":
        match = re.search(r"由(?!于)", sentence)
        index = match.start() if match else -1
    else:
        index = sentence.find(trigger)
    if index < 0:
        return ""
    span = sentence[index + len(trigger) :]
    stop_markers = [
        "，可使用",
        "，可以使用",
        ",可以使用",
        ",可以用",
        ",可以",
        "，可应用",
        "，给予",
        "，使用",
        "，服用",
        "，可采用",
        "，可能",
        ",可能",
        "，可行",
        ",可行",
        "，并可",
        ",并可",
        "，病情",
        ",病情",
        "，能够",
        ",能够",
        "，治疗",
        "，检查",
        "，诊断",
        "，并发",
        "，可并发",
        "，建议",
        "，需要",
        "，应",
        "，需",
        "，所导致",
        ",所导致",
        ",一般",
        "；",
        ";",
        "。",
    ]
    cut = len(span)
    for marker in stop_markers:
        marker_index = span.find(marker)
        if marker_index >= 0:
            cut = min(cut, marker_index)
    return span[:cut]


def split_terms(span: str) -> list[str]:
    cleaned = re.sub(r"(等症状|等表现|等药物|等检查|等方法|等等|等)$", "", span.strip())
    pieces = re.split(r"[、,，/]|以及|或者|和|及|与", cleaned)
    terms: list[str] = []
    for piece in pieces:
        term = clean_term(piece)
        if 2 <= len(term) <= 24 and term not in terms:
            terms.append(term)
    return terms


def normalize_object_value(value: str, object_type: str) -> str:
    cleaned = clean_term(value)
    if object_type == "症状":
        for symptom in SORTED_COMMON_SYMPTOMS:
            if symptom in cleaned and len(cleaned) > len(symptom):
                return normalize_object_value(symptom, object_type)
    if object_type == "症状" and cleaned == "体温升高":
        return "发热"
    if object_type == "症状" and cleaned == "发烧":
        return "发热"
    if object_type == "疾病":
        for disease in SORTED_COMMON_DISEASES:
            if disease in cleaned:
                return disease
    if object_type == "药物":
        for drug in SORTED_COMMON_DRUGS:
            if drug in cleaned:
                return drug
    if object_type == "治疗方法":
        for treatment in SORTED_COMMON_TREATMENTS:
            if treatment in cleaned:
                return treatment
    if object_type == "病因":
        for cause in SORTED_COMMON_CAUSES:
            if cause in cleaned:
                return cause
    if object_type == "医学检查项目":
        lowered = cleaned.lower()
        if "血常规" in cleaned:
            return "血常规"
        if "白细胞计数" in cleaned:
            return "白细胞计数"
        if "中性粒细胞计数" in cleaned:
            return "中性粒细胞计数"
        if "胸片" in cleaned:
            return "胸片"
        if "胸部ct" in lowered:
            return "胸部CT"
        if "肺ct" in lowered or "肺部ct" in lowered:
            return "肺部CT"
        if "胃镜" in cleaned:
            return "胃镜"
        if "核酸检测" in cleaned:
            return "核酸检测"
        if "冠状动脉ct" in lowered:
            return "冠状动脉CT"
        if "心电图" in cleaned:
            return "心电图"
        if "头颅ct" in lowered:
            return "头颅CT"
        if "头颅磁共振" in cleaned:
            return "头颅磁共振"
        if "x线" in cleaned and "肺部" in cleaned:
            return "肺部x线片"
        if "结核菌素试验" in cleaned:
            return "结核菌素试验"
        if "抽血化验" in cleaned:
            return "抽血化验"
    return cleaned


def should_keep_object_value(value: str, object_type: str) -> bool:
    if object_type == "病因":
        cleaned = value.lstrip("的")
        if "症状" in cleaned or cleaned in COMMON_SYMPTOMS:
            return False
        if any(disease in cleaned for disease in COMMON_DISEASES):
            return False
        if re.search(r"(患者|出现|导致|表现|治疗|检查)", cleaned):
            return False
        return 2 <= len(cleaned) <= 40
    if object_type == "医学检查项目":
        return bool(
            value in COMMON_CHECKS
            or re.search(r"(检查|CT|ct|x线|X线|镜|血常规|心电图|核酸检测|胸片|超声|功能|电解质)", value)
        )
    if object_type == "治疗方法":
        return bool(
            value in COMMON_TREATMENTS
            or re.search(r"(治疗|饮食|锻炼|戒烟|限酒|休息|隔离|降温|止咳|化痰|平喘|诊治)", value)
        )
    if object_type == "疾病":
        return value in COMMON_DISEASES
    if object_type != "症状":
        return True
    if value in COMMON_SYMPTOMS:
        return True
    if re.search(r"(检查|诊断|治疗|药物|感染|培养|结果|比例|指标|胸片|CT|ct|x线|影像|血常规|白细胞|淋巴细胞)", value):
        return False
    if re.search(r"(性质|可以是|不能用|无法|部分|患者|临床|疾病|系统|功能)", value):
        return False
    return bool(
        re.search(
            r"(痛|热|咳|痰|喘|闷|胀|泻|吐|晕|痒|疹|黄疸|乏力|鼻塞|流涕|"
            r"气短|困难|酸痛|心慌|手抖|饥饿感|烧心|反酸|口干|口苦|"
            r"便秘|苍白|耳鸣|食欲|眩晕|共济失调|复视|偏瘫|嘶哑|嗳气|吞咽)",
            value,
        )
    )


def extract_context_symptom_relations(
    *,
    doc_id: str,
    sentence_index: int,
    disease: str,
    sentence: str,
    seen: set[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    if not re.search(r"(症状|表现|不适|疼痛|痛|压迫感|胸闷|气短|心绞痛|^\s*\d+[、.])", sentence):
        return []
    relations: list[dict[str, Any]] = []
    for symptom in SORTED_COMMON_SYMPTOMS:
        if symptom not in sentence:
            continue
        object_value = normalize_object_value(symptom, "症状")
        key = (disease, "临床表现", object_value)
        if key in seen:
            continue
        seen.add(key)
        relations.append(
            make_relation(
                doc_id=doc_id,
                sentence_index=sentence_index,
                subject=disease,
                predicate="临床表现",
                object_value=object_value,
                object_type="症状",
                evidence=sentence,
            )
        )
    return relations


def extract_context_check_relations(
    *,
    doc_id: str,
    sentence_index: int,
    disease: str,
    sentence: str,
    seen: set[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    if not re.search(r"(检查|诊断|确诊|筛查|胸片|CT|ct|x线|X线|胃镜|心电图|超声|化验|监测|试验)", sentence):
        return []
    relations: list[dict[str, Any]] = []
    matched_checks: set[str] = set()
    for check in SORTED_COMMON_CHECKS:
        if check not in sentence:
            continue
        object_value = normalize_object_value(check, "医学检查项目")
        if object_value == "CT" and any(item in matched_checks for item in {"胸部ct", "肺部ct", "头颅ct", "冠状动脉ct"}):
            continue
        matched_checks.add(object_value)
        key = (disease, "辅助检查", object_value)
        if key in seen:
            continue
        seen.add(key)
        relations.append(
            make_relation(
                doc_id=doc_id,
                sentence_index=sentence_index,
                subject=disease,
                predicate="辅助检查",
                object_value=object_value,
                object_type="医学检查项目",
                evidence=sentence,
            )
        )
    return relations


def extract_context_treatment_relations(
    *,
    doc_id: str,
    sentence_index: int,
    disease: str,
    sentence: str,
    seen: set[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    if not re.search(r"(治疗|用药|药物|服用|口服|预防|提倡|建议|需要|应|避免|戒烟|饮食|休息|隔离|锻炼)", sentence):
        return []
    relations: list[dict[str, Any]] = []
    for drug in SORTED_COMMON_DRUGS:
        if drug not in sentence:
            continue
        key = (disease, "药物治疗", drug)
        if key in seen:
            continue
        seen.add(key)
        relations.append(
            make_relation(
                doc_id=doc_id,
                sentence_index=sentence_index,
                subject=disease,
                predicate="药物治疗",
                object_value=drug,
                object_type="药物",
                evidence=sentence,
            )
        )
    for treatment in SORTED_COMMON_TREATMENTS:
        if treatment not in sentence:
            continue
        key = (disease, "辅助治疗", treatment)
        if key in seen:
            continue
        seen.add(key)
        relations.append(
            make_relation(
                doc_id=doc_id,
                sentence_index=sentence_index,
                subject=disease,
                predicate="辅助治疗",
                object_value=treatment,
                object_type="治疗方法",
                evidence=sentence,
            )
        )
    return relations


def extract_context_cause_relations(
    *,
    doc_id: str,
    sentence_index: int,
    disease: str,
    sentence: str,
    seen: set[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    if not re.search(r"(病因|原因|由于|因为|导致|所引起|感染|危险因素|诱发因素|相关)", sentence):
        return []
    relations: list[dict[str, Any]] = []
    for cause in SORTED_COMMON_CAUSES:
        if cause not in sentence:
            continue
        key = (disease, "病因", cause)
        if key in seen:
            continue
        seen.add(key)
        relations.append(
            make_relation(
                doc_id=doc_id,
                sentence_index=sentence_index,
                subject=disease,
                predicate="病因",
                object_value=cause,
                object_type="病因",
                evidence=sentence,
            )
        )
    return relations


def extract_context_subtype_relations(
    *,
    doc_id: str,
    sentence_index: int,
    disease: str,
    sentence: str,
    seen: set[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    has_subtype_marker = bool(re.search(r"(分为|包括|类型|分型|最为常见|常见)", sentence))
    has_parent_by_name = any(parent != disease and disease.endswith(parent) for parent in COMMON_DISEASES)
    if not has_subtype_marker and not has_parent_by_name:
        return []
    relations: list[dict[str, Any]] = []
    subtype_pairs: list[tuple[str, str]] = []
    for subtype in SORTED_COMMON_DISEASES:
        if subtype == disease or subtype not in sentence:
            continue
        if disease not in subtype:
            continue
        subtype_pairs.append((disease, subtype))
    for parent in SORTED_COMMON_DISEASES:
        if parent == disease or not disease.endswith(parent):
            continue
        subtype_pairs.append((parent, disease))
    for parent, subtype in subtype_pairs:
        key = (parent, "病理分型", subtype)
        if key in seen:
            continue
        seen.add(key)
        relations.append(
            make_relation(
                doc_id=doc_id,
                sentence_index=sentence_index,
                subject=parent,
                predicate="病理分型",
                object_value=subtype,
                object_type="疾病",
                evidence=sentence,
            )
        )
    return relations


def clean_term(value: str) -> str:
    cleaned = str(value or "").strip()
    cleaned = re.sub(
        r"^(的|但是|个别人会出现|部分患者是以|可|可以是|可能是|会|常|多伴有|伴有|伴|伴随|出现|进行|使用|服用|给予|采用|例如|如|明显|轻度|重度|持续)",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(为首发症状进行求治|等症状|等表现|等药物|等检查|等方法|的表现|的情况|治疗|检查|为主|为主要表现|为特征|明显|较明显|持续|制剂|等等|等)$",
        "",
        cleaned,
    )
    cleaned = cleaned.strip(" ：:，,。；;、（）()[]【】")
    return cleaned.strip()


def split_sentences(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"[\n。！？!?；;]+", text) if item.strip()]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())
