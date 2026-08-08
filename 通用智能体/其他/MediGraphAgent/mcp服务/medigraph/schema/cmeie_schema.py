"""Complete CMeIE-V2 schema adapter.

The project graph keeps a compact clinical ontology for user-facing QA.  This
adapter preserves all 53 official schema rows (44 predicate labels) for strict
benchmark evaluation, avoiding the misleading practice of scoring only an easy
relation subset while calling it "full CMeIE".
"""
from __future__ import annotations

import json
from pathlib import Path

CMEIE_ENTITY_TYPES: dict[str, str] = {
    "疾病": "Disease",
    "症状": "Symptom",
    "检查": "Examination",
    "药物": "Drug",
    "部位": "Body",
    "手术治疗": "Procedure",
    "其他治疗": "Procedure",
    "流行病学": "Epidemiology",
    "社会学": "SocialFactor",
    "预后": "Prognosis",
    "其他": "Other",
}

CMEIE_PREDICATE_KEYS: dict[str, str] = {
    "预防": "prevention",
    "阶段": "stage",
    "就诊科室": "department",
    "同义词": "synonym",
    "辅助治疗": "adjuvant_therapy",
    "化疗": "chemotherapy",
    "放射治疗": "radiotherapy",
    "手术治疗": "surgical_treatment",
    "实验室检查": "laboratory_examination",
    "影像学检查": "imaging_examination",
    "辅助检查": "auxiliary_examination",
    "组织学检查": "histological_examination",
    "内窥镜检查": "endoscopic_examination",
    "筛查": "screening",
    "多发群体": "susceptible_population",
    "发病率": "incidence",
    "发病年龄": "onset_age",
    "多发地区": "high_incidence_area",
    "发病性别倾向": "sex_tendency",
    "死亡率": "mortality",
    "多发季节": "seasonality",
    "传播途径": "transmission_route",
    "并发症": "complication",
    "病理分型": "pathological_classification",
    "相关（导致）": "related_cause",
    "鉴别诊断": "differential_diagnosis",
    "相关（转化）": "related_transformation",
    "相关（症状）": "related_symptom",
    "临床表现": "clinical_manifestation",
    "治疗后症状": "post_treatment_symptom",
    "侵及周围组织转移的症状": "invasion_symptom",
    "病因": "etiology",
    "高危因素": "high_risk_factor",
    "风险评估因素": "risk_assessment_factor",
    "病史": "medical_history",
    "遗传因素": "genetic_factor",
    "发病机制": "pathogenesis",
    "病理生理": "pathophysiology",
    "药物治疗": "drug_treatment",
    "发病部位": "onset_site",
    "转移部位": "metastatic_site",
    "外侵部位": "invasion_site",
    "预后状况": "prognosis_status",
    "预后生存率": "prognosis_survival_rate",
}


def predicate_key(predicate: str) -> str:
    key = CMEIE_PREDICATE_KEYS.get(predicate)
    return f"cmeie:{key}" if key else ""


#: cmeie:<key> -> Chinese predicate, for displaying edges that only carry the raw
#: schema key (as graph_scaled.json's does -- it stores the full 53-row CMeIE
#: predicate space, not the compact CM3KG-style relation set that ships a
#: `relation_zh` on each edge). Built once at import time from the same table
#: `predicate_key()` uses, so the two directions can never drift apart.
CMEIE_KEY_TO_PREDICATE: dict[str, str] = {
    f"cmeie:{key}": predicate for predicate, key in CMEIE_PREDICATE_KEYS.items()
}


def predicate_zh(relation: str) -> str:
    """Best-effort Chinese label for a relation key.

    Handles both relation vocabularies used across the two graph artefacts:
    CM3KG-style bare keys (resolved via `medigraph.schema.ontology.RELATION_TYPES`
    by the caller) and CMeIE-style `cmeie:xxx` keys (resolved here). Falls back to
    the raw key so a caller can always safely display *something*.
    """
    return CMEIE_KEY_TO_PREDICATE.get(relation, relation)


def load_schema(path: str | Path) -> list[dict]:
    """Load the official schema file, tolerating its leading comment line."""
    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("#"):
        text = "\n".join(text.splitlines()[1:])
    rows = json.loads(text)
    result = []
    for row in rows:
        object_type = row.get("object_type", "")
        if isinstance(object_type, dict):
            object_type = object_type.get("@value", "")
        result.append(
            {
                "subject_type": str(row.get("subject_type", "")),
                "predicate": str(row.get("predicate", "")),
                "predicate_key": predicate_key(str(row.get("predicate", ""))),
                "object_type": str(object_type),
                "direction": int(row.get("direction", 1)),
            }
        )
    return result
