"""Expand task-2 holdout gold corpora to 30 records (plan phase 1.2)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXTRACTION_RECORDS = [
    {
        "record_id": "holdout_11",
        "text": "记录: 青年女性，甲亢（Graves病）确诊，主诉消瘦、手抖、心悸，甲巯咪唑控制，复查甲状腺超声。",
        "entities": {
            "Disease": ["甲亢"],
            "Symptom": ["消瘦", "手抖", "心悸"],
            "Drug": ["甲巯咪唑"],
            "Examination": ["甲状腺超声"],
            "Treatment": [],
        },
    },
    {
        "record_id": "holdout_12",
        "text": "记录: 甲状腺功能减退被明确诊断，患者怕冷、乏力，口服左甲状腺素，定期查血常规。",
        "entities": {
            "Disease": ["甲状腺功能减退"],
            "Symptom": ["怕冷", "乏力"],
            "Drug": ["左甲状腺素"],
            "Examination": ["血常规"],
            "Treatment": [],
        },
    },
    {
        "record_id": "holdout_13",
        "text": "记录: 疑似冠心病，行冠脉造影及CTA评估，胸闷气短，长期阿司匹林抗血小板。",
        "entities": {
            "Disease": ["冠心病"],
            "Symptom": ["胸闷", "气短"],
            "Drug": ["阿司匹林"],
            "Examination": ["冠脉造影", "CTA"],
            "Treatment": [],
        },
    },
    {
        "record_id": "holdout_14",
        "text": "记录: 系统性红斑狼疮活动期，关节痛、发热，泼尼松免疫抑制治疗，查抗核抗体。",
        "entities": {
            "Disease": ["系统性红斑狼疮"],
            "Symptom": ["关节痛", "发热"],
            "Drug": ["泼尼松"],
            "Examination": ["抗核抗体"],
            "Treatment": ["免疫抑制"],
        },
    },
    {
        "record_id": "holdout_15",
        "text": "记录: 患者否认糖尿病，但口渴多尿仍被记录；最终按2型糖尿病管理，二甲双胍治疗。",
        "entities": {
            "Disease": ["2型糖尿病"],
            "Symptom": ["口渴", "多尿"],
            "Drug": ["二甲双胍"],
            "Examination": [],
            "Treatment": [],
        },
    },
    {
        "record_id": "holdout_16",
        "text": "记录: 高血压未被有效控制，头晕头痛反复，氨氯地平加量，建议低盐饮食。",
        "entities": {
            "Disease": ["高血压"],
            "Symptom": ["头晕", "头痛"],
            "Drug": ["氨氯地平"],
            "Examination": [],
            "Treatment": ["低盐饮食"],
        },
    },
    {
        "record_id": "holdout_17",
        "text": "记录: 由肺炎引起的咳嗽、发热，血常规异常，阿莫西林抗感染，胸片复查。",
        "entities": {
            "Disease": ["肺炎"],
            "Symptom": ["咳嗽", "发热"],
            "Drug": ["阿莫西林"],
            "Examination": ["血常规", "胸片"],
            "Treatment": ["抗感染"],
        },
    },
    {
        "record_id": "holdout_18",
        "text": "记录: 慢性胃炎与反流性食管炎并存，上腹痛、反酸，奥美拉唑抑酸，胃镜随访。",
        "entities": {
            "Disease": ["胃炎"],
            "Symptom": ["上腹痛", "反酸"],
            "Drug": ["奥美拉唑"],
            "Examination": ["胃镜"],
            "Treatment": [],
        },
    },
    {
        "record_id": "holdout_19",
        "text": "记录: 心力衰竭被评估为冠心病的并发症，活动后气短，心电图监测，辛伐他汀调脂。",
        "entities": {
            "Disease": ["心力衰竭", "冠心病"],
            "Symptom": ["气短"],
            "Drug": ["辛伐他汀"],
            "Examination": ["心电图"],
            "Treatment": ["调节血脂"],
        },
    },
    {
        "record_id": "holdout_20",
        "text": "记录: 支气管哮喘急性发作夜间加重，喘息、呼吸困难，肺功能检查，对症治疗。",
        "entities": {
            "Disease": ["支气管哮喘急性发作"],
            "Symptom": ["喘息", "呼吸困难"],
            "Drug": [],
            "Examination": ["肺功能"],
            "Treatment": ["对症治疗"],
        },
    },
    {
        "record_id": "holdout_21",
        "text": "Record: Male 62y, hypertension and type 2 diabetes, dizziness, on amlodipine and metformin, ECG ordered.",
        "entities": {
            "Disease": ["高血压", "2型糖尿病"],
            "Symptom": ["头晕"],
            "Drug": ["氨氯地平", "二甲双胍"],
            "Examination": ["心电图"],
            "Treatment": [],
        },
    },
    {
        "record_id": "holdout_22",
        "text": "Note: CHD patient with chest tightness; aspirin and statin; coronary CTA planned.",
        "entities": {
            "Disease": ["冠心病"],
            "Symptom": ["胸闷"],
            "Drug": ["阿司匹林", "辛伐他汀"],
            "Examination": ["CTA"],
            "Treatment": [],
        },
    },
    {
        "record_id": "holdout_23",
        "text": "Summary: hyperthyroidism, weight loss and tremor, methimazole, thyroid ultrasound follow-up.",
        "entities": {
            "Disease": ["甲亢"],
            "Symptom": ["消瘦", "手抖"],
            "Drug": ["甲巯咪唑"],
            "Examination": ["甲状腺超声"],
            "Treatment": [],
        },
    },
    {
        "record_id": "holdout_24",
        "text": "Mixed 混合病历: pneumonia 肺炎 with fever 发热, CBC 血常规, amoxicillin 阿莫西林.",
        "entities": {
            "Disease": ["肺炎"],
            "Symptom": ["发热"],
            "Drug": ["阿莫西林"],
            "Examination": ["血常规"],
            "Treatment": [],
        },
    },
    {
        "record_id": "holdout_25",
        "text": "Discharge: asthma exacerbation, wheezing and dyspnea, pulmonary function test, symptomatic care.",
        "entities": {
            "Disease": ["支气管哮喘急性发作"],
            "Symptom": ["喘息", "呼吸困难"],
            "Drug": [],
            "Examination": ["肺功能"],
            "Treatment": ["对症治疗"],
        },
    },
    {
        "record_id": "holdout_26",
        "text": "记录: 脑卒中后遗留头晕，阿司匹林二级预防，MRI评估病灶。",
        "entities": {
            "Disease": ["脑卒中"],
            "Symptom": ["头晕"],
            "Drug": ["阿司匹林"],
            "Examination": ["MRI"],
            "Treatment": [],
        },
    },
    {
        "record_id": "holdout_27",
        "text": "记录: 多疾病并存——高血压合并2型糖尿病，口渴、头痛，氨氯地平与二甲双胍联用，血糖监测。",
        "entities": {
            "Disease": ["高血压", "2型糖尿病"],
            "Symptom": ["口渴", "头痛"],
            "Drug": ["氨氯地平", "二甲双胍"],
            "Examination": ["血糖"],
            "Treatment": [],
        },
    },
    {
        "record_id": "holdout_28",
        "text": "记录: 若无明显感染证据则不必加用抗生素；当前肺炎证据充分，阿莫西林抗感染。",
        "entities": {
            "Disease": ["肺炎"],
            "Symptom": [],
            "Drug": ["阿莫西林"],
            "Examination": [],
            "Treatment": ["抗感染"],
        },
    },
    {
        "record_id": "holdout_29",
        "text": "记录: 甲亢患者碘131治疗后被随访，心悸减轻，复查甲状腺超声及血常规。",
        "entities": {
            "Disease": ["甲亢"],
            "Symptom": ["心悸"],
            "Drug": [],
            "Examination": ["甲状腺超声", "血常规"],
            "Treatment": ["碘131治疗"],
        },
    },
    {
        "record_id": "holdout_30",
        "text": "记录: 系统性红斑狼疮并发肾炎可能，发热、乏力，泼尼松与免疫抑制并用，尿常规异常。",
        "entities": {
            "Disease": ["系统性红斑狼疮"],
            "Symptom": ["发热", "乏力"],
            "Drug": ["泼尼松"],
            "Examination": ["尿常规"],
            "Treatment": ["免疫抑制"],
        },
    },
]

RELATION_RECORDS = [
    {
        "record_id": "holdout_11",
        "text": EXTRACTION_RECORDS[0]["text"],
        "relations": [
            {"subject": "甲亢", "predicate": "has_symptom", "object": "消瘦"},
            {"subject": "甲亢", "predicate": "has_symptom", "object": "手抖"},
            {"subject": "甲亢", "predicate": "has_symptom", "object": "心悸"},
            {"subject": "甲亢", "predicate": "treated_by", "object": "甲巯咪唑"},
            {"subject": "甲亢", "predicate": "diagnosed_by", "object": "甲状腺超声"},
        ],
    },
    {
        "record_id": "holdout_12",
        "text": EXTRACTION_RECORDS[1]["text"],
        "relations": [
            {"subject": "甲状腺功能减退", "predicate": "has_symptom", "object": "怕冷"},
            {"subject": "甲状腺功能减退", "predicate": "has_symptom", "object": "乏力"},
            {"subject": "甲状腺功能减退", "predicate": "treated_by", "object": "左甲状腺素"},
            {"subject": "甲状腺功能减退", "predicate": "diagnosed_by", "object": "血常规"},
        ],
    },
    {
        "record_id": "holdout_13",
        "text": EXTRACTION_RECORDS[2]["text"],
        "relations": [
            {"subject": "冠心病", "predicate": "has_symptom", "object": "胸闷"},
            {"subject": "冠心病", "predicate": "has_symptom", "object": "气短"},
            {"subject": "冠心病", "predicate": "treated_by", "object": "阿司匹林"},
            {"subject": "冠心病", "predicate": "diagnosed_by", "object": "冠脉造影"},
            {"subject": "冠心病", "predicate": "diagnosed_by", "object": "CTA"},
        ],
    },
    {
        "record_id": "holdout_14",
        "text": EXTRACTION_RECORDS[3]["text"],
        "relations": [
            {"subject": "系统性红斑狼疮", "predicate": "has_symptom", "object": "关节痛"},
            {"subject": "系统性红斑狼疮", "predicate": "has_symptom", "object": "发热"},
            {"subject": "系统性红斑狼疮", "predicate": "treated_by", "object": "泼尼松"},
            {"subject": "系统性红斑狼疮", "predicate": "diagnosed_by", "object": "抗核抗体"},
            {"subject": "系统性红斑狼疮", "predicate": "recommended_treatment", "object": "免疫抑制"},
        ],
    },
    {
        "record_id": "holdout_15",
        "text": EXTRACTION_RECORDS[4]["text"],
        "relations": [
            {"subject": "2型糖尿病", "predicate": "has_symptom", "object": "口渴"},
            {"subject": "2型糖尿病", "predicate": "has_symptom", "object": "多尿"},
            {"subject": "2型糖尿病", "predicate": "treated_by", "object": "二甲双胍"},
        ],
    },
    {
        "record_id": "holdout_16",
        "text": EXTRACTION_RECORDS[5]["text"],
        "relations": [
            {"subject": "高血压", "predicate": "has_symptom", "object": "头晕"},
            {"subject": "高血压", "predicate": "has_symptom", "object": "头痛"},
            {"subject": "高血压", "predicate": "treated_by", "object": "氨氯地平"},
            {"subject": "高血压", "predicate": "recommended_treatment", "object": "低盐饮食"},
        ],
    },
    {
        "record_id": "holdout_17",
        "text": EXTRACTION_RECORDS[6]["text"],
        "relations": [
            {"subject": "肺炎", "predicate": "has_symptom", "object": "咳嗽"},
            {"subject": "肺炎", "predicate": "has_symptom", "object": "发热"},
            {"subject": "肺炎", "predicate": "treated_by", "object": "阿莫西林"},
            {"subject": "肺炎", "predicate": "diagnosed_by", "object": "血常规"},
            {"subject": "肺炎", "predicate": "diagnosed_by", "object": "胸片"},
            {"subject": "肺炎", "predicate": "recommended_treatment", "object": "抗感染"},
        ],
    },
    {
        "record_id": "holdout_18",
        "text": EXTRACTION_RECORDS[7]["text"],
        "relations": [
            {"subject": "胃炎", "predicate": "has_symptom", "object": "上腹痛"},
            {"subject": "胃炎", "predicate": "has_symptom", "object": "反酸"},
            {"subject": "胃炎", "predicate": "treated_by", "object": "奥美拉唑"},
            {"subject": "胃炎", "predicate": "diagnosed_by", "object": "胃镜"},
        ],
    },
    {
        "record_id": "holdout_19",
        "text": EXTRACTION_RECORDS[8]["text"],
        "relations": [
            {"subject": "心力衰竭", "predicate": "has_symptom", "object": "气短"},
            {"subject": "冠心病", "predicate": "has_symptom", "object": "气短"},
            {"subject": "心力衰竭", "predicate": "treated_by", "object": "辛伐他汀"},
            {"subject": "心力衰竭", "predicate": "diagnosed_by", "object": "心电图"},
            {"subject": "心力衰竭", "predicate": "recommended_treatment", "object": "调节血脂"},
            {"subject": "心力衰竭", "predicate": "complication_of", "object": "冠心病"},
        ],
    },
    {
        "record_id": "holdout_20",
        "text": EXTRACTION_RECORDS[9]["text"],
        "relations": [
            {"subject": "支气管哮喘急性发作", "predicate": "has_symptom", "object": "喘息"},
            {"subject": "支气管哮喘急性发作", "predicate": "has_symptom", "object": "呼吸困难"},
            {"subject": "支气管哮喘急性发作", "predicate": "diagnosed_by", "object": "肺功能"},
            {"subject": "支气管哮喘急性发作", "predicate": "recommended_treatment", "object": "对症治疗"},
        ],
    },
    {
        "record_id": "holdout_21",
        "text": EXTRACTION_RECORDS[10]["text"],
        "relations": [
            {"subject": "高血压", "predicate": "has_symptom", "object": "头晕"},
            {"subject": "2型糖尿病", "predicate": "has_symptom", "object": "头晕"},
            {"subject": "高血压", "predicate": "treated_by", "object": "氨氯地平"},
            {"subject": "2型糖尿病", "predicate": "treated_by", "object": "二甲双胍"},
            {"subject": "高血压", "predicate": "diagnosed_by", "object": "心电图"},
            {"subject": "2型糖尿病", "predicate": "diagnosed_by", "object": "心电图"},
        ],
    },
    {
        "record_id": "holdout_22",
        "text": EXTRACTION_RECORDS[11]["text"],
        "relations": [
            {"subject": "冠心病", "predicate": "has_symptom", "object": "胸闷"},
            {"subject": "冠心病", "predicate": "treated_by", "object": "阿司匹林"},
            {"subject": "冠心病", "predicate": "treated_by", "object": "辛伐他汀"},
            {"subject": "冠心病", "predicate": "diagnosed_by", "object": "CTA"},
        ],
    },
    {
        "record_id": "holdout_23",
        "text": EXTRACTION_RECORDS[12]["text"],
        "relations": [
            {"subject": "甲亢", "predicate": "has_symptom", "object": "消瘦"},
            {"subject": "甲亢", "predicate": "has_symptom", "object": "手抖"},
            {"subject": "甲亢", "predicate": "treated_by", "object": "甲巯咪唑"},
            {"subject": "甲亢", "predicate": "diagnosed_by", "object": "甲状腺超声"},
        ],
    },
    {
        "record_id": "holdout_24",
        "text": EXTRACTION_RECORDS[13]["text"],
        "relations": [
            {"subject": "肺炎", "predicate": "has_symptom", "object": "发热"},
            {"subject": "肺炎", "predicate": "treated_by", "object": "阿莫西林"},
            {"subject": "肺炎", "predicate": "diagnosed_by", "object": "血常规"},
        ],
    },
    {
        "record_id": "holdout_25",
        "text": EXTRACTION_RECORDS[14]["text"],
        "relations": [
            {"subject": "支气管哮喘急性发作", "predicate": "has_symptom", "object": "喘息"},
            {"subject": "支气管哮喘急性发作", "predicate": "has_symptom", "object": "呼吸困难"},
            {"subject": "支气管哮喘急性发作", "predicate": "diagnosed_by", "object": "肺功能"},
            {"subject": "支气管哮喘急性发作", "predicate": "recommended_treatment", "object": "对症治疗"},
        ],
    },
    {
        "record_id": "holdout_26",
        "text": EXTRACTION_RECORDS[15]["text"],
        "relations": [
            {"subject": "脑卒中", "predicate": "has_symptom", "object": "头晕"},
            {"subject": "脑卒中", "predicate": "treated_by", "object": "阿司匹林"},
            {"subject": "脑卒中", "predicate": "diagnosed_by", "object": "MRI"},
        ],
    },
    {
        "record_id": "holdout_27",
        "text": EXTRACTION_RECORDS[16]["text"],
        "relations": [
            {"subject": "高血压", "predicate": "has_symptom", "object": "头痛"},
            {"subject": "2型糖尿病", "predicate": "has_symptom", "object": "口渴"},
            {"subject": "高血压", "predicate": "treated_by", "object": "氨氯地平"},
            {"subject": "2型糖尿病", "predicate": "treated_by", "object": "二甲双胍"},
            {"subject": "2型糖尿病", "predicate": "diagnosed_by", "object": "血糖"},
        ],
    },
    {
        "record_id": "holdout_28",
        "text": EXTRACTION_RECORDS[17]["text"],
        "relations": [
            {"subject": "肺炎", "predicate": "treated_by", "object": "阿莫西林"},
            {"subject": "肺炎", "predicate": "recommended_treatment", "object": "抗感染"},
        ],
    },
    {
        "record_id": "holdout_29",
        "text": EXTRACTION_RECORDS[18]["text"],
        "relations": [
            {"subject": "甲亢", "predicate": "has_symptom", "object": "心悸"},
            {"subject": "甲亢", "predicate": "diagnosed_by", "object": "甲状腺超声"},
            {"subject": "甲亢", "predicate": "diagnosed_by", "object": "血常规"},
            {"subject": "甲亢", "predicate": "recommended_treatment", "object": "碘131治疗"},
        ],
    },
    {
        "record_id": "holdout_30",
        "text": EXTRACTION_RECORDS[19]["text"],
        "relations": [
            {"subject": "系统性红斑狼疮", "predicate": "has_symptom", "object": "发热"},
            {"subject": "系统性红斑狼疮", "predicate": "has_symptom", "object": "乏力"},
            {"subject": "系统性红斑狼疮", "predicate": "treated_by", "object": "泼尼松"},
            {"subject": "系统性红斑狼疮", "predicate": "diagnosed_by", "object": "尿常规"},
            {"subject": "系统性红斑狼疮", "predicate": "recommended_treatment", "object": "免疫抑制"},
        ],
    },
]


def main() -> None:
    extraction_path = ROOT / "benchmarks" / "data" / "kg_extraction_gold.json"
    relation_path = ROOT / "benchmarks" / "data" / "kg_relation_gold.json"

    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    relation = json.loads(relation_path.read_text(encoding="utf-8"))

    extraction["records"].extend(EXTRACTION_RECORDS)
    relation["records"].extend(RELATION_RECORDS)

    extraction_path.write_text(
        json.dumps(extraction, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    relation_path.write_text(
        json.dumps(relation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Expanded extraction gold to {len(extraction['records'])} records")
    print(f"Expanded relation gold to {len(relation['records'])} records")


if __name__ == "__main__":
    main()
