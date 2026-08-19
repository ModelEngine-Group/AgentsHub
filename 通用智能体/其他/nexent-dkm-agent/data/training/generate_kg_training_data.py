"""Generate synthetic training data for task-2 KG entity extraction fine-tuning.

Produces JSONL files suitable for QLoRA fine-tuning of small language models
on the medical entity extraction task.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.operators.kg_ops.kg_prompts import KG_INSTRUCTION

random.seed(42)

_DISEASES = [
    "高血压", "2型糖尿病", "支气管哮喘急性发作", "冠心病", "心力衰竭",
    "胃炎", "肺炎", "脑卒中", "慢性阻塞性肺疾病", "肝硬化",
    "甲状腺功能亢进", "类风湿关节炎", "系统性红斑狼疮", "骨质疏松",
    "慢性肾病", "贫血", "消化性溃疡", "急性阑尾炎", "支气管炎",
    "心房颤动", "帕金森病", "阿尔茨海默病", "抑郁症", "焦虑症",
]

_SYMPTOMS = [
    "头晕", "头痛", "口渴", "多尿", "喘息", "呼吸困难", "胸闷", "气短",
    "上腹痛", "恶心", "呕吐", "发热", "咳嗽", "咳痰", "心悸", "乏力",
    "食欲减退", "体重下降", "失眠", "水肿", "腹痛", "腹泻", "便秘",
    "关节疼痛", "肌肉酸痛", "皮疹", "视力模糊", "耳鸣", "腰痛",
]

_DRUGS = [
    "氨氯地平", "阿司匹林", "二甲双胍", "辛伐他汀", "布洛芬",
    "奥美拉唑", "阿莫西林", "头孢克洛", "氯雷他定", "甲硝唑",
    "华法林", "地高辛", "呋塞米", "螺内酯", "硝苯地平",
    "卡托普利", "缬沙坦", "阿托伐他汀", "氯吡格雷", "美托洛尔",
]

_EXAMINATIONS = [
    "血常规", "肝功能", "尿常规", "血糖", "肺功能", "心电图", "CT",
    "MRI", "超声", "胸部X光", "腹部B超", "血气分析", "甲状腺功能",
    "凝血功能", "C反应蛋白", "血沉", "血脂", "肾功能", "电解质",
]

_TREATMENTS = [
    "调节血脂", "对症治疗", "抗感染", "继续服用", "手术治疗",
    "物理治疗", "心理疏导", "放射治疗", "化疗", "免疫治疗",
    "中药调理", "康复训练", "饮食控制", "运动疗法", "戒烟戒酒",
]

_NAMES = [
    "张三", "李四", "王五", "赵六", "钱七", "孙八", "周九", "吴十",
    "郑一", "冯二", "陈明", "林华", "黄芳", "刘强", "杨丽",
]

_RECORD_TEMPLATES = [
    (
        "患者{name}，{gender}，{age}岁。主诉：{symptoms} {duration}。"
        "既往有{disease}病史{history_years}年，长期服用{drug}。"
        "查体：{findings}。初步诊断：{disease}。"
        "建议：{exams}检查，{treatment}。"
    ),
    (
        "患者{name}，{gender}，{age}岁。因{symptoms}入院。"
        "既往史：{disease}，服用{drug}。"
        "实验室检查：{exam_findings}。诊断：{disease}。"
        "治疗方案：{treatment}，{drug}。"
    ),
    (
        "{name}，{gender}，{age}岁。{symptoms} {duration}。"
        "有{disease}病史。体检发现{findings}。"
        "诊断：{disease}。医嘱：{exams}，{drug}，{treatment}。"
    ),
    (
        "姓名：{name}  性别：{gender}  年龄：{age}\n"
        "主诉：{symptoms}\n"
        "现病史：患者{history_years}年前确诊{disease}，规律服用{drug}。\n"
        "辅助检查：{exams}\n"
        "诊断：{disease}\n"
        "处理：{treatment}"
    ),
    (
        "门诊记录：患者{name}，{gender}，{age}岁。\n"
        "主诉：{symptoms} {duration}。\n"
        "既往{disease}史，服药{drug}。\n"
        "体格检查：{findings}。\n"
        "诊断：{disease}。\n"
        "建议完善{exams}，予{drug}及{treatment}。"
    ),
]

_ENTITY_LABELS = {
    "Disease": _DISEASES,
    "Symptom": _SYMPTOMS,
    "Drug": _DRUGS,
    "Examination": _EXAMINATIONS,
    "Treatment": _TREATMENTS,
}


def _choice(seq):
    return random.choice(seq)


def _choices(seq, n):
    return random.sample(seq, min(n, len(seq)))


def _generate_record():
    template = _choice(_RECORD_TEMPLATES)
    name = _choice(_NAMES)
    gender = _choice(["男", "女"])
    age = random.randint(25, 80)
    disease = _choice(_DISEASES)
    symptoms = "、".join(_choices(_SYMPTOMS, random.randint(2, 4)))
    drug = _choice(_DRUGS)
    drugs = [drug] + _choices(_DRUGS, random.randint(0, 2))
    drug_str = "、".join(drugs)
    exams = "、".join(_choices(_EXAMINATIONS, random.randint(1, 3)))
    treatment = _choice(_TREATMENTS)
    duration = _choice(["1天", "3天", "1周", "2周", "1个月", "3天", "5天"])
    history_years = str(random.randint(1, 15))
    findings = _choice([
        f"血压{random.randint(130, 180)}/{random.randint(80, 110)}mmHg",
        f"体温{random.uniform(36.5, 39.0):.1f}°C",
        f"心率{random.randint(60, 120)}次/分",
        "双肺呼吸音粗", "腹部压痛阳性", "双下肢水肿",
    ])
    exam_findings = _choice([
        f"空腹血糖{random.uniform(5.0, 15.0):.1f}mmol/L",
        f"血红蛋白{random.randint(80, 160)}g/L",
        f"血肌酐{random.randint(50, 200)}μmol/L",
        "心电图提示ST-T改变",
        "胸部CT未见明显异常",
    ])

    text = template.format(
        name=name, gender=gender, age=age, disease=disease,
        symptoms=symptoms, drug=drug_str, exams=exams, treatment=treatment,
        duration=duration, history_years=history_years, findings=findings,
        exam_findings=exam_findings,
    )

    # Build entity annotations
    entities = {}
    for etype, terms in _ENTITY_LABELS.items():
        found = [t for t in terms if t in text]
        if found:
            entities[etype] = found
        else:
            entities[etype] = []

    # Build relations
    relations = []
    diseases_in_text = entities.get("Disease", [])
    for d in diseases_in_text:
        for s in entities.get("Symptom", []):
            relations.append({"subject": d, "predicate": "has_symptom", "object": s})
        for dr in entities.get("Drug", []):
            relations.append({"subject": d, "predicate": "treated_by", "object": dr})
        for ex in entities.get("Examination", []):
            relations.append({"subject": d, "predicate": "diagnosed_by", "object": ex})
        for tr in entities.get("Treatment", []):
            relations.append({"subject": d, "predicate": "recommended_treatment", "object": tr})

    return text, entities, relations


def generate_samples(n: int = 2000):
    samples = []
    for _ in range(n):
        text, entities, relations = _generate_record()
        output = json.dumps({"entities": entities, "relations": relations}, ensure_ascii=False)
        samples.append({
            "instruction": KG_INSTRUCTION,
            "input": text,
            "output": output,
        })
    return samples


def main():
    output_dir = Path(__file__).resolve().parent
    samples = generate_samples(2000)

    random.shuffle(samples)
    split = int(len(samples) * 0.9)
    train = samples[:split]
    val = samples[split:]

    train_path = output_dir / "kg_extraction_train.jsonl"
    val_path = output_dir / "kg_extraction_val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for s in train:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for s in val:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Generated {len(train)} train + {len(val)} val samples.")
    print(f"Train: {train_path}")
    print(f"Val: {val_path}")


if __name__ == "__main__":
    main()
