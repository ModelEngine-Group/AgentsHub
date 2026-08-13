"""合成中文电子病历生成器（确定性种子）。

产出四类相互对齐的数据：
1. data/raw/emr_NNNN.txt      —— 含噪声的病历文本（任务一清洗对象）
2. data/raw/structured/*.csv  —— patients / visits 结构化表（任务三分析对象）
3. data/gold/entities|triples —— 金标准标注（任务二抽取质量评测）
4. data/gold/corpus_meta.json —— 语料元信息

关键设计：
- 金标准在"生成时"同步记录（不是事后抽取），保证与文本严格一致；
- 实体提及 30% 概率用别名替换（制造归一化难度），gold 一律记规范名；
- 关系表述模板每类 ≥4 种，其中部分模板不含触发词（只有共现），
  用于评测抽取器的召回分层；
- 噪声注入发生在"金标准记录之后"（HTML/全角/空格/PII/重复句/繁体），
  正是任务一清洗算子要去除的内容。
"""

import argparse
import csv
import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dki.paths import GOLD_DATA_DIR, RAW_DATA_DIR

DICT_PATH = Path(__file__).resolve().parent.parent / "kg" / "dictionaries" / "medical_dict.json"

SURNAMES = "王李张刘陈杨黄赵吴周徐孙马朱胡郭何林罗高"
GIVEN = ["伟", "芳", "娜", "敏", "静", "磊", "军", "洋", "勇", "艳", "杰", "涛",
         "明", "超", "秀英", "建华", "文", "红", "志强", "玉兰"]
REGIONS = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安", "南京", "长沙"]

# 繁体噪声映射（少量常用字）
TRAD_MAP = {"头": "頭", "疼": "疼", "痛": "痛", "医": "醫", "药": "藥", "检": "檢",
            "查": "查", "诊": "診", "断": "斷", "护": "護", "病": "病", "术": "術"}

# ---------------------------------------------------------------- 关系表述模板
# {d}=疾病 {s}=症状 {g}=药物 {e}=检查 {t}=治疗 {c}=并发症
SYMPTOM_TPLS = [  # HAS_SYMPTOM（前3种含触发词，后2种弱触发/共现）
    "患者主诉{s}，伴{s2}。",
    "患者出现{s}，表现为{s2}。",
    "自觉{s}，并伴有{s2}。",
    "近期反复{s}，同时有{s2}症状。",
    "患者{s}、{s2}明显。",
]
DRUG_TPLS = [  # TREATED_BY_DRUG
    "给予{g}口服治疗。",
    "予{g}规律服用。",
    "处方{g}，嘱按时服药。",
    "服用{g}后症状好转。",
    "继续{g}维持治疗。",
]
EXAM_TPLS = [  # DIAGNOSED_BY
    "行{e}检查，结果提示异常。",
    "完善{e}，明确诊断。",
    "查{e}示有异常改变。",
    "{e}结果支持上述诊断。",
]
TREAT_TPLS = [  # TREATED_BY
    "行{t}，术程顺利。",
    "建议行{t}。",
    "已接受{t}，恢复良好。",
    "择期行{t}。",
]
COMP_TPLS = [  # HAS_COMPLICATION
    "病程中并发{c}。",
    "继发{c}，予以对症处理。",
    "警惕{c}等并发症的发生。",
    "合并{c}，转专科随诊。",
]
CONTRA_TPLS = [  # CONTRAINDICATED_FOR（drug 对 disease2 禁忌）
    "因患者既往有{c}病史，{g}禁用。",
    "考虑到{c}，避免使用{g}。",
    "{g}对{c}患者慎用，已停用。",
]
DEPT_TPLS = [  # BELONGS_TO_DEPT
    "收入{dept}住院治疗。",
    "于{dept}门诊就诊。",
    "转入{dept}进一步诊治。",
    "至{dept}随诊复查。",
]


def _make_id_number(rng: random.Random) -> str:
    """生成校验位合法的 18 位身份证号（GB11643），确保能被严格校验的脱敏算子识别。"""
    area = rng.choice(["110101", "310101", "440103", "330102", "510104"])
    birth = date(rng.randint(1950, 2005), rng.randint(1, 12), rng.randint(1, 28))
    seq = f"{rng.randint(0, 999):03d}"
    body = area + birth.strftime("%Y%m%d") + seq
    weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    check_map = "10X98765432"
    s = sum(int(c) * w for c, w in zip(body, weights))
    return body + check_map[s % 11]


def load_dict() -> Dict[str, Any]:
    return json.loads(DICT_PATH.read_text(encoding="utf-8"))


class EmrGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.med = load_dict()
        self.diseases = self.med["diseases"]

    # ------------------------------------------------------ 提及（别名替换）
    def _mention(self, canonical: str, section: Dict[str, Any]) -> str:
        aliases = section.get(canonical, {}).get("aliases", [])
        if aliases and self.rng.random() < 0.3:
            return self.rng.choice(aliases)
        return canonical

    # ------------------------------------------------------ 单份病历
    def generate_one(self, doc_id: str, patient: Dict[str, Any]) -> Tuple[str, List, List, Dict]:
        """返回 (纯净文本, gold_entities, gold_triples, visit_row)。"""
        rng = self.rng
        med = self.med
        main_d = rng.choice(list(self.diseases))
        info = self.diseases[main_d]

        entities: List[Dict[str, Any]] = []
        triples: List[Dict[str, Any]] = []

        def ent(name, etype, mention):
            entities.append({"name": name, "type": etype, "mention": mention, "span": None})

        def tri(h, ht, r, t, tt):
            row = {"head": h, "head_type": ht, "relation": r, "tail": t, "tail_type": tt}
            if row not in triples:
                triples.append(row)

        d_mention = self._mention(main_d, self.diseases)
        symptoms = rng.sample(info["symptoms"], k=min(len(info["symptoms"]), rng.randint(2, 4)))
        drugs = rng.sample(info["drugs"], k=min(len(info["drugs"]), rng.randint(1, 3))) if info["drugs"] else []
        exams = rng.sample(info["exams"], k=min(len(info["exams"]), rng.randint(1, 3))) if info["exams"] else []
        treatments = rng.sample(info["treatments"], k=1) if info["treatments"] and rng.random() < 0.6 else []
        comps = rng.sample(info["complications"], k=1) if info["complications"] and rng.random() < 0.5 else []

        # ---- 章节文本
        lines: List[str] = []
        lines.append(f"基本信息：{patient['name']}，{patient['gender']}，{patient['age']}岁，来自{patient['region']}。")

        # 主诉：1-2 个症状
        s0 = self._mention(symptoms[0], med["symptoms"])
        ent(symptoms[0], "Symptom", s0)
        tri(main_d, "Disease", "HAS_SYMPTOM", symptoms[0], "Symptom")
        dur = rng.choice(["3天", "1周", "2周", "1个月", "3个月", "半年"])
        lines.append(f"主诉：{s0}{dur}。")

        # 现病史：症状模板（含触发词/共现分层）
        hx = [f"患者{dur}前无明显诱因出现{s0}。"]
        if len(symptoms) >= 2:
            s1 = self._mention(symptoms[1], med["symptoms"])
            ent(symptoms[1], "Symptom", s1)
            tri(main_d, "Disease", "HAS_SYMPTOM", symptoms[1], "Symptom")
            extra = ""
            if len(symptoms) >= 3:
                s2m = self._mention(symptoms[2], med["symptoms"])
                ent(symptoms[2], "Symptom", s2m)
                tri(main_d, "Disease", "HAS_SYMPTOM", symptoms[2], "Symptom")
                extra = rng.choice(SYMPTOM_TPLS).format(s=s1, s2=s2m)
            else:
                extra = f"伴有{s1}。"
            hx.append(extra)
        if len(symptoms) >= 4:
            s3m = self._mention(symptoms[3], med["symptoms"])
            ent(symptoms[3], "Symptom", s3m)
            tri(main_d, "Disease", "HAS_SYMPTOM", symptoms[3], "Symptom")
            hx.append(f"病程中出现{s3m}。")
        lines.append("现病史：" + "".join(hx))

        # 既往史：0-2 个其他疾病（只作实体，不给关系；10% 触发禁忌叙述）
        past: List[str] = []
        other_pool = [d for d in self.diseases if d != main_d]
        past_diseases = rng.sample(other_pool, k=rng.randint(0, 2))
        for pd in past_diseases:
            pm = self._mention(pd, self.diseases)
            ent(pd, "Disease", pm)
            past.append(f"{pm}病史{rng.choice(['2年', '5年', '10年', '多年'])}")
        lines.append("既往史：" + ("，".join(past) + "。" if past else "无特殊。"))

        # 辅助检查
        exam_lines = []
        for e in exams:
            em = self._mention(e, med["exams"])
            ent(e, "Exam", em)
            tri(main_d, "Disease", "DIAGNOSED_BY", e, "Exam")
            exam_lines.append(rng.choice(EXAM_TPLS).format(e=em))
        if exam_lines:
            lines.append("辅助检查：" + "".join(exam_lines))

        # 诊断
        ent(main_d, "Disease", d_mention)
        dept = info["department"]
        ent(dept, "Department", dept)
        tri(main_d, "Disease", "BELONGS_TO_DEPT", dept, "Department")
        diag_line = f"诊断：{d_mention}。"
        if comps:
            c = comps[0]
            cm = self._mention(c, self.diseases)
            ent(c, "Disease", cm)
            tri(main_d, "Disease", "HAS_COMPLICATION", c, "Disease")
            diag_line += rng.choice(COMP_TPLS).format(c=cm)
        lines.append(diag_line)

        # 治疗方案：药物 + 治疗方式 + 禁忌
        treat_lines = []
        for g in drugs:
            gm = self._mention(g, med["drugs"])
            ent(g, "Drug", gm)
            tri(main_d, "Disease", "TREATED_BY_DRUG", g, "Drug")
            treat_lines.append(rng.choice(DRUG_TPLS).format(g=gm))
        for t in treatments:
            tm = self._mention(t, med["treatments"])
            ent(t, "Treatment", tm)
            tri(main_d, "Disease", "TREATED_BY", t, "Treatment")
            treat_lines.append(rng.choice(TREAT_TPLS).format(t=tm))
        # 禁忌叙述：从词典找一个对"既往疾病"禁忌的药物
        for pd in past_diseases:
            contra_drugs = [
                dn for dn, dd in med["drugs"].items() if pd in dd.get("contraindications", [])
            ]
            if contra_drugs and rng.random() < 0.7:
                cg = rng.choice(contra_drugs)
                cgm = self._mention(cg, med["drugs"])
                ent(cg, "Drug", cgm)
                ent(pd, "Disease", self._mention(pd, self.diseases))
                tri(cg, "Drug", "CONTRAINDICATED_FOR", pd, "Disease")
                treat_lines.append(rng.choice(CONTRA_TPLS).format(g=cgm, c=self._mention(pd, self.diseases)))
                break
        if treat_lines:
            lines.append("治疗方案：" + "".join(treat_lines))

        # 医嘱
        lines.append("医嘱：" + rng.choice(DEPT_TPLS).format(dept=dept)
                     + rng.choice(["低盐低脂饮食，规律作息。", "注意休息，避免劳累。",
                                   "定期复查，不适随诊。", "戒烟限酒，适量运动。"]))

        clean_text = "\n".join(lines)

        # ---- 结构化 visit 行
        visit = {
            "visit_id": doc_id,
            "patient_id": patient["patient_id"],
            "visit_date": patient["visit_date"],
            "department": dept,
            "disease": main_d,
            "drug_prescribed": drugs[0] if drugs else "",
            "exam": exams[0] if exams else "",
            "cost": self._cost(dept, bool(treatments)),
            "los_days": rng.choice([0, 0, 0, rng.randint(3, 15)]),
        }
        return clean_text, entities, triples, visit

    def _cost(self, dept: str, has_surgery: bool) -> float:
        base = {
            "心血管内科": 1200, "内分泌科": 800, "呼吸内科": 900, "消化内科": 1000,
            "神经内科": 1100, "肾内科": 1300, "泌尿外科": 1500, "血液科": 1400,
            "感染科": 950, "骨科": 1600, "风湿免疫科": 850, "精神心理科": 600,
            "眼科": 700, "普通外科": 1800,
        }.get(dept, 1000)
        v = base * self.rng.uniform(0.6, 2.2)
        if has_surgery:
            v += self.rng.uniform(8000, 30000)
        return round(v, 2)

    # ------------------------------------------------------ 噪声注入
    def add_noise(self, text: str) -> str:
        rng = self.rng
        noises = rng.sample(
            ["html", "fullwidth", "spaces", "invisible", "dup", "pii", "trad"],
            k=rng.randint(2, 4),
        )
        lines = text.split("\n")
        if "html" in noises:
            i = rng.randrange(len(lines))
            lines[i] = f"<p>{lines[i]}</p>"
            j = rng.randrange(len(lines))
            lines[j] = lines[j].replace("：", "：&nbsp;", 1) + "<br/>"
        if "dup" in noises:
            i = rng.randrange(1, len(lines))
            lines.insert(i, lines[i])  # 整句重复
        text = "\n".join(lines)
        if "fullwidth" in noises:
            trans = str.maketrans("0123456789:", "０１２３４５６７８９：")
            # 只对部分数字全角化
            parts = list(text)
            for k, ch in enumerate(parts):
                if ch.isdigit() and rng.random() < 0.5:
                    parts[k] = ch.translate(trans)
            text = "".join(parts)
        if "spaces" in noises:
            text = text.replace("，", "， ", 3).replace("。", "。  ", 2)
        if "invisible" in noises:
            pos = rng.randrange(len(text))
            text = text[:pos] + "​" + text[pos:]
            text = text.replace("：", ":​", 1) if rng.random() < 0.3 else text
        if "pii" in noises:
            phone = "1" + rng.choice("3589") + "".join(rng.choice("0123456789") for _ in range(9))
            idno = _make_id_number(rng)
            text += f"\n联系电话：{phone}。身份证号：{idno}。"
        if "trad" in noises and rng.random() < 0.6:
            for k, v in list(TRAD_MAP.items())[:5]:
                if rng.random() < 0.4:
                    text = text.replace(k, v, 1)
        return text

    # ------------------------------------------------------ 患者
    def gen_patient(self, idx: int) -> Dict[str, Any]:
        rng = self.rng
        start = date(2024, 7, 1)
        d = start + timedelta(days=rng.randint(0, 729))  # 近24个月（至2026-06-30）
        return {
            "patient_id": f"P{idx:05d}",
            "name": rng.choice(SURNAMES) + rng.choice(GIVEN),
            "gender": rng.choice(["男", "女"]),
            "age": rng.randint(18, 88),
            "region": rng.choice(REGIONS),
            "visit_date": d.isoformat(),
        }


def generate_corpus(
    n_docs: int = 200,
    seed: int = 42,
    out_dir: Path = None,
    gold_dir: Path = None,
) -> Dict[str, Any]:
    out_dir = Path(out_dir or RAW_DATA_DIR)
    gold_dir = Path(gold_dir or GOLD_DATA_DIR)
    (out_dir / "structured").mkdir(parents=True, exist_ok=True)
    (gold_dir / "entities").mkdir(parents=True, exist_ok=True)
    (gold_dir / "triples").mkdir(parents=True, exist_ok=True)

    gen = EmrGenerator(seed=seed)
    patients_rows, visits_rows, meta = [], [], {}
    n_triples = 0
    diseases_used = set()

    # 部分患者复诊（患者数 < 文档数）
    n_patients = max(1, int(n_docs * 0.8))
    patients = [gen.gen_patient(i + 1) for i in range(n_patients)]
    for p in patients:
        patients_rows.append({k: p[k] for k in ("patient_id", "name", "gender", "age", "region")})

    for i in range(1, n_docs + 1):
        doc_id = f"emr_{i:04d}"
        patient = dict(gen.rng.choice(patients))
        # 复诊时换日期
        base = date.fromisoformat(patient["visit_date"])
        patient["visit_date"] = (base + timedelta(days=gen.rng.randint(0, 60))).isoformat()
        if patient["visit_date"] > "2026-06-30":
            patient["visit_date"] = "2026-06-30"

        clean_text, entities, triples, visit = gen.generate_one(doc_id, patient)
        noisy_text = gen.add_noise(clean_text)

        (out_dir / f"{doc_id}.txt").write_text(noisy_text, encoding="utf-8")
        (gold_dir / "entities" / f"{doc_id}.json").write_text(
            json.dumps(entities, ensure_ascii=False, indent=1), encoding="utf-8")
        (gold_dir / "triples" / f"{doc_id}.json").write_text(
            json.dumps(triples, ensure_ascii=False, indent=1), encoding="utf-8")

        visits_rows.append(visit)
        n_triples += len(triples)
        diseases_used.add(visit["disease"])
        meta[doc_id] = {"disease": visit["disease"], "department": visit["department"]}

    with open(out_dir / "structured" / "patients.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "name", "gender", "age", "region"])
        w.writeheader()
        w.writerows(patients_rows)
    with open(out_dir / "structured" / "visits.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["visit_id", "patient_id", "visit_date", "department",
                                          "disease", "drug_prescribed", "exam", "cost", "los_days"])
        w.writeheader()
        w.writerows(visits_rows)
    (gold_dir / "corpus_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    stats = {
        "n_docs": n_docs,
        "n_patients": n_patients,
        "n_gold_triples": n_triples,
        "n_diseases_used": len(diseases_used),
        "out_dir": str(out_dir),
        "gold_dir": str(gold_dir),
    }
    return stats


def main():
    ap = argparse.ArgumentParser(description="生成合成医疗数据集")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    stats = generate_corpus(n_docs=args.n, seed=args.seed)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
