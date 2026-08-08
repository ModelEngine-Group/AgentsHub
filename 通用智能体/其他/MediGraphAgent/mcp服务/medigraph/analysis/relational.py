"""Relational layer for Task 3: build a SQLite analytics DB.

Zero-config (file-based SQLite; the plan's PostgreSQL is a documented production
swap). The clinical *vocabulary and relationships* are derived from the task-2
knowledge graph (disease->department via treated_in_department, disease->drug via
recommend_drug, disease->exam via need_examination) -- this is the concrete
"effective reuse" of Task 2. On top of that vocabulary we generate deterministic
(seeded) synthetic visit records so statistical/trend analysis has data to query.

Tables:
  patient_visits(visit_id, patient_id, age, gender, disease, department, visit_date, cost)
  prescriptions(rx_id, visit_id, drug, days)
  lab_tests(test_id, visit_id, test_name, abnormal)
  kg_entities(name, type)
  kg_triples(head, head_type, relation, tail, tail_type, confidence, source)
"""
from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from medigraph.graph.local_store import LocalGraphStore

SCHEMA = """
CREATE TABLE patient_visits (
    visit_id INTEGER PRIMARY KEY,
    patient_id INTEGER NOT NULL,
    age INTEGER NOT NULL,
    gender TEXT NOT NULL,            -- '男' | '女'
    disease TEXT NOT NULL,           -- diagnosed disease name
    department TEXT NOT NULL,        -- visited department
    visit_date TEXT NOT NULL,        -- 'YYYY-MM-DD'
    cost REAL NOT NULL               -- visit cost in CNY
);
CREATE TABLE prescriptions (
    rx_id INTEGER PRIMARY KEY,
    visit_id INTEGER NOT NULL,
    drug TEXT NOT NULL,
    days INTEGER NOT NULL,
    FOREIGN KEY (visit_id) REFERENCES patient_visits(visit_id)
);
CREATE TABLE lab_tests (
    test_id INTEGER PRIMARY KEY,
    visit_id INTEGER NOT NULL,
    test_name TEXT NOT NULL,
    abnormal INTEGER NOT NULL,       -- 1 if result abnormal else 0
    FOREIGN KEY (visit_id) REFERENCES patient_visits(visit_id)
);
CREATE TABLE kg_entities (name TEXT, type TEXT);
CREATE TABLE kg_triples (
    head TEXT, head_type TEXT, relation TEXT,
    tail TEXT, tail_type TEXT, confidence REAL, source TEXT
);
"""

# Fallback vocabulary if the KG lacks the needed relations.
_FALLBACK = {
    "高血压": {"department": "心内科", "drugs": ["硝苯地平", "氨氯地平"], "exams": ["心电图"]},
    "2型糖尿病": {"department": "内分泌科", "drugs": ["二甲双胍"], "exams": ["糖化血红蛋白"]},
    "急性心肌梗死": {"department": "心内科", "drugs": ["阿司匹林", "氯吡格雷"], "exams": ["冠状动脉造影"]},
    "冠心病": {"department": "心内科", "drugs": ["阿司匹林"], "exams": ["心电图"]},
    "糖尿病肾病": {"department": "肾内科", "drugs": ["缬沙坦"], "exams": ["尿微量白蛋白"]},
}

#: Diseases the rest of the codebase already names by literal string (NL2SQL
#: few-shot pool, the stress-set generator, hand-written gold questions,
#: `_MEDICAL_ALIASES`). Force-included in the curated vocabulary below so those
#: literals are guaranteed to resolve to real, non-empty visit/prescription rows
#: instead of an accidental empty-set match against an equally-empty gold query.
ANCHOR_DISEASES: tuple[str, ...] = (
    "高血压", "冠心病", "糖尿病肾病", "急性心肌梗死", "糖尿病足", "糖尿病酮症酸中毒",
    "小儿肺炎", "急性阑尾炎", "肺炎", "强直性脊柱炎", "骨关节炎", "老年痴呆",
    "肝硬化", "急性胃炎",
)

#: Cap on distinct diseases sampled into the synthetic visit generator.
DEFAULT_MAX_DISEASES = 50


def derive_vocab(
    store: LocalGraphStore,
    max_diseases: int = DEFAULT_MAX_DISEASES,
    anchors: tuple[str, ...] = ANCHOR_DISEASES,
) -> dict:
    """Build {disease: {department, drugs[], exams[]}} from KG triples.

    Why curate rather than use every Disease node
    ----------------------------------------------
    The full CM3KG import carries ~1,200 diseases with a department/drug edge.
    Spreading 600 synthetic visits across all of them leaves most diseases with
    zero or one visit -- any NL2SQL question that filters on a specific disease
    or drug name then has an *empty* result set on both the gold and the
    predicted side, so it "passes" by vacuous agreement rather than by the
    predicted SQL being structurally correct. That silently inflates reported
    accuracy without validating anything.

    Curating to the `max_diseases` richest entries (ranked by how many drugs
    and exams the KG associates with them, a proxy for how well-documented the
    disease is) keeps a synthetic hospital's realistic diagnosis mix instead of
    a uniform draw over rare/obscure conditions, and gives literal-filter
    questions real, countable rows to match against.
    """
    vocab: dict[str, dict] = {}
    for head, tail, data in store.g.edges(data=True):
        htype = store.g.nodes[head].get("type", "")
        if htype not in ("Disease", "Tumor"):
            continue
        rel = data.get("relation")
        slot = vocab.setdefault(head, {"department": None, "drugs": [], "exams": []})
        if rel == "treated_in_department" and not slot["department"]:
            slot["department"] = tail
        elif rel == "recommend_drug":
            slot["drugs"].append(tail)
        elif rel == "need_examination":
            slot["exams"].append(tail)
    # keep only diseases that have at least a department or drug; fill gaps
    cleaned = {}
    for dz, slot in vocab.items():
        if not slot["department"] and not slot["drugs"]:
            continue
        slot["department"] = slot["department"] or "全科"
        slot["drugs"] = slot["drugs"] or ["对症治疗药"]
        slot["exams"] = slot["exams"] or ["常规检查"]
        cleaned[dz] = slot
    if not cleaned:
        return dict(_FALLBACK)
    if max_diseases and len(cleaned) > max_diseases:
        ranked = sorted(
            cleaned.items(),
            key=lambda item: (len(item[1]["drugs"]) * 2 + len(item[1]["exams"]), item[0]),
            reverse=True,
        )
        selected = dict(ranked[:max_diseases])
        for name in anchors:
            if name in cleaned:
                selected[name] = cleaned[name]
        cleaned = selected
    return cleaned


def generate_rows(
    store: LocalGraphStore,
    n_visits: int = 600,
    seed: int = 42,
    year: int = 2024,
    max_diseases: int = DEFAULT_MAX_DISEASES,
) -> dict:
    """Deterministically generate all analytics rows from the Task-2 KG vocabulary.

    Engine-independent on purpose: the SQLite builder below and the PostgreSQL
    builder (`pg_relational.build_pg_db`) both insert exactly these rows, so
    running the same generated SQL on both engines is a true logical-equivalence
    check rather than a comparison across two different datasets.
    """
    vocab = derive_vocab(store, max_diseases=max_diseases)
    diseases = list(vocab.keys())
    rng = random.Random(seed)

    visits: list[tuple] = []
    prescriptions: list[tuple] = []
    lab_tests: list[tuple] = []

    # disease popularity weights (stable), age profiles per disease
    weights = [rng.randint(2, 10) for _ in diseases]
    rx_id = test_id = 1
    start = date(year, 1, 1)
    for vid in range(1, n_visits + 1):
        dz = rng.choices(diseases, weights=weights, k=1)[0]
        slot = vocab[dz]
        # age skewed older for cardiac/diabetes
        age = max(18, min(95, int(rng.gauss(58, 14))))
        gender = rng.choice(["男", "女"])
        # visits trend upward across the year (seasonality-ish)
        day_offset = int(abs(rng.gauss(0, 1)) / 3 * 364) % 365
        # bias later months a bit to create a visible trend
        day_offset = min(364, int(day_offset * 0.6 + (vid / n_visits) * 364 * 0.4))
        vdate = (start + timedelta(days=day_offset)).isoformat()
        cost = round(rng.uniform(120, 3200), 2)
        visits.append(
            (vid, rng.randint(10000, 99999), age, gender, dz, slot["department"], vdate, cost)
        )
        for drug in rng.sample(slot["drugs"], k=min(len(slot["drugs"]), rng.randint(1, 2))):
            prescriptions.append((rx_id, vid, drug, rng.choice([7, 14, 28, 30])))
            rx_id += 1
        for exam in rng.sample(slot["exams"], k=min(len(slot["exams"]), rng.randint(1, 2))):
            lab_tests.append((test_id, vid, exam, 1 if rng.random() < 0.35 else 0))
            test_id += 1

    entities = [(n, d.get("type", "")) for n, d in store.g.nodes(data=True)]
    triples = [
        (head, store.g.nodes[head].get("type", ""), data.get("relation", ""),
         tail, store.g.nodes[tail].get("type", ""),
         data.get("confidence", 1.0), data.get("source", ""))
        for head, tail, data in store.g.edges(data=True)
    ]
    return {
        "diseases": diseases,
        "patient_visits": visits,
        "prescriptions": prescriptions,
        "lab_tests": lab_tests,
        "kg_entities": entities,
        "kg_triples": triples,
    }


def build_db(
    db_path: str | Path,
    store: LocalGraphStore,
    n_visits: int = 600,
    seed: int = 42,
    year: int = 2024,
    max_diseases: int = DEFAULT_MAX_DISEASES,
) -> dict:
    """Create the SQLite analytics DB. Returns a summary dict."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    rows = generate_rows(store, n_visits=n_visits, seed=seed, year=year, max_diseases=max_diseases)

    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    cur = conn.cursor()
    cur.executemany("INSERT INTO patient_visits VALUES (?,?,?,?,?,?,?,?)", rows["patient_visits"])
    cur.executemany("INSERT INTO prescriptions VALUES (?,?,?,?)", rows["prescriptions"])
    cur.executemany("INSERT INTO lab_tests VALUES (?,?,?,?)", rows["lab_tests"])
    # KG mirror tables (for graph-aware SQL and reuse traceability)
    cur.executemany("INSERT INTO kg_entities VALUES (?,?)", rows["kg_entities"])
    cur.executemany("INSERT INTO kg_triples VALUES (?,?,?,?,?,?,?)", rows["kg_triples"])
    conn.commit()
    conn.close()
    return {
        "db_path": str(db_path),
        "diseases": rows["diseases"],
        "n_visits": n_visits,
        "n_prescriptions": len(rows["prescriptions"]),
        "n_lab_tests": len(rows["lab_tests"]),
    }


def schema_text() -> str:
    """Schema description injected into the NL2SQL prompt.

    The `relation` hint spells out that kg_triples.relation stores the English
    ontology key, not the Chinese word the question used -- without it, a model
    asked "冠心病的并发症有哪些" reasonably generates `relation = '并发症'`
    (the Chinese word it just read in the question), which matches zero rows
    because the column actually holds `'complication'`. Found via
    benchmarks/nl2sql_hard_natural.json's kg-aware questions, which return
    genuinely wrong (not just differently-shaped) results without this hint.
    """
    from medigraph.schema.ontology import RELATION_TYPES

    relation_hint = "、".join(f"{key}({label})" for key, label in RELATION_TYPES.items())
    return (
        "表 patient_visits(visit_id, patient_id, age 年龄, gender 性别('男'/'女'), "
        "disease 疾病, department 科室, visit_date 就诊日期'YYYY-MM-DD', cost 费用)\n"
        "表 prescriptions(rx_id, visit_id, drug 药物, days 用药天数)\n"
        "表 lab_tests(test_id, visit_id, test_name 检查名称, abnormal 是否异常(1/0))\n"
        "表 kg_entities(name 实体名, type 类型)\n"
        "表 kg_triples(head, head_type, relation 关系, tail, tail_type, confidence, source)\n"
        f"kg_triples.relation 的取值是英文代码，不是中文，例如：{relation_hint}"
    )
