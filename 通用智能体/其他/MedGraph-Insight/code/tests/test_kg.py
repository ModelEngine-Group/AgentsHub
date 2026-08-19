"""知识图谱存储层测试：构建、查询、持久化、SQLite 导出。"""

from __future__ import annotations

import sqlite3

from mgi.kg import MedicalKG
from mgi.kg.schema import Triple


def _sample_triples() -> list[Triple]:
    return [
        Triple("2型糖尿病", "Disease", "HAS_SYMPTOM", "多尿", "Symptom", confidence=0.9),
        Triple("2型糖尿病", "Disease", "TREATED_BY_DRUG", "二甲双胍", "Drug", confidence=0.9),
        Triple("2型糖尿病", "Disease", "COMPLICATION_OF", "糖尿病肾病", "Disease", confidence=0.8),
    ]


def test_add_and_query_neighbors() -> None:
    kg = MedicalKG()
    kg.add_triples(_sample_triples())
    syms = kg.neighbors("2型糖尿病", "HAS_SYMPTOM")
    assert any(tail == "多尿" for _, tail, _ in syms)


def test_stats() -> None:
    kg = MedicalKG()
    kg.add_triples(_sample_triples())
    stats = kg.stats()
    assert stats["num_relations"] == 3
    assert stats["num_entities"] == 4


def test_save_load_roundtrip(tmp_path) -> None:
    kg = MedicalKG()
    kg.add_triples(_sample_triples())
    p = tmp_path / "kg.json"
    kg.save(p)
    loaded = MedicalKG.load(p)
    assert loaded.stats() == kg.stats()


def test_sqlite_export(tmp_path) -> None:
    kg = MedicalKG()
    kg.add_triples(_sample_triples())
    db = tmp_path / "kg.sqlite"
    kg.to_sqlite(db)
    con = sqlite3.connect(db)
    n_ent = con.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    n_rel = con.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
    con.close()
    assert n_ent == 4
    assert n_rel == 3
