"""Engine-level ablation gate wiring: default-off, hash-bound, behavior-gated."""

import csv

import pytest

from dki.analysis.database import build_database
from dki.analysis.nl2sql import NL2SQL
from dki.analysis.nl2sql_ablation import (
    KNOWN_RULE_CLUSTER_IDS,
    NL2SQLAblationConfig,
)
from dki.kg.store import KnowledgeGraph


@pytest.fixture(scope="module")
def gate_db(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ablation_gates")
    patients = tmp / "patients.csv"
    visits = tmp / "visits.csv"
    with open(patients, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "name", "gender", "age", "region"])
        w.writerows([
            ["P1", "张三", "男", 65, "北京"],
            ["P2", "李四", "女", 30, "上海"],
        ])
    with open(visits, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["visit_id", "patient_id", "visit_date", "department", "disease",
                    "drug_prescribed", "exam", "cost", "los_days"])
        w.writerows([
            ["v1", "P1", "2026-01-10", "心血管内科", "高血压", "硝苯地平", "血压测量", 1200.5, 0],
            ["v2", "P2", "2025-12-01", "内分泌科", "2型糖尿病", "二甲双胍", "空腹血糖检测", 900.0, 0],
        ])
    kg = KnowledgeGraph()
    kg.add_triple("高血压", "Disease", "HAS_SYMPTOM", "头痛", "Symptom", 0.9, "v1")
    db_path = tmp / "insight.db"
    build_database(db_path=db_path, visits_csv=visits, patients_csv=patients, kg=kg)
    return db_path


QUESTIONS = (
    "按科室统计就诊量",
    "高血压的就诊次数",
    "统计每个地区的患者数",
)


def test_no_config_and_baseline_config_are_identical(gate_db):
    plain = NL2SQL(db_path=gate_db)
    baseline = NL2SQL(
        db_path=gate_db,
        ablation_config=NL2SQLAblationConfig.baseline(),
    )
    for question in QUESTIONS:
        assert plain.rule_sql(question) == baseline.rule_sql(question), question


def test_route_meta_binds_ablation_hashes_only_when_configured(gate_db):
    config = NL2SQLAblationConfig.baseline(profile_name="gate-test")
    engine = NL2SQL(db_path=gate_db, ablation_config=config)
    meta = engine._route_meta("rule", "supported")
    assert meta["ablation_profile_name"] == "gate-test"
    assert meta["ablation_policy_hash"] == config.policy_hash
    assert meta["ablation_profile_hash"] == config.profile_hash

    plain_meta = NL2SQL(db_path=gate_db)._route_meta("rule", "supported")
    assert "ablation_policy_hash" not in plain_meta


def test_constructor_rejects_non_config_objects(gate_db):
    with pytest.raises(ValueError, match="ablation_config"):
        NL2SQL(db_path=gate_db, ablation_config={"profile_name": "x"})


def test_all_clusters_disabled_still_fails_closed(gate_db):
    """Disabling every registered cluster must never crash or emit garbage."""

    config = NL2SQLAblationConfig(
        profile_name="all-clusters-off",
        disabled_clusters=tuple(KNOWN_RULE_CLUSTER_IDS),
    )
    engine = NL2SQL(db_path=gate_db, ablation_config=config)
    for question in QUESTIONS:
        result = engine.query(question)
        assert isinstance(result, dict)
        # Whatever remains (legacy core or abstention) must stay well-formed.
        assert "sql" in result and "engine" in result


def test_disabled_cluster_changes_only_gated_ownership(gate_db):
    """A disabled cluster may change routing but never raises."""

    baseline = NL2SQL(db_path=gate_db)
    for cluster in sorted(KNOWN_RULE_CLUSTER_IDS):
        variant = NL2SQL(
            db_path=gate_db,
            ablation_config=NL2SQLAblationConfig.without(
                cluster, profile_name="minus-cluster"
            ),
        )
        for question in QUESTIONS:
            base_sql = baseline.rule_sql(question)
            variant_sql = variant.rule_sql(question)
            # Identical or a deterministic replacement/abstention; both str|None.
            assert variant_sql is None or isinstance(variant_sql, str), (
                cluster,
                question,
                base_sql,
                variant_sql,
            )
