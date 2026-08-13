"""BI 增强的确定性单元测试：热力图/桑基图数据形状 + 关键洞察 + 报告含新图型。"""
import pytest

from dki.paths import OUTPUT_DIR


@pytest.fixture(scope="module")
def db():
    p = OUTPUT_DIR / "insight.db"
    if not p.is_file():
        pytest.skip("缺少 insight.db，请先 run.ps1 demo-all")
    return p


def test_age_disease_heatmap_shape(db):
    from dki.analysis.metrics import age_disease_heatmap
    h = age_disease_heatmap(db)
    assert h["ycats"] == ["0-18", "19-40", "41-60", "60+"]
    assert h["xcats"] and len(h["xcats"]) <= 8
    # data 是 [x,y,val] 且索引在范围内
    for x, y, v in h["data"]:
        assert 0 <= x < len(h["xcats"]) and 0 <= y < len(h["ycats"])


def test_disease_department_sankey_valid(db):
    from dki.analysis.metrics import disease_department_sankey
    s = disease_department_sankey(db)
    names = {n["name"] for n in s["nodes"]}
    for link in s["links"]:
        assert link["source"] in names and link["target"] in names
        assert link["value"] > 0


def test_key_insights_nonempty(db):
    from dki.analysis.metrics import key_insights, overview_stats, standard_metrics
    ins = key_insights(db, standard_metrics(db), overview_stats(db))
    assert len(ins) >= 3 and all(isinstance(s, str) and s for s in ins)


def test_report_contains_new_charts(db, tmp_path):
    from dki.analysis.bi import build_default_report
    from dki.kg.store import KnowledgeGraph
    kg_path = OUTPUT_DIR / "kg" / "medical_kg.json"
    kg = KnowledgeGraph.load(kg_path) if kg_path.exists() else None
    out = build_default_report(db, kg=kg, out_path=tmp_path / "r.html")
    html = out.read_text(encoding="utf-8")
    assert '"heatmap"' in html and '"sankey"' in html
    assert "关键洞察" in html
