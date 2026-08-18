from analysis.open_nl2sql.disease_alias import normalize_diseases
from analysis.open_nl2sql.indicator_alias import normalize_indicators
from analysis.open_nl2sql.synonym_rewrite import extract_future_window_days, rewrite_question


def test_normalize_diseases_supports_chinese_and_english() -> None:
    assert normalize_diseases("高血压合并 diabetes") == ["hypertension", "diabetes"]


def test_normalize_diseases_deduplicates_aliases() -> None:
    assert normalize_diseases("慢阻肺和慢性阻塞性肺疾病") == ["copd"]


def test_normalize_indicators_supports_aliases() -> None:
    assert normalize_indicators("HbA1c、LDL-C和收缩压") == ["hba1c", "ldl_c", "systolic_bp"]


def test_extract_future_window_days_arabic_number() -> None:
    assert extract_future_window_days("未来 180 天需要随访多少患者？") == 180


def test_extract_future_window_days_chinese_number() -> None:
    assert extract_future_window_days("接下来十五日需要随访多少患者？") == 15


def test_extract_future_window_days_uses_default() -> None:
    assert extract_future_window_days("查询随访人数", default=30) == 30


def test_rewrite_future_followup_preserves_requested_window() -> None:
    result = rewrite_question("未来 17 天需要随访的患者有多少？")
    assert result["canonical_id"] == "future_followup_chart_bundle"
    assert result["window_days"] == 17


def test_rewrite_disease_inventory() -> None:
    result = rewrite_question("当前常见病有哪些？")
    assert result["canonical_id"] == "kg_disease_inventory"


def test_rewrite_unknown_question_is_identity() -> None:
    result = rewrite_question("请介绍项目架构")
    assert result["canonical_id"] == "standard_or_unknown"
    assert result["question"] == "请介绍项目架构"
