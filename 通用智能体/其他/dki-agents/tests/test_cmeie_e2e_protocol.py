import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_re_cmeie_e2e.py"
SPEC = importlib.util.spec_from_file_location("eval_re_cmeie_e2e", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_sentence_context_supports_chinese_and_ascii_boundaries():
    text = "甲乙。丙丁?戊己"
    assert MODULE._sentence_context(text, [0, 1])[:3] == (0, 0, 3)
    assert MODULE._sentence_context(text, [3, 4])[:3] == (1, 3, 6)
    assert MODULE._sentence_context(text, [6, 7])[:3] == (2, 6, 8)


def test_span_gap_is_zero_for_touching_or_overlapping_spans():
    assert MODULE._span_gap([1, 3], [3, 5]) == 0
    assert MODULE._span_gap([1, 4], [3, 5]) == 0
    assert MODULE._span_gap([1, 3], [6, 8]) == 3


def test_rows_hash_is_order_sensitive_and_stable():
    first = [{"text": "甲"}, {"text": "乙"}]
    second = list(reversed(first))
    assert MODULE._rows_sha256(first) == MODULE._rows_sha256(first)
    assert MODULE._rows_sha256(first) != MODULE._rows_sha256(second)


def test_predicate_breakdown_keeps_relation_specific_counts():
    gold = [{("甲", "关系一", "乙"), ("甲", "关系二", "丙")}]
    predicted = [{("甲", "关系一", "乙"), ("甲", "关系一", "丁")}]
    result = MODULE._predicate_breakdown(
        gold, predicted, ["关系一", "关系二"],
    )
    assert result["关系一"]["tp"] == 1
    assert result["关系一"]["fp"] == 1
    assert result["关系一"]["fn"] == 0
    assert result["关系二"]["fn"] == 1


def test_decode_supports_per_predicate_thresholds_and_disable():
    predicted, stats = MODULE._decode_spo_rows(
        n_rows=1,
        pairs=[(0, "甲", "乙"), (0, "丙", "丁")],
        allowed_rows=[{0, 1}, {0, 1}],
        probability_rows=[[0.8, 0.2], [0.9, 0.7]],
        predicates=["关系一", "关系二"],
        default_threshold=0.5,
        predicate_thresholds={"关系一": None, "关系二": 0.5},
    )
    assert predicted == [{("丙", "关系二", "丁")}]
    assert stats["n_predicted_candidate_pairs"] == 1


def test_internal_calibration_raises_a_noisy_predicate_threshold():
    predicted, _, report = MODULE._calibrate_threshold_policy(
        threshold_grid=[0.5, 0.9],
        gold_rows=[{("甲", "有效", "乙")}, set()],
        pairs=[(0, "甲", "乙"), (1, "丙", "丁")],
        allowed_rows=[{0, 1}, {0, 1}],
        probability_rows=[[0.8, 0.8], [0.1, 0.8]],
        predicates=["有效", "噪声"],
    )
    assert report["selected_thresholds"] == {"有效": 0.5, "噪声": 0.9}
    assert report["selected_metrics"]["micro_f1"] == 1.0
    assert predicted == [{("甲", "有效", "乙")}, set()]
