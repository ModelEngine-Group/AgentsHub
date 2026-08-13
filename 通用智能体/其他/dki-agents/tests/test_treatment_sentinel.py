from dki.nlp.treatment_sentinel import assess_profile


def _reference():
    return {
        "f1": 0.97,
        "macro_f1": 0.96,
        "worst_type_f1": 0.93,
        "throughput_samples_s": 5000,
        "by_type": {"Treatment": {"recall": 0.94, "tp": 94, "fn": 6}},
    }


def test_treatment_zero_recall_blocks_extreme_profile():
    candidate = _reference()
    candidate["by_type"] = {"Treatment": {"recall": 0.0, "tp": 0, "fn": 81}}
    candidate["throughput_samples_s"] = 5961
    result = assess_profile(candidate, _reference(), {"throughput_samples_s": 5961})
    assert not result["promotion"]
    assert not result["checks"]["treatment_recall"]


def test_balanced_profile_can_pass_all_gates():
    candidate = _reference()
    candidate["throughput_samples_s"] = 5500
    result = assess_profile(candidate, _reference(), {"throughput_samples_s": 5961})
    assert result["promotion"]


def test_missing_treatment_support_is_inconclusive():
    candidate = _reference()
    candidate["by_type"] = {"Treatment": {"recall": 1.0, "tp": 0, "fn": 0}}
    result = assess_profile(candidate, _reference(), {"throughput_samples_s": 5961})
    assert not result["promotion"]
    assert not result["checks"]["treatment_support"]
