import json

from scripts.nl2sql_curator_grounding_check import (
    missing_grounded_literals,
    validate_corpus,
)


def test_curator_grounding_check_matches_numeric_string_and_empty_contract(tmp_path):
    assert missing_grounded_literals(
        "查询年龄至少 18 岁且地区为上海的前 10 名患者",
        "SELECT patient_id FROM patients WHERE age>=18 AND region='\u4e0a\u6d77' LIMIT 10",
    ) == ()
    assert missing_grounded_literals(
        "查询成年患者", "SELECT patient_id FROM patients WHERE age>=18"
    ) == ("18",)
    try:
        missing_grounded_literals(
            "查询无药物记录", "SELECT visit_id FROM visits WHERE drug_prescribed=''"
        )
    except ValueError as exc:
        assert "empty predicate literal" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("empty predicate literal must fail closed")

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "question": "查询年龄至少 18 岁的患者",
                "gold_sql": "SELECT patient_id FROM patients WHERE age>=18",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    assert validate_corpus(corpus) == {
        "n_rows": 1,
        "n_passed": 1,
        "n_failed": 0,
        "gate_passed": True,
    }
