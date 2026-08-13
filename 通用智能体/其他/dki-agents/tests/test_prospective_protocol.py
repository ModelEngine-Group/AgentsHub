import sys
from pathlib import Path

import pytest

from dki.nlp.assertion import AssertionValidationError, evaluate_assertions, validate_assertion

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from blind_protocol import consume, seal, verify


def _entity(assertion="present"):
    return {
        "span": [0, 2],
        "type": "Disease",
        "assertion": {
            "polarity_certainty": assertion,
            "temporality": "current",
            "experiencer": "patient",
        },
    }


def test_assertion_schema_and_separate_metrics():
    assert validate_assertion(_entity(), "疾病")["assertion"]["experiencer"] == "patient"
    result = evaluate_assertions({"d": [_entity("negated")]}, {"d": [_entity()]})
    assert result["extraction_exact"]["f1"] == 1.0
    assert result["oracle_span_assertion_accuracy"] == 0.0
    assert result["end_to_end_joint_accuracy"] == 0.0


def test_invalid_assertion_fails_closed():
    entity = _entity()
    entity["assertion"]["experiencer"] = "guess"
    with pytest.raises(AssertionValidationError):
        validate_assertion(entity, "疾病")


def test_create_once_seal_and_receipt(tmp_path):
    protocol = tmp_path / "protocol.json"
    artifact = tmp_path / "model.manifest.json"
    seal_path = tmp_path / "seal.json"
    receipt = tmp_path / "receipt.json"
    protocol.write_text("{}", encoding="utf-8")
    artifact.write_text("{}", encoding="utf-8")
    seal(protocol, [artifact], seal_path)
    assert verify(seal_path)["threat_model"].startswith("internally")
    consume(seal_path, receipt)
    with pytest.raises(RuntimeError, match="already consumed|already marked consumed"):
        consume(seal_path, receipt)
    with pytest.raises(RuntimeError, match="already marked consumed"):
        consume(seal_path, tmp_path / "alternate-receipt.json")
    assert verify(seal_path)["consumed"] is True
    with pytest.raises(RuntimeError, match="already exists"):
        seal(protocol, [artifact], seal_path)


def test_sealed_artifact_tamper_detected(tmp_path):
    protocol = tmp_path / "protocol.json"
    artifact = tmp_path / "model.json"
    seal_path = tmp_path / "seal.json"
    protocol.write_text("{}", encoding="utf-8")
    artifact.write_text("one", encoding="utf-8")
    seal(protocol, [artifact], seal_path)
    artifact.write_text("two", encoding="utf-8")
    with pytest.raises(RuntimeError, match="mismatch"):
        verify(seal_path)
