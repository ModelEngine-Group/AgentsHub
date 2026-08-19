import json

from operators import SaveJsonOperator


def test_save_json_operator_writes_utf8_json(tmp_path):
    output_path = tmp_path / "nested" / "result.json"
    payload = {
        "status": "success",
        "items": [
            {"name": "肺炎", "type": "Disease"},
            {"name": "咳嗽", "type": "Symptom"},
        ],
    }

    saved_path = SaveJsonOperator().execute(payload, str(output_path))

    assert saved_path == str(output_path)
    assert output_path.exists()
    with open(output_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == payload
