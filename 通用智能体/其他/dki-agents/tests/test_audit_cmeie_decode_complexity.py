import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_cmeie_decode_complexity.py"
SPEC = importlib.util.spec_from_file_location("audit_cmeie_decode_complexity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_decode_complexity_audit_is_create_once(tmp_path):
    output = tmp_path / "audit.json"
    MODULE.create_once(output, {"contains_source_text_or_predictions": False})
    assert output.exists()
    try:
        MODULE.create_once(output, {"contains_source_text_or_predictions": False})
    except SystemExit:
        pass
    else:
        raise AssertionError("decode complexity audit overwrite was allowed")
