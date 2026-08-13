import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "freeze_python_environment.py"
SPEC = importlib.util.spec_from_file_location("freeze_python_environment", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_environment_lock_is_hash_bound_and_create_once(tmp_path):
    payload = MODULE.build_lock("a" * 64)
    assert payload["kind"] == "python_environment_lock"
    assert any(
        item["name"].casefold() == "sqlglot" for item in payload["packages"]
    )
    output = tmp_path / "environment.lock.json"
    MODULE._publish_once(output, payload)
    assert output.is_file()
    with pytest.raises(FileExistsError):
        MODULE._publish_once(output, payload)
