import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.common.llm_config import load_llm_config, openai_extra_kwargs
from src.common.path_security import resolve_allowed_path


def test_load_llm_config_accepts_env_and_drops_unrelated_keys(tmp_path):
    config = tmp_path / "llm_config.env"
    config.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=sk-test",
                "OPENAI_BASE_URL=https://example.test/v1",
                "OPENAI_MODEL=glm-5.1",
                "IGNORED=value",
            ]
        ),
        encoding="utf-8",
    )

    values = load_llm_config(config)

    assert values == {
        "api_key": "sk-test",
        "base_url": "https://example.test/v1",
        "model_name": "glm-5.1",
    }


def test_load_llm_config_rejects_base_url_credentials(tmp_path):
    config = tmp_path / "llm_config.json"
    config.write_text(
        json.dumps(
            {
                "api_key": "sk-test",
                "base_url": "https://user:pass@example.test/v1",
                "model_name": "glm-5.1",
            }
        ),
        encoding="utf-8",
    )

    assert load_llm_config(config) is None


def test_load_llm_config_accepts_uppercase_json_aliases(tmp_path):
    config = tmp_path / "llm_config.json"
    config.write_text(
        json.dumps(
            {
                "OPENAI_API_KEY": "sk-test",
                "OPENAI_BASE_URL": "https://example.test/v1",
                "OPENAI_MODEL": "glm-5.1",
            }
        ),
        encoding="utf-8",
    )

    assert load_llm_config(config) == {
        "api_key": "sk-test",
        "base_url": "https://example.test/v1",
        "model_name": "glm-5.1",
    }


def test_load_llm_config_accepts_provider_thinking_option(tmp_path):
    config = tmp_path / "llm_config.env"
    config.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=sk-test",
                "OPENAI_BASE_URL=https://api.deepseek.com",
                "OPENAI_MODEL=deepseek-v4-flash",
                "OPENAI_THINKING=disabled",
            ]
        ),
        encoding="utf-8",
    )

    values = load_llm_config(config)

    assert values == {
        "api_key": "sk-test",
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-v4-flash",
        "thinking": {"type": "disabled"},
    }
    assert openai_extra_kwargs(values) == {
        "extra_body": {"thinking": {"type": "disabled"}}
    }


def test_resolve_allowed_path_rejects_escape(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    safe_file = allowed / "input.txt"
    safe_file.write_text("ok", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    assert resolve_allowed_path(safe_file, allowed_roots=[allowed]) == safe_file.resolve()
    with pytest.raises(ValueError, match="outside allowed roots"):
        resolve_allowed_path(outside, allowed_roots=[allowed])


def test_task3_api_rejects_graph_file_outside_allowed_roots():
    from src.pipelines.task3_api_server import app

    client = TestClient(app)
    response = client.post(
        "/api/task3/centrality",
        json={"graph_file": str(Path.home() / "private_graph.json")},
    )

    assert response.status_code == 400
    assert "outside allowed roots" in response.json()["detail"]


def test_task1_api_rejects_input_path_outside_allowed_roots():
    from src.pipelines.task1_api_server import app, _tasks

    _tasks.clear()
    client = TestClient(app)
    response = client.post(
        "/api/task1/process",
        json={"input_path": str(Path.home() / "private_patients.csv")},
    )

    assert response.status_code == 400
    assert "outside allowed roots" in response.json()["detail"]


def test_task2_api_rejects_input_path_outside_allowed_roots():
    from src.pipelines.task2_api_server import app, _tasks

    _tasks.clear()
    client = TestClient(app)
    response = client.post(
        "/api/task2/process",
        json={"input_path": str(Path.home() / "private_notes.txt")},
    )

    assert response.status_code == 400
    assert "outside allowed roots" in response.json()["detail"]
