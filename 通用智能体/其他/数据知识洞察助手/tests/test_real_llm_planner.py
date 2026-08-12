import json
import os
import urllib.request

import pytest

from planners.task_planner import LLMTaskPlanner


def _load_project_env() -> None:
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    if not os.path.exists(env_path):
        return

    with open(env_path, encoding="utf-8") as env_file:
        for line in env_file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_project_env()


class FailingFallbackPlanner:
    def plan(self, task_description: str) -> list[dict[str, str]]:
        raise AssertionError(f"LLM planner unexpectedly fell back for task: {task_description}")


def _real_llm(prompt: str) -> str:
    api_key = os.environ["DKI_REAL_LLM_API_KEY"]
    base_url = os.environ.get("DKI_REAL_LLM_BASE_URL", "http://82.157.105.8:37853/v1").rstrip("/")
    model = os.environ.get("DKI_REAL_LLM_MODEL", "MiniMax-M2.7-highspeed")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You return strict JSON only. Do not include markdown fences or explanations.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    request = urllib.request.Request(
        url=f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data["choices"][0]["message"]["content"])


@pytest.mark.skipif(
    os.environ.get("RUN_REAL_LLM_TESTS") != "1" or not os.environ.get("DKI_REAL_LLM_API_KEY"),
    reason="Set RUN_REAL_LLM_TESTS=1 and DKI_REAL_LLM_API_KEY to run real LLM planner test.",
)
def test_real_minimax_llm_planner_returns_valid_plan():
    planner = LLMTaskPlanner(llm=_real_llm, fallback_planner=FailingFallbackPlanner())

    plan = planner.plan("只处理医疗文本，不要处理CSV")

    assert [item["step"] for item in plan] == ["read_text", "split_paragraph", "save_text_json"]
