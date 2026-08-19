from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LORA_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
DEFAULT_LORA_ADAPTER = PROJECT_ROOT / "artifacts" / "task1" / "models" / "flow_planner_lora_final"
DEFAULT_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


SYSTEM_PROMPT = (
    "你是医疗数据处理流程编排智能体。请根据任务描述和数据画像，"
    "选择合适的数据处理算子，并只输出 JSON。"
)


def lora_planner_available(adapter_path: Path = DEFAULT_LORA_ADAPTER, python_bin: Path = DEFAULT_PYTHON) -> bool:
    return python_bin.exists() and (adapter_path / "adapters.safetensors").exists()


def infer_profile_buckets(profile: dict[str, Any]) -> dict[str, str]:
    record_count = int(profile.get("record_count") or 0)
    text_coverage = float(profile.get("text_coverage") or 0)
    structured_coverage = float(profile.get("structured_field_coverage") or 0)
    data_mode = str(profile.get("detected_mode") or "hybrid")
    return {
        "data_mode": data_mode,
        "record_bucket": "small" if record_count < 100 else "medium" if record_count < 10000 else "large",
        "text_coverage_bucket": _coverage_bucket(text_coverage),
        "structured_coverage_bucket": _coverage_bucket(structured_coverage),
    }


def build_lora_prompt(task: str, profile: dict[str, Any], optimization_goal: str = "quality_first") -> str:
    buckets = infer_profile_buckets(profile)
    noise_types = profile.get("noise_types") or []
    missing_fields = profile.get("missing_fields") or []
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"任务：{task}\n"
        f"数据模式：{buckets['data_mode']}\n"
        f"优化目标：{optimization_goal}\n"
        f"记录规模：{buckets['record_bucket']}\n"
        f"文本覆盖率：{buckets['text_coverage_bucket']}\n"
        f"结构化字段覆盖率：{buckets['structured_coverage_bucket']}\n"
        f"噪声类型：{', '.join(str(item) for item in noise_types) or '无'}\n"
        f"缺失字段：{', '.join(str(item) for item in missing_fields) or '无'}\n\n"
        "请输出包含 operators、execution_policy、rationale 的 JSON。"
    )


def run_lora_flow_planner(
    task: str,
    profile: dict[str, Any],
    model: str = DEFAULT_LORA_MODEL,
    adapter_path: Path = DEFAULT_LORA_ADAPTER,
    python_bin: Path = DEFAULT_PYTHON,
    timeout_seconds: float = 120,
    optimization_goal: str = "quality_first",
) -> dict[str, Any]:
    if not lora_planner_available(adapter_path, python_bin):
        return {
            "status": "unavailable",
            "reason": "LoRA adapter or Python runtime is missing.",
            "model": model,
            "adapter_path": str(adapter_path),
        }

    prompt = build_lora_prompt(task, profile, optimization_goal)
    command = [
        str(python_bin),
        "-m",
        "mlx_lm",
        "generate",
        "--model",
        model,
        "--adapter-path",
        str(adapter_path),
        "--prompt",
        prompt,
        "--max-tokens",
        "400",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "reason": f"LoRA planner exceeded {timeout_seconds}s timeout.",
            "model": model,
            "adapter_path": str(adapter_path),
            "stdout": exc.stdout,
            "stderr": exc.stderr,
        }

    parsed = extract_json(result.stdout)
    status = "ok" if result.returncode == 0 and "operators" in parsed else "error"
    return {
        "status": status,
        "model": model,
        "adapter_path": str(adapter_path),
        "prompt": prompt,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "parsed_plan": parsed,
    }


def extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {"raw_output": text, "parse_error": "No JSON object found."}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return {"raw_output": text, "parse_error": str(exc)}
    return payload if isinstance(payload, dict) else {"raw_output": text, "parse_error": "JSON root is not an object."}


def _coverage_bucket(value: float) -> str:
    if value >= 0.8:
        return "high"
    if value >= 0.4:
        return "medium"
    return "low"
