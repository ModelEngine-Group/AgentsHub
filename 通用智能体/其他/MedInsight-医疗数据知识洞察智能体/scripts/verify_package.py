from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "agent.json",
    "README.md",
    "LICENSE",
    "mcps/server.py",
    "mcps/medinsight_tools.py",
    "mcps/start.sh",
    "mcps/requirements.txt",
    "skills/medinsight-workflow.md",
    "knowledge/medical_schema.json",
    "knowledge/empty_graph.json",
    "config/mcp.json",
    "config/env.example",
    "scripts/register_in_nexent.py",
)
EXPECTED_TOOLS = {
    "process_medical_data",
    "build_medical_knowledge_graph",
    "ask_medical_knowledge_graph",
    "analyze_medical_graph",
    "query_medical_graph_analysis",
    "run_data_knowledge_insight_pipeline",
}
TEXT_SUFFIXES = {".json", ".md", ".py", ".sh", ".txt", ".example"}
FORBIDDEN_PATTERNS = {
    "local absolute path": re.compile(r"/Users/[A-Za-z0-9._-]+/"),
    "possible API key": re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
}
SKIP_DIRS = {".git", ".pytest_cache", ".venv", "__pycache__"}


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def load_json(relative_path: str) -> object:
    path = ROOT / relative_path
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"invalid JSON: {relative_path}: {exc}")


def main() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    if missing:
        fail(f"missing required files: {', '.join(missing)}")

    agent = load_json("agent.json")
    if not isinstance(agent, dict) or set(agent) != {"agent_id", "agent_info", "mcp_info"}:
        fail("agent.json must contain agent_id, agent_info and mcp_info")

    agent_info = agent.get("agent_info")
    if not isinstance(agent_info, dict) or not agent_info:
        fail("agent_info is empty")
    definition = next(iter(agent_info.values()))
    if not isinstance(definition, dict):
        fail("agent definition is invalid")
    if definition.get("model_ids") != [] or definition.get("business_logic_model_id") is not None:
        fail("agent.json must not bind an environment-specific model")
    tools = definition.get("tools")
    if not isinstance(tools, list):
        fail("agent tools must be a list")
    tool_names = {tool.get("name") for tool in tools if isinstance(tool, dict)}
    if tool_names != EXPECTED_TOOLS:
        fail(f"unexpected MCP tool set: {sorted(str(name) for name in tool_names)}")
    for tool in tools:
        try:
            inputs = ast.literal_eval(str(tool.get("inputs") or "{}"))
        except (SyntaxError, ValueError) as exc:
            fail(f"invalid inputs declaration for {tool.get('name')}: {exc}")
        if not isinstance(inputs, dict):
            fail(f"inputs declaration for {tool.get('name')} must resolve to a dict")

    mcp_info = agent.get("mcp_info")
    if not isinstance(mcp_info, list) or not any(
        isinstance(item, dict) and item.get("mcp_server_name") == "medinsight"
        for item in mcp_info
    ):
        fail("medinsight MCP declaration is missing")
    medinsight_mcp = next(
        item for item in mcp_info
        if isinstance(item, dict) and item.get("mcp_server_name") == "medinsight"
    )
    if medinsight_mcp.get("mcp_url") != "http://host.docker.internal:18100/sse":
        fail("agent.json contains an unexpected MCP URL")

    load_json("knowledge/medical_schema.json")
    load_json("knowledge/empty_graph.json")
    load_json("config/mcp.json")

    python_files = sorted((ROOT / "mcps").rglob("*.py"))
    for path in python_files:
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except (OSError, SyntaxError) as exc:
            fail(f"Python syntax error: {path.relative_to(ROOT)}: {exc}")

    findings: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if not path.is_file() or path.name == ".gitkeep":
            continue
        if "runtime" in relative.parts:
            findings.append(f"runtime artifact: {relative}")
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"LICENSE", ".gitignore"}:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(content):
                findings.append(f"{label}: {relative}")
    if findings:
        fail("unsafe archive contents:\n  - " + "\n  - ".join(findings))

    print("[PASS] AgentsHub package structure")
    print("[PASS] agent.json and MCP declarations")
    print(f"[PASS] Python syntax ({len(python_files)} files)")
    print("[PASS] no local absolute paths, API keys or runtime artifacts")
    print(f"[PASS] six MCP tools: {', '.join(sorted(EXPECTED_TOOLS))}")


if __name__ == "__main__":
    main()
