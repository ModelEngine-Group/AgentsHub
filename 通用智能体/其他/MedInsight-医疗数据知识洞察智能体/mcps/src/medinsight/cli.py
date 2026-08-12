from __future__ import annotations

import argparse
import json
import locale
from pathlib import Path
import subprocess
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from medinsight.common.utils import read_csv_records, read_jsonl
from medinsight.task1.api import build_nexent_data_processing_plan, run_nexent_task1_agent_tool
from medinsight.task1.service import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    DEFAULT_TASK,
)
from medinsight.task2.api import build_nexent_kg_plan, run_nexent_task2_kg_agent_tool
from medinsight.task2.operators.qa import answer_question
from medinsight.task2.service import (
    DEFAULT_TASK2_INPUT,
    DEFAULT_TASK2_OUTPUT,
    DEFAULT_TASK2_TASK,
    DEFAULT_TASK2_TEXT_OUTPUT,
    DEFAULT_TASK2_TEXT_TASK,
    run_task2_text_kg_build,
)

try:
    import readline
except ImportError:  # pragma: no cover - readline is unavailable on some platforms.
    readline = None

try:
    locale.setlocale(locale.LC_CTYPE, "")
except locale.Error:
    pass

TASK2_QA_TOOL_NAME = "medinsight_answer_medical_kg_question"
TASK2_TEXT_KG_TOOL_NAME = "medinsight_generate_medical_kg_from_text"
ANSI_ANSWER = "\033[1;32m"
ANSI_EVIDENCE = "\033[2;36m"
ANSI_RESET = "\033[0m"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MedInsight Task1 data processing agent.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("task1", help="Run Task1 through the Nexent tool entry.")
    demo.add_argument("--input", default=str(DEFAULT_INPUT), help="Input JSONL or CSV path.")
    demo.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output artifact directory.")
    demo.add_argument("--task", default=DEFAULT_TASK, help="Natural-language processing task.")
    demo.add_argument(
        "--planner-runtime",
        choices=["deterministic", "lora", "auto"],
        default="deterministic",
        help="Planner used by the Nexent planning tool.",
    )
    demo.add_argument("--lora-adapter-path", default=None, help="Optional LoRA adapter path for planner-runtime=lora.")
    demo.add_argument("--lora-timeout-seconds", type=float, default=120.0)

    task2 = subparsers.add_parser("task2", help="Run Task2 medical knowledge graph and QA agent.")
    task2.add_argument("--input-dir", default=str(DEFAULT_TASK2_INPUT), help="CBLUE task2 raw data directory.")
    task2.add_argument("--output", default=str(DEFAULT_TASK2_OUTPUT), help="Task2 output artifact directory.")
    task2.add_argument(
        "--shared-output",
        default="data/shared/knowledge_graph",
        help="Shared graph output directory for Task3 reuse.",
    )
    task2.add_argument("--task", default=DEFAULT_TASK2_TASK, help="Natural-language KG build task.")
    task2.add_argument("--cmeee-split", default="train", help="train, dev, test, train+dev, or all.")
    task2.add_argument("--cmeie-split", default="train", help="train, dev, test, train+dev, or all.")
    task2.add_argument("--cmeee-limit", type=int, default=2000, help="0 means all records.")
    task2.add_argument("--cmeie-limit", type=int, default=2000, help="0 means all records.")
    task2.add_argument("--chip-cdn-split", default="train", help="train, dev, test, train+dev, or all.")
    task2.add_argument("--cmedcausal-split", default="train", help="train, dev, test, train+dev, or all.")
    task2.add_argument("--chip-cdn-limit", type=int, default=0, help="0 means all CHIP-CDN records.")
    task2.add_argument("--cmedcausal-limit", type=int, default=0, help="0 means all CMedCausal records.")
    task2.add_argument("--icd10-limit", type=int, default=0, help="0 means all ICD-10 lexicon rows.")
    task2.add_argument("--surgery-lexicon-limit", type=int, default=0, help="0 means all surgery lexicon rows.")
    task2.add_argument("--disable-enrichment", action="store_true", help="Disable CHIP-CDN/CMedCausal graph enrichment.")

    task2_text = subparsers.add_parser("task2-text-kg", help="Generate a Task2 medical KG from user-provided medical text.")
    task2_text.add_argument("--text", default="", help="Medical text to extract entities and relations from.")
    task2_text.add_argument("--input", default="", help="Optional UTF-8 text file. Used when --text is empty.")
    task2_text.add_argument("--output", default=str(DEFAULT_TASK2_TEXT_OUTPUT), help="Output artifact directory.")
    task2_text.add_argument(
        "--shared-output",
        default="data/shared/knowledge_graph",
        help="Shared graph output directory for Task3 reuse.",
    )
    task2_text.add_argument("--task", default=DEFAULT_TASK2_TEXT_TASK, help="Natural-language KG generation task.")
    task2_text.add_argument("--doc-id", default="custom-medical-text-000001")
    task2_text.add_argument("--question", default="", help="Optional QA question after KG generation.")
    task2_text.add_argument("--interactive", action="store_true", help="Prompt for medical text in the terminal.")
    task2_text.add_argument(
        "--multiline",
        action="store_true",
        help="With --interactive, keep reading lines until an empty line. Default reads one line and runs immediately.",
    )
    task2_text.add_argument(
        "--mode",
        choices=["tool", "local"],
        default="tool",
        help="tool calls the Nexent-imported Outer API Tool; local is for offline debugging.",
    )
    task2_text.add_argument("--task2-api", default="http://127.0.0.1:18081")
    task2_text.add_argument("--nexent-config-api", default="http://127.0.0.1:5010")
    task2_text.add_argument("--authorization", default=None)
    task2_text.add_argument("--timeout", type=int, default=300)

    task2_qa = subparsers.add_parser("task2-qa", help="Ask questions through the Task2 Nexent medical KG QA agent.")
    task2_qa.add_argument(
        "--mode",
        choices=["tool", "rule", "local"],
        default="tool",
        help=(
            "tool/rule runs the Task2 QA agent through Nexent /tool/validate; "
            "local is for offline debugging only."
        ),
    )
    task2_qa.add_argument("--question", default="", help="Question to answer. Omit with --interactive for a terminal loop.")
    task2_qa.add_argument("--graph-path", default="data/shared/knowledge_graph/task2_user_medical_kg_library.json")
    task2_qa.add_argument("--interactive", action="store_true", help="Start a terminal Q&A loop.")
    task2_qa.add_argument("--task2-api", default="http://127.0.0.1:18081")
    task2_qa.add_argument("--nexent-config-api", default="http://127.0.0.1:5010")
    task2_qa.add_argument("--agent-name", default="medinsight_medical_kg_qa_agent")
    task2_qa.add_argument("--agent-id", type=int, default=0)
    task2_qa.add_argument("--authorization", default=None)
    task2_qa.add_argument("--timeout", type=int, default=300)
    task2_qa.add_argument("--clean-output", action="store_true", help="Print only the answer body for interactive demos.")
    task2_qa.add_argument("--title", default="医疗知识图谱问答（规则模式）", help="Interactive QA page title.")
    return parser


def load_records_for_planning(input_path: Path) -> list[dict]:
    if input_path.suffix.lower() == ".csv":
        return read_csv_records(input_path)
    return read_jsonl(input_path)


def resolve_task2_text_input(args: argparse.Namespace) -> str:
    if args.text:
        return normalize_task2_medical_source(str(args.text))
    if args.input:
        return normalize_task2_medical_source(str(args.input))
    if args.interactive:
        if not args.multiline:
            print("请输入任务一 processed_documents.json 路径、医疗文本，或一行 JSON（字段 text/records）。回车后立即生成知识图谱。")
            reset_terminal_input()
            return normalize_task2_medical_source(input("> ").strip())
        print("请输入医疗文本，支持多行；输入空行结束。")
        lines: list[str] = []
        while True:
            reset_terminal_input()
            line = input("> ")
            if not line.strip():
                break
            lines.append(line)
        return normalize_task2_medical_source("\n".join(lines))
    return (
        "肺炎主要表现为发热、咳嗽、咳痰和胸痛，重症患者可出现呼吸困难。"
        "检查可进行血常规、胸片和肺部CT，细菌性肺炎可使用抗生素治疗。"
    )


def normalize_task2_medical_source(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""
    content = value
    try:
        candidate = Path(value).expanduser()
        if candidate.exists() and candidate.is_file():
            content = candidate.read_text(encoding="utf-8")
    except OSError:
        content = value
    stripped = content.strip()
    if not stripped:
        return ""
    if stripped[0] not in "[{":
        return content
    try:
        json.loads(stripped)
    except json.JSONDecodeError:
        return content
    return content


def extract_texts_from_medical_payload(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        for key in ("records", "documents", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return extract_texts_from_medical_payload(value)
        for key in ("text", "content", "medical_text", "description"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return [value.strip()]
        return []
    if isinstance(payload, list):
        texts: list[str] = []
        for item in payload:
            texts.extend(extract_texts_from_medical_payload(item))
        return texts
    return []


def print_nexent_result(plan_result: dict, run_result: dict) -> None:
    result = run_result["result"]
    quality = result["quality"]
    plan = plan_result.get("plan", {})
    explainability = plan.get("explainability", {})
    task_understanding = explainability.get("task_understanding", {})
    steps = plan_result.get("recommended_execution", [])
    report = {}
    report_path = Path(str(result["report_path"]))
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    fallback_policy = report.get("fallback_policy") or {}
    print("MedInsight Task1 Nexent data processing agent completed.")
    print(f"Nexent runtime: {run_result.get('nexent_runtime')}")
    print(f"Planner runtime: {plan_result.get('planner_runtime')}")
    print(f"Task understanding: {task_understanding.get('canonical_task', plan_result.get('task'))}")
    print(f"DataMate connected: {report.get('datamate_integration', {}).get('connected')}")
    print(f"Fallback used: {fallback_policy.get('fallback_used')}")
    if fallback_policy.get("fallback_used"):
        print(f"Fallback reason: {fallback_policy.get('fallback_reason')}")
        missing = fallback_policy.get("missing_services") or []
        print(f"Missing services: {', '.join(missing) if missing else 'unknown'}")
    print("Operator chain:")
    for item in steps:
        print(f"  {item['step']}. {item['tool']} - {item['reason']}")
    print(f"Input records: {quality['input_records']}")
    print(f"Normalized records: {quality['normalized_records']}")
    print(f"Documents: {quality['documents']}")
    print(f"Segments: {quality['segments']}")
    print(f"Warnings: {quality['warning_count']}")
    print(f"Errors: {quality['error_count']}")
    print(f"Report: {result['report_path']}")


def print_task2_result(plan_result: dict, run_result: dict) -> None:
    result = run_result["result"]
    quality = result["quality"]
    gate = result.get("kg_quality_gate") or {}
    performance = result["performance"]
    print("MedInsight Task2 Nexent medical KG QA agent completed.")
    print(f"Nexent runtime: {run_result.get('nexent_runtime')}")
    print(f"Agent: {run_result.get('agent')}")
    print(f"Task understanding: {plan_result.get('task_understanding', {}).get('canonical_task')}")
    print("Operator chain:")
    for item in plan_result.get("recommended_execution", []):
        print(f"  {item['step']}. {item['tool']} - {item['reason']}")
    print(f"Entities: {quality['entities']}")
    print(f"Relations: {quality['relations']}")
    print(f"Valid triples: {quality['valid_triples']}")
    print(f"Invalid triples: {quality['invalid_triples']}")
    if "repaired_triples" in quality:
        print(f"Repaired triples: {quality['repaired_triples']}")
    if "conflict_count" in quality:
        print(f"Conflicts: {quality['conflict_count']}")
    print(f"Graph nodes: {quality['graph_nodes']}")
    print(f"Graph edges: {quality['graph_edges']}")
    print(f"Triple valid rate: {quality['triple_valid_rate']}")
    if gate:
        print(f"KG quality gate: {gate.get('status')} (score={gate.get('score')})")
    print(f"Records/s: {performance['records_per_second']}")
    print(f"Report: {result['report_path']}")
    print(f"Graph: {result['graph_path']}")
    print(f"Shared graph for Task3: {result['shared_graph_path']}")


def print_task2_text_kg_result(result: dict[str, Any]) -> None:
    quality = result["quality"]
    gate = result.get("kg_quality_gate") or {}
    print("MedInsight 任务二医疗文本知识图谱智能体执行完成。")
    print(f"运行方式：{result.get('nexent_runtime') or 'custom_text_kg_agent'}")
    if result.get("tool"):
        print(f"Nexent 工具：{result.get('tool')}")
    print("任务理解：用户医疗文本知识图谱生成")
    print("算子链：")
    for item in result.get("plan", {}).get("operator_chain", []):
        print(f"  {item['step']}. {item['tool']} - {item['reason']}")
    print(f"实体数量：{quality['entities']}")
    print(f"关系数量：{quality['relations']}")
    print(f"有效三元组：{quality['valid_triples']}")
    print(f"无效三元组：{quality['invalid_triples']}")
    if "repaired_triples" in quality:
        print(f"自动修复三元组：{quality['repaired_triples']}")
    if "conflict_count" in quality:
        print(f"冲突数量：{quality['conflict_count']}")
    print(f"图谱节点数：{quality['graph_nodes']}")
    print(f"图谱关系数：{quality['graph_edges']}")
    print(f"三元组有效率：{quality['triple_valid_rate']}")
    if gate:
        status_label = "通过" if gate.get("passed") else "未通过"
        print(f"图谱质量门控：{status_label}（状态={gate.get('status')}，得分={gate.get('score')}）")
        gaps = gate.get("knowledge_gaps") or []
        if gaps:
            print("知识缺口：")
            for gap in gaps[:3]:
                print(f"  - {gap.get('suggestion')}")
    performance = result.get("performance") or {}
    panel = performance.get("operator_panel") or []
    if panel:
        print("算子性能：")
        for row in panel[:8]:
            print(
                f"  - {row.get('operator')}：{row.get('latency_ms')} ms，"
                f"{row.get('throughput_records_per_second')} 条/秒"
            )
    print(f"报告文件：{result['report_path']}")
    print(f"图谱文件：{result['graph_path']}")
    print(f"待确认图谱：{result['shared_graph_path']}")
    if result.get("visualization_path"):
        print(f"可视化文件：{result['visualization_path']}")
    print_terminal_kg_preview(Path(str(result.get("graph_path") or result.get("shared_graph_path"))))


def print_terminal_kg_preview(graph_path: Path, max_nodes: int = 16, max_subjects: int = 8) -> None:
    if not graph_path.exists():
        return
    try:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return

    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    node_by_id = {str(node.get("node_id")): node for node in nodes}
    print("\n知识图谱预览")
    print("-" * 64)
    print(f"实体节点：{len(nodes)} 个")
    for node in nodes[:max_nodes]:
        print(f"- [{node.get('type')}] {node.get('name')}")
    if len(nodes) > max_nodes:
        print(f"- ... 还有 {len(nodes) - max_nodes} 个节点")

    grouped: dict[str, dict[str, list[str]]] = {}
    for edge in edges:
        source = node_by_id.get(str(edge.get("source"))) or {}
        target = node_by_id.get(str(edge.get("target"))) or {}
        source_name = str(source.get("name") or edge.get("source") or "")
        target_name = str(target.get("name") or edge.get("target") or "")
        predicate = str(edge.get("predicate") or "相关")
        if not source_name or not target_name:
            continue
        grouped.setdefault(source_name, {}).setdefault(predicate, [])
        if target_name not in grouped[source_name][predicate]:
            grouped[source_name][predicate].append(target_name)

    print(f"\n图谱关系：{len(edges)} 条")
    shown_subjects = 0
    for subject, predicates in grouped.items():
        print(f"- {subject}")
        for predicate, objects in predicates.items():
            print(f"  - {predicate} -> {'、'.join(objects[:8])}")
        shown_subjects += 1
        if shown_subjects >= max_subjects:
            break
    if len(grouped) > max_subjects:
        print(f"- ... 还有 {len(grouped) - max_subjects} 个主体")
    print("-" * 64)


def run_local_qa(graph_path: Path, question: str) -> dict[str, Any]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    return answer_question(graph, question)


def print_local_answer(answer: dict[str, Any], graph_path: Path) -> None:
    print("\nMedInsight Task2 Local QA")
    print("Runtime: local_debug_kg_query")
    print(f"Graph: {graph_path}")
    print(f"Status: {answer.get('status')}")
    print(f"问题：{answer.get('question')}")
    print("回答：")
    print(format_qa_answer_text(str(answer.get("answer") or "")))
    facts = answer.get("facts") or []
    if facts:
        print("图谱事实：")
        for fact in facts[:10]:
            print(f"- {fact.get('subject')} --{fact.get('predicate')}--> {fact.get('object')}")


def normalize_task2_qa_mode(mode: str) -> str:
    aliases = {
        "rule": "tool",
    }
    return aliases.get(mode, mode)


def format_qa_answer_text(answer_text: str) -> str:
    if not sys.stdout.isatty():
        return answer_text
    lines = answer_text.splitlines()
    formatted: list[str] = []
    in_evidence = False
    for line in lines:
        if line.startswith("证据路径"):
            in_evidence = True
        if not line:
            formatted.append(line)
            continue
        color = ANSI_EVIDENCE if in_evidence else ANSI_ANSWER
        formatted.append(f"{color}{line}{ANSI_RESET}")
    return "\n".join(formatted)


def as_tool_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "message", "tools"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                nested = as_tool_list(value)
                if nested:
                    return nested
    return []


def find_nexent_tool(tools: list[dict[str, Any]], operation_id: str) -> dict[str, Any] | None:
    for tool in tools:
        names = {
            str(tool.get("class_name") or ""),
            str(tool.get("name") or ""),
            str(tool.get("origin_name") or ""),
        }
        if operation_id in names:
            return tool
    return None


def parse_nexent_tool_result(payload: Any) -> Any:
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {"status": "ok", "raw_text": payload}
    return payload


def extract_qa_answer(payload: Any) -> dict[str, Any]:
    parsed = parse_nexent_tool_result(payload)
    if isinstance(parsed, dict):
        answer = parsed.get("answer")
        if isinstance(answer, dict):
            return answer
        result = parsed.get("result")
        if isinstance(result, dict):
            nested_answer = result.get("answer")
            if isinstance(nested_answer, dict):
                return nested_answer
    return {
        "status": "raw",
        "question": "",
        "answer": json.dumps(parsed, ensure_ascii=False) if not isinstance(parsed, str) else parsed,
        "facts": [],
    }


def extract_tool_result(payload: Any) -> dict[str, Any]:
    parsed = parse_nexent_tool_result(payload)
    if isinstance(parsed, dict):
        result = parsed.get("result")
        if isinstance(result, dict):
            return result
        nested = parsed.get("data")
        if isinstance(nested, dict):
            nested_result = nested.get("result")
            if isinstance(nested_result, dict):
                return nested_result
        return parsed
    return {"status": "raw", "raw_text": str(parsed)}


def request_json(url: str, method: str = "GET", payload: dict[str, Any] | None = None, authorization: str | None = None, timeout: int = 30) -> Any:
    headers = {"Content-Type": "application/json"}
    if authorization:
        headers["Authorization"] = authorization
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=data, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return json.loads(body) if body.strip() else {}


def format_request_exception(exc: HTTPError | URLError | RuntimeError) -> str:
    if isinstance(exc, HTTPError):
        body = exc.read().decode("utf-8", errors="replace")
        suffix = f"\n{body}" if body.strip() else ""
        return f"HTTP {exc.code} from {exc.url}{suffix}"
    return str(exc)


def run_nexent_text_kg(args: argparse.Namespace, text: str) -> dict[str, Any]:
    request_json(
        f"{args.task2_api.rstrip('/')}/health",
        authorization=args.authorization,
        timeout=min(args.timeout, 30),
    )
    tools_payload = request_json(
        f"{args.nexent_config_api.rstrip('/')}/tool/list",
        authorization=args.authorization,
        timeout=min(args.timeout, 30),
    )
    tool = find_nexent_tool(as_tool_list(tools_payload), TASK2_TEXT_KG_TOOL_NAME)
    if not tool:
        raise RuntimeError(
            f"Nexent tool `{TASK2_TEXT_KG_TOOL_NAME}` is not registered. "
            "Run `scripts/task2/create_task2_nexent_agent.sh` first."
        )
    tool_name = str(tool.get("class_name") or tool.get("name") or TASK2_TEXT_KG_TOOL_NAME)
    validate_payload = {
        "name": tool_name,
        "source": "mcp",
        "usage": "outer-apis",
        "inputs": {
            "text": text,
            "task": args.task,
            "doc_id": args.doc_id,
            "output_dir": args.output,
            "shared_output_dir": args.shared_output,
            "question": args.question,
        },
        "params": {},
    }
    raw_result = request_json(
        f"{args.nexent_config_api.rstrip('/')}/tool/validate",
        method="POST",
        payload=validate_payload,
        authorization=args.authorization,
        timeout=args.timeout,
    )
    result = extract_tool_result(raw_result)
    if result.get("result") and isinstance(result["result"], dict):
        result = result["result"]
    result.setdefault("nexent_runtime", "nexent_tool_validate")
    result.setdefault("tool", tool_name)
    return result


def reset_terminal_input() -> None:
    try:
        locale.setlocale(locale.LC_CTYPE, "")
    except locale.Error:
        pass
    if readline is not None:
        readline.parse_and_bind('"\\C-h": backward-delete-char')
        readline.parse_and_bind('"\\C-?": backward-delete-char')
        readline.parse_and_bind('"\\e[3~": delete-char')
    if not sys.stdin.isatty():
        return
    subprocess.run(["stty", "sane"], check=False)
    subprocess.run(["stty", "echo", "icanon", "isig", "iexten"], check=False)
    subprocess.run(["stty", "erase", "^?"], check=False)


def run_nexent_tool_qa(args: argparse.Namespace, question: str) -> dict[str, Any]:
    request_json(
        f"{args.task2_api.rstrip('/')}/health",
        authorization=args.authorization,
        timeout=min(args.timeout, 30),
    )
    tools_payload = request_json(
        f"{args.nexent_config_api.rstrip('/')}/tool/list",
        authorization=args.authorization,
        timeout=min(args.timeout, 30),
    )
    tool = find_nexent_tool(as_tool_list(tools_payload), TASK2_QA_TOOL_NAME)
    if not tool:
        raise RuntimeError(
            f"Nexent tool `{TASK2_QA_TOOL_NAME}` is not registered. "
            "Run `scripts/task2/create_task2_nexent_agent.sh` or "
            "`REGISTER_NEXENT=true scripts/task2/run_task2_nexent_demo.sh` first."
        )
    agent_id = resolve_nexent_agent_id(args.nexent_config_api, args.agent_name, args.authorization, args.agent_id)
    tool_name = str(tool.get("class_name") or tool.get("name") or TASK2_QA_TOOL_NAME)
    validate_payload = {
        "name": tool_name,
        "source": "mcp",
        "usage": "outer-apis",
        "inputs": {
            "graph_path": args.graph_path,
            "question": question,
        },
        "params": {},
    }
    raw_result = request_json(
        f"{args.nexent_config_api.rstrip('/')}/tool/validate",
        method="POST",
        payload=validate_payload,
        authorization=args.authorization,
        timeout=args.timeout,
    )
    parsed_result = parse_nexent_tool_result(raw_result)
    return {
        "runtime": "nexent_rule_agent_tool_validate",
        "agent_name": args.agent_name,
        "agent_id": agent_id,
        "nexent_config_api": args.nexent_config_api,
        "task2_api": args.task2_api,
        "tool": tool_name,
        "tool_id": tool.get("tool_id"),
        "question": question,
        "raw_result": parsed_result,
        "answer": extract_qa_answer(parsed_result),
    }


def print_nexent_tool_answer(result: dict[str, Any], clean: bool = False) -> None:
    answer = result["answer"]
    if clean:
        print(format_qa_answer_text(str(answer.get("answer") or "没有找到可回答的图谱结果。")))
        return
    print("\nMedInsight Task2 QA Agent - Rule Mode")
    print(f"Runtime: {result['runtime']}")
    print(f"Agent: {result['agent_name']} (agent_id={result['agent_id']})")
    print(f"Nexent config API: {result['nexent_config_api']}")
    print(f"Task2 API: {result['task2_api']}")
    print(f"Tool: {result['tool']} (tool_id={result.get('tool_id')})")
    print(f"Status: {answer.get('status')}")
    print(f"问题：{answer.get('question') or result['question']}")
    print("回答：")
    print(format_qa_answer_text(str(answer.get("answer") or "")))
    facts = answer.get("facts") or []
    if facts:
        print("图谱事实：")
        for fact in facts[:10]:
            print(f"- {fact.get('subject')} --{fact.get('predicate')}--> {fact.get('object')}")


def resolve_nexent_agent_id(config_api: str, agent_name: str, authorization: str | None, explicit_agent_id: int = 0) -> int:
    if explicit_agent_id:
        return explicit_agent_id
    payload = request_json(
        f"{config_api.rstrip('/')}/agent/by-name/{quote(agent_name)}",
        authorization=authorization,
    )
    data = payload.get("data") if isinstance(payload, dict) else payload
    for item in (data, payload):
        if isinstance(item, dict) and isinstance(item.get("agent_id"), int):
            return item["agent_id"]
    raise RuntimeError(f"Nexent agent id not found for {agent_name}. Register the Task2 agent first.")


def run_task2_qa_cli(args: argparse.Namespace) -> None:
    def answer_once(question: str) -> None:
        mode = normalize_task2_qa_mode(args.mode)
        if mode == "local":
            graph_path = Path(args.graph_path)
            if not graph_path.exists():
                raise SystemExit(f"Knowledge graph not found: {graph_path}. Run `python3 -m medinsight.cli task2` first.")
            print_local_answer(run_local_qa(graph_path, question), graph_path)
            return
        if mode == "tool":
            try:
                print_nexent_tool_answer(run_nexent_tool_qa(args, question), clean=args.clean_output)
            except (HTTPError, URLError, RuntimeError) as exc:
                raise SystemExit(f"Nexent tool QA failed: {format_request_exception(exc)}") from exc
            return
        raise SystemExit(f"unsupported Task2 QA mode: {args.mode}")

    if args.interactive:
        mode = normalize_task2_qa_mode(args.mode)
        if args.clean_output:
            print(args.title)
            print("输入 q、quit 或 exit 返回菜单。直接回车使用示例问题：肺炎有什么症状？\n")
        else:
            print(f"MedInsight Task2 QA CLI mode={mode}. 输入 exit 或 quit 退出。")
        while True:
            reset_terminal_input()
            query = input("你 > ").strip()
            if query.lower() in {"exit", "quit", "q"}:
                break
            if not query:
                query = "肺炎有什么症状？"
            answer_once(query)
        return
    if not args.question:
        raise SystemExit("--question is required unless --interactive is used.")
    answer_once(args.question)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "task1":
        input_path = Path(args.input)
        output_dir = Path(args.output)
        plan_payload = {
            "task": args.task,
            "records": load_records_for_planning(input_path),
            "planner_runtime": args.planner_runtime,
            "lora_timeout_seconds": args.lora_timeout_seconds,
        }
        if args.lora_adapter_path:
            plan_payload["lora_adapter_path"] = args.lora_adapter_path
        plan_result = build_nexent_data_processing_plan(plan_payload)
        run_result = run_nexent_task1_agent_tool(
            {
                "input_path": str(input_path),
                "output_dir": str(output_dir),
                "task": args.task,
            }
        )
        print_nexent_result(plan_result, run_result)
    elif args.command == "task2":
        plan_payload = {
            "task": args.task,
            "input_dir": args.input_dir,
            "cmeee_split": args.cmeee_split,
            "cmeie_split": args.cmeie_split,
            "cmeee_limit": args.cmeee_limit,
            "cmeie_limit": args.cmeie_limit,
            "chip_cdn_split": args.chip_cdn_split,
            "cmedcausal_split": args.cmedcausal_split,
            "chip_cdn_limit": args.chip_cdn_limit,
            "cmedcausal_limit": args.cmedcausal_limit,
            "icd10_limit": args.icd10_limit,
            "surgery_lexicon_limit": args.surgery_lexicon_limit,
            "enrichment_enabled": not args.disable_enrichment,
        }
        plan_result = build_nexent_kg_plan(plan_payload)
        run_result = run_nexent_task2_kg_agent_tool(
            {
                **plan_payload,
                "output_dir": args.output,
                "shared_output_dir": args.shared_output,
            }
        )
        print_task2_result(plan_result, run_result)
    elif args.command == "task2-text-kg":
        text = resolve_task2_text_input(args)
        if not text.strip():
            raise SystemExit("medical text is empty.")
        if args.mode == "tool":
            try:
                result = run_nexent_text_kg(args, text)
            except (HTTPError, URLError, RuntimeError) as exc:
                raise SystemExit(f"Nexent text KG generation failed: {format_request_exception(exc)}") from exc
        else:
            result = run_task2_text_kg_build(
                text=text,
                output_dir=Path(args.output),
                shared_output_dir=Path(args.shared_output),
                task=args.task,
                doc_id=args.doc_id,
                question=args.question.strip() or None,
            )
        print_task2_text_kg_result(result)
    elif args.command == "task2-qa":
        run_task2_qa_cli(args)


if __name__ == "__main__":
    main()
