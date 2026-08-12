from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from medinsight.common.schemas import Document
from medinsight.task1.agent import DataProcessingAgent
from medinsight.task1.models.lora_flow_planner import DEFAULT_LORA_ADAPTER, DEFAULT_LORA_MODEL, run_lora_flow_planner
from medinsight.task1.operators.text_processing import (
    MedicalRecordNormalizer,
    MedicalTextCleaner,
    MedicalTextLoader,
    MedicalTextSegmenter,
    RawRecordProfiler,
)
from medinsight.task1.operators.table_processing import (
    MedicalTableDocumentBuilder,
    MedicalTableNormalizer,
    MedicalTableProfiler,
)
from medinsight.task1.scheduler import build_batch_scheduler
from medinsight.task1.service import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    run_anomaly_demo,
    run_task1,
    run_task1_from_file,
    run_task1_structured_table_from_csv,
)


NEXENT_TOOL_OPERATORS = {
    RawRecordProfiler.spec.name: RawRecordProfiler,
    MedicalRecordNormalizer.spec.name: MedicalRecordNormalizer,
    MedicalTextLoader.spec.name: MedicalTextLoader,
    MedicalTextCleaner.spec.name: MedicalTextCleaner,
    MedicalTextSegmenter.spec.name: MedicalTextSegmenter,
    MedicalTableProfiler.spec.name: MedicalTableProfiler,
    MedicalTableNormalizer.spec.name: MedicalTableNormalizer,
    MedicalTableDocumentBuilder.spec.name: MedicalTableDocumentBuilder,
}


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def html_response(handler: BaseHTTPRequestHandler, status: int, html: str) -> None:
    body = html.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    if not length:
        return {}
    raw = handler.rfile.read(length).decode("utf-8")
    return json.loads(raw) if raw.strip() else {}


class MedInsightAPIHandler(BaseHTTPRequestHandler):
    server_version = "MedInsightTask1API/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/task1/planner-chat", "/task1/chat"}:
            html_response(self, 200, build_task1_planner_chat_html_v2())
            return
        if parsed.path == "/health":
            json_response(self, 200, {"status": "ok", "service": "medinsight-task1-api"})
            return
        if parsed.path == "/api/task1/report":
            params = parse_qs(parsed.query)
            report_path = Path(params.get("path", [str(DEFAULT_OUTPUT / "data_processing_report.json")])[0])
            if not report_path.exists():
                json_response(self, 404, {"status": "error", "message": f"report not found: {report_path}"})
                return
            json_response(self, 200, json.loads(report_path.read_text(encoding="utf-8")))
            return
        if parsed.path in {"/openapi.json", "/api/nexent/openapi.json"}:
            params = parse_qs(parsed.query)
            server_url = params.get("server_url", ["http://host.docker.internal:18080"])[0]
            json_response(self, 200, build_nexent_openapi_spec(server_url))
            return
        if parsed.path == "/api/nexent/tools":
            json_response(self, 200, {"status": "ok", "tools": build_nexent_tool_catalog()})
            return
        json_response(self, 404, {"status": "error", "message": f"unknown endpoint: {parsed.path}"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = read_json_body(self)
            if parsed.path == "/api/task1/run":
                result = run_task1_payload(payload)
                json_response(self, 200, result)
                return
            if parsed.path == "/api/nexent/task1/run":
                result = run_nexent_task1_agent_tool(payload)
                json_response(self, 200, result)
                return
            if parsed.path == "/api/nexent/task1/plan":
                result = build_nexent_data_processing_plan(payload)
                json_response(self, 200, result)
                return
            if parsed.path.startswith("/api/nexent/tools/"):
                tool_name = parsed.path.rsplit("/", 1)[-1]
                result = run_nexent_operator_tool(tool_name, payload)
                json_response(self, 200, result)
                return
            if parsed.path == "/api/task1/anomaly-demo":
                output_dir = Path(payload.get("output_dir", "artifacts/task1/anomaly_demo"))
                result = run_anomaly_demo(output_dir)
                json_response(self, 200, result)
                return
            json_response(self, 404, {"status": "error", "message": f"unknown endpoint: {parsed.path}"})
        except Exception as exc:
            json_response(self, 500, {"status": "error", "message": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        return


def run_server(host: str = "127.0.0.1", port: int = 18080) -> None:
    server = ThreadingHTTPServer((host, port), MedInsightAPIHandler)
    print(f"MedInsight Task1 API running at http://{host}:{port}")
    print("Health: POST/GET examples are documented in docs/task1_api_demo.md")
    server.serve_forever()


def build_nexent_tool_catalog() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "description": operator_class.spec.description,
            "input_schema": operator_class.spec.input_schema,
            "output_schema": operator_class.spec.output_schema,
            "nexent_endpoint": f"/api/nexent/tools/{name}",
        }
        for name, operator_class in NEXENT_TOOL_OPERATORS.items()
    ]


def run_task1_payload(payload: dict[str, Any]) -> dict[str, Any]:
    task = str(payload.get("task") or "清洗医疗数据，完成数据画像、规范化、文本清洗和句子切分。")
    output_dir = Path(payload.get("output_dir", DEFAULT_OUTPUT))
    scheduler_enabled = bool(payload.get("scheduler_enabled", True))
    if isinstance(payload.get("records"), list):
        return run_task1(payload["records"], task, output_dir, scheduler_enabled=scheduler_enabled)
    input_path = Path(payload.get("input_path", DEFAULT_INPUT))
    if input_path.suffix.lower() == ".csv":
        return run_task1_structured_table_from_csv(input_path, task, output_dir, scheduler_enabled=scheduler_enabled)
    return run_task1_from_file(input_path, task, output_dir, scheduler_enabled=scheduler_enabled)


def run_nexent_task1_agent_tool(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute the same entry tool that Nexent imports through OpenAPI."""
    result = run_task1_payload(payload)
    enrich_task1_report_with_nexent_context(result)
    return {
        "status": "ok",
        "nexent_runtime": "outer_api_tool",
        "tool": "medinsight_run_data_processing_agent",
        "result": result,
    }


def build_task1_planner_chat_html() -> str:
    examples = {
        "text": {
            "task": "请清洗公开医疗文本数据，完成数据画像、文本去噪和句子切分，并输出可执行算子链。",
            "records": [
                {
                    "doc_id": "demo-text-1",
                    "title": "糖尿病",
                    "text": "糖尿病常见症状包括多饮、多尿、多食和消瘦。诊断检查可参考血糖、尿葡萄糖和血脂。",
                    "source_dataset": "QASystemOnMedicalGraph/data/disease.csv",
                }
            ],
        },
        "table": {
            "task": "请处理医疗 CSV 表格，检查字段缺失，统一字段名，并生成标准化文本片段。",
            "records": [
                {
                    "disease": "糖尿病",
                    "symptom": "多饮 多尿 多食 消瘦",
                    "drug": "二甲双胍",
                    "checklist": "血糖 尿葡萄糖 血脂",
                    "treatment": "药物治疗 饮食疗法 运动治疗",
                    "department": "内分泌科",
                    "complication": "糖尿病肾病 糖尿病足",
                }
            ],
        },
    }
    payload = json.dumps(examples, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>任务一规划对话框 | MedInsight</title>
  <style>
    :root {{
      --ink: #17212b;
      --muted: #657282;
      --line: #d9e1e8;
      --paper: #f6f8fa;
      --panel: #ffffff;
      --blue: #3867d6;
      --teal: #0f8b8d;
      --green: #2f9e44;
      --gold: #b7791f;
      --red: #c92a2a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      font-size: 15px;
      line-height: 1.45;
    }}
    header {{
      background: #fff;
      border-bottom: 1px solid var(--line);
      padding: 18px 24px;
    }}
    h1 {{
      margin: 0 0 5px;
      font-size: 22px;
      letter-spacing: 0;
    }}
    .subhead {{ color: var(--muted); max-width: 980px; }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 18px;
      display: grid;
      grid-template-columns: minmax(320px, 420px) 1fr;
      gap: 14px;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-width: 0;
    }}
    .composer {{ padding: 16px; }}
    h2 {{ margin: 0 0 12px; font-size: 16px; letter-spacing: 0; }}
    label {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin: 12px 0 6px;
    }}
    textarea, select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      padding: 10px;
      font: inherit;
      outline: none;
    }}
    textarea:focus, select:focus {{
      border-color: var(--blue);
      box-shadow: 0 0 0 3px rgba(56, 103, 214, .12);
    }}
    textarea.task {{ min-height: 118px; resize: vertical; }}
    textarea.records {{
      min-height: 230px;
      resize: vertical;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
      line-height: 1.5;
    }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }}
    button {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      min-height: 36px;
      padding: 8px 12px;
      font: inherit;
      cursor: pointer;
    }}
    button.primary {{
      border-color: var(--blue);
      background: var(--blue);
      color: #fff;
      font-weight: 650;
    }}
    button:disabled {{ opacity: .6; cursor: wait; }}
    .conversation {{
      display: grid;
      grid-template-rows: auto 1fr;
      min-height: 720px;
    }}
    .summary {{
      padding: 16px;
      border-bottom: 1px solid var(--line);
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
    }}
    .metric {{
      border: 1px solid #e7edf3;
      border-radius: 8px;
      padding: 10px;
      background: #fbfdff;
      min-height: 74px;
    }}
    .metric .label {{ color: var(--muted); font-size: 12px; }}
    .metric .value {{ font-weight: 720; font-size: 18px; margin-top: 4px; word-break: break-word; }}
    .output {{
      padding: 16px;
      display: grid;
      gap: 14px;
      align-content: start;
    }}
    .dialog {{
      display: grid;
      gap: 10px;
    }}
    .bubble {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }}
    .bubble.user {{
      border-left: 4px solid var(--blue);
      background: #fbfdff;
    }}
    .bubble.agent {{
      border-left: 4px solid var(--teal);
    }}
    .title-row {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      margin-bottom: 8px;
    }}
    .role {{ font-weight: 700; }}
    .status {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 9px;
      background: #f8fafb;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    .dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--green);
    }}
    .dot.warn {{ background: var(--gold); }}
    .dot.bad {{ background: var(--red); }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .chip {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 10px;
      background: #f8fafb;
      font-size: 13px;
    }}
    .steps {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
    }}
    .step {{
      border: 1px solid var(--line);
      border-left: 4px solid var(--blue);
      border-radius: 8px;
      padding: 10px;
      min-height: 96px;
      background: #fbfdff;
    }}
    .step .idx {{ color: var(--blue); font-weight: 700; font-size: 12px; }}
    .step .name {{ font-weight: 700; margin: 4px 0; word-break: break-word; }}
    .step .why {{ color: var(--muted); font-size: 12px; }}
    .json {{
      margin: 0;
      max-height: 300px;
      overflow: auto;
      background: #101820;
      color: #e6edf3;
      border-radius: 8px;
      padding: 12px;
      font-size: 12px;
      line-height: 1.5;
    }}
    .empty {{
      color: var(--muted);
      padding: 40px 16px;
      text-align: center;
    }}
    @media (max-width: 920px) {{
      main {{ grid-template-columns: 1fr; padding: 12px; }}
      .conversation {{ min-height: auto; }}
      .summary {{ grid-template-columns: 1fr 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>任务一规划对话框</h1>
    <div class="subhead">输入自然语言数据处理任务，Agent 会先做数据画像，再输出任务理解、规划器类型、推荐算子链和调度理由。</div>
  </header>
  <main>
    <section class="composer">
      <h2>输入</h2>
      <label for="sample">样例数据</label>
      <select id="sample">
        <option value="text">医疗文本 JSONL 样例</option>
        <option value="table">结构化医疗表格样例</option>
      </select>
      <label for="task">自然语言任务</label>
      <textarea id="task" class="task"></textarea>
      <label for="runtime">规划器</label>
      <select id="runtime">
        <option value="deterministic">稳定规划器</option>
        <option value="lora">LoRA 小模型规划器</option>
      </select>
      <label for="records">输入记录 JSON 数组</label>
      <textarea id="records" class="records" spellcheck="false"></textarea>
      <div class="actions">
        <button class="primary" id="send">发送任务</button>
        <button id="textExample">文本样例</button>
        <button id="tableExample">表格样例</button>
      </div>
    </section>
    <section class="conversation">
      <div class="summary" id="summary">
        <div class="metric"><div class="label">数据模式</div><div class="value">—</div></div>
        <div class="metric"><div class="label">任务意图</div><div class="value">—</div></div>
        <div class="metric"><div class="label">算子数量</div><div class="value">—</div></div>
        <div class="metric"><div class="label">规划器</div><div class="value">—</div></div>
      </div>
      <div class="output" id="output"><div class="empty">等待输入任务。</div></div>
    </section>
  </main>
  <script id="examples" type="application/json">{payload}</script>
  <script>
    const examples = JSON.parse(document.getElementById('examples').textContent);
    const taskEl = document.getElementById('task');
    const recordsEl = document.getElementById('records');
    const sampleEl = document.getElementById('sample');
    const runtimeEl = document.getElementById('runtime');
    const sendEl = document.getElementById('send');
    const outputEl = document.getElementById('output');
    const summaryEl = document.getElementById('summary');
    const esc = value => String(value ?? '—').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));

    function loadExample(name) {{
      const item = examples[name];
      sampleEl.value = name;
      taskEl.value = item.task;
      recordsEl.value = JSON.stringify(item.records, null, 2);
    }}

    function activeIntents(intent) {{
      return Object.entries(intent || {{}})
        .filter(([, value]) => value === true)
        .map(([key]) => key);
    }}

    function setSummary(result) {{
      const profile = result.profile || {{}};
      const steps = result.recommended_execution || [];
      const intents = activeIntents(result.plan?.intent || {{}}).join(' / ') || '—';
      summaryEl.innerHTML = `
        <div class="metric"><div class="label">数据模式</div><div class="value">${{esc(profile.detected_mode)}}</div></div>
        <div class="metric"><div class="label">任务意图</div><div class="value">${{esc(intents)}}</div></div>
        <div class="metric"><div class="label">算子数量</div><div class="value">${{steps.length}}</div></div>
        <div class="metric"><div class="label">规划器</div><div class="value">${{esc(result.planner_runtime)}}</div></div>
      `;
    }}

    function render(result, task, records) {{
      const profile = result.profile || {{}};
      const intent = result.plan?.intent || {{}};
      const steps = result.recommended_execution || [];
      const intentChips = Object.entries(intent).map(([key, value]) =>
        `<span class="chip">${{esc(key)}}: ${{value ? 'true' : 'false'}}</span>`
      ).join('');
      const profileChips = [
        ['record_count', profile.record_count],
        ['field_count', profile.field_count],
        ['text_coverage', profile.text_coverage],
        ['structured_coverage', profile.structured_field_coverage],
        ['detected_mode', profile.detected_mode],
      ].map(([key, value]) => `<span class="chip">${{esc(key)}}: ${{esc(value)}}</span>`).join('');
      const stepHtml = steps.map(step => `
        <div class="step">
          <div class="idx">STEP ${{esc(step.step)}}</div>
          <div class="name">${{esc(step.tool)}}</div>
          <div class="why">${{esc(step.reason)}}</div>
        </div>
      `).join('');
      outputEl.innerHTML = `
        <div class="dialog">
          <div class="bubble user">
            <div class="title-row"><span class="role">用户任务</span><span class="status"><span class="dot"></span>${{records.length}} 条输入记录</span></div>
            <div>${{esc(task)}}</div>
          </div>
          <div class="bubble agent">
            <div class="title-row"><span class="role">任务理解</span><span class="status"><span class="dot"></span>${{esc(result.tool)}}</span></div>
            <div class="chips">${{intentChips}}</div>
          </div>
          <div class="bubble agent">
            <div class="title-row"><span class="role">数据画像</span><span class="status"><span class="dot"></span>${{esc(profile.detected_mode)}}</span></div>
            <div class="chips">${{profileChips}}</div>
          </div>
          <div class="bubble agent">
            <div class="title-row"><span class="role">自主规划算子链</span><span class="status"><span class="dot"></span>${{esc(result.planner_runtime)}}</span></div>
            <div class="steps">${{stepHtml}}</div>
          </div>
          <div class="bubble agent">
            <div class="title-row"><span class="role">原始返回</span><span class="status"><span class="dot"></span>JSON</span></div>
            <pre class="json">${{esc(JSON.stringify(result, null, 2))}}</pre>
          </div>
        </div>
      `;
    }}

    async function send() {{
      const task = taskEl.value.trim();
      let records;
      try {{
        records = JSON.parse(recordsEl.value);
        if (!Array.isArray(records)) throw new Error('输入记录必须是 JSON 数组');
      }} catch (error) {{
        outputEl.innerHTML = `<div class="bubble agent"><div class="title-row"><span class="role">输入错误</span><span class="status"><span class="dot bad"></span>JSON</span></div>${{esc(error.message)}}</div>`;
        return;
      }}
      sendEl.disabled = true;
      sendEl.textContent = '规划中';
      outputEl.innerHTML = '<div class="empty">正在调用任务一规划工具。</div>';
      try {{
        const response = await fetch('/api/nexent/task1/plan', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            task,
            records,
            planner_runtime: runtimeEl.value,
            lora_timeout_seconds: 120,
          }}),
        }});
        const result = await response.json();
        if (!response.ok || result.status !== 'ok') {{
          throw new Error(result.message || `HTTP ${{response.status}}`);
        }}
        setSummary(result);
        render(result, task, records);
      }} catch (error) {{
        outputEl.innerHTML = `<div class="bubble agent"><div class="title-row"><span class="role">调用失败</span><span class="status"><span class="dot bad"></span>error</span></div>${{esc(error.message)}}</div>`;
      }} finally {{
        sendEl.disabled = false;
        sendEl.textContent = '发送任务';
      }}
    }}

    sampleEl.addEventListener('change', () => loadExample(sampleEl.value));
    document.getElementById('textExample').addEventListener('click', () => loadExample('text'));
    document.getElementById('tableExample').addEventListener('click', () => loadExample('table'));
    sendEl.addEventListener('click', send);
    loadExample('text');
  </script>
</body>
</html>"""


def build_task1_planner_chat_html_v2() -> str:
    examples = {
        "text": {
            "label": "医疗文本",
            "task": "请清洗公开医疗文本数据，完成数据画像、文本去噪和句子切分，并输出可执行算子链。",
            "records": [
                {
                    "doc_id": "demo-text-1",
                    "title": "糖尿病",
                    "text": "糖尿病常见症状包括多饮、多尿、多食和消瘦。诊断检查可参考血糖、尿葡萄糖和血脂。",
                    "source_dataset": "QASystemOnMedicalGraph/data/disease.csv",
                }
            ],
        },
        "table": {
            "label": "结构化表",
            "task": "请处理医疗 CSV 表格，检查字段缺失，统一字段名，并生成标准化文本片段。",
            "records": [
                {
                    "disease": "糖尿病",
                    "symptom": "多饮 多尿 多食 消瘦",
                    "drug": "二甲双胍",
                    "checklist": "血糖 尿葡萄糖 血脂",
                    "treatment": "药物治疗 饮食疗法 运动治疗",
                    "department": "内分泌科",
                    "complication": "糖尿病肾病 糖尿病足",
                }
            ],
        },
    }
    payload = json.dumps(examples, ensure_ascii=False).replace("</", "<\\/")
    html = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>任务一智能规划演示 | MedInsight</title>
  <style>
    :root {
      --ink: #17212b;
      --muted: #667789;
      --line: #d8e0e8;
      --paper: #f5f7fa;
      --panel: #ffffff;
      --blue: #2f5fd0;
      --teal: #0f8b8d;
      --green: #2f9e44;
      --gold: #b7791f;
      --red: #c92a2a;
      --soft-blue: #edf4ff;
      --soft-teal: #eaf7f6;
      --soft-gold: #fff7e6;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      font-size: 15px;
      line-height: 1.45;
    }
    header {
      background: #fff;
      border-bottom: 1px solid var(--line);
      padding: 18px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }
    h1 { margin: 0; font-size: 22px; letter-spacing: 0; }
    .subtitle { color: var(--muted); margin-top: 4px; }
    .top-badge {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 12px;
      color: var(--muted);
      white-space: nowrap;
      background: #fbfdff;
      font-size: 13px;
    }
    main {
      max-width: 1360px;
      margin: 0 auto;
      padding: 18px;
      display: grid;
      grid-template-columns: 430px 1fr;
      gap: 16px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-width: 0;
    }
    .input-panel { padding: 16px; }
    h2 { margin: 0 0 12px; font-size: 16px; letter-spacing: 0; }
    .segmented {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      padding: 4px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f8fafc;
    }
    .segmented button {
      border: 0;
      background: transparent;
      border-radius: 6px;
      min-height: 34px;
      color: var(--muted);
      cursor: pointer;
      font: inherit;
    }
    .segmented button.active {
      background: #fff;
      color: var(--ink);
      box-shadow: 0 1px 4px rgba(23,33,43,.12);
      font-weight: 700;
    }
    label {
      display: block;
      margin: 14px 0 6px;
      color: var(--muted);
      font-size: 13px;
    }
    textarea, select, input[type="file"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      color: var(--ink);
      background: #fff;
      font: inherit;
      outline: none;
    }
    textarea:focus, select:focus, input[type="file"]:focus {
      border-color: var(--blue);
      box-shadow: 0 0 0 3px rgba(47,95,208,.12);
    }
    .task-input { min-height: 128px; resize: vertical; }
    .records-input {
      min-height: 150px;
      resize: vertical;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
      line-height: 1.5;
    }
    .file-drop {
      border: 1px dashed #b9c7d6;
      border-radius: 8px;
      background: #fbfdff;
      padding: 12px;
    }
    .file-meta {
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      min-height: 18px;
      overflow-wrap: anywhere;
    }
    .actions {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      margin-top: 12px;
    }
    .primary, .ghost {
      border-radius: 8px;
      min-height: 38px;
      padding: 8px 12px;
      cursor: pointer;
      font: inherit;
    }
    .primary {
      border: 1px solid var(--blue);
      background: var(--blue);
      color: #fff;
      font-weight: 700;
    }
    .ghost {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
    }
    .primary:disabled { opacity: .65; cursor: wait; }
    .workspace {
      display: grid;
      grid-template-rows: auto auto 1fr;
      gap: 14px;
      min-width: 0;
    }
    .scoreboard {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
      padding: 14px;
    }
    .metric {
      border: 1px solid #e7edf3;
      border-radius: 8px;
      background: #fbfdff;
      padding: 11px;
      min-height: 78px;
    }
    .metric .label { color: var(--muted); font-size: 12px; }
    .metric .value { margin-top: 5px; font-size: 18px; font-weight: 800; word-break: break-word; }
    .reason-grid {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 10px;
      padding: 0 14px 14px;
    }
    .reason-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
      min-height: 118px;
      min-width: 0;
    }
    .reason-card strong { display: block; margin-bottom: 8px; }
    .chips { display: flex; flex-wrap: wrap; gap: 7px; }
    .chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #f8fafc;
      padding: 5px 9px;
      font-size: 12px;
      color: var(--ink);
      max-width: 100%;
      overflow-wrap: anywhere;
    }
    .chip.blue { border-color: #cbdafa; background: var(--soft-blue); }
    .chip.teal { border-color: #c7e7e3; background: var(--soft-teal); }
    .chip.gold { border-color: #f2d59c; background: var(--soft-gold); }
    .chat {
      padding: 14px;
      display: grid;
      gap: 12px;
      align-content: start;
    }
    .bubble {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 13px;
      min-width: 0;
    }
    .bubble.user { border-left: 4px solid var(--blue); background: #fbfdff; }
    .bubble.agent { border-left: 4px solid var(--teal); }
    .bubble.evidence { border-left: 4px solid var(--gold); }
    .bubble-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
    }
    .role { font-weight: 800; }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      border: 1px solid var(--line);
      background: #f8fafc;
      border-radius: 999px;
      padding: 4px 9px;
      white-space: nowrap;
    }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); }
    .dot.warn { background: var(--gold); }
    .dot.bad { background: var(--red); }
    .pipeline {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
      gap: 10px;
    }
    .step {
      border: 1px solid var(--line);
      border-left: 4px solid var(--blue);
      border-radius: 8px;
      padding: 11px;
      background: #fbfdff;
      min-height: 108px;
    }
    .step .idx { color: var(--blue); font-size: 12px; font-weight: 800; }
    .step .name { margin: 5px 0; font-weight: 800; word-break: break-word; }
    .step .why { color: var(--muted); font-size: 12px; }
    .compare {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
      gap: 10px;
      align-items: stretch;
      min-width: 0;
    }
    .compare-box {
      border: 1px solid #e7edf3;
      border-radius: 8px;
      padding: 10px;
      background: #fbfdff;
      min-height: 86px;
      min-width: 0;
    }
    .arrow {
      display: grid;
      place-items: center;
      color: var(--muted);
      font-weight: 800;
    }
    pre {
      margin: 0;
      max-height: 320px;
      overflow: auto;
      border-radius: 8px;
      background: #101820;
      color: #e6edf3;
      padding: 12px;
      font-size: 12px;
      line-height: 1.5;
    }
    .empty {
      padding: 56px 18px;
      text-align: center;
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: #fbfdff;
    }
    @media (max-width: 980px) {
      header { display: block; }
      .top-badge { display: inline-flex; margin-top: 10px; }
      main { grid-template-columns: 1fr; padding: 12px; }
      .scoreboard, .reason-grid { grid-template-columns: 1fr 1fr; }
      .compare { grid-template-columns: 1fr; }
      .arrow { min-height: 20px; }
    }
    @media (max-width: 620px) {
      .scoreboard, .reason-grid { grid-template-columns: 1fr; }
      .actions { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>任务一智能规划演示</h1>
      <div class="subtitle">把自然语言任务转成数据画像、意图识别和可执行算子链。</div>
    </div>
    <div class="top-badge">Nexent Tool: medinsight_plan_data_processing_flow</div>
  </header>
  <main>
    <section class="panel input-panel">
      <h2>输入任务</h2>
      <div class="segmented">
        <button id="textMode" class="active" type="button">医疗文本</button>
        <button id="tableMode" type="button">结构化表格</button>
      </div>
      <label for="task">自然语言任务</label>
      <textarea id="task" class="task-input"></textarea>
      <label for="runtime">规划器</label>
      <select id="runtime">
        <option value="deterministic">可复现规划器</option>
        <option value="lora">LoRA 小模型规划器</option>
      </select>
      <label for="fileInput">上传数据文件</label>
      <div class="file-drop">
        <input id="fileInput" type="file" accept=".csv,.txt,.json,.jsonl,text/plain,application/json,text/csv" />
        <div class="file-meta" id="fileMeta">可上传 CSV、TXT、JSON 或 JSONL；未上传时使用当前样例。</div>
      </div>
      <label for="records">解析预览</label>
      <textarea id="records" class="records-input" spellcheck="false"></textarea>
      <div class="actions">
        <button id="send" class="primary" type="button">生成智能体规划</button>
        <button id="reset" class="ghost" type="button">重置样例</button>
      </div>
    </section>

    <section class="workspace">
      <section class="panel scoreboard" id="scoreboard">
        <div class="metric"><div class="label">数据模式</div><div class="value">等待输入</div></div>
        <div class="metric"><div class="label">任务意图</div><div class="value">—</div></div>
        <div class="metric"><div class="label">算子数量</div><div class="value">—</div></div>
        <div class="metric"><div class="label">规划器</div><div class="value">—</div></div>
      </section>
      <section class="panel reason-grid" id="reasonGrid">
        <div class="reason-card"><strong>输入影响</strong><div class="chips"><span class="chip">等待任务</span></div></div>
        <div class="reason-card"><strong>数据证据</strong><div class="chips"><span class="chip">等待画像</span></div></div>
        <div class="reason-card"><strong>选择依据</strong><div class="chips"><span class="chip">等待规划</span></div></div>
      </section>
      <section class="panel chat" id="chat">
        <div class="empty">输入任务后，这里会展示“用户输入如何影响规划结果”。</div>
      </section>
    </section>
  </main>

  <script id="examples" type="application/json">__EXAMPLES__</script>
  <script>
    const examples = JSON.parse(document.getElementById('examples').textContent);
    const taskEl = document.getElementById('task');
    const recordsEl = document.getElementById('records');
    const runtimeEl = document.getElementById('runtime');
    const fileInputEl = document.getElementById('fileInput');
    const fileMetaEl = document.getElementById('fileMeta');
    const sendEl = document.getElementById('send');
    const resetEl = document.getElementById('reset');
    const chatEl = document.getElementById('chat');
    const scoreboardEl = document.getElementById('scoreboard');
    const reasonGridEl = document.getElementById('reasonGrid');
    const textModeEl = document.getElementById('textMode');
    const tableModeEl = document.getElementById('tableMode');
    let activeMode = 'text';
    let currentFileName = '内置样例';

    const esc = value => String(value ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const keywords = ['清洗','去噪','切分','CSV','表格','缺失','统一字段','规范化','抽取','转换','质量','文本','医疗'];

    function setMode(mode) {
      activeMode = mode;
      textModeEl.classList.toggle('active', mode === 'text');
      tableModeEl.classList.toggle('active', mode === 'table');
      taskEl.value = examples[mode].task;
      recordsEl.value = JSON.stringify(examples[mode].records, null, 2);
      currentFileName = '内置样例';
      fileInputEl.value = '';
      fileMetaEl.textContent = mode === 'table' ? '可上传 CSV 文件；未上传时使用结构化表样例。' : '可上传 TXT、JSONL 或 JSON 文件；未上传时使用医疗文本样例。';
    }

    function parseCsv(text) {
      const rows = [];
      let row = [];
      let cell = '';
      let quoted = false;
      for (let i = 0; i < text.length; i += 1) {
        const ch = text[i];
        const next = text[i + 1];
        if (ch === '"' && quoted && next === '"') {
          cell += '"';
          i += 1;
        } else if (ch === '"') {
          quoted = !quoted;
        } else if (ch === ',' && !quoted) {
          row.push(cell);
          cell = '';
        } else if ((ch === '\\n' || ch === '\\r') && !quoted) {
          if (ch === '\\r' && next === '\\n') i += 1;
          row.push(cell);
          if (row.some(value => value.trim() !== '')) rows.push(row);
          row = [];
          cell = '';
        } else {
          cell += ch;
        }
      }
      row.push(cell);
      if (row.some(value => value.trim() !== '')) rows.push(row);
      if (rows.length < 2) return [];
      const headers = rows[0].map(item => item.trim());
      return rows.slice(1).map(values => {
        const item = {};
        headers.forEach((header, index) => { item[header || `field_${index + 1}`] = (values[index] || '').trim(); });
        return item;
      });
    }

    function parseJsonLike(text) {
      const trimmed = text.trim();
      if (!trimmed) return [];
      if (trimmed.startsWith('[')) return JSON.parse(trimmed);
      if (trimmed.startsWith('{')) return [JSON.parse(trimmed)];
      return trimmed.split(/\\n+/).map(line => line.trim()).filter(Boolean).map(line => JSON.parse(line));
    }

    function parseTextFile(text, fileName) {
      const chunks = text.split(/\\n\\s*\\n|\\n/).map(item => item.trim()).filter(Boolean);
      const usable = chunks.length ? chunks : [text.trim()].filter(Boolean);
      return usable.map((content, index) => ({
        doc_id: `${fileName || 'uploaded-text'}-${index + 1}`,
        title: fileName || `uploaded-text-${index + 1}`,
        text: content,
        source_dataset: 'user_uploaded_file'
      }));
    }

    async function handleFileUpload(file) {
      if (!file) return;
      const text = await file.text();
      const lower = file.name.toLowerCase();
      let records;
      if (lower.endsWith('.csv')) {
        records = parseCsv(text);
        activeMode = 'table';
        textModeEl.classList.remove('active');
        tableModeEl.classList.add('active');
      } else if (lower.endsWith('.json') || lower.endsWith('.jsonl')) {
        records = parseJsonLike(text);
      } else {
        records = parseTextFile(text, file.name);
        activeMode = 'text';
        textModeEl.classList.add('active');
        tableModeEl.classList.remove('active');
      }
      if (!Array.isArray(records) || records.length === 0) throw new Error('文件没有解析出有效记录');
      currentFileName = file.name;
      recordsEl.value = JSON.stringify(records, null, 2);
      fileMetaEl.textContent = `${file.name}，已解析 ${records.length} 条记录`;
    }

    function activeIntents(intent) {
      return Object.entries(intent || {}).filter(([, value]) => value === true).map(([key]) => key);
    }

    function taskKeywords(task) {
      return keywords.filter(word => task.includes(word));
    }

    function fieldNames(records) {
      const names = new Set();
      for (const row of records) Object.keys(row || {}).forEach(key => names.add(key));
      return [...names];
    }

    function chips(items, cls = '') {
      const list = items.length ? items : ['—'];
      return list.map(item => `<span class="chip ${cls}">${esc(item)}</span>`).join('');
    }

    function setScoreboard(result) {
      const profile = result.profile || {};
      const intents = activeIntents(result.plan?.intent || {});
      const steps = result.recommended_execution || [];
      scoreboardEl.innerHTML = `
        <div class="metric"><div class="label">数据模式</div><div class="value">${esc(profile.detected_mode)}</div></div>
        <div class="metric"><div class="label">任务意图</div><div class="value">${esc(intents.join(' / ') || '—')}</div></div>
        <div class="metric"><div class="label">算子数量</div><div class="value">${steps.length}</div></div>
        <div class="metric"><div class="label">规划器</div><div class="value">${esc(result.planner_runtime)}</div></div>
      `;
    }

    function setReasonGrid(result, task, records) {
      const profile = result.profile || {};
      const steps = result.recommended_execution || [];
      const selected = steps.map(item => item.tool);
      const explain = result.plan?.explainability || {};
      const taskInfo = explain.task_understanding || {};
      const evidence = explain.data_evidence || {};
      const operatorEvidence = (explain.operator_selection || []).map(item => item.evidence || item.reason || item.tool);
      reasonGridEl.innerHTML = `
        <div class="reason-card">
          <strong>输入影响</strong>
          <div class="chips">${chips([taskInfo.canonical_task, ...(taskInfo.active_intents || taskKeywords(task))].filter(Boolean), 'blue')}</div>
        </div>
        <div class="reason-card">
          <strong>数据证据</strong>
          <div class="chips">
            ${chips([
              `字段 ${profile.field_count ?? 0}`,
              `文本覆盖 ${evidence.text_coverage ?? profile.text_coverage ?? 0}`,
              `结构化覆盖 ${evidence.structured_field_coverage ?? profile.structured_field_coverage ?? 0}`,
              `记录 ${evidence.record_count ?? profile.record_count ?? records.length}`
            ], 'teal')}
          </div>
        </div>
        <div class="reason-card">
          <strong>选择依据</strong>
          <div class="chips">${chips(operatorEvidence.length ? operatorEvidence.slice(0, 4) : selected, 'gold')}</div>
        </div>
      `;
    }

    function render(result, task, records) {
      const profile = result.profile || {};
      const intent = result.plan?.intent || {};
      const steps = result.recommended_execution || [];
      const inputFields = fieldNames(records);
      const keywordList = taskKeywords(task);
      const stepHtml = steps.map(item => `
        <div class="step">
          <div class="idx">STEP ${esc(item.step)}</div>
          <div class="name">${esc(item.tool)}</div>
          <div class="why">${esc(item.reason)}</div>
        </div>
      `).join('');
      chatEl.innerHTML = `
        <div class="bubble user">
          <div class="bubble-title"><span class="role">用户输入</span><span class="status"><span class="dot"></span>${esc(currentFileName)} · ${records.length} 条记录</span></div>
          <div>${esc(task)}</div>
          <div class="chips" style="margin-top:10px">${chips(keywordList, 'blue')}</div>
        </div>
        <div class="bubble agent">
          <div class="bubble-title"><span class="role">系统如何理解这句话</span><span class="status"><span class="dot"></span>intent</span></div>
          <div class="compare">
            <div class="compare-box"><strong>识别到的字段</strong><div class="chips" style="margin-top:8px">${chips(inputFields, 'teal')}</div></div>
            <div class="arrow">→</div>
            <div class="compare-box"><strong>识别到的意图</strong><div class="chips" style="margin-top:8px">${chips(Object.entries(intent).map(([k,v]) => `${k}: ${v}`), 'blue')}</div></div>
          </div>
        </div>
        <div class="bubble evidence">
          <div class="bubble-title"><span class="role">数据画像决定流程分支</span><span class="status"><span class="dot"></span>${esc(profile.detected_mode)}</span></div>
          <div class="compare">
            <div class="compare-box"><strong>画像结果</strong><div class="chips" style="margin-top:8px">${chips([
              `record_count=${profile.record_count}`,
              `field_count=${profile.field_count}`,
              `text_coverage=${profile.text_coverage}`,
              `structured_coverage=${profile.structured_field_coverage}`
            ], 'teal')}</div></div>
            <div class="arrow">→</div>
            <div class="compare-box"><strong>流程分支</strong><div style="margin-top:8px">${esc(profile.detected_mode === 'structured' ? '结构化表格处理链路' : '医疗文本处理链路')}</div></div>
          </div>
        </div>
        <div class="bubble agent">
          <div class="bubble-title"><span class="role">自主规划算子链</span><span class="status"><span class="dot"></span>${esc(result.planner_runtime)}</span></div>
          <div class="pipeline">${stepHtml}</div>
        </div>
        <div class="bubble evidence">
          <div class="bubble-title"><span class="role">原始 JSON 证据</span><span class="status"><span class="dot"></span>API response</span></div>
          <pre>${esc(JSON.stringify(result, null, 2))}</pre>
        </div>
      `;
    }

    async function submitTask() {
      const task = taskEl.value.trim();
      let records;
      try {
        records = JSON.parse(recordsEl.value);
        if (!Array.isArray(records)) throw new Error('输入记录必须是 JSON 数组');
      } catch (error) {
        chatEl.innerHTML = `<div class="bubble evidence"><div class="bubble-title"><span class="role">输入错误</span><span class="status"><span class="dot bad"></span>JSON</span></div>${esc(error.message)}</div>`;
        return;
      }
      sendEl.disabled = true;
      sendEl.textContent = '规划中';
      chatEl.innerHTML = '<div class="empty">正在调用真实规划接口。</div>';
      try {
        const response = await fetch('/api/nexent/task1/plan', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ task, records, planner_runtime: runtimeEl.value, lora_timeout_seconds: 120 })
        });
        const result = await response.json();
        if (!response.ok || result.status !== 'ok') throw new Error(result.message || `HTTP ${response.status}`);
        setScoreboard(result);
        setReasonGrid(result, task, records);
        render(result, task, records);
      } catch (error) {
        chatEl.innerHTML = `<div class="bubble evidence"><div class="bubble-title"><span class="role">调用失败</span><span class="status"><span class="dot bad"></span>error</span></div>${esc(error.message)}</div>`;
      } finally {
        sendEl.disabled = false;
        sendEl.textContent = '生成智能体规划';
      }
    }

    textModeEl.addEventListener('click', () => setMode('text'));
    tableModeEl.addEventListener('click', () => setMode('table'));
    resetEl.addEventListener('click', () => setMode(activeMode));
    fileInputEl.addEventListener('change', async () => {
      try {
        await handleFileUpload(fileInputEl.files[0]);
      } catch (error) {
        fileMetaEl.textContent = `解析失败：${error.message}`;
      }
    });
    sendEl.addEventListener('click', submitTask);
    setMode('text');
  </script>
</body>
</html>"""
    return html.replace("__EXAMPLES__", payload)


def normalize_tool_payload(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    if tool_name not in {"medical_text_cleaner", "medical_text_segmenter"}:
        return payload

    documents = []
    for item in payload.get("documents", []):
        if isinstance(item, Document):
            documents.append(item)
        elif isinstance(item, dict):
            documents.append(
                Document(
                    doc_id=str(item.get("doc_id") or ""),
                    title=str(item.get("title") or ""),
                    text=str(item.get("text") or ""),
                    metadata=dict(item.get("metadata") or {}),
                )
            )
    return {**payload, "documents": documents}


def build_nexent_data_processing_plan(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a Nexent-facing operator plan without executing the full ETL."""
    task = str(payload.get("task") or "清洗医疗数据，完成数据画像、规范化、文本清洗和句子切分。")
    records = payload.get("records") if isinstance(payload.get("records"), list) else []
    profile = RawRecordProfiler().run({"records": records}).get("profile", {})
    tools = list(NEXENT_TOOL_OPERATORS)
    agent = DataProcessingAgent()
    executable_tools = agent._select_tools(tools, profile)
    plan = agent._build_plan(task, executable_tools, profile)
    scheduler = build_batch_scheduler(
        task,
        profile,
        bool(payload.get("scheduler_enabled", True)),
    )
    plan["batch_scheduler"] = scheduler
    requested_runtime = str(payload.get("planner_runtime") or "deterministic").lower()
    if requested_runtime == "fallback":
        requested_runtime = "deterministic"
    lora_result = None
    if requested_runtime in {"lora", "lora_sft", "finetuned", "auto"}:
        lora_result = run_lora_flow_planner(
            task=task,
            profile=profile,
            model=str(payload.get("lora_model") or DEFAULT_LORA_MODEL),
            adapter_path=Path(payload.get("lora_adapter_path") or DEFAULT_LORA_ADAPTER),
            timeout_seconds=float(payload.get("lora_timeout_seconds") or 120),
            optimization_goal=str(payload.get("optimization_goal") or "quality_first"),
        )
        if lora_result.get("status") == "ok":
            plan = merge_lora_plan(plan, lora_result, executable_tools)
            plan["batch_scheduler"] = scheduler

    plan["explainability"] = agent._explain_plan(task, plan, profile, scheduler, executable_tools)

    return {
        "status": "ok",
        "nexent_runtime": "outer_api_tool",
        "tool": "medinsight_plan_data_processing_flow",
        "task": task,
        "profile": profile,
        "planner_runtime": plan.get("runtime"),
        "lora_planner": summarize_lora_result(lora_result) if lora_result else {
            "status": "not_requested",
            "hint": "Set planner_runtime=lora to use the fine-tuned Qwen2.5-0.5B LoRA planner.",
        },
        "plan": plan,
        "recommended_execution": [
            {
                "step": item["order"],
                "tool": item["tool"],
                "nexent_tool_operation": f"medinsight_{item['tool']}",
                "reason": item["reason"],
            }
            for item in plan.get("steps", [])
        ],
    }


def merge_lora_plan(deterministic_plan: dict[str, Any], lora_result: dict[str, Any], tools: list[str]) -> dict[str, Any]:
    parsed_plan = lora_result.get("parsed_plan") or {}
    lora_operators = [str(item) for item in parsed_plan.get("operators", [])]
    selected = [tool for tool in order_tools(lora_operators) if tool in tools]
    if not selected:
        return deterministic_plan
    lora_plan = dict(deterministic_plan)
    lora_plan["runtime"] = "lora_sft+deterministic_rules"
    lora_plan["steps"] = [
        {"order": index, "tool": tool, "reason": f"LoRA 微调流程编排模型推荐调用 {tool}。"}
        for index, tool in enumerate(selected, start=1)
    ]
    lora_plan["lora_flow_planner"] = {
        "model": lora_result.get("model"),
        "adapter_path": lora_result.get("adapter_path"),
        "status": lora_result.get("status"),
        "operators": lora_operators,
        "mapped_tools": selected,
        "execution_policy": parsed_plan.get("execution_policy", {}),
        "rationale": parsed_plan.get("rationale", []),
    }
    return lora_plan


def summarize_lora_result(lora_result: dict[str, Any] | None) -> dict[str, Any]:
    if not lora_result:
        return {"status": "not_requested"}
    parsed_plan = lora_result.get("parsed_plan") if isinstance(lora_result.get("parsed_plan"), dict) else {}
    return {
        "status": lora_result.get("status"),
        "model": lora_result.get("model"),
        "adapter_path": lora_result.get("adapter_path"),
        "returncode": lora_result.get("returncode"),
        "operators": parsed_plan.get("operators", []),
        "execution_policy": parsed_plan.get("execution_policy", {}),
        "parse_error": parsed_plan.get("parse_error"),
        "reason": lora_result.get("reason"),
    }


def order_tools(tools: list[str]) -> list[str]:
    canonical_order = [
        "raw_record_profiler",
        "medical_table_profiler",
        "medical_table_normalizer",
        "medical_table_document_builder",
        "medical_record_normalizer",
        "medical_text_loader",
        "medical_text_cleaner",
        "medical_text_segmenter",
        "table_schema_mapper",
        "deduplication_operator",
        "quality_anomaly_detector",
        "batch_scheduler",
    ]
    selected = set(tools)
    return [tool for tool in canonical_order if tool in selected]


def run_nexent_operator_tool(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    operator_class = NEXENT_TOOL_OPERATORS.get(tool_name)
    if operator_class is None:
        return {
            "status": "error",
            "message": f"unknown Nexent tool: {tool_name}",
            "available_tools": sorted(NEXENT_TOOL_OPERATORS),
        }
    operator = operator_class()
    normalized_payload = normalize_tool_payload(tool_name, payload)
    return {
        "status": "ok",
        "nexent_runtime": "outer_api_tool",
        "tool": tool_name,
        "description": operator.spec.description,
        "result": operator.run(normalized_payload),
    }


def enrich_task1_report_with_nexent_context(result: dict[str, Any]) -> None:
    report_path = result.get("report_path")
    if not report_path:
        return
    path = Path(str(report_path))
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    report["nexent_integration"] = {
        "mode": "outer_api_tool",
        "agent_name": "medinsight_data_processing_agent",
        "planner_tool": "medinsight_plan_data_processing_flow",
        "entry_tool": "medinsight_run_data_processing_agent",
        "operator_tools": [f"medinsight_{name}" for name in NEXENT_TOOL_OPERATORS],
        "role": "Nexent is the agent orchestration layer; MedInsight/DataMate is the data processing tool layer.",
    }
    path.write_text(json.dumps(to_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")


def build_nexent_openapi_spec(server_url: str = "http://host.docker.internal:18080") -> dict[str, Any]:
    paths: dict[str, Any] = {
        "/api/nexent/task1/plan": {
            "post": {
                "operationId": "medinsight_plan_data_processing_flow",
                "summary": "Plan a Nexent data processing flow",
                "description": "由 Nexent 调用的任务理解与流程规划工具：根据自然语言任务和输入数据画像，返回推荐算子链、调度理由和后续工具调用顺序。",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/Task1PlanRequest"}
                        }
                    },
                },
                "responses": {"200": {"description": "Nexent-facing operator plan."}},
            }
        },
        "/api/nexent/task1/run": {
            "post": {
                "operationId": "medinsight_run_data_processing_agent",
                "summary": "Run the MedInsight task-one data processing agent",
                "description": "由 Nexent 作为外部 API 工具触发任务一 Data Processing Agent，自动完成数据画像、规划、DataMate 能力发现、清洗、规范化、切分、状态追踪和异常处理。",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/Task1RunRequest"}
                        }
                    },
                },
                "responses": {"200": {"description": "Task run result with quality metrics and plan."}},
            }
        }
    }

    for name, operator_class in NEXENT_TOOL_OPERATORS.items():
        paths[f"/api/nexent/tools/{name}"] = {
            "post": {
                "operationId": f"medinsight_{name}",
                "summary": operator_class.spec.description,
                "description": f"Nexent 可调用的数据处理算子：{operator_class.spec.description}",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": f"#/components/schemas/{_operator_request_schema_name(name)}"}
                        }
                    },
                },
                "responses": {"200": {"description": f"{name} execution result."}},
            }
        }

    spec = {
        "openapi": "3.0.3",
        "info": {
            "title": "MedInsight Nexent Data Processing Tools",
            "version": "0.1.0",
            "description": "任务一数据处理智能体的 Nexent 工具契约，可通过 Nexent Outer API Tools 导入并调用。",
        },
        "servers": [{"url": server_url.rstrip("/")}],
        "paths": paths,
        "components": {
            "schemas": {
                "Task1PlanRequest": {
                    "type": "object",
                    "required": ["task"],
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "自然语言数据处理任务。",
                            "default": "清洗医疗数据，完成数据画像、规范化、文本清洗和句子切分。",
                        },
                        "records": {
                            "type": "array",
                            "description": "可选原始记录列表，用于生成数据画像并辅助规划。",
                            "items": {"type": "object"},
                        },
                        "planner_runtime": {
                            "type": "string",
                            "description": "规划器选择：deterministic 使用稳定可复现规划器；lora 使用已微调的 Qwen2.5-0.5B LoRA 小模型。",
                            "enum": ["deterministic", "lora", "auto"],
                            "default": "deterministic",
                        },
                        "lora_model": {
                            "type": "string",
                            "description": "可选 LoRA 基座模型。",
                            "default": DEFAULT_LORA_MODEL,
                        },
                        "lora_adapter_path": {
                            "type": "string",
                            "description": "可选 LoRA adapter 目录。",
                            "default": str(DEFAULT_LORA_ADAPTER.relative_to(Path.cwd())) if DEFAULT_LORA_ADAPTER.is_relative_to(Path.cwd()) else str(DEFAULT_LORA_ADAPTER),
                        },
                        "lora_timeout_seconds": {
                            "type": "number",
                            "description": "LoRA 推理超时时间。",
                            "default": 120,
                        },
                    },
                },
                "Task1RunRequest": {
                    "type": "object",
                    "required": ["task"],
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "自然语言数据处理任务。",
                            "default": "清洗医疗数据，完成数据画像、规范化、文本清洗和句子切分。",
                        },
                        "input_path": {
                            "type": "string",
                            "description": "JSONL 输入文件路径；未传 records 时使用。",
                            "default": "data/task1/raw/qamedkg_samples.jsonl",
                        },
                        "output_dir": {
                            "type": "string",
                            "description": "输出目录。",
                            "default": "artifacts/task1/nexent",
                        },
                        "records": {
                            "type": "array",
                            "description": "可选的原始医疗记录列表。",
                            "items": {"type": "object"},
                        },
                        "scheduler_enabled": {
                            "type": "boolean",
                            "description": "是否启用任务一自适应批调度和优化文本算子。",
                            "default": True,
                        },
                    },
                },
                "RecordsPayload": {
                    "type": "object",
                    "required": ["records"],
                    "properties": {
                        "records": {
                            "type": "array",
                            "description": "原始或规范化后的医疗记录。",
                            "items": {"type": "object"},
                        }
                    },
                },
                "DocumentsPayload": {
                    "type": "object",
                    "required": ["documents"],
                    "properties": {
                        "documents": {
                            "type": "array",
                            "description": "Document JSON 列表，包含 doc_id、title、text、metadata。",
                            "items": {"type": "object"},
                        }
                    },
                },
                "RowsPayload": {
                    "type": "object",
                    "required": ["rows"],
                    "properties": {
                        "rows": {
                            "type": "array",
                            "description": "结构化医疗表格行记录。",
                            "items": {"type": "object"},
                        }
                    },
                },
            }
        },
    }
    return normalize_object_required(spec)


def normalize_object_required(value: Any) -> Any:
    """Nexent validates tool schemas strictly; keep object.required as an array."""
    if isinstance(value, dict):
        if value.get("type") == "object" and value.get("required") is None:
            value["required"] = []
        for item in value.values():
            normalize_object_required(item)
    elif isinstance(value, list):
        for item in value:
            normalize_object_required(item)
    return value


def _operator_request_schema_name(tool_name: str) -> str:
    if tool_name in {"raw_record_profiler", "medical_record_normalizer", "medical_text_loader"}:
        return "RecordsPayload"
    if tool_name in {"medical_table_profiler", "medical_table_normalizer", "medical_table_document_builder"}:
        return "RowsPayload"
    return "DocumentsPayload"
