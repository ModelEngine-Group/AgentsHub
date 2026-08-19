# ModelEngine medical agent OpenAPI/MCP service

This directory is the runnable service behind the three Nexent agents. Run
commands from this directory so the bundled sample data and graph artifacts
resolve correctly.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
TASK3_ANALYSIS_ENABLE_TENSOR=false MEDICAL_DEMO_USE_LLM=false \
python -m demo.run_medical_demo orchestrate \
  --output-root . \
  --input data/raw/medical_demo.jsonl \
  --no-llm \
  --kg-backend rule
python -m demo.app_server --host 0.0.0.0 --port 8088
```

After the server starts:

- Dashboard: `http://127.0.0.1:8088/dashboard`
- Task 1 OpenAPI: `http://127.0.0.1:8088/api/task1/openapi.json`
- Task 2 OpenAPI: `http://127.0.0.1:8088/api/task2/openapi.json`
- Task 3 OpenAPI: `http://127.0.0.1:8088/api/task3/openapi.json`

The rule-based offline demo is the portable baseline. Task 1's real DataMate
execution and Nexent orchestration require the endpoints documented in the
top-level README. Ascend NPU and LoRA support are optional enhancements.
