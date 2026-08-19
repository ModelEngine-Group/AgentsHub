# Task 1 Enhancement Implementation Plan

> **执行约束**：本计划后续实施时不使用后台 agent；只使用本地直接工具（Read/Grep/LSP/uv/pytest/git）。Python 依赖统一通过 `uv` 管理。每个实现步骤先写测试，再写实现，再运行验证。

**Goal:** 把任务一从“固定流程 MVP”增强为更接近 Nexent + DataMate 要求的数据处理智能体：支持 Nexent 风格工具入口、DataMate 风格算子适配、可扩展任务规划、算子级异常处理与状态追踪。

**Architecture:** 保持当前 Demo 独立可运行，不直接改动 `DataMate/` 和 `nexent/` 主项目源码。增强层放在 Demo 内部：`planners/` 负责把自然语言映射为计划，`adapters/` 负责兼容 DataMate/Nexent，`agents/data_processing_agent.py` 只做编排和状态追踪。

**Tech Stack:** Python 3.11, uv, pytest, pandas, FastMCP-compatible tool pattern, DataMate-style `sample -> sample` mapper pattern.

---

## 1. Current Evidence

### 1.1 Current task 1 implementation

- File: `agents/data_processing_agent.py`
- Current behavior:
  - `DataProcessingAgent.plan(task_description)` only checks that the task text is not empty.
  - It always returns the same 8 hard-coded steps:
    - `read_csv`
    - `clean_missing`
    - `normalize_text`
    - `extract_fields`
    - `save_csv`
    - `read_text`
    - `split_paragraph`
    - `save_text_json`
  - `DataProcessingAgent.run(...)` does not execute from the returned plan; it calls `_process_structured_records()` and `_process_text_records()` directly.
  - Step records currently include `running` and `success`; failure paths are mostly inherited from operator exceptions and are not summarized into a resilient execution report.

### 1.2 Current operator implementation

- File: `operators/base.py`
- Current behavior:
  - `BaseOperator.execute(...)` sets `status = running`.
  - On success, it stores `result` and marks `status = success`.
  - On failure, it marks `status = failed`, stores `error_message`, and raises `OperatorRuntimeError`.
- Gap:
  - There is no retry policy.
  - There is no skip/continue policy for non-critical steps.
  - There is no unified structured error report for the whole agent run.

### 1.3 Current DataMate adapter

- File: `adapters/datamate_adapter.py`
- Current behavior:
  - Only `CleanTextDataMateMapper` exists.
  - It follows DataMate-style `execute(sample) -> sample` for a dictionary containing `text`.
- DataMate local evidence:
  - `DataMate/runtime/ops/README.md` requires custom operators to provide:
    - `__init__.py`
    - `metadata.yml`
    - `process.py`
  - DataMate operators should inherit from `datamate.core.base_op.Mapper` or `Filter`.
  - The key runtime interface is `execute(self, sample: Dict[str, Any]) -> Dict[str, Any]`.
  - Registration uses `OPERATORS.register_module(module_name=..., module_path=...)`.
- Gap:
  - Current Demo adapter is conceptually compatible, but not packaged as a DataMate operator package with `metadata.yml`, `process.py`, and `__init__.py` registration example.
  - Current task 1 pipeline does not use DataMate-style adapters yet.

### 1.4 Current Nexent-compatible evidence

- File: `nexent/doc/docs/zh/backend/tools/mcp.md`
- File: `nexent/backend/tool_collection/mcp/local_mcp_service.py`
- Nexent local evidence:
  - Nexent supports connecting external MCP servers.
  - Python + FastMCP is explicitly recommended in the Nexent docs.
  - Nexent local example exposes tools with `@local_mcp_service.tool(name=..., description=...)`.
  - MCP tool docstrings/type hints become discoverable schema.
- Gap:
  - Current Demo has FastAPI endpoints, but not a Nexent/FastMCP-style tool file.
  - Directly subclassing or modifying Nexent `CoreAgent` is higher risk and unnecessary for this Demo stage.

---

## 2. Recommended Direction

Use a **lightweight compatibility route** first:

1. Keep `DataProcessingAgent` as the independent Demo orchestrator.
2. Add a Nexent-compatible MCP tool wrapper around `DataProcessingAgent.run(...)`.
3. Add DataMate-style adapters for the task 1 operators and optionally a sample package layout.
4. Add a planner abstraction so current fixed planning becomes one implementation, not hard-coded agent behavior.
5. Add resilient operator execution helpers: retry, skip, fail-fast, structured error report.

This is better than directly modifying DataMate or Nexent because:

- It keeps the Demo runnable with `uv run pytest` and `uv run python run_demo.py task1`.
- It gives visible integration artifacts reviewers can inspect.
- It avoids requiring Docker services or full Nexent/DataMate deployment during development.
- It creates a clean migration path if we later run real Nexent/DataMate services.

---

## 3. File Structure Plan

### Create

- `planners/__init__.py`
  - Exports planner classes.
- `planners/task_planner.py`
  - Defines `PlannedStep`, `TaskPlan`, `TaskPlanner`, `RuleBasedTaskPlanner`.
- `adapters/nexent_mcp_tools.py`
  - FastMCP-compatible tool wrapper for task 1.
- `adapters/datamate_task1_mappers.py`
  - DataMate-style sample mappers for task 1 text/record transformations.
- `adapters/datamate_task1_package/metadata.yml`
  - Example DataMate operator metadata.
- `adapters/datamate_task1_package/process.py`
  - Example DataMate-style operator implementation for task 1.
- `adapters/datamate_task1_package/__init__.py`
  - Example `OPERATORS.register_module(...)` registration.
- `tests/test_task_planner.py`
  - Planner tests.
- `tests/test_nexent_mcp_tools.py`
  - Nexent/FastMCP wrapper tests without starting Nexent.
- `tests/test_datamate_task1_mappers.py`
  - DataMate-style mapper tests.
- `tests/test_data_processing_agent_resilience.py`
  - Retry/skip/failure-report tests.

### Modify

- `agents/data_processing_agent.py`
  - Inject/use `TaskPlanner`.
  - Keep backward-compatible default behavior.
  - Execute from a plan where practical, or at minimum record the selected plan in the summary.
  - Add structured failure handling.
- `adapters/datamate_adapter.py`
  - Keep current `CleanTextDataMateMapper` and optionally re-export new mappers.
- `docs/task1_data_processing_agent.md`
  - Document new planner, Nexent tool, DataMate adapter, and resilience behavior.
- `README.md`
  - Add short usage examples for the new task 1 enhancement.
- `../data_knowledge_insight_project_plan.md`
  - Add/check task 1 enhancement checklist items as they are completed.

---

## 4. Implementation Tasks

### Task 1: Add a planner abstraction while preserving current behavior

**Why:** 当前 `plan()` 是硬编码在 agent 内部的固定列表。先抽象 planner，后续才能替换成 LLM planner 或更复杂规则，不影响现有 CLI/API。

**Files:**
- Create: `planners/__init__.py`
- Create: `planners/task_planner.py`
- Modify: `agents/data_processing_agent.py`
- Test: `tests/test_task_planner.py`
- Test: `tests/test_data_processing_agent.py`

- [x] Step 1: Write failing tests for `RuleBasedTaskPlanner`.
  - It should reject empty task descriptions.
  - It should return both CSV and text steps for “清洗医疗CSV数据，并处理医疗文本”.
  - It should return only text-related steps for text-only task descriptions.
  - It should return only CSV-related steps for CSV-only task descriptions.

- [x] Step 2: Run targeted test and confirm failure.
  - Command: `uv run pytest tests/test_task_planner.py -v`
  - Expected: fails because planner module does not exist.

- [x] Step 3: Implement planner classes.
  - `PlannedStep`: stores `step`, `operator`, `goal`, `required`.
  - `TaskPlan`: stores `steps` and helper methods.
  - `TaskPlanner`: abstract/protocol interface.
  - `RuleBasedTaskPlanner`: rule-based implementation using keywords.

- [x] Step 4: Modify `DataProcessingAgent` to use the planner.
  - Constructor accepts optional `planner`.
  - `plan()` delegates to `self.planner.plan(task_description)`.
  - Default behavior remains backward-compatible for current tests.

- [x] Step 5: Run verification.
  - Command: `uv run pytest tests/test_task_planner.py tests/test_data_processing_agent.py -v`
  - Expected: all pass.

---

### Task 2: Add operator execution resilience

**Why:** 当前算子失败会直接抛异常，不能表达“重试、跳过、降级、最终失败报告”。评分项要求异常处理和任务状态跟踪更完整。

**Files:**
- Modify: `agents/data_processing_agent.py`
- Test: `tests/test_data_processing_agent_resilience.py`

- [x] Step 1: Write failing tests for failure recording.
  - Missing CSV file should produce a summary with `status = "failed"` or raise a clear domain error with recorded failed step.
  - Optional text processing failure can be configured as skipped while structured CSV still succeeds.
  - Failed step record includes `step`, `status`, `detail`, and `error`.

- [x] Step 2: Run targeted test and confirm failure.
  - Command: `uv run pytest tests/test_data_processing_agent_resilience.py -v`

- [x] Step 3: Add a small execution helper.
  - Example responsibility: `_execute_step(step_name, callable, required=True, retries=0)`.
  - It records `running`, `success`, `failed`, and optionally `skipped`.
  - It retries only when configured; default remains fail-fast for current behavior.

- [x] Step 4: Wire helper into task 1 structured/text flows.
  - Do not swallow critical errors silently.
  - Store errors in `summary["execution_state"]["errors"]`.

- [x] Step 5: Run verification.
  - Command: `uv run pytest tests/test_data_processing_agent_resilience.py tests/test_data_processing_agent.py -v`
  - Expected: all pass.

---

### Task 3: Add DataMate-style task 1 mappers and package example

**Why:** 当前只有一个 text mapper，且任务一 pipeline 没有展示 DataMate package 结构。这里补足 DataMate 可迁移证据。

**Files:**
- Create: `adapters/datamate_task1_mappers.py`
- Create: `adapters/datamate_task1_package/metadata.yml`
- Create: `adapters/datamate_task1_package/process.py`
- Create: `adapters/datamate_task1_package/__init__.py`
- Modify: `adapters/datamate_adapter.py`
- Test: `tests/test_datamate_task1_mappers.py`

- [x] Step 1: Write failing mapper tests.
  - `NormalizeMedicalRecordMapper.execute(sample)` normalizes text fields in a record sample.
  - `ExtractMedicalFieldsMapper.execute(sample)` keeps only configured fields.
  - Missing required fields raises readable errors.

- [x] Step 2: Run targeted test and confirm failure.
  - Command: `uv run pytest tests/test_datamate_task1_mappers.py -v`

- [x] Step 3: Implement mappers with DataMate-style interface.
  - Input: `dict[str, object]` sample.
  - Output: updated `dict[str, object]` sample.
  - Avoid importing DataMate runtime so tests stay lightweight.

- [x] Step 4: Add package example files.
  - `metadata.yml` uses `raw_id` matching the process class name.
  - `process.py` shows how the mapper would inherit from DataMate `Mapper` when running inside DataMate.
  - `__init__.py` shows `OPERATORS.register_module(...)` registration.

- [x] Step 5: Run verification.
  - Command: `uv run pytest tests/test_datamate_adapter.py tests/test_datamate_task1_mappers.py -v`
  - Expected: all pass.

---

### Task 4: Add a Nexent-compatible MCP tool wrapper

**Why:** Nexent docs show MCP as the practical extension point. A FastMCP-style wrapper gives reviewers a concrete Nexent integration artifact without requiring full Nexent deployment.

**Files:**
- Create: `adapters/nexent_mcp_tools.py`
- Test: `tests/test_nexent_mcp_tools.py`
- Modify: `README.md`
- Modify: `docs/task1_data_processing_agent.md`

- [x] Step 1: Write failing tests for plain tool function.
  - `process_medical_data_tool(task_description)` returns a JSON-serializable dict.
  - It includes `status`, `structured_result`, `text_result`, and `execution_state`.

- [x] Step 2: Run targeted test and confirm failure.
  - Command: `uv run pytest tests/test_nexent_mcp_tools.py -v`

- [x] Step 3: Implement plain callable first.
  - Keep the core function testable without starting an MCP server.
  - Use project-local `DATA_DIR` and `OUTPUTS_DIR` defaults.

- [x] Step 4: Add optional FastMCP registration.
  - If `fastmcp` is installed, expose `mcp = FastMCP("data-knowledge-insight")` and decorate the function.
  - If `fastmcp` is unavailable, importing the module should still work for tests.
  - Do not require Nexent service startup.

- [x] Step 5: Document how Nexent would connect.
  - Run command: `uv run python -m adapters.nexent_mcp_tools`
  - Server URL example: `http://127.0.0.1:8001/sse`
  - In Docker-based Nexent: use `host.docker.internal` instead of `127.0.0.1`.

- [x] Step 6: Run verification.
  - Command: `uv run pytest tests/test_nexent_mcp_tools.py -v`
  - Command: `uv run pytest -v`
  - Expected: all pass.

---

### Task 5: Document final task 1 architecture and update checklist

**Why:** 任务一增强不仅要能跑，还要能被评审理解：它如何对应 Nexent、DataMate、任务规划和异常处理评分点。

**Files:**
- Modify: `docs/task1_data_processing_agent.md`
- Modify: `docs/datamate_integration.md`
- Modify: `docs/datamate_nexent_mapping.md`
- Modify: `README.md`
- Modify: `../data_knowledge_insight_project_plan.md`

- [x] Step 1: Update task 1 document with the new architecture.
  - Include planner flow.
  - Include operator state flow.
  - Include error strategy.
  - Include Nexent MCP tool example.
  - Include DataMate mapper/package example.

- [x] Step 2: Update project checklist.
  - Mark Nexent tool sample complete only after tests pass.
  - Add explicit task 1 enhancement checklist items.

- [x] Step 3: Run final verification.
  - Command: `uv run pytest -v`
  - Command: `uv run python run_demo.py task1`
  - Expected: tests pass and task 1 outputs regenerate successfully.

---

## 5. Acceptance Criteria

- `uv run pytest -v` passes.
- `uv run python run_demo.py task1` still works.
- `DataProcessingAgent.plan(...)` is no longer hard-coded directly in the agent; it delegates to a planner.
- Current default behavior stays backward-compatible.
- At least one Nexent-compatible MCP tool wrapper exists and is tested.
- At least one DataMate task 1 mapper package example exists with `metadata.yml`, `process.py`, and `__init__.py`.
- Agent execution state can represent success, failure, skipped steps, and readable error details.
- Documentation explains how this maps to Nexent and DataMate without requiring their services to be running.

---

## 6. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| FastMCP dependency may not be installed | Keep the core tool callable importable and testable without FastMCP; only server startup requires FastMCP. |
| DataMate runtime imports may be unavailable in Demo tests | Keep lightweight adapters independent; provide package example as migration artifact. |
| Dynamic planner becomes too complex | Start with `RuleBasedTaskPlanner`; leave LLM planner as a later optional implementation. |
| Error handling hides real failures | Default critical steps remain fail-fast; skip behavior must be explicit and tested. |
| Existing CLI/API behavior breaks | Preserve default planner output and run existing full test suite after each task. |

---

## 7. Suggested Implementation Order

1. Planner abstraction.
2. Resilient execution helper.
3. DataMate-style mappers/package example.
4. Nexent MCP tool wrapper.
5. Documentation and checklist update.

This order keeps each step small, testable, and reversible.

---

## 8. Nexent Remote MCP Integration Plan

### 8.1 Recommended integration route

Use an **independent FastMCP server** in `data_knowledge_insight_demo`, then connect it to Nexent as a remote MCP server.

This is the preferred route over native Nexent SDK tools because:

1. `adapters/nexent_mcp_tools.py` already exposes task 1 as a FastMCP-compatible tool, so extending it to four tools is a natural next step.
2. Nexent documentation explicitly supports Python + FastMCP remote MCP servers.
3. Demo dependencies such as `pandas`, `networkx`, `matplotlib`, and `pyvis` stay isolated in the Demo environment instead of being added to Nexent.
4. Deployment is easy to explain and demonstrate:
- Local Nexent: `http://127.0.0.1:8001/sse`
- Docker Nexent: `http://host.docker.internal:8001/sse`
5. The defense demo is clearer: start the Demo MCP server independently, then call the tools from the Nexent UI.

### 8.2 Target tool surface

Expose four business-level tools, not low-level operators:

| MCP tool | Backing Demo agent | Input | Output |
|---|---|---|---|
| `process_medical_data` | `DataProcessingAgent` | `task_description: str` | Task 1 execution summary |
| `build_medical_knowledge_graph` | `KnowledgeGraphAgent` | `task_description: str` | Triples, graph counts, output paths |
| `ask_medical_kg` | `KGQAAgent` | `question: str` | Answer, evidence triples, status |
| `analyze_medical_data` | `DataAnalysisAgent` | `task_description: str` | Statistics, chart paths, report path |

### 8.3 What to do

- [x] Choose independent FastMCP server + Nexent remote MCP as the recommended integration route.
- [x] Keep the Demo as the owner of medical workflow dependencies and output files.
- [x] Use Nexent remote MCP registration instead of modifying Nexent `CoreAgent`.
- [x] Extend `adapters/nexent_mcp_tools.py` from one tool to four tools.
- [x] Keep each MCP tool callable as a plain Python function first, then register it with FastMCP.
- [x] Add dependency guards for tools with prerequisites:
  - `ask_medical_kg` should ensure `outputs/medical_kg.json` exists or return a clear message.
  - `analyze_medical_data` should ensure `outputs/cleaned_records.csv` and `outputs/medical_kg.json` exist or return a clear message.
- [x] Add targeted tests in `tests/test_nexent_mcp_tools.py` for all four plain functions.
- [x] Document Nexent connection steps in README and task docs.
- [x] Verify with `uv run pytest tests/test_nexent_mcp_tools.py -v` and `uv run pytest -q`.

### 8.4 What not to do

- [x] Do not modify Nexent `CoreAgent` for this stage.
- [x] Do not add Demo dependencies directly into Nexent `sdk/pyproject.toml` or `backend/pyproject.toml` for this stage.
- [x] Do not create native Nexent SDK tools as the first implementation path.
- [x] Do not expose every low-level Demo operator as a Nexent tool; keep the tool surface business-level.
- [x] Do not require DataMate or Nexent Docker services for Demo unit tests.
- [x] Do not commit `.env` or any real LLM API key.
- [x] Do not assume `127.0.0.1` works from inside Docker; document `host.docker.internal`.

### 8.5 Implementation phases

#### Phase 1: Expand MCP wrappers

- Modify: `adapters/nexent_mcp_tools.py`
- Test: `tests/test_nexent_mcp_tools.py`
- Add plain functions:
  - `process_medical_data_tool(...)`
  - `build_medical_knowledge_graph_tool(...)`
  - `ask_medical_kg_tool(...)`
  - `analyze_medical_data_tool(...)`

#### Phase 2: Add prerequisite handling

- For KGQA, check `outputs/medical_kg.json` before calling `KGQAAgent`.
- For analysis, check both `outputs/medical_kg.json` and `outputs/cleaned_records.csv`.
- Preferred behavior: return a structured, readable error such as `{"status": "missing_prerequisite", "required": [...]}` instead of crashing.

#### Phase 3: Register all tools with FastMCP

- Keep import behavior lightweight when `fastmcp` is absent.
- Register all four tools under `FastMCP("data-knowledge-insight")` when available.
- Keep the server startup command:

```bash
uv run python -m adapters.nexent_mcp_tools
```

#### Phase 4: Update docs and demo instructions

- README should include:
  - Server startup command.
- Local URL: `http://127.0.0.1:8001/sse`.
- Docker URL: `http://host.docker.internal:8001/sse`.
  - Suggested Nexent prompts for each tool.
- Task docs should explain task dependencies and expected output files.

#### Phase 5: Verification

Run:

```bash
uv run pytest tests/test_nexent_mcp_tools.py -v
uv run pytest -q
```

Expected:

- All MCP wrapper tests pass.
- Full test suite passes.
- `.env` remains ignored.
- No Nexent source-code modification is required.

### 8.6 Acceptance criteria

- [x] Recommended Nexent integration route is documented.
- [x] Do / do-not-do boundaries are documented.
- [x] Four FastMCP-compatible tools are implemented.
- [x] Four tools have plain-function tests.
- [x] Prerequisite handling is covered by tests.
- [x] README and task docs contain remote MCP connection instructions.
