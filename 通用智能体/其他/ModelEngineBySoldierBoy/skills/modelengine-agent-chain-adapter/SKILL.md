---
name: modelengine-agent-chain-adapter
description: Use this skill when an AI coding agent needs to develop, adapt, verify, or maintain the ModelEngineBySoldierBoy competition project, including DataMate/Nexent integration, data processing agents, medical knowledge graph agents, graph-driven analysis, model-side orchestration policy, NPU operator optimization, reproducible demos, and final submission materials.
metadata:
  short-description: Develop and verify the ModelEngine DataMate/Nexent agent chain
---

# ModelEngine Agent Chain Adapter

This skill describes how an AI coding agent should assist programming and adaptation work for the ModelEngineBySoldierBoy project. It is intended to guide AI-assisted development from basic implementation through enhanced engineering, verification, and final delivery.

## Purpose

Build and maintain a competition-ready intelligent agent system based on DataMate and Nexent:

- Task 1: Data Processing Agent based on Nexent and DataMate.
- Task 2: Medical knowledge graph generation and QA agent.
- Task 3: Graph-driven data analysis and visualization agent.
- Optional enhancement: small orchestration policy model under 1B parameters.
- Bonus enhancement: Ascend NPU operator optimization and CPU/NPU benchmark evidence.

The AI agent should help with real implementation, integration, tests, benchmarks, and documentation. It should not replace required platform links with unrelated mock flows.

## Repository Areas

Use these areas when working in the project:

- `agents/`: core agents and DataMate/Nexent integration logic.
- `operators/`: reusable data processing, KG, analysis, tensor, and NPU-related operators.
- `demo/`: CLI, HTTP API, OpenAPI descriptions, dashboard, traces, and generated reports.
- `data/`: raw, external, and processed data artifacts.
- `graph/`: entities, triples, schema, validation, and graph exports.
- `benchmarks/`: CPU/NPU benchmark entry points and result files.
- `scripts/`: reproducible setup, conversion, validation, and submission scripts.
- `docs/`: project book, requirement mapping, platform evidence, NPU reports, and final checklist.
- `models/`: model-side selector configuration, training/evaluation evidence, and policy artifacts that are small enough for source control.

## Required Inputs

Before making changes, inspect the current repository state:

```bash
git status --short
rg -n "任务一|任务二|任务三|DataMate|Nexent|NPU|NL2SQL" docs README.md
```

For platform integration work, confirm the relevant services and configuration:

- DataMate gateway and UI endpoint.
- Nexent UI/config endpoint.
- Demo API endpoint.
- Knowledge base ID, if DataMate retrieval evidence must be refreshed.
- OpenAI-compatible LLM settings only when Nexent natural-language agent calls need an LLM.

Do not hard-code secrets, personal absolute paths, machine-specific drive letters, passwords, tokens, or temporary remote server addresses into repository files.

## Development Workflow

### 1. Understand the target requirement

Map the requested change to the competition task:

- Task 1: task understanding, DataMate data access/processing, multi-operator planning, status tracking, exception handling, runnable demo.
- Task 2: medical data source, entity extraction, relation extraction, triple generation/validation, KG QA, modular operators, CPU/NPU performance evidence.
- Task 3: reuse Task 1/2 outputs, graph structure understanding, NL2SQL, statistical/association/trend analysis, BI and graph visualization, extensibility.
- Optional model: construct data processing flow optimization data and fine-tune or distill a small orchestration model under 1B parameters.
- NPU bonus: implement real operator-level NPU optimization and compare CPU/NPU throughput, latency, and available resource metrics.

When a user asks for implementation, prefer code changes and verification over only writing a plan.

### 2. Read existing patterns first

Before editing, inspect the nearest existing implementation:

```bash
rg -n "class .*Agent|OpenAPI|MCP|DataMate|tensor|npu|nl2sql" agents operators demo tests
rg --files agents operators demo scripts benchmarks tests docs | sort
```

Follow local style and module boundaries. Extend existing agents/operators instead of creating parallel one-off scripts unless a new entry point is clearly required.

### 3. Implement basic functionality first

For a new or repaired feature, first make the basic competition requirement work end to end:

- Add or update the relevant agent method.
- Add or update reusable operators.
- Connect CLI and/or HTTP API entry points when the feature is user-facing.
- Emit trace/report artifacts so execution can be inspected.
- Keep outputs deterministic where possible.

Do not silently replace DataMate/Nexent-required behavior with unrelated local-only logic. If an official platform dependency cannot run, report the blocker clearly and keep fallback behavior explicitly labeled.

### 4. Strengthen the implementation

After the basic path works, improve real capability:

- Task understanding: support natural language intent extraction, input profiling, constraints, and multi-intent tasks.
- Planning: use multi-operator DAG planning, capability coverage, quality constraints, cost, and compatibility checks.
- Data quality: detect empty text, duplicates, missing fields, schema drift, low information content, and failed contracts.
- Exception handling: record failure category, recovery action, retry result, and final status.
- KG quality: strengthen entity typing, predicate mapping, confidence scoring, conflict filtering, evidence aggregation, and relation validation.
- Analysis: strengthen schema linking, safe SQL generation, graph metrics, BI chart data, graph subgraphs, and explanation text.
- Performance: benchmark real operators, avoid measuring unrelated overhead as operator acceleration.

### 5. Model-side work

For the optional orchestration model:

- Construct data from real project files, DataMate/Nexent feedback, OOD cases, and constraint-aware DAG calibration.
- Keep training/evaluation splits distinct.
- Report holdout/OOD and blind evaluation results separately.
- Use the small model as a candidate strategy generator, not an unchecked executor.
- Validate generated workflow/operator candidates through a guard and deterministic planner.

The model package may be distributed outside Git when too large. In that case, keep scripts, configuration, metrics, SHA256, and download instructions in the repository.

### 6. NPU work

For NPU adaptation:

- Optimize real tensorizable operators, such as ranking, candidate scoring, top-k, or graph similarity.
- Keep CPU and NPU benchmark inputs, dimensions, repeats, dtype, and output checks comparable.
- Include CPU/NPU throughput and latency when available.
- Include resource snapshots such as CPU time, HBM, power, or `npu-smi` output when available.
- State the boundary clearly: operator-level acceleration is not the same as full end-to-end platform acceleration.

Do not claim DataMate official Docker internals are NPU-accelerated unless the project actually changes and verifies that internal code path.

## Validation

Use the narrowest validation that proves the change, then run broader checks before final submission.

Common checks:

```bash
python3 -m unittest discover -s tests
python3 -m demo.run_medical_demo process --output-root . --input data/raw/medical_demo.jsonl
python3 -m demo.run_medical_demo build --output-root .
python3 -m demo.run_medical_demo analyze --output-root .
python3 -m demo.run_medical_demo nl2sql-eval --output-root .
```

Submission-level check:

```bash
bash scripts/prepare_submission.sh \
  --knowledge-base-id <DataMate knowledge base id> \
  --cblue-source data/external/cblue \
  --cblue-limit 300 \
  --no-screenshots
```

NPU checks depend on the available Ascend environment. If NPU is unavailable locally, do not fake NPU results; run on the remote Ascend container or clearly mark the check as not run.

## Deliverables

When the AI agent completes a substantial change, ensure the relevant deliverables are updated:

- Source code for agents/operators/scripts.
- CLI/API/OpenAPI entry points when needed.
- Tests or focused validation commands.
- Runtime reports and JSON evidence when the task requires measurable output.
- `README.md` when run instructions or model package instructions change.
- `docs/ModelEngine初赛项目书终稿.tex` and generated PDF when final project-book content changes.
- NPU reports under `docs/` when NPU benchmark results change.
- Model package link, SHA256, and restore instructions when model artifacts are external.

## Safety and Submission Boundaries

- Do not commit secrets, credentials, API keys, Git tokens, passwords, or private remote access details.
- Do not rely on user-machine absolute paths in documented commands.
- Do not present a fallback or local helper as an official DataMate/Nexent path.
- Do not overstate benchmark results beyond the measured operator or dataset.
- Do not claim arbitrary NL2SQL accuracy beyond the current evaluation set.
- Do not add large model weights to Git unless explicitly required and approved.

## Human-Led Development Responsibilities

This project is human-led. The human developer is responsible for requirement understanding, architecture decisions, implementation direction, environment operation, code acceptance, and final delivery. The AI agent assists the human developer with programming, adaptation, debugging, verification, benchmark organization, and documentation.

Human-led work includes:

- Understanding the competition requirements and deciding the final technical route.
- Designing the overall DataMate + Nexent + multi-agent architecture.
- Deciding which modules to implement, refactor, strengthen, or remove.
- Providing and operating the required development environments, including local Docker/WSL, DataMate, Nexent, GitLink, cloud resources, and Ascend NPU containers.
- Preparing official or access-controlled datasets through authorized access.
- Leading code implementation decisions and accepting or rejecting AI-assisted code changes.
- Uploading external model artifacts and confirming public download availability.
- Confirming final technical claims, benchmark scope, project book content, and submission materials.
- Completing the final competition registration and work submission.

AI-assisted work includes:

- Reading the codebase and proposing implementation paths.
- Generating or editing code under human direction.
- Helping implement agents, operators, CLI/API demos, tests, reports, and benchmark scripts.
- Running local commands, interpreting errors, and suggesting fixes.
- Organizing reproducible evidence, project documentation, model package instructions, and final checklists.

## AI Assistance Record

This project used AI-assisted programming under human direction for:

- Implementing and refining agents, operators, CLI/API demos, reports, and tests.
- Enhancing task understanding, DAG planning, exception recovery, KG extraction, NL2SQL, visualization, model-side strategy selection, and NPU tensor operators.
- Debugging DataMate/Nexent integration, Docker/WSL environment issues, remote NPU workflow, and final submission packaging.
- Maintaining project documentation, project book, model package instructions, and evidence checklists.

The human developer remained the primary owner of the project direction, implementation acceptance, environment access, data preparation, model package upload, GitLink submission, and competition page submission.
