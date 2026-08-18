---
name: medical-kg-report
description: Generate a medical knowledge-graph report (HTML) from a graph.json, and optionally extract medical relation triples from raw text. Use when the user wants a readable report/summary of a medical knowledge graph, entity/relation statistics, or wants to turn medical text into structured triples.
output_type: file
output_mime_types:
  - text/html
  - application/json
---

# Medical Knowledge-Graph Report Skill

Two scripts, used independently:

## 1. `scripts/render_report.py` — graph.json -> HTML report (no LLM, stdlib only)
Render a knowledge graph (produced by MediGraph's KG build, `graph.json` with
`{nodes:[{id,type}], edges:[{head,relation,tail,...}]}`) into a self-contained
HTML report: entity-type distribution, relation-type distribution, top entities,
and the full triple list.

```python
result = run_skill_script(
    "medical-kg-report",
    "scripts/render_report.py",
    params='--graph "/mnt/nexent/graph.json" --output "medical_kg_report.html"'
)
return result
```
Returns JSON: `{"status":"success","file_path":...,"absolute_path":...,"num_entities":N,"num_triples":M}`.

## 2. `scripts/extract_triples.py` — medical text -> relation triples (uses LLM)
Extract ontology-constrained relation triples from raw medical text via an
OpenAI-compatible API (key/base/model passed as args or env). Outputs JSON.

```python
result = run_skill_script(
    "medical-kg-report",
    "scripts/extract_triples.py",
    params='--text "高血压常用药物包括硝苯地平，可并发冠心病。" --api-key "$KEY" --model "Qwen/Qwen3.5-35B-A3B"'
)
return result
```

## Usage rules
- For a report from an existing graph, use `render_report.py` (fast, no API).
- To turn free text into triples, use `extract_triples.py`.
- Return the raw JSON result from the script directly; do not reformat.

## Storage
- Default output directory: `/mnt/nexent`. Override with `--output`/`--working-dir`.
