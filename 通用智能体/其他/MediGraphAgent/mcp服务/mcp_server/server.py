"""FastMCP server exposing the MediGraph operators as Nexent MCP tools.

Run:
  python mcp_server/server.py            # serves SSE at http://127.0.0.1:8000/sse by default
  MEDIGRAPH_MCP_HOST=0.0.0.0 MEDIGRAPH_MCP_PORT=8011 python mcp_server/server.py

Register in Nexent: 智能体开发 -> 选择Agent的工具 -> MCP配置, add the URL.
If Nexent runs in Docker and this server runs on the host, use
http://host.docker.internal:<port>/sse instead of 127.0.0.1.

This is the "Operator-as-MCP-Tool" bridge: the SAME operator implementations used
by the local agents are surfaced here as callable MCP tools.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastmcp import FastMCP  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse  # noqa: E402

from medigraph.operators.base import load_default_operators, get_operator  # noqa: E402

# Register operators once at startup (LLM-backed ones lazily create a client).
load_default_operators()

mcp = FastMCP(name="MediGraphOperators")

# Base URL the *user's browser* should use to fetch generated artifacts. The chat
# UI runs on the host, so container-internal paths like /app/outputs/x.html are
# useless there -- every tool that produces a file also returns a URL built from
# this, which is directly clickable in the Nexent conversation.
ARTIFACT_BASE_URL = os.getenv(
    "MEDIGRAPH_ARTIFACT_BASE_URL",
    f"http://localhost:{os.getenv('MEDIGRAPH_MCP_PORT', '8011')}",
).rstrip("/")

# Whether this process runs inside a container. Matters when fetching chat-uploaded
# files: Nexent mints presigned URLs against localhost (right for the user's browser,
# wrong from in here), so those hosts get redirected to the Docker host gateway.
_IN_CONTAINER = Path("/.dockerenv").exists()


def _output_path(file_name: str, default: str, suffix: str) -> Path:
    """Resolve a user-facing artifact name safely under outputs/."""
    from config.settings import OUTPUTS_DIR

    name = Path(file_name or default).name
    if not name.lower().endswith(suffix):
        name += suffix
    return OUTPUTS_DIR / name


def _artifact_url(path: "str | Path | None") -> str:
    """Map an outputs/ artifact to a browser-clickable URL served by /outputs/.

    Returns "" for anything that is not a real file under outputs/, so callers can
    surface the field unconditionally without inventing dead links.
    """
    if not path:
        return ""
    from config.settings import OUTPUTS_DIR

    try:
        resolved = Path(path).resolve()
        resolved.relative_to(Path(OUTPUTS_DIR).resolve())
    except (ValueError, OSError):
        return ""
    if not resolved.is_file():
        return ""
    return f"{ARTIFACT_BASE_URL}/outputs/{resolved.name}"


@mcp.custom_route("/outputs/{file_name}", methods=["GET"])
async def serve_artifact(request: Request):
    """Serve a generated artifact so chat users can open/download it in one click.

    Only plain files directly under outputs/ are served: the name is reduced to its
    basename and the resolved path is re-checked against outputs/, so "..", absolute
    paths and symlinks escaping the directory are all rejected.
    """
    from config.settings import OUTPUTS_DIR

    root = Path(OUTPUTS_DIR).resolve()
    target = (root / Path(request.path_params["file_name"]).name).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return PlainTextResponse("forbidden", status_code=403)
    if not target.is_file():
        return PlainTextResponse("not found", status_code=404)
    # inline so .html reports render in the browser instead of downloading blindly
    return FileResponse(target, headers={"Content-Disposition": f'inline; filename="{target.name}"'})


@mcp.custom_route("/outputs", methods=["GET"])
async def list_artifacts(request: Request):
    """List downloadable artifacts (handy when a user lost the link from the chat)."""
    from config.settings import OUTPUTS_DIR

    root = Path(OUTPUTS_DIR).resolve()
    if not root.is_dir():
        return JSONResponse({"artifacts": []})
    items = [
        {"name": p.name, "size": p.stat().st_size, "url": f"{ARTIFACT_BASE_URL}/outputs/{p.name}"}
        for p in sorted(root.iterdir())
        if p.is_file() and p.suffix.lower() in {".html", ".json", ".csv", ".md", ".png", ".svg"}
    ]
    return JSONResponse({"count": len(items), "artifacts": items})


def _ontology_summary() -> dict:
    from medigraph.schema.ontology import ENTITY_TYPES, RELATION_CONSTRAINTS, RELATION_TYPES

    constraints = {}
    for relation, (head_types, tail_types) in RELATION_CONSTRAINTS.items():
        constraints[relation] = {
            "head_types": sorted(head_types),
            "tail_types": sorted(tail_types),
        }
    return {
        "entity_types": ENTITY_TYPES,
        "relation_types": RELATION_TYPES,
        "relation_constraints": constraints,
        "entity_type_count": len(ENTITY_TYPES),
        "relation_type_count": len(RELATION_TYPES),
    }


def _read_output_json(file_name: str) -> dict:
    from config.settings import OUTPUTS_DIR

    path = OUTPUTS_DIR / file_name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _metric_view(data: dict) -> dict:
    """Compact a benchmark report for chat-friendly MCP output."""
    if not data:
        return {}
    return {
        "samples": data.get("samples"),
        "extractor": data.get("extractor"),
        "model_version": data.get("model_version"),
        "encoder": data.get("encoder"),
        "entity_micro": data.get("entity_micro"),
        "end_to_end_triple_micro": data.get("end_to_end_triple_micro"),
        "end_to_end_triple_micro_strict": data.get("end_to_end_triple_micro_strict"),
        "latency_ms": data.get("latency_ms"),
    }


@mcp.tool(
    name="inspect_extraction_models",
    description=(
        "只读审计最新抽取级联：神经 GPLinker 权重是否可用、当前路由配置、CMeIE-V1/V2 实测指标、"
        "双编码器集成提升、词典基线、实体链接与自产图谱规模。用于网页录屏展示新更新已接入。"
    ),
)
def inspect_extraction_models() -> str:
    import importlib.util

    from config.settings import get_extraction_config
    from medigraph.extraction.neural_gplinker import NeuralGPLinkerExtractor

    cfg = get_extraction_config()
    neural_dir = cfg.neural_model_dir
    model_bin = neural_dir / "pytorch_model.bin"
    checkpoint_available = NeuralGPLinkerExtractor.available(neural_dir)
    torch_available = importlib.util.find_spec("torch") is not None
    transformers_available = importlib.util.find_spec("transformers") is not None
    neural_available = (
        checkpoint_available
        and torch_available
        and transformers_available
    )
    ensemble_v2 = _read_output_json("eval_ensemble_cmeie_dev.json")
    ensemble_v1 = _read_output_json("eval_ensemble_cmeie_v1_dev.json")
    scale_report = _read_output_json("kg_scale_report.json")
    return json.dumps(
        {
            "runtime_config": {
                "backend": cfg.backend,
                "priority": [
                    "L1_neural_gplinker",
                    "L1_lexicon_fast_path",
                    "L2_llm_fallback" if cfg.llm_fallback else "L2_llm_disabled",
                    "L3_entity_linking_and_validation",
                ],
                "route_threshold": cfg.route_threshold,
                "neural_threshold": cfg.neural_threshold,
                "neural_rel_threshold": cfg.neural_rel_threshold,
            },
            "neural_gplinker": {
                "available": neural_available,
                "checkpoint_available": checkpoint_available,
                "runtime_dependencies": {
                    "torch": torch_available,
                    "transformers": transformers_available,
                },
                "model_dir": str(neural_dir),
                "checkpoint_bytes": model_bin.stat().st_size if model_bin.exists() else 0,
                "relation_space": "full CMeIE predicate space; online graph maps safe subset to the clinical ontology and keeps cmeie_relation provenance",
                "v2_dev": _metric_view(_read_output_json("eval_neural_cmeie_dev.json")),
                "v1_dev_public_comparison": _metric_view(_read_output_json("eval_neural_cmeie_v1_dev.json")),
                "ensemble_v2": ensemble_v2.get("triple_micro", {}),
                "ensemble_v1": ensemble_v1.get("triple_micro", {}),
            },
            "fallback_and_governance": {
                "lexicon_baseline": _metric_view(_read_output_json("eval_fast_cmeie_dev.json")),
                "confidence_calibration": _read_output_json("calibration_report.json"),
                "entity_linking": _read_output_json("eval_entity_linking.json"),
            },
            "self_produced_graph": {
                "self_produced": scale_report.get("self_produced"),
                "third_party_graph_import": scale_report.get("third_party_graph_import"),
                "graph": scale_report.get("graph"),
                "source": "outputs/graph_scaled.json",
            },
        },
        ensure_ascii=False,
    )


@mcp.tool(name="text_clean", description="清洗医疗文本：去网页噪声/页眉页脚/链接/LaTeX/短碎片。输入 text，返回清洗后的 text。")
def text_clean(text: str) -> str:
    return json.dumps(get_operator("text_clean").run({"text": text}), ensure_ascii=False)


@mcp.tool(name="chunker", description="把长文本按标题与字数切块。输入 text 和可选 max_chars，返回 chunks 列表。")
def chunker(text: str, max_chars: int = 1200) -> str:
    return json.dumps(get_operator("chunker").run({"text": text, "max_chars": max_chars}), ensure_ascii=False)


@mcp.tool(name="medical_ner", description="从医疗文本抽取疾病/症状/药物/标志物/基因等医学实体，返回带类型与置信度的实体列表。")
def medical_ner(text: str) -> str:
    out = get_operator("medical_ner").run({"text": text})
    # Compact per-entity records so the calling agent gets a small, summarizable observation.
    if isinstance(out, dict) and isinstance(out.get("entities"), list):
        out = dict(out)
        out["entities"] = [
            {k: e[k] for k in ("name", "type", "start", "end", "confidence") if isinstance(e, dict) and k in e}
            for e in out["entities"]
        ]
    return json.dumps(out, ensure_ascii=False)


@mcp.tool(name="medical_re", description="从医疗文本抽取实体间关系三元组（受本体约束）。输入 text 和可选 entities(JSON)。")
def medical_re(text: str, entities: str = "[]") -> str:
    try:
        ents = json.loads(entities) if isinstance(entities, str) else (entities or [])
    except json.JSONDecodeError:
        ents = []
    return json.dumps(get_operator("medical_re").run({"text": text, "entities": ents}), ensure_ascii=False)


@mcp.tool(name="triple_validator", description="校验候选三元组：schema合法性/去重/置信度/冲突检测。输入 triples(JSON 数组)。")
def triple_validator(triples: str, min_confidence: float = 0.5) -> str:
    try:
        tlist = json.loads(triples) if isinstance(triples, str) else (triples or [])
    except json.JSONDecodeError:
        tlist = []
    return json.dumps(
        get_operator("triple_validator").run({"triples": tlist, "min_confidence": min_confidence}),
        ensure_ascii=False,
    )


@mcp.tool(
    name="load_documents",
    description="读取 TXT/MD/HTML/CSV/JSON/JSONL/DOCX/PDF 文档并统一为文本。输入 paths(JSON 数组)。",
)
def load_documents(paths: str) -> str:
    try:
        values = json.loads(paths) if isinstance(paths, str) else paths
    except json.JSONDecodeError:
        values = []
    return json.dumps(
        get_operator("document_loader").run({"paths": values if isinstance(values, list) else []}),
        ensure_ascii=False,
    )


@mcp.tool(
    name="profile_data_quality",
    description=(
        "检查文档空值、重复、字段缺失和长度异常。documents 可直接传一段文本，"
        "也可传文档 JSON 数组；单段病历无需自行用 Python 组装或读取文件。"
    ),
)
def profile_data_quality(documents: str) -> str:
    try:
        values = json.loads(documents) if isinstance(documents, str) else documents
    except json.JSONDecodeError:
        values = [{"fileName": "inline_medical_record.txt", "text": str(documents)}]
    if isinstance(values, str):
        values = [{"fileName": "inline_medical_record.txt", "text": values}]
    return json.dumps(
        get_operator("data_quality").run({"documents": values if isinstance(values, list) else []}),
        ensure_ascii=False,
    )


@mcp.tool(name="redact_medical_pii", description="脱敏医疗文本中的身份证、电话、邮箱、银行卡和病历号，返回审计计数。")
def redact_medical_pii(text: str) -> str:
    return json.dumps(get_operator("pii_redact").run({"text": text}), ensure_ascii=False)


@mcp.tool(name="link_medical_entities", description="把实体链接到 CM3KG/本地稳定规范 ID。输入 entities(JSON 数组)。")
def link_medical_entities(entities: str) -> str:
    try:
        values = json.loads(entities) if isinstance(entities, str) else entities
    except json.JSONDecodeError:
        values = []
    return json.dumps(
        get_operator("entity_linker").run({"entities": values if isinstance(values, list) else []}),
        ensure_ascii=False,
    )


# ----------------------------------------------------------------------------
# High-level orchestration tools (let a Nexent agent drive the whole pipeline).
# ----------------------------------------------------------------------------


@mcp.tool(
    name="plan_datamate_pipeline",
    description=(
        "调用微调后的 Qwen3.5-0.8B 编排模型，把自然语言数据处理需求规划为可执行算子 DAG。"
        "只规划、不执行；返回模型名、DAG 和可传给 run_datamate_pipeline 的 ops。输入 goal。"
    ),
)
def plan_datamate_pipeline(goal: str) -> str:
    from finetune.api_planner import DEFAULT_MODEL, plan_via_api

    valid_ops = {"text_clean", "chunker", "medical_ner", "medical_re", "triple_validator"}
    dag = plan_via_api(goal)
    selected_ops = [str(node.get("op")) for node in dag if node.get("op") in valid_ops]
    if not selected_ops:
        return json.dumps({"error": "fine-tuned planner returned no supported operators"}, ensure_ascii=False)
    return json.dumps(
        {
            "planner_model": os.getenv("FINETUNED_ORCHESTRATOR_MODEL", DEFAULT_MODEL),
            "goal": goal,
            "dag": dag,
            "ops": ",".join(selected_ops),
        },
        ensure_ascii=False,
    )


def _fetch_uploaded_document(url: str, max_bytes: int = 2_000_000) -> dict:
    """Download a chat-uploaded document (MinIO presigned URL) into a doc dict.

    Nexent appends uploaded files to the query as a `presigned_url`, so accepting a
    URL here is what makes "drag a file into the chat and build a graph from it"
    actually work end to end. Only http(s) is allowed and the body is size-capped.
    """
    import urllib.parse
    import urllib.request

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported URL scheme: {parsed.scheme or '(none)'}")
    # Nexent hands out presigned URLs pointing at localhost (correct for the user's
    # browser). Inside this container localhost is the container itself, so redirect
    # host-local addresses to the Docker host gateway.
    fetch_url = url
    if _IN_CONTAINER and (parsed.hostname or "") in {"localhost", "127.0.0.1"}:
        fetch_url = parsed._replace(
            netloc=f"host.docker.internal:{parsed.port}" if parsed.port else "host.docker.internal"
        ).geturl()
    with urllib.request.urlopen(fetch_url, timeout=30) as resp:  # noqa: S310 - scheme checked above
        raw = resp.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError(f"uploaded file exceeds {max_bytes} bytes")
    name = Path(urllib.parse.unquote(parsed.path)).name or "uploaded.txt"
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        import io

        from pypdf import PdfReader

        text = "\n".join((page.extract_text() or "") for page in PdfReader(io.BytesIO(raw)).pages)
    else:
        text = raw.decode("utf-8", errors="ignore")
    if not text.strip():
        raise ValueError(f"no extractable text in {name}")
    return {"fileName": name, "text": text}


@mcp.tool(
    name="build_medical_kg",
    description=(
        "把医疗文本或语料自动编排为知识图谱：chunker→medical_ner→medical_re→triple_validator→graph_upsert。"
        "输入三选一：text=一段医疗文本；source_path=项目内文件/目录（如 data/demo_cases、data/corpus）；"
        "或 source_url=用户在对话中上传的文件链接（把文件信息里的 presigned_url 原样传进来即可，"
        "支持 .txt/.md/.pdf，多个文件用英文逗号分隔）。"
        "工具会自动读取内容——请不要自己用 python(os/urllib) 去读文件或列目录。"
        "默认写入独立的 nexent_demo_graph.json，不覆盖主图；返回质量统计、样例实体/三元组、溯源与图谱链接。"
    ),
)
def build_medical_kg(text: str = "",
                     source_path: str = "",
                     source_url: str = "",
                     source_name: str = "nexent_demo_input.txt",
                     graph_name: str = "nexent_demo_graph.json",
                     append: bool = False,
                     max_chunks: int = 3,
                     max_docs: int = 3) -> str:
    from medigraph.agents.kg_agent import KGGenAgent
    from medigraph.graph.local_store import LocalGraphStore
    from config.settings import PROJECT_ROOT

    docs = []
    if source_url:
        urls = [u.strip() for u in str(source_url).split(",") if u.strip()]
        for url in urls[: max(1, min(int(max_docs), 10))]:
            try:
                docs.append(_fetch_uploaded_document(url))
            except Exception as exc:
                return json.dumps(
                    {"error": f"failed to read uploaded file: {exc}"}, ensure_ascii=False
                )
    if not docs and source_path:
        try:
            base = (PROJECT_ROOT / source_path).resolve()
            if base != PROJECT_ROOT and PROJECT_ROOT not in base.parents:
                return json.dumps({"error": f"source_path must be inside the project: {source_path}"}, ensure_ascii=False)
            if base.is_dir():
                files = sorted(p for p in base.rglob("*") if p.suffix.lower() in {".txt", ".md"})
                files = files[: max(1, min(int(max_docs), 10))]
            elif base.is_file():
                files = [base]
            else:
                return json.dumps({"error": f"source_path not found: {source_path}"}, ensure_ascii=False)
            for p in files:
                docs.append({"fileName": p.name, "text": p.read_text(encoding="utf-8", errors="ignore")})
        except Exception as exc:
            return json.dumps({"error": f"failed to read source_path {source_path}: {exc}"}, ensure_ascii=False)

    text = (text or "").strip()
    if not docs and text:
        docs = [{"fileName": Path(source_name or "nexent_demo_input.txt").name, "text": text}]
    if not docs:
        return json.dumps(
            {"error": "provide either text=... or source_path=data/demo_cases (a project file or directory)"},
            ensure_ascii=False,
        )

    graph_path = _output_path(graph_name, "nexent_demo_graph.json", ".json")
    store = LocalGraphStore.load_json(graph_path) if append and graph_path.exists() else LocalGraphStore()
    agent = KGGenAgent(store=store, build_vectors=False)
    stats = agent.build(
        docs,
        verbose=False,
        max_chunks_per_doc=max(1, min(int(max_chunks), 8)),
    )
    store.export_json(graph_path)
    html_path = graph_path.with_suffix(".html")
    try:
        store.export_html(html_path)
        graph_html = str(html_path)
    except Exception:
        graph_html = ""
    graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
    candidate = int(stats.get("candidate_triples", 0) or 0)
    valid = int(stats.get("valid_triples", 0) or 0)
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    pass_rate = round(valid / candidate, 4) if candidate else 0.0
    # Show every entity/triple for single-document builds so the caller can name
    # them all. Truncating to a handful made agents invent placeholders such as
    # "Drug: 1（从病历中抽取）" for the entities they never received.
    SAMPLE_LIMIT = 40
    sample_entities = [f"{n.get('id')}（{n.get('type')}）" for n in nodes[:SAMPLE_LIMIT]]
    sample_triples = [
        f"{e.get('head')} -[{e.get('relation_zh') or e.get('relation')}]-> {e.get('tail')}"
        for e in edges[:SAMPLE_LIMIT]
    ]
    # Tell the caller whether the lists are exhaustive, so it never presents a
    # truncated slice as the complete extraction result.
    samples_complete = len(nodes) <= SAMPLE_LIMIT and len(edges) <= SAMPLE_LIMIT
    # Compact, chat-friendly result: a ready-to-echo summary + short samples (no giant JSON dump).
    # The viewable link goes inside `summary` on purpose -- agents reliably echo the
    # summary, so the user gets a clickable artifact without every agent prompt having
    # to name the *_url fields explicitly.
    html_url = _artifact_url(graph_html)
    return json.dumps(
        {
            "summary": (
                f"知识图谱构建完成（{'增量扩充' if append else '新建'}）：抽取实体 "
                f"{stats.get('entities_extracted', 0)} 个，候选三元组 {candidate} 条，"
                f"有效三元组 {valid} 条，校验通过率 {pass_rate}；抽取路由 {stats.get('routing', {})}。"
                + (f" 图谱可视化可直接打开：{html_url}" if html_url else "")
            ),
            "pipeline": ["chunker", "medical_ner", "medical_re", "triple_validator", "graph_upsert"],
            "graph": stats.get("graph", {}),
            "sample_entities": sample_entities,
            "sample_triples": sample_triples,
            "samples_complete": samples_complete,
            "graph_file": str(graph_path),
            "graph_html": graph_html,
            # Clickable links for the chat user (container paths above are not).
            "graph_json_url": _artifact_url(graph_path),
            "graph_html_url": _artifact_url(graph_html),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    name="medical_kg_qa",
    description=(
        "基于医疗知识图谱做意图感知 GraphRAG：识别问题实体与目标关系，图遍历后按关系意图排序证据，"
        "返回中文答案、命中实体、检索模式、证据三元组及来源引用。默认查询 2.68 万三元组主图 graph.json。"
    ),
)
def medical_kg_qa(question: str, graph_name: str = "graph.json", hops: int = 2) -> str:
    from medigraph.agents.qa_agent import QAAgent
    from medigraph.graph.local_store import LocalGraphStore

    graph_json = _output_path(graph_name, "graph.json", ".json")
    if not graph_json.exists():
        return json.dumps(
            {"error": f"graph not found: {graph_json.name}; call build_medical_kg first or use graph_name=graph.json"},
            ensure_ascii=False,
        )
    store = LocalGraphStore.load_json(graph_json)
    res = QAAgent(store=store, hops=max(1, min(int(hops), 3))).answer(question)
    # Compact result: keep the natural-language answer + a few evidence/citation lines the agent can echo.
    evidence = [
        f"{e.get('head')} -[{e.get('relation_zh') or e.get('relation')}]-> {e.get('tail')}"
        if isinstance(e, dict) else str(e)
        for e in (res.get("evidence") or [])[:6]
    ]
    return json.dumps(
        {
            "question": question,
            "answer": res["answer"],
            "refused": res.get("refused", False),
            "retrieval_mode": res["retrieval_mode"],
            "resolved_entities": res["resolved_entities"],
            "evidence_used": res["evidence_used"],
            "evidence": evidence,
            "citations": (res.get("citations") or [])[:6],
            "graph_file": str(graph_json),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    name="inspect_medical_kg",
    description=(
        "只读审计医疗知识图谱能力证据：返回实体/关系类型约束、图规模、规范ID/溯源覆盖和样例，"
        "并读取全量L1、标定、LLM抽取与算子性能评测。不会调用LLM，也不会修改图谱。"
    ),
)
def inspect_medical_kg(graph_name: str = "graph.json", sample_limit: int = 8) -> str:
    from medigraph.graph.local_store import LocalGraphStore
    from config.settings import OUTPUTS_DIR

    graph_json = _output_path(graph_name, "graph.json", ".json")
    if not graph_json.exists():
        return json.dumps({"error": f"graph not found: {graph_json.name}"}, ensure_ascii=False)
    store = LocalGraphStore.load_json(graph_json)
    graph_data = json.loads(graph_json.read_text(encoding="utf-8"))

    evaluations = {}
    for key, file_name in {
        "public_cmeie_diakg": "eval_extraction_public.json",
        "cm3kg_controlled": "eval_extraction_cm3kg.json",
        "pathology_coverage_probe": "eval_extraction_pathology.json",
    }.items():
        path = OUTPUTS_DIR / file_name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        evaluations[key] = {
            "samples": data.get("samples"),
            "model": data.get("model"),
            "ner_strict_f1": (data.get("ner_strict") or {}).get("f1"),
            "ner_relaxed_f1": (data.get("ner_relaxed") or {}).get("f1"),
            "re_strict_f1": (data.get("re_strict") or {}).get("f1"),
            "re_relaxed_f1": (data.get("re_relaxed") or {}).get("f1"),
        }

    operator_benchmark = {}
    benchmark_path = OUTPUTS_DIR / "benchmark_operators.json"
    if benchmark_path.exists():
        operator_benchmark = json.loads(benchmark_path.read_text(encoding="utf-8")).get("operators", {})

    offline_evaluations = {}
    for key, file_name in {
        "neural_gplinker_cmeie_v2_dev": "eval_neural_cmeie_dev.json",
        "neural_gplinker_cmeie_v1_dev": "eval_neural_cmeie_v1_dev.json",
        "neural_ensemble_cmeie_v2_dev": "eval_ensemble_cmeie_dev.json",
        "neural_ensemble_cmeie_v1_dev": "eval_ensemble_cmeie_v1_dev.json",
        "fast_core_250": "eval_fast_core.json",
        "fast_cmeie_full_dev": "eval_fast_cmeie_dev.json",
        "confidence_calibration": "calibration_report.json",
    }.items():
        path = OUTPUTS_DIR / file_name
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            offline_evaluations[key] = {
                "extractor": data.get("extractor"),
                "model_version": data.get("model_version"),
                "encoder": data.get("encoder"),
                "samples": data.get("samples"),
                "entity_micro": data.get("entity_micro"),
                "end_to_end_triple_micro": data.get("end_to_end_triple_micro"),
                "end_to_end_triple_micro_strict": data.get("end_to_end_triple_micro_strict"),
                "triple_micro": data.get("triple_micro"),
                "latency_ms": data.get("latency_ms"),
                "ece_before": data.get("ece_before"),
                "ece_after": data.get("ece_after"),
            }

    limit = max(1, min(int(sample_limit), 20))
    return json.dumps(
        {
            "graph_file": str(graph_json),
            "ontology": _ontology_summary(),
            "graph_stats": store.stats(),
            "graph_audit": store.audit(),
            "sample_entities": graph_data.get("nodes", [])[:limit],
            "sample_triples": graph_data.get("edges", [])[:limit],
            "evaluations": evaluations,
            "offline_evaluations": offline_evaluations,
            "operator_benchmark": operator_benchmark,
        },
        ensure_ascii=False,
    )


@mcp.tool(
    name="analyze_medical_data",
    description=(
        "图谱驱动医疗分析：感知医疗知识图谱后规划 SQL(统计/排名/趋势) 或 GRAPH(关联) 路由，"
        "执行只读 NL2SQL/图遍历，自动选择柱状/折线/饼图/关系网络并生成图文 HTML 报告。"
        "返回路由理由、数据血缘、SQL纠错轨迹、数据、图谱证据、洞察和报告路径。"
    ),
)
def analyze_medical_data(question: str,
                         graph_name: str = "graph.json",
                         report_name: str = "",
                         n_visits: int = 600,
                         seed: int = 42) -> str:
    import time
    from medigraph.analysis.analysis_agent import AnalysisAgent
    from medigraph.analysis.graph_profile import load_graph
    from medigraph.analysis.relational import build_db
    from config.settings import OUTPUTS_DIR

    db = OUTPUTS_DIR / "analytics.db"
    graph_json = _output_path(graph_name, "graph.json", ".json")
    if not graph_json.exists():
        return json.dumps({"error": f"graph not found: {graph_json.name}"}, ensure_ascii=False)
    store, used_example = load_graph(graph_json)
    db_summary = build_db(
        db,
        store,
        n_visits=max(100, min(int(n_visits), 5000)),
        seed=int(seed),
    )
    agent = AnalysisAgent(str(db), graph_json=str(graph_json) if graph_json.exists() else None)
    default_report = f"analysis_mcp_{time.strftime('%m%d_%H%M%S')}.html"
    out_html = _output_path(report_name, default_report, ".html")
    res = agent.analyze(question, out_html=out_html, verbose=False)
    # Keep graph evidence relation-complete within the agent's 24-row budget.
    # A fixed 10-row preview could contain only one relation (for example,
    # recommended drugs) and omit a lower-ranked complication entirely.
    row_limit = 24 if res["route"] == "GRAPH" else 10
    return json.dumps(
        {
            "question": question,
            "route": res["route"],
            "plan": {
                "route": res["route"],
                "reason": res["route_reason"],
                "planner": res["planner"],
            },
            "chart_type": res["chart_type"],
            "columns": res["columns"],
            "rows": res["rows"][:row_limit],
            "row_count": len(res["rows"]),
            "sql": res["sql"],
            "sql_attempts": res["attempts"],
            "anchors": res["anchors"],
            "intent_relations": res["intent_relations"],
            "evidence_total": res["evidence_total"],
            "evidence_used": res["evidence_used"],
            "citations": res["citations"],
            "insight": res["insight"],
            "report_html": res["html"],
            # Clickable link for the chat user (report_html above is a server path).
            "report_url": _artifact_url(res["html"]),
            "data_provenance": {
                "medical_kg_graph": str(graph_json),
                "analytics_db": str(db),
                "demo_record_note": "就诊/处方/检查记录由医疗知识图谱词表按固定seed确定性生成，仅用于可复现分析演示。",
            },
        },
        ensure_ascii=False,
    )


@mcp.tool(
    name="inspect_analysis_assets",
    description=(
        "只读审计图谱驱动分析能力证据：知识图谱复用、分析数据库表与记录数、NL2SQL执行准确率、"
        "支持的SQL/GRAPH路由和BI图表类型、已生成报告。不会调用LLM或修改数据。"
    ),
)
def inspect_analysis_assets(graph_name: str = "graph.json") -> str:
    import sqlite3
    from config.settings import OUTPUTS_DIR
    from medigraph.analysis.graph_profile import build_profile, load_graph
    from medigraph.analysis.relational import schema_text

    graph_json = _output_path(graph_name, "graph.json", ".json")
    if not graph_json.exists():
        return json.dumps({"error": f"graph not found: {graph_json.name}"}, ensure_ascii=False)
    store, used_example = load_graph(graph_json)

    evaluation = {}
    eval_path = OUTPUTS_DIR / "eval_nl2sql.json"
    if eval_path.exists():
        raw = json.loads(eval_path.read_text(encoding="utf-8"))
        evaluation = {
            "model": raw.get("model"),
            "samples": raw.get("samples"),
            "correct": raw.get("correct"),
            "execution_accuracy": raw.get("execution_accuracy"),
            "dual_database_execution_accuracy": raw.get("dual_database_execution_accuracy"),
            "target": 0.85,
            "passed": float(raw.get("execution_accuracy", 0.0) or 0.0) >= 0.85,
        }

    table_counts = {}
    db = OUTPUTS_DIR / "analytics.db"
    if db.exists():
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            for table in ("patient_visits", "prescriptions", "lab_tests", "kg_entities", "kg_triples"):
                table_counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        finally:
            conn.close()

    report_paths = sorted({*OUTPUTS_DIR.glob("analysis*.html"), *OUTPUTS_DIR.glob("task3*.html")})
    reports = [
        {
            "name": path.name,
            "path": str(path),
            "bytes": path.stat().st_size,
            "url": _artifact_url(path),
        }
        for path in report_paths
    ]
    return json.dumps(
        {
            "kg_reuse": {
                "graph_file": str(graph_json),
                "used_example_graph": used_example,
                "graph_profile": build_profile(store),
                "kg_mirror_tables": ["kg_entities", "kg_triples"],
                "analytics_vocab_from_relations": [
                    "treated_in_department",
                    "recommend_drug",
                    "need_examination",
                ],
            },
            "analytics_schema": schema_text(),
            "table_counts": table_counts,
            "nl2sql_evaluation": evaluation,
            "analysis_routes": {
                "SQL": ["统计", "分组", "排名", "趋势", "比例", "费用", "异常率"],
                "GRAPH": ["症状", "药物", "检查", "并发症", "标志物", "基因", "多跳关联"],
            },
            "visualizations": ["bar", "line", "pie", "graph", "heatmap", "scatter", "table"],
            "generated_reports": reports,
            "provenance_note": "KG与词表来自医疗知识图谱主图；就诊/处方/检查明细为固定seed生成的可复现实验数据。",
        },
        ensure_ascii=False,
    )


@mcp.tool(
    name="run_datamate_pipeline",
    description=(
        "在 DataMate 运行或查询数据处理任务。新任务会建数据集→上传文档→按算子DAG提交执行，并立即返回 task_id，"
        "不会在一次对话中等待长任务完成；后续把 task_id 传回本工具即可查询 status/progress，且不会重复创建任务。"
        "可输入 goal，让微调后的0.8B模型先规划DAG；也可直接传 ops。"
        "当 dry_run=true 时不创建DataMate任务，只做两件事：扫描 input_dir 列出将被处理的文件"
        "（文件名、字数、开头预览，见返回的 corpus 字段），并调用0.8B模型规划算子DAG——"
        "用户问“这批数据里有什么”或“先看看再规划”时用它。"
        "file_names 可指定逗号分隔文件名，适合小批量演示。"
    ),
)
def run_datamate_pipeline(input_dir: str = "data/corpus",
                          ops: str = "text_clean,chunker,medical_ner,medical_re,triple_validator",
                          max_files: int = 3,
                          goal: str = "",
                          file_names: str = "",
                          dry_run: bool = False,
                          task_id: str = "",
                          wait_for_completion: bool = False) -> str:
    import time
    from integration.datamate.datamate_client import DataMateClient
    from config.settings import OUTPUTS_DIR, get_llm_config

    client = DataMateClient()
    if task_id.strip():
        task = client.get_task(task_id.strip())
        progress = (task or {}).get("progress", {}) or {}
        return json.dumps(
            {
                "mode": "status",
                "task_id": task_id.strip(),
                "status": (task or {}).get("status"),
                "progress": progress,
                "file_count": (task or {}).get("fileCount"),
                "src_dataset_id": (task or {}).get("srcDatasetId"),
                "dest_dataset_id": (task or {}).get("destDatasetId"),
                "dest_dataset_name": (task or {}).get("destDatasetName"),
                "created_at": (task or {}).get("createdAt"),
                "started_at": (task or {}).get("startedAt"),
                "finished_at": (task or {}).get("finishedAt"),
                "terminal": (task or {}).get("status") in {"COMPLETED", "PARTIAL_SUCCESS", "STOPPED", "FAILED"},
                "datamate_url": "http://localhost:8080",
            },
            ensure_ascii=False,
        )

    planner_model = None
    planner_dag = []
    planner_error = None
    planned_ops = []
    if goal.strip():
        try:
            from finetune.api_planner import DEFAULT_MODEL, plan_via_api

            planner_dag = plan_via_api(goal)
            planner_model = os.getenv("FINETUNED_ORCHESTRATOR_MODEL", DEFAULT_MODEL)
            supported_ops = {"text_clean", "chunker", "medical_ner", "medical_re", "triple_validator"}
            planned_ops = [
                str(node.get("op")) for node in planner_dag
                if str(node.get("op")) in supported_ops
            ]
            if planned_ops:
                ops = ",".join(planned_ops)
            else:
                planner_error = "planner returned no supported DataMate operators"
        except Exception as exc:  # keep the pipeline usable if the optional planner is offline
            planner_error = f"{exc.__class__.__name__}: {exc}; using explicit ops"

    def is_corpus_file(path: Path) -> bool:
        """A README sitting in a corpus folder documents the corpus, it is not
        part of it. Feeding it to the medical operators fails, and when it does
        not fail it pollutes the graph with entities lifted from the prose."""
        return path.suffix.lower() in (".md", ".txt") and path.stem.lower() != "readme"

    def discover_docs() -> list[Path]:
        root = Path(input_dir).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"input directory does not exist: {input_dir}")
        if file_names.strip():
            requested = [Path(name.strip()).name for name in file_names.split(",") if name.strip()]
            found = [
                path for name in requested
                for path in ([root / name] if (root / name).is_file() else sorted(root.rglob(name)))
                if path.is_file() and is_corpus_file(path)
            ]
        else:
            found = sorted(p for p in root.iterdir() if p.is_file() and is_corpus_file(p))
            if not found:
                # Corpora are commonly grouped into per-case subfolders (e.g.
                # data/demo_cases/<case>/*.txt). Fall back to a recursive scan so a
                # caller naming the parent directory gets the documents underneath
                # instead of "no .md/.txt under ...".
                found = sorted(p for p in root.rglob("*") if p.is_file() and is_corpus_file(p))
        return found[:max_files]

    def describe(paths: list[Path], preview_chars: int = 180) -> list[dict]:
        """Name + size + opening lines, so the caller can tell the user what is
        actually in the directory instead of just echoing the path back."""
        described = []
        for path in paths:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                described.append({"file_name": path.name, "error": str(exc)})
                continue
            snippet = " ".join(text.split())
            described.append({
                "file_name": path.name,
                "relative_path": str(path.relative_to(Path(input_dir).resolve())),
                "chars": len(text),
                "preview": snippet[:preview_chars] + ("…" if len(snippet) > preview_chars else ""),
            })
        return described

    if dry_run:
        if not goal.strip():
            return json.dumps({"error": "dry_run requires a natural-language goal"}, ensure_ascii=False)
        # A dry run that only prints a DAG leaves the user unable to see what
        # would actually be processed, so preview the discovered corpus too.
        try:
            preview_docs = discover_docs()
            corpus = {
                "input_dir": input_dir,
                "file_count": len(preview_docs),
                "documents": describe(preview_docs),
                "max_files": max_files,
            }
        except FileNotFoundError as exc:
            corpus = {"input_dir": input_dir, "file_count": 0, "documents": [], "error": str(exc)}
        return json.dumps(
            {
                "mode": "plan_only",
                "planner_model": planner_model,
                "goal": goal,
                "corpus": corpus,
                "planner_dag": planner_dag,
                "selected_ops": planned_ops,
                "planner_error": planner_error,
                "status": "planned" if planner_dag else "planning_failed",
                "datamate_task_created": False,
            },
            ensure_ascii=False,
        )

    ids_file = OUTPUTS_DIR / "datamate_operator_ids.json"
    if not ids_file.exists():
        return json.dumps({"error": "operators not uploaded; run integration/datamate/upload_operators.py first"}, ensure_ascii=False)
    op_ids = json.loads(ids_file.read_text(encoding="utf-8"))
    try:
        docs = discover_docs()
    except FileNotFoundError as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)
    if not docs:
        return json.dumps({"error": f"no .md/.txt under {input_dir}"}, ensure_ascii=False)

    cfg = get_llm_config()
    stamp = time.strftime("%m%d_%H%M%S")
    ds = client.create_dataset(f"mcp_src_{stamp}", "TEXT", "MCP-triggered pipeline")
    for d in docs:
        client.upload_file_to_dataset(ds["id"], d)
    client.wait_dataset_files(ds["id"], expected=len(docs))
    instance = []
    for name in [o.strip() for o in ops.split(",") if o.strip()]:
        oid = op_ids.get(name)
        if not oid:
            continue
        # Forward the configured LLM timeout too: the operators used to hardcode
        # 120s, which times out on content-rich clinical records.
        ov = (
            {"apiBase": cfg.base_url, "apiKey": cfg.api_key, "model": cfg.model, "timeout": cfg.timeout}
            if name in ("medical_ner", "medical_re")
            else {}
        )
        instance.append({"id": oid, "name": name, "inputs": "text", "outputs": "text", "overrides": ov})
    tpl = client.create_template(f"mcp_tpl_{stamp}", "MCP pipeline", instance)
    # Submit expanded instances because the current backend can accept
    # templateId yet create an empty process.
    task = client.create_task(
        f"mcp_task_{stamp}",
        ds["id"],
        ds["name"],
        f"mcp_out_{stamp}",
        "TEXT",
        instance=instance,
    )
    final = client.poll_task(task["id"], max_wait=300) if wait_for_completion else client.get_task(task["id"])
    progress = (final or {}).get("progress", {}) or {}
    terminal = final.get("status") in {"COMPLETED", "PARTIAL_SUCCESS", "STOPPED", "FAILED"}
    if not wait_for_completion:
        mode = "submitted"
    elif terminal:
        mode = "completed"
    else:
        mode = "wait_timeout"
    return json.dumps(
        {
            "mode": mode,
            "planner_model": planner_model,
            "planner_dag": planner_dag,
            "planner_error": planner_error,
            "selected_ops": [item["name"] for item in instance],
            "files": [doc.name for doc in docs],
            "dataset_id": ds["id"],
            "template_id": tpl["id"],
            "task_id": task["id"],
            "status": final.get("status"),
            "progress": progress,
            "terminal": terminal,
            "wait_for_completion": wait_for_completion,
            "datamate_url": "http://localhost:8080",
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    host = os.getenv("MEDIGRAPH_MCP_HOST", "127.0.0.1")
    port = int(os.getenv("MEDIGRAPH_MCP_PORT", "8000"))
    print(f"Starting MediGraph MCP server at http://{host}:{port}/sse")
    mcp.run(transport="sse", host=host, port=port)
