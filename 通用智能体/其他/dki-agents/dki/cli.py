"""dki 统一命令行入口。

用法（在项目根目录）：
    python -m dki gen-data --n 200 --seed 42
    python -m dki task1 [--input data/raw] [--config configs/pipeline_medical_etl.yaml]
    python -m dki task2 [--eval]
    python -m dki task2-qa --question "高血压有哪些症状？"
    python -m dki task3 [--report] [--ask "各科室就诊量是多少"]
    python -m dki eval-nl2sql [--engine rule]
    python -m dki eval-kg
    python -m dki bench-ops / bench-ner --device cpu
    python -m dki agent --which data|kg|insight --task "..."
    python -m dki demo-all
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath, PureWindowsPath

from dki.paths import (
    GOLD_DATA_DIR,
    OUTPUT_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    ensure_dirs,
)


def _print_header(title: str):
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


def _publish_json_once(path: Path, payload: dict) -> None:
    """Atomically publish one JSON artifact without overwriting history."""

    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"create-once JSON artifact exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_RUNTIME_EVALUATION_FILENAMES = {
    "kg_extraction": "kg_extraction_eval.json",
    "nl2sql": "nl2sql_eval.json",
}

_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
_GIT_HEAD_RE = re.compile(r"[0-9a-f]{40}")
_IMMUTABLE_RID_RE = re.compile(
    r"\d{8}T\d{6}Z-g[0-9a-f]{7}-t[0-9a-f]{12}"
)
_INCLUDED_TREE_ALGORITHM = "sha256 of sorted path NUL size NUL file_sha256 LF"


def runtime_evaluation_path(kind: str) -> Path:
    """Return a transient evaluation path that cannot replace promoted claims."""

    try:
        filename = _RUNTIME_EVALUATION_FILENAMES[kind]
    except KeyError as exc:
        raise ValueError(f"unknown runtime evaluation kind: {kind}") from exc
    return OUTPUT_DIR / "runtime_eval" / filename


def _strict_sha256(value, *, source: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{source} must be exactly 64 hexadecimal characters")
    return value.lower()


def _load_json_object(path: Path, *, label: str) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _canonical_included_entry(entry, *, index: int) -> dict:
    if not isinstance(entry, dict):
        raise ValueError(f"included-tree entry {index} must be an object")
    relative = entry.get("path")
    size = entry.get("size")
    digest = entry.get("sha256")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"included-tree entry {index}.path must be non-empty text")
    posix = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    parts = relative.split("/")
    if (
        "\\" in relative
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in parts)
        or str(posix) != relative
    ):
        raise ValueError(f"included-tree entry {index}.path is not canonical repository-relative POSIX text")
    if type(size) is not int or size < 0:
        raise ValueError(f"included-tree entry {index}.size must be a non-negative integer")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"included-tree entry {index}.sha256 must be lowercase SHA-256")
    return {"path": relative, "size": size, "sha256": digest}


def _included_tree_hash(entries: list[dict]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _verify_included_tree_files(entries: list[dict]) -> None:
    """Require the executable worktree to match every manifest entry.

    Recomputing only the manifest's aggregate hash proves that the manifest is
    internally consistent, but it does not prove that the files Streamlit will
    execute still have those bytes.  Bind the receipt to the live source tree
    before any demo evidence directory is created.
    """

    root = PROJECT_ROOT.resolve(strict=True)
    for entry in entries:
        relative = PurePosixPath(entry["path"])
        candidate = PROJECT_ROOT.joinpath(*relative.parts)
        current = PROJECT_ROOT
        for part in relative.parts:
            current = current / part
            if current.is_symlink() or (
                hasattr(current, "is_junction") and current.is_junction()
            ):
                raise ValueError(
                    f"included-tree entry traverses a link or junction: {entry['path']}"
                )
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(
                f"included-tree entry is missing or outside the repository: {entry['path']}"
            ) from exc
        if not resolved.is_file():
            raise ValueError(f"included-tree entry is not a regular file: {entry['path']}")
        if resolved.stat().st_size != entry["size"]:
            raise ValueError(f"included-tree entry size differs from manifest: {entry['path']}")
        digest = hashlib.sha256()
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"included-tree entry SHA-256 differs from manifest: {entry['path']}")


def _included_tree_binding_from_manifest(
    path: Path,
    *,
    expected_rid: str | None = None,
) -> dict:
    """Validate a v1 manifest and the live files it binds.

    A naked aggregate digest is never source authority: it can be copied from a
    different checkout while the process executes unrelated bytes.  The
    authoritative binding therefore includes the manifest itself, its RID and
    git head, and a fresh verification of every included live file.
    """

    path = Path(path).resolve(strict=True)
    payload = _load_json_object(path, label="included-tree manifest")
    if payload.get("schema_version") != 1:
        raise ValueError("included-tree manifest schema_version must be 1")
    rid = payload.get("rid")
    if not isinstance(rid, str) or not rid:
        raise ValueError("included-tree manifest rid must be non-empty text")
    if expected_rid is not None and rid != expected_rid:
        raise ValueError("included-tree manifest rid does not match --rid")
    git_head = payload.get("git_head")
    if not isinstance(git_head, str) or _GIT_HEAD_RE.fullmatch(git_head) is None:
        raise ValueError("included-tree manifest git_head must be lowercase 40-hex")
    if payload.get("algorithm") != _INCLUDED_TREE_ALGORITHM:
        raise ValueError("included-tree manifest algorithm is missing or unsupported")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("included-tree manifest entries must be a non-empty list")
    entries = [
        _canonical_included_entry(entry, index=index)
        for index, entry in enumerate(raw_entries)
    ]
    paths = [entry["path"] for entry in entries]
    if paths != sorted(paths):
        raise ValueError("included-tree manifest entries are not canonically sorted")
    if len(paths) != len(set(paths)) or len(paths) != len({path.casefold() for path in paths}):
        raise ValueError("included-tree manifest contains duplicate or case-alias paths")
    claimed = _strict_sha256(
        payload.get("included_tree_sha256"),
        source="included-tree manifest included_tree_sha256",
    )
    actual = _included_tree_hash(entries)
    if claimed != actual:
        raise ValueError("included-tree manifest aggregate hash does not match its entries")
    _verify_included_tree_files(entries)
    return {
        "manifest_path": str(path),
        "manifest_bytes": path.stat().st_size,
        "manifest_sha256": _sha256_file(path),
        "included_tree_sha256": claimed,
        "rid": rid,
        "git_head": git_head,
        "n_files": len(entries),
    }


def _source_tree_from_manifest(path: Path, *, expected_rid: str | None = None) -> str:
    return _included_tree_binding_from_manifest(
        path,
        expected_rid=expected_rid,
    )["included_tree_sha256"]


def _source_tree_from_final_status(path: Path) -> str:
    payload = _load_json_object(Path(path), label="FINAL_STATUS.json")
    candidates: list[tuple[str, object]] = []
    for field in ("source_tree_sha256", "included_tree_sha256"):
        if field in payload:
            candidates.append((field, payload[field]))
    source = payload.get("source")
    if isinstance(source, dict):
        for field in ("source_tree_sha256", "included_tree_sha256"):
            if field in source:
                candidates.append((f"source.{field}", source[field]))
    included_tree = payload.get("included_tree")
    if isinstance(included_tree, dict):
        for field in ("sha256", "source_tree_sha256", "included_tree_sha256"):
            if field in included_tree:
                candidates.append((f"included_tree.{field}", included_tree[field]))
    if not candidates:
        raise ValueError(
            "FINAL_STATUS.json does not contain an explicit source/included-tree SHA-256 field"
        )
    hashes = {
        _strict_sha256(value, source=f"FINAL_STATUS.json {field}")
        for field, value in candidates
    }
    if len(hashes) != 1:
        raise ValueError("FINAL_STATUS.json contains conflicting source/included-tree hashes")
    return hashes.pop()


def _discover_current_demo_source_binding() -> dict | None:
    """Find one unambiguous immutable manifest matching the live worktree.

    This keeps the documented bare ``demo-ui`` command usable after a source
    snapshot has been created, without treating a ``latest`` pointer or a
    naked digest as authority.  Multiple valid manifests are accepted only
    when they bind the identical included tree; otherwise discovery fails.
    """

    remote_runs = PROJECT_ROOT / "outputs" / "remote_runs"
    if not remote_runs.is_dir() or remote_runs.is_symlink():
        return None
    bindings: list[dict] = []
    try:
        children = sorted(remote_runs.iterdir(), key=lambda item: item.name)
    except OSError:
        return None
    for child in children:
        if (
            _IMMUTABLE_RID_RE.fullmatch(child.name) is None
            or not child.is_dir()
            or child.is_symlink()
        ):
            continue
        manifest = child / "included_tree_manifest.json"
        try:
            binding = _included_tree_binding_from_manifest(
                manifest,
                expected_rid=child.name,
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            continue
        bindings.append(binding)
    if not bindings:
        return None
    tree_hashes = {binding["included_tree_sha256"] for binding in bindings}
    if len(tree_hashes) != 1:
        raise RuntimeError(
            "multiple different included-tree manifests match the live worktree; "
            "pass --rid or --included-tree-manifest explicitly"
        )
    return min(
        bindings,
        key=lambda binding: (binding["manifest_sha256"], binding["manifest_path"]),
    )


def _resolve_demo_source_binding(args) -> dict:
    environment_value = os.getenv("DKI_INCLUDED_TREE_SHA256")
    environment_tree = (
        _strict_sha256(environment_value, source="DKI_INCLUDED_TREE_SHA256")
        if environment_value is not None
        else None
    )
    environment_manifest = os.getenv("DKI_INCLUDED_TREE_MANIFEST")
    explicit_manifest = getattr(args, "included_tree_manifest", None)
    selected_manifest: Path | None = None
    expected_rid: str | None = None

    if explicit_manifest is not None:
        selected_manifest = Path(explicit_manifest)
    elif environment_manifest:
        selected_manifest = Path(environment_manifest)

    rid = getattr(args, "rid", None)
    if selected_manifest is None and rid:
        rid_root = PROJECT_ROOT / "outputs" / "remote_runs" / rid
        rid_manifest = rid_root / "included_tree_manifest.json"
        if rid_manifest.is_file():
            selected_manifest = rid_manifest
            expected_rid = rid

    if selected_manifest is None:
        if environment_tree is not None:
            raise RuntimeError(
                "DKI_INCLUDED_TREE_SHA256 is not source authority without "
                "DKI_INCLUDED_TREE_MANIFEST"
            )
        discovered = _discover_current_demo_source_binding()
        if discovered is not None:
            return discovered
        raise RuntimeError(
            "demo-ui requires a verified included-tree manifest before it can "
            "create a run directory. Pass --included-tree-manifest <path>, set "
            "DKI_INCLUDED_TREE_MANIFEST=<path>, or seal "
            "outputs/remote_runs/<rid>/included_tree_manifest.json and pass --rid <rid>."
        )

    binding = _included_tree_binding_from_manifest(
        selected_manifest,
        expected_rid=expected_rid,
    )
    if environment_tree is not None:
        if environment_tree != binding["included_tree_sha256"]:
            raise ValueError(
                "DKI_INCLUDED_TREE_SHA256 does not match the verified included-tree manifest"
            )
    expected_manifest_sha = os.getenv("DKI_INCLUDED_TREE_MANIFEST_SHA256")
    if expected_manifest_sha is not None:
        expected_manifest_sha = _strict_sha256(
            expected_manifest_sha,
            source="DKI_INCLUDED_TREE_MANIFEST_SHA256",
        )
        if expected_manifest_sha != binding["manifest_sha256"]:
            raise ValueError(
                "DKI_INCLUDED_TREE_MANIFEST_SHA256 does not match the manifest bytes"
            )
    return binding


def _resolve_demo_source_tree(args) -> str:
    return _resolve_demo_source_binding(args)["included_tree_sha256"]


def _preflight_demo_port(host: str, port: int) -> None:
    if not isinstance(host, str) or not host.strip():
        raise ValueError("demo-ui host must be non-empty text")
    if type(port) is not int or not (1 <= port <= 65535):
        raise ValueError("demo-ui port must be an integer from 1 to 65535")
    try:
        addresses = socket.getaddrinfo(
            host,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
            flags=socket.AI_PASSIVE,
        )
    except OSError as exc:
        raise OSError(f"demo-ui cannot resolve bind host {host!r}: {exc}") from exc
    failures: list[str] = []
    for family, socktype, protocol, _canonical_name, address in addresses:
        probe = socket.socket(family, socktype, protocol)
        try:
            probe.bind(address)
        except OSError as exc:
            failures.append(str(exc))
        else:
            return
        finally:
            probe.close()
    detail = failures[-1] if failures else "no bindable address"
    raise OSError(f"demo-ui cannot bind {host}:{port}; port is occupied or unavailable: {detail}")


# ---------------------------------------------------------------- 子命令实现
def cmd_gen_data(args):
    from dki.datagen.generate import generate_corpus

    _print_header(f"生成合成医疗数据集 n={args.n} seed={args.seed}")
    stats = generate_corpus(n_docs=args.n, seed=args.seed)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def _run_pipeline_from_config(config_path: Path, input_dir: Path, pattern: str, name: str):
    from dki.pipeline import LocalPipelineRunner, samples_from_dir
    from dki.pipeline.config import load_pipeline_config

    process, meta = load_pipeline_config(config_path)
    export_path = str(OUTPUT_DIR / meta.get("export_path", "").replace("outputs/", "")) if meta.get("export_path") else ""
    samples = samples_from_dir(input_dir, pattern=pattern, export_path=export_path)
    if not samples:
        print(f"[!] 输入目录没有匹配文件: {input_dir}/{pattern}")
        sys.exit(2)
    runner = LocalPipelineRunner()
    result = runner.run(process, samples, name=name or meta.get("name", ""), export_path=export_path)
    print(result.report.summary())
    return result


def cmd_task1(args):
    _print_header("任务一：数据处理（医疗文本 ETL 流水线）")
    if args.agent:
        return _agent_mode("data", args.task or "请把 data/raw 目录下的医疗病历文本做一次完整的清洗 ETL：先看看有哪些可用算子和数据长什么样，然后编排合适的清洗流水线执行，最后检查任务状态并汇报结果。", offline=args.offline)
    result = _run_pipeline_from_config(
        PROJECT_ROOT / args.config, Path(args.input), args.pattern, "task1-etl"
    )
    if result.report.status.value not in {"COMPLETED", "PARTIAL_SUCCESS"}:
        raise RuntimeError(
            f"任务一流水线状态为 {result.report.status.value}，拒绝发布 manifest"
        )
    # 每次任务一运行发布独立快照，任务二/三只读 manifest 指向的这份产物。
    out_dir = PROCESSED_DIR / "runs" / result.report.task_id / "clean"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for s in result.ok_samples:
        if s.get("text"):
            (out_dir / f"{s['fileId']}.txt").write_text(s["text"], encoding="utf-8")
            n += 1
    from dki.pipeline.artifacts import write_task1_manifest

    manifest = write_task1_manifest(
        task_id=result.report.task_id,
        clean_dir=out_dir,
        source_text_dir=Path(args.input),
        structured_source_dir=RAW_DATA_DIR / "structured",
    )
    print(f"\n[√] 清洗后文本已导出 {n} 份 → {out_dir}")
    print(f"[√] 任务一产物清单 → {manifest}")
    report_path = OUTPUT_DIR / "tasks" / f"{result.report.task_id}.json"
    print(f"[√] 任务报告 → {report_path}")


def cmd_task2(args):
    _print_header("任务二：知识图谱构建（实体→关系→三元组→校验→聚合）")
    if args.agent:
        return _agent_mode("kg", args.task or "请基于任务一 manifest 的病历文本自动编排知识抽取算子流水线，构建医疗知识图谱，并汇报图谱规模统计。", offline=args.offline)
    from dki.kg.build import build_kg_from_samples

    from dki.pipeline.artifacts import (
        load_task1_manifest,
        task1_clean_dir,
        write_task2_receipt,
    )

    manifest = load_task1_manifest()
    input_dir = task1_clean_dir(manifest)
    if args.input and Path(args.input).resolve() != input_dir.resolve():
        raise ValueError("任务二只能读取任务一 manifest 产物；请先运行 task1")
    result = _run_pipeline_from_config(
        PROJECT_ROOT / args.config, input_dir, args.pattern, "task2-kg"
    )
    if result.report.status.value not in {"COMPLETED", "PARTIAL_SUCCESS"}:
        raise RuntimeError(
            f"任务二流水线状态为 {result.report.status.value}，拒绝发布知识图谱"
        )
    kg = build_kg_from_samples(result.samples)
    kg_path = OUTPUT_DIR / "kg" / "medical_kg.json"
    kg.save(kg_path)
    receipt = write_task2_receipt(
        kg_path,
        task_id=result.report.task_id,
    )
    print(f"\n[√] 知识图谱 → {kg_path}")
    print(f"[√] 图谱 provenance → {receipt}")
    print(json.dumps(kg.stats(), ensure_ascii=False, indent=2))

    if args.eval:
        _eval_kg_result(result)


def _eval_kg_result(result):
    from dki.kg.eval import evaluate_triples, format_eval_report, predictions_from_samples

    preds = predictions_from_samples(result.samples)
    if not (GOLD_DATA_DIR / "triples").exists():
        print("[!] 没有金标准目录 data/gold/triples，跳过抽取质量评测")
        return
    res = evaluate_triples(preds)
    _print_header("抽取质量评测（vs 金标准）")
    print(format_eval_report(res))
    out = runtime_evaluation_path("kg_extraction")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[√] 评测明细 → {out}")


def cmd_task2_qa(args):
    from dki.kg.qa import answer_question
    from dki.kg.store import KnowledgeGraph
    from dki.pipeline.artifacts import validate_task2_artifact

    kg_path = OUTPUT_DIR / "kg" / "medical_kg.json"
    validate_task2_artifact(kg_path)
    kg = KnowledgeGraph.load(kg_path)
    res = answer_question(kg, args.question)
    _print_header(f"图谱问答：{args.question}")
    print(res["answer"])
    if res["evidence"]:
        print("\n证据（原文片段）：")
        for e in res["evidence"][:3]:
            print(f"  - [{e.get('doc','')}] {e.get('sent','')}")


def cmd_task3(args):
    _print_header("任务三：数据分析与 BI 可视化")
    if args.agent:
        return _agent_mode("insight", args.task or args.ask or "请基于知识图谱与就诊数据做一次全面分析：先了解图谱结构与数据概况，再做统计分析与指标计算，生成图文结合的 BI 报告。", offline=args.offline)
    from dki.analysis.database import build_database
    from dki.kg.store import KnowledgeGraph
    from dki.pipeline.artifacts import (
        load_task1_manifest,
        manifest_path,
        validate_task2_artifact,
    )

    kg_path = OUTPUT_DIR / "kg" / "medical_kg.json"
    validate_task2_artifact(kg_path)
    kg = KnowledgeGraph.load(kg_path)
    db_path = OUTPUT_DIR / "insight.db"
    load_task1_manifest()
    build_database(
        db_path=db_path,
        manifest_path=manifest_path(),
        kg=kg,
        kg_path=kg_path,
    )
    print(f"[√] 分析数据库 → {db_path}")

    if args.ask:
        from dki.analysis.nl2sql_profile_factory import build_engine

        engine = build_engine(db_path)
        res = engine.query(args.ask)
        print(f"\n问题: {args.ask}")
        print(f"SQL ({res.get('engine')}): {res.get('sql')}")
        cols = res.get("columns") or []
        rows = res.get("rows") or []
        print(" | ".join(map(str, cols)))
        for r in rows[:20]:
            print(" | ".join(map(str, r)))
        return

    from dki.analysis.bi import build_default_report

    out = build_default_report(
        db_path=db_path, kg=kg, out_path=OUTPUT_DIR / "reports" / "insight_report.html"
    )
    print(f"[√] BI 分析报告 → {out}")


def cmd_eval_nl2sql(args):
    from dki.analysis.database import build_database, database_matches_manifest
    from dki.analysis.nl2sql_eval import evaluate
    from dki.kg.store import KnowledgeGraph
    from dki.pipeline.artifacts import (
        load_task1_manifest,
        manifest_path,
        validate_task2_artifact,
    )

    db_path = OUTPUT_DIR / "insight.db"
    kg_path = OUTPUT_DIR / "kg" / "medical_kg.json"
    validate_task2_artifact(kg_path)
    kg = KnowledgeGraph.load(kg_path)
    load_task1_manifest()
    if not database_matches_manifest(db_path, manifest_path(), kg_path):
        build_database(
            db_path=db_path,
            manifest_path=manifest_path(),
            kg=kg,
            kg_path=kg_path,
        )
    _print_header(f"NL2SQL 准确率评测（engine={args.engine}）")
    res = evaluate(engine=args.engine, db_path=db_path)
    print(f"准确率: {res['accuracy']:.2%}  ({res['n_correct']}/{res['n']})")
    for d in res.get("details", []):
        if not d.get("correct"):
            print(f"  [X] {d['question']}\n      pred: {d.get('pred_sql')}\n      gold: {d.get('gold_sql')}")
    out = runtime_evaluation_path("nl2sql")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[√] 明细 → {out}")


def cmd_eval_kg(args):
    from dki.pipeline import LocalPipelineRunner, samples_from_dir
    from dki.pipeline.artifacts import load_task1_manifest, task1_clean_dir
    from dki.pipeline.config import load_pipeline_config

    _print_header("知识抽取质量评测（重跑任务二流水线 vs 金标准）")
    process, _ = load_pipeline_config(PROJECT_ROOT / "configs" / "pipeline_kg_build.yaml")
    samples = samples_from_dir(task1_clean_dir(load_task1_manifest()), pattern="*.txt")
    runner = LocalPipelineRunner()
    result = runner.run(process, samples, name="eval-kg")
    _eval_kg_result(result)


def cmd_bench_ops(args):
    from dki.bench.operator_bench import run_bench

    _print_header("算子基准测试（CPU）")
    out = run_bench(repeat=args.repeat, limit=args.limit)
    print(f"[√] 结果 → {out}")


def cmd_bench_ner(args):
    from dki.bench.npu.run_npu_bench import main as npu_main

    npu_main(["--device", args.device, "--batch", args.batch, "--iters", str(args.iters)])


def _offline_responses(which: str):
    """Deterministic ReAct traces for demos without external LLM credentials."""
    if which == "data":
        return [
            '<code>\nprint(list_operators(category="medical"))\n'
            'print(preview_data(path="data/raw", n=300))\n</code>',
            '<code>\nimport json\n'
            'pipeline = ['
            '{"op":"DemoTransientOnce","params":{}},'
            '{"op":"MedicalTextCleaner","params":{"maskPii":True}},'
            '{"op":"FileWithShortOrLongLengthFilter","params":{"fileLength":[30,1000000]}}'
            ']\n'
            'submitted = json.loads(run_pipeline('
            'pipeline=json.dumps(pipeline), input_dir="data/raw", pattern="*.txt", '
            'name="offline-agent-task1"))\n'
            'tid = submitted["task_id"]\nprint(submitted)\n</code>',
            '<code>\nimport json\n'
            'status = json.loads(get_task_status(task_id=tid, wait_seconds=5))\n'
            'if status.get("is_running"):\n'
            '    status = json.loads(get_task_status(task_id=tid, wait_seconds=5))\n'
            'print(status)\n</code>',
            '<code>\nimport json\n'
            'published = json.loads(publish_task1_artifacts(task_id=tid, source_text_dir="data/raw"))\n'
            'events = status["task"].get("recovery_events", [])\n'
            'final_answer("任务一已发布；状态=" + status["task"]["status"] + '
            '"；恢复事件=" + str(len(events)) + "；manifest=" + '
            'str(published.get("manifest_path", "")))\n</code>',
        ]
    if which == "kg":
        return [
            '<code>\nimport json\nprint(kg_query(action="stats"))\n</code>',
            '<code>\nfinal_answer("离线 KnowledgeGraphAgent 已读取图谱结构与关系分布。")\n</code>',
        ]
    return [
        '<code>\nprint(nl2sql_query(question="按科室统计就诊量"))\n</code>',
        '<code>\nfinal_answer("离线 InsightAgent 已通过 NL2SQL 验证分析数据库。")\n</code>',
    ]


def _summarise_tool_result(value) -> dict:
    """Return a non-sensitive outcome summary for an actual tool result."""

    parsed = None
    if isinstance(value, str):
        try:
            candidate = json.loads(value)
        except json.JSONDecodeError:
            candidate = None
        if isinstance(candidate, dict):
            parsed = candidate
    elif isinstance(value, dict):
        parsed = value
    summary = {"outcome": "returned"}
    if isinstance(parsed, dict):
        if parsed.get("ok") is False or parsed.get("status") in {"FAILED", "ERROR"}:
            summary["outcome"] = "reported_failure"
        elif parsed.get("ok") is True or parsed.get("status") in {
            "PENDING", "RUNNING", "COMPLETED", "PARTIAL_SUCCESS"
        }:
            summary["outcome"] = "reported_success"
        for key in ("status", "disposition", "terminal", "is_running"):
            item = parsed.get(key)
            if isinstance(item, (str, bool)) or item is None:
                summary[key] = item
    return summary


def _instrument_agent_tools(agent, calls: list[dict]) -> tuple[int, list[str]]:
    """Instrument exposed CoreAgent tools; never infer calls from a prompt."""

    raw_tools = getattr(agent, "tools", None)
    if isinstance(raw_tools, dict):
        tools = list(raw_tools.values())
    elif isinstance(raw_tools, (list, tuple)):
        tools = list(raw_tools)
    else:
        tools = []
    instrumented = 0
    failures: list[str] = []
    attempts: dict[str, int] = {}
    previous_calls: dict[str, dict] = {}
    for tool in tools:
        name = getattr(tool, "name", None)
        original = getattr(tool, "forward", None)
        if not isinstance(name, str) or not name or not callable(original):
            continue

        def traced_forward(*call_args, __name=name, __original=original, **call_kwargs):
            attempt = attempts.get(__name, 0) + 1
            attempts[__name] = attempt
            previous = previous_calls.get(__name, {})
            started = time.monotonic()
            row = {
                "tool": __name,
                "attempt": attempt,
                "repeated_call": attempt > 1,
                "retry": bool(
                    previous.get("outcome") in {"raised", "reported_failure"}
                    or previous.get("is_running") is True
                ),
                "observed": True,
            }
            try:
                result = __original(*call_args, **call_kwargs)
            except BaseException as exc:
                row.update({"outcome": "raised", "error_type": type(exc).__name__})
                raise
            else:
                row.update(_summarise_tool_result(result))
                return result
            finally:
                row["duration_ms"] = round((time.monotonic() - started) * 1000, 3)
                calls.append(row)
                previous_calls[__name] = row

        try:
            setattr(tool, "forward", traced_forward)
        except (AttributeError, TypeError) as exc:
            failures.append(f"{name}:{type(exc).__name__}")
        else:
            instrumented += 1
    return instrumented, failures


def _publish_agent_observation(
    *,
    which: str,
    task: str,
    offline: bool,
    agent,
    calls: list[dict],
    instrumented_tools: int,
    instrumentation_failures: list[str],
    status: str,
) -> Path:
    trace_dir = OUTPUT_DIR / "agent_observations"
    trace_dir.mkdir(parents=True, exist_ok=True)
    model = getattr(agent, "model", None)
    model = getattr(model, "_model", model)
    model_implementation = (
        f"{type(model).__module__}.{type(model).__qualname__}"
        if model is not None else "unavailable"
    )
    payload = {
        "schema_version": 1,
        "kind": "dki_demo_agent_observation",
        "agent": which,
        "status": status,
        "offline": bool(offline),
        "model_backend": "ScriptedModel" if offline else "OpenAIModel",
        "model_implementation": model_implementation,
        "core_agent": f"{type(agent).__module__}.{type(agent).__qualname__}",
        "task_sha256": hashlib.sha256(task.encode("utf-8")).hexdigest(),
        "task_length": len(task),
        "trace_source": (
            "partial_instrumentation" if instrumented_tools > 0 and instrumentation_failures
            else ("instrumented_tool_forward" if instrumented_tools > 0 else "unavailable")
        ),
        "instrumented_tool_count": instrumented_tools,
        "instrumentation_failures": instrumentation_failures,
        "tool_calls": calls,
        "privacy": {
            "contains_full_user_input": False,
            "contains_tool_arguments": False,
            "contains_tool_results": False,
            "contains_secrets": False,
        },
    }
    for index in range(1, 1000):
        path = trace_dir / f"{index:03d}-{which}.json"
        try:
            _publish_json_once(path, payload)
        except FileExistsError:
            continue
        return path
    raise RuntimeError("too many demo agent observation artifacts")


def _agent_mode(which: str, task: str, offline: bool = False):
    from dki.agents.runtime import run_agent_sync

    if which == "data":
        from dki.agents.data_agent import build_data_processing_agent as build
    elif which == "kg":
        from dki.agents.kg_agent import build_kg_agent as build
    else:
        from dki.agents.insight_agent import build_insight_agent as build

    try:
        if offline:
            from dki.agents.runtime import ScriptedModel, build_observer

            agent = build(model=ScriptedModel(_offline_responses(which)), observer=build_observer())
        else:
            agent = build()
    except Exception as e:
        print(f"[!] 智能体创建失败：{e}")
        print("    请在 .env 配置 LLM_API_KEY / LLM_MODEL_NAME / LLM_API_URL 后重试；")
        print("    或用 --offline 启动 ScriptedModel 离线智能体演示。")
        sys.exit(2)
    _print_header(f"Nexent 智能体（{which}）任务：{task[:60]}...")
    calls: list[dict] = []
    instrumented_tools, instrumentation_failures = _instrument_agent_tools(agent, calls)
    answer = run_agent_sync(agent, task)
    observation = _publish_agent_observation(
        which=which,
        task=task,
        offline=offline,
        agent=agent,
        calls=calls,
        instrumented_tools=instrumented_tools,
        instrumentation_failures=instrumentation_failures,
        status="run_returned" if answer else "outcome_unavailable",
    )
    _print_header("最终回答")
    print(answer)
    print(f"[agent-observation] {observation}")
    return answer


def cmd_agent(args):
    task = args.task
    if not task:
        print("请用 --task 传入任务描述")
        sys.exit(2)
    _agent_mode(args.which, task, offline=args.offline)


def _manifest_input_snapshot(manifest: dict) -> dict:
    """Bind the exact Task1 input snapshot without rereading global input files."""

    source = manifest.get("source")
    rows = source.get("raw_text_files") if isinstance(source, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("Task1 manifest lacks a non-empty raw input snapshot")
    canonical: list[tuple[str, int, str]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Task1 input snapshot row {index} is invalid")
        path = row.get("path")
        size = row.get("bytes")
        sha256 = row.get("sha256")
        normalised_path = path.replace("\\", "/") if isinstance(path, str) else ""
        posix_path = PurePosixPath(normalised_path)
        windows_path = PureWindowsPath(normalised_path)
        path_key = normalised_path.casefold()
        if (
            not isinstance(path, str)
            or not path
            or posix_path.is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
            or any(part in {"", ".", ".."} for part in normalised_path.split("/"))
            or path_key in seen
            or not isinstance(size, int)
            or size < 0
            or not isinstance(sha256, str)
            or _SHA256_RE.fullmatch(sha256) is None
        ):
            raise ValueError(f"Task1 input snapshot row {index} is not hash-bound")
        seen.add(path_key)
        canonical.append((normalised_path, size, sha256.lower()))
    digest = hashlib.sha256()
    for path, size, sha256 in sorted(canonical):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256.encode("ascii"))
        digest.update(b"\n")
    return {
        "algorithm": "sha256 of sorted path NUL size NUL file_sha256 LF",
        "n_files": len(canonical),
        "sha256": digest.hexdigest(),
    }


def _bound_task_report(manifest: dict) -> tuple[Path, dict]:
    binding = manifest.get("task_report")
    if not isinstance(binding, dict):
        raise ValueError("Task1 manifest lacks task report provenance")
    raw_path = binding.get("path")
    expected_size = binding.get("bytes")
    expected_sha256 = binding.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("Task1 task report path is invalid")
    path = (PROJECT_ROOT / raw_path).resolve()
    output_root = OUTPUT_DIR.resolve()
    try:
        path.relative_to(output_root)
    except ValueError as exc:
        raise ValueError("Task1 task report escapes the isolated demo output") from exc
    if (
        not path.is_file()
        or path.stat().st_size != expected_size
        or _sha256_file(path) != expected_sha256
    ):
        raise ValueError("Task1 task report binding failed")
    payload = _load_json_object(path, label="Task1 task report")
    if payload.get("status") != "COMPLETED" or payload.get("task_id") != manifest.get("task_id"):
        raise ValueError("Task1 task report is not a completed manifest match")
    return path, payload


def _runtime_source_binding() -> dict | None:
    """Revalidate the authoritative manifest inherited from demo-ui."""

    manifest_value = (os.getenv("DKI_INCLUDED_TREE_MANIFEST") or "").strip()
    expected_tree = (os.getenv("DKI_INCLUDED_TREE_SHA256") or "").strip()
    expected_manifest = (
        os.getenv("DKI_INCLUDED_TREE_MANIFEST_SHA256") or ""
    ).strip()
    ui_bound = bool((os.getenv("DKI_DEMO_SERVER_RID") or "").strip())
    if not manifest_value:
        if ui_bound:
            raise RuntimeError(
                "demo-ui child lacks authoritative DKI_INCLUDED_TREE_MANIFEST"
            )
        return None
    binding = _included_tree_binding_from_manifest(Path(manifest_value))
    if expected_tree and _strict_sha256(
        expected_tree,
        source="DKI_INCLUDED_TREE_SHA256",
    ) != binding["included_tree_sha256"]:
        raise ValueError("runtime source tree differs from the bound manifest")
    if expected_manifest and _strict_sha256(
        expected_manifest,
        source="DKI_INCLUDED_TREE_MANIFEST_SHA256",
    ) != binding["manifest_sha256"]:
        raise ValueError("runtime included-tree manifest bytes changed after UI startup")
    return binding


def _load_demo_agent_observations(*, agent_enabled: bool, offline: bool) -> dict:
    trace_dir = OUTPUT_DIR / "agent_observations"
    paths = sorted(trace_dir.glob("*.json")) if trace_dir.is_dir() else []
    if not agent_enabled:
        if paths:
            raise ValueError("non-agent demo contains unexpected agent observations")
        return {
            "trace_source": "not_requested",
            "tool_calls": [],
            "observations": [],
            "retry_count": 0,
        }
    if not paths:
        raise ValueError("agent demo lacks actual agent observation artifacts")

    expected_backend = "ScriptedModel" if offline else "OpenAIModel"
    observations: list[dict] = []
    calls: list[dict] = []
    seen_agents: list[str] = []
    unavailable = 0
    partial = 0
    core_agents: set[str] = set()
    model_implementations: set[str] = set()
    expected_privacy = {
        "contains_full_user_input": False,
        "contains_tool_arguments": False,
        "contains_tool_results": False,
        "contains_secrets": False,
    }
    for path in paths:
        payload = _load_json_object(path, label="agent observation")
        if (
            payload.get("schema_version") != 1
            or payload.get("kind") != "dki_demo_agent_observation"
            or payload.get("offline") is not offline
            or payload.get("model_backend") != expected_backend
        ):
            raise ValueError(f"agent observation mode binding failed: {path.name}")
        agent = payload.get("agent")
        core_agent = payload.get("core_agent")
        model_implementation = payload.get("model_implementation")
        tool_calls = payload.get("tool_calls")
        if (
            agent not in {"data", "kg", "insight"}
            or not isinstance(core_agent, str)
            or not core_agent
            or not isinstance(model_implementation, str)
            or not model_implementation
            or not isinstance(tool_calls, list)
            or payload.get("privacy") != expected_privacy
        ):
            raise ValueError(f"agent observation contract failed: {path.name}")
        seen_agents.append(agent)
        core_agents.add(core_agent)
        model_implementations.add(model_implementation)
        if payload.get("trace_source") == "unavailable":
            unavailable += 1
        elif payload.get("trace_source") == "partial_instrumentation":
            partial += 1
        for row in tool_calls:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("tool"), str)
                or row.get("observed") is not True
                or type(row.get("attempt")) is not int
                or row["attempt"] < 1
                or type(row.get("retry")) is not bool
                or not isinstance(row.get("outcome"), str)
            ):
                raise ValueError(f"agent observation contains invalid tool call: {path.name}")
            calls.append({"agent": agent, **row})
        observations.append({
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "agent": agent,
            "status": payload.get("status"),
            "trace_source": payload.get("trace_source"),
            "task_sha256": payload.get("task_sha256"),
            "task_length": payload.get("task_length"),
        })
    if sorted(seen_agents) != ["data", "insight", "kg"]:
        raise ValueError("agent observations must bind exactly data, kg, and insight agents")
    if len(core_agents) != 1:
        raise ValueError("agent observations disagree on CoreAgent implementation")
    if len(model_implementations) != 1:
        raise ValueError("agent observations disagree on model implementation")
    trace_source = (
        "unavailable" if unavailable == len(paths) else
        ("partially_unavailable" if unavailable or partial else "instrumented_tool_forward")
    )
    return {
        "trace_source": trace_source,
        "tool_calls": calls,
        "observations": observations,
        "retry_count": sum(1 for row in calls if row["retry"]),
        "core_agent": core_agents.pop(),
        "model_backend": expected_backend,
        "model_implementation": model_implementations.pop(),
    }


def _demo_artifact_binding(path: Path, *, label: str) -> dict:
    output_root = OUTPUT_DIR.resolve(strict=True)
    candidate = Path(path)
    current = OUTPUT_DIR
    try:
        relative = candidate.relative_to(OUTPUT_DIR)
    except ValueError as exc:
        raise ValueError(f"{label} is outside the isolated demo output") from exc
    for part in relative.parts:
        current = current / part
        if current.is_symlink() or (
            hasattr(current, "is_junction") and current.is_junction()
        ):
            raise ValueError(f"{label} traverses a link or junction")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(output_root)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"{label} is missing or outside the isolated demo output") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _publish_demo_runtime(args) -> Path:
    """Publish the structured runtime truth consumed by the browser receipt."""

    manifest_path = PROCESSED_DIR / "manifest.json"
    manifest = _load_json_object(manifest_path, label="Task1 manifest")
    task_report_path, task_report = _bound_task_report(manifest)
    from dki import dmcompat

    source_binding = _runtime_source_binding()
    source_tree = (
        source_binding["included_tree_sha256"] if source_binding is not None else None
    )
    agent_enabled = bool(args.agent)
    offline = bool(args.offline)
    if offline and not agent_enabled:
        raise ValueError("offline mode is valid only when agent mode is enabled")
    agent_trace = _load_demo_agent_observations(
        agent_enabled=agent_enabled,
        offline=offline,
    )
    datamate_root = getattr(dmcompat, "_DATAMATE_RUNTIME_DIR", None)
    task_report_binding = {
        "path": str(task_report_path),
        "bytes": task_report_path.stat().st_size,
        "sha256": _sha256_file(task_report_path),
    }
    manifest_binding = {
        "path": str(manifest_path.resolve()),
        "bytes": manifest_path.stat().st_size,
        "sha256": _sha256_file(manifest_path),
    }
    artifact_paths = {
        "kg": OUTPUT_DIR / "kg" / "medical_kg.json",
        "kg_provenance": OUTPUT_DIR / "kg" / "medical_kg.provenance.json",
        "database": OUTPUT_DIR / "insight.db",
        "database_provenance": OUTPUT_DIR / "insight.provenance.json",
        "bi_report": OUTPUT_DIR / "reports" / "insight_report.html",
    }
    artifacts = {
        name: _demo_artifact_binding(path, label=name)
        for name, path in artifact_paths.items()
    }
    payload = {
        "schema_version": 1,
        "kind": "dki_demo_runtime_receipt",
        "status": "completed",
        "source_tree_sha256": source_tree,
        "source": source_binding,
        "evidence_eligible": source_binding is not None,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_snapshot": _manifest_input_snapshot(manifest),
        "task1_manifest": manifest_binding,
        "task1_report": task_report_binding,
        "artifacts": artifacts,
        "runtime": {
            "agent_enabled": agent_enabled,
            "core_agent": agent_trace.get("core_agent") if agent_enabled else None,
            "model_backend": agent_trace.get("model_backend") if agent_enabled else None,
            "model_implementation": (
                agent_trace.get("model_implementation") if agent_enabled else None
            ),
            "offline": offline,
            "datamate_mode": (
                "real-datamate" if dmcompat.DATAMATE_AVAILABLE else "compat-shim"
            ),
            "datamate_available": bool(dmcompat.DATAMATE_AVAILABLE),
            "datamate_runtime": str(datamate_root) if datamate_root else None,
        },
        "agent_trace": agent_trace,
        "pipeline": {
            "status": task_report.get("status"),
            "n_input": task_report.get("n_input"),
            "n_output": task_report.get("n_output"),
            "n_failed": task_report.get("n_failed"),
            "n_filtered": task_report.get("n_filtered"),
            "duration_s": task_report.get("duration_s"),
            "operators": task_report.get("op_metrics", []),
            "recovery_events": task_report.get("recovery_events", []),
        },
        "privacy": {
            "contains_secrets": False,
            "contains_private_blind_data": False,
            "contains_full_user_input": False,
        },
    }
    output = OUTPUT_DIR / "demo_runtime.json"
    _publish_json_once(output, payload)
    return output


def cmd_demo_all(args):
    """端到端演示：数据 → 清洗 → 图谱 → 分析报告。"""
    ns = argparse.Namespace
    if args.offline and not args.agent:
        raise ValueError("demo-all --offline requires --agent")
    if not any(RAW_DATA_DIR.glob("*.txt")):
        cmd_gen_data(ns(n=200, seed=42))
    data_agent_already_run = False
    if args.agent and args.offline:
        from dki.pipeline.artifacts import load_task1_manifest
        from dki.pipeline.demo_ops import register_demo_transient_operator

        try:
            previous_task_id = str(load_task1_manifest()["task_id"])
        except (FileNotFoundError, ValueError):
            previous_task_id = ""
        register_demo_transient_operator()
        _agent_mode(
            "data",
            "规划并执行任务一 ETL，跟踪异步状态，展示瞬时故障恢复后发布 manifest",
            offline=True,
        )
        current_task_id = str(load_task1_manifest()["task_id"])
        if current_task_id == previous_task_id:
            raise RuntimeError("离线 DataProcessingAgent 未发布新的任务一 manifest")
        data_agent_already_run = True
    else:
        cmd_task1(ns(agent=False, task=None, input=str(RAW_DATA_DIR),
                     config="configs/pipeline_medical_etl.yaml", pattern="*.txt"))
    cmd_task2(ns(agent=False, task=None, input=None,
                 config="configs/pipeline_kg_build.yaml", pattern="*.txt", eval=True))
    cmd_task2_qa(ns(question="高血压有哪些症状？"))
    cmd_task2_qa(ns(question="2型糖尿病有什么并发症？"))
    cmd_task3(ns(agent=False, task=None, ask=None, report=True))
    cmd_eval_nl2sql(ns(engine="rule"))
    if args.agent:
        agent_tasks = [
            ("kg", "读取当前知识图谱的实体和关系分布"),
            ("insight", "用 NL2SQL 复查各科室就诊量"),
        ]
        if not data_agent_already_run:
            agent_tasks.insert(0, ("data", "审阅任务一 manifest 和已发布的算子链"))
        for which, task in agent_tasks:
            _agent_mode(which, task, offline=args.offline)
    runtime_receipt = _publish_demo_runtime(args)
    _print_header("端到端演示完成")
    print(f"产物目录: {OUTPUT_DIR}")
    print(f"  - 任务一清单 outputs/processed/manifest.json")
    print(f"  - 知识图谱   outputs/kg/medical_kg.json")
    print(f"  - 分析报告   outputs/reports/insight_report.html")
    print(f"  - 运行时评测 outputs/runtime_eval/")
    print(f"  - 结构化回执 {runtime_receipt}")


def _installed_streamlit_version() -> str | None:
    try:
        return importlib.metadata.version("streamlit")
    except importlib.metadata.PackageNotFoundError:
        return None


def _streamlit_environment(target: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(target).resolve()) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _install_and_verify_streamlit(demo_root: Path) -> tuple[Path, Path, dict, dict]:
    """Install and import the exact Streamlit pin from the RID-local target."""

    target = Path(demo_root) / ".demo_deps"
    target.mkdir(parents=True, exist_ok=False)
    install = subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "--disable-pip-version-check", "--no-input",
            "--target", str(target), "streamlit==1.54.0",
        ],
        cwd=PROJECT_ROOT,
    )
    if install.returncode != 0:
        raise RuntimeError("unable to install RID-local streamlit==1.54.0")

    probe_code = r'''
import hashlib, importlib.metadata, json
from pathlib import Path
import streamlit
dist = importlib.metadata.distribution("streamlit")
def digest(path):
    if path is None or not Path(path).is_file():
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
record = metadata = None
for item in dist.files or ():
    text = str(item).replace("\\", "/")
    if text.endswith(".dist-info/RECORD"):
        record = dist.locate_file(item)
    elif text.endswith(".dist-info/METADATA"):
        metadata = dist.locate_file(item)
payload = {
    "version": importlib.metadata.version("streamlit"),
    "module_path": str(Path(streamlit.__file__).resolve()),
    "distribution_path": str(Path(dist._path).resolve()),
    "record_sha256": digest(record),
    "metadata_sha256": digest(metadata),
}
print("DKI_STREAMLIT_PROBE=" + json.dumps(payload, sort_keys=True))
'''
    env = _streamlit_environment(target)
    probe = subprocess.run(
        [sys.executable, "-c", probe_code],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if probe.returncode != 0:
        raise RuntimeError("RID-local Streamlit import probe failed")
    prefix = "DKI_STREAMLIT_PROBE="
    encoded = next(
        (line[len(prefix):] for line in probe.stdout.splitlines() if line.startswith(prefix)),
        None,
    )
    try:
        runtime = json.loads(encoded) if encoded is not None else None
    except json.JSONDecodeError as exc:
        raise RuntimeError("RID-local Streamlit probe returned invalid JSON") from exc
    if not isinstance(runtime, dict) or runtime.get("version") != "1.54.0":
        raise RuntimeError("RID-local Streamlit version is not exactly 1.54.0")
    try:
        Path(runtime["module_path"]).resolve(strict=True).relative_to(target.resolve())
        Path(runtime["distribution_path"]).resolve(strict=True).relative_to(target.resolve())
    except (KeyError, FileNotFoundError, ValueError) as exc:
        raise RuntimeError("Streamlit was imported outside the RID-local target") from exc
    for field in ("record_sha256", "metadata_sha256"):
        value = runtime.get(field)
        if value is not None:
            _strict_sha256(value, source=f"Streamlit {field}")
    receipt = Path(demo_root) / "streamlit_runtime.json"
    _publish_json_once(receipt, {
        "schema_version": 1,
        "kind": "dki_demo_streamlit_runtime",
        "status": "verified",
        "python": sys.executable,
        "target": str(target.resolve()),
        **runtime,
    })
    return target.resolve(), receipt.resolve(), runtime, env


def _python_device_runtime() -> dict:
    payload = {
        "python": sys.version.split()[0],
        "torch": None,
        "torch_npu": None,
        "npu_available": False,
        "device_name": None,
        "error_type": None,
    }
    try:
        import torch

        payload["torch"] = str(torch.__version__)
        try:
            import torch_npu

            payload["torch_npu"] = str(torch_npu.__version__)
            available = bool(hasattr(torch, "npu") and torch.npu.is_available())
            payload["npu_available"] = available
            if available:
                payload["device_name"] = str(torch.npu.get_device_name(0))
        except (ImportError, RuntimeError, OSError) as exc:
            payload["error_type"] = type(exc).__name__
    except (ImportError, RuntimeError, OSError) as exc:
        payload["error_type"] = type(exc).__name__
    return payload


def _capture_demo_device_snapshot(path: Path, *, streamlit_runtime: dict) -> Path:
    """Create a sanitised current-device snapshot without exposing env values."""

    binary = shutil.which("npu-smi")
    npu_smi = {
        "binary_available": binary is not None,
        "returncode": None,
        "stdout_sha256": None,
        "summary_lines": [],
        "error_type": None,
    }
    if binary is not None:
        try:
            result = subprocess.run(
                [binary, "info"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
            )
            stdout = result.stdout or ""
            summary_lines = []
            for line in stdout.splitlines():
                if not line.strip():
                    continue
                if re.search(
                    r"(?:api[_-]?key|authorization|password|secret|token)",
                    line,
                    flags=re.IGNORECASE,
                ):
                    summary_lines.append("[redacted-sensitive-line]")
                else:
                    summary_lines.append(line[:500])
                if len(summary_lines) >= 80:
                    break
            npu_smi.update({
                "returncode": int(result.returncode),
                "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
                "summary_lines": summary_lines,
            })
        except (OSError, subprocess.SubprocessError) as exc:
            npu_smi["error_type"] = type(exc).__name__
    python_runtime = _python_device_runtime()
    summary_text = "\n".join(npu_smi["summary_lines"])
    smi_detected = bool(
        npu_smi["returncode"] == 0
        and re.search(r"(?:Ascend\s*910|Ascend910|\b910[A-Z0-9_]*\b)", summary_text, flags=re.IGNORECASE)
    )
    torch_detected = bool(python_runtime.get("npu_available"))
    detected = smi_detected or torch_detected
    detected_name = python_runtime.get("device_name") if torch_detected else None
    detection_sources = [
        source
        for source, active in (
            ("torch_npu", torch_detected),
            ("npu-smi", smi_detected),
        )
        if active
    ]
    payload = {
        "schema_version": 2,
        "kind": "dki_demo_device_snapshot",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "requested_logical_device": "npu:0",
        "detected_available": detected,
        "detected_device_name": detected_name,
        "environment_label": "910C 验收环境" if detected else None,
        "detection_sources": detection_sources,
        "streamlit_runtime": {
            "version": streamlit_runtime.get("version"),
            "module_path": streamlit_runtime.get("module_path"),
            "distribution_path": streamlit_runtime.get("distribution_path"),
            "record_sha256": streamlit_runtime.get("record_sha256"),
            "metadata_sha256": streamlit_runtime.get("metadata_sha256"),
        },
        "commands": {
            "npu_smi": npu_smi,
            "python_runtime": python_runtime,
        },
        "privacy": {
            "contains_environment": False,
            "contains_secrets": False,
        },
    }
    _publish_json_once(path, payload)
    return Path(path).resolve()


def cmd_demo_ui(args):
    """Launch the optional Streamlit evidence cockpit.

    The UI itself is a separate process so that its per-run ``DKI_OUTPUT_DIR``
    can never be confused with the CLI process that launches it.
    """
    rid = args.rid or (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-ui"
    )
    # Both checks are deliberately read-only and precede demo-root creation or
    # dependency installation.  A port race after the probe remains possible,
    # but an already occupied port cannot leave a misleading evidence folder.
    source_binding = _resolve_demo_source_binding(args)
    source_tree_sha256 = source_binding["included_tree_sha256"]
    _preflight_demo_port(args.host, args.port)
    app = PROJECT_ROOT / "dki" / "demo" / "app.py"
    if not app.is_file():
        raise FileNotFoundError(f"demo app not found: {app}")
    demo_root = OUTPUT_DIR / "demo_runs" / rid
    demo_root.mkdir(parents=True, exist_ok=False)
    demo_deps, streamlit_receipt, streamlit_runtime, env = (
        _install_and_verify_streamlit(demo_root)
    )
    device_snapshot = _capture_demo_device_snapshot(
        demo_root / "device_snapshot.json",
        streamlit_runtime=streamlit_runtime,
    )
    env.pop("DKI_OUTPUT_DIR", None)
    env["DKI_DEMO_ROOT"] = str(demo_root)
    env["DKI_DEMO_SERVER_RID"] = rid
    env["DKI_DEMO_PYTHON"] = sys.executable
    env["DKI_INCLUDED_TREE_SHA256"] = source_tree_sha256
    env["DKI_INCLUDED_TREE_MANIFEST"] = source_binding["manifest_path"]
    env["DKI_INCLUDED_TREE_MANIFEST_SHA256"] = source_binding["manifest_sha256"]
    env["DKI_DEMO_DEVICE_SNAPSHOT"] = str(device_snapshot)
    env["DKI_DEMO_STREAMLIT_RECEIPT"] = str(streamlit_receipt)
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--global.developmentMode",
        "false",
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    print(f"[demo-ui] verified RID-local streamlit 1.54.0: {demo_deps}")
    print(f"[demo-ui] run root: {demo_root}")
    print(f"[demo-ui] listening on {args.host}:{args.port}")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    raise SystemExit(completed.returncode)


# ---------------------------------------------------------------- 参数解析
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dki", description="数据—知识—洞察 智能体套件")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("gen-data", help="生成合成医疗数据集")
    g.add_argument("--n", type=int, default=200)
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(func=cmd_gen_data)

    t1 = sub.add_parser("task1", help="任务一：数据处理流水线")
    t1.add_argument("--input", default=str(RAW_DATA_DIR))
    t1.add_argument("--config", default="configs/pipeline_medical_etl.yaml")
    t1.add_argument("--pattern", default="*.txt")
    t1.add_argument("--agent", action="store_true", help="用 Nexent 智能体自主编排（需 LLM）")
    t1.add_argument("--offline", action="store_true", help="使用 ScriptedModel 离线演示智能体工具调用")
    t1.add_argument("--task", default=None, help="agent 模式下的自然语言任务")
    t1.set_defaults(func=cmd_task1)

    t2 = sub.add_parser("task2", help="任务二：知识图谱构建")
    t2.add_argument("--input", default=None, help="仅允许当前任务一 manifest 的清洗文本目录")
    t2.add_argument("--config", default="configs/pipeline_kg_build.yaml")
    t2.add_argument("--pattern", default="*.txt")
    t2.add_argument("--eval", action="store_true", help="对金标准评测抽取质量")
    t2.add_argument("--agent", action="store_true")
    t2.add_argument("--offline", action="store_true")
    t2.add_argument("--task", default=None)
    t2.set_defaults(func=cmd_task2)

    qa = sub.add_parser("task2-qa", help="任务二：图谱问答")
    qa.add_argument("--question", required=True)
    qa.set_defaults(func=cmd_task2_qa)

    t3 = sub.add_parser("task3", help="任务三：数据分析与 BI 报告")
    t3.add_argument("--ask", default=None, help="自然语言分析问题（NL2SQL）")
    t3.add_argument("--report", action="store_true", help="生成完整 BI 报告（默认行为）")
    t3.add_argument("--agent", action="store_true")
    t3.add_argument("--offline", action="store_true")
    t3.add_argument("--task", default=None)
    t3.set_defaults(func=cmd_task3)

    e1 = sub.add_parser("eval-nl2sql", help="NL2SQL 准确率评测")
    e1.add_argument("--engine", default="rule", choices=["rule", "llm", "auto"])
    e1.set_defaults(func=cmd_eval_nl2sql)

    e2 = sub.add_parser("eval-kg", help="知识抽取质量评测")
    e2.set_defaults(func=cmd_eval_kg)

    b1 = sub.add_parser("bench-ops", help="算子吞吐/延迟基准")
    b1.add_argument("--repeat", type=int, default=3)
    b1.add_argument("--limit", type=int, default=100)
    b1.set_defaults(func=cmd_bench_ops)

    b2 = sub.add_parser("bench-ner", help="BERT NER 后端基准（CPU/NPU）")
    b2.add_argument("--device", default="cpu", choices=["cpu", "cuda", "npu"])
    b2.add_argument("--batch", default="8")
    b2.add_argument("--iters", type=int, default=20)
    b2.set_defaults(func=cmd_bench_ner)

    a = sub.add_parser("agent", help="Nexent 智能体模式（需 LLM 配置）")
    a.add_argument("--which", default="data", choices=["data", "kg", "insight"])
    a.add_argument("--task", required=True)
    a.add_argument("--offline", action="store_true", help="使用 ScriptedModel，不读取 LLM 环境变量")
    a.set_defaults(func=cmd_agent)

    d = sub.add_parser("demo-all", help="端到端演示（数据→知识→洞察）")
    d.add_argument("--agent", action="store_true", help="在端到端产物完成后展示三类智能体工具轨迹")
    d.add_argument("--offline", action="store_true", help="demo-all --agent 时使用 ScriptedModel")
    d.set_defaults(func=cmd_demo_all)

    ui = sub.add_parser("demo-ui", help="启动中文证据驾驶舱（需要 dki-agents[demo]）")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=7860)
    ui.add_argument("--rid", default=None, help="显式服务 RID；目录存在时直接失败")
    ui.add_argument(
        "--included-tree-manifest",
        type=Path,
        default=None,
        help="显式绑定 v1 included_tree_manifest.json；未设置时按环境变量/RID/FINAL_STATUS 解析",
    )
    ui.set_defaults(func=cmd_demo_ui)
    return p


def main(argv=None):
    ensure_dirs()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
