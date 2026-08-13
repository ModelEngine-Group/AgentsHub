"""任务一工具集：算子目录 / 数据预览 / 流水线执行 / 任务状态。"""

import json
from pathlib import Path
from typing import Optional

import yaml
from smolagents.tools import Tool

from dki.paths import DATA_DIR, OUTPUT_DIR, PROCESSED_DIR, PROJECT_ROOT, RAW_DATA_DIR


def _err(error: str, hint: str = "", **extra) -> str:
    """失败返回。``extra`` 用于携带结构化标记（如 terminal），供智能体判定是否可重试。"""
    return json.dumps(
        {"ok": False, "error": error, "hint": hint, **extra},
        ensure_ascii=False,
        default=str,
    )


def _ok(**payload) -> str:
    return json.dumps({"ok": True, **payload}, ensure_ascii=False, default=str)


def _safe_path(path_str: str) -> Path:
    """限制智能体只能访问 data/ 与 outputs/（防越界）。"""
    p = (PROJECT_ROOT / path_str).resolve() if not Path(path_str).is_absolute() else Path(path_str).resolve()
    for allowed in (DATA_DIR.resolve(), OUTPUT_DIR.resolve()):
        if p == allowed or allowed in p.parents:
            return p
    raise PermissionError(f"路径越界：只允许访问 data/ 与 outputs/ 目录，收到 {path_str}")


def _notify(observer, text: str):
    if observer is not None:
        try:
            from nexent.core.utils.observer import ProcessType

            observer.add_message("", ProcessType.TOOL, text)
        except Exception:
            pass


class ListOperatorsTool(Tool):
    name = "list_operators"
    description = (
        "列出可用数据处理算子：本项目 5 个医疗算子（含参数说明）与 DataMate 内置算子名清单。"
        "可选按类别过滤。返回 JSON。"
    )
    inputs = {
        "category": {
            "type": "string",
            "description": "过滤类别：medical（本项目医疗算子）/ builtin（DataMate内置）/ all（默认）",
            "nullable": True,
        }
    }
    output_type = "string"

    def __init__(self, observer=None):
        super().__init__()
        self.observer = observer

    def forward(self, category: Optional[str] = None) -> str:
        try:
            _notify(self.observer, "查询算子目录")
            from dki import dmcompat
            from dki.dmcompat import OPERATORS

            dmcompat.load_builtin_ops()
            dmcompat.register_dki_ops()

            medical = []
            ops_dir = Path(__file__).resolve().parent.parent.parent / "operators"
            for meta_path in sorted(ops_dir.glob("*/metadata.yml")):
                meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
                if meta.get("catalog_visible", True) is False:
                    continue
                settings = {
                    k: {"说明": v.get("description", ""), "默认值": v.get("defaultVal")}
                    for k, v in (meta.get("settings") or {}).items()
                }
                medical.append({
                    "op": meta["raw_id"],
                    "name": meta.get("name", ""),
                    "description": meta.get("description", ""),
                    "params": settings,
                })
            medical_names = {m["op"] for m in medical}
            builtin = sorted(n for n in OPERATORS.modules if n not in medical_names)

            cat = (category or "all").lower()
            payload = {}
            if cat in ("medical", "all"):
                payload["medical_operators"] = medical
            if cat in ("builtin", "all"):
                payload["builtin_operators"] = builtin
            return _ok(**payload)
        except Exception as e:
            return _err(str(e), "检查 DataMate runtime 是否可用；category 仅支持 medical/builtin/all")


class PreviewDataTool(Tool):
    name = "preview_data"
    description = (
        "预览数据：path 为目录时返回文件清单与数量；为文件时返回前 n 个字符。"
        "仅允许访问 data/ 与 outputs/。返回 JSON。"
    )
    inputs = {
        "path": {"type": "string", "description": "目录或文件路径（相对项目根，如 data/raw）"},
        "n": {"type": "integer", "description": "文件预览的字符数，默认 500", "nullable": True},
    }
    output_type = "string"

    def __init__(self, observer=None):
        super().__init__()
        self.observer = observer

    def forward(self, path: str, n: Optional[int] = None) -> str:
        try:
            _notify(self.observer, f"预览数据 {path}")
            p = _safe_path(path)
            n = n or 500
            if p.is_dir():
                files = sorted(p.iterdir())
                return _ok(
                    kind="dir",
                    n_files=len([f for f in files if f.is_file()]),
                    entries=[f.name for f in files[:30]],
                    truncated=len(files) > 30,
                )
            if p.is_file():
                text = p.read_text(encoding="utf-8", errors="replace")[:n]
                return _ok(kind="file", size=p.stat().st_size, preview=text)
            return _err(f"路径不存在: {path}", "先用 preview_data 查看上级目录确认文件名")
        except Exception as e:
            return _err(str(e), "path 需在 data/ 或 outputs/ 内")


class RunPipelineTool(Tool):
    name = "run_pipeline"
    description = (
        "异步提交数据处理流水线（与 DataMate process.yaml 同构语义）。"
        'pipeline 为 JSON 数组字符串，元素 {"op": "算子名", "params": {...}}，按序执行。'
        "立即返回 task_id 与 PENDING/RUNNING 状态；随后必须调用 get_task_status 跟踪完成、失败和恢复动作。"
    )
    inputs = {
        "pipeline": {"type": "string", "description": 'JSON数组：[{"op":"MedicalTextCleaner","params":{}}]'},
        "input_dir": {"type": "string", "description": "输入目录（如 data/raw）"},
        "pattern": {"type": "string", "description": "文件通配符，默认 *.txt", "nullable": True},
        "name": {"type": "string", "description": "任务名（便于追踪）", "nullable": True},
        "retry": {"type": "integer", "description": "单样本失败重试次数，默认0", "nullable": True},
        "parent_task_id": {"type": "string", "description": "基于失败任务重试时填写原 task_id，用于恢复链路追踪", "nullable": True},
    }
    output_type = "string"

    def __init__(self, observer=None):
        super().__init__()
        self.observer = observer

    def forward(self, pipeline: str, input_dir: str, pattern: Optional[str] = None,
                name: Optional[str] = None, retry: Optional[int] = None,
                parent_task_id: Optional[str] = None) -> str:
        try:
            from dki.pipeline import LocalPipelineRunner, resolve_operator, samples_from_dir
            from dki.pipeline.config import _normalize

            process = _normalize(json.loads(pipeline))
            runner = LocalPipelineRunner()
            # Preserve immediate feedback for spelling/registration errors while
            # leaving normal execution asynchronous.
            for item in process:
                resolve_operator(next(iter(item)))
            in_dir = _safe_path(input_dir)
            samples = samples_from_dir(in_dir, pattern=pattern or "*.txt")
            if not samples:
                return _err(f"目录 {input_dir} 下没有匹配 {pattern or '*.txt'} 的文件",
                            "先用 preview_data 确认数据位置")
            _notify(self.observer, f"异步提交流水线（{len(samples)} 个文件，{len(process)} 个算子）")
            rep = runner.submit(process, samples, name=name or "agent-pipeline",
                                retry=retry or 0, parent_task_id=parent_task_id or "")
            return _ok(
                task_id=rep.task_id,
                status=rep.status.value,
                n_input=rep.n_input, n_output=rep.n_output,
                n_failed=rep.n_failed, n_filtered=rep.n_filtered,
                next_action="调用 get_task_status(task_id) 直到任务结束；失败时检查 failure_category/recovery_action",
            )
        except KeyError as e:
            return _err(f"算子未注册: {e}", "先用 list_operators 查看可用算子名（区分大小写）")
        except json.JSONDecodeError as e:
            return _err(f"pipeline 不是合法 JSON: {e}",
                        'pipeline 应形如 [{"op":"MedicalTextCleaner","params":{}}]')
        except Exception as e:
            return _err(str(e), "检查算子参数是否符合 list_operators 给出的说明")


class TaskStatusTool(Tool):
    name = "get_task_status"
    description = (
        "查询流水线任务状态与逐算子指标（吞吐/延迟/失败明细）。用于执行跟踪与失败诊断。返回 JSON。"
    )
    inputs = {
        "task_id": {"type": "string", "description": "run_pipeline 返回的 task_id"},
        "wait_seconds": {"type": "number", "description": "可选等待秒数，最多 5 秒；用于轮询任务状态", "nullable": True},
    }
    output_type = "string"

    def __init__(self, observer=None):
        super().__init__()
        self.observer = observer

    def forward(self, task_id: str, wait_seconds: Optional[float] = None) -> str:
        try:
            from dki.pipeline import TASKS

            if wait_seconds:
                TASKS.wait(task_id, min(5.0, max(0.0, float(wait_seconds))))
            data = TASKS.get(task_id)
            if data is None:
                known = TASKS.list_ids()[-10:]
                return _err(f"任务 {task_id} 不存在", f"最近的任务: {known}")
            report = TASKS.get_report(task_id)
            summary = report.summary() if report else ""
            return _ok(task=data, summary=summary, is_running=TASKS.is_running(task_id))
        except Exception as e:
            return _err(str(e))


class PublishTask1ArtifactsTool(Tool):
    name = "publish_task1_artifacts"
    description = (
        "把已完成的 run_pipeline 任务发布为任务一标准产物：清洗文本、结构化表快照与 manifest。"
        "只接受 COMPLETED/PARTIAL_SUCCESS 任务；任务二和任务三只能消费这个 manifest。"
    )
    inputs = {
        "task_id": {"type": "string", "description": "已完成的 run_pipeline task_id"},
        "source_text_dir": {"type": "string", "description": "原始文本目录，默认 data/raw", "nullable": True},
    }
    output_type = "string"

    def __init__(self, observer=None):
        super().__init__()
        self.observer = observer

    def forward(self, task_id: str, source_text_dir: Optional[str] = None) -> str:
        try:
            from dki.pipeline import TASKS
            from dki.pipeline.artifacts import write_task1_manifest

            report = TASKS.get_report(task_id)
            result = TASKS.get_result(task_id)
            if report is None or result is None:
                data = TASKS.get(task_id)
                status = data.get("status") if data else "UNKNOWN"
                return _err(f"任务 {task_id} 尚未完成（当前 {status}）", "先用 get_task_status 轮询完成")
            if report.status.value not in {"COMPLETED", "PARTIAL_SUCCESS"}:
                return _err(f"任务状态为 {report.status.value}，不可发布产物", "修复失败后重新提交")
            clean_dir = PROCESSED_DIR / "runs" / task_id / "clean"
            clean_dir.mkdir(parents=True, exist_ok=True)
            n = 0
            for sample in result.ok_samples:
                if sample.get("text"):
                    (clean_dir / f"{sample['fileId']}.txt").write_text(sample["text"], encoding="utf-8")
                    n += 1
            manifest = write_task1_manifest(
                task_id=task_id,
                clean_dir=clean_dir,
                source_text_dir=_safe_path(source_text_dir or "data/raw"),
                structured_source_dir=RAW_DATA_DIR / "structured",
            )
            _notify(self.observer, f"已发布任务一 manifest: {manifest}")
            return _ok(task_id=task_id, clean_text_count=n, manifest_path=str(manifest))
        except Exception as exc:
            return _err(str(exc), "确认任务包含文本输出，且 data/raw/structured 中存在 patients.csv 与 visits.csv")
