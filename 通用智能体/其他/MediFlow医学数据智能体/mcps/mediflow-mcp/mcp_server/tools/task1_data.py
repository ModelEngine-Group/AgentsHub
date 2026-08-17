"""
任务一 DataMate 数据清洗 MCP 工具。
"""

import os, re, json, time, subprocess, tempfile, shutil, sys
from pathlib import Path
from mcp_server.tools import mcp
from mcp_server.config import DATAMATE_BASE, DATASET_VOLUME, SUDO_PW, KG_DB
_DATASET_VOLUME = DATASET_VOLUME  # 兼容旧变量名
ROOT = Path(__file__).resolve().parents[2]
from mcp_server.datamate.client import dm_post, dm_get, gateway_get, write_temp_dataset, _sudo_command
from mcp_server.datamate.resolver import _task2_resolve_datamate_dataset, _task2_read_datamate_dataset
from mcp_server.task1.dataset_service import (
    inspect_datamate_dataset as _inspect_datamate_dataset,
    upload_text_dataset as _upload_text_dataset,
)
from mcp_server.task1.chains import (
    task1_mixed_chain_descriptions as _task1_mixed_chain_descriptions,
    task1_mixed_chain_map as _task1_mixed_chain_map,
)
from mcp_server.task1.datasets import (
    TASK1_FILE_GROUPS,
    classify_source_file,
    count_source_file_groups,
    datamate_dataset_host_path,
)
from mcp_server.task1.evidence import summarize_cleaning_evidence
from mcp_server.task1.inspection import (
    build_preview_samples,
    recommend_chain,
    summarize_file_types,
)
from mcp_server.task1.postprocess import postprocess_output as _task1_postprocess_output
from mcp_server.task1.pipeline_service import (
    get_datamate_cleaning_result as _get_datamate_cleaning_result,
    run_datamate_cleaning_pipeline as _run_datamate_cleaning_pipeline,
)
from mcp_server.task1.mixed_cleaning_service import (
    run_task1_mixed_cleaning_service as _run_task1_mixed_cleaning_service,
)
from mcp_server.task1.status import (
    task1_async_status_path as _task1_async_status_path,
    write_task1_async_status as _write_task1_async_status,
)

_DM_BASE = DATAMATE_BASE
_DEMO_DATASET_ID   = '1b2f84a6-7842-4bce-bb77-e235cfa12c81'
_DEMO_DATASET_NAME = 'CMeEE-V2医疗文本测试集'


# === Nexent 可调用的任务一工具入口 ===
@mcp.tool
def inspect_dataset(dataset_id: str) -> dict:
    """探查数据集构成，供智能体决策走哪条清洗链（结构化表 vs 非结构化文本 vs 多段病历）。

    dataset_id 可以传 DataMate UUID，也可以传数据集完整名称；如果用户给出的名称
    与真实名称只有“清理/清洗、保留/保持来源格式”等轻微差异，会尽量按时间戳兜底解析。

    返回：
      file_count          文件总数
      type_distribution   {文件类型: 数量}，如 {"txt": 8, "csv": 2}
      samples             前若干文件的预览 [{name, type, preview, looks_multi_record}]
      recommendation      "table_chain" | "text_chain" | "pdf_chain" | "mixed"
      multi_record_hint   是否检测到文件内含多段病历（建议先分段）

    智能体据此选择工具：
      table_chain 使用 DataMate 表格清洗链；
      text_chain 使用 DataMate 文本清洗链；
      pdf_chain 使用 MinerU 先提取 TXT，再执行文本清洗链；
      mixed 使用 run_task1_mixed_cleaning 按类型分批处理。
    """
    return _inspect_datamate_dataset(dataset_id, dataset_volume=_DATASET_VOLUME)


@mcp.tool
def upload_text_to_datamate(text: str, name: str = "user_input") -> dict:
    """将用户提供的医疗文本注册为 DataMate 数据集，返回真实可用的 dataset_id。
    本工具接收 text 和 name 两个参数；text 必须是用户原始文本，name 应为中文数据集名称。
    成功后返回 status、dataset_id、dataset_name、char_count、file_name。
    下一步通常选择 inspect_dataset 检查数据集，或选择 run_task1_mixed_cleaning 执行任务一清洗。"""
    return _upload_text_dataset(text, name, dataset_volume=_DATASET_VOLUME)


@mcp.tool
def list_datamate_operators(keyword: str = "", category: str = "", page: int = 0) -> dict:
    """列出 DataMate 算子市场中可用的算子，返回名称、描述和功能分类，供智能体理解并自主选择。

    【推荐查询方式】优先用 category 参数按大类筛选，比关键词更精准：
      category="用户上传"  — 自研医疗专用算子（4个）：术语标准化、实体识别、关系抽取、三元组生成
      category="清洗"      — DataMate内置文本清洗算子（200+个），含乱码去除、HTML清理、全角转半角等
      category="DataJuicer"— DataJuicer集成算子，含去重、质量过滤、文本分析等
      category=""          — 不过滤，返回全部（每页100个，用 page 参数翻页）

    【关键词搜索】keyword 在算子名称和描述中全文搜索，可与 category 组合使用
      例：category="清洗", keyword="HTML" → 只看清洗类里和HTML相关的算子

    参数：
      keyword:  关键词（留空=不过滤）
      category: 大类筛选（见上方可选值，留空=全部类别）
      page:     页码，从0开始，每页100条

    返回：{page, total_count, page_count, count, has_more, category_hint,
           operators: [{id, name, description, categories, inputs, outputs}]}"""
    import requests as _req, math as _math

    _CAT_MAP = {
        "清洗":       "8c09476a-a922-418f-a908-733f8a0de521",
        "用户上传":   "ec2cdd17-8b93-4a81-88c4-ac9e98d10757",
        "系统预置":   "96a3b07a-3439-4557-a835-525faad60ac3",
        "DataMate":   "431e7798-5426-4e1a-aae6-b9905a836b34",
        "DataJuicer": "79b385b4-fde8-4617-bcba-02a176938996",
        "文本":       "d8a5df7a-52a9-42c2-83c4-01062e60f597",
    }
    _ID_TO_CAT = {v: k for k, v in _CAT_MAP.items()}

    PAGE_SIZE = 100
    payload = {"page": page, "size": PAGE_SIZE}
    if keyword:
        payload["keyword"] = keyword
    if category and category in _CAT_MAP:
        payload["categories"] = [[_CAT_MAP[category]]]

    r = _req.post(f"{_DM_BASE}/api/operators/list", json=payload, timeout=10)
    if not r.ok:
        return {"error": f"HTTP {r.status_code}", "total_count": 0, "count": 0, "operators": []}

    data = r.json().get("data", {})
    items = data.get("content", []) if isinstance(data, dict) else []
    total = data.get("totalElements", len(items)) if isinstance(data, dict) else len(items)
    page_count = _math.ceil(total / PAGE_SIZE) if total else 1

    def _fmt(op):
        cat_ids = op.get("categories") or []
        cat_names = [_ID_TO_CAT.get(c, c) for c in cat_ids if c in _ID_TO_CAT]
        return {
            "id":          op.get("id", ""),
            "name":        op.get("name", ""),
            "description": op.get("description", ""),
            "categories":  cat_names,
            "inputs":      op.get("inputs", ""),
            "outputs":     op.get("outputs", ""),
        }

    result = [_fmt(op) for op in items]
    hint = (f"第{page+1}/{page_count}页，共{total}个算子"
            + (f"（分类：{category}）" if category else "")
            + (f"（关键词：{keyword}）" if keyword else ""))
    return {
        "page":          page,
        "total_count":   total,
        "page_count":    page_count,
        "count":         len(result),
        "has_more":      (page + 1) < page_count,
        "category_hint": hint,
        "operators": result if result else [{"note": "未找到匹配算子，请换关键词或分类重试"}],
    }


def _postprocess_output(dest_dataset_id: str, task_id: str) -> dict:
    """修复 DataMate 输出的三类瑕疵 + 给输出文件加区分名：
       1) Ray 多块写入会产生 0B 幻影结果记录（磁盘上不存在），清掉
          t_clean_result / t_dm_dataset_files 里 dest_size/file_size=0 的记录
       2) Ray 多块写入有时产生内容完全相同的重复块 → 按内容去重，只留一个
       3) 输出文件默认与源文件同名 → 加 .cleaned 后缀，明确区分"处理后"
     返回 {removed_phantoms, removed_dups, renamed:[[old,new]], real_count}"""
    return _task1_postprocess_output(dest_dataset_id, task_id, _DATASET_VOLUME, _sudo_command)


@mcp.tool
def run_datamate_pipeline(dataset_id: str = "", task_name: str = "",
                          operators: list = None,
                          with_llm_filter: bool = False,
                          output_jsonl: bool = False) -> dict:
    """为已有 DataMate 数据集执行清洗流水线。

    dataset_id must be a real DataMate dataset UUID or resolvable dataset name.
    operators can explicitly override the default Task 1 operator chain.
    with_llm_filter appends LLMNoiseFilter when it is not already present.
    output_jsonl appends UnifiedJsonlExporter unless the chain already produces JSONL.
    """
    return _run_datamate_cleaning_pipeline(
        dataset_id=dataset_id,
        task_name=task_name,
        operators=operators,
        with_llm_filter=with_llm_filter,
        output_jsonl=output_jsonl,
        datamate_base=_DM_BASE,
        postprocess=_postprocess_output,
    )


@mcp.tool
def run_task1_mixed_cleaning(
    dataset_id: str,
    task_name: str = "",
    wait: bool = True,
    async_file_threshold: int = 50,
) -> dict:
    """执行任务一混合格式数据集清洗编排。

    Use this tool for mixed txt/csv/json/jsonl/pdf DataMate datasets. It keeps
    source formats except that PDF is explicitly converted to TXT, runs per-type chains,
    and returns lineage, quality evidence and DataMate task IDs. The normal Nexent
    path waits for the final result in this same request. Use wait=False only for
    an explicitly non-blocking caller; that path returns a real run_id for the
    status tool and is not the normal Agent conversation path.
    """
    return _run_task1_mixed_cleaning_service(
        dataset_id=dataset_id,
        task_name=task_name,
        wait=wait,
        async_file_threshold=async_file_threshold,
    )


@mcp.tool
def get_task1_mixed_cleaning_status(run_id: str) -> dict:
    """查询任务一混合清洗异步任务状态。

    该工具只服务于调用方明确选择 wait=False 的非阻塞任务。正常的
    run_task1_mixed_cleaning 调用会在同一次请求中等待最终结果。拿到 run_id
    后调用本工具，可获得：
    - queued/running/success/error 状态；
    - 源数据集和文件类型分布；
    - 每类格式实际使用的算子链；
    - 完成后的最终 DataMate 数据集、子任务 ID、质量报告。

    如果 status 仍是 queued/running，最终回答应明确告诉用户“后台仍在处理”，
    并展示 operators_plan 与 next_action，不要编造最终数据集 ID。
    """
    rid = (run_id or "").strip()
    if not rid:
        return {"status": "error", "error": "run_id is required"}
    status_path = _task1_async_status_path(rid)
    if not status_path.exists():
        return {
            "status": "not_found",
            "run_id": rid,
            "error": f"async status file not found: {status_path}",
        }
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "error", "run_id": rid, "error": f"failed to read status: {exc}"}
    payload.setdefault("run_id", rid)
    if payload.get("status") in {"queued", "running", "async_started"}:
        payload.setdefault("next_action", "call get_task1_mixed_cleaning_status with the returned run_id")
    return payload


@mcp.tool
def get_datamate_result(task_id: str) -> dict:
    """执行任务一混合格式数据集清洗编排。"""
    return _get_datamate_cleaning_result(task_id, datamate_base=_DM_BASE)


if __name__ == "__main__":
    host = CFG.get("mcp_server", {}).get("host", "0.0.0.0")
    port = CFG.get("mcp_server", {}).get("port", 8900)
    # streamable-http transport，默认路径 /mcp，Nexent 以 StreamableHttpTransport 连接
    mcp.run(transport="streamable-http", host=host, port=port)
