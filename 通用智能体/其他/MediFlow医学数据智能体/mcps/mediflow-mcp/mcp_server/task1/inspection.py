"""
任务一数据集探查模块。
"""

from __future__ import annotations

import re
from collections import Counter

from mcp_server.task1.datasets import datamate_dataset_host_path


MULTI_RECORD_PATTERN = re.compile(r"患者[:：]|主诉[:：]|病历号|住院号|姓名[:：]|病例\s*\d|={3,}|-{3,}")


def recommend_chain(file_types: set[str], multi_record_hint: bool = False) -> tuple[str, str]:
    """返回 DataMate 清洗链推荐和用户可读提示。"""

    normalized_types = {value.lower() for value in file_types}
    if normalized_types and normalized_types <= {"pdf"}:
        recommendation = "pdf_chain"
    elif normalized_types <= {"csv", "xlsx", "xls"}:
        recommendation = "table_chain"
    elif normalized_types <= {"json", "jsonl"}:
        recommendation = "json_chain"
    elif "pdf" in normalized_types or normalized_types & {"csv", "xlsx", "xls"}:
        recommendation = "mixed"
    else:
        recommendation = "text_chain"

    advice = {
        "table_chain": "纯表格：选择 DataMate 表格清洗链，默认保持 CSV 格式",
        "json_chain": "JSON/JSONL：选择 DataMate JSON 字段清洗链，默认保持 JSON/JSONL 格式",
        "pdf_chain": "PDF：先用 MinerU 提取为 TXT，再进入文本清洗链；执行前必须确认 PDF 解析服务可用",
        "text_chain": (
            "纯文本：选择 DataMate 文本清洗链，默认保持 TXT 格式"
            + ("；仅在明确要求切分记录或交给任务二统一入口时，才追加病历分段" if multi_record_hint else "")
        ),
        "mixed": "混合：优先选择 run_task1_mixed_cleaning，按文件类型分批处理；PDF 先转换为 TXT，其余格式保持源格式",
    }.get(recommendation, "")
    return recommendation, advice


def summarize_file_types(rows: list[list[str]]) -> dict[str, int]:
    """按任务一实际可分派的格式统计，兼容 MIME 类型声明。"""
    from mcp_server.task1.datasets import classify_source_file

    values = []
    for row in rows:
        file_name = row[0] if row else ""
        declared = row[2] if len(row) >= 3 else ""
        _group, output_type = classify_source_file(file_name, declared)
        values.append(output_type or declared.lower() or "unknown")
    return dict(Counter(values))


def build_preview_samples(
    rows: list[list[str]],
    dataset_volume: str,
    dataset_id: str,
    read_file,
    limit: int = 3,
) -> tuple[list[dict], bool]:
    """生成文件预览并识别多记录文本标记。"""

    samples: list[dict] = []
    multi_hint = False
    for row in rows[:limit]:
        fname = row[0]
        stored_path = row[1] if len(row) >= 2 else ""
        ftype = row[2] if len(row) >= 3 else ""
        is_pdf = ftype.lower() == "pdf" or fname.lower().endswith(".pdf")
        host_path = datamate_dataset_host_path(dataset_volume, dataset_id, fname, stored_path)
        raw = "" if is_pdf else (read_file(host_path) or "")
        n_sig = len(MULTI_RECORD_PATTERN.findall(raw))
        is_multi = n_sig >= 2
        if is_multi:
            multi_hint = True
        samples.append(
            {
                "name": fname,
                "type": ftype,
                "preview": "PDF 文件，执行时将先提取文本" if is_pdf else raw[:200],
                "looks_multi_record": is_multi,
            }
        )
    return samples, multi_hint
