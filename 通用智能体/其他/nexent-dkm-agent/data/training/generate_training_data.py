"""Generate training data for the task-1 data orchestration small model."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.data_processing_agent.planner import plan_data_task


_REQUEST_TEMPLATES = [
    # Full pipeline (deduplicate + fill missing + normalize + export)
    "请清洗患者CSV，删除重复记录，填补缺失值并导出",
    "清洗CSV数据，去重、填补缺失值、规范化字段类型并导出结果",
    "对这份医疗数据进行去重和缺失值处理",
    "去除重复行，填补空值，统一类型后保存",
    "请帮我处理这份数据，去掉重复记录，填充空白",
    "Deduplicate the CSV, fill missing values, normalize types",
    "Clean the patient data: remove duplicates and fill nulls",
    "去重、填补缺失、类型转换、导出",
    "处理结构化数据，清洗并标准化",
    "Clean and normalize the dataset for analysis",
    "把这份表去重、补空、统一格式然后存起来",
    "数据清洗全流程：去重补缺标准化导出",
    "Perform full data cleaning: dedup, impute, normalize, export",
    "帮我跑一遍数据清洗，去重补缺失改类型导出",
    # Deduplicate only
    "处理CSV文件，删除完全相同的行",
    "仅去重，不需要其他处理",
    "Remove duplicate records from this spreadsheet",
    "去重处理",
    "帮我去掉重复的数据行",
    "只要去重就好",
    "Dedup only please",
    # Fill missing only
    "填补空值并导出",
    "帮我补齐缺失数据",
    "Fill all missing values in this CSV",
    "空值太多了，帮我填一下",
    "缺失值填充",
    # Normalize types only
    "规范化字段类型",
    "统一列的数据类型",
    "Convert column types to proper formats",
    "类型标准化处理",
    "把数值列的类型修正一下",
    # Export / profiling
    "请对数据进行ETL处理",
    "分析数据质量并生成报告",
    "帮我做个数据画像",
    "Profile this dataset and give me a summary",
    # Text processing
    "请清洗文本数据，去除HTML标签和特殊字符",
    "处理医疗文本，抽取诊断和药品信息",
    "Extract medical entities from clinical notes",
    "把HTML和乱码清理掉",
    "文本清洗，去掉标签保留纯文本",
    # Transform
    "转换列名和数据类型",
    "重命名列，筛选符合条件的行",
    "Transform columns: rename patient_id to pid",
    "只要前3列，其他删掉",
    # Mixed intent with different phrasings
    "数据有点脏，帮我整理一下",
    "这份数据需要清洗",
    "数据预处理：清洗和标准化",
    "Preprocess this data file for analysis",
    "帮我做一下数据治理",
    "数据质量太差了，先清洗再导出",
    "I need this CSV cleaned up with dedup and imputation",
    "标准化数据格式并导出干净的数据集",
    "去掉脏数据，补全字段，整理格式",
    "把这份CSV清理干净：删除重复行、补全缺失值、规范类型",
    "跑个清洗流水线：去重→补缺→规范→导出",
    "Data cleaning pipeline: remove dupes, fill nulls, cast types, export",
    # Edge cases
    "这份数据看起来还行，帮我检查一下质量",
    "Just profile and export, no cleaning needed",
    "做一次完整的数据质量评估",
    "数据审计并生成报告",
]

_PROFILE_TEMPLATES = [
    {
        "file_name": "patients.csv",
        "row_count": 100,
        "column_count": 4,
        "duplicate_rows": 5,
        "missing_cells": {"age": 12, "diagnosis": 3},
        "columns": [
            {"name": "patient_id", "inferred_type": "text"},
            {"name": "age", "inferred_type": "integer"},
            {"name": "diagnosis", "inferred_type": "text"},
            {"name": "cost", "inferred_type": "float"},
        ],
    },
    {
        "file_name": "records.csv",
        "row_count": 500,
        "column_count": 3,
        "duplicate_rows": 0,
        "missing_cells": {"name": 8},
        "columns": [
            {"name": "id", "inferred_type": "integer"},
            {"name": "name", "inferred_type": "text"},
            {"name": "score", "inferred_type": "float"},
        ],
    },
    {
        "file_name": "patients.csv",
        "row_count": 50,
        "column_count": 5,
        "duplicate_rows": 10,
        "missing_cells": {"age": 2, "cost": 5, "diagnosis": 1},
        "columns": [
            {"name": "patient_id", "inferred_type": "text"},
            {"name": "age", "inferred_type": "integer"},
            {"name": "diagnosis", "inferred_type": "text"},
            {"name": "cost", "inferred_type": "float"},
            {"name": "admitted", "inferred_type": "boolean"},
        ],
    },
    {
        "file_name": "clean_data.csv",
        "row_count": 200,
        "column_count": 3,
        "duplicate_rows": 0,
        "missing_cells": {},
        "columns": [
            {"name": "id", "inferred_type": "text"},
            {"name": "value", "inferred_type": "float"},
            {"name": "label", "inferred_type": "text"},
        ],
    },
    {
        "file_name": "medical_notes.txt",
        "row_count": 30,
        "column_count": 0,
        "duplicate_rows": 0,
        "missing_cells": {},
        "columns": [],
    },
    {
        "file_name": "hospital_data.csv",
        "row_count": 1000,
        "column_count": 8,
        "duplicate_rows": 50,
        "missing_cells": {"patient_name": 20, "blood_type": 5, "weight": 15},
        "columns": [
            {"name": "patient_id", "inferred_type": "text"},
            {"name": "patient_name", "inferred_type": "text"},
            {"name": "age", "inferred_type": "integer"},
            {"name": "blood_type", "inferred_type": "text"},
            {"name": "weight", "inferred_type": "float"},
            {"name": "height", "inferred_type": "float"},
            {"name": "diagnosis", "inferred_type": "text"},
            {"name": "is_critical", "inferred_type": "boolean"},
        ],
    },
    {
        "file_name": "lab_results.csv",
        "row_count": 300,
        "column_count": 6,
        "duplicate_rows": 3,
        "missing_cells": {"hemoglobin": 10, "glucose": 8},
        "columns": [
            {"name": "sample_id", "inferred_type": "text"},
            {"name": "patient_id", "inferred_type": "text"},
            {"name": "test_date", "inferred_type": "text"},
            {"name": "hemoglobin", "inferred_type": "float"},
            {"name": "glucose", "inferred_type": "float"},
            {"name": "wbc_count", "inferred_type": "integer"},
        ],
    },
    {
        "file_name": "sales.csv",
        "row_count": 5000,
        "column_count": 4,
        "duplicate_rows": 100,
        "missing_cells": {"price": 30, "quantity": 15},
        "columns": [
            {"name": "product_id", "inferred_type": "text"},
            {"name": "price", "inferred_type": "float"},
            {"name": "quantity", "inferred_type": "integer"},
            {"name": "is_active", "inferred_type": "boolean"},
        ],
    },
    {
        "file_name": "survey_data.csv",
        "row_count": 250,
        "column_count": 5,
        "duplicate_rows": 0,
        "missing_cells": {"response": 40, "rating": 12},
        "columns": [
            {"name": "respondent_id", "inferred_type": "text"},
            {"name": "question_id", "inferred_type": "text"},
            {"name": "response", "inferred_type": "text"},
            {"name": "rating", "inferred_type": "integer"},
            {"name": "timestamp", "inferred_type": "text"},
        ],
    },
    {
        "file_name": "clinical_notes.txt",
        "row_count": 50,
        "column_count": 0,
        "duplicate_rows": 0,
        "missing_cells": {},
        "columns": [],
    },
]


def generate_samples(count: int = 2000) -> list[dict[str, str]]:
    """Generate training samples using the rule-based planner as ground truth."""

    samples = []
    for _ in range(count):
        request = random.choice(_REQUEST_TEMPLATES)
        profile = random.choice(_PROFILE_TEMPLATES)

        plan = plan_data_task(request, data_profile=profile)

        output = {
            "operators": plan.operators,
            "rationale": plan.rationale,
            "task_type": plan.understanding.task_type,
            "data_type": plan.understanding.data_type,
            "intent_keywords": plan.understanding.intent_keywords,
            "confidence": plan.confidence,
        }

        # Mirror exactly what inference (local_model_planner.predict_plan) places
        # after "Input: " -- a data_profile-only JSON. The request itself is
        # already carried by the "Task: " line, so it is not duplicated here.
        input_data = {
            "data_profile": {
                k: profile[k]
                for k in ("file_name", "row_count", "column_count", "duplicate_rows", "missing_cells")
                if k in profile
            },
        }

        samples.append({
            "instruction": request,
            "input": json.dumps(input_data, ensure_ascii=False),
            "output": json.dumps(output, ensure_ascii=False),
        })

    return samples


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    all_samples = generate_samples(2000)

    train_samples = all_samples[:1600]
    val_samples = all_samples[1600:]

    train_path = output_dir / "task_orchestration_train.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    val_path = output_dir / "task_orchestration_val.jsonl"
    with val_path.open("w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Generated {len(train_samples)} train samples -> {train_path}")
    print(f"Generated {len(val_samples)} val samples -> {val_path}")


if __name__ == "__main__":
    main()
