"""Generate diverse task-1 orchestration fine-tuning samples (plan phase 2.1)."""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "training" / "task_orchestration_train.jsonl"
VAL_PATH = ROOT / "data" / "training" / "task_orchestration_val.jsonl"

INSTRUCTIONS = [
    "清洗CSV数据，去重、填补缺失值、规范化字段类型并导出结果",
    "仅去重，不需要其他处理",
    "空值太多了，帮我填一下",
    "跳过导出，只做画像分析",
    "部分列缺失严重，删除无用列后导出",
    "文本文件太大，只清洗不重试",
    "异常输入：空文件，给出最小可行计划",
    "全缺失列占比高，建议删列并补缺失",
    "全文本列，抽取实体后导出",
    "条件分支：有重复才去重，否则跳过",
    "重试失败的清洗步骤并导出",
    "超大行数CSV，只做去重和类型规范",
    "Skip dedup when profile shows zero duplicates",
    "Partial clean: impute only, no type normalization",
    "Extract entities from unstructured notes then validate",
]

PROFILES = [
    {
        "file_name": "patients.csv",
        "row_count": 120,
        "column_count": 6,
        "duplicate_rows": 8,
        "missing_cells": {"age": 4, "diagnosis": 2},
        "columns": [
            {"name": "age", "inferred_type": "integer"},
            {"name": "diagnosis", "inferred_type": "text"},
        ],
    },
    {
        "file_name": "clean_data.csv",
        "row_count": 200,
        "column_count": 3,
        "duplicate_rows": 0,
        "missing_cells": {},
        "columns": [
            {"name": "id", "inferred_type": "integer"},
            {"name": "value", "inferred_type": "float"},
        ],
    },
    {
        "file_name": "sparse.csv",
        "row_count": 80,
        "column_count": 4,
        "duplicate_rows": 0,
        "missing_cells": {"notes": 70, "comment": 65, "score": 10},
        "columns": [
            {"name": "notes", "inferred_type": "text"},
            {"name": "comment", "inferred_type": "text"},
            {"name": "score", "inferred_type": "float"},
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
        "file_name": "empty.csv",
        "row_count": 0,
        "column_count": 0,
        "duplicate_rows": 0,
        "missing_cells": {},
        "columns": [],
    },
    {
        "file_name": "wide.csv",
        "row_count": 50000,
        "column_count": 40,
        "duplicate_rows": 1200,
        "missing_cells": {"col_a": 100},
        "columns": [{"name": f"col_{i}", "inferred_type": "text"} for i in range(5)],
    },
    {
        "file_name": "notes_only.csv",
        "row_count": 400,
        "column_count": 2,
        "duplicate_rows": 0,
        "missing_cells": {"note": 0},
        "columns": [
            {"name": "note", "inferred_type": "text"},
            {"name": "summary", "inferred_type": "text"},
        ],
    },
]

PLAN_VARIANTS: list[dict] = [
    {
        "operators": [
            "load_csv", "profile_schema", "drop_duplicate_rows", "fill_missing_values",
            "normalize_column_types", "export_clean_dataset", "validate_clean_dataset",
        ],
        "rationale": ["Profile input.", "Remove duplicates.", "Impute missing.", "Normalize types."],
        "task_type": "cleaning",
        "data_type": "structured_csv",
        "intent_keywords": ["deduplicate", "fill_missing", "normalize_types", "export"],
        "confidence": 0.92,
    },
    {
        "operators": ["load_csv", "profile_schema", "drop_duplicate_rows", "export_clean_dataset", "validate_clean_dataset"],
        "rationale": ["Profile input.", "User requested dedup only."],
        "task_type": "cleaning",
        "data_type": "structured_csv",
        "intent_keywords": ["deduplicate", "export"],
        "confidence": 0.88,
    },
    {
        "operators": ["load_csv", "profile_schema", "fill_missing_values", "export_clean_dataset", "validate_clean_dataset"],
        "rationale": ["Profile input.", "Impute missing values only."],
        "task_type": "cleaning",
        "data_type": "structured_csv",
        "intent_keywords": ["fill_missing", "export"],
        "confidence": 0.86,
    },
    {
        "operators": ["load_csv", "profile_schema", "drop_column", "fill_missing_values", "export_clean_dataset", "validate_clean_dataset"],
        "rationale": ["Profile input.", "Drop columns with >50% missing.", "Impute remaining gaps."],
        "task_type": "cleaning",
        "data_type": "structured_csv",
        "intent_keywords": ["fill_missing", "export"],
        "confidence": 0.84,
    },
    {
        "operators": ["load_text", "clean_text", "extract_entities", "export_clean_dataset", "validate_clean_dataset"],
        "rationale": ["Load text.", "Clean text.", "Extract entities from unstructured content."],
        "task_type": "cleaning",
        "data_type": "text",
        "intent_keywords": ["extract", "export"],
        "confidence": 0.9,
    },
    {
        "operators": ["load_csv", "profile_schema", "normalize_column_types", "transform_columns", "export_clean_dataset", "validate_clean_dataset"],
        "rationale": ["Profile input.", "Normalize types.", "Apply column transforms."],
        "task_type": "cleaning",
        "data_type": "structured_csv",
        "intent_keywords": ["normalize_types", "transform", "export"],
        "confidence": 0.87,
    },
    {
        "operators": ["load_csv", "profile_schema", "export_clean_dataset", "validate_clean_dataset"],
        "rationale": ["Empty or unusable input; emit minimal profiling pipeline."],
        "task_type": "profiling",
        "data_type": "structured_csv",
        "intent_keywords": ["export"],
        "confidence": 0.55,
    },
    {
        "operators": ["load_csv", "profile_schema", "drop_duplicate_rows", "normalize_column_types", "export_clean_dataset", "validate_clean_dataset"],
        "rationale": ["Large file: dedup and cast types only."],
        "task_type": "cleaning",
        "data_type": "structured_csv",
        "intent_keywords": ["deduplicate", "normalize_types", "export"],
        "confidence": 0.83,
    },
    {
        "operators": ["load_text", "clean_text", "export_clean_dataset", "validate_clean_dataset"],
        "rationale": ["Text-only clean without entity extraction."],
        "task_type": "profiling",
        "data_type": "text",
        "intent_keywords": ["export"],
        "confidence": 0.78,
    },
    {
        "operators": ["load_csv", "profile_schema", "fill_missing_values", "transform_columns", "export_clean_dataset", "validate_clean_dataset"],
        "rationale": ["Partial clean requested.", "Transform after imputation."],
        "task_type": "cleaning",
        "data_type": "structured_csv",
        "intent_keywords": ["fill_missing", "transform", "export"],
        "confidence": 0.85,
    },
]


def _sample(rng: random.Random) -> dict:
    instruction = rng.choice(INSTRUCTIONS)
    profile = rng.choice(PROFILES)
    plan = rng.choice(PLAN_VARIANTS)
    return {
        "instruction": instruction,
        "input": json.dumps({"data_profile": profile}, ensure_ascii=False),
        "output": json.dumps(plan, ensure_ascii=False),
    }


def main() -> int:
    rng = random.Random(42)
    samples = [_sample(rng) for _ in range(500)]
    val_samples = [_sample(rng) for _ in range(50)]

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_PATH.write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in samples) + "\n",
        encoding="utf-8",
    )
    VAL_PATH.write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in val_samples) + "\n",
        encoding="utf-8",
    )
    combos = {tuple(json.loads(s["output"])["operators"]) for s in samples}
    print(f"Wrote {len(samples)} train / {len(val_samples)} val samples")
    print(f"Unique operator combinations: {len(combos)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
