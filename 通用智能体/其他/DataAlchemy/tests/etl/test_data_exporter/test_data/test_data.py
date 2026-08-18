# -*- coding: utf-8 -*-
"""
DataExporter 测试数据 fixtures

提供测试用的样本数据、预期结果和辅助函数。
可被 test_data_exporter.py 导入使用。
"""

import os
import tempfile
import pandas as pd
import numpy as np


# ============================================================================
# 样本数据（标准 input_data 格式）
# ============================================================================

SAMPLE_DATA = {
    # 基础数据（3行3列）
    "basic": [
        {"name": "张三", "age": 25, "dept": "内科"},
        {"name": "李四", "age": 30, "dept": "外科"},
        {"name": "王五", "age": 35, "dept": "内科"},
    ],

    # 包含中文特殊字符
    "with_special": [
        {"name": "张\"三", "note": "包含,逗号", "desc": "有\n换行"},
        {"name": "李'四", "note": "含=等号", "desc": "正常描述"},
    ],

    # 包含 NaN 和 None
    "with_null": [
        {"name": "张三", "age": 25, "dept": "内科"},
        {"name": None, "age": None, "dept": "外科"},
        {"name": "王五", "age": 35, "dept": None},
    ],

    # 包含 numpy 类型（Int64 兼容性测试）
    "with_numpy_types": [
        {"name": "张三", "age": np.int64(25), "score": np.float64(85.5)},
        {"name": "李四", "age": np.int64(30), "score": np.float64(90.0)},
    ],

    # 单行数据（边界测试）
    "single_row": [
        {"name": "张三", "age": 25},
    ],

    # 大量行（性能测试用）
    "many_rows": [{"id": i, "value": f"data_{i}"} for i in range(1000)],

    # 空数据
    "empty": [],
}

# 标准输入格式（模拟上游算子输出）
BASIC_INPUT = {
    "data": SAMPLE_DATA["basic"],
    "schema": {
        "columns": ["name", "age", "dept"],
        "row_count": 3
    },
    "report": {
        "input_rows": 3,
        "output_rows": 3,
        "warnings": [],
        "summary": "上游处理完成。"
    }
}


# ============================================================================
# 默认参数
# ============================================================================

CSV_PARAMS = {
    "outputFormat": "csv",
    "outputDir": "",
    "outputFileName": "test_export",
    "encoding": "utf-8",
    "includeHeader": True,
    "indexColumn": False,
    "overwrite": True,
}

JSON_PARAMS = {
    "outputFormat": "json",
    "outputDir": "",
    "outputFileName": "test_export",
    "encoding": "utf-8",
    "overwrite": True,
}

JSONL_PARAMS = {
    "outputFormat": "jsonl",
    "outputDir": "",
    "outputFileName": "test_export",
    "encoding": "utf-8",
    "overwrite": True,
}


# ============================================================================
# 辅助函数
# ============================================================================

def get_temp_dir():
    """获取临时测试目录"""
    return tempfile.mkdtemp(prefix="data_exporter_test_")


def make_params(overrides: dict) -> dict:
    """生成参数组合（用 overrides 覆盖默认值）"""
    base = CSV_PARAMS.copy()
    base.update(overrides)
    return base


def get_upstream_input(data_key: str) -> dict:
    """生成标准上游输入格式"""
    data = SAMPLE_DATA.get(data_key, SAMPLE_DATA["basic"])
    return {
        "data": data,
        "schema": {
            "columns": list(data[0].keys()) if data else [],
            "row_count": len(data)
        }
    }
