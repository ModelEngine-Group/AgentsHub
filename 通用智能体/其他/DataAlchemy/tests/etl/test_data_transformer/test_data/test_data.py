# -*- coding: utf-8 -*-
"""
DataTransformer 测试数据 fixtures

提供测试用的样本数据、预期结果和辅助函数。
可被 test_data_transformer.py 和 test_e2e.py 共同导入使用。
"""

import pandas as pd
import numpy as np


# ============================================================================
# 样本数据
# ============================================================================

SAMPLE_DATA = {
    # 基础数据（3行3列）
    "basic": [
        {"name": "张三", "age": "25", "dept": "内科"},
        {"name": "李四", "age": "30", "dept": "外科"},
        {"name": "王五", "age": "35", "dept": "内科"},
    ],

    # 包含多种类型的数据
    "with_types": [
        {"name": "张三", "age": "25", "price": "3.14", "admit_date": "2023/5/3", "gender": "M"},
        {"name": "李四", "age": "30", "price": "6.28", "admit_date": "2023-06-15", "gender": "F"},
    ],

    # 列名含空格
    "with_spaces": [
        {"patient name": "张三", "age": "25"},
        {"patient name": "李四", "age": "30"},
    ],

    # 包含零值（用于除零测试）
    "with_zero": [
        {"weight": 70, "height": 175},
        {"weight": 60, "height": 0},
    ],

    # 类型混合列
    "type_mixed": [
        {"age": "25"},
        {"age": 30},
        {"age": "thirty"},
    ],

    # 含 NaN 值
    "with_null": [
        {"name": "张三", "age": "25", "dept": "内科"},
        {"name": None, "age": None, "dept": "外科"},
        {"name": "王五", "age": "35", "dept": None},
    ],

    # 医疗数据（完整 ETL 场景）
    "medical": [
        {"name": "张三", "age": "25", "admit_date": "2023/5/3", "dept": "内科", "gender": "M", "weight": 70, "height": 175},
        {"name": "李四", "age": "30", "admit_date": "2023-06-15", "dept": "外科", "gender": "F", "weight": 55, "height": 160},
        {"name": "王五", "age": "35", "admit_date": "20230701", "dept": "内科", "gender": "M", "weight": 80, "height": 180},
    ],

    # 空数据
    "empty": [],
}


# ============================================================================
# 全空参数（不执行任何操作）
# ============================================================================

EMPTY_PARAMS = {
    "renameFields": "",
    "selectFields": "",
    "dropFields": "",
    "typeConversions": "",
    "valueMappings": "",
    "filterCondition": "",
    "deriveColumns": ""
}


# ============================================================================
# 辅助函数
# ============================================================================

def get_dataframe(data_key: str) -> pd.DataFrame:
    """
    根据数据键名获取 DataFrame

    Args:
        data_key: SAMPLE_DATA 中的键名

    Returns:
        对应的 DataFrame
    """
    return pd.DataFrame(SAMPLE_DATA[data_key])


def create_input_data(data_key: str) -> dict:
    """
    根据数据键名创建标准输入格式

    Args:
        data_key: SAMPLE_DATA 中的键名

    Returns:
        {"data": [...], "schema": {...}} 格式的字典
    """
    data = SAMPLE_DATA[data_key]
    df = pd.DataFrame(data)
    return {
        "data": data,
        "schema": {
            "columns": list(df.columns),
            "row_count": len(df)
        }
    }


def create_large_data(rows: int = 10000) -> dict:
    """
    创建大数据集（性能测试用）

    Args:
        rows: 数据行数

    Returns:
        大数据集的标准输入格式
    """
    data = [
        {
            "name": f"user_{i}",
            "age": str(20 + i % 50),
            "dept": "内科" if i % 2 == 0 else "外科"
        }
        for i in range(rows)
    ]
    return {
        "data": data,
        "schema": {"columns": ["name", "age", "dept"], "row_count": rows}
    }


# ============================================================================
# 预期结果
# ============================================================================

EXPECTED_RESULTS = {
    # 基础数据的列名
    "basic_columns": ["name", "age", "dept"],

    # 医疗数据的列名
    "medical_columns": ["name", "age", "admit_date", "dept", "gender", "weight", "height"],

    # 日期标准化预期
    "date_standardization": {
        "2023/5/3": "2023-05-03",
        "2023-06-15": "2023-06-15",
        "20230701": "2023-07-01",
    }
}
