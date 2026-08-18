# -*- coding: utf-8 -*-
"""
DataCleaner 测试数据 fixtures

提供各类测试场景的标准化测试数据
"""

import pandas as pd

# ============================================================================
# 基础测试数据
# ============================================================================

SAMPLE_DATA = {
    # 基本数据
    "basic": [
        {"name": "张三", "phone": "13812345678", "age": 25},
        {"name": "李四", "phone": "", "age": None}
    ],

    # 去重测试数据
    "duplicates": [
        {"name": "张三", "phone": "13812345678"},
        {"name": "张三", "phone": "13812345678"},  # 完全重复
        {"name": "李四", "phone": "13900000000"}
    ],

    # 隐私数据
    "privacy": [
        {"name": "张三", "phone": "13812345678", "id_card": "310101199001011234"},
        {"name": "李四", "phone": "13898765432", "id_card": "110101199001011234"}
    ],

    # 空格数据
    "whitespace": [
        {"name": " 张三 ", "phone": " 13812345678"},
        {"name": "李四 ", "phone": " 13900000000 "}
    ],

    # 空数据
    "empty": [],

    # 混合场景
    "mixed": [
        {"name": " 张三 ", "phone": "13812345678", "age": 25},
        {"name": "张三", "phone": "13812345678", "age": 25},  # 重复（空格处理后）
        {"name": "李四", "phone": "", "age": None},  # 空值
        {"name": "王五", "phone": "13811111111", "id_card": "310101199001011234"}
    ],

    # 日期数据
    "dates": [
        {"name": "张三", "birthday": "2026/04/28"},
        {"name": "李四", "birthday": "20260428"},
        {"name": "王五", "birthday": "04/28/2026"},
        {"name": "赵六", "birthday": "2026-04-28"}
    ],

    # 复杂姓名（含称呼）
    "complex_names": [
        {"name": "张三先生"},
        {"name": "李四女士"},
        {"name": "王老师"}
    ],

    # 无效数据
    "invalid_phone": [
        {"phone": "12345678901"},  # 12位，非法
        {"phone": "1234567890"},   # 10位，非法
    ]
}

# ============================================================================
# 期望结果定义
# ============================================================================

EXPECTED_RESULTS = {
    "remove_duplicates": {
        "count": 1,  # 删除1条重复
    },
    "fill_missing": {
        "value": "未知"
    },
    "masked": {
        "phone": "138****5678",
        "id_card": "310***********1234",
        "name": "患者1"
    },
    "whitespace": {
        "trimmed_name": "张三",
        "trimmed_phone": "13812345678"
    }
}

# ============================================================================
# DataFrame 格式测试数据
# ============================================================================

def get_dataframe(data_key: str) -> pd.DataFrame:
    """获取 DataFrame 格式的测试数据"""
    return pd.DataFrame(SAMPLE_DATA.get(data_key, []))

# ============================================================================
# 大规模测试数据
# ============================================================================

def generate_large_dataset(row_count: int = 10000) -> list:
    """生成大规模测试数据"""
    import random

    names = ["张三", "李四", "王五", "赵六", "钱七", "孙八", "周九", "吴十"]
    base_data = []

    for i in range(row_count):
        # 10% 重复
        if i > 0 and random.random() < 0.1:
            row = base_data[i - 1].copy()
        else:
            row = {
                "name": random.choice(names),
                "phone": f"138{random.randint(10000000, 99999999)}",
                "age": random.randint(18, 80) if random.random() > 0.1 else None
            }
        base_data.append(row)

    return base_data

# ============================================================================
# 辅助函数
# ============================================================================

def create_input_schema(data: list) -> dict:
    """创建标准的输入 schema"""
    if not data:
        return {"columns": [], "row_count": 0}

    return {
        "columns": list(data[0].keys()),
        "row_count": len(data)
    }

def create_input_data(data_key: str) -> dict:
    """创建完整的输入数据结构"""
    data = SAMPLE_DATA.get(data_key, [])
    return {
        "data": data,
        "schema": create_input_schema(data)
    }
