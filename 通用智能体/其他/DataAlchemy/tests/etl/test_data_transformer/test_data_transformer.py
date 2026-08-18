# -*- coding: utf-8 -*-
"""
DataTransformer 算子完整测试套件

测试金字塔：
- 单元测试 (Unit Tests): 30个
- 集成测试 (Integration Tests): 5个
- 边界测试 (Boundary Tests): 6个
- 异常测试 (Exception Tests): 5个
- 端到端测试 (E2E Tests): 3个

运行方式：
    pytest test_data_transformer.py -v
    pytest test_data_transformer.py -v -k "unit"
    pytest test_data_transformer.py -v -k "integration"
    pytest test_data_transformer.py -v -k "boundary"
    pytest test_data_transformer.py -v -k "exception"
    pytest test_data_transformer.py -v -k "e2e"
"""

import sys
import json
import pytest
import pandas as pd
import numpy as np
import time

# 路径配置
sys.path.insert(0, r"D:\PythonProject\ModelEngine\operators")

from data_transformer import DataTransformer, ValidationError, ProcessingError


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def transformer():
    """DataTransformer 算子实例"""
    return DataTransformer()


@pytest.fixture
def basic_data():
    """基础测试数据（3行3列）"""
    return {
        "data": [
            {"name": "张三", "age": "25", "dept": "内科"},
            {"name": "李四", "age": "30", "dept": "外科"},
            {"name": "王五", "age": "35", "dept": "内科"},
        ],
        "schema": {"columns": ["name", "age", "dept"], "row_count": 3}
    }


@pytest.fixture
def basic_params():
    """全空参数（不执行任何操作）"""
    return {
        "renameFields": "",
        "selectFields": "",
        "dropFields": "",
        "typeConversions": "",
        "valueMappings": "",
        "filterCondition": "",
        "deriveColumns": ""
    }


@pytest.fixture
def data_with_types():
    """包含不同类型的数据"""
    return {
        "data": [
            {"name": "张三", "age": "25", "price": "3.14", "admit_date": "2023/5/3", "gender": "M"},
            {"name": "李四", "age": "30", "price": "6.28", "admit_date": "2023-06-15", "gender": "F"},
        ],
        "schema": {"columns": ["name", "age", "price", "admit_date", "gender"], "row_count": 2}
    }


@pytest.fixture
def data_with_spaces():
    """列名含空格的数据"""
    return {
        "data": [
            {"patient name": "张三", "age": "25"},
            {"patient name": "李四", "age": "30"},
        ],
        "schema": {"columns": ["patient name", "age"], "row_count": 2}
    }


@pytest.fixture
def data_with_zero():
    """包含零值的数据（用于除零测试）"""
    return {
        "data": [
            {"weight": 70, "height": 175},
            {"weight": 60, "height": 0},
        ],
        "schema": {"columns": ["weight", "height"], "row_count": 2}
    }


@pytest.fixture
def large_data():
    """大数据集（性能测试用）"""
    rows = [{"name": f"user_{i}", "age": str(20 + i % 50), "dept": "内科" if i % 2 == 0 else "外科"}
            for i in range(10000)]
    return {"data": rows, "schema": {"columns": ["name", "age", "dept"], "row_count": 10000}}


def _run(transformer, data, params):
    """辅助方法：执行 process 并返回结果"""
    return transformer.process(data, params)


def _get_warnings(result, warning_type=None):
    """辅助方法：获取指定类型的 warnings"""
    warnings = result["report"]["warnings"]
    if warning_type:
        return [w for w in warnings if w["type"] == warning_type]
    return warnings


# ============================================================================
# 单元测试 - select_fields
# ============================================================================

class TestSelectFields:
    """字段选择测试"""

    def test_ut01_normal_select(self, transformer, basic_data, basic_params):
        """UT-01: 正常选择，保留 name 和 age"""
        params = {**basic_params, "selectFields": "name,age"}
        result = _run(transformer, basic_data, params)

        assert list(result["schema"]["columns"]) == ["name", "age"]
        assert result["report"]["transform_summary"]["select_count"] == 2
        assert result["report"]["input_rows"] == 3
        assert result["report"]["output_rows"] == 3

    def test_ut02_field_not_found(self, transformer, basic_data, basic_params):
        """UT-02: 选择不存在的字段，保留存在的字段 + warning"""
        params = {**basic_params, "selectFields": "name,xxx"}
        result = _run(transformer, basic_data, params)

        assert list(result["schema"]["columns"]) == ["name"]
        assert result["report"]["transform_summary"]["select_count"] == 1
        assert len(_get_warnings(result, "field_not_found")) >= 1

    def test_ut03_empty_select(self, transformer, basic_data, basic_params):
        """UT-03: 空参数，保留全部字段"""
        params = {**basic_params, "selectFields": ""}
        result = _run(transformer, basic_data, params)

        assert list(result["schema"]["columns"]) == ["name", "age", "dept"]
        assert result["report"]["transform_summary"]["select_count"] == 0

    def test_ut04_all_fields(self, transformer, basic_data, basic_params):
        """UT-04: 选择全部字段"""
        params = {**basic_params, "selectFields": "name,age,dept"}
        result = _run(transformer, basic_data, params)

        assert list(result["schema"]["columns"]) == ["name", "age", "dept"]
        assert result["report"]["transform_summary"]["select_count"] == 3


# ============================================================================
# 单元测试 - drop_fields
# ============================================================================

class TestDropFields:
    """字段删除测试"""

    def test_ut05_normal_drop(self, transformer, basic_data, basic_params):
        """UT-05: 正常删除 dept 字段"""
        params = {**basic_params, "dropFields": "dept"}
        result = _run(transformer, basic_data, params)

        assert "dept" not in result["schema"]["columns"]
        assert result["report"]["transform_summary"]["drop_count"] == 1

    def test_ut06_field_not_found(self, transformer, basic_data, basic_params):
        """UT-06: 删除不存在的字段，无变化 + warning"""
        params = {**basic_params, "dropFields": "xxx"}
        result = _run(transformer, basic_data, params)

        assert list(result["schema"]["columns"]) == ["name", "age", "dept"]
        assert result["report"]["transform_summary"]["drop_count"] == 0
        assert len(_get_warnings(result, "field_not_found")) >= 1

    def test_ut07_empty_drop(self, transformer, basic_data, basic_params):
        """UT-07: 空参数，无变化"""
        params = {**basic_params, "dropFields": ""}
        result = _run(transformer, basic_data, params)

        assert list(result["schema"]["columns"]) == ["name", "age", "dept"]
        assert result["report"]["transform_summary"]["drop_count"] == 0


# ============================================================================
# 单元测试 - rename_fields
# ============================================================================

class TestRenameFields:
    """字段重命名测试"""

    def test_ut08_normal_rename(self, transformer, basic_data, basic_params):
        """UT-08: 正常重命名 name → patient_name"""
        params = {**basic_params, "renameFields": json.dumps({"name": "patient_name"})}
        result = _run(transformer, basic_data, params)

        assert "patient_name" in result["schema"]["columns"]
        assert "name" not in result["schema"]["columns"]
        assert result["report"]["transform_summary"]["rename_count"] == 1

    def test_ut09_new_name_conflict(self, transformer, basic_data, basic_params):
        """UT-09: 新名称已被其他字段占用，记录 warning"""
        params = {**basic_params, "renameFields": json.dumps({"name": "age"})}
        result = _run(transformer, basic_data, params)

        # name 被重命名为 age，原来的 age 列被覆盖
        assert result["report"]["transform_summary"]["rename_count"] == 1
        assert len(_get_warnings(result, "field_not_found")) >= 1

    def test_ut10_source_not_found(self, transformer, basic_data, basic_params):
        """UT-10: 源字段不存在，warning + 跳过"""
        params = {**basic_params, "renameFields": json.dumps({"xxx": "yyy"})}
        result = _run(transformer, basic_data, params)

        assert result["report"]["transform_summary"]["rename_count"] == 0
        assert len(_get_warnings(result, "field_not_found")) >= 1

    def test_ut11_multiple_to_same_new(self, transformer, basic_data, basic_params):
        """UT-11: 多个字段映射到同一个新名称，后者覆盖前者"""
        params = {**basic_params, "renameFields": json.dumps({"name": "new_col", "age": "new_col"})}
        result = _run(transformer, basic_data, params)

        assert "new_col" in result["schema"]["columns"]
        assert len(_get_warnings(result, "field_not_found")) >= 1


# ============================================================================
# 单元测试 - type_conversion
# ============================================================================

class TestTypeConversion:
    """类型转换测试"""

    def test_ut12_str_to_int(self, transformer, basic_data, basic_params):
        """UT-12: 字符串 → 整数"""
        params = {**basic_params, "typeConversions": json.dumps({"age": "int"})}
        result = _run(transformer, basic_data, params)

        assert result["data"][0]["age"] == 25
        assert result["report"]["transform_summary"]["type_convert_count"] == 1

    def test_ut13_str_to_float(self, transformer, data_with_types, basic_params):
        """UT-13: 字符串 → 浮点"""
        params = {**basic_params, "typeConversions": json.dumps({"price": "float"})}
        result = _run(transformer, data_with_types, params)

        assert abs(result["data"][0]["price"] - 3.14) < 0.01

    def test_ut14_int_to_str(self, transformer, basic_data, basic_params):
        """UT-14: 数字 → 字符串"""
        # 先转成 int
        params1 = {**basic_params, "typeConversions": json.dumps({"age": "int"})}
        result1 = _run(transformer, basic_data, params1)

        # 再转回 str
        params2 = {**basic_params, "typeConversions": json.dumps({"age": "str"})}
        result2 = _run(transformer, result1, params2)

        assert isinstance(result2["data"][0]["age"], str)

    def test_ut15_date_standardization(self, transformer, data_with_types, basic_params):
        """UT-15: 日期格式标准化 → YYYY-MM-DD"""
        params = {**basic_params, "typeConversions": json.dumps({"admit_date": "date"})}
        result = _run(transformer, data_with_types, params)

        assert result["data"][0]["admit_date"] == "2023-05-03"
        assert result["data"][1]["admit_date"] == "2023-06-15"

    def test_ut16_invalid_conversion(self, transformer, basic_params):
        """UT-16: 无效转换（abc → int），保留原值 + warning"""
        data = {"data": [{"val": "abc"}, {"val": "def"}], "schema": {"columns": ["val"], "row_count": 2}}
        params = {**basic_params, "typeConversions": json.dumps({"val": "int"})}
        result = _run(transformer, data, params)

        # "abc" 会被 coerce 成 NaN
        assert result["report"]["transform_summary"]["type_convert_count"] == 1

    def test_ut17_field_not_found(self, transformer, basic_data, basic_params):
        """UT-17: 不存在的字段，warning + 跳过"""
        params = {**basic_params, "typeConversions": json.dumps({"xxx": "int"})}
        result = _run(transformer, basic_data, params)

        assert result["report"]["transform_summary"]["type_convert_count"] == 0
        assert len(_get_warnings(result, "field_not_found")) >= 1


# ============================================================================
# 单元测试 - value_mapping
# ============================================================================

class TestValueMapping:
    """值替换测试"""

    def test_ut18_normal_mapping(self, transformer, data_with_types, basic_params):
        """UT-18: 正常替换 M → 男"""
        params = {**basic_params, "valueMappings": json.dumps({"gender": {"M": "男"}})}
        result = _run(transformer, data_with_types, params)

        assert result["data"][0]["gender"] == "男"
        assert result["data"][1]["gender"] == "F"  # F 不变
        assert result["report"]["transform_summary"]["value_replace_count"] == 1

    def test_ut19_value_not_found(self, transformer, data_with_types, basic_params):
        """UT-19: 值不存在，count=0，不记录 warning"""
        params = {**basic_params, "valueMappings": json.dumps({"gender": {"X": "未知"}})}
        result = _run(transformer, data_with_types, params)

        assert result["report"]["transform_summary"]["value_replace_count"] == 0
        # 不应该有 warning（值不存在是正常业务场景）
        assert len(_get_warnings(result, "value_mapping_type_mismatch")) == 0

    def test_ut20_batch_mapping(self, transformer, basic_data, basic_params):
        """UT-20: 批量值替换"""
        params = {**basic_params, "valueMappings": json.dumps({"dept": {"内科": "内", "外科": "外"}})}
        result = _run(transformer, basic_data, params)

        assert result["data"][0]["dept"] == "内"
        assert result["data"][1]["dept"] == "外"
        assert result["report"]["transform_summary"]["value_replace_count"] == 3


# ============================================================================
# 单元测试 - filter_condition
# ============================================================================

class TestFilterCondition:
    """条件筛选测试"""

    def test_ut21_numeric_filter(self, transformer, basic_data, basic_params):
        """UT-21: 数值筛选 age > 25"""
        # 先转换类型
        params = {
            **basic_params,
            "typeConversions": json.dumps({"age": "int"}),
            "filterCondition": "age > 25"
        }
        result = _run(transformer, basic_data, params)

        assert result["report"]["output_rows"] == 2  # 30 和 35
        assert result["report"]["transform_summary"]["filter_count"] == 1

    def test_ut22_string_filter(self, transformer, basic_data, basic_params):
        """UT-22: 字符串筛选 dept == 内科"""
        params = {**basic_params, "filterCondition": 'dept == "内科"'}
        result = _run(transformer, basic_data, params)

        assert result["report"]["output_rows"] == 2  # 张三和王五
        assert result["report"]["transform_summary"]["filter_count"] == 1

    def test_ut23_column_with_spaces(self, transformer, data_with_spaces, basic_params):
        """UT-23: 列名有空格时自动标准化"""
        params = {**basic_params, "filterCondition": 'patient_name == "张三"'}
        result = _run(transformer, data_with_spaces, params)

        # 列名应该恢复为原始的 "patient name"
        assert "patient name" in result["schema"]["columns"]
        assert result["report"]["output_rows"] == 1

    def test_ut24_expression_error(self, transformer, basic_data, basic_params):
        """UT-24: 表达式语法错误 → ValidationError"""
        params = {**basic_params, "filterCondition": "age >>> 18"}
        with pytest.raises(ValidationError):
            _run(transformer, basic_data, params)

    def test_ut25_empty_condition(self, transformer, basic_data, basic_params):
        """UT-25: 空条件，无变化"""
        params = {**basic_params, "filterCondition": ""}
        result = _run(transformer, basic_data, params)

        assert result["report"]["output_rows"] == 3
        assert result["report"]["transform_summary"]["filter_count"] == 0


# ============================================================================
# 单元测试 - derive_columns
# ============================================================================

class TestDeriveColumns:
    """派生列测试"""

    def test_ut26_normal_derive(self, transformer, data_with_zero, basic_params):
        """UT-26: 正常计算 BMI"""
        params = {**basic_params, "deriveColumns": json.dumps({"bmi": "weight / (height/100)**2"})}
        result = _run(transformer, data_with_zero, params)

        assert "bmi" in result["schema"]["columns"]
        assert result["report"]["transform_summary"]["derive_count"] == 1
        # 第一行：70 / (1.75)^2 ≈ 22.86
        assert abs(result["data"][0]["bmi"] - 22.86) < 0.1

    def test_ut27_divide_by_zero(self, transformer, data_with_zero, basic_params):
        """UT-27: 除零处理 → NaN（不是 inf）"""
        params = {**basic_params, "deriveColumns": json.dumps({"rate": "weight / height"})}
        result = _run(transformer, data_with_zero, params)

        assert "rate" in result["schema"]["columns"]
        # 第一行正常
        assert result["data"][0]["rate"] == 70 / 175
        # 第二行 height=0 → NaN（不是 inf）
        assert pd.isna(result["data"][1]["rate"])

    def test_ut28_column_name_conflict(self, transformer, basic_data, basic_params):
        """UT-28: 派生列名已存在 → warning + 覆盖"""
        params = {**basic_params, "deriveColumns": json.dumps({"name": "100"})}
        result = _run(transformer, basic_data, params)

        assert len(_get_warnings(result, "field_not_found")) >= 1
        assert result["report"]["transform_summary"]["derive_count"] == 1

    def test_ut29_expression_error(self, transformer, basic_data, basic_params):
        """UT-29: 表达式错误 → NaN + warning"""
        params = {**basic_params, "deriveColumns": json.dumps({"new_col": "nonexistent + 1"})}
        result = _run(transformer, basic_data, params)

        assert len(_get_warnings(result, "derive_error")) >= 1
        assert result["report"]["transform_summary"]["derive_count"] == 0

    def test_ut30_empty_derive(self, transformer, basic_data, basic_params):
        """UT-30: 空参数，无变化"""
        params = {**basic_params, "deriveColumns": ""}
        result = _run(transformer, basic_data, params)

        assert result["report"]["transform_summary"]["derive_count"] == 0


# ============================================================================
# 集成测试 - Pipeline 顺序
# ============================================================================

class TestIntegration:
    """集成测试 - Pipeline 顺序"""

    def test_it01_select_then_drop(self, transformer, basic_data, basic_params):
        """IT-01: select → drop，先选后删"""
        params = {
            **basic_params,
            "selectFields": "name,age,dept",
            "dropFields": "dept"
        }
        result = _run(transformer, basic_data, params)

        assert list(result["schema"]["columns"]) == ["name", "age"]
        # 应该有 select_drop_conflict warning
        assert len(_get_warnings(result, "select_drop_conflict")) >= 1

    def test_it02_select_then_rename(self, transformer, basic_data, basic_params):
        """IT-02: select → rename，先选再重命名"""
        params = {
            **basic_params,
            "selectFields": "name,age",
            "renameFields": json.dumps({"name": "patient_name"})
        }
        result = _run(transformer, basic_data, params)

        assert list(result["schema"]["columns"]) == ["patient_name", "age"]
        assert result["data"][0]["patient_name"] == "张三"

    def test_it03_rename_then_type(self, transformer, data_with_types, basic_params):
        """IT-03: rename → type_conversion，重命名后用新名做类型转换"""
        params = {
            **basic_params,
            "renameFields": json.dumps({"age": "patient_age"}),
            "typeConversions": json.dumps({"patient_age": "int"})
        }
        result = _run(transformer, data_with_types, params)

        assert "patient_age" in result["schema"]["columns"]
        assert result["data"][0]["patient_age"] == 25

    def test_it04_type_then_value(self, transformer, basic_data, basic_params):
        """IT-04: type_conversion → value_mapping，类型转换后值替换生效"""
        params = {
            **basic_params,
            "typeConversions": json.dumps({"age": "int"}),
            "valueMappings": json.dumps({"age": {"25": "青年", "30": "壮年", "35": "中年"}})
        }
        result = _run(transformer, basic_data, params)

        # value_mapping 将 int 列替换为字符串时，列会转为 object
        assert result["data"][0]["age"] == "青年"
        assert result["data"][1]["age"] == "壮年"

    def test_it05_rename_then_derive(self, transformer, basic_data, basic_params):
        """IT-05: rename + type + derive，重命名后用新名做类型转换再派生"""
        params = {
            **basic_params,
            "renameFields": json.dumps({"age": "patient_age"}),
            "typeConversions": json.dumps({"patient_age": "int"}),
            "deriveColumns": json.dumps({"age_group": "patient_age // 10 * 10"})
        }
        result = _run(transformer, basic_data, params)

        assert "patient_age" in result["schema"]["columns"]
        assert "age_group" in result["schema"]["columns"]
        assert result["data"][0]["age_group"] == 20  # 25 // 10 * 10 = 20

    def test_it06_type_then_filter(self, transformer, basic_data, basic_params):
        """IT-06: type_conversion → filter_condition，类型转换后筛选生效"""
        params = {
            **basic_params,
            "typeConversions": json.dumps({"age": "int"}),
            "filterCondition": "age > 25"
        }
        result = _run(transformer, basic_data, params)

        assert result["report"]["output_rows"] == 2
        assert result["report"]["transform_summary"]["type_convert_count"] == 1

    def test_it07_full_pipeline(self, transformer, data_with_types, basic_params):
        """IT-07: 全流程 7 步全部执行"""
        params = {
            "selectFields": "name,age,admit_date,gender",
            "dropFields": "",
            "renameFields": json.dumps({"name": "patient_name"}),
            "typeConversions": json.dumps({"age": "int", "admit_date": "date"}),
            "valueMappings": json.dumps({"gender": {"M": "男", "F": "女"}}),
            "filterCondition": "age >= 25",
            "deriveColumns": json.dumps({"age_group": 'age // 10 * 10'})
        }
        result = _run(transformer, data_with_types, params)

        report = result["report"]
        assert report["output_rows"] == 2
        assert report["transform_summary"]["rename_count"] == 1
        assert report["transform_summary"]["type_convert_count"] == 2
        assert report["transform_summary"]["value_replace_count"] == 2
        assert report["transform_summary"]["derive_count"] == 1
        assert "patient_name" in result["schema"]["columns"]
        assert "age_group" in result["schema"]["columns"]
        # performance 字段存在
        assert "performance" in report
        assert "total_ms" in report["performance"]

    def test_it08_rename_then_filter(self, transformer, basic_data, basic_params):
        """IT-08: rename 后 filter 使用新列名"""
        params = {
            **basic_params,
            "renameFields": json.dumps({"age": "patient_age"}),
            "typeConversions": json.dumps({"patient_age": "int"}),
            "filterCondition": "patient_age > 25"
        }
        result = _run(transformer, basic_data, params)

        assert result["report"]["output_rows"] == 2
        assert "patient_age" in result["schema"]["columns"]


# ============================================================================
# 边界测试
# ============================================================================

class TestBoundary:
    """边界测试"""

    def test_bt01_empty_data(self, transformer, basic_params):
        """BT-01: 空数据，返回空结果"""
        data = {"data": [], "schema": {"columns": [], "row_count": 0}}
        result = _run(transformer, data, basic_params)

        assert result["report"]["input_rows"] == 0
        assert result["report"]["output_rows"] == 0
        assert result["data"] == []

    def test_bt02_single_row(self, transformer, basic_params):
        """BT-02: 单行数据，正常处理"""
        data = {"data": [{"name": "张三", "age": "25"}], "schema": {"columns": ["name", "age"], "row_count": 1}}
        params = {**basic_params, "typeConversions": json.dumps({"age": "int"})}
        result = _run(transformer, data, params)

        assert result["report"]["output_rows"] == 1
        assert result["data"][0]["age"] == 25

    def test_bt03_all_params_empty(self, transformer, basic_data, basic_params):
        """BT-03: 全参数为空，原样返回"""
        result = _run(transformer, basic_data, basic_params)

        assert result["report"]["input_rows"] == 3
        assert result["report"]["output_rows"] == 3
        assert list(result["schema"]["columns"]) == ["name", "age", "dept"]
        assert result["report"]["transform_summary"]["rename_count"] == 0
        assert result["report"]["transform_summary"]["type_convert_count"] == 0

    def test_bt04_filter_to_empty(self, transformer, basic_data, basic_params):
        """BT-04: 筛选后为空"""
        params = {**basic_params, "typeConversions": json.dumps({"age": "int"}), "filterCondition": "age > 100"}
        result = _run(transformer, basic_data, params)

        assert result["report"]["output_rows"] == 0
        assert result["data"] == []

    def test_bt05_select_all_fields(self, transformer, basic_data, basic_params):
        """BT-05: 选择全部字段，等于原字段"""
        params = {**basic_params, "selectFields": "name,age,dept"}
        result = _run(transformer, basic_data, params)

        assert list(result["schema"]["columns"]) == ["name", "age", "dept"]
        assert result["report"]["transform_summary"]["select_count"] == 3

    def test_bt06_type_mixed_column(self, transformer, basic_params):
        """BT-06: 字段类型混合"""
        data = {"data": [{"age": "25"}, {"age": 30}, {"age": "thirty"}], "schema": {"columns": ["age"], "row_count": 3}}
        params = {**basic_params, "typeConversions": json.dumps({"age": "int"})}
        result = _run(transformer, data, params)

        assert result["report"]["output_rows"] == 3
        # "25" → 25, 30 → 30, "thirty" → NaN
        assert result["data"][0]["age"] == 25
        assert result["data"][1]["age"] == 30
        assert pd.isna(result["data"][2]["age"])


# ============================================================================
# 异常测试
# ============================================================================

class TestException:
    """异常测试"""

    def test_et01_json_format_error(self, transformer, basic_data, basic_params):
        """ET-01: JSON 格式错误 → ValidationError"""
        params = {**basic_params, "typeConversions": '{"a": }'}
        with pytest.raises(ValidationError):
            _run(transformer, basic_data, params)

    def test_et02_unsupported_type(self, transformer, basic_data, basic_params):
        """ET-02: 不支持的目标类型 → ValidationError"""
        params = {**basic_params, "typeConversions": json.dumps({"age": "boolean"})}
        with pytest.raises(ValidationError):
            _run(transformer, basic_data, params)

    def test_et03_filter_syntax_error(self, transformer, basic_data, basic_params):
        """ET-03: filter 表达式语法错误 → ValidationError"""
        params = {**basic_params, "filterCondition": "age >>> 18"}
        with pytest.raises(ValidationError):
            _run(transformer, basic_data, params)

    def test_et04_derive_divide_by_zero(self, transformer, data_with_zero, basic_params):
        """ET-04: derive 除零 → NaN + warning"""
        params = {**basic_params, "deriveColumns": json.dumps({"result": "1 / height"})}
        result = _run(transformer, data_with_zero, params)

        # 第二行 height=0 → inf → NaN
        assert pd.isna(result["data"][1]["result"])
        # 第一行正常
        assert result["data"][0]["result"] == 1 / 175

    def test_et05_invalid_input_format(self, transformer, basic_params):
        """ET-05: 输入格式无效 → 返回空结果"""
        result = _run(transformer, "invalid", basic_params)

        assert result["report"]["input_rows"] == 0
        assert result["data"] == []


# ============================================================================
# 端到端测试
# ============================================================================

class TestE2E:
    """端到端测试 - 完整场景"""

    def test_e2e01_medical_data_pipeline(self, transformer):
        """E2E-01: 医疗数据完整 ETL 转换"""
        data = {
            "data": [
                {"name": "张三", "age": "25", "admit_date": "2023/5/3", "dept": "内科", "gender": "M", "weight": 70, "height": 175},
                {"name": "李四", "age": "30", "admit_date": "2023-06-15", "dept": "外科", "gender": "F", "weight": 55, "height": 160},
                {"name": "王五", "age": "35", "admit_date": "20230701", "dept": "内科", "gender": "M", "weight": 80, "height": 180},
            ],
            "schema": {"columns": ["name", "age", "admit_date", "dept", "gender", "weight", "height"], "row_count": 3}
        }
        params = {
            "selectFields": "name,age,admit_date,gender,weight,height",
            "dropFields": "",
            "renameFields": json.dumps({"name": "patient_name"}),
            "typeConversions": json.dumps({"age": "int", "admit_date": "date"}),
            "valueMappings": json.dumps({"gender": {"M": "男", "F": "女"}}),
            "filterCondition": "",
            "deriveColumns": json.dumps({"bmi": "weight / (height/100)**2"})
        }
        result = _run(transformer, data, params)

        report = result["report"]
        assert report["input_rows"] == 3
        assert report["output_rows"] == 3
        assert "patient_name" in result["schema"]["columns"]
        assert "dept" not in result["schema"]["columns"]  # 被 select 排除
        assert "bmi" in result["schema"]["columns"]

        # 验证数据正确性
        assert result["data"][0]["patient_name"] == "张三"
        assert result["data"][0]["age"] == 25
        assert result["data"][0]["admit_date"] == "2023-05-03"
        assert result["data"][0]["gender"] == "男"
        assert abs(result["data"][0]["bmi"] - 22.86) < 0.1

        # 验证 report 完整性
        assert report["transform_summary"]["rename_count"] == 1
        assert report["transform_summary"]["select_count"] == 6
        assert report["transform_summary"]["type_convert_count"] == 2
        assert report["transform_summary"]["value_replace_count"] == 3  # 2×M→男 + 1×F→女
        assert report["transform_summary"]["derive_count"] == 1

        # 验证性能分析
        assert "performance" in report
        assert report["performance"]["total_ms"] > 0
        assert len(report["performance"]["steps"]) == 7

        # 验证 summary
        assert "数据转换完成" in report["summary"]

    def test_e2e02_performance_large_dataset(self, transformer, large_data, basic_params):
        """E2E-02: 大数据集性能测试"""
        params = {
            **basic_params,
            "typeConversions": json.dumps({"age": "int"}),
            "filterCondition": "age > 30"
        }
        start = time.perf_counter()
        result = _run(transformer, large_data, params)
        elapsed = (time.perf_counter() - start) * 1000

        assert result["report"]["input_rows"] == 10000
        assert result["report"]["output_rows"] > 0
        assert result["report"]["performance"]["total_ms"] > 0
        # 10000 行应在 5 秒内完成
        assert elapsed < 5000, f"10000 行处理耗时 {elapsed:.0f}ms，超过 5s 阈值"

    def test_e2e03_null_handling(self, transformer, basic_params):
        """E2E-03: NaN 值在各步骤中的处理"""
        data = {
            "data": [
                {"name": "张三", "age": "25", "dept": "内科"},
                {"name": None, "age": None, "dept": "外科"},
                {"name": "王五", "age": "35", "dept": None},
            ],
            "schema": {"columns": ["name", "age", "dept"], "row_count": 3}
        }
        params = {
            **basic_params,
            "typeConversions": json.dumps({"age": "int"}),
            "filterCondition": "age > 20"
        }
        result = _run(transformer, data, params)

        # NaN 的 age 行被 filter 排除
        assert result["report"]["output_rows"] == 2
        assert result["data"][0]["name"] == "张三"


# ============================================================================
# 性能测试
# ============================================================================

class TestPerformance:
    """性能测试"""

    def test_pt01_large_dataset_performance(self, transformer, large_data, basic_params):
        """PT-01: 10000 行数据集性能"""
        params = {
            **basic_params,
            "renameFields": json.dumps({"name": "user_name"}),
            "typeConversions": json.dumps({"age": "int"}),
            "valueMappings": json.dumps({"dept": {"内科": "内", "外科": "外"}}),
            "deriveColumns": json.dumps({"age_group": "age // 10 * 10"})
        }
        result = _run(transformer, large_data, params)

        assert result["report"]["input_rows"] == 10000
        assert result["report"]["performance"]["total_ms"] > 0
        # 每个步骤都有耗时记录
        assert len(result["report"]["performance"]["steps"]) == 7

    def test_pt02_step_timing(self, transformer, large_data, basic_params):
        """PT-02: 步骤级耗时分析"""
        params = {
            **basic_params,
            "typeConversions": json.dumps({"age": "int"}),
            "filterCondition": "age > 30"
        }
        result = _run(transformer, large_data, params)

        steps = result["report"]["performance"]["steps"]
        # 每个步骤都有 duration_ms 和 rows_after
        for step_name, step_info in steps.items():
            assert "duration_ms" in step_info
            assert "rows_after" in step_info
            assert step_info["duration_ms"] >= 0

    def test_it03_step_profiling_detail(self, transformer, large_data, basic_params):
        """PT-03: 步骤级性能详细分析"""
        params = {
            **basic_params,
            "renameFields": json.dumps({"name": "user_name"}),
            "typeConversions": json.dumps({"age": "int"}),
            "filterCondition": "age > 30"
        }
        result = _run(transformer, large_data, params)

        perf = result["report"]["performance"]
        assert perf["total_ms"] > 0
        # 各步骤耗时之和应接近总耗时
        sum_steps = sum(s["duration_ms"] for s in perf["steps"].values())
        assert abs(sum_steps - perf["total_ms"]) < perf["total_ms"] * 0.5  # 允许50%误差


# ============================================================================
# 补充测试 - 清单查漏补缺
# ============================================================================

class TestSupplementary:
    """补充测试 - 覆盖清单中缺失的检查项"""

    def test_type_already_target(self, transformer, basic_params):
        """补充: 类型已是目标类型 → 透传不报错"""
        # age 已经是字符串，再转 str 应该无异常
        data = {"data": [{"age": "25"}, {"age": "30"}], "schema": {"columns": ["age"], "row_count": 2}}
        params = {**basic_params, "typeConversions": json.dumps({"age": "str"})}
        result = _run(transformer, data, params)

        assert result["report"]["transform_summary"]["type_convert_count"] == 1
        assert result["data"][0]["age"] == "25"

    def test_output_format_detailed(self, transformer, basic_data, basic_params):
        """补充: 输出格式详细校验 - report 包含所有必要字段"""
        params = {**basic_params, "typeConversions": json.dumps({"age": "int"})}
        result = _run(transformer, basic_data, params)

        # 顶层字段
        assert "data" in result
        assert "schema" in result
        assert "report" in result

        # schema 字段
        assert "columns" in result["schema"]
        assert "row_count" in result["schema"]

        # report 字段
        report = result["report"]
        assert "input_rows" in report
        assert "output_rows" in report
        assert "transform_summary" in report
        assert "quality_metrics" in report
        assert "warnings" in report
        assert "summary" in report
        assert "performance" in report

        # quality_metrics 字段
        assert "input_columns" in report["quality_metrics"]
        assert "output_columns" in report["quality_metrics"]
        assert "type_mismatch_after" in report["quality_metrics"]

        # performance 字段
        assert "total_ms" in report["performance"]
        assert "steps" in report["performance"]

    def test_data_cleaner_compatibility(self, transformer, basic_params):
        """补充: data_cleaner 输出格式兼容性（模拟真实上游输出）"""
        # 模拟 data_cleaner 的输出格式
        cleaner_output = {
            "data": [
                {"name": "张三", "age": "25", "dept": "内科"},
                {"name": "李四", "age": "30", "dept": "外科"},
            ],
            "schema": {"columns": ["name", "age", "dept"], "row_count": 2},
            "report": {
                "input_rows": 2,
                "output_rows": 2,
                "duplicates_removed": 0,
                "missing_handled": 0,
                "summary": "数据清洗完成"
            }
        }
        params = {
            **basic_params,
            "renameFields": json.dumps({"name": "patient_name"}),
            "typeConversions": json.dumps({"age": "int"})
        }
        result = _run(transformer, cleaner_output, params)

        assert result["report"]["input_rows"] == 2
        assert "patient_name" in result["schema"]["columns"]
        assert result["data"][0]["age"] == 25

    def test_performance_medium_dataset(self, transformer, basic_params):
        """补充: 1000行中等规模 + 多次运行取均值"""
        # 构造 1000 行数据
        rows = [{"name": f"user_{i}", "age": str(20 + i % 50), "dept": "内科" if i % 2 == 0 else "外科"}
                for i in range(1000)]
        data = {"data": rows, "schema": {"columns": ["name", "age", "dept"], "row_count": 1000}}

        params = {
            **basic_params,
            "renameFields": json.dumps({"name": "user_name"}),
            "typeConversions": json.dumps({"age": "int"}),
            "filterCondition": "age > 30",
            "deriveColumns": json.dumps({"age_group": "age // 10 * 10"})
        }

        # 多次运行取均值
        times = []
        for _ in range(5):
            start = time.perf_counter()
            result = _run(transformer, data, params)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = sum(times) / len(times)

        assert result["report"]["input_rows"] == 1000
        assert result["report"]["output_rows"] > 0
        assert result["report"]["performance"]["total_ms"] > 0
        # 1000 行均值应在 1 秒内
        assert avg_time < 1000, f"1000 行平均耗时 {avg_time:.0f}ms，超过 1s 阈值"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
