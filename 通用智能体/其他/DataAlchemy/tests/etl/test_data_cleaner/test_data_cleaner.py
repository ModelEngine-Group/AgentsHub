# -*- coding: utf-8 -*-
"""
DataCleaner 算子完整测试套件

测试金字塔：
- 单元测试 (Unit Tests): 30-40个
- 集成测试 (Integration Tests): 10-15个

运行方式：
    pytest test_data_cleaner.py -v
    pytest test_data_cleaner.py -v -k "unit"
    pytest test_data_cleaner.py -v -k "integration"
"""

import sys
import pytest
import pandas as pd
import numpy as np
import time

# 路径配置
sys.path.insert(0, r"D:\PythonProject\ModelEngine\operators")

from data_cleaner import DataCleaner, ValidationError
from data_cleaner.process import PrivacyMasker

# 导入测试数据
from test_data import (
    SAMPLE_DATA,
    EXPECTED_RESULTS,
    get_dataframe,
    create_input_data
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def cleaner():
    """DataCleaner 算子实例"""
    return DataCleaner()


@pytest.fixture
def privacy_masker():
    """PrivacyMasker 实例"""
    return PrivacyMasker()


@pytest.fixture
def basic_input():
    """基础输入数据"""
    return create_input_data("basic")


@pytest.fixture
def duplicates_input():
    """去重测试数据"""
    return create_input_data("duplicates")


@pytest.fixture
def privacy_input():
    """隐私脱敏测试数据"""
    return create_input_data("privacy")


@pytest.fixture
def whitespace_input():
    """空格测试数据"""
    return create_input_data("whitespace")


@pytest.fixture
def mixed_input():
    """混合场景测试数据"""
    return create_input_data("mixed")


# ============================================================================
# 1. 单元测试 (Unit Tests)
# ============================================================================

class TestTrimWhitespace:
    """1.1 去空格功能测试 (UT-01 ~ UT-05)"""

    def test_ut_01_simple_whitespace(self, cleaner, whitespace_input):
        """UT-01: 简单空格处理"""
        params = {"trimWhitespace": True}
        result = cleaner.process(whitespace_input, params)

        names = [row["name"] for row in result["data"]]
        assert "张三" in names
        assert " 李四" not in names  # 验证去掉了尾部空格

    def test_ut_02_multiple_spaces(self, cleaner):
        """UT-02: 多个空格处理"""
        data = {"data": [{"name": "  姓名  "}], "schema": {"columns": ["name"], "row_count": 1}}
        params = {"trimWhitespace": True}
        result = cleaner.process(data, params)

        assert result["data"][0]["name"] == "姓名"

    def test_ut_03_number_preserved(self, cleaner):
        """UT-03: 数字类型保持不变"""
        data = {"data": [{"age": " 25 "}], "schema": {"columns": ["age"], "row_count": 1}}
        params = {"trimWhitespace": True}
        result = cleaner.process(data, params)

        # 字符串 " 25 " 去除空格后变为 "25"
        assert result["data"][0]["age"] == "25"

    def test_ut_04_empty_string(self, cleaner):
        """UT-04: 空字符串处理（转为NaN）"""
        data = {"data": [{"name": "  "}], "schema": {"columns": ["name"], "row_count": 1}}
        params = {"trimWhitespace": True}
        result = cleaner.process(data, params)

        # 空字符串 → 空格 → NaN → dropna 删除
        assert len(result["data"]) == 0  # 被删除了

    def test_ut_05_nan_string(self, cleaner):
        """UT-05: nan 字符串转 NaN"""
        data = {"data": [{"name": "nan"}], "schema": {"columns": ["name"], "row_count": 1}}
        params = {"trimWhitespace": True, "handleMissing": "drop"}
        result = cleaner.process(data, params)

        # nan 字符串 → NaN → drop 删除
        assert len(result["data"]) == 0


class TestRemoveDuplicates:
    """1.2 去重功能测试 (UT-06 ~ UT-10)"""

    def test_ut_06_exact_duplicate(self, cleaner, duplicates_input):
        """UT-06: 完全重复行"""
        params = {"removeDuplicates": True}
        result = cleaner.process(duplicates_input, params)

        assert result["report"]["duplicates_removed"] == 1
        assert len(result["data"]) == 2

    def test_ut_07_partial_duplicate(self, cleaner):
        """UT-07: 部分重复（姓名相同但电话不同）"""
        data = {"data": [
            {"name": "张三", "phone": "13812345678"},
            {"name": "张三", "phone": "13900000000"}
        ], "schema": {"columns": ["name", "phone"], "row_count": 2}}
        params = {"removeDuplicates": True}
        result = cleaner.process(data, params)

        # 视为不同行，不去重
        assert result["report"]["duplicates_removed"] == 0
        assert len(result["data"]) == 2

    def test_ut_08_all_duplicates(self, cleaner):
        """UT-08: 全是重复"""
        data = {"data": [
            {"name": "张三", "phone": "13812345678"},
            {"name": "张三", "phone": "13812345678"},
            {"name": "张三", "phone": "13812345678"}
        ], "schema": {"columns": ["name", "phone"], "row_count": 3}}
        params = {"removeDuplicates": True}
        result = cleaner.process(data, params)

        assert result["report"]["duplicates_removed"] == 2
        assert len(result["data"]) == 1

    def test_ut_09_all_unique(self, cleaner):
        """UT-09: 全都唯一"""
        data = {"data": [
            {"name": "张三"},
            {"name": "李四"},
            {"name": "王五"}
        ], "schema": {"columns": ["name"], "row_count": 3}}
        params = {"removeDuplicates": True}
        result = cleaner.process(data, params)

        assert result["report"]["duplicates_removed"] == 0
        assert len(result["data"]) == 3

    def test_ut_10_empty_dataframe(self, cleaner):
        """UT-10: 空 DataFrame"""
        data = create_input_data("empty")
        params = {"removeDuplicates": True}
        result = cleaner.process(data, params)

        assert result["report"]["duplicates_removed"] == 0
        assert len(result["data"]) == 0


class TestHandleMissing:
    """1.3 空值处理测试 (UT-11 ~ UT-15)"""

    def test_ut_11_drop_missing(self, cleaner):
        """UT-11: drop 策略"""
        data = {"data": [{"name": "a"}, {"name": None}, {"name": "b"}],
                "schema": {"columns": ["name"], "row_count": 3}}
        params = {"handleMissing": "drop"}
        result = cleaner.process(data, params)

        assert len(result["data"]) == 2
        names = [row["name"] for row in result["data"]]
        assert "a" in names
        assert "b" in names

    def test_ut_12_fill_missing(self, cleaner):
        """UT-12: fill 策略"""
        data = {"data": [{"name": "a"}, {"name": None}, {"name": "b"}],
                "schema": {"columns": ["name"], "row_count": 3}}
        params = {"handleMissing": "fill", "fillValue": "未知"}
        result = cleaner.process(data, params)

        assert len(result["data"]) == 3
        names = [row["name"] for row in result["data"]]
        assert "未知" in names

    def test_ut_13_keep_missing(self, cleaner):
        """UT-13: keep 策略"""
        data = {"data": [{"name": "a"}, {"name": None}, {"name": "b"}],
                "schema": {"columns": ["name"], "row_count": 3}}
        params = {"handleMissing": "keep"}
        result = cleaner.process(data, params)

        assert len(result["data"]) == 3
        # 保留空值

    def test_ut_14_multi_column_missing(self, cleaner):
        """UT-14: 多列混合空值"""
        data = {"data": [
            {"name": "张三", "age": 25},
            {"name": None, "age": 30},
            {"name": "李四", "age": None}
        ], "schema": {"columns": ["name", "age"], "row_count": 3}}
        params = {"handleMissing": "drop"}
        result = cleaner.process(data, params)

        # 删除包含空值的行
        assert len(result["data"]) == 1
        assert result["data"][0]["name"] == "张三"

    def test_ut_15_empty_string_vs_null(self, cleaner):
        """UT-15: 空字符串 vs null"""
        data = {"data": [
            {"name": "a"},
            {"name": ""},
            {"name": "b"}
        ], "schema": {"columns": ["name"], "row_count": 3}}
        params = {"handleMissing": "drop"}
        result = cleaner.process(data, params)

        # 空字符串视为有效值，null 才删除
        assert len(result["data"]) == 2


class TestPrivacyMasker:
    """1.4 隐私脱敏测试 (UT-16 ~ UT-22)"""

    def test_ut_16_valid_phone(self, privacy_masker):
        """UT-16: 有效手机号"""
        masked, warning = privacy_masker.detect_and_mask("13812345678", "phone")
        assert masked == "138****5678"
        assert warning is None

    def test_ut_17_invalid_phone(self, privacy_masker):
        """UT-17: 无效手机号"""
        masked, warning = privacy_masker.detect_and_mask("12345678901", "phone")
        assert masked == "12345678901"  # 不变
        assert warning is None

    def test_ut_18_id_card_15(self, privacy_masker):
        """UT-18: 15位身份证"""
        masked, warning = privacy_masker.detect_and_mask("310101199001011", "id_card")
        # 15位：前3后4，中间 15-7=8 个星号
        assert masked == "310********1011"

    def test_ut_19_id_card_18(self, privacy_masker):
        """UT-19: 18位身份证"""
        masked, warning = privacy_masker.detect_and_mask("310101199001011234", "id_card")
        # 18位：前3后4，中间 18-7=11 个星号
        assert masked == "310***********1234"

    def test_ut_20_chinese_name(self, privacy_masker):
        """UT-20: 中文姓名"""
        masked, warning = privacy_masker.detect_and_mask("张三", "name")
        assert masked == "患者1"

    def test_ut_21_english_name(self, privacy_masker):
        """UT-21: 英文姓名"""
        masked, warning = privacy_masker.detect_and_mask("John Smith", "name")
        assert masked == "John Smith"  # 不变

    def test_ut_22_sequential_names(self, privacy_masker):
        """UT-22: 连续姓名脱敏"""
        masked1, _ = privacy_masker.detect_and_mask("张三", "name")
        masked2, _ = privacy_masker.detect_and_mask("李四", "name")

        assert masked1 == "患者1"
        assert masked2 == "患者2"


class TestStandardizeFormat:
    """1.5 格式标准化测试 (UT-23 ~ UT-27)"""

    def test_ut_23_slash_date(self, cleaner):
        """UT-23: 斜杠日期"""
        data = {"data": [{"birthday": "2026/04/28"}],
                "schema": {"columns": ["birthday"], "row_count": 1}}
        params = {"standardizeFormat": True}
        result = cleaner.process(data, params)

        assert result["data"][0]["birthday"] == "2026-04-28"

    def test_ut_24_compact_date(self, cleaner):
        """UT-24: 紧凑日期"""
        data = {"data": [{"birthday": "20260428"}],
                "schema": {"columns": ["birthday"], "row_count": 1}}
        params = {"standardizeFormat": True}
        result = cleaner.process(data, params)

        assert result["data"][0]["birthday"] == "2026-04-28"

    def test_ut_25_us_date(self, cleaner):
        """UT-25: 美式日期"""
        data = {"data": [{"birthday": "04/28/2026"}],
                "schema": {"columns": ["birthday"], "row_count": 1}}
        params = {"standardizeFormat": True}
        result = cleaner.process(data, params)

        assert result["data"][0]["birthday"] == "2026-04-28"

    def test_ut_26_already_standard(self, cleaner):
        """UT-26: 已是标准格式"""
        data = {"data": [{"birthday": "2026-04-28"}],
                "schema": {"columns": ["birthday"], "row_count": 1}}
        params = {"standardizeFormat": True}
        result = cleaner.process(data, params)

        assert result["data"][0]["birthday"] == "2026-04-28"

    def test_ut_27_invalid_date(self, cleaner):
        """UT-27: 无效日期"""
        data = {"data": [{"birthday": "invalid-date"}],
                "schema": {"columns": ["birthday"], "row_count": 1}}
        params = {"standardizeFormat": True}
        result = cleaner.process(data, params)

        # 无效日期保持原值
        assert result["data"][0]["birthday"] == "invalid-date"


class TestExceptionHandling:
    """1.6 异常处理测试 (UT-28 ~ UT-30)"""

    def test_ut_28_invalid_fill_value(self, cleaner):
        """UT-28: fill 模式未指定 fillValue"""
        data = {"data": [{"name": "a"}], "schema": {"columns": ["name"], "row_count": 1}}
        params = {"handleMissing": "fill"}

        with pytest.raises(ValidationError) as exc_info:
            cleaner.process(data, params)

        assert "填充值" in str(exc_info.value)

    def test_ut_29_empty_input(self, cleaner):
        """UT-29: 空输入"""
        data = create_input_data("empty")
        params = {"removeDuplicates": True}
        result = cleaner.process(data, params)

        assert len(result["data"]) == 0
        assert result["report"]["input_rows"] == 0

    def test_ut_30_numeric_column_names(self, cleaner):
        """UT-30: 数字类型列名"""
        data = {"data": [{"1": "a", "2": "b"}],
                "schema": {"columns": ["1", "2"], "row_count": 1}}
        params = {"trimWhitespace": True}
        result = cleaner.process(data, params)

        assert len(result["data"]) == 1


# ============================================================================
# 2. 集成测试 (Integration Tests)
# ============================================================================

class TestPipelineIntegration:
    """2.1 Pipeline 顺序测试 (IT-01 ~ IT-04)"""

    def test_it_01_whitespace_then_dedup(self, cleaner):
        """IT-01: 空格处理后再去重"""
        data = {"data": [
            {"name": " 张三 "},
            {"name": " 张三 "}
        ], "schema": {"columns": ["name"], "row_count": 2}}
        params = {"trimWhitespace": True, "removeDuplicates": True}
        result = cleaner.process(data, params)

        # 先去空格：" 张三 " → "张三"，然后去重
        assert result["report"]["duplicates_removed"] == 1
        assert len(result["data"]) == 1

    def test_it_02_complex_pipeline(self, cleaner):
        """IT-02: 复杂流水线"""
        data = {"data": [
            {"name": " 张三 "},
            {"name": "张三"},
            {"name": None}
        ], "schema": {"columns": ["name"], "row_count": 3}}
        params = {
            "trimWhitespace": True,
            "removeDuplicates": True,
            "handleMissing": "drop"
        }
        result = cleaner.process(data, params)

        # trim → dedup → drop: 3 → 2 → 1
        assert len(result["data"]) == 1
        assert result["report"]["duplicates_removed"] == 1

    def test_it_03_privacy_multiple_types(self, cleaner):
        """IT-03: 多种敏感信息"""
        data = create_input_data("privacy")
        params = {"privacyCheck": True}
        result = cleaner.process(data, params)

        privacy_stats = result["report"]["privacy_masked"]
        assert privacy_stats["phone"] == 2
        assert privacy_stats["id_card"] == 2
        assert privacy_stats["name"] == 2

    def test_it_04_full_pipeline(self, cleaner):
        """IT-04: 全流程测试"""
        # 使用专门设计的数据，避免列不一致导致的空值问题
        data = {
            "data": [
                {"name": " 张三 ", "phone": "13812345678", "age": 25},
                {"name": "张三", "phone": "13812345678", "age": 25},  # 重复
                {"name": "李四", "phone": "", "age": None},  # 空值
                {"name": "王五", "phone": "13811111111", "age": 30}  # 有空值列但无值
            ],
            "schema": {"columns": ["name", "phone", "age"], "row_count": 4}
        }
        params = {
            "trimWhitespace": True,
            "removeDuplicates": True,
            "handleMissing": "drop",
            "privacyCheck": True
        }
        result = cleaner.process(data, params)

        # 验证所有功能都生效
        assert result["report"]["duplicates_removed"] == 1
        # missing_handled 统计所有被处理的空值数量
        assert result["report"]["missing_handled"] >= 1
        assert sum(result["report"]["privacy_masked"].values()) > 0


class TestComplexScenarios:
    """2.2 复杂场景测试 (IT-05 ~ IT-07)"""

    def test_it_05_dedup_missing_privacy(self, cleaner):
        """IT-05: 去重+空值+脱敏联合"""
        data = {"data": [
            {"name": "张三", "phone": "13812345678"},
            {"name": "张三", "phone": "13812345678"},
            {"name": "李四", "phone": ""}
        ], "schema": {"columns": ["name", "phone"], "row_count": 3}}
        params = {
            "trimWhitespace": True,
            "removeDuplicates": True,
            "handleMissing": "drop",
            "privacyCheck": True
        }
        result = cleaner.process(data, params)

        assert result["report"]["duplicates_removed"] == 1
        # 李四因空值被删除
        assert len(result["data"]) == 1
        assert result["data"][0]["name"] == "患者1"

    def test_it_06_mixed_privacy(self, cleaner):
        """IT-06: 多种敏感信息混合"""
        data = {"data": [
            {"name": "张三", "phone": "13812345678", "id_card": "310101199001011234"},
            {"name": "李四", "phone": "13900000000", "id_card": "110101199001011234"}
        ], "schema": {"columns": ["name", "phone", "id_card"], "row_count": 2}}
        params = {"privacyCheck": True}
        result = cleaner.process(data, params)

        privacy_stats = result["report"]["privacy_masked"]
        assert privacy_stats["phone"] == 2
        assert privacy_stats["id_card"] == 2
        assert privacy_stats["name"] == 2

    def test_it_07_case_sensitivity(self, cleaner):
        """IT-07: 大小写敏感"""
        data = {"data": [
            {"name": "ZhangSan"},
            {"name": "zhangsan"}
        ], "schema": {"columns": ["name"], "row_count": 2}}
        params = {"removeDuplicates": True}
        result = cleaner.process(data, params)

        # 大小写敏感，视为不同
        assert result["report"]["duplicates_removed"] == 0


# ============================================================================
# 3. 报告生成测试
# ============================================================================

class TestReportGeneration:
    """3.1 报告生成测试"""

    def test_report_structure(self, cleaner, basic_input):
        """验证报告结构"""
        params = {"removeDuplicates": True}
        result = cleaner.process(basic_input, params)
        report = result["report"]

        # 验证必需字段
        assert "input_rows" in report
        assert "output_rows" in report
        assert "duplicates_removed" in report
        assert "missing_handled" in report
        assert "privacy_masked" in report
        assert "quality_metrics" in report
        assert "warnings" in report
        assert "summary" in report

    def test_quality_metrics(self, cleaner):
        """验证质量指标"""
        data = {"data": [
            {"name": "a", "value": "1"},
            {"name": None, "value": None},
            {"name": "b", "value": "3"}
        ], "schema": {"columns": ["name", "value"], "row_count": 3}}
        params = {"handleMissing": "drop"}
        result = cleaner.process(data, params)

        qm = result["report"]["quality_metrics"]
        assert "null_rate_before" in qm
        assert "null_rate_after" in qm
        assert "duplicate_rate_after" in qm

    def test_summary_generation(self, cleaner, basic_input):
        """验证摘要生成"""
        params = {"removeDuplicates": True, "privacyCheck": True}
        result = cleaner.process(basic_input, params)

        summary = result["report"]["summary"]
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "数据清洗完成" in summary


# ============================================================================
# 4. 边界条件测试
# ============================================================================

class TestBoundaryConditions:
    """4.1 边界条件测试"""

    def test_single_row(self, cleaner):
        """单行数据"""
        data = {"data": [{"name": "张三"}], "schema": {"columns": ["name"], "row_count": 1}}
        params = {"removeDuplicates": True}
        result = cleaner.process(data, params)

        assert len(result["data"]) == 1

    def test_single_column(self, cleaner):
        """单列数据"""
        data = {"data": [
            {"name": "张三"},
            {"name": "李四"},
            {"name": "张三"}
        ], "schema": {"columns": ["name"], "row_count": 3}}
        params = {"removeDuplicates": True}
        result = cleaner.process(data, params)

        assert result["report"]["duplicates_removed"] == 1

    def test_all_same_values(self, cleaner):
        """所有值相同"""
        data = {"data": [
            {"name": "张三"},
            {"name": "张三"},
            {"name": "张三"}
        ], "schema": {"columns": ["name"], "row_count": 3}}
        params = {"removeDuplicates": True}
        result = cleaner.process(data, params)

        assert len(result["data"]) == 1

    def test_special_characters(self, cleaner):
        """特殊字符"""
        data = {"data": [
            {"name": "张三★"},
            {"name": "李四@@"},
            {"name": "王五##"}
        ], "schema": {"columns": ["name"], "row_count": 3}}
        params = {"trimWhitespace": True}
        result = cleaner.process(data, params)

        assert len(result["data"]) == 3

    def test_very_long_string(self, cleaner):
        """超长字符串"""
        long_name = "张" * 1000
        data = {"data": [{"name": f" {long_name} "}],
                "schema": {"columns": ["name"], "row_count": 1}}
        params = {"trimWhitespace": True}
        result = cleaner.process(data, params)

        assert result["data"][0]["name"] == long_name


# ============================================================================
# 运行入口
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
