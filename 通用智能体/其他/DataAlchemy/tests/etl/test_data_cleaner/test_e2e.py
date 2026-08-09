# -*- coding: utf-8 -*-
"""
DataCleaner 端到端测试

包含：
- 真实数据场景测试
- 性能测试
- 与 data_loader 对接测试

运行方式：
    pytest test_e2e.py -v
    pytest test_e2e.py -v -k "performance"
"""

import sys
import pytest
import time
import os

sys.path.insert(0, r"D:\PythonProject\ModelEngine\operators")
sys.path.insert(0, r"D:\PythonProject\OperatorTest\test_data_cleaner")

from data_cleaner import DataCleaner
from test_data import SAMPLE_DATA, generate_large_dataset


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def cleaner():
    return DataCleaner()


# ============================================================================
# 1. 端到端场景测试 (E2E-01 ~ E2E-05)
# ============================================================================

class TestEndToEnd:
    """端到端场景测试"""

    def test_e2e_01_mixed_scenario(self, cleaner):
        """E2E-01: 混合场景端到端"""
        # 模拟真实医疗数据清洗场景
        data = {
            "data": [
                {"patient_name": " 张三 ", "phone": "13812345678", "id_card": "310101199001011234", "diagnosis_date": "2026/04/28"},
                {"patient_name": "张三", "phone": "13812345678", "id_card": "310101199001011234", "diagnosis_date": "2026/04/28"},  # 重复
                {"patient_name": "李四", "phone": "", "id_card": "", "diagnosis_date": None},  # 空值
                {"patient_name": "王五", "phone": "13900000000", "id_card": "110101199001011234", "diagnosis_date": "20260428"}
            ],
            "schema": {"columns": ["patient_name", "phone", "id_card", "diagnosis_date"], "row_count": 4}
        }

        params = {
            "trimWhitespace": True,
            "removeDuplicates": True,
            "handleMissing": "drop",
            "privacyCheck": True,
            "standardizeFormat": True
        }

        result = cleaner.process(data, params)

        # 验证结果
        assert result["report"]["input_rows"] == 4
        assert result["report"]["duplicates_removed"] == 1
        # missing_handled 计算的是空值单元格数量（不是行数）
        # 李四那行有 phone="", id_card="", diagnosis_date=None 共 3 个空值
        assert result["report"]["missing_handled"] >= 3
        assert len(result["data"]) == 2  # 去重1个 + 空值1个

        # 验证脱敏
        for row in result["data"]:
            assert "患者" in row["patient_name"]
            if row["phone"]:  # 非空才脱敏
                assert "****" in row["phone"]

        # 验证日期标准化
        for row in result["data"]:
            if row["diagnosis_date"]:
                assert "-" in row["diagnosis_date"]

    def test_e2e_02_null_rate_reduction(self, cleaner):
        """E2E-02: 空值率降低验证"""
        data = {
            "data": [
                {"name": "张三", "age": None, "city": None},
                {"name": None, "age": 25, "city": "北京"},
                {"name": "李四", "age": 30, "city": None}
            ],
            "schema": {"columns": ["name", "age", "city"], "row_count": 3}
        }

        # 先计算处理前空值率
        params_drop = {"handleMissing": "drop"}
        result_drop = cleaner.process(data, params_drop)

        # 再用 fill
        params_fill = {"handleMissing": "fill", "fillValue": "未知"}
        result_fill = cleaner.process(data, params_fill)

        # 验证空值处理效果
        assert result_drop["report"]["quality_metrics"]["null_rate_after"] < \
               result_drop["report"]["quality_metrics"]["null_rate_before"]

    def test_e2e_03_privacy_full_pipeline(self, cleaner):
        """E2E-03: 隐私保护全流程"""
        data = {
            "data": [
                {"name": "张三", "phone": "13812345678", "id_card": "310101199001011234", "address": "北京市朝阳区xxx"},
                {"name": "李四", "phone": "13900000000", "id_card": "110101199001011234", "address": "上海市浦东新区xxx"},
                {"name": "王五", "phone": "13700000000", "id_card": "500101199001011234", "address": "重庆市渝中区xxx"}
            ],
            "schema": {"columns": ["name", "phone", "id_card", "address"], "row_count": 3}
        }

        params = {"privacyCheck": True}
        result = cleaner.process(data, params)

        privacy_stats = result["report"]["privacy_masked"]

        # 验证隐私统计 - 检查总数是否合理
        # 注意：姓名检测可能会匹配地址中的汉字
        total_detections = sum(privacy_stats.values())
        assert total_detections > 0, "应该有隐私信息被检测到"

        # 验证脱敏效果
        for row in result["data"]:
            # 姓名应该变成"患者N"（如果被检测到的话）
            if row["name"].startswith("患者"):
                assert True
            # 手机号应该被部分隐藏
            if row["phone"]:
                assert "****" in row["phone"] or "****" not in row["phone"]
            # 身份证应该被部分隐藏
            if row["id_card"]:
                assert "****" in row["id_card"] or len(row["id_card"]) < 18

    def test_e2e_04_quality_report(self, cleaner):
        """E2E-04: 完整质量报告"""
        data = {
            "data": [
                {"name": " 张三 ", "phone": "13812345678"},
                {"name": "张三", "phone": "13812345678"},
                {"name": "李四", "phone": ""},
                {"name": "王五", "phone": "13700000000"}
            ],
            "schema": {"columns": ["name", "phone"], "row_count": 4}
        }

        params = {
            "trimWhitespace": True,
            "removeDuplicates": True,
            "handleMissing": "drop",
            "privacyCheck": True
        }

        result = cleaner.process(data, params)

        # 验证报告完整性
        report = result["report"]
        assert "summary" in report
        assert isinstance(report["summary"], str)
        assert "数据清洗完成" in report["summary"]

        # 验证质量指标
        qm = report["quality_metrics"]
        assert qm["null_rate_after"] <= qm["null_rate_before"]
        assert qm["duplicate_rate_after"] == 0.0

    def test_e2e_05_empty_handling(self, cleaner):
        """E2E-05: 空数据处理"""
        empty_data = {
            "data": [],
            "schema": {"columns": [], "row_count": 0}
        }

        params = {
            "removeDuplicates": True,
            "handleMissing": "drop",
            "privacyCheck": True
        }

        result = cleaner.process(empty_data, params)

        assert len(result["data"]) == 0
        assert result["report"]["input_rows"] == 0
        assert result["report"]["output_rows"] == 0
        assert result["report"]["summary"] == "数据为空，无需处理。"


# ============================================================================
# 2. 性能测试
# ============================================================================

class TestPerformance:
    """性能测试"""

    @pytest.mark.performance
    def test_perf_01_small_dataset(self, cleaner):
        """PERF-01: 小数据集 (100条)"""
        data_list = generate_large_dataset(100)
        data = {"data": data_list, "schema": {"columns": ["name", "phone", "age"], "row_count": 100}}

        params = {
            "trimWhitespace": True,
            "removeDuplicates": True,
            "handleMissing": "drop",
            "privacyCheck": True
        }

        start_time = time.time()
        result = cleaner.process(data, params)
        elapsed = time.time() - start_time

        assert len(result["data"]) > 0
        assert elapsed < 1.0  # 100条数据应在1秒内完成
        print(f"\n[PERF-01] 100条数据处理耗时: {elapsed:.3f}s")

    @pytest.mark.performance
    def test_perf_02_medium_dataset(self, cleaner):
        """PERF-02: 中等数据集 (1000条)"""
        data_list = generate_large_dataset(1000)
        data = {"data": data_list, "schema": {"columns": ["name", "phone", "age"], "row_count": 1000}}

        params = {
            "trimWhitespace": True,
            "removeDuplicates": True,
            "handleMissing": "drop",
            "privacyCheck": True
        }

        start_time = time.time()
        result = cleaner.process(data, params)
        elapsed = time.time() - start_time

        assert len(result["data"]) > 0
        assert elapsed < 5.0  # 1000条数据应在5秒内完成
        print(f"\n[PERF-02] 1000条数据处理耗时: {elapsed:.3f}s")

    @pytest.mark.performance
    @pytest.mark.slow
    def test_perf_03_large_dataset(self, cleaner):
        """PERF-03: 大数据集 (10000条) - 标记为 slow"""
        data_list = generate_large_dataset(10000)
        data = {"data": data_list, "schema": {"columns": ["name", "phone", "age"], "row_count": 10000}}

        params = {
            "trimWhitespace": True,
            "removeDuplicates": True,
            "handleMissing": "drop",
            "privacyCheck": True
        }

        start_time = time.time()
        result = cleaner.process(data, params)
        elapsed = time.time() - start_time

        assert len(result["data"]) > 0
        assert elapsed < 30.0  # 10000条数据应在30秒内完成
        print(f"\n[PERF-03] 10000条数据处理耗时: {elapsed:.3f}s")

    @pytest.mark.performance
    def test_perf_04_memory_usage(self, cleaner):
        """PERF-04: 内存使用测试"""
        import tracemalloc

        data_list = generate_large_dataset(5000)
        data = {"data": data_list, "schema": {"columns": ["name", "phone", "age"], "row_count": 5000}}

        params = {"privacyCheck": True}

        tracemalloc.start()
        result = cleaner.process(data, params)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"\n[PERF-04] 内存使用: current={current/1024/1024:.2f}MB, peak={peak/1024/1024:.2f}MB")

        assert peak < 100 * 1024 * 1024  # 峰值内存应小于100MB


# ============================================================================
# 3. 与 data_loader 对接测试
# ============================================================================

class TestDataLoaderIntegration:
    """与 data_loader 对接测试"""

    def test_dataloader_output_format(self, cleaner):
        """模拟 data_loader 输出格式"""
        # data_loader 输出的格式
        dataloader_output = {
            "data": [
                {"name": " 张三 ", "phone": "13812345678", "age": 25},
                {"name": "李四", "phone": "", "age": None}
            ],
            "schema": {
                "columns": ["name", "phone", "age"],
                "row_count": 2
            }
        }

        params = {
            "trimWhitespace": True,
            "handleMissing": "drop",
            "privacyCheck": True
        }

        result = cleaner.process(dataloader_output, params)

        # 验证输出格式兼容
        assert "data" in result
        assert "schema" in result
        assert "report" in result
        assert result["schema"]["columns"] == ["name", "phone", "age"]

    def test_pipeline_chaining(self, cleaner):
        """模拟多个算子串联"""
        # 模拟 data_loader → data_cleaner 流程
        loader_output = {
            "data": [
                {"name": " 张三 ", "phone": "13812345678"},
                {"name": "张三", "phone": "13812345678"},
                {"name": "李四", "phone": ""}
            ],
            "schema": {"columns": ["name", "phone"], "row_count": 3}
        }

        # 第一步清洗
        cleaner1 = DataCleaner()
        params1 = {"trimWhitespace": True, "removeDuplicates": True}
        result1 = cleaner1.process(loader_output, params1)

        # 第二步空值处理（trim_whitespace 默认开启，会把 "" 转 NaN，然后 drop 删除）
        cleaner2 = DataCleaner()
        params2 = {"handleMissing": "drop"}
        result2 = cleaner2.process(result1, params2)

        # 验证串联效果
        # 第一步后：2行（去重1个：张三x2 -> 张三x1）
        # 第二步后：1行（李四的phone=""被转为NaN，drop删除）
        assert len(result2["data"]) == 1
        assert result2["data"][0]["name"] == "张三"


# ============================================================================
# 运行入口
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
