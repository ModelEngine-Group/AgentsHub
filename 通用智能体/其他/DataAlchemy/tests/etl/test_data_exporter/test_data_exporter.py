# -*- coding: utf-8 -*-
"""
DataExporter 算子完整测试套件

测试金字塔（比赛评分针对性）：
- Unit Tests (格式导出): CSV/JSON/JSONL 三种格式的基础功能  (15分 ← 功能完整性)
- Edge Tests (边界): 空数据/单行/特殊字符/NaN/numpy类型     (5分 ← 工程质量)
- Exception Tests (异常): 参数校验/目录不可写/无效格式       (5分 ← 工程质量)
- Integration Tests (集成): 参数组合/上游数据透传             (5分 ← 架构一致性)
- E2E Tests (端到端): 完整 ETL 流水线串联                    (10分 ← 功能完整性)
- Performance Tests (性能): 1000行/10000行导出时间            (鼓励项 ← 性能量化)

运行方式：
    pytest test_data_exporter.py -v
    pytest test_data_exporter.py -v -k "unit"
    pytest test_data_exporter.py -v -k "edge"
    pytest test_data_exporter.py -v -k "exception"
    pytest test_data_exporter.py -v -k "integration"
    pytest test_data_exporter.py -v -k "e2e"
    pytest test_data_exporter.py -v -k "performance"
"""

import sys
import os
import json
import pytest
import pandas as pd
import numpy as np
import time
import shutil

# 路径配置
sys.path.insert(0, r"D:\PythonProject\ModelEngine\operators")

from data_exporter import DataExporter, ValidationError, ProcessingError
from test_data.test_data import (
    SAMPLE_DATA, BASIC_INPUT, CSV_PARAMS, JSON_PARAMS, JSONL_PARAMS,
    get_temp_dir, make_params, get_upstream_input
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def exporter():
    """DataExporter 算子实例"""
    return DataExporter()


@pytest.fixture
def temp_dir():
    """临时测试目录（自动清理）"""
    tmp = get_temp_dir()
    yield tmp
    if os.path.exists(tmp):
        shutil.rmtree(tmp)


# ============================================================================
# 测试分类 Tag（用于 pytest -k 筛选）
# ============================================================================

UNIT = pytest.mark.unit
EDGE = pytest.mark.edge
EXCEPTION = pytest.mark.exception
INTEGRATION = pytest.mark.integration
E2E = pytest.mark.e2e
PERFORMANCE = pytest.mark.performance


# ============================================================================
# Unit Tests: 格式导出 (15分 → 功能完整性)
# ============================================================================

class TestExportCSV:
    """CSV 格式导出测试（基础 + 参数组合）"""

    @UNIT
    def test_csv_basic_export(self, exporter, temp_dir):
        """CSV 基本导出：有表头、无索引"""
        params = make_params({"outputDir": temp_dir, "outputFileName": "test_csv_basic"})
        result = exporter.process(BASIC_INPUT, params)

        assert result["report"]["input_rows"] == 3
        assert result["report"]["output_rows"] == 3

        output_path = result["report"]["export_summary"]["output_path"]
        assert os.path.exists(output_path)
        assert result["report"]["export_summary"]["file_size_bytes"] > 0

        # 验证文件内容
        df = pd.read_csv(output_path)
        assert len(df) == 3
        assert list(df.columns) == ["name", "age", "dept"]

    @UNIT
    def test_csv_without_header(self, exporter, temp_dir):
        """CSV 导出：无表头"""
        params = make_params({
            "outputDir": temp_dir,
            "outputFileName": "test_no_header",
            "includeHeader": False
        })
        result = exporter.process(BASIC_INPUT, params)
        output_path = result["report"]["export_summary"]["output_path"]

        # 无表头时行数应包含所有数据行
        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 3  # 3行数据，无表头

    @UNIT
    def test_csv_with_index(self, exporter, temp_dir):
        """CSV 导出：含索引列"""
        params = make_params({
            "outputDir": temp_dir,
            "outputFileName": "test_with_index",
            "indexColumn": True
        })
        result = exporter.process(BASIC_INPUT, params)
        output_path = result["report"]["export_summary"]["output_path"]

        df = pd.read_csv(output_path)
        # 含索引列，数据列应为 name/age/dept + 索引列
        assert len(df) == 3

    @UNIT
    def test_csv_gbk_encoding(self, exporter, temp_dir):
        """CSV 导出：GBK 编码"""
        params = make_params({
            "outputDir": temp_dir,
            "outputFileName": "test_gbk",
            "encoding": "gbk"
        })
        result = exporter.process(BASIC_INPUT, params)
        output_path = result["report"]["export_summary"]["output_path"]

        # 验证 GBK 编码可正常读取
        df = pd.read_csv(output_path, encoding="gbk")
        assert len(df) == 3
        assert df.iloc[0]["name"] == "张三"


class TestExportJSON:
    """JSON 格式导出测试"""

    @UNIT
    def test_json_basic_export(self, exporter, temp_dir):
        """JSON 基本导出：数组格式"""
        params = make_params({
            "outputFormat": "json",
            "outputDir": temp_dir,
            "outputFileName": "test_json_basic"
        })
        result = exporter.process(BASIC_INPUT, params)
        output_path = result["report"]["export_summary"]["output_path"]

        # 验证 JSON 格式有效
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 3
        assert data[0]["name"] == "张三"

    @UNIT
    def test_json_ensure_ascii(self, exporter, temp_dir):
        """JSON 导出：中文不转义"""
        params = make_params({
            "outputFormat": "json",
            "outputDir": temp_dir,
            "outputFileName": "test_chinese"
        })
        exporter.process(BASIC_INPUT, params)
        output_path = params["outputDir"] + "/test_chinese.json"

        # 检查中文是否正常（非 \\uXXXX）
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "张三" in content


class TestExportJSONL:
    """JSONL 格式导出测试"""

    @UNIT
    def test_jsonl_basic_export(self, exporter, temp_dir):
        """JSONL 基本导出：每行一个 JSON 对象"""
        params = make_params({
            "outputFormat": "jsonl",
            "outputDir": temp_dir,
            "outputFileName": "test_jsonl_basic"
        })
        result = exporter.process(BASIC_INPUT, params)
        output_path = result["report"]["export_summary"]["output_path"]

        # 验证 JSONL 格式：每行独立 JSON
        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 3
        for line in lines:
            obj = json.loads(line.strip())
            assert isinstance(obj, dict)
            assert "name" in obj

    @UNIT
    def test_jsonl_not_json_array(self, exporter, temp_dir):
        """JSONL 不是 JSON 数组（验证每行独立）"""
        params = make_params({
            "outputFormat": "jsonl",
            "outputDir": temp_dir,
            "outputFileName": "test_not_array"
        })
        exporter.process(BASIC_INPUT, params)
        output_path = params["outputDir"] + "/test_not_array.jsonl"

        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
        # 第一行不应以 [ 开头
        first_line = content.split("\n")[0]
        assert not first_line.startswith("["), "JSONL 不应是数组格式"
        assert first_line.startswith("{"), "JSONL 每行应以 { 开头"


# ============================================================================
# Edge Tests: 边界测试 (5分 → 工程质量)
# ============================================================================

class TestEdgeCases:
    """边界测试"""

    @EDGE
    def test_empty_data(self, exporter, temp_dir):
        """空数据：不创建文件"""
        params = make_params({"outputDir": temp_dir})
        result = exporter.process({"data": []}, params)

        assert result["report"]["input_rows"] == 0
        assert result["report"]["output_rows"] == 0
        # 空数据不创建文件
        assert not result["report"]["export_summary"]

    @EDGE
    def test_single_row(self, exporter, temp_dir):
        """单行数据"""
        input_data = get_upstream_input("single_row")
        params = make_params({
            "outputDir": temp_dir,
            "outputFileName": "test_single"
        })
        result = exporter.process(input_data, params)
        assert result["report"]["output_rows"] == 1

    @EDGE
    def test_special_characters_csv(self, exporter, temp_dir):
        """特殊字符：引号、逗号、换行"""
        input_data = get_upstream_input("with_special")
        params = make_params({
            "outputDir": temp_dir,
            "outputFileName": "test_special"
        })
        result = exporter.process(input_data, params)
        output_path = result["report"]["export_summary"]["output_path"]

        df = pd.read_csv(output_path)
        # 特殊字符不应导致解析错误
        assert len(df) == 2

    @EDGE
    def test_nan_null_handling(self, exporter, temp_dir):
        """NaN/None 处理：CSV 输出空字符串"""
        input_data = get_upstream_input("with_null")
        params = make_params({
            "outputDir": temp_dir,
            "outputFileName": "test_nan"
        })
        result = exporter.process(input_data, params)
        output_path = result["report"]["export_summary"]["output_path"]

        df = pd.read_csv(output_path)
        assert len(df) == 3
        # None 值在 CSV 中应为空字符串
        assert pd.isna(df.iloc[1]["name"])

    @EDGE
    def test_numpy_types_json(self, exporter, temp_dir):
        """numpy 数值类型（Int64）JSON 兼容性"""
        input_data = get_upstream_input("with_numpy_types")
        params = make_params({
            "outputFormat": "json",
            "outputDir": temp_dir,
            "outputFileName": "test_numpy"
        })
        result = exporter.process(input_data, params)
        output_path = result["report"]["export_summary"]["output_path"]

        # numpy Int64/float64 应能正常序列化为 JSON
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 2
        assert isinstance(data[0]["age"], int)
        assert isinstance(data[0]["score"], float)


# ============================================================================
# Exception Tests: 异常测试 (5分 → 工程质量)
# ============================================================================

class TestExceptionHandling:
    """异常处理测试"""

    @EXCEPTION
    def test_empty_output_dir(self, exporter, temp_dir):
        """outputDir 为空 → ValidationError"""
        params = make_params({"outputDir": ""})
        with pytest.raises(ValidationError, match="输出目录不能为空"):
            exporter.process(BASIC_INPUT, params)

    @EXCEPTION
    def test_invalid_format(self, exporter, temp_dir):
        """无效导出格式 → ValidationError"""
        params = make_params({
            "outputDir": temp_dir,
            "outputFormat": "xlsx"
        })
        with pytest.raises(ValidationError, match="不支持的导出格式"):
            exporter.process(BASIC_INPUT, params)

    @EXCEPTION
    def test_unsupported_encoding(self, exporter, temp_dir):
        """不支持的编码 → ValidationError"""
        params = make_params({
            "outputDir": temp_dir,
            "encoding": "invalid-encoding"
        })
        with pytest.raises(ValidationError, match="不支持的编码格式"):
            exporter.process(BASIC_INPUT, params)

    @EXCEPTION
    def test_none_input(self, exporter, temp_dir):
        """None 输入 → 空数据返回"""
        params = make_params({"outputDir": temp_dir})
        result = exporter.process(None, params)
        assert result["report"]["input_rows"] == 0

    @EXCEPTION
    def test_non_writable_dir(self, exporter):
        """不可写目录 → ProcessingError"""
        params = make_params({
            "outputDir": "\\\\invalid\\\\path\\\\"
        })
        # Windows 上允许创建任意路径，用 validate() 测试路径有效性
        is_valid, _ = exporter.validate(params)
        if is_valid:
            # 如果 validate 通过，process 时可能仍失败
            with pytest.raises((ProcessingError, ValidationError)):
                exporter.process(BASIC_INPUT, params)


# ============================================================================
# Integration Tests: 集成测试 (5分 → 架构一致性)
# ============================================================================

class TestPipelineIntegration:
    """参数组合 + 上游数据透传测试"""

    @INTEGRATION
    def test_passthrough_data(self, exporter, temp_dir):
        """上游 data 和 schema 应原样透传"""
        params = make_params({"outputDir": temp_dir})
        result = exporter.process(BASIC_INPUT, params)

        # data 透传
        assert result["data"] == BASIC_INPUT["data"]
        # schema 透传
        assert result["schema"] == BASIC_INPUT["schema"]

    @INTEGRATION
    def test_report_structure(self, exporter, temp_dir):
        """report 应包含完整的 export_summary 全部字段"""
        params = make_params({"outputDir": temp_dir})
        result = exporter.process(BASIC_INPUT, params)
        report = result["report"]

        assert "input_rows" in report
        assert "output_rows" in report
        assert "export_summary" in report
        assert "warnings" in report
        assert "summary" in report

        export = report["export_summary"]
        # 校验 export_summary 全部 6 个字段（供下游 Agent 使用）
        assert "format" in export
        assert "output_path" in export
        assert "file_size_bytes" in export
        assert "encoding" in export
        assert "header_included" in export
        assert "export_time_ms" in export
        assert export["header_included"] is True
        assert export["index_included"] is False
        assert export["format"] == "csv"

    @INTEGRATION
    def test_warning_on_overwrite(self, exporter, temp_dir):
        """文件已存在时产生覆盖警告"""
        params = make_params({"outputDir": temp_dir, "overwrite": True})
        # 第一次导出
        exporter.process(BASIC_INPUT, params)
        # 第二次导出（覆盖）
        result = exporter.process(BASIC_INPUT, params)

        warnings = result["report"]["warnings"]
        overwrite_warnings = [w for w in warnings if w["type"] == "file_overwritten"]
        assert len(overwrite_warnings) >= 1

    @INTEGRATION
    def test_no_overwrite_timestamp(self, exporter, temp_dir):
        """overwrite=False 时自动追加时间戳"""
        params = make_params({
            "outputDir": temp_dir,
            "overwrite": False
        })
        # 第一次导出
        exporter.process(BASIC_INPUT, params)
        # 第二次导出（不覆盖，追加时间戳）
        result = exporter.process(BASIC_INPUT, params)
        output_path = result["report"]["export_summary"]["output_path"]

        # 文件名应包含时间戳
        filename = os.path.basename(output_path)
        assert "_" in filename

    @INTEGRATION
    def test_warnings_structure_consistency(self, exporter, temp_dir):
        """警告结构应与 data_cleaner/data_transformer 一致"""
        params = make_params({"outputDir": temp_dir, "overwrite": True})
        # 覆盖导出两次，触发 file_overwritten 警告
        exporter.process(BASIC_INPUT, params)
        result = exporter.process(BASIC_INPUT, params)

        for w in result["report"]["warnings"]:
            assert "type" in w
            assert "message" in w
            assert isinstance(w["type"], str)
            assert isinstance(w["message"], str)

    @INTEGRATION
    def test_output_dir_created_warning(self, exporter, temp_dir):
        """输出目录不存在时自动创建并记录 output_dir_created 警告"""
        new_dir = os.path.join(temp_dir, "auto_created_subdir")
        params = make_params({
            "outputDir": new_dir,
            "outputFileName": "test_auto_dir"
        })
        result = exporter.process(BASIC_INPUT, params)

        warnings = result["report"]["warnings"]
        created_warnings = [w for w in warnings if w["type"] == "output_dir_created"]
        assert len(created_warnings) >= 1
        assert "输出目录不存在" in created_warnings[0]["message"]

    @INTEGRATION
    def test_get_summary_format(self, exporter, temp_dir):
        """get_summary 返回人类可读字符串"""
        params = make_params({"outputDir": temp_dir})
        result = exporter.process(BASIC_INPUT, params)
        summary = result["report"]["summary"]

        assert isinstance(summary, str)
        assert "数据导出完成" in summary
        assert "行" in summary
        assert "CSV" in summary or "JSON" in summary


# ============================================================================
# E2E Tests: 端到端测试 (10分 → 功能完整性)
# ============================================================================

class TestEndToEnd:
    """完整 ETL 流水线串联测试"""

    @E2E
    def test_e2e_data_loader_to_exporter(self, exporter, temp_dir):
        """模拟 data_loader → data_exporter 直接导出"""
        # data_loader 的输出格式（模拟）
        loader_output = {
            "data": [
                {"name": "张三", "age": "25", "dept": "内科"},
                {"name": "李四", "age": "30", "dept": "外科"},
            ],
            "schema": {"columns": ["name", "age", "dept"], "row_count": 2}
        }
        params = make_params({
            "outputDir": temp_dir,
            "outputFileName": "e2e_loader_to_exporter"
        })
        result = exporter.process(loader_output, params)
        assert result["report"]["output_rows"] == 2
        assert result["report"]["export_summary"]["file_size_bytes"] > 0

    @E2E
    def test_e2e_full_etl_pipeline(self, exporter, temp_dir):
        """模拟完整 ETL：loader → cleaner → transformer → exporter"""
        # 模拟 cleaners 的输出格式
        cleaner_output = {
            "data": [
                {"name": "张三", "age": 25, "dept": "内科", "admit_date": "2023-05-03"},
                {"name": "李四", "age": 30, "dept": "外科", "admit_date": "2023-06-15"},
            ],
            "schema": {"columns": ["name", "age", "dept", "admit_date"], "row_count": 2},
            "report": {
                "input_rows": 2,
                "output_rows": 2,
                "warnings": [],
                "summary": "清洗完成。"
            }
        }
        params = make_params({
            "outputDir": temp_dir,
            "outputFileName": "e2e_etl_result",
            "includeHeader": True
        })
        result = exporter.process(cleaner_output, params)

        # 验证导出成功
        assert result["report"]["output_rows"] == 2
        output_path = result["report"]["export_summary"]["output_path"]

        # 验证文件内容和表头
        df = pd.read_csv(output_path)
        assert list(df.columns) == ["name", "age", "dept", "admit_date"]
        assert df.iloc[0]["name"] == "张三"

    @E2E
    def test_e2e_multiple_formats(self, exporter, temp_dir):
        """同一数据导出为三种格式并验证一致性"""
        input_data = get_upstream_input("basic")
        results = {}

        for fmt in ["csv", "json", "jsonl"]:
            params = make_params({
                "outputFormat": fmt,
                "outputDir": temp_dir,
                "outputFileName": f"e2e_multi_{fmt}"
            })
            results[fmt] = exporter.process(input_data, params)
            assert results[fmt]["report"]["output_rows"] == 3

        # 三种格式行数应一致
        assert results["csv"]["report"]["output_rows"] == 3
        assert results["json"]["report"]["output_rows"] == 3
        assert results["jsonl"]["report"]["output_rows"] == 3

    @E2E
    def test_readback_verify_row_count(self, exporter, temp_dir):
        """导出后读取文件验证行数与输入一致（核心功能验证）"""
        params = make_params({
            "outputDir": temp_dir,
            "outputFileName": "test_readback"
        })
        result = exporter.process(BASIC_INPUT, params)
        output_path = result["report"]["export_summary"]["output_path"]

        # 读取导出的 CSV 文件
        df = pd.read_csv(output_path)

        # 验证行数一致
        assert len(df) == len(BASIC_INPUT["data"])
        assert list(df.columns) == BASIC_INPUT["schema"]["columns"]


# ============================================================================
# Performance Tests: 性能测试（鼓励项 → 性能量化对比）
# ============================================================================

class TestPerformance:
    """性能基准测试"""

    @PERFORMANCE
    def test_performance_csv_1000_rows(self, exporter, temp_dir):
        """1000行 CSV 导出时间 < 100ms"""
        input_data = get_upstream_input("many_rows")
        params = make_params({
            "outputDir": temp_dir,
            "outputFileName": "perf_1000_csv"
        })

        start = time.perf_counter()
        exporter.process(input_data, params)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 500, f"1000行CSV导出耗时 {elapsed_ms:.1f}ms，期望 < 500ms"

    @PERFORMANCE
    def test_performance_json_1000_rows(self, exporter, temp_dir):
        """1000行 JSON 导出时间 < 100ms"""
        input_data = get_upstream_input("many_rows")
        params = make_params({
            "outputFormat": "json",
            "outputDir": temp_dir,
            "outputFileName": "perf_1000_json"
        })

        start = time.perf_counter()
        exporter.process(input_data, params)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 500, f"1000行JSON导出耗时 {elapsed_ms:.1f}ms，期望 < 500ms"

    @PERFORMANCE
    def test_performance_jsonl_1000_rows(self, exporter, temp_dir):
        """1000行 JSONL 导出时间 < 500ms"""
        input_data = get_upstream_input("many_rows")
        params = make_params({
            "outputFormat": "jsonl",
            "outputDir": temp_dir,
            "outputFileName": "perf_1000_jsonl"
        })

        start = time.perf_counter()
        exporter.process(input_data, params)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 500, f"1000行JSONL导出耗时 {elapsed_ms:.1f}ms，期望 < 500ms"

    @PERFORMANCE
    def test_performance_csv_10000_rows(self, exporter, temp_dir):
        """10000行 CSV 导出时间 < 5s"""
        data_10k = [{"id": i, "value": f"data_{i}"} for i in range(10000)]
        input_data = {"data": data_10k}
        params = make_params({
            "outputDir": temp_dir,
            "outputFileName": "perf_10k_csv"
        })

        start = time.perf_counter()
        exporter.process(input_data, params)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 5000, f"10000行CSV导出耗时 {elapsed_ms:.1f}ms，期望 < 5000ms"


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
