# -*- coding: utf-8 -*-
"""
DataExporter - 数据导出算子

支持 CSV/JSON/JSONL 格式导出的数据导出算子，完成 ETL 流水线的 Load 阶段
版本: 1.0.0

架构：
- Layer 4（公开接口）：validate / process / get_summary / get_schema
- Layer 3（流水线引擎）：PIPELINE_ORDER → STEP_HANDLERS → _run_step_safe
- Layer 2（格式导出器）：_export_csv / _export_json / _export_jsonl
- Layer 1（基础设施）：异常类 / 路径映射 / 警告系统 / 输入标准化 / 输出构建

与 data_cleaner / data_transformer 保持一致的架构模式
"""

import sys
import os
import json
import time
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime

# 日志配置
logger = logging.getLogger("DataExporter")

# 路径配置（容器中自动跳过，不影响部署）
try:
    _OP_DIR = os.path.dirname(os.path.abspath(__file__))
    if _OP_DIR not in sys.path:
        sys.path.insert(0, _OP_DIR)
except Exception:
    pass


# ============================================================================
# Layer 1: 异常类定义
# ============================================================================

class DataExporterError(Exception):
    """数据导出算子基础异常"""
    pass


class ValidationError(DataExporterError):
    """参数校验失败"""
    pass


class ProcessingError(DataExporterError):
    """处理步骤执行失败"""
    pass


# ============================================================================
# Layer 4: 主处理类
# ============================================================================

class DataExporter:
    """
    数据导出算子

    功能：
    1. 校验输入数据（validate_input）
    2. 准备导出数据（prepare_data）
    3. 写入文件（export_file）—— CSV / JSON / JSONL
    4. 验证输出文件（verify_output）

    处理顺序：validate_input → prepare_data → export_file → verify_output
    """

    # =========================================================================
    # 常量定义
    # =========================================================================

    PIPELINE_ORDER = [
        "validate_input",    # 1. 校验输入数据
        "prepare_data",      # 2. 准备导出数据
        "export_file",       # 3. 写入文件
        "verify_output",     # 4. 验证输出文件
    ]

    SUPPORTED_FORMATS = {"csv", "json", "jsonl"}

    FORMAT_EXTENSIONS = {
        "csv": ".csv",
        "json": ".json",
        "jsonl": ".jsonl",
    }

    WARNING_TYPES = {
        "empty_input": "输入数据为空，不创建导出文件",
        "output_dir_created": "输出目录不存在，已自动创建",
        "output_dir_not_writable": "输出目录不可写",
        "file_overwritten": "输出文件已存在，将覆盖",
        "file_not_overwritten": "输出文件已存在，文件名追加时间戳",
        "encoding_fallback": "指定编码不支持，回退到 UTF-8",
        "step_failed": "步骤执行失败",
    }

    # =========================================================================
    # Layer 4: 公开接口
    # =========================================================================

    def __init__(self):
        self.name = "DataExporter"
        self.version = "1.0.0"
        self.warnings = []

    def validate(self, params: dict) -> Tuple[bool, Optional[str]]:
        """
        参数校验

        Args:
            params: 用户参数

        Returns:
            (是否有效, 错误信息)
        """
        # outputDir 必填
        output_dir = params.get("outputDir", "")
        if not output_dir or not isinstance(output_dir, str) or not output_dir.strip():
            return False, "输出目录不能为空"

        # outputFormat 合法性
        output_format = params.get("outputFormat", "csv")
        if output_format not in self.SUPPORTED_FORMATS:
            return False, f"不支持的导出格式 '{output_format}'，支持: {self.SUPPORTED_FORMATS}"

        # encoding 合法性
        encoding = params.get("encoding", "utf-8")
        try:
            "测试".encode(encoding)
        except (LookupError, TypeError):
            return False, f"不支持的编码格式 '{encoding}'"

        return True, None

    def process(self, input_data: Any, params: dict) -> Dict[str, Any]:
        """
        核心处理逻辑

        流程：
        1. 输入标准化 → 2. 参数校验 → 3. 流水线执行 → 4. 报告生成

        Args:
            input_data: 输入数据（DataFrame / 字典列表 / 标准输出格式）
            params: 处理参数

        Returns:
            包含 data（透传）, schema（透传）, report 的字典
        """
        # 1. 输入标准化
        df = self._normalize_input(input_data)

        # 2. 空数据快速返回
        if df is None or df.empty:
            report = {
                "input_rows": 0,
                "output_rows": 0,
                "export_summary": {},
                "warnings": [],
                "summary": "输入数据为空，未导出文件。"
            }
            return self._build_passthrough_output(input_data, report)

        # 3. 参数校验
        is_valid, error_msg = self.validate(params)
        if not is_valid:
            raise ValidationError(error_msg)

        # 4. 重置状态
        self.warnings = []

        # 5. 初始化流水线状态（state 在各步骤间传递）
        pipeline_state = {
            "df": df,
            "params": params,
            "original_rows": len(df),
            "output_path": None,
            "file_size_bytes": 0,
        }

        # 6. 执行流水线
        for step in self.PIPELINE_ORDER:
            pipeline_state = self._run_step_safe(step, pipeline_state)

        # 7. 生成报告
        report = self._generate_report(
            original_rows=len(df),
            pipeline_state=pipeline_state
        )

        # 8. 构建透传输出
        return self._build_passthrough_output(input_data, report)

    def get_summary(self, report: dict) -> str:
        """
        生成人类可读的处理摘要

        Args:
            report: 处理报告

        Returns:
            摘要文本
        """
        if report.get("input_rows", 0) == 0:
            return "输入数据为空，未导出文件。"

        export = report.get("export_summary", {})
        fmt = export.get("format", "unknown")
        path = export.get("output_path", "unknown")
        size = export.get("file_size_bytes", 0)
        rows = report.get("output_rows", 0)

        size_str = self._format_file_size(size)
        return f"数据导出完成：{rows} 行，{fmt.upper()} 格式，输出到 {path}，文件大小 {size_str}。"

    def get_schema(self, params: dict) -> Dict[str, Any]:
        """获取输出 Schema（静态接口）"""
        return {"columns": [], "row_count": 0}

    # =========================================================================
    # Layer 1: 基础设施 - 输入标准化
    # =========================================================================

    def _normalize_input(self, input_data: Any) -> Optional[pd.DataFrame]:
        """
        输入格式标准化

        支持：
        - 标准输出格式：{"data": [...], "schema": {...}, "report": {...}}
        - 字典列表：[{"col1": val1}, ...]
        - DataFrame
        """
        if input_data is None:
            return None

        if isinstance(input_data, pd.DataFrame):
            return input_data.copy()

        if isinstance(input_data, dict) and "data" in input_data:
            data = input_data.get("data")
            if isinstance(data, list):
                return self._list_to_df(data)
            return None

        if isinstance(input_data, list):
            return self._list_to_df(input_data)

        return None

    def _list_to_df(self, data: list) -> Optional[pd.DataFrame]:
        """字典列表 → DataFrame（替换 NaN）"""
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df = df.replace({None: np.nan, "None": np.nan})
        return df

    # =========================================================================
    # Layer 1: 基础设施 - 路径与文件操作
    # =========================================================================

    def _validate_path(self, path: str) -> str:
        """
        Docker 路径映射（参考 data_loader）

        将容器的 /mnt/data/ 路径映射到宿主机的 D:/data/
        """
        if path.startswith("/mnt/data/"):
            return path.replace("/mnt/data/", "D:/data/")
        return path

    def _validate_output_dir(self, output_dir: str) -> Tuple[bool, Optional[str]]:
        """
        校验输出目录是否存在且可写

        策略：
        1. 目录不存在 → 自动创建 + 记录 warning
        2. 创建成功后 → 写入临时文件验证可写性
        3. 清理临时文件
        """
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                self._add_warning("output_dir_created",
                                  f"输出目录不存在，已自动创建: {output_dir}",
                                  "validate_input")
            test_file = os.path.join(output_dir, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            return True, None
        except (PermissionError, OSError) as e:
            return False, f"输出目录不可写: {str(e)}"

    def _get_output_path(self, params: dict) -> str:
        """
        计算输出文件的完整路径

        流程：
        1. 路径映射（Docker 兼容）
        2. 拼接文件名 + 扩展名
        3. 文件冲突处理（覆盖 / 追加时间戳）
        """
        output_dir = self._validate_path(params.get("outputDir", "").strip())
        output_name = params.get("outputFileName", "export_output").strip() or "export_output"
        output_format = params.get("outputFormat", "csv")
        extension = self.FORMAT_EXTENSIONS.get(output_format, ".csv")
        overwrite = params.get("overwrite", True)

        file_name = f"{output_name}{extension}"
        full_path = os.path.join(output_dir, file_name)

        # 文件冲突：不覆盖 → 追加时间戳
        if os.path.exists(full_path) and not overwrite:
            file_name = self._append_timestamp(output_name, extension)
            full_path = os.path.join(output_dir, file_name)

        return full_path

    def _append_timestamp(self, filename: str, extension: str) -> str:
        """
        文件名追加时间戳

        Example:
            cleaned_data.csv → cleaned_data_20260430_221500.csv
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{filename}_{timestamp}{extension}"

    def _get_file_size(self, path: str) -> int:
        """获取文件字节数，文件不存在返回 0"""
        try:
            return os.path.getsize(path)
        except (OSError, FileNotFoundError):
            return 0

    def _calculate_row_count(self, file_path: str, fmt: str, params: dict = None) -> int:
        """
        读取导出文件验证行数

        不同格式的计数方式：
        - CSV：行数 - 1（有表头时），行数（无表头时）
        - JSONL：行数 = 总行数
        - JSON：解析后 len(data)
        """
        try:
            if fmt == "csv":
                with open(file_path, "r", encoding="utf-8") as f:
                    total_lines = sum(1 for _ in f)
                    # 只有包含表头时才减 1
                    if params and params.get("includeHeader", True):
                        return max(0, total_lines - 1)
                    return max(0, total_lines)
            elif fmt == "jsonl":
                with open(file_path, "r", encoding="utf-8") as f:
                    return sum(1 for _ in f)
            elif fmt == "json":
                with open(file_path, "r", encoding="utf-8") as f:
                    return len(json.load(f))
        except Exception:
            pass
        return 0

    def _format_file_size(self, size_bytes: int) -> str:
        """格式化文件大小展示"""
        if size_bytes < 1024:
            return f"{size_bytes}B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f}KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f}MB"

    # =========================================================================
    # Layer 1: 基础设施 - 警告系统与输出构建
    # =========================================================================

    def _add_warning(self, warning_type: str, message: str, context: str = None):
        """添加结构化警告"""
        warning = {
            "type": warning_type,
            "message": message,
        }
        if context:
            warning["context"] = context
        self.warnings.append(warning)

    def _generate_report(self, original_rows: int, pipeline_state: dict) -> Dict:
        """生成处理报告"""
        export_summary = {
            "format": pipeline_state.get("output_format", "csv"),
            "output_path": pipeline_state.get("output_path", ""),
            "file_size_bytes": pipeline_state.get("file_size_bytes", 0),
            "encoding": pipeline_state.get("params", {}).get("encoding", "utf-8"),
            "header_included": pipeline_state.get("params", {}).get("includeHeader", True),
            "index_included": pipeline_state.get("params", {}).get("indexColumn", False),
            "export_time_ms": round(pipeline_state.get("export_time_ms", 0), 2),
        }

        report = {
            "input_rows": original_rows,
            "output_rows": original_rows,
            "export_summary": export_summary,
            "warnings": self.warnings.copy(),
            "summary": "",
        }
        report["summary"] = self.get_summary(report)
        return report

    def _build_passthrough_output(self, input_data: Any, report: Dict) -> Dict[str, Any]:
        """
        构建透传输出

        data_exporter 透传上游的 data 和 schema，只追加 report

        支持：
        - dict 格式：{"data": [...], "schema": {...}}
        - DataFrame 格式（传入的 input_data 是 df）
        - list 格式：[{"col1": val1}, ...]
        """
        data = []
        schema = {"columns": [], "row_count": 0}

        if isinstance(input_data, pd.DataFrame):
            data = input_data.to_dict(orient="records")
            schema = {"columns": list(input_data.columns), "row_count": len(input_data)}
        elif isinstance(input_data, dict):
            if "data" in input_data:
                data = input_data["data"]
                schema = input_data.get("schema", schema)
        elif isinstance(input_data, list):
            data = input_data
            if data:
                schema = {"columns": list(data[0].keys()), "row_count": len(data)}

        return {"data": data, "schema": schema, "report": report}

    # =========================================================================
    # Layer 3: 流水线引擎
    # =========================================================================

    def _run_step_safe(self, step: str, state: dict) -> dict:
        """
        带容错的步骤执行

        三种错误策略：
        - ValidationError → 直接抛出（用户配置错误必须告知）
        - ValueError/KeyError/TypeError → warning + 继续
        - 其他异常 → raise ProcessingError（系统级错误中断流程）
        """
        handler = self.STEP_HANDLERS.get(step)
        if handler is None:
            return state

        try:
            return handler(self, state)
        except ValidationError:
            raise
        except (ValueError, KeyError, TypeError) as e:
            self._add_warning("step_failed", f"步骤 {step} 执行异常: {str(e)}", step)
            return state
        except Exception as e:
            raise ProcessingError(f"步骤 {step} 执行失败: {str(e)}")

    # =========================================================================
    # Layer 3: 流水线步骤
    # =========================================================================

    def _handle_validate_input(self, state: dict) -> dict:
        """
        步骤 1：校验输入数据

        检查：
        - outputDir 不为空（二次校验）
        - outputFormat 合法
        - 目录可写
        """
        df = state["df"]
        params = state["params"]

        # 记录输出格式到 state
        output_format = params.get("outputFormat", "csv")
        state["output_format"] = output_format

        # outputDir 二次校验
        output_dir = params.get("outputDir", "")
        if not output_dir or not output_dir.strip():
            raise ValidationError("输出目录不能为空")

        # outputFormat 校验
        if output_format not in self.SUPPORTED_FORMATS:
            raise ValidationError(f"不支持的导出格式 '{output_format}'")

        # 目录可写性校验
        output_dir = self._validate_path(output_dir.strip())
        is_writable, error_msg = self._validate_output_dir(output_dir)
        if not is_writable:
            raise ProcessingError(error_msg)

        return state

    def _handle_prepare_data(self, state: dict) -> dict:
        """
        步骤 2：准备导出数据

        计算输出路径并处理文件冲突
        """
        params = state["params"]

        # 计算输出路径（含覆盖处理）
        state["output_path"] = self._get_output_path(params)

        return state

    def _handle_export_file(self, state: dict) -> dict:
        """
        步骤 3：写入文件

        根据 outputFormat 选择对应的导出方式：
        - CSV → pd.DataFrame.to_csv()
        - JSON → json.dumps(ensure_ascii=False)
        - JSONL → 每行一个 JSON 对象
        """
        df = state["df"]
        params = state["params"]
        output_path = state["output_path"]
        output_format = state.get("output_format", "csv")
        encoding = params.get("encoding", "utf-8")

        # 记录覆盖警告
        if os.path.exists(output_path):
            if params.get("overwrite", True):
                self._add_warning("file_overwritten", f"输出文件已存在，将覆盖: {output_path}")
            else:
                # 理论上 _get_output_path 已经处理了不覆盖的情况，但为安全再检查
                self._add_warning("file_not_overwritten",
                                  f"输出文件已存在，文件名已追加时间戳: {os.path.basename(output_path)}")

        # 记录开始时间
        self._export_start_time = time.perf_counter()

        try:
            if output_format == "csv":
                self._export_csv(df, output_path, params, encoding)
            elif output_format == "json":
                self._export_json(df, output_path, encoding)
            elif output_format == "jsonl":
                self._export_jsonl(df, output_path, encoding)
            else:
                raise ValidationError(f"不支持的导出格式: {output_format}")
        except ValidationError:
            raise
        except Exception as e:
            raise ProcessingError(f"文件写入失败: {str(e)}")

        # 计算耗时
        state["export_time_ms"] = (time.perf_counter() - self._export_start_time) * 1000

        # 记录文件信息
        state["file_size_bytes"] = self._get_file_size(output_path)

        return state

    def _handle_verify_output(self, state: dict) -> dict:
        """
        步骤 4：验证输出文件

        检查：
        1. 文件存在
        2. 文件大小 > 0
        3. 行数一致性（可选，仅当有输出路径时）
        """
        output_path = state.get("output_path")
        file_size = state.get("file_size_bytes", 0)
        output_format = state.get("output_format", "csv")
        original_rows = state.get("original_rows", 0)

        # 检查文件存在
        if not output_path or not os.path.exists(output_path):
            raise ProcessingError(f"导出文件不存在: {output_path}")

        # 检查文件大小
        if file_size == 0:
            raise ProcessingError(f"导出文件大小为 0: {output_path}")

        # 验证行数一致性
        params = state.get("params", {})
        actual_rows = self._calculate_row_count(output_path, output_format, params)
        if actual_rows > 0 and actual_rows != original_rows:
            self._add_warning(
                "step_failed",
                f"行数不一致：期望 {original_rows} 行，实际 {actual_rows} 行",
                "verify_output"
            )

        return state

    # =========================================================================
    # Layer 2: 格式导出器
    # =========================================================================

    def _export_csv(self, df: pd.DataFrame, output_path: str, params: dict, encoding: str):
        """CSV 格式导出"""
        include_header = params.get("includeHeader", True)
        include_index = params.get("indexColumn", False)

        df.to_csv(
            output_path,
            encoding=encoding,
            index=include_index,
            header=include_header,
        )

    def _export_json(self, df: pd.DataFrame, output_path: str, encoding: str):
        """JSON 数组格式导出"""
        data = df.to_dict(orient="records")
        data = self._clean_nan_for_json(data)

        with open(output_path, "w", encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def _export_jsonl(self, df: pd.DataFrame, output_path: str, encoding: str):
        """
        JSONL 格式导出

        ⚠️ JSONL 不是 JSON 数组！每行必须是独立的 JSON 对象，用换行分隔。
        ✅ {record1}\n{record2}\n{record3}
        ❌ [{record1}, {record2}, {record3}]
        """
        data = df.to_dict(orient="records")
        data = self._clean_nan_for_json(data)

        with open(output_path, "w", encoding=encoding) as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _clean_nan_for_json(self, data: list) -> list:
        """
        清理 JSON 中的非法值

        NaN / Inf 在 JSON 中是非法的，需要转换为 None（序列化为 null）
        同时处理 numpy 数值类型（如 Int64、float64）→ Python 原生类型
        """
        cleaned = []
        for record in data:
            cleaned_record = {}
            for key, value in record.items():
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    cleaned_record[key] = None
                elif pd.isna(value):
                    cleaned_record[key] = None
                elif isinstance(value, (np.integer,)):
                    cleaned_record[key] = int(value)
                elif isinstance(value, (np.floating,)):
                    cleaned_record[key] = float(value)
                else:
                    cleaned_record[key] = value
            cleaned.append(cleaned_record)
        return cleaned

    # =========================================================================
    # Layer 3: 步骤处理器映射（必须放在类体末尾！）
    # =========================================================================

    STEP_HANDLERS = {
        "validate_input": _handle_validate_input,
        "prepare_data": _handle_prepare_data,
        "export_file": _handle_export_file,
        "verify_output": _handle_verify_output,
    }
