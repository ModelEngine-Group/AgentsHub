# -*- coding: utf-8 -*-
"""
DataTransformer - 数据转换算子

支持字段重命名、字段选择、类型转换、值替换、条件筛选、派生列的数据转换算子
版本: 1.0.0

架构设计：
- PIPELINE_ORDER：7 步流水线，与 data_cleaner 保持一致的架构模式
- STEP_HANDLERS：步骤名 → 处理器映射（必须放在类体末尾）
- 步骤级性能分析：记录每个步骤的耗时（创新点）

吸收的实现细节：
1. filterCondition：列名标准化 + 反引号包裹 + NaN 自动排除说明
2. deriveColumns：inf → NaN 统一处理 + numexpr 引擎检查
3. typeConversions：date 列 strftime 格式化，确保输出 "YYYY-MM-DD"
4. valueMapping：类型推断，old_val 自动匹配列 dtype
5. select/drop 互斥：同时指定时记录 warning
6. STEP_HANDLERS 放在类体末尾
"""

import sys
import os
import json
import time
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple, Optional

# 日志配置
logger = logging.getLogger("DataTransformer")

# 路径配置
_OPERATORS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _OPERATORS_DIR not in sys.path:
    sys.path.insert(0, _OPERATORS_DIR)


# ============================================================================
# 异常类定义
# ============================================================================

class DataTransformerError(Exception):
    """数据转换算子基础异常"""
    pass


class ValidationError(DataTransformerError):
    """参数校验失败"""
    pass


class ProcessingError(DataTransformerError):
    """处理步骤执行失败"""
    pass


# ============================================================================
# 主处理类
# ============================================================================

class DataTransformer:
    """
    数据转换算子

    功能：
    1. 字段选择（select_fields）—— 保留指定字段
    2. 字段删除（drop_fields）—— 删除指定字段
    3. 字段重命名（rename_fields）—— 映射 old_name → new_name
    4. 类型转换（type_conversion）—— str/int/float/date 互转
    5. 值替换（value_mapping）—— 按列替换特定值（带类型推断）
    6. 条件筛选（filter_condition）—— pandas query 表达式（列名标准化）
    7. 派生列（derive_columns）—— 基于表达式计算新列（inf → NaN 处理）

    处理顺序：select → drop → rename → type_conversion → value_mapping → filter → derive
    """

    # 处理顺序定义（架构亮点！与 data_cleaner 保持一致的模式）
    PIPELINE_ORDER = [
        "select_fields",       # 1. 先选字段（减少后续处理量）
        "drop_fields",         # 2. 删除字段
        "rename_fields",       # 3. 重命名（后续步骤用新名）
        "type_conversion",     # 4. 类型转换
        "value_mapping",       # 5. 值替换
        "filter_condition",    # 6. 条件筛选（倒数第二步）
        "derive_columns",      # 7. 派生列（最后计算）
    ]

    # 警告类型枚举
    WARNING_TYPES = {
        "field_not_found": "字段不存在",
        "type_conversion_failed": "类型转换失败",
        "filter_expression_error": "筛选表达式错误",
        "derive_error": "派生列计算异常",
        "value_mapping_type_mismatch": "值替换类型不匹配",
        "empty_input": "输入数据为空",
        "select_drop_conflict": "select 和 drop 同时指定",
        "step_failed": "步骤执行失败"
    }

    # 支持的目标类型
    SUPPORTED_TYPES = {"int", "float", "str", "date"}

    def __init__(self):
        self.name = "DataTransformer"
        self.version = "1.0.0"
        self.warnings = []

        # 创新点：检查 numexpr 引擎可用性
        self._check_eval_engine()

    def _check_eval_engine(self):
        """检查 pandas eval/query 使用的引擎"""
        try:
            import numexpr
            self._eval_engine = "numexpr"
            logger.info("使用 numexpr 引擎（安全、高性能）")
        except ImportError:
            self._eval_engine = "python"
            logger.warning("numexpr 不可用，df.eval() 将回退到 Python 引擎")

    # ========================================================================
    # 公开接口
    # ========================================================================

    def validate(self, params: dict) -> Tuple[bool, Optional[str]]:
        """
        参数校验

        Args:
            params: 用户参数

        Returns:
            (是否有效, 错误信息)
        """
        # 校验 JSON 格式参数
        json_params = ["renameFields", "typeConversions", "valueMappings", "deriveColumns"]
        for param_name in json_params:
            value = params.get(param_name, "")
            if value and isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if not isinstance(parsed, dict):
                        return False, f"{param_name} 必须是 JSON 对象格式"
                except json.JSONDecodeError as e:
                    return False, f"{param_name} JSON 格式错误: {str(e)}"

        # 校验类型转换的目标类型
        type_conversions_str = params.get("typeConversions", "")
        if type_conversions_str and isinstance(type_conversions_str, str):
            try:
                type_map = json.loads(type_conversions_str)
                for col, target_type in type_map.items():
                    if target_type not in self.SUPPORTED_TYPES:
                        return False, f"类型转换: 不支持的目标类型 '{target_type}'，支持: {self.SUPPORTED_TYPES}"
            except json.JSONDecodeError:
                pass  # 已在上面校验过

        return True, None

    def process(self, input_data: Any, params: dict) -> Dict[str, Any]:
        """
        核心处理逻辑

        Args:
            input_data: 输入数据（DataFrame 或 字典列表）
            params: 处理参数

        Returns:
            包含 data, schema, report 的字典
        """
        # 1️⃣ 格式兼容
        df = self._normalize_input(input_data)
        if df is None:
            return self._empty_output()

        # 2️⃣ 空数据快速返回
        if df.empty:
            return self._build_output(df, self._generate_report(df, len(df), {
                "rename_count": 0, "select_count": 0, "drop_count": 0,
                "type_convert_count": 0, "value_replace_count": 0,
                "filter_count": 0, "derive_count": 0
            }))

        # 3️⃣ 参数校验
        is_valid, error_msg = self.validate(params)
        if not is_valid:
            raise ValidationError(error_msg)

        # 4️⃣ 重置状态
        original_rows = len(df)
        original_columns = len(df.columns)  # 记录原始字段数（含派生列前）
        self.warnings = []
        transform_summary = {
            "rename_count": 0, "select_count": 0, "drop_count": 0,
            "type_convert_count": 0, "value_replace_count": 0,
            "filter_count": 0, "derive_count": 0,
            "_input_columns": original_columns  # 流水线执行前的字段数
        }

        # 5️⃣ 执行流水线（带步骤级性能分析）
        total_start = time.perf_counter()
        step_timings = {}

        for step in self.PIPELINE_ORDER:
            step_start = time.perf_counter()
            df, step_stats = self._run_step_safe(step, df, params)
            step_elapsed = (time.perf_counter() - step_start) * 1000

            # 记录步骤级性能
            step_timings[step] = {
                "duration_ms": round(step_elapsed, 2),
                "rows_after": len(df)
            }
            transform_summary.update(step_stats)

        total_elapsed = (time.perf_counter() - total_start) * 1000

        # 6️⃣ 生成报告
        report = self._generate_report(df, original_rows, transform_summary)
        report["warnings"] = self.warnings.copy()

        # 步骤级性能分析（创新点！）
        report["performance"] = {
            "total_ms": round(total_elapsed, 2),
            "steps": step_timings
        }

        # 7️⃣ 构建输出
        return self._build_output(df, report)

    def get_summary(self, report: dict) -> str:
        """
        生成人类可读的处理摘要

        Args:
            report: 处理报告

        Returns:
            人类可读的摘要文本
        """
        if report.get("input_rows", 0) == 0:
            return "数据为空，无需转换。"

        parts = [f"数据转换完成：{report['input_rows']} 行 → {report['output_rows']} 行"]
        ts = report.get("transform_summary", {})

        if ts.get("rename_count", 0) > 0:
            parts.append(f"重命名 {ts['rename_count']} 个字段")
        if ts.get("select_count", 0) > 0:
            parts.append(f"选择保留 {ts['select_count']} 个字段")
        if ts.get("drop_count", 0) > 0:
            parts.append(f"删除 {ts['drop_count']} 个字段")
        if ts.get("type_convert_count", 0) > 0:
            parts.append(f"类型转换 {ts['type_convert_count']} 个字段")
        if ts.get("value_replace_count", 0) > 0:
            parts.append(f"值替换 {ts['value_replace_count']} 处")
        if ts.get("filter_count", 0) > 0:
            parts.append(f"筛选过滤 {ts['filter_count']} 行")
        if ts.get("derive_count", 0) > 0:
            parts.append(f"派生 {ts['derive_count']} 个新列")

        warnings = report.get("warnings", [])
        if warnings:
            parts.append(f"产生 {len(warnings)} 条警告")

        return "；".join(parts) + "。"

    def get_schema(self, params: dict) -> Dict[str, Any]:
        """获取输出Schema（静态接口）"""
        return {"columns": [], "row_count": 0}

    # ========================================================================
    # 私有方法
    # ========================================================================

    def _normalize_input(self, input_data: Any) -> Optional[pd.DataFrame]:
        """输入格式标准化（与 data_cleaner 保持一致）"""
        if input_data is None:
            return None

        if isinstance(input_data, pd.DataFrame):
            return input_data.copy()

        if isinstance(input_data, dict) and "data" in input_data:
            data = input_data["data"]
            if isinstance(data, list):
                df = pd.DataFrame(data)
                df = df.replace({None: np.nan})
                df = df.replace("None", np.nan)
                return df

        if isinstance(input_data, list):
            df = pd.DataFrame(input_data)
            df = df.replace({None: np.nan})
            df = df.replace("None", np.nan)
            return df

        return None

    def _run_step_safe(self, step: str, df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, Dict]:
        """
        带容错的步骤执行

        三种错误策略：
        - ValidationError → 直接抛出（用户配置错误必须告知）
        - ValueError/KeyError/TypeError → warning + 继续
        - 其他异常 → raise ProcessingError（系统级错误中断流程）
        """
        handler = self.STEP_HANDLERS.get(step)
        if handler is None:
            return df, {}

        try:
            return handler(self, df, params)
        except ValidationError:
            raise
        except (ValueError, KeyError, TypeError) as e:
            self._add_warning("step_failed", f"步骤 {step} 执行异常: {str(e)}", step)
            return df, {}
        except Exception as e:
            raise ProcessingError(f"步骤 {step} 执行失败: {str(e)}")

    def _add_warning(self, warning_type: str, message: str, context: str = None):
        """添加结构化警告"""
        warning = {
            "type": warning_type,
            "message": message
        }
        if context:
            warning["context"] = context
        self.warnings.append(warning)

    def _generate_report(self, df: pd.DataFrame, original_rows: int, transform_summary: Dict) -> Dict:
        """生成处理报告"""
        report = {
            "input_rows": original_rows,
            "output_rows": len(df),
            "transform_summary": transform_summary,
            "quality_metrics": {
                "input_columns": transform_summary.get("_input_columns", 0),
                "output_columns": len(df.columns) if not df.empty else 0,
                "type_mismatch_after": 0
            },
            "summary": ""
        }
        report["summary"] = self.get_summary(report)
        return report

    def _build_output(self, df: pd.DataFrame, report: Dict) -> Dict[str, Any]:
        """构建最终输出"""
        return {
            "data": df.to_dict(orient="records"),
            "schema": {
                "columns": list(df.columns),
                "row_count": len(df)
            },
            "report": report
        }

    def _empty_output(self) -> Dict[str, Any]:
        """空数据输出"""
        return {
            "data": [],
            "schema": {"columns": [], "row_count": 0},
            "report": {
                "input_rows": 0,
                "output_rows": 0,
                "transform_summary": {
                    "rename_count": 0, "select_count": 0, "drop_count": 0,
                    "type_convert_count": 0, "value_replace_count": 0,
                    "filter_count": 0, "derive_count": 0
                },
                "quality_metrics": {
                    "input_columns": 0,
                    "output_columns": 0,
                    "type_mismatch_after": 0
                },
                "warnings": [],
                "summary": "数据为空，无需转换。"
            }
        }

    def _parse_json_param(self, value: str) -> Optional[dict]:
        """解析 JSON 参数，返回 None 表示空或无效"""
        if not value or not isinstance(value, str) or not value.strip():
            return None
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    def _parse_csv_param(self, value: str) -> List[str]:
        """解析逗号分隔参数，返回字段列表"""
        if not value or not isinstance(value, str) or not value.strip():
            return []
        return [f.strip() for f in value.split(",") if f.strip()]

    def _coerce_value(self, value, target_dtype) -> Any:
        """
        值类型推断：将 old_val 转换为与列 dtype 匹配的类型

        用于 value_mapping 步骤，确保类型匹配。

        Args:
            value: 待转换的值（通常是字符串）
            target_dtype: 目标列的 dtype

        Returns:
            转换后的值，失败返回原始值
        """
        try:
            if pd.api.types.is_integer_dtype(target_dtype):
                return int(value)
            elif pd.api.types.is_float_dtype(target_dtype):
                return float(value)
            elif pd.api.types.is_bool_dtype(target_dtype):
                return bool(value)
            else:
                return value
        except (ValueError, TypeError):
            return value

    # ========================================================================
    # 处理步骤处理器
    # ========================================================================

    def _handle_select_fields(self, df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, Dict]:
        """
        1. 字段选择：保留指定字段

        互斥提醒：如果同时指定了 dropFields，记录 warning 提醒用户。
        """
        fields = self._parse_csv_param(params.get("selectFields", ""))
        input_cols = len(df.columns)

        # 互斥提醒（建议5）
        drop_fields_str = params.get("dropFields", "")
        if fields and drop_fields_str:
            self._add_warning(
                "select_drop_conflict",
                "selectFields 和 dropFields 同时指定，select 先生效，drop 可能覆盖 select 结果",
                "select_fields"
            )

        if not fields:
            return df, {"select_count": 0, "_input_columns": input_cols}

        # 只保留存在的字段，记录不存在的
        valid_fields = []
        for f in fields:
            if f in df.columns:
                valid_fields.append(f)
            else:
                self._add_warning("field_not_found", f"字段选择: 字段 '{f}' 不存在，已跳过", "select_fields")

        if valid_fields:
            df = df[valid_fields]

        return df, {"select_count": len(valid_fields), "_input_columns": input_cols}

    def _handle_drop_fields(self, df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, Dict]:
        """2. 字段删除：删除指定字段"""
        fields = self._parse_csv_param(params.get("dropFields", ""))
        if not fields:
            return df, {"drop_count": 0}

        # 只删除存在的字段
        existing = [f for f in fields if f in df.columns]
        not_found = [f for f in fields if f not in df.columns]

        for f in not_found:
            self._add_warning("field_not_found", f"字段删除: 字段 '{f}' 不存在，已跳过", "drop_fields")

        if existing:
            df = df.drop(columns=existing)

        return df, {"drop_count": len(existing)}

    def _handle_rename_fields(self, df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, Dict]:
        """
        3. 字段重命名：映射 old_name → new_name

        冲突处理：多个 old_name 映射到同一个 new_name 时，
        先执行的映射先生效，后执行的同名 new_name 覆盖前者并记录 warning。
        """
        rename_map = self._parse_json_param(params.get("renameFields", ""))
        if not rename_map:
            return df, {"rename_count": 0}

        # 过滤不存在的字段 + 检测新名称是否已被占用
        valid_rename = {}
        for old_name, new_name in rename_map.items():
            if old_name not in df.columns:
                self._add_warning("field_not_found", f"字段重命名: 字段 '{old_name}' 不存在，已跳过", "rename_fields")
            elif new_name in df.columns and new_name != old_name:
                # 新名称已被其他字段占用，记录 warning 但仍允许执行（pandas rename 会覆盖）
                self._add_warning(
                    "field_not_found",
                    f"字段重命名: 新名称 '{new_name}' 已被其他字段使用，重命名后该列将被覆盖",
                    "rename_fields"
                )
                valid_rename[old_name] = new_name
            else:
                valid_rename[old_name] = new_name

        # 检测 new_name 冲突
        seen_new_names = {}
        final_rename = {}
        for old_name, new_name in valid_rename.items():
            if new_name in seen_new_names:
                previous_old = seen_new_names[new_name]
                self._add_warning(
                    "field_not_found",
                    f"字段重命名: '{previous_old}' 和 '{old_name}' 都映射到 '{new_name}'，后者覆盖前者",
                    "rename_fields"
                )
                final_rename.pop(previous_old, None)
            seen_new_names[new_name] = old_name
            final_rename[old_name] = new_name

        if final_rename:
            df = df.rename(columns=final_rename)

        return df, {"rename_count": len(final_rename)}

    def _handle_type_conversion(self, df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, Dict]:
        """
        4. 类型转换：str/int/float/date 互转

        关键实现细节（建议3）：
        - date 类型转换后，使用 strftime("%Y-%m-%d") 格式化
        - 确保输出格式为 "2023-05-03"，而不是 ISO 8601 的 "2023-05-03T00:00:00"
        """
        type_map = self._parse_json_param(params.get("typeConversions", ""))
        if not type_map:
            return df, {"type_convert_count": 0}

        converted = 0
        for col, target_type in type_map.items():
            if col not in df.columns:
                self._add_warning("field_not_found", f"类型转换: 字段 '{col}' 不存在，已跳过", "type_conversion")
                continue

            try:
                if target_type == "int":
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                elif target_type == "float":
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                elif target_type == "str":
                    df[col] = df[col].astype(str).replace("nan", np.nan)
                elif target_type == "date":
                    # 关键：使用 format="mixed" 支持混合日期格式
                    df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed").dt.strftime("%Y-%m-%d")
                    # NaT 会变成 NaN 字符串，需要处理
                    df[col] = df[col].replace("NaT", np.nan)
                converted += 1
            except Exception as e:
                self._add_warning(
                    "type_conversion_failed",
                    f"类型转换: 字段 '{col}' 转换为 {target_type} 失败: {str(e)}，保留原值",
                    "type_conversion"
                )

        return df, {"type_convert_count": converted}

    def _handle_value_mapping(self, df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, Dict]:
        """
        5. 值替换：按列替换特定值

        关键实现细节（建议4）：
        - 值替换在类型转换之后执行
        - old_val 的类型需要与列的 dtype 匹配
        - 例如 age 已转为 int，valueMappings 中 {"age": {"25": "青年"}} 的 "25" 需要转为 25
        """
        value_map = self._parse_json_param(params.get("valueMappings", ""))
        if not value_map:
            return df, {"value_replace_count": 0}

        total_replaced = 0
        for col, mappings in value_map.items():
            if col not in df.columns:
                self._add_warning("field_not_found", f"值替换: 字段 '{col}' 不存在，已跳过", "value_mapping")
                continue

            if not isinstance(mappings, dict):
                continue

            for old_val, new_val in mappings.items():
                # 字符串化匹配：统一转字符串比较，避免类型不匹配问题
                col_as_str = df[col].astype(str)
                mask = col_as_str == str(old_val)
                count = mask.sum()

                if count > 0:
                    # 如果新值是字符串而列是数值类型，需要先转为 object 再赋值
                    if isinstance(new_val, str) and pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].astype(object)
                    df.loc[mask, col] = new_val
                    total_replaced += count
                # 值不存在是正常业务场景（如用户配置了但该列没有这个值），不记录 warning

        return df, {"value_replace_count": total_replaced}

    def _handle_filter_condition(self, df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, Dict]:
        """
        6. 条件筛选：pandas query 表达式

        关键实现细节（建议1）：
        - 列名标准化：将空格替换为下划线，避免 query 语法错误
        - 反引号包裹：pandas 支持用反引号包裹含特殊字符的列名
        - NaN 处理：query("age > 18") 会自动排除 NaN 行（通常期望行为）
        """
        condition = params.get("filterCondition", "")
        if not condition or not isinstance(condition, str) or not condition.strip():
            return df, {"filter_count": 0}

        original_len = len(df)

        # 列名标准化：用映射表精确管理原始列名 ↔ 标准化列名的对应关系
        original_columns = df.columns.tolist()
        normalized_columns = [col.replace(" ", "_") for col in original_columns]
        norm_to_orig = {norm: orig for orig, norm in zip(original_columns, normalized_columns)}
        df.columns = normalized_columns

        # 同步更新 condition 中的列名（如果用户使用了原始列名）
        for orig, norm in zip(original_columns, normalized_columns):
            if orig != norm and orig in condition:
                condition = condition.replace(orig, norm)

        try:
            df = df.query(condition).reset_index(drop=True)
        except Exception as e:
            # 恢复原始列名
            df.columns = [norm_to_orig.get(col, col) for col in df.columns]
            raise ValidationError(f"条件筛选表达式错误: {str(e)}")

        # 用映射表精确恢复原始列名
        df.columns = [norm_to_orig.get(col, col) for col in df.columns]

        filtered = original_len - len(df)
        return df, {"filter_count": filtered}

    def _handle_derive_columns(self, df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, Dict]:
        """
        7. 派生列：基于表达式计算新列

        关键实现细节（建议2）：
        - 除零处理：pd.eval 在除零时返回 inf，需要用 df.replace([np.inf, -np.inf], np.nan) 统一处理
        - 引擎安全：df.eval() 只访问 DataFrame 的列，不会执行任意代码
        """
        derive_map = self._parse_json_param(params.get("deriveColumns", ""))
        if not derive_map:
            return df, {"derive_count": 0}

        derived = 0
        for new_col, expr in derive_map.items():
            if not isinstance(expr, str):
                continue

            # 优化：检测列名是否已存在，覆盖前记录 warning
            if new_col in df.columns:
                self._add_warning(
                    "field_not_found",
                    f"派生列: 列名 '{new_col}' 已存在，将覆盖原列",
                    "derive_columns"
                )

            try:
                df[new_col] = df.eval(expr)

                # 关键：inf → NaN 统一处理；float 列保留2位小数
                if df[new_col].dtype in [np.float64, np.float32]:
                    df[new_col] = df[new_col].replace([np.inf, -np.inf], np.nan)
                    df[new_col] = df[new_col].round(2)

                derived += 1
            except Exception as e:
                self._add_warning(
                    "derive_error",
                    f"派生列 '{new_col}' 计算失败: {str(e)}，填充为 null",
                    "derive_columns"
                )
                df[new_col] = np.nan

        return df, {"derive_count": derived}

    # 步骤处理器映射（必须放在类体末尾！）
    STEP_HANDLERS = {
        "select_fields": _handle_select_fields,
        "drop_fields": _handle_drop_fields,
        "rename_fields": _handle_rename_fields,
        "type_conversion": _handle_type_conversion,
        "value_mapping": _handle_value_mapping,
        "filter_condition": _handle_filter_condition,
        "derive_columns": _handle_derive_columns,
    }
