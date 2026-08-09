# -*- coding: utf-8 -*-
"""
DataCleaner - 数据清洗算子

支持去重、空值处理、格式标准化、隐私脱敏的结构化数据清洗算子
版本: 1.0.0
"""

import sys
import os
import re
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple, Optional

# 路径配置
_OPERATORS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _OPERATORS_DIR not in sys.path:
    sys.path.insert(0, _OPERATORS_DIR)


# ============================================================================
# 异常类定义
# ============================================================================

class DataCleanerError(Exception):
    """数据清洗算子基础异常"""
    pass


class ValidationError(DataCleanerError):
    """参数校验失败"""
    pass


class ProcessingError(DataCleanerError):
    """处理步骤执行失败"""
    pass


# ============================================================================
# 隐私脱敏模块（独立类设计，支持扩展）
# ============================================================================

class PrivacyMasker:
    """
    隐私脱敏专家模块

    设计亮点：
    - 独立模块化，可单独测试
    - 支持规则引擎和模型两种方案
    - 正则匹配 + 替换，脱敏效果可预期
    """

    # 检测规则
    DETECTION_RULES = {
        "phone": {
            "pattern": r"1[3-9]\d{9}",
            "description": "手机号（11位，以1开头）"
        },
        "id_card": {
            # 注意：18位模式在前（因为15位是18位的前缀）
            "pattern": r"\d{17}[\dXx]|\d{15}",
            "description": "身份证号（15位或18位）"
        },
        "name": {
            "pattern": r"[\u4e00-\u9fa5]{2,4}(?:先生|女士|老师)?",
            "description": "中文姓名（2-4个汉字）"
        }
    }

    # 脱敏规则
    MASK_RULES = {
        "phone": lambda x: f"{x[:3]}****{x[-4:]}",
        # 身份证：前3后4，中间填充星号
        "id_card": lambda x: f"{x[:3]}{'*' * (len(x) - 7)}{x[-4:]}",
        "name": lambda x: None  # 动态生成
    }

    def __init__(self):
        self.name_counter = 1  # 姓名脱敏计数器
        self.stats = {"phone": 0, "id_card": 0, "name": 0}
        self.warnings = []

    def reset(self):
        """重置状态（每次处理前调用）"""
        self.name_counter = 1
        self.stats = {"phone": 0, "id_card": 0, "name": 0}
        self.warnings = []

    def detect_and_mask(self, value: Any, field: str) -> Tuple[Any, Optional[Dict]]:
        """
        检测并脱敏单个值

        Args:
            value: 待处理的值
            field: 字段名（用于日志）

        Returns:
            (脱敏后的值, 警告信息或None)
        """
        if not isinstance(value, str) or not value.strip():
            return value, None

        original = value
        warning = None

        try:
            # 先检查是否是纯数字字符串（可能是身份证）
            is_numeric = value.isdigit()

            # 如果是纯数字且长度 >= 15，优先检查身份证（避免误匹配）
            if is_numeric and len(value) >= 15:
                # 先检查身份证
                if re.search(self.DETECTION_RULES["id_card"]["pattern"], value):
                    masked = self.MASK_RULES["id_card"](value)
                    self.stats["id_card"] += 1
                    return masked, None
                # 15位以下才检查手机号
                if len(value) == 11 and re.match(r"^1[3-9]\d{9}$", value):
                    masked = self.MASK_RULES["phone"](value)
                    self.stats["phone"] += 1
                    return masked, None
            else:
                # 非纯数字字符串（可能有名字），按原有顺序检测
                # 检测手机号（只匹配纯手机号格式）
                if re.match(r"^1[3-9]\d{9}$", value):
                    masked = self.MASK_RULES["phone"](value)
                    self.stats["phone"] += 1
                    return masked, None

            # 检测姓名（最后检测，因为中文名不会与数字冲突）
            name_match = re.search(self.DETECTION_RULES["name"]["pattern"], value)
            if name_match:
                masked = f"患者{self.name_counter}"
                self.name_counter += 1
                self.stats["name"] += 1
                return masked, None

        except Exception as e:
            warning = {
                "type": "privacy_detection_failed",
                "field": field,
                "message": f"隐私检测失败: {str(e)}",
                "original_value": original[:20] if len(str(original)) > 20 else original
            }
            self.warnings.append(warning)

        return value, warning

    def get_stats(self) -> Dict[str, int]:
        """获取脱敏统计"""
        return self.stats.copy()

    def get_warnings(self) -> List[Dict]:
        """获取警告列表"""
        return self.warnings.copy()


# ============================================================================
# 主处理类
# ============================================================================

class DataCleaner:
    """
    数据清洗算子

    功能：
    1. 去重（remove_duplicates）
    2. 空值处理（handle_missing: drop/fill/keep）
    3. 去空格（trim_whitespace）
    4. 格式标准化（standardize_format）
    5. 隐私脱敏（privacy_check）

    处理顺序：trim_whitespace → remove_duplicates → handle_missing → privacy_check → standardize_format
    """

    # 处理顺序定义（架构亮点！）
    PIPELINE_ORDER = [
        "trim_whitespace",      # 1. 先去空格（清理原始数据）
        "remove_duplicates",    # 2. 去重（基于清理后数据）
        "handle_missing",       # 3. 空值处理
        "privacy_check",        # 4. 脱敏（最后处理）
        "standardize_format"    # 5. 格式标准化
    ]

    # 警告类型枚举
    WARNING_TYPES = {
        "privacy_detection_failed": "隐私检测异常",
        "format_mismatch": "格式不符",
        "missing_field": "字段不存在",
        "processing_skipped": "步骤被跳过",
        "step_failed": "步骤执行失败"
    }

    def __init__(self):
        self.name = "DataCleaner"
        self.version = "1.0.0"
        self.privacy_masker = PrivacyMasker()
        self.warnings = []

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
        # fill_value 检查：只有选择 fill 时才需要
        if params.get("handleMissing") == "fill" and not params.get("fillValue"):
            return False, "选择填充空值时，必须指定填充值"

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
        # 1️⃣ 格式兼容：DataFrame 或 字典列表 都支持
        df = self._normalize_input(input_data)
        if df is None:
            return self._empty_output()

        # 2️⃣ 参数校验：快速失败
        is_valid, error_msg = self.validate(params)
        if not is_valid:
            raise ValidationError(error_msg)

        # 3️⃣ 重置状态
        original_rows = len(df)
        self.warnings = []
        self.privacy_masker.reset()
        stats = self._init_stats(df)

        # 4️⃣ 执行流水线
        for step in self.PIPELINE_ORDER:
            step_key = self._to_camel_case(step)
            if params.get(step_key, self._get_default(step_key)):
                df, step_stats = self._run_step_safe(step, df, params)
                stats.update(step_stats)

        # 5️⃣ 生成报告
        report = self._generate_report(df, original_rows, stats)
        report["warnings"].extend(self.privacy_masker.get_warnings())

        # 6️⃣ 构建输出
        return self._build_output(df, report)

    def get_summary(self, report: dict) -> str:
        """
        生成人类可读的处理摘要（智能体友好）

        Args:
            report: 处理报告

        Returns:
            人类可读的摘要文本
        """
        # 空数据处理
        if report.get("input_rows", 0) == 0:
            return "数据为空，无需处理。"

        qm = report.get("quality_metrics", {})
        pm = report.get("privacy_masked", {})

        summary_parts = [
            f"数据清洗完成：原始数据 {report['input_rows']} 行 → 清洗后 {report['output_rows']} 行",
        ]

        if report.get("duplicates_removed", 0) > 0:
            summary_parts.append(f"删除重复行 {report['duplicates_removed']} 行")

        if report.get("missing_handled", 0) > 0:
            summary_parts.append(f"处理空值 {report['missing_handled']} 个")

        null_before = qm.get("null_rate_before", 0)
        null_after = qm.get("null_rate_after", 0)
        if null_before > 0:
            summary_parts.append(f"空值率从 {null_before:.1%} 降至 {null_after:.1%}")

        total_privacy = sum(pm.values())
        if total_privacy > 0:
            summary_parts.append(f"脱敏处理 {total_privacy} 处敏感信息")

        if report.get("warnings"):
            summary_parts.append(f"产生 {len(report['warnings'])} 条警告")

        return "；".join(summary_parts) + "。"

    def get_schema(self, params: dict) -> Dict[str, Any]:
        """获取输出Schema（静态接口）"""
        return {
            "columns": [],
            "row_count": 0
        }

    # ========================================================================
    # 私有方法
    # ========================================================================

    def _normalize_input(self, input_data: Any) -> Optional[pd.DataFrame]:
        """输入格式标准化"""
        if input_data is None:
            return None

        if isinstance(input_data, pd.DataFrame):
            return input_data.copy()

        if isinstance(input_data, dict) and "data" in input_data:
            data = input_data["data"]
            if isinstance(data, list):
                df = pd.DataFrame(data)
                # 特殊处理：Python None -> NaN
                df = df.replace({None: np.nan})
                # 特殊处理：字符串 "None" -> NaN（来自 JSON 序列化）
                df = df.replace("None", np.nan)
                return df

        if isinstance(input_data, list):
            df = pd.DataFrame(input_data)
            df = df.replace({None: np.nan})
            df = df.replace("None", np.nan)
            return df

        return None

    def _get_default(self, key: str) -> bool:
        """获取参数默认值"""
        defaults = {
            "trimWhitespace": True,
            "removeDuplicates": False,
            "handleMissing": "drop",
            "standardizeFormat": False,
            "privacyCheck": False
        }
        return defaults.get(key, False)

    def _to_camel_case(self, snake_str: str) -> str:
        """下划线转驼峰"""
        components = snake_str.split('_')
        return components[0] + ''.join(x.title() for x in components[1:])

    def _init_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """初始化统计信息"""
        return {
            "input_rows": len(df),
            "output_rows": len(df),
            "duplicates_removed": 0,
            "missing_handled": 0,
            "privacy_masked": {"phone": 0, "id_card": 0, "name": 0},
            "warnings": []
        }

    def _run_step_safe(self, step: str, df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, Dict]:
        """
        带容错的步骤执行

        设计思路：
        - 可恢复错误（ValueError, KeyError）：记录警告，继续执行
        - 不可恢复错误：抛出 ProcessingError
        """
        handler = self.STEP_HANDLERS.get(step)
        if handler is None:
            return df, {}

        try:
            return handler(self, df, params)
        except (ValueError, KeyError, TypeError) as e:
            # 可恢复错误：记录警告，继续
            self._add_warning("step_failed", f"步骤 {step} 执行异常: {str(e)}", step)
            return df, {}
        except Exception as e:
            # 不可恢复错误：中断流程
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

    def _generate_report(self, df: pd.DataFrame, original_rows: int, stats: Dict) -> Dict:
        """生成处理报告"""
        # 质量指标
        null_before = stats.get("null_rate_before", 0)
        null_after = self._calc_null_rate(df)

        report = {
            "input_rows": original_rows,
            "output_rows": len(df),
            "duplicates_removed": stats.get("duplicates_removed", 0),
            "missing_handled": stats.get("missing_handled", 0),
            "privacy_masked": self.privacy_masker.get_stats(),
            "quality_metrics": {
                "null_rate_before": null_before,
                "null_rate_after": null_after,
                "duplicate_rate_after": 0.0  # 去重后应始终为0
            },
            "warnings": self.warnings.copy()
        }

        # 添加处理摘要
        report["summary"] = self.get_summary(report)

        return report

    def _calc_null_rate(self, df: pd.DataFrame) -> float:
        """计算空值率"""
        if df.empty:
            return 0.0
        total_cells = df.shape[0] * df.shape[1]
        null_cells = df.isna().sum().sum()
        return null_cells / total_cells if total_cells > 0 else 0.0

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
                "duplicates_removed": 0,
                "missing_handled": 0,
                "privacy_masked": {"phone": 0, "id_card": 0, "name": 0},
                "quality_metrics": {
                    "null_rate_before": 0.0,
                    "null_rate_after": 0.0,
                    "duplicate_rate_after": 0.0
                },
                "warnings": [],
                "summary": "数据为空，无需处理。"
            }
        }

    # ========================================================================
    # 处理步骤处理器
    # ========================================================================

    def _handle_trim_whitespace(self, df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, Dict]:
        """1. 去除首尾空格"""
        df = df.copy()
        for col in df.columns:
            # 兼容 pandas 2.0+ 的 str 类型和旧版的 object 类型
            col_dtype = df[col].dtype
            is_string_type = col_dtype == object or col_dtype.name == 'str' or pd.api.types.is_string_dtype(col_dtype)
            if is_string_type:
                # 先处理字符串，去除首尾空格
                df[col] = df[col].astype(str).str.strip()
                # 将 "nan" 字符串转回 NaN（来自 pandas 的 None 处理）
                df[col] = df[col].replace("nan", np.nan)
                # 处理 None 字符串
                df[col] = df[col].replace("None", np.nan)
                # 处理空字符串
                df[col] = df[col].replace("", np.nan)
        return df, {}

    def _handle_remove_duplicates(self, df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, Dict]:
        """2. 删除重复行"""
        original_len = len(df)
        df = df.drop_duplicates(keep='first').reset_index(drop=True)
        removed = original_len - len(df)
        return df, {"duplicates_removed": removed}

    def _handle_missing(self, df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, Dict]:
        """3. 空值处理"""
        original_nulls = df.isna().sum().sum()
        null_rate_before = self._calc_null_rate(df)  # 处理前的空值率
        mode = params.get("handleMissing", "drop")

        if mode == "drop":
            df = df.dropna().reset_index(drop=True)
        elif mode == "fill":
            fill_value = params.get("fillValue", "")
            df = df.fillna(fill_value)
        elif mode == "keep":
            pass  # 保持原样

        handled = original_nulls - df.isna().sum().sum()
        return df, {"missing_handled": max(0, handled), "null_rate_before": null_rate_before}

    def _handle_privacy_check(self, df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, Dict]:
        """4. 隐私脱敏"""
        df = df.copy()
        privacy_fields = params.get("privacyFields", [])  # 指定字段脱敏

        for col in df.columns:
            # 如果指定了字段，只处理指定字段；否则处理所有文本字段
            if privacy_fields and col not in privacy_fields:
                continue

            col_dtype = df[col].dtype

            # 判断是否为字符串类型（兼容 pandas 2.0+）
            is_string_type = col_dtype == object or col_dtype.name == 'str' or pd.api.types.is_string_dtype(col_dtype)
            # 判断是否为数值类型（如 int64 的手机号/身份证）
            is_numeric_type = pd.api.types.is_numeric_dtype(col_dtype)

            if is_string_type:
                # 字符串类型：直接检测
                for idx, value in df[col].items():
                    if isinstance(value, str):
                        masked, warning = self.privacy_masker.detect_and_mask(value, col)
                        df.at[idx, col] = masked
            elif is_numeric_type and privacy_fields and col in privacy_fields:
                # 数值类型但被用户指定为隐私字段（如 int64 的手机号/身份证）：
                # 先转 object 类型以兼容字符串脱敏值，再逐行处理
                df[col] = df[col].astype(object)
                for idx, value in df[col].items():
                    if pd.isna(value):
                        continue
                    str_value = str(int(value)) if isinstance(value, (int, np.integer)) else str(value)
                    masked, warning = self.privacy_masker.detect_and_mask(str_value, col)
                    df.at[idx, col] = masked

        return df, {}

    def _handle_standardize_format(self, df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, Dict]:
        """5. 日期格式标准化"""
        df = df.copy()
        date_fields = params.get("dateFields", [])  # 指定日期字段

        for col in df.columns:
            if date_fields and col not in date_fields:
                continue

            # 兼容 pandas 2.0+ 的 str 类型和旧版的 object 类型
            col_dtype = df[col].dtype
            is_string_type = col_dtype == object or col_dtype.name == 'str' or pd.api.types.is_string_dtype(col_dtype)
            if is_string_type:
                # 尝试标准化日期格式
                for idx, value in df[col].items():
                    if isinstance(value, str):
                        standardized = self._standardize_date(value)
                        if standardized:
                            df.at[idx, col] = standardized

        return df, {}

    def _standardize_date(self, value: str) -> Optional[str]:
        """将各种日期格式标准化为 YYYY-MM-DD（带前导零）"""
        if not value or pd.isna(value):
            return None

        # 格式1: YYYY-MM-DD 或 YYYY/MM/DD（可能缺少前导零）
        match = re.match(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", value)
        if match:
            y, m, d = match.groups()
            return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"

        # 格式2: YYYYMMDD（无分隔符）
        match = re.match(r"(\d{4})(\d{2})(\d{2})", value)
        if match:
            y, m, d = match.groups()
            return f"{y}-{m}-{d}"

        # 格式3: MM/DD/YYYY 或 MM-DD-YYYY（美式日期）
        match = re.match(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", value)
        if match:
            m, d, y = match.groups()
            return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"

        return None

    # 步骤处理器映射
    STEP_HANDLERS = {
        "trim_whitespace": _handle_trim_whitespace,
        "remove_duplicates": _handle_remove_duplicates,
        "handle_missing": _handle_missing,
        "privacy_check": _handle_privacy_check,
        "standardize_format": _handle_standardize_format
    }
