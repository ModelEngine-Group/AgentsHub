"""
operators/csv_operators.py
=========================
CSV 数据处理算子集合。

包含：
- ReadCsvOperator：读取 CSV 文件
- CleanMissingOperator：清洗空值
- NormalizeTextOperator：文本规范化
- ExtractFieldOperator：字段抽取
- SaveCsvOperator：保存 CSV 文件
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from .base import BaseOperator, OperatorRuntimeError


class ReadCsvOperator(BaseOperator):
    """读取 CSV 文件"""

    name = "ReadCsvOperator"
    description = "读取 CSV 文件，返回 pandas DataFrame"
    input_schema = {"file_path": "str"}
    output_schema = {"df": "DataFrame"}

    def __init__(self):
        super().__init__()

    def run(self, file_path: str, encoding: str = "utf-8") -> pd.DataFrame:
        """
        读取 CSV 文件。

        Args:
            file_path: CSV 文件路径
            encoding: 文件编码，默认 utf-8

        Returns:
            pandas DataFrame
        """
        path = Path(file_path)
        if not path.exists():
            raise OperatorRuntimeError(f"文件不存在: {file_path}")

        df = pd.read_csv(file_path, encoding=encoding)
        print(f"[ReadCsvOperator] 读取成功，共 {len(df)} 行，{len(df.columns)} 列")
        print(f"[ReadCsvOperator] 字段: {list(df.columns)}")
        return df


class CleanMissingOperator(BaseOperator):
    """清洗 DataFrame 中的空值"""

    name = "CleanMissingOperator"
    description = "清洗 DataFrame 中的空值"
    input_schema = {"df": "DataFrame"}
    output_schema = {"df": "DataFrame"}

    def __init__(
        self,
        strategy: str = "drop",  # drop / fill / forward
        fill_value: Any = None,
        columns: Optional[List[str]] = None
    ):
        """
        Args:
            strategy: 清洗策略
                - drop: 删除含有空值的行
                - fill: 用 fill_value 填充
                - forward: 用前向填充
            fill_value: 填充值（strategy=fill 时使用）
            columns: 指定要清洗的列，None 表示全部列
        """
        super().__init__()
        self.strategy = strategy
        self.fill_value = fill_value
        self.columns = columns

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.columns:
            df = df[self.columns]

        original_len = len(df)

        if self.strategy == "drop":
            df = df.dropna()
        elif self.strategy == "fill":
            df = df.fillna(value=self.fill_value)
        elif self.strategy == "forward":
            df = df.ffill()
        else:
            raise OperatorRuntimeError(f"未知清洗策略: {self.strategy}")

        removed = original_len - len(df)
        print(f"[CleanMissingOperator] 策略={self.strategy}，删除了 {removed} 行空值记录")
        return df


class NormalizeTextOperator(BaseOperator):
    """文本规范化算子"""

    name = "NormalizeTextOperator"
    description = "规范化文本字段，去除多余空格和特殊字符"
    input_schema = {"df": "DataFrame", "columns": "List[str]"}
    output_schema = {"df": "DataFrame"}

    def __init__(self):
        super().__init__()

    def run(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        规范化文本字段。

        Args:
            df: 输入 DataFrame
            columns: 要规范化的列名列表，None 表示所有 object 类型列
        """
        if columns is None:
            columns = df.select_dtypes(include=["object"]).columns.tolist()

        df = df.copy()
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

        print(f"[NormalizeTextOperator] 规范化了列: {columns}")
        return df


class ExtractFieldOperator(BaseOperator):
    """字段抽取算子"""

    name = "ExtractFieldOperator"
    description = "从 DataFrame 中抽取指定字段"
    input_schema = {"df": "DataFrame", "fields": "List[str]"}
    output_schema = {"df": "DataFrame"}

    def __init__(self):
        super().__init__()

    def run(self, df: pd.DataFrame, fields: List[str]) -> pd.DataFrame:
        """
        抽取指定字段。

        Args:
            df: 输入 DataFrame
            fields: 要抽取的字段名列表

        Returns:
            只包含指定字段的 DataFrame
        """
        missing = [f for f in fields if f not in df.columns]
        if missing:
            raise OperatorRuntimeError(f"字段不存在: {missing}")

        extracted = df[fields].copy()
        print(f"[ExtractFieldOperator] 抽取字段: {fields}，共 {len(extracted)} 行")
        return extracted


class SaveCsvOperator(BaseOperator):
    """保存 DataFrame 为 CSV 文件"""

    name = "SaveCsvOperator"
    description = "将 DataFrame 保存为 CSV 文件"
    input_schema = {"df": "DataFrame", "file_path": "str"}
    output_schema = {"saved_path": "str"}

    def __init__(self):
        super().__init__()

    def run(self, df: pd.DataFrame, file_path: str, encoding: str = "utf-8") -> str:
        """
        保存 DataFrame 为 CSV。

        Args:
            df: 要保存的 DataFrame
            file_path: 保存路径
            encoding: 文件编码，默认 utf-8

        Returns:
            保存的文件路径
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(file_path, index=False, encoding=encoding)
        print(f"[SaveCsvOperator] 已保存至: {file_path}，共 {len(df)} 行")
        return str(path)
