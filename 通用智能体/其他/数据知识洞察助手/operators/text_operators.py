"""
operators/text_operators.py
===========================
文本处理算子集合。

包含：
- ReadTextOperator：读取文本文件
- SplitParagraphOperator：段落分割
- CleanTextOperator：文本清洗
- SaveTextOperator：保存文本文件
"""

from pathlib import Path
from typing import Optional, List
from .base import BaseOperator, OperatorRuntimeError


class ReadTextOperator(BaseOperator):
    """读取文本文件"""

    name = "ReadTextOperator"
    description = "读取文本文件，返回字符串内容"
    input_schema = {"file_path": "str"}
    output_schema = {"text": "str"}

    def __init__(self):
        super().__init__()

    def run(self, file_path: str, encoding: str = "utf-8") -> str:
        path = Path(file_path)
        if not path.exists():
            raise OperatorRuntimeError(f"文件不存在: {file_path}")

        with open(file_path, "r", encoding=encoding) as f:
            text = f.read()

        print(f"[ReadTextOperator] 读取成功，字符数: {len(text)}")
        return text


class SplitParagraphOperator(BaseOperator):
    """按段落分割文本"""

    name = "SplitParagraphOperator"
    description = "将长文本按段落分割为列表"
    input_schema = {"text": "str"}
    output_schema = {"paragraphs": "List[str]"}

    def __init__(self):
        super().__init__()

    def run(self, text: str, min_length: int = 10) -> List[str]:
        """
        分割文本为段落。

        Args:
            text: 原始文本
            min_length: 最小段落长度，过短的段落会被丢弃

        Returns:
            段落列表
        """
        import re
        # 按换行+空行分割
        paragraphs = re.split(r"\n\s*\n", text.strip())
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) >= min_length]

        print(f"[SplitParagraphOperator] 分割为 {len(paragraphs)} 个段落")
        return paragraphs


class CleanTextOperator(BaseOperator):
    """文本清洗算子"""

    name = "CleanTextOperator"
    description = "清洗文本：去除特殊字符、多余空格"
    input_schema = {"text": "str"}
    output_schema = {"text": "str"}

    def __init__(self):
        super().__init__()

    def run(
        self,
        text: str,
        remove_digits: bool = False,
        remove_punctuation: bool = False,
    ) -> str:
        """
        清洗文本。

        Args:
            text: 原始文本
            remove_digits: 是否去除数字
            remove_punctuation: 是否去除标点
        """
        import re

        cleaned = text

        # 去除多余空格
        cleaned = re.sub(r"\s+", " ", cleaned)

        if remove_digits:
            cleaned = re.sub(r"\d+", "", cleaned)

        if remove_punctuation:
            cleaned = re.sub(r"[^\w\s]", "", cleaned)

        print(f"[CleanTextOperator] 清洗完成，字符数: {len(cleaned)}")
        return cleaned.strip()


class SaveTextOperator(BaseOperator):
    """保存文本到文件"""

    name = "SaveTextOperator"
    description = "将文本保存到文件"
    input_schema = {"text": "str", "file_path": "str"}
    output_schema = {"saved_path": "str"}

    def __init__(self):
        super().__init__()

    def run(self, text: str, file_path: str, encoding: str = "utf-8") -> str:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding=encoding) as f:
            f.write(text)

        print(f"[SaveTextOperator] 已保存至: {file_path}，字符数: {len(text)}")
        return str(path)
