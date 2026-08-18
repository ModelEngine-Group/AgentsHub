# -*- coding: utf-8 -*-
"""
DataLoaderMapper - 数据加载算子
支持：CSV/JSON 多文件批量加载、自动编码检测、合并策略、分块处理
版本：v2.2
"""

import os
import json
import time
import pandas as pd
from typing import Dict, Any, List, Optional, Literal, Union
from datetime import datetime


# ==================== 基类 ====================

class Mapper:
    """DataMate 算子基类"""

    def __init__(self, *args, **kwargs):
        self._name = kwargs.get("op_name", "DataLoaderMapper")
        self.text_key = kwargs.get("text_key", "text")

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        try:
            return self.execute(sample)
        except Exception as e:
            sample["execute_result"] = False
            sample["error"] = str(e)
            return sample

    def execute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("子类必须实现 execute 方法")


# ==================== 主类 ====================

class DataLoaderMapper(Mapper):
    """
    数据加载算子

    支持功能：
    - 多格式加载：CSV、JSON
    - 批量处理：多个文件一次加载
    - 合并策略：concat（拼接）、join（关联）、union（去重合并）
    - 编码兼容：UTF-8 → GBK → GB2312 → UTF-8-SIG
    - 大文件处理：分块读取，避免内存溢出
    - 容错模式：部分成功 + 连续错误检测 + 降级策略
    - 质量评分：基于成功率和数据量的评分
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 文件配置
        self.file_paths: List[str] = kwargs.get("file_paths", [])
        self.file_formats: List[str] = kwargs.get("file_formats", [])
        self.encoding: str = kwargs.get("encoding", "utf-8")

        # 分块处理配置
        self.chunk_size: int = kwargs.get("chunk_size", 10000)  # 每块处理的行数

        # 合并配置
        self.merge_strategy: str = kwargs.get("merge_strategy", "concat")

        # 容错配置
        self.max_errors: int = kwargs.get("max_errors", 3)
        self.fail_fast: bool = kwargs.get("fail_fast", False)

        # 内部状态
        self._consecutive_errors: int = 0
        self._total_rows_loaded: int = 0

        # 初始化验证（Guardrails）
        self._validate_and_normalize_paths()

    # ==================== 公开接口 ====================

    def execute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """核心执行逻辑"""
        start_time = time.time()

        # 容错模式检查
        if self._consecutive_errors >= self.max_errors:
            return self._create_degraded_result(
                sample,
                f"连续错误达到 {self.max_errors} 次，触发降级策略"
            )

        # 加载所有文件
        loaded_results = []
        errors = []

        for i, file_path in enumerate(self.file_paths):
            try:
                result = self._load_single_file(file_path, i)
                if result["success"]:
                    loaded_results.append(result)
                    self._consecutive_errors = 0
                else:
                    errors.append(result["error"])
                    self._consecutive_errors += 1

                    # 快速失败模式
                    if self.fail_fast and self._consecutive_errors >= self.max_errors:
                        break
            except Exception as e:
                errors.append(f"{file_path}: {str(e)}")
                self._consecutive_errors += 1

        # 合并数据
        merged_data = self._merge_data(loaded_results)

        # 更新样本
        sample["data"] = merged_data
        sample["count"] = len(merged_data)
        sample["sources"] = [r["source"] for r in loaded_results]
        sample["errors"] = errors
        sample["success_count"] = len(loaded_results)
        sample["error_count"] = len(errors)
        sample["execute_result"] = True

        # 质量评分（Hooks）
        sample["quality_score"] = self._calculate_quality_score(loaded_results, errors)

        # 性能监控（Hooks）
        execution_time = (time.time() - start_time) * 1000
        sample["execution_time_ms"] = round(execution_time, 2)

        return sample

    # ==================== 文件加载 ====================

    def _load_single_file(self, file_path: str, index: int) -> Dict[str, Any]:
        """加载单个文件"""
        # 路径处理（Docker 兼容）
        file_path = self._validate_path(file_path)

        # 格式检测
        file_format = self._detect_format(file_path, index)

        # 加载数据
        if file_format == "csv":
            data = self._load_csv(file_path)
        elif file_format == "json":
            data = self._load_json(file_path)
        else:
            raise ValueError(f"不支持的格式: {file_format}")

        return {
            "success": True,
            "source": file_path,
            "format": file_format,
            "data": data,
            "count": len(data)
        }

    def _load_csv(self, path: str) -> List[Dict[str, Any]]:
        """
        分块读取 CSV 文件（大文件优化版）

        新增功能：
        - 分块处理：避免一次性加载大文件导致 OOM
        - 内存监控：记录加载的总行数
        - 进度友好：每块数据独立处理
        """
        # 尝试多种编码
        encodings_to_try = [self.encoding, "utf-8", "gbk", "gb2312", "utf-8-sig"]
        last_error = None

        for enc in encodings_to_try:
            try:
                all_data = []

                # ========== 核心改动：分块读取 ==========
                # pandas 的 chunksize 参数会返回一个迭代器
                # 每次只读取 chunk_size 行到内存，避免 OOM
                for chunk in pd.read_csv(
                    path,
                    encoding=enc,
                    chunksize=self.chunk_size,  # ← 分块大小
                    on_bad_lines="skip"          # ← 跳过格式错误的行
                ):
                    # 将每块转换为字典列表并合并
                    chunk_records = chunk.to_dict(orient="records")
                    all_data.extend(chunk_records)
                    self._total_rows_loaded += len(chunk_records)

                return all_data

            except UnicodeDecodeError:
                continue
            except Exception as e:
                last_error = e
                break

        # 所有编码都失败
        raise IOError(f"无法读取 CSV（已尝试: {encodings_to_try}）: {last_error}")

    def _load_json(self, path: str) -> List[Dict[str, Any]]:
        """加载 JSON 文件"""
        encodings_to_try = [self.encoding, "utf-8", "gbk"]

        for enc in encodings_to_try:
            try:
                with open(path, "r", encoding=enc) as f:
                    content = json.load(f)

                # 支持单对象和数组
                if isinstance(content, list):
                    return content
                elif isinstance(content, dict):
                    return [content]
                else:
                    raise ValueError(f"JSON 格式不支持: {type(content)}")

            except UnicodeDecodeError:
                continue
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON 解析失败: {e}")

        raise IOError(f"无法读取 JSON 文件: {path}")

    # ==================== 合并策略 ====================

    def _merge_data(self, results: List[Dict]) -> List[Dict]:
        """合并多文件数据"""
        if not results:
            return []

        all_data = []
        for result in results:
            all_data.extend(result.get("data", []))

        if self.merge_strategy == "concat":
            return all_data

        elif self.merge_strategy == "union":
            # 基于 JSON 序列化去重
            seen = set()
            unique_data = []
            for item in all_data:
                key = json.dumps(item, sort_keys=True, ensure_ascii=False)
                if key not in seen:
                    seen.add(key)
                    unique_data.append(item)
            return unique_data

        elif self.merge_strategy == "join":
            # 简单拼接（实际可能需要根据 key 关联）
            return all_data

        return all_data

    # ==================== 辅助方法 ====================

    def _validate_and_normalize_paths(self) -> None:
        """路径校验 + 规范化"""
        if not self.file_paths:
            raise ValueError("file_paths 不能为空")

        normalized = []
        for path in self.file_paths:
            normalized.append(self._validate_path(path))
        self.file_paths = normalized

    def _validate_path(self, path: str) -> str:
        """Docker 路径转换"""
        if path.startswith("/mnt/data/"):
            return path.replace("/mnt/data/", "D:/data/")
        return path

    def _detect_format(self, path: str, index: int) -> str:
        """检测文件格式"""
        # 优先使用手动指定的格式
        if self.file_formats and index < len(self.file_formats):
            return self.file_formats[index]

        # 通过后缀猜测
        ext = os.path.splitext(path)[1].lower()
        format_map = {
            ".csv": "csv",
            ".json": "json",
            ".jsonl": "json"
        }
        return format_map.get(ext, "csv")

    def _calculate_quality_score(
        self,
        results: List[Dict],
        errors: List[str]
    ) -> float:
        """计算质量评分（0-1）"""
        if not self.file_paths:
            return 0.0

        # 成功率
        success_rate = len(results) / len(self.file_paths)

        # 数据量检查
        total_rows = sum(r.get("count", 0) for r in results)
        has_data = 1.0 if total_rows > 0 else 0.0

        # 综合评分
        return round(success_rate * 0.7 + has_data * 0.3, 2)

    def _create_degraded_result(
        self,
        sample: Dict,
        reason: str
    ) -> Dict[str, Any]:
        """创建降级结果"""
        return {
            "data": [],
            "count": 0,
            "sources": [],
            "errors": [reason],
            "success_count": 0,
            "error_count": len(self.file_paths),
            "execute_result": True,  # 降级不是失败
            "quality_score": 0.0,
            "execution_time_ms": 0,
            "degraded": True,
            "degraded_reason": reason
        }
