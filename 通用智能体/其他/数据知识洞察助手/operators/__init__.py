"""
operators
=========
数据处理与知识抽取算子集合。

所有算子都继承自 BaseOperator（定义在 base.py）。
"""

from .base import BaseOperator, OperatorRuntimeError

# CSV 相关算子
from .csv_operators import (
    ReadCsvOperator,
    CleanMissingOperator,
    NormalizeTextOperator,
    ExtractFieldOperator,
    SaveCsvOperator,
)

# 文本处理算子
from .text_operators import (
    ReadTextOperator,
    SplitParagraphOperator,
    CleanTextOperator,
    SaveTextOperator,
)

# JSON 相关算子
from .json_operators import SaveJsonOperator

# 知识图谱相关算子
from .kg_operators import (
    EntityRecognitionOperator,
    GraphBuildOperator,
    RelationExtractionOperator,
    TripleGenerationOperator,
    TripleValidationOperator,
)

# 图谱分析相关算子
from .analysis_operators import (
    CentralityAnalysisOperator,
    DiseaseSymptomCountOperator,
    DrugDiseaseCountOperator,
    GraphStatisticsOperator,
    NL2AnalysisOperator,
)

# NL2SQL 相关算子
from .nl2sql_operators import (
    BuildMedicalSqliteOperator,
    EvaluateNL2SQLOperator,
    ExecuteSQLOperator,
    NL2SQLOperator,
)

__all__ = [
    "BaseOperator",
    "OperatorRuntimeError",
    # CSV
    "ReadCsvOperator",
    "CleanMissingOperator",
    "NormalizeTextOperator",
    "ExtractFieldOperator",
    "SaveCsvOperator",
    # Text
    "ReadTextOperator",
    "SplitParagraphOperator",
    "CleanTextOperator",
    "SaveTextOperator",
    # JSON
    "SaveJsonOperator",
    # KG
    "EntityRecognitionOperator",
    "GraphBuildOperator",
    "RelationExtractionOperator",
    "TripleGenerationOperator",
    "TripleValidationOperator",
    # Analysis
    "CentralityAnalysisOperator",
    "DiseaseSymptomCountOperator",
    "DrugDiseaseCountOperator",
    "GraphStatisticsOperator",
    "NL2AnalysisOperator",
    # NL2SQL
    "BuildMedicalSqliteOperator",
    "EvaluateNL2SQLOperator",
    "ExecuteSQLOperator",
    "NL2SQLOperator",
]
