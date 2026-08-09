#!/usr/bin/env python3
"""
DataMate MCP Server v3.0
========================
给Nexent智能体提供DataMate算子调用能力的MCP服务器。

【架构说明】
- list_operators: 调用 DataMate 真实API（POST /api/operators/list）
- run_etl_pipeline: 一键ETL（本地执行，默认参数与本地脚本一致）
- execute_operator: 本地Python直接执行单个算子

【启动方式】
    set DATAMATE_TOKEN=your_token_here
    python server.py

【在Nexent中连接】
    - 方式：自定义MCP服务器（方式2）
    - 地址：http://host.docker.internal:8089
"""

import json
import os
import sys
import subprocess
import tempfile
import traceback
import time
import re
from typing import Optional, Dict, Any, List
from enum import Enum

import httpx
from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP

# ============================================================
# 配置常量
# ============================================================

SERVER_PORT = int(os.getenv("MCP_PORT", "8089"))
DATAMATE_GATEWAY = os.getenv("DATAMATE_GATEWAY", "http://localhost:8080")
DATAMATE_TOKEN = os.getenv("DATAMATE_TOKEN", "")

# 算子源目录（本地Python执行用）— 相对路径，评委机器也能跑
_OPERATORS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "operators")

# ============================================================
# 服务器初始化
# ============================================================

mcp = FastMCP("datamate_mcp", host="0.0.0.0", port=SERVER_PORT, streamable_http_path="/")

# ============================================================
# 本地支持的算子列表（始终可用，无论DataMate状态如何）
# ============================================================

_LOCAL_OPERATORS = [
    {
        "name": "data_loader",
        "display_name": "数据加载器",
        "version": "v2.2.0",
        "modal": "text",
        "types": ["cleaning"],
        "description": "加载原始数据（CSV/JSON/TXT），解析为标准格式。ETL流程第一步",
        "module": "data_loader",
        "class_name": "DataLoaderMapper",
        "interface": "execute",
    },
    {
        "name": "data_cleaner",
        "display_name": "数据清洗器",
        "version": "v1.0.0",
        "modal": "text",
        "types": ["cleaning"],
        "description": "数据清洗：去重、空值填补、数据脱敏。ETL流程第二步",
        "module": "data_cleaner",
        "class_name": "DataCleaner",
        "interface": "process",
    },
    {
        "name": "data_transformer",
        "display_name": "数据转换器",
        "version": "v1.0.0",
        "modal": "text",
        "types": ["cleaning"],
        "description": "数据转换：类型标准化、字段映射、格式转换。ETL流程第三步",
        "module": "data_transformer",
        "class_name": "DataTransformer",
        "interface": "process",
    },
    {
        "name": "data_exporter",
        "display_name": "数据导出器",
        "version": "v1.0.0",
        "modal": "text",
        "types": ["cleaning"],
        "description": "导出处理后的数据为CSV/JSON文件。ETL流程第四步",
        "module": "data_exporter",
        "class_name": "DataExporter",
        "interface": "process",
    },
    # === KG 知识图谱算子（任务二 Week 5）===
    {
        "name": "kg_entity_recognizer",
        "display_name": "医疗实体识别器",
        "version": "v1.0.0",
        "modal": "text",
        "types": ["cleaning"],
        "description": "基于医疗词典+首字索引，从文本中识别4类医疗实体：疾病、症状、药物、检查。输出实体类型、名称、位置信息",
        "module": "kg_entity_recognizer",
        "class_name": "KGEntityRecognizer",
        "interface": "process",
    },
    {
        "name": "kg_relation_extractor",
        "display_name": "医疗关系抽取器",
        "version": "v1.0.0",
        "modal": "text",
        "types": ["cleaning"],
        "description": "基于模式匹配+共现推理，抽取4类关系：导致(疾病→症状)、治疗(药物→疾病)、用于(药物→症状)、禁忌(药物→疾病)。需先执行EntityRecognizer提供实体",
        "module": "kg_relation_extractor",
        "class_name": "KGRelationExtractor",
        "interface": "process",
    },
    {
        "name": "kg_triple_generator",
        "display_name": "知识图谱三元组生成器",
        "version": "v1.0.0",
        "modal": "text",
        "types": ["cleaning"],
        "description": "将实体+关系组合为标准化KG三元组，含去重、冲突检测（直接冲突+禁忌冲突）、实体类型校验。需先执行EntityRecognizer和RelationExtractor",
        "module": "kg_triple_generator",
        "class_name": "KGTripleGenerator",
        "interface": "process",
    },
]


# ============================================================
# 枚举与模型定义
# ============================================================


class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"


class ExecuteOperatorInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    operator_name: str = Field(
        ...,
        description="算子名称，例如: data_loader, data_cleaner, data_transformer, data_exporter",
        min_length=1,
        max_length=100,
    )
    sample: Dict[str, Any] = Field(
        default_factory=dict,
        description="输入样本数据。对data_loader传 {\"text\": \"csv内容\"}，对其他算子传前一步返回结果中的data字段",
    )
    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="算子参数（可选）。例如: data_cleaner可用 {\"remove_duplicates\": true, \"fill_na\": true}",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="返回格式: markdown（人类可读）/ json（机器可读，包含data和report字段）",
    )


class RunETLPipelineInput(BaseModel):
    """一键执行完整ETL流水线的输入参数"""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    csv_text: str = Field(
        ...,
        description="CSV文本内容（从read_file读取的完整内容）",
        min_length=1,
    )
    cleaner_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="data_cleaner的参数（可选）。如: {\"remove_duplicates\": True, \"fill_na\": True, \"privacyCheck\": True}",
    )
    transformer_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="data_transformer的参数（可选）。如: {\"standardize_types\": True}",
    )
    exporter_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="data_exporter的参数（可选）。关键参数: {\"outputDir\": \"/mnt/nexent/output\"} 指定输出目录，不传则返回CSV文本。注意参数名是outputDir（驼峰），不是output_path",
    )


class ListOperatorsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    page: Optional[int] = Field(default=1, description="页码（从1开始）", ge=1, le=100)
    size: Optional[int] = Field(default=10, description="每页数量", ge=1, le=100)
    keyword: Optional[str] = Field(default=None, description="搜索关键词（可选）")
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="输出格式: markdown/json",
    )


class KGEntityRecognitionInput(BaseModel):
    """医疗文本实体识别的输入参数"""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    text: str = Field(
        ...,
        description="待识别的医疗文本，例如: \"患者糖尿病多年，出现多饮多尿症状\"",
        min_length=1,
        max_length=10000,
    )


class KGRelationExtractionInput(BaseModel):
    """医疗实体关系抽取的输入参数"""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    text: str = Field(
        ...,
        description="待抽取关系的医疗文本，例如: \"患者糖尿病多年，服用二甲双胍治疗\"",
        min_length=1,
        max_length=10000,
    )
    entities: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="已识别的实体列表（可选）。若不提供，会自动调用实体识别器先识别实体。"
                     "格式: [{\"type\": \"疾病\", \"name\": \"糖尿病\"}]",
    )


class KGTripleGenerationInput(BaseModel):
    """知识图谱三元组生成的输入参数"""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    entities: List[Dict[str, Any]] = Field(
        ...,
        description="实体列表，格式: [{\"type\": \"疾病\", \"name\": \"糖尿病\"}, {\"type\": \"药物\", \"name\": \"二甲双胍\"}]",
        min_length=1,
    )
    relations: List[Dict[str, Any]] = Field(
        ...,
        description="关系列表，格式: [{\"head\": \"二甲双胍\", \"relation\": \"治疗\", \"tail\": \"糖尿病\", \"confidence\": \"high\"}]",
        min_length=1,
    )
    text: Optional[str] = Field(
        default=None,
        description="原始医疗文本（可选，用于冲突检测上下文）",
        max_length=10000,
    )


class VerifyConnectionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ============================================================
# DataMate API 调用工具
# ============================================================


def _get_auth_headers() -> Dict[str, str]:
    if not DATAMATE_TOKEN:
        return {}
    return {"Authorization": f"Bearer {DATAMATE_TOKEN}"}


async def _call_datamate_api(
    path: str,
    method: str = "POST",
    json_data: Optional[Dict] = None,
    timeout: float = 30.0,
) -> Dict:
    """调用 DataMate 真实 API（Python后端，通过网关）"""
    url = f"{DATAMATE_GATEWAY.rstrip('/')}{path}"
    headers = _get_auth_headers()
    headers["Content-Type"] = "application/json"

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.request(
            method=method,
            url=url,
            headers=headers,
            json=json_data,
        )
        response.raise_for_status()
        return response.json()


def _format_error(e: Exception) -> str:
    """格式化错误信息"""
    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        try:
            body = e.response.json()
            msg = body.get("message", body.get("error", str(e)))
        except (json.JSONDecodeError, AttributeError):
            msg = str(e)
        if status == 401:
            return "❌ 认证失败（401）：DATAMATE_TOKEN 无效或已过期"
        elif status == 403:
            return f"❌ 权限不足（403）：{msg}"
        elif status == 404:
            return "❌ API 路径不存在（404）：请检查 DataMate 版本"
        elif status >= 500:
            return f"❌ DataMate服务端错误（{status}）：{msg}"
        return f"❌ API请求失败（{status}）：{msg}"
    if isinstance(e, httpx.ConnectError):
        return f"❌ 无法连接到 DataMate：{DATAMATE_GATEWAY}，请确认 DataMate 已启动"
    if isinstance(e, httpx.TimeoutException):
        return "❌ 请求超时：DataMate 响应时间过长"
    return f"❌ 错误：{type(e).__name__}: {str(e)}"


# ============================================================
# 本地算子执行引擎
# ============================================================


def _import_operator_class(module_name: str, class_name: str):
    """动态导入算子类"""
    if _OPERATORS_DIR not in sys.path:
        sys.path.insert(0, _OPERATORS_DIR)
    module = __import__(module_name)
    return getattr(module, class_name)


def _sanitize_value(v):
    """将numpy类型转为Python原生类型，确保可JSON序列化"""
    import math
    if isinstance(v, dict):
        return {k: _sanitize_value(v) for k, v in v.items()}
    elif isinstance(v, (list, tuple)):
        return [_sanitize_value(x) for x in v]
    elif isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    elif hasattr(v, 'item'):  # numpy int64, float64 etc.
        return v.item()
    return v


def _sanitize_data(data):
    """深度清洗数据中的numpy类型，返回纯Python类型"""
    return _sanitize_value(data)


def _ensure_camel_case(params: Optional[Dict]) -> Dict:
    """将参数名统一转为驼峰（cleaner内部要求驼峰参数名）"""
    if not params:
        return {}
    result = {}
    snake_to_camel = {
        "remove_duplicates": "removeDuplicates",
        "fill_na": "handleMissing",
        "fillna": "handleMissing",
        "trim_whitespace": "trimWhitespace",
        "standardize_format": "standardizeFormat",
        "standardize_types": "standardizeFormat",
        "privacy_check": "privacyCheck",
        "handle_missing": "handleMissing",
        "fill_value": "fillValue",
        "fillvalue": "fillValue",
        "output_path": "outputDir",
        "output_dir": "outputDir",
    }
    for k, v in params.items():
        camel_k = snake_to_camel.get(k, k)
        # 特殊值转换：fill_na=True → handleMissing="fill"
        if k in ("fill_na", "fillna") and v is True:
            v = "fill"
        # 如果已存在驼峰key，优先保留用户传的驼峰值
        if camel_k not in result:
            result[camel_k] = v
    return result


def _run_operator_locally(
    op_name: str,
    sample: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """本地执行算子，返回结构化结果"""
    # 查找算子定义
    op_def = None
    for op in _LOCAL_OPERATORS:
        if op["name"] == op_name:
            op_def = op
            break
    if not op_def:
        return {"success": False, "error": f"未知算子: {op_name}"}

    start = time.perf_counter()
    temp_file = None

    try:
        # 导入算子类
        cls = _import_operator_class(op_def["module"], op_def["class_name"])

        interface = op_def["interface"]
        if interface == "execute":
            # DataLoaderMapper: 需要 file_paths 参数初始化
            file_paths = []

            # 如果传了 text 内容，写入临时文件
            if sample and "text" in sample:
                temp_file = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".csv", delete=False, encoding="utf-8"
                )
                csv_text = sample["text"]
                # 确保换行符统一
                if not csv_text.endswith("\n"):
                    csv_text += "\n"
                temp_file.write(csv_text)
                temp_file.flush()
                temp_file.close()
                file_paths = [temp_file.name]
            elif params and "file_paths" in params:
                file_paths = params["file_paths"]

            # 用 file_paths 初始化 DataLoaderMapper
            init_kwargs = {"file_paths": file_paths}
            if params:
                for k in ("encoding", "chunk_size", "merge_strategy", "max_errors"):
                    if k in params:
                        init_kwargs[k] = params[k]

            instance = cls(**init_kwargs)
            result = instance.execute(sample or {})

        elif interface == "process":
            # DataCleaner / DataTransformer / DataExporter 使用 process(data, params)
            instance = cls()
            input_data = sample.get("data", sample)
            proc_params = dict(params or {})
            # data_exporter 参数名兼容：output_path → outputDir
            if op_name == "data_exporter":
                if "output_path" in proc_params and "outputDir" not in proc_params:
                    path = proc_params.pop("output_path")
                    proc_params["outputDir"] = os.path.dirname(path) if os.path.basename(path) else path
                if "output_dir" in proc_params and "outputDir" not in proc_params:
                    proc_params["outputDir"] = proc_params.pop("output_dir")
            result = instance.process(input_data, proc_params)

        else:
            return {"success": False, "error": f"不支持接口: {interface}"}

        # 清洗结果中的numpy类型，确保可JSON序列化
        result = _sanitize_data(result)
        elapsed = round((time.perf_counter() - start) * 1000, 2)

        return {
            "success": True,
            "result": result,
            "elapsed_ms": elapsed,
            "operator": op_name,
        }

    except Exception as e:
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        return {
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "elapsed_ms": elapsed,
            "operator": op_name,
        }
    finally:
        # 清理临时文件
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except (OSError, PermissionError):
                pass


# ============================================================
# 工具1：验证连接
# ============================================================


@mcp.tool(
    name="datamate_verify_connection",
    annotations={
        "title": "验证DataMate连接",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def datamate_verify_connection(params: VerifyConnectionInput) -> str:
    """验证MCP服务器是否能正常连接到DataMate API。

    在开始数据处理前，建议先用此工具确认连接是否正常。
    它会检查Token是否有效、API网关是否可达、算子列表能否获取。

    Args:
        params (VerifyConnectionInput): 无需任何参数

    Returns:
        str: 连接测试结果
    """
    results = ["# 🔌 DataMate 连接验证", ""]

    # 1. 检查Token
    if not DATAMATE_TOKEN:
        results.append("⚠️  DATAMATE_TOKEN 未设置")
        results.append("   获取方法：浏览器打开 http://localhost:30000 → F12 → Application → Local Storage")
        results.append("")
    else:
        results.append(f"✅ DATAMATE_TOKEN 已设置（长度: {len(DATAMATE_TOKEN)}）")
        results.append("")

    # 2. 测试网关连通性
    try:
        results.append(f"📡 正在连接网关: {DATAMATE_GATEWAY} ...")
        headers = _get_auth_headers()
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                DATAMATE_GATEWAY.rstrip("/") + "/",
                headers=headers,
                follow_redirects=False,
            )
            results.append(f"✅ 网关可达（HTTP {resp.status_code}）")
    except httpx.ConnectError:
        results.append(f"❌ 无法连接到网关，请确认 DataMate 已启动")
        results.append("")
        return "\n".join(results)
    except Exception as e:
        results.append(f"⚠️ 网关检查异常: {type(e).__name__}")
    results.append("")

    # 3. 测试算子列表API
    try:
        results.append("📡 正在测试算子列表API ...")
        data = await _call_datamate_api("/api/operators/list", json_data={"size": 1})
        total = data.get("data", {}).get("totalElements", 0)
        results.append(f"✅ 算子列表API正常！DataMate 共 {total} 个内置算子")
    except Exception as e:
        results.append(_format_error(e))

    results.append("")

    # 4. 本地算子状态
    results.append("---")
    results.append(f"📦 本地算子（{len(_LOCAL_OPERATORS)} 个，可直接执行）：")
    for op in _LOCAL_OPERATORS:
        results.append(f"   - `{op['name']}`: {op['description']}")

    return "\n".join(results)


# ============================================================
# 工具2：查询算子列表
# ============================================================


@mcp.tool(
    name="datamate_list_operators",
    annotations={
        "title": "列出DataMate算子",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def datamate_list_operators(params: ListOperatorsInput) -> str:
    """查询DataMate平台上所有可用的算子列表。

    同时返回 DataMate 内置算子（215+个）和本地自定义算子（4个）。
    支持关键词搜索和分页。

    Args:
        params (ListOperatorsInput): 包含:
            - page (int): 页码，从1开始（默认: 1）
            - size (int): 每页数量（默认: 10，最大: 100）
            - keyword (str): 搜索关键词（可选）
            - response_format (str): 输出格式 markdown/json（默认: markdown）

    Returns:
        str: 格式化后的算子列表
    """
    all_operators = []

    # 1. 从DataMate API获取内置算子
    try:
        body = {"page": params.page, "size": params.size}
        if params.keyword:
            body["keyword"] = params.keyword
        data = await _call_datamate_api("/api/operators/list", json_data=body)
        api_ops = data.get("data", {}).get("content", [])
        total = data.get("data", {}).get("totalElements", 0)
        total_pages = data.get("data", {}).get("totalPages", 1)

        for op in api_ops:
            all_operators.append({
                "name": op.get("id", op.get("name", "?")),
                "display_name": op.get("name", op.get("id", "?")),
                "version": op.get("version", "-"),
                "description": op.get("description", "暂无描述"),
                "modal": op.get("inputs", "-"),
                "types": ["datamate"],
                "source": "datamate",
            })
    except Exception:
        total = 0
        total_pages = 0

    # 2. 合并本地算子（总是显示）
    all_operators.extend([
        {**op, "source": "local"} for op in _LOCAL_OPERATORS
    ])

    if not all_operators:
        return "📭 没有获取到算子列表。"

    if params.response_format == ResponseFormat.JSON:
        response = {
            "total_datamate": total,
            "total_local": len(_LOCAL_OPERATORS),
            "page": params.page,
            "size": params.size,
            "operators": [
                {
                    "name": op["name"],
                    "display_name": op["display_name"],
                    "version": op["version"],
                    "description": op["description"],
                    "modal": op["modal"],
                    "source": op.get("source", "datamate"),
                }
                for op in all_operators
            ],
        }
        return json.dumps(response, indent=2, ensure_ascii=False)

    # Markdown 格式
    lines = ["# DataMate 算子列表", ""]

    if total > 0:
        lines.append(f"📡 DataMate 内置算子: 共 {total} 个（第 {params.page}/{total_pages} 页）")
    else:
        lines.append("📡 DataMate API 暂不可用")

    # 分组显示：先显示DataMate的，再显示本地的
    datamate_ops = [op for op in all_operators if op.get("source") == "datamate"]
    local_ops = [op for op in all_operators if op.get("source") == "local"]

    if datamate_ops:
        lines.append("")
        lines.append("## 🔷 DataMate 内置算子")
        lines.append("")
        for op in datamate_ops:
            lines.append(f"### {op['display_name']} (`{op['name']}`)")
            lines.append(f"- **版本**: {op['version']}")
            lines.append(f"- **模态**: {op['modal']}")
            lines.append(f"- **描述**: {op['description']}")
            lines.append("")

    if local_ops:
        lines.append("---")
        lines.append("## 💚 本地自定义算子（可直接执行）")
        lines.append("")
        for op in local_ops:
            lines.append(f"### {op['display_name']} (`{op['name']}`)")
            lines.append(f"- **版本**: {op['version']}")
            lines.append(f"- **类型**: {', '.join(op['types'])}")
            lines.append(f"- **模态**: {op['modal']}")
            lines.append(f"- **描述**: {op['description']}")
            lines.append("")

    lines.append("---")
    lines.append("💡 推荐使用 `datamate_run_etl_pipeline` 工具一键执行完整ETL流程")
    lines.append("   或使用 `datamate_execute_operator` 分步执行")

    return "\n".join(lines)


# ============================================================
# 工具3：执行算子（本地执行）
# ============================================================


@mcp.tool(
    name="datamate_execute_operator",
    annotations={
        "title": "执行DataMate算子",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def datamate_execute_operator(params: ExecuteOperatorInput) -> str:
    """执行数据处理算子或知识图谱算子（本地Python引擎）。

    这是最核心的工具，直接在本地Python环境中执行算子代码。
    支持ETL数据处理和知识图谱（KG）两类算子。

    ETL流程按顺序调用:
    1. data_loader - 加载原始数据
    2. data_cleaner - 清洗数据（去重、空值填补、脱敏等）
    3. data_transformer - 转换数据（类型标准化、字段映射等）
    4. data_exporter - 导出数据为CSV/JSON

    KG流程按顺序调用:
    1. kg_entity_recognizer - 医疗实体识别（疾病/症状/药物/检查）
    2. kg_relation_extractor - 医疗关系抽取（导致/治疗/用于/禁忌）
    3. kg_triple_generator - 知识图谱三元组生成（含冲突检测）

    💡 知识图谱专用工具推荐使用独立的 knowledge_graph_search / entity_recognition / relation_extraction 工具，接口更简洁。

    Args:
        params (ExecuteOperatorInput): 包含:
            - operator_name (str): 算子名称（必填）
              支持: data_loader, data_cleaner, data_transformer, data_exporter, kg_entity_recognizer, kg_relation_extractor, kg_triple_generator
            - sample (dict): 输入数据
              对 data_loader: {"text": "CSV原始内容"} 或 {"data": [...]}
              对其他算子: 前一步返回结果中的 data 字段
            - params (Optional[dict]): 算子参数
              data_cleaner: {"remove_duplicates": true, "fill_na": true}
              data_transformer: {"standardize_types": true}
              data_exporter: {"format": "csv", "outputDir": "/mnt/nexent/output"}

    Returns:
        str: 算子执行结果
    """
    if not params.sample:
        return "❌ 请提供 sample 参数（输入样本数据）"

    # 检查算子是否支持
    op_names = [op["name"] for op in _LOCAL_OPERATORS]
    if params.operator_name not in op_names:
        return (
            f"❌ 不支持的算子: `{params.operator_name}`\n\n"
            f"支持的算子: {', '.join(op_names)}\n\n"
            "💡 可以先调用 `datamate_list_operators` 查看所有可用算子"
        )

    # 执行
    result = _run_operator_locally(params.operator_name, params.sample, params.params)

    # JSON格式返回
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(result, indent=2, ensure_ascii=False)

    # Markdown格式
    lines = [f"## 算子执行结果: {params.operator_name}", ""]

    if result["success"]:
        lines.append(f"✅ **执行成功**（耗时: {result['elapsed_ms']}ms）")
        lines.append("")

        op_result = result.get("result", {})
        if isinstance(op_result, dict):
            # 提取data和report
            data = op_result.get("data", op_result.get("result", {}))
            report = op_result.get("report", op_result.get("step_report", {}))

            if report:
                lines.append("### 📊 处理报告")
                lines.append("")
                if isinstance(report, dict):
                    for k, v in report.items():
                        if k in ("step", "time_ms"):
                            continue
                        if isinstance(v, (dict, list)):
                            v = json.dumps(v, ensure_ascii=False)
                        lines.append(f"- **{k}**: {v}")
                elif isinstance(report, str):
                    lines.append(report)
                lines.append("")

            if data:
                data_str = json.dumps(data, indent=2, ensure_ascii=False)
                if len(data_str) > 3000:
                    lines.append(f"### 📦 结果数据（共 {len(data_str)} 字符，已截断）")
                    lines.append("")
                    lines.append("```json")
                    lines.append(data_str[:2000])
                    lines.append("\n...（截断）")
                    lines.append("```")
                else:
                    lines.append("### 📦 结果数据")
                    lines.append("")
                    lines.append("```json")
                    lines.append(data_str)
                    lines.append("```")

            # 处理导出路径
            export_path = (
                op_result.get("export_summary", {})
                .get("output_path")
                or op_result.get("output_path")
                or op_result.get("path")
            )
            if export_path:
                lines.append("")
                lines.append(f"📄 **输出文件**: `{export_path}`")

        elif isinstance(op_result, str):
            if len(op_result) > 2000:
                lines.append("**结果**:（较长，已截断）")
                lines.append("")
                lines.append(op_result[:2000] + "\n...")
            else:
                lines.append("**结果**:")
                lines.append("")
                lines.append(op_result)
        else:
            lines.append(f"**结果**: {op_result}")

        lines.append("")
        lines.append(f"⏱ 耗时: {result['elapsed_ms']}ms")

    else:
        lines.append(f"❌ **执行失败**")
        lines.append("")
        lines.append(f"错误: {result.get('error', '未知错误')}")
        lines.append("")
        lines.append("可能的原因：")
        lines.append("1. 算子代码路径 `_OPERATORS_DIR` 配置不正确")
        lines.append("2. 算子依赖的Python包未安装")
        lines.append("3. 输入数据格式不符合算子要求")
        lines.append("")
        lines.append("💡 可以先调用 `datamate_list_operators` 查看可用算子")

    return "\n".join(lines)


# ============================================================
# 工具4：一键ETL流水线
# ============================================================


@mcp.tool(
    name="datamate_run_etl_pipeline",
    annotations={
        "title": "一键执行完整ETL流程",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def datamate_run_etl_pipeline(params: RunETLPipelineInput) -> str:
    """一键执行完整ETL数据处理流程：加载→清洗→转换→导出。

    这是推荐使用的ETL工具！只需调用一次即可完成全部4步处理。
    支持自定义每步的参数，返回每一步的详细结果。

    Args:
        params (RunETLPipelineInput): 包含:
            - csv_text (str): CSV文本内容（必填，从read_file读取）
            - cleaner_params (Optional[dict]): 清洗参数（可选）
              默认: {"remove_duplicates": true, "fill_na": true}
            - transformer_params (Optional[dict]): 转换参数（可选）
              默认: {"standardize_types": true}
            - exporter_params (Optional[dict]): 导出参数（可选）
              关键参数: {"format": "csv", "outputDir": "/mnt/nexent/output"}
              注意参数名是 outputDir（驼峰），不传则返回CSV文本内容

    Returns:
        str: 每一步的执行结果JSON，包含data和report

    Examples:
        - 完整流程: {"csv_text": "name,age\\n张三,25", "cleaner_params": {"remove_duplicates": true}, "exporter_params": {"format": "csv", "outputDir": "/mnt/nexent/output"}}
        - 仅加载+清洗: {"csv_text": "..."}（不传exporter_params会返回CSV文本）
    """
    pipeline_start = time.perf_counter()
    temp_file = None

    try:
        # Step 1: data_loader - 写入临时文件加载
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        )
        csv_text = params.csv_text
        if not csv_text.endswith("\n"):
            csv_text += "\n"
        temp_file.write(csv_text)
        temp_file.flush()
        temp_file.close()

        cls_loader = _import_operator_class("data_loader", "DataLoaderMapper")
        loader = cls_loader(file_paths=[temp_file.name])
        load_result = loader.execute({"data": [], "count": 0})
        if not load_result.get("execute_result", False):
            return json.dumps({
                "success": False, "step": "data_loader",
                "error": load_result.get("error", "未知错误"),
            }, ensure_ascii=False)

        data = load_result.get("data", [])
        data = _sanitize_data(data)
        step1 = {
            "step": "data_loader",
            "success": True,
            "rows": len(data),
            "columns": list(data[0].keys()) if data else [],
        }

        # Step 2: data_cleaner
        cls_cleaner = _import_operator_class("data_cleaner", "DataCleaner")
        cleaner = cls_cleaner()
        clean_params = _ensure_camel_case(params.cleaner_params) or {
            "removeDuplicates": True,
            "handleMissing": "drop",
            "trimWhitespace": True,
            "standardizeFormat": True,
            "privacyCheck": True,
            "privacyFields": ["name", "phone", "id_card"],
        }
        clean_result = cleaner.process(data, clean_params)
        clean_data = clean_result.get("data", data)
        clean_data = _sanitize_data(clean_data)
        step2 = {
            "step": "data_cleaner",
            "success": True,
            "rows_before": len(data),
            "rows_after": len(clean_data),
            "report": clean_result.get("report", {}),
        }

        # Step 3: data_transformer
        cls_transformer = _import_operator_class("data_transformer", "DataTransformer")
        transformer = cls_transformer()
        trans_params = _ensure_camel_case(params.transformer_params) or {
            "renameFields": '{"name": "patient_name"}',
            "typeConversions": '{"age": "int", "weight": "float", "height": "float"}',
            "valueMappings": '{"gender": {"M": "男", "F": "女"}}',
            "deriveColumns": '{"bmi": "weight/((height/100)**2)"}',
        }
        trans_result = transformer.process(clean_data, trans_params)
        trans_data = trans_result.get("data", clean_data)
        trans_data = _sanitize_data(trans_data)
        step3 = {
            "step": "data_transformer",
            "success": True,
            "rows": len(trans_data),
            "report": trans_result.get("report", {}),
        }

        # Step 4: data_exporter - 始终生成CSV文本，可选写入文件
        import io as io_module
        import csv as csv_module
        csv_output = ""
        if trans_data:
            output = io_module.StringIO()
            writer = csv_module.DictWriter(output, fieldnames=list(trans_data[0].keys()))
            writer.writeheader()
            writer.writerows(trans_data)
            csv_output = output.getvalue()

        step4 = {
            "step": "data_exporter",
            "success": True,
            "output_text": csv_output,
            "output_path": None,
        }

        # 如果指定了输出路径，尝试写入文件
        if params.exporter_params:
            export_params = _ensure_camel_case(params.exporter_params)
            # 如果 outputDir 指向的是文件路径，提取目录部分
            od = export_params.get("outputDir", "")
            if od and os.path.basename(od):
                export_params["outputDir"] = os.path.dirname(od)

            # 尝试用出口器写入文件
            try:
                cls_exporter = _import_operator_class("data_exporter", "DataExporter")
                exporter = cls_exporter()
                export_result = exporter.process(trans_data, export_params)
                export_path = (
                    export_result.get("export_summary", {}).get("output_path")
                    or export_result.get("output_path", "")
                )
                if export_path:
                    step4["output_path"] = export_path
                    step4["report"] = export_result.get("report", {})
                    # 读取文件内容验证
                    try:
                        with open(export_path, "r", encoding="utf-8") as f:
                            step4["output_text"] = f.read()
                    except Exception:
                        pass
            except Exception as e:
                step4["write_warning"] = f"文件写入失败（路径问题），但CSV文本已返回: {str(e)}"

        # 如果导出器没生成文件，但有CSV文本和outputDir，尝试用docker exec写入容器
        if not step4.get("output_path") and csv_output and params.exporter_params:
            export_params = _ensure_camel_case(params.exporter_params)
            od = export_params.get("outputDir", "")
            if od:
                # 生成文件名
                safe_fn = f"medical_data_cleaned.csv"
                container_path = f"{od.rstrip('/')}/{safe_fn}"
                try:
                    proc = subprocess.run(
                        ["docker", "exec", "-i", "nexent-config", "sh", "-c",
                         f"cat > '{container_path}'"],
                        input=csv_output,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if proc.returncode == 0:
                        step4["output_path"] = container_path
                        step4.pop("write_warning", None)
                except Exception:
                    pass

        pipeline_elapsed = round((time.perf_counter() - pipeline_start) * 1000, 2)

        result = {
            "success": True,
            "pipeline_time_ms": pipeline_elapsed,
            "steps": [step1, step2, step3, step4],
            "summary": {
                "total_steps": 4,
                "final_rows": len(trans_data),
                "final_columns": list(trans_data[0].keys()) if trans_data else [],
            },
        }
        return json.dumps(_sanitize_data(result), indent=2, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "pipeline_time_ms": round((time.perf_counter() - pipeline_start) * 1000, 2),
        }, ensure_ascii=False)

    finally:
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except (OSError, PermissionError):
                pass


# ============================================================
# 工具5：医疗实体识别（知识图谱专用）
# ============================================================


@mcp.tool(
    name="knowledge_graph_search",
    annotations={
        "title": "医疗知识图谱搜索",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def knowledge_graph_search(params: KGEntityRecognitionInput) -> str:
    """从医疗文本中识别知识图谱实体（疾病、症状、药物、检查）。

    这是知识图谱问答的第一步。输入一段医疗文本，返回识别出的所有医疗实体。
    支持4类实体：疾病、症状、药物、检查。

    使用场景:
    - 用户问"发热可能是什么原因？" → 识别出实体"发热"（症状）
    - 用户问"糖尿病怎么治疗？" → 识别出实体"糖尿病"（疾病）
    - 用户问"阿莫西林能治什么？" → 识别出实体"阿莫西林"（药物）

    Args:
        params (KGEntityRecognitionInput): 包含:
            - text (str): 待识别的医疗文本（必填）

    Returns:
        str: JSON格式的实体识别结果，包含entities列表和entity_count
    """
    start = time.perf_counter()
    try:
        result = _run_operator_locally("kg_entity_recognizer", {"text": params.text}, {})
        elapsed = round((time.perf_counter() - start) * 1000, 2)

        if result["success"]:
            op_data = result.get("result", {})
            if isinstance(op_data, dict) and "entities" in op_data:
                op_data["elapsed_ms"] = elapsed
                op_data["tool"] = "knowledge_graph_search"
                return json.dumps(op_data, indent=2, ensure_ascii=False)
            return json.dumps({
                "success": True,
                "entities": [],
                "entity_count": 0,
                "elapsed_ms": elapsed,
                "tool": "knowledge_graph_search",
            }, indent=2, ensure_ascii=False)
        else:
            return json.dumps({
                "success": False,
                "error": result.get("error", "未知错误"),
                "elapsed_ms": elapsed,
                "tool": "knowledge_graph_search",
            }, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "tool": "knowledge_graph_search",
        }, indent=2, ensure_ascii=False)


# ============================================================
# 工具6：医疗关系抽取（知识图谱专用）
# ============================================================


@mcp.tool(
    name="entity_recognition",
    annotations={
        "title": "医疗实体关系抽取",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def entity_recognition(params: KGRelationExtractionInput) -> str:
    """抽取医疗实体之间的关系（导致、治疗、用于、禁忌）。

    这是知识图谱问答的第二步。基于已识别的实体，抽取它们之间的语义关系。
    支持4类关系：导致(疾病→症状)、治疗(药物→疾病)、用于(药物→症状)、禁忌(药物→疾病)。

    使用场景:
    - 已识别"糖尿病"和"二甲双胍" → 抽取关系：二甲双胍→治疗→糖尿病
    - 已识别"高血压"和"头痛" → 抽取关系：高血压→导致→头痛
    - 不传entities参数时会自动先调用实体识别器

    Args:
        params (KGRelationExtractionInput): 包含:
            - text (str): 待抽取关系的医疗文本（必填）
            - entities (Optional[list]): 已识别的实体列表（可选，不传则自动识别）

    Returns:
        str: JSON格式的关系抽取结果，包含relations列表和relation_count
    """
    start = time.perf_counter()
    try:
        # 构建输入数据
        input_data = {"text": params.text}
        if params.entities:
            input_data["entities"] = params.entities

        result = _run_operator_locally("kg_relation_extractor", input_data, {})
        elapsed = round((time.perf_counter() - start) * 1000, 2)

        if result["success"]:
            op_data = result.get("result", {})
            if isinstance(op_data, dict) and "relations" in op_data:
                op_data["elapsed_ms"] = elapsed
                op_data["tool"] = "entity_recognition"
                return json.dumps(op_data, indent=2, ensure_ascii=False)
            return json.dumps({
                "success": True,
                "relations": [],
                "relation_count": 0,
                "elapsed_ms": elapsed,
                "tool": "entity_recognition",
            }, indent=2, ensure_ascii=False)
        else:
            return json.dumps({
                "success": False,
                "error": result.get("error", "未知错误"),
                "elapsed_ms": elapsed,
                "tool": "entity_recognition",
            }, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "tool": "entity_recognition",
        }, indent=2, ensure_ascii=False)


# ============================================================
# 工具7：KG三元组生成（知识图谱专用）
# ============================================================


@mcp.tool(
    name="relation_extraction",
    annotations={
        "title": "知识图谱三元组生成",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def relation_extraction(params: KGTripleGenerationInput) -> str:
    """基于实体和关系生成标准化的知识图谱三元组，含冲突检测。

    这是知识图谱问答的第三步。将实体和关系组合为标准化KG三元组，
    自动执行去重和冲突检测（直接冲突+禁忌冲突）。

    使用场景:
    - 已有entities和relations → 生成标准化三元组
    - 自动检查是否存在矛盾（如同一药物同时标注治疗和禁忌）
    - 输出可直接存入知识图谱的结构化三元组

    Args:
        params (KGTripleGenerationInput): 包含:
            - entities (list): 实体列表（必填）
            - relations (list): 关系列表（必填）
            - text (Optional[str]): 原始文本（可选，用于冲突上下文）

    Returns:
        str: JSON格式的三元组生成结果，包含triples、conflicts、warnings
    """
    start = time.perf_counter()
    try:
        # 构建输入数据
        input_data = {
            "entities": params.entities,
            "relations": params.relations,
        }
        if params.text:
            input_data["text"] = params.text

        result = _run_operator_locally("kg_triple_generator", input_data, {})
        elapsed = round((time.perf_counter() - start) * 1000, 2)

        if result["success"]:
            op_data = result.get("result", {})
            if isinstance(op_data, dict) and "triples" in op_data:
                op_data["elapsed_ms"] = elapsed
                op_data["tool"] = "relation_extraction"
                return json.dumps(op_data, indent=2, ensure_ascii=False)
            return json.dumps({
                "success": True,
                "triples": [],
                "triple_count": 0,
                "conflicts": [],
                "conflict_count": 0,
                "elapsed_ms": elapsed,
                "tool": "relation_extraction",
            }, indent=2, ensure_ascii=False)
        else:
            return json.dumps({
                "success": False,
                "error": result.get("error", "未知错误"),
                "elapsed_ms": elapsed,
                "tool": "relation_extraction",
            }, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "tool": "relation_extraction",
        }, indent=2, ensure_ascii=False)


# ============================================================
# 任务三新增：kg_analyze — 图谱统计分析
# ============================================================

import os as _os2

# 加载三元组数据
_KG_DATA_DIR = _os2.path.join(_os2.path.dirname(__file__), "kg_data")
_KG_TRIPLES_CACHE = None


def _load_kg_data():
    """加载知识图谱三元组数据（首次调用时加载，后续用缓存）"""
    global _KG_TRIPLES_CACHE
    if _KG_TRIPLES_CACHE is not None:
        return _KG_TRIPLES_CACHE
    path = _os2.path.join(_KG_DATA_DIR, "knowledge_graph_triples.json")
    if not _os2.path.exists(path):
        _KG_TRIPLES_CACHE = []
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _KG_TRIPLES_CACHE = data.get("triples", [])
    return _KG_TRIPLES_CACHE


class KgAnalyzeInput(BaseModel):
    entity: Optional[str] = Field(
        default=None,
        description="要分析的实体名称（如'糖尿病'），返回涉及该实体的统计结果"
    )
    relation: Optional[str] = Field(
        default=None,
        description="按关系类型筛选（导致/治疗/用于/禁忌）"
    )
    analysis_type: Optional[str] = Field(
        default="关系分布",
        description="分析类型：关系分布（默认）/ 症状分布 / 药物统计 / 关联分析"
    )
    keyword: Optional[str] = Field(
        default=None,
        description="关键词搜索（head/tail中模糊匹配，不区分大小写）"
    )
    top_k: int = Field(
        default=20,
        description="最多返回的统计条数",
        ge=1, le=100
    )


# ---------- Layer 1: 基础设施 ----------

def _filter_triples(triples, entity=None, relation=None, keyword=None):
    """筛选三元组，支持entity/relation/keyword组合条件"""
    results = []
    for t in triples:
        # entity匹配方向：治疗关系中entity(疾病)在tail，其他关系中在head
        if entity:
            if relation == "治疗":
                if t["tail"] != entity:
                    continue
            else:
                if t["head"] != entity:
                    continue
        if relation and t["relation"] != relation:
            continue
        if keyword:
            kw = keyword.lower()
            if kw not in t["head"].lower() and kw not in t["tail"].lower():
                continue
        results.append(t)
    return results


def _count_and_rank(items, top_k):
    """计数+排序+计算百分比"""
    from collections import Counter
    counter = Counter(items)
    total = sum(counter.values())
    ranked = counter.most_common(top_k)
    return [
        {
            "item": item,
            "count": count,
            "percentage": round(count / total * 100, 1) if total > 0 else 0.0
        }
        for item, count in ranked
    ]


def _error_json(message, tool="kg_analyze"):
    """统一的错误返回格式"""
    return json.dumps({
        "success": False,
        "error": message,
        "tool": tool,
    }, indent=2, ensure_ascii=False)


# ---------- Layer 2: 统计处理器 ----------

def _handle_relation_dist(triples, entity, top_k):
    """关系分布：统计各种关系类型的数量"""
    items = [t["relation"] for t in triples]
    return _count_and_rank(items, top_k)


def _handle_symptom_dist(triples, entity, top_k):
    """症状分布：entity→症状的分布（relation已强制为'导致'）"""
    items = [t["tail"] for t in triples]
    return _count_and_rank(items, top_k)


def _handle_drug_stats(triples, entity, top_k):
    """药物统计：治疗entity的药物（relation已强制为'治疗'）"""
    items = [t["head"] for t in triples]
    return _count_and_rank(items, top_k)


def _handle_entity_relations(triples, entity, top_k):
    """关联分析：与entity关联的所有实体"""
    items = []
    for t in triples:
        if t["head"] == entity:
            items.append(t["tail"])
        else:
            items.append(t["head"])
    return _count_and_rank(items, top_k)


# ---------- Layer 3: 处理器映射 ----------

ANALYSIS_HANDLERS = {
    "关系分布": _handle_relation_dist,
    "症状分布": _handle_symptom_dist,
    "药物统计": _handle_drug_stats,
    "关联分析": _handle_entity_relations,
}


# ---------- Layer 4: MCP工具入口 ----------

@mcp.tool(
    name="kg_analyze",
    annotations={
        "title": "图谱统计分析",
        "readOnlyHint": True,
        "destructiveHint": False,
        "description": "对知识图谱三元组做统计分析。输入实体名称或关键词，返回统计结果（计数/百分比）。"
    }
)
async def kg_analyze(params: KgAnalyzeInput) -> str:
    """
    图谱统计分析工具，支持4种analysis_type：
    - 关系分布：统计各种关系类型的分布
    - 症状分布：统计entity的所有症状
    - 药物统计：统计治疗entity的药物
    - 关联分析：统计与entity关联的所有实体
    """
    start = time.perf_counter()
    warning_parts = []

    try:
        triples = _load_kg_data()
        entity = params.entity
        relation = params.relation
        analysis_type = params.analysis_type or "关系分布"
        keyword = params.keyword
        top_k = params.top_k

        # Step 0: entity必填检查
        if analysis_type in ["症状分布", "药物统计", "关联分析"] and not entity:
            return _error_json(f"analysis_type='{analysis_type}'需要参数entity，但entity为空")

        # Step 1: 参数覆盖逻辑
        if analysis_type == "关系分布":
            if relation is not None:
                warning_parts.append(f"已忽略relation={relation}，关系分布不受relation筛选影响")
                relation = None

        elif analysis_type == "症状分布":
            if relation is not None and relation != "导致":
                warning_parts.append(f"已忽略relation={relation}，症状分布自动使用relation=导致")
            relation = "导致"

        elif analysis_type == "药物统计":
            if relation is not None and relation != "治疗":
                warning_parts.append(f"已忽略relation={relation}，药物统计自动使用relation=治疗")
            relation = "治疗"

        # 关联分析：保留用户传入的relation

        # Step 2: 筛选三元组
        filtered = _filter_triples(triples, entity, relation, keyword)

        # Step 3: 按analysis_type分发到处理器
        handler = ANALYSIS_HANDLERS.get(analysis_type)
        if handler is None:
            warning_parts.append(f"未知analysis_type='{analysis_type}'，已回退到关系分布")
            handler = _handle_relation_dist
            analysis_type = "关系分布"

        stats = handler(filtered, entity, top_k)
        elapsed = round((time.perf_counter() - start) * 1000, 2)

        result = {
            "success": True,
            "entity": entity,
            "analysis_type": analysis_type,
            "total_triples": len(filtered),
            "stats": stats,
            "elapsed_ms": elapsed,
            "tool": "kg_analyze",
        }
        if warning_parts:
            result["warning"] = "; ".join(warning_parts)

        return json.dumps(result, indent=2, ensure_ascii=False)

    except Exception as e:
        return _error_json(f"{type(e).__name__}: {str(e)}")


# ============================================================
# 任务三新增：kg_visualize — 图谱图表可视化
# ============================================================

# ---------- Layer 0: 共享安全工具 ----------

def _escape_html(s: str) -> str:
    """HTML上下文转义（用于<title>、<div>等）"""
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    s = s.replace('"', "&quot;")
    s = s.replace("'", "&#x27;")
    return s


def _escape_js_visualize(s: str) -> str:
    """JS字符串上下文转义（用于ECharts option内部）"""
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\\", "\\\\")
    s = s.replace("'", "\\'")
    s = s.replace('"', '\\"')
    s = s.replace("\n", "\\n")
    s = s.replace("\r", "\\r")
    s = s.replace("/", "\\/")
    return s


_W_SIZE_RE = re.compile(r'^\d+(%|px|vw|vh)?$')
_VALID_THEMES = {"light", "dark"}

def _validate_size(val: str) -> str:
    """尺寸白名单验证"""
    if not _W_SIZE_RE.match(val):
        raise ValueError(f"无效尺寸值: {val}，示例: 100%, 800px, 80vw")
    return val

def _validate_theme(theme: str) -> str:
    """主题枚举验证"""
    if theme not in _VALID_THEMES:
        raise ValueError(f"无效主题: {theme}，可选: {_VALID_THEMES}")
    return theme


class KgVisualizeInput(BaseModel):
    data: Optional[list] = Field(
        default=None,
        description="统计数据[{item, count, percentage}]，bar/pie/line使用"
    )
    triples: Optional[list] = Field(
        default=None,
        description="三元组[{head, relation, tail}]，仅graph使用"
    )
    chart_type: str = Field(
        default="bar",
        description="图表类型：bar / pie / graph / line"
    )
    entity: Optional[str] = Field(
        default=None,
        description="中心实体名称，仅graph必填"
    )
    title: str = Field(
        default="知识图谱分析",
        description="图表标题"
    )
    width: str = Field(
        default="100%",
        description="宽度（如100%、800px、80vw）"
    )
    height: str = Field(
        default="500px",
        description="高度（如500px、80vh）"
    )
    theme: str = Field(
        default="light",
        description="主题：light / dark"
    )


# ---------- Layer 1: 数据转换 + ECharts option 构建 ----------

def _stats_to_labels_values(data: list) -> tuple:
    """将 [{item, count, percentage}] 转为 ([labels], [values])"""
    labels = []
    values = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = item.get("item", "")
        cnt = item.get("count", 0)
        # 跳过异常数据
        if not name or name.strip() == "":
            continue
        if cnt is None or (isinstance(cnt, (int, float)) and cnt < 0):
            continue
        labels.append(str(name))
        values.append(int(cnt) if isinstance(cnt, (int, float)) else 0)
    return labels, values


def _build_option_json(chart_type: str, labels: list, values: list) -> str:
    """生成 bar/pie/line 共用的 ECharts option JSON"""
    labels_js = "[" + ",".join(f"'{_escape_js_visualize(l)}'" for l in labels) + "]"
    values_js = "[" + ",".join(str(v) for v in values) + "]"

    if chart_type == "pie":
        data_js = "[" + ",".join(
            f"{{name:'{_escape_js_visualize(l)}',value:{v}}}"
            for l, v in zip(labels, values)
        ) + "]"
        return (
            f'{{"tooltip":{{"trigger":"item"}},'
            f'"series":[{{"type":"pie","data":{data_js}}}]}}'
        )
    else:
        return (
            f'{{"tooltip":{{"trigger":"axis"}},'
            f'"xAxis":{{"type":"category","data":{labels_js}}},'
            f'"yAxis":{{"type":"value"}},'
            f'"series":[{{"type":"{chart_type}","data":{values_js}}}]}}'
        )


def _build_graph_option_json(entity: str, triples: list) -> str:
    """生成 graph 图表的 ECharts option JSON"""
    # 动态提取节点和类型
    nodes = {}
    for t in triples:
        head = t.get("head", "")
        tail = t.get("tail", "")
        if head:
            nodes[head] = t.get("head_type", t.get("type", "实体"))
        if tail:
            nodes[tail] = t.get("tail_type", "实体")

    data_js = "[" + ",".join(
        f"{{name:'{_escape_js_visualize(n)}',category:'{_escape_js_visualize(c)}'}}"
        for n, c in nodes.items()
    ) + "]"

    links_js = "[" + ",".join(
        f"{{source:'{_escape_js_visualize(t.get('head',''))}',"
        f"target:'{_escape_js_visualize(t.get('tail',''))}',"
        f"label:{{show:true,formatter:'{_escape_js_visualize(t.get('relation',''))}'}}}}"
        for t in triples
    ) + "]"

    return (
        f'{{"tooltip":{{}},"series":[{{'
        f'"type":"graph","layout":"force","roam":true,'
        f'"data":{data_js},"links":{links_js},'
        f'"force":{{"repulsion":300,"gravity":0.1,"edgeLength":150}}'
        f'}}]}}'
    )


def _assemble_html(title: str, option: str, theme: str,
                   width: str = "100%", height: str = "500px") -> str:
    """拼装完整HTML，所有动态值都经过转义或验证"""
    title_html = _escape_html(title)
    w = _validate_size(width)
    h = _validate_size(height)
    t = _validate_theme(theme)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>{title_html}</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"
    onerror="document.getElementById('chart').innerHTML=
    '&lt;p style=\\"text-align:center;padding:40px;color:#c00\\"&gt;⚠️ ECharts加载失败，请检查网络连接&lt;/p&gt;'">
  </script>
  <style>
    body {{ margin:0; padding:20px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
    body.dark {{ background:#1a1a2e; color:#eee; }}
    .dark .title {{ color:#eee; }}
    .chart-container {{ width:{w}; height:{h}; }}
    .title {{ font-size:18px; font-weight:bold; margin-bottom:16px; color:#333; }}
  </style>
</head>
<body class="{t}">
  <div class="title">{title_html}</div>
  <div id="chart" class="chart-container"></div>
  <script>
    try {{
      var chart = echarts.init(document.getElementById('chart'), '{t}');
      var option = {option};
      chart.setOption(option);
      window.addEventListener('resize', function() {{ chart.resize(); }});
    }} catch(e) {{
      document.getElementById('chart').innerHTML =
        '<p style="text-align:center;padding:40px;color:#c00">⚠️ 图表渲染失败: ' + e.message + '</p>';
    }}
  </script>
</body>
</html>"""


# ---------- Layer 2: 图表构建函数（统一签名）----------

def _build_bar_html(data: list, title: str, theme: str,
                    width: str = "100%", height: str = "500px") -> str:
    labels, values = _stats_to_labels_values(data)
    option = _build_option_json("bar", labels, values)
    return _assemble_html(title, option, theme, width, height)


def _build_pie_html(data: list, title: str, theme: str,
                    width: str = "100%", height: str = "500px") -> str:
    labels, values = _stats_to_labels_values(data)
    option = _build_option_json("pie", labels, values)
    return _assemble_html(title, option, theme, width, height)


def _build_line_html(data: list, title: str, theme: str,
                     width: str = "100%", height: str = "500px") -> str:
    labels, values = _stats_to_labels_values(data)
    option = _build_option_json("line", labels, values)
    return _assemble_html(title, option, theme, width, height)


def _build_graph_html(data: list, title: str, theme: str,
                      width: str = "100%", height: str = "500px") -> str:
    entity = data.get("entity", "") if isinstance(data, dict) else ""
    triples = data.get("triples", []) if isinstance(data, dict) else []
    option = _build_graph_option_json(entity, triples)
    return _assemble_html(title, option, theme, width, height)


CHART_BUILDERS = {
    "bar": _build_bar_html,
    "pie": _build_pie_html,
    "graph": _build_graph_html,
    "line": _build_line_html,
}


# ---------- Layer 3: MCP公开接口 ----------

@mcp.tool(
    name="kg_visualize",
    annotations={
        "title": "图谱图表可视化",
        "readOnlyHint": True,
        "destructiveHint": False,
        "description": "将知识图谱统计数据渲染为HTML图表。支持bar(柱状图)/pie(饼图)/graph(关系图)/line(折线图)。"
    }
)
async def kg_visualize(params: KgVisualizeInput) -> str:
    start = time.perf_counter()

    try:
        # Step 1: 白名单验证
        _validate_theme(params.theme)
        _validate_size(params.width)
        _validate_size(params.height)

        # Step 2: 参数路由
        if params.chart_type == "graph":
            if not params.entity or not params.triples:
                return _error_json("graph图表需要entity和triples参数",
                                   tool="kg_visualize")
            if len(params.triples) == 0:
                return _error_json("triples为空，无法生成关系图",
                                   tool="kg_visualize")
            data = {"entity": params.entity, "triples": params.triples}
            data_count = len(params.triples)
        else:
            if not params.data or len(params.data) == 0:
                return _error_json(
                    f"{params.chart_type}图表需要data参数，但data为空",
                    tool="kg_visualize"
                )
            data = params.data
            data_count = len(data)
            # 截断保护
            if len(data) > 50:
                data = data[:50]
                warning_val = f"数据{data_count}条过多，截取前50条展示"

        # Step 3: 校验图表类型+分发
        chart_type = params.chart_type
        if chart_type not in CHART_BUILDERS:
            chart_type = "bar"
            warning_val = (
                f"未知chart_type='{_escape_html(params.chart_type)}'，已回退到bar"
            )

        build_func = CHART_BUILDERS[chart_type]
        if chart_type == "graph":
            html = build_func(data, params.title, params.theme,
                              params.width, params.height)
        else:
            html = build_func(data, params.title, params.theme,
                              params.width, params.height)

        # HTML非空检查
        if not html or html.strip() == "":
            return _error_json("HTML生成失败，输出为空", tool="kg_visualize")

        # 保存HTML文件到 /mnt/nexent/charts/
        import hashlib
        _WORK_DIR = "/mnt/nexent/charts"
        os.makedirs(_WORK_DIR, exist_ok=True)
        chart_name = f"{chart_type}_{params.title.replace(' ','_')[:30]}_{hashlib.md5(html.encode()).hexdigest()[:6]}"
        file_path = os.path.join(_WORK_DIR, f"{chart_name}.html")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html)

        elapsed = round((time.perf_counter() - start) * 1000, 2)

        result = {
            "success": True,
            "html": html,
            "file_path": file_path,
            "chart_type": chart_type,
            "title": params.title,
            "data_count": data_count,
            "elapsed_ms": elapsed,
            "tool": "kg_visualize",
        }
        if 'warning_val' in dir():
            result["warning"] = warning_val

        return json.dumps(result, indent=2, ensure_ascii=False)

    except ValueError as e:
        return _error_json(str(e), tool="kg_visualize")
    except Exception as e:
        return _error_json(f"{type(e).__name__}: {str(e)}",
                           tool="kg_visualize")


# ============================================================
# 任务三新增：kg_report — 分析报告生成
# ============================================================

from dataclasses import dataclass, field

# ---------- Insight 数据结构（v2.1核心设计）----------

@dataclass
class Insight:
    rule_id: int
    rule_name: str
    title: str
    detail: str
    severity: str           # "highlight" / "info" / "warning"
    source_index: int
    confidence: float       # 0-1
    evidence: list = field(default_factory=list)
    limitation: str = ""


# ---------- Pydantic 输入模型 ----------

class ChartItem(BaseModel):
    html: str = Field(description="图表HTML代码")
    title: Optional[str] = Field(default=None, description="图表标题")
    source_index: int = Field(default=0, description="对应all_results的索引")


class AnalysisResult(BaseModel):
    entity: str = Field(description="分析实体名称")
    analysis_type: str = Field(
        description="分析类型：关系分布/症状分布/药物统计/关联分析"
    )
    total_triples: int = Field(default=0)
    stats: list = Field(default=[], description="[{item, count, percentage}]")


class KgReportInput(BaseModel):
    all_results: list = Field(
        default_factory=list,
        description="kg_analyze完整输出列表"
    )
    charts: Optional[list] = Field(
        default=None,
        description="ChartItem列表"
    )
    query: Optional[str] = Field(default=None, description="用户原始提问")
    report_type: str = Field(default="full", description="full/summary/technical")
    max_length: int = Field(default=15000, ge=1000, le=50000)
    max_insights: int = Field(default=8, ge=0, le=20)


# ---------- L0: 共享工具层 ----------

TYPE_LABELS = {
    "关系分布": "关系类型", "症状分布": "症状",
    "药物统计": "治疗药物", "关联分析": "关联实体",
}


def _md_escape(text: str) -> str:
    """Markdown特殊字符转义"""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("|", "&#124;")
    text = text.replace("`", "\u200b`")
    text = text.replace("_", "\\_")
    text = text.replace("*", "\\*")
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    lines = text.split("\n")
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped and stripped[0] in ("#", "-", "*", ">"):
            lines[i] = "\\" + line
    return "\n".join(lines)


def _validate_analysis(results: list) -> tuple:
    """校验analysis_result，返回 (valid_count, warnings)"""
    warnings = []
    for i, r in enumerate(results):
        if not isinstance(r, dict):
            warnings.append(f"result[{i}]不是字典，已跳过")
            continue
        if "entity" not in r:
            warnings.append(f"result[{i}]缺少entity字段")
        if "analysis_type" not in r:
            warnings.append(f"result[{i}]缺少analysis_type字段")
        if "stats" not in r:
            warnings.append(f"result[{i}]缺少stats字段")
    return len(results), warnings


def _build_md_table(rows: list, headers: list) -> str:
    """生成Markdown表格（列数补齐+转义）"""
    col_count = len(headers)
    safe_rows = []
    for r in rows:
        r = list(r[:col_count]) + [""] * max(0, col_count - len(r))
        safe_rows.append([_md_escape(str(c)) for c in r])
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join([":---"] * col_count) + "|")
    for row in safe_rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _format_timestamp() -> str:
    """当前时间格式化"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _generate_chart_comment(stats: list, analysis_type: str) -> str:
    """为图表生成图文结合说明"""
    if not stats or len(stats) < 2:
        return ""
    top1, top2 = stats[0], stats[1]
    label = TYPE_LABELS.get(analysis_type, "条目")
    divisor = max(top2.get("count", 1), 1)
    ratio = round(top1.get("count", 0) / divisor, 1)
    return (
        f"> 📊 从上图可见，**{_md_escape(str(top1.get('item','')))}** "
        f"是最常见的{label}（{top1.get('percentage',0)}%），"
        f"约为第二名 **{_md_escape(str(top2.get('item','')))}** "
        f"（{top2.get('percentage',0)}%）的 {ratio} 倍。"
    )


def _safe_truncate(text: str, max_len: int) -> str:
    """段落边界安全截断"""
    if len(text) <= max_len:
        return text
    # 在最后一个完整段落处截断
    cutoff = text.rfind("\n\n", 0, max_len)
    if cutoff < max_len // 2:
        cutoff = text.rfind("\n", 0, max_len)
    if cutoff < max_len // 2:
        cutoff = max_len
    return text[:cutoff] + "\n\n*[报告过长，已截断]*"


# ---------- L1: 报告引擎层 ----------

def _build_header(entity: str, query: str, result_count: int) -> str:
    ts = _format_timestamp()
    q = _md_escape(query or "无")
    return (
        f"# {_md_escape(entity)}知识图谱分析报告\n\n"
        f"> 分析时间：{ts}\n"
        f"> 数据来源：医疗知识图谱（任务二产出）\n"
        f"> 分析维度：{result_count} 项\n"
        f"> 用户提问：{q}\n"
    )


def _build_overview(results: list) -> str:
    total = sum(r.get("total_triples", 0) for r in results)
    types = [_md_escape(r.get("analysis_type", "未知")) for r in results]
    return (
        f"## 一、概述\n\n"
        f"知识图谱中共关联 **{total}** 条三元组。\n"
        f"本报告整合了 {len(results)} 项统计分析，涵盖：{'、'.join(types)}。\n"
    )


def _build_analysis_section(idx: int, r: dict) -> str:
    atype = r.get("analysis_type", "统计")
    label = TYPE_LABELS.get(atype, "条目")
    entity = _md_escape(r.get("entity", "未知"))
    title = f"{entity}的{label}" if atype != "关系分布" else "关系类型分布"
    stats = r.get("stats", [])
    total = r.get("total_triples", len(stats))

    md = f"### 2.{idx} {title}\n\n"
    if not stats:
        return md + "*暂无统计数据*\n"
    safe_stats = []
    for s in stats:
        if not isinstance(s, dict):
            continue
        if not s.get("item") or s["item"] == "":
            continue
        cnt = s.get("count")
        if cnt is None or (isinstance(cnt, (int, float)) and cnt < 0):
            continue
        pct = s.get("percentage", 0)
        if isinstance(pct, (int, float)) and (pct < 0 or pct > 100):
            continue
        safe_stats.append(s)

    rows = [
        [s.get("item", ""), str(s.get("count", "")), f"{s.get('percentage',0)}%"]
        for s in safe_stats
    ]
    if rows:
        md += _build_md_table(rows, ["名称", "计数", "占比"])
        md += "\n\n"
    else:
        md += "*数据异常，无法统计*\n"
    return md


def _build_chart_section(chart: dict, r: dict) -> str:
    """图表嵌入节（含图文结合说明）"""
    atype = r.get("analysis_type", "统计")
    comment = _generate_chart_comment(r.get("stats", []), atype)
    title = chart.get("title", "") if isinstance(chart, dict) else ""
    md = f"### 图表：{_md_escape(title)}\n\n" if title else ""
    if isinstance(chart, dict) and chart.get("html"):
        md += chart["html"] + "\n\n"
    if comment:
        md += comment + "\n\n"
    return md


def _build_insights_section(insights: list) -> str:
    if not insights:
        return ""
    md = "## 四、关键洞察\n\n"
    md += "| # | 洞察 | 置信度 | 依据 |\n"
    md += "|:---:|:---|:---:|:---|\n"
    for i, ins in enumerate(insights):
        icon = {"highlight": "🔴", "info": "🔵", "warning": "🟡"}.get(
            ins.severity, "⚪"
        )
        conf_icon = "🟢" if ins.confidence >= 0.8 else (
            "🟡" if ins.confidence >= 0.6 else "🔴"
        )
        evidence = "; ".join(ins.evidence[:2]) if ins.evidence else "—"
        md += (
            f"| {i+1} | {icon} **{_md_escape(ins.title)}**<br>"
            f"{_md_escape(ins.detail[:80])}{'...' if len(ins.detail)>80 else ''} | "
            f"{conf_icon} {ins.confidence:.2f} | "
            f"{_md_escape(evidence)} |\n"
        )
    return md + "\n"


def _build_footer() -> str:
    return (
        "## 五、注意事项\n\n"
        "- 本报告基于知识图谱已有数据生成，可能存在数据覆盖不全的情况\n"
        "- 统计关系不构成医疗建议，请咨询专业医生\n\n"
        "---\n\n"
        "*报告由 Nexent 数据分析智能体自动生成*"
    )


def _build_raw_data_section(results: list) -> str:
    md = "## 附录：原始数据\n\n"
    for i, r in enumerate(results):
        atype = _md_escape(r.get("analysis_type", "未知"))
        md += f"### A.{i+1} {atype}\n\n"
        stats = r.get("stats", [])
        if stats:
            rows = [
                [s.get("item",""), str(s.get("count","")), f"{s.get('percentage',0)}%"]
                for s in stats if isinstance(s, dict) and s.get("item")
            ]
            if rows:
                md += _build_md_table(rows, ["条目", "计数", "占比"]) + "\n\n"
    return md


# ---------- 洞察规则引擎 ----------

def _score_confidence(rule_id: int, stats: list, total: int) -> float:
    base = 0.70
    if total >= 30:
        base += 0.10
    elif total < 5:
        base -= 0.30
    if rule_id == 1 and stats:
        pct = stats[0].get("percentage", 0)
        if pct > 50:
            base += 0.10
        elif pct < 40:
            base -= 0.05
    return round(min(max(base, 0.0), 1.0), 2)


def _generate_insights(results: list) -> list:
    """规则引擎：从分析结果生成洞察列表"""
    insights = []

    for idx, r in enumerate(results):
        if not isinstance(r, dict):
            continue
        stats = r.get("stats", [])
        total = r.get("total_triples", len(stats))
        atype = r.get("analysis_type", "")
        entity = r.get("entity", "")
        label = TYPE_LABELS.get(atype, "条目")

        if not stats:
            continue

        top1 = stats[0]
        p1 = top1.get("percentage", 0) if top1 else 0

        # 规则1：高集中度
        if p1 > 35:
            insights.append(Insight(
                rule_id=1, rule_name="高集中度",
                title=f"{top1['item']}是最突出的{label}",
                detail=f"{top1['item']}占比 {p1}%，远超其他{label}，"
                       f"是{entity}知识图谱中的核心{label}。",
                severity="highlight", source_index=idx,
                confidence=_score_confidence(1, stats, total),
                evidence=[f"{atype}/Top-1/{p1}%"],
                limitation=f"基于{total}条三元组统计",
            ))

        # 规则2：均衡分布
        if len(stats) >= 2:
            p2 = stats[1].get("percentage", 0)
            if p2 > 0 and (p1 - p2) < 5:
                insights.append(Insight(
                    rule_id=2, rule_name="均衡分布",
                    title=f"{top1['item']}与{stats[1]['item']}分布接近",
                    detail=f"二者占比分别为 {p1}% 和 {p2}%，差距仅 "
                           f"{round(p1-p2,1)}%，无明显主导项。",
                    severity="info", source_index=idx,
                    confidence=0.65,
                    evidence=[f"{atype}/Top-1/{p1}%", f"{atype}/Top-2/{p2}%"],
                    limitation="仅比较前两名",
                ))

        # 规则3：Top-3集中
        if len(stats) >= 3:
            top3_sum = sum(s.get("percentage", 0) for s in stats[:3])
            if top3_sum > 60:
                insights.append(Insight(
                    rule_id=3, rule_name="Top-3集中",
                    title=f"前三项合计占比 {round(top3_sum,1)}%",
                    detail=f"{stats[0]['item']}、{stats[1]['item']}、"
                           f"{stats[2]['item']}构成{label}的主要部分。",
                    severity="info", source_index=idx,
                    confidence=0.70,
                    evidence=[f"{atype}/Top-3/{round(top3_sum,1)}%"],
                    limitation="",
                ))

        # 规则4：长尾显著
        if len(stats) >= 3:
            top3_sum = sum(s.get("percentage", 0) for s in stats[:3])
            if top3_sum < 40:
                insights.append(Insight(
                    rule_id=4, rule_name="长尾显著",
                    title=f"{label}分布分散，长尾效应明显",
                    detail=f"前三项合计仅占 {round(top3_sum,1)}%，"
                           f"{label}呈现高度分散的长尾分布。",
                    severity="info", source_index=idx,
                    confidence=0.65,
                    evidence=[f"{atype}/Top-3/{round(top3_sum,1)}%"],
                    limitation="",
                ))

        # 规则5：禁忌警告
        if "禁忌" in atype:
            taboo_items = [s for s in stats if s.get("count", 0) > 0]
            if taboo_items:
                insights.append(Insight(
                    rule_id=5, rule_name="禁忌警告",
                    title=f"检测到与{entity}相关的禁忌关系",
                    detail=f"知识图谱中存在 {len(taboo_items)} 条禁忌，"
                           f"涉及：{'、'.join(s['item'] for s in taboo_items[:3])}。",
                    severity="warning", source_index=idx,
                    confidence=1.0,
                    evidence=[f"{atype}/{len(taboo_items)}条禁忌"],
                    limitation="禁忌关系来自图谱，需结合实际临床判断",
                ))

        # 规则6：数据稀疏
        if total < 5:
            insights.append(Insight(
                rule_id=6, rule_name="数据稀疏",
                title=f"{entity}数据稀疏（仅{total}条）",
                detail=f"仅{total}条三元组，分析结果置信度有限，"
                       f"建议结合其他数据源验证。",
                severity="warning", source_index=idx,
                confidence=0.95,
                evidence=[f"total_triples={total}"],
                limitation="",
            ))

        # 规则7：数据丰富
        if total > 30:
            insights.append(Insight(
                rule_id=7, rule_name="数据丰富",
                title=f"{entity}数据丰富（{total}条），分析可信度较高",
                detail=f"基于{total}条三元组的统计数据较为可靠。",
                severity="info", source_index=idx,
                confidence=0.85,
                evidence=[f"total_triples={total}"],
                limitation="数据量不代表数据质量",
            ))

    # 规则8：跨分析因果链
    by_type = {r.get("analysis_type"): r for r in results if isinstance(r, dict)}
    if "症状分布" in by_type and "药物统计" in by_type:
        sr = by_type["症状分布"]
        dr = by_type["药物统计"]
        ss = sr.get("stats", [])
        ds = dr.get("stats", [])
        if ss and ds:
            top_s = ss[0]["item"]
            top_d = ds[0]["item"]
            sentity = sr.get("entity", "")
            insights.append(Insight(
                rule_id=8, rule_name="跨分析因果链",
                title=f"{top_s}与{top_d}的治疗关联",
                detail=f"图谱中{top_s}是{sentity}最常见症状，{top_d}是最常用药物，"
                       f"二者形成「症状→治疗」因果链。",
                severity="highlight", source_index=0,
                confidence=0.75,
                evidence=["症状分布/Top-1", "药物统计/Top-1"],
                limitation="因果链基于共现统计，不构成因果关系",
            ))

    if "关系分布" in by_type:
        rr = by_type["关系分布"]
        has_taboo = any(
            s.get("item") == "禁忌" and s.get("count", 0) > 0
            for s in rr.get("stats", [])
        )
        if has_taboo and "药物统计" in by_type:
            insights.append(Insight(
                rule_id=8, rule_name="跨分析安全警告",
                title="禁忌关系与多药物共存",
                detail="知识图谱中同时存在禁忌关系和药物治疗，"
                       "建议进行药物相互作用检查。",
                severity="warning", source_index=0,
                confidence=0.90,
                evidence=["关系分布/禁忌", "药物统计"],
                limitation="需结合drug_interaction_check工具验证",
            ))

    return insights


def _resolve_conflicts(insights: list) -> list:
    """冲突消解：1↔2互斥（规则1优先），3↔4互斥（规则3优先）"""
    rule_ids = {i.rule_id for i in insights}
    if 1 in rule_ids and 2 in rule_ids:
        insights = [i for i in insights if i.rule_id != 2]
    if 3 in rule_ids and 4 in rule_ids:
        insights = [i for i in insights if i.rule_id != 4]
    priority = {"highlight": 0, "warning": 1, "info": 2}
    insights.sort(key=lambda i: priority.get(i.severity, 99))
    return insights


# ---------- 统一报告引擎 ----------

REPORT_CONFIGS = {
    "full": {
        "include_header": True,
        "include_overview": True,
        "include_analysis_table": True,
        "include_charts": True,
        "include_insights": "all",
        "include_footer": True,
        "include_raw_data": False,
    },
    "summary": {
        "include_header": True,
        "include_overview": False,
        "include_analysis_table": False,
        "include_charts": False,
        "include_insights": "top3",
        "include_footer": False,
        "include_raw_data": False,
        "summary_mode": True,
    },
    "technical": {
        "include_header": True,
        "include_overview": False,
        "include_analysis_table": False,
        "include_charts": False,
        "include_insights": "none",
        "include_footer": False,
        "include_raw_data": True,
    },
}


def _build_report(results: list, charts: list, query: str,
                  config: dict, max_length: int, max_insights: int) -> str:
    sections = []

    # 头部
    if config.get("include_header"):
        entity = results[0].get("entity", "未知") if results else "未知"
        sections.append(_build_header(
            entity, query, len(results)
        ))

    # 概述
    if config.get("include_overview"):
        sections.append(_build_overview(results))

    # 分析节+图表
    for i, r in enumerate(results):
        if not isinstance(r, dict):
            continue
        if config.get("include_analysis_table"):
            sections.append(_build_analysis_section(i + 1, r))
        if config.get("include_charts") and charts:
            for c in charts:
                c_dict = c if isinstance(c, dict) else (
                    c.model_dump() if hasattr(c, 'model_dump') else {}
                )
                if c_dict.get("source_index", 0) == i:
                    sections.append(_build_chart_section(c_dict, r))

    # 洞察
    cfg_insights = config.get("include_insights", "all")
    if cfg_insights != "none":
        raw_insights = _generate_insights(results)
        raw_insights = _resolve_conflicts(raw_insights)
        if cfg_insights == "top3":
            raw_insights = raw_insights[:3]
        else:
            raw_insights = raw_insights[:max_insights]
        if raw_insights:
            sections.append(_build_insights_section(raw_insights))

    # 尾部
    if config.get("include_footer"):
        sections.append(_build_footer())

    # 原始数据
    if config.get("include_raw_data"):
        sections.append(_build_raw_data_section(results))

    return _safe_truncate("\n\n".join(sections), max_length)


# ---------- L2: MCP 公开接口 ----------

@mcp.tool(
    name="kg_report",
    annotations={
        "title": "图谱分析报告",
        "readOnlyHint": True,
        "destructiveHint": False,
        "description": "生成知识图谱分析报告。整合kg_analyze统计结果和kg_visualize图表，输出结构化Markdown报告（含洞察+置信度）。"
    }
)
async def kg_report(params: KgReportInput) -> str:
    start = time.perf_counter()
    try:
        results = params.all_results
        charts = params.charts or []
        report_type = params.report_type
        max_length = params.max_length
        max_insights = params.max_insights
        warning_parts = []

        # 参数校验
        if not results or len(results) == 0:
            return _error_json("all_results不能为空", tool="kg_report")

        # 校验分析结果
        for i, r in enumerate(results):
            if not isinstance(r, dict):
                return _error_json(f"all_results[{i}]必须是字典", tool="kg_report")
            if "entity" not in r:
                return _error_json(f"all_results[{i}]缺少entity字段", tool="kg_report")

        # 校验报告类型
        config = REPORT_CONFIGS.get(report_type)
        if not config:
            warning_parts.append(
                f"未知report_type='{_md_escape(report_type)}'，已回退到full"
            )
            config = REPORT_CONFIGS["full"]
            report_type = "full"

        # 构建报告
        report = _build_report(
            results, charts, params.query,
            config, max_length, max_insights
        )

        if not report or report.strip() == "":
            return _error_json("报告生成失败，输出为空", tool="kg_report")

        # 保存Markdown报告到 /mnt/nexent/reports/
        import hashlib
        _REPORT_DIR = "/mnt/nexent/reports"
        os.makedirs(_REPORT_DIR, exist_ok=True)
        entity_name = results[0].get("entity", "report") if results else "report"
        report_path = os.path.join(_REPORT_DIR, f"{entity_name}_分析报告.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        elapsed = round((time.perf_counter() - start) * 1000, 2)

        # 统计洞察数
        insight_count = report.count("\n| 1 |") + report.count("\n| 2 |") + \
                        report.count("\n| 3 |") + report.count("\n| 4 |") + \
                        report.count("\n| 5 |") + report.count("\n| 6 |")

        result = {
            "success": True,
            "report": report,
            "report_path": report_path,
            "report_type": report_type,
            "analysis_count": len(results),
            "chart_count": len(charts),
            "insights_generated": min(insight_count, 99),
            "elapsed_ms": elapsed,
            "tool": "kg_report",
        }
        if warning_parts:
            result["warning"] = "; ".join(warning_parts)
        return json.dumps(result, indent=2, ensure_ascii=False)

    except Exception as e:
        return _error_json(f"{type(e).__name__}: {str(e)}", tool="kg_report")


# ============================================================
# 更新datamate_execute_operator的描述，加入KG算子
# ============================================================


# ============================================================
# 入口
# ============================================================


def main():
    """启动MCP服务器"""
    if not DATAMATE_TOKEN:
        print("⚠️  DATAMATE_TOKEN 未设置！")
        print("   获取方法：")
        print("   1. 浏览器打开 http://localhost:30000")
        print("   2. 登录DataMate Web UI")
        print("   3. F12 → Application → Local Storage → 复制token值")
        print("   4. set DATAMATE_TOKEN=你的token")
        print("")

    print(f"🚀 DataMate MCP Server v3.0 启动中...")
    print(f"📡 DataMate Gateway: {DATAMATE_GATEWAY}")
    print(f"🔌 监听端口:         {SERVER_PORT}")
    print(f"🔑 Token已设置:     {'✅' if DATAMATE_TOKEN else '❌'}")
    print(f"📦 本地算子:         {len(_LOCAL_OPERATORS)} 个")
    print(f"📂 算子源目录:       {_OPERATORS_DIR}")
    print(f"")
    print(f"在Nexent中用方式2连接:")
    print(f"  地址: http://host.docker.internal:{SERVER_PORT}")
    print(f"")

    # 使用 Streamable HTTP 传输
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
