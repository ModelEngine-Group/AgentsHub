# MediFlow MCP 服务

本目录提供三个 Nexent 智能体共用的 MCP 服务。服务使用 Streamable HTTP 暴露工具，智能体在 Nexent 中通过工具调用完成数据处理、知识图谱和数据分析。

## 主要模块

| 目录 | 作用 |
| --- | --- |
| `mcp_server/` | MCP 协议入口、工具注册和服务编排 |
| `core/` | 医学语义处理、关系抽取、查询规划和结果组织 |
| `mcp_server/task1/` | 数据集注册、文件整理和数据处理流程 |
| `mcp_server/task2/` | 实体关系抽取和图谱写入流程 |
| `task3/` | 数据分析服务和查询结果交付 |
| `operators/` | 可上传到 DataMate 的自定义算子 |
| `clients/` | Nexent、DataMate 等外部平台适配 |
| `kg/` | 知识图谱库和分析库的离线构建代码 |
| `scripts/runtime_env.py` | 运行时环境变量读取辅助模块 |
| `docs/` | 配置和架构参考 |

源代码中的 `task1`、`task2`、`task3` 是历史模块目录名；面向使用者的智能体名称以归档根目录 README 为准。

## 启动服务

先在归档根目录展开数据库资产：

```bash
python prepare_runtime_assets.py
```

然后在本目录安装依赖并启动：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env.runtime
python mcp_server/server.py
```

默认地址是 `http://<host>:8900/mcp`。配置字段见 [`docs/CONFIGURATION_GUIDE.md`](docs/CONFIGURATION_GUIDE.md)。

DataMate 和 Nexent 需要在目标环境单独提供。数据处理智能体还需要在 DataMate 中登记 `operators/` 下的算子；这一步使用 DataMate 自身的算子管理功能完成。
