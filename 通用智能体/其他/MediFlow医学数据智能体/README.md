# MediFlow 医疗数据智能体

MediFlow 是一套面向 Nexent 的医疗数据智能体运行包，提供数据处理、知识图谱和数据分析三类能力。三个智能体共用一个 MCP 服务，以及随包提供的医学数据库和规则资产。

## 智能体组成

| 智能体 | 能力 | 配置位置 |
| --- | --- | --- |
| 数据处理智能体 | 注册并整理 TXT、CSV、JSON、JSONL、PDF 等医疗资料，完成格式清理、术语规范化和质量检查 | `agents/数据处理智能体/agent.json` |
| 知识图谱智能体 | 识别医学实体和关系，生成三元组，写入知识图谱并保留来源 | `agents/知识图谱智能体/agent.json` |
| 数据分析智能体 | 将中文问题转换为受控查询，返回疾病事实、统计结果、表格和图表 | `agents/数据分析智能体/agent.json` |

三个 `agent.json` 已由 Nexent 当前环境真实导出。导入前需要在 Nexent 中重新选择目标环境的模型、核对 MCP 工具绑定，并将 MCP 地址和知识库索引改为目标环境的配置。

## 目录结构

```text
MediFlow医学数据智能体/
├── README.md
├── agents/                    # 三个 Nexent 智能体的配置文件
├── mcps/mediflow-mcp/         # MCP 服务、抽取逻辑、查询逻辑和 DataMate 算子
├── knowledge_base/            # 压缩后的数据库、词典和规则资产
├── config/                    # 不含密钥的配置模板
├── skills/                    # 可选的 Nexent Skill 说明
└── prepare_runtime_assets.py  # 校验并展开数据库资产
```

## 运行条件

- Linux；
- Python 3.10 或更高版本；
- 已部署并可访问的 Nexent；
- 数据处理智能体需要已部署并可访问的 DataMate；
- 可访问的模型服务。

Nexent 和 DataMate 是外部平台，本目录提供与它们连接所需的 MCP 服务、算子和配置模板，不复制平台源码。

## 快速开始

### 1. 展开运行资产

在本目录执行：

```bash
python prepare_runtime_assets.py
```

脚本会校验文件完整性，并将数据库和词典展开到 `mcps/mediflow-mcp/` 的默认位置。只检查、不展开时执行：

```bash
python prepare_runtime_assets.py --verify-only
```

随包提供的运行资产和 Nexent 外部向量知识库的关系见 [`knowledge_base/README.md`](knowledge_base/README.md)。如需复建文本检索知识库，源文件位于 `knowledge_base/nexent/reconstructed_source_documents/`。

### 2. 启动 MCP 服务

```bash
cd mcps/mediflow-mcp
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env.runtime
python mcp_server/server.py
```

MCP 默认地址为 `http://<host>:8900/mcp`。配置模板见 [`config/README.md`](config/README.md) 和 [`mcps/mediflow-mcp/docs/CONFIGURATION_GUIDE.md`](mcps/mediflow-mcp/docs/CONFIGURATION_GUIDE.md)。

### 3. 在 Nexent 中接入

1. 在 Nexent 中添加可用模型；
2. 注册 MCP 地址并扫描工具，服务名使用 `medical-ai`；
3. 将三个目录中的 `agent.json` 分别导入 Nexent；
4. 为智能体选择目标环境模型并确认工具绑定；
5. 如果需要文本型知识检索，在 Nexent 中导入源文档并绑定知识库；
6. 保存并发布智能体。

如果只需要验证 MCP 服务，也可以先直接在 Nexent 的开始问答页面选择智能体进行调用。

## 使用示例

数据处理智能体：

> 请将这批 TXT、CSV、JSON 和 JSONL 医疗资料注册并完成格式清理，返回处理结果和质量信息。

知识图谱智能体：

> 请从这份医学资料中抽取疾病、症状和治疗关系，生成三元组并说明来源。

数据分析智能体：

> 同时包含症状“咳嗽”和检查“X线”的疾病有哪些？请返回结果表格并给出统计图。

## 数据与知识库

`knowledge_base/` 已包含 MCP 直接使用的预构建资产：知识图谱数据库、分析数据库、医学术语映射、语义噪声规则和抽取辅助词表。这些文件以 ZIP 保存，首次运行时展开；同时提供可重新导入 Nexent 的文本源文件。

Nexent 的向量知识库由 Nexent 服务管理，不会随 `agent.json` 自动迁移。归档中的文本源文件是依据当前服务端只读文档内容重建的可导入文件，不宣称是原始数据提供方的发行包；使用前仍应确认数据的再分发权限。目标环境需要文本检索时，应按 [`knowledge_base/README.md`](knowledge_base/README.md) 的说明重新建立索引并绑定。

## 自定义 Skill

当前三个智能体的业务能力由 MCP 工具提供，项目没有必须额外上传的自定义 Nexent Skill，因此 `skills/` 只保留说明。后续若新增可复用的提示词、工具编排或文件处理能力，可以编写独立 Skill 包、绑定到对应智能体，并将 Skill 文件和使用说明一并放入该目录。

## 说明

本目录是面向 Nexent 的运行归档包，聚焦三个智能体的运行组件，不包含完整工程的实验材料、答辩材料或独立可视化前端。可选前端若在目标环境单独部署，MCP 只读取其访问地址，不影响三个智能体的核心服务。
