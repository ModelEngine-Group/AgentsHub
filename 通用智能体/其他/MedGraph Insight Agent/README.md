# MedGraph Insight Agent

MedGraph Insight Agent 是 OS 战队提交至 ModelEngine 开源项目贡献赛的离线优先医疗数据智能体。它通过一个可独立部署的 MCP 服务完成医疗数据清洗、知识图谱构建、图谱问答、NL2SQL 风格分析和质量审计，并可导入 Nexent 进行编排。

> 本项目仅用于数据处理、技术研究和演示，不构成医疗诊断、处方或治疗建议。输出必须由具备资质的专业人员复核。

## 核心能力

- `run_pipeline`：清洗 JSONL/CSV/TXT 数据并构建医疗知识图谱。
- `query_graph`：查询实体、关系与图谱统计。
- `answer_medical_question`：依据图谱证据回答医疗知识问题。
- `run_analysis`：执行图谱驱动的统计分析并返回图表数据。
- `get_benchmark`：报告 CPU、CUDA 与 Ascend NPU 后端可用性。
- `get_quality_report`：检查图谱结构、证据覆盖率和已知错误三元组。

## 目录结构

```text
MedGraph Insight Agent/
├── agent.json                         # Nexent 导出的智能体配置
├── README.md
├── CHANGELOG.md
├── LICENSE
├── configs/
│   ├── nexent_mcp.json                # Nexent MCP 接入参数
│   └── datamate_operator_manifest.json
├── knowledge/
│   ├── medical_cases.jsonl            # 可公开复现的医疗样例数据
│   └── nl2sql_eval.json               # NL2SQL 意图评测集
├── mcps/medgraph-insight-agent/
│   ├── run_server.py                  # SSE/Streamable HTTP 服务入口
│   ├── pyproject.toml
│   └── src/medgraph_agent/
└── verification/verify_tools.py       # 六工具离线验证脚本
```

本版本不依赖自定义 Skill。所有运行所需能力均由随目录提供的 MCP 服务实现。

## 环境要求

- Python 3.11 或更高版本
- Nexent 2.x（导入和编排时需要）
- Windows、Linux 或 macOS；命令示例使用通用 POSIX 写法

## 安装与启动

在本目录执行：

```bash
python -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -c mcps/medgraph-insight-agent/constraints.txt \
  -e mcps/medgraph-insight-agent

python mcps/medgraph-insight-agent/run_server.py \
  --transport sse --host 0.0.0.0 --port 8100
```

MCP 地址为 `http://127.0.0.1:8100/sse`。当 Nexent 运行在 Docker 中时，使用 `http://host.docker.internal:8100/sse`。

## 导入 Nexent

1. 启动 MCP 服务。
2. 在 Nexent 的“智能体开发 → MCP 配置”中添加 URL 类型服务：
   - 服务名：`MedGraphInsight`
   - 服务地址：`http://host.docker.internal:8100/sse`
3. 确认六个工具均已同步且健康检查通过。
4. 在“智能体开发 → 导入”中上传 `agent.json`。
5. 根据部署环境选择可用模型，保存后进入“开始问答”验证。

## 使用示例

- `请处理 knowledge/medical_cases.jsonl，构建医疗知识图谱并报告质量检查结果。`
- `高血压有哪些症状、治疗和检查证据？请只依据图谱回答。`
- `统计知识图谱中的关系类型分布，并返回适合柱状图的数据。`
- `检查图谱是否存在悬空关系或无证据三元组。`

## 独立验证

```bash
python verification/verify_tools.py
python verification/verify_transport.py  # MCP 服务运行时执行
```

成功时输出 `verified_tools=6`。验证脚本使用临时输出目录，不会在仓库内生成结果文件。

## 数据与安全

- `knowledge/` 中仅包含用于比赛复现的合成或公开样例，不包含真实患者身份信息。
- 服务默认离线运行，不要求 API Key，也不会上传输入数据。
- MCP 工具不会给出确定性诊断或替代医生决策。
- 仓库不包含账号、密码、令牌或私有服务地址。

## 来源与许可证

- 战队：OS
- 作者：李佳斌、李佳禾、姚舒文
- 原始作品仓库：<https://gitlink.org.cn/Lijiabin1234/medgraph-insight-agent>
- 归档基线提交：`658b0d2`
- 许可证：MIT
