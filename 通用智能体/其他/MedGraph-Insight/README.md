# MedGraph-Insight · 医疗数据洞察智能体

> 医疗领域「**数据 → 知识 → 洞察**」一体化智能体：清洗医疗文本、构建医疗知识图谱、
> 基于图谱做检索问答与多跳推理、用自然语言查询关系数据（NL2SQL）、生成图文 BI 报告。
>
> 2026 ModelEngine 开源项目贡献赛参赛作品归档 · 基于 **Nexent** 智能体框架与 **DataMate** 数据工程理念

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

---

## 1. 核心功能

| 能力 | 说明 | 关键指标 |
| --- | --- | --- |
| **数据处理** | DAG 编排 7 个清洗算子（去 HTML、繁简转换、手机号/身份证脱敏、空白归一、去重、短文本过滤），输出逐节点数据质量指标与血缘链路 | 拓扑排序 + 环检测；PII 泄漏率、重复率、HTML 残留率可量化 |
| **知识图谱** | 实体识别 → 关系抽取 → 三元组校验，自动构建医疗知识图谱 | **212 实体 / 351 关系**；NER F1 **1.000**、RE F1 **0.997** |
| **图谱问答** | GraphRAG 语义检索，支持别名/英文/缩写链接与多跳推理，答案附图谱证据 | `T2DM → 2型糖尿病`；并发症链二跳汇聚 |
| **分析洞察** | 自然语言转 SQL 查询关系数据，并生成自包含 HTML 图文 BI 报告 | NL2SQL 执行准确率 **100% (40/40)** |

配套 **MCP 服务**将上述能力暴露为 6 个工具，可直接接入 Nexent 平台。

## 2. 运行环境依赖

| 项目 | 要求 |
| --- | --- |
| Python | **3.10+**（验证环境 3.11.4） |
| 核心依赖 | `networkx>=3.0`、`numpy>=1.24`、`pandas>=2.0`、`matplotlib>=3.7`、`requests>=2.31` |
| MCP 服务 | `mcp>=1.2.0` |
| 可选 | `nexent`（接入 Nexent CoreAgent）、`openai>=1.0`（LLM 规划器）、`torch` / `torch_npu`（NPU 算子） |
| 中文字体 | 图表含中文，建议安装：Ubuntu `sudo apt-get install -y fonts-noto-cjk`；缺失时图表中文显示为方框，不影响数据正确性 |
| 外部服务 | **无需**。默认全链路离线可跑，不依赖大模型或联网服务 |

## 3. 目录结构

```text
MedGraph-Insight/
├── README.md                    # 本文档
├── agent.json                   # Nexent 智能体配置（导入平台用）
├── LICENSE                      # MIT
├── mcps/                        # MCP 服务
│   ├── mgi_mcp_server.py        # 6 个 MCP 工具的服务实现
│   ├── requirements.txt
│   └── README.md                # 启动方式与工具清单
├── skills/                      # 技能包（算子设计与提示词规范）
│   ├── 医疗知识抽取算子.md
│   └── 医疗图谱问答与NL2SQL.md
├── knowledge/                   # 知识库文件
│   ├── dictionaries/            # 医疗术语词典（7 类 200+ 词条）+ 别名/英文/缩写表
│   ├── corpus/                  # 医疗语料（40 篇疾病案例/指南）
│   └── eval/                    # 金标准：NER/RE 标注 + NL2SQL 40 条评测集
├── code/                        # 可复现运行的完整源码
│   ├── mgi/                     # operators / kg / agents / analysis / datamate / eval
│   ├── data/                    # 与 knowledge/ 同源，供代码默认相对路径直接读取
│   ├── tests/                   # 51 项单元测试
│   ├── pyproject.toml
│   └── Dockerfile
└── docs/                        # NPU 优化、昇腾真机基准、SOTA 指标
```

> `knowledge/` 是面向阅读与知识库导入的组织形式；`code/data/` 是同源副本，
> 使 `python -m mgi.cli` 无需改配置即可直接运行。

## 4. 快速开始

### 4.1 方式一：作为 MCP 服务接入 Nexent（推荐）

```bash
# 1) 安装依赖
cd mcps
pip install -r requirements.txt

# 2) 启动 MCP 服务（SSE）
python mgi_mcp_server.py --transport sse --host 0.0.0.0 --port 8765
# 输出 MCP URL: http://<服务器地址>:8765/sse
```

然后在 Nexent 平台：

1. **添加模型**：模型配置页填入可用大模型的名称、URL、API Key 并验证连通性；
2. **配置 MCP**：智能体开发 → 选择工具 → MCP 配置，服务名 `medgraph-insight`，
   URL 填上一步输出的 `http://<服务器地址>:8765/sse`；
3. **导入智能体**：智能体开发 → 导入，上传本目录的 `agent.json`，选择大模型；
4. **发布**：确认信息后发布，即可在问答中使用。

（平台操作截图见仓库根目录 [`操作指导.md`](../../../操作指导.md)）

### 4.2 方式二：命令行直接运行（无需平台）

```bash
cd code
pip install -e .          # 或：pip install -r ../mcps/requirements.txt

mgi pipeline              # 端到端：清洗 → 建图 → 问答 → 分析报告 → 评测
```

分任务运行：

```bash
mgi clean                                   # 任务一：清洗语料
mgi dag                                     # 任务一：DAG 编排 + 数据质量 + 血缘
mgi build                                   # 任务二：构建知识图谱
mgi ask --question "痛风好发于哪些人群？"      # 任务二：图谱问答
mgi ask --rag --question "心力衰竭的并发症用什么药？"   # 任务二：GraphRAG 多跳
mgi analyze                                 # 任务三：生成 BI 报告
mgi eval-kg                                 # NER/RE 金标准 P/R/F1
mgi eval-sql                                # NL2SQL 执行准确率
mgi bench                                   # CPU/NPU 算子性能对比
```

未安装为包时，可用 `python -m mgi.cli <子命令>` 等价调用。

### 4.3 方式三：Docker

```bash
cd code
docker build -t medgraph-insight .
docker run --rm -v $(pwd)/output:/app/output medgraph-insight
```

## 5. 使用示例

### 5.1 清洗医疗文本（MCP 工具 `clean_medical_texts`）

输入：

```json
{"texts": [
  "<p>患者血压 160/95 mmHg，聯繫電話 13800001111。</p>",
  "<p>患者血压 160/95 mmHg，聯繫電話 13800001111。</p>",
  "短"
]}
```

返回（节选）：

```json
{
  "topological_order": ["clean_html", "t2s", "anon_phone", "anon_id", "ws", "dedup", "minlen"],
  "input_count": 3,
  "output_count": 1,
  "nodes": [{"id": "clean_html", "operator": "HtmlTagCleaner",
             "rows": "3 -> 3", "html_rate": "0.67 -> 0.0"}]
}
```

HTML 被剥离、手机号脱敏为 `<PHONE>`、重复文本去重、过短文本被过滤，每一步的数据质量变化可审计。

### 5.2 图谱问答（`ask_medical_kg`）

```text
Q: T2DM有哪些症状？
A: 2型糖尿病 的疾病-临床表现包括：乏力、体重下降、多尿、多食、多饮、视物模糊。
   （实体链接：T2DM → 2型糖尿病；关系：HAS_SYMPTOM；证据附语料原文与置信度）
```

多跳推理：

```text
Q: 心力衰竭的并发症用什么药？
A: 冠心病（硝酸甘油、阿司匹林、阿托伐他汀）；心房颤动（利伐沙班、华法林、美托洛尔）；
   急性心肌梗死（氯吡格雷、阿司匹林、阿托伐他汀）。
   （关系链：COMPLICATION_OF + TREATED_BY_DRUG，hops=2）
```

### 5.3 自然语言查询（`query_kg_with_nl2sql`）

```text
Q: 既有胸痛又有心悸的疾病
SQL: SELECT head FROM (SELECT head FROM relations WHERE relation='HAS_SYMPTOM' AND tail='胸痛'
     INTERSECT SELECT head FROM relations WHERE relation='HAS_SYMPTOM' AND tail='心悸') ORDER BY head
结果: [["冠心病"]]
```

### 5.4 生成 BI 报告（`generate_bi_report`）

产出 `code/output/analysis_report.html`（自包含单文件），含图谱概览、实体类型×关系类型
覆盖热力图、TopN 关联排行、并发症关系网络与实体中心子图。

## 6. 验证与复现

```bash
cd code
pip install -e ".[dev]"

pytest -q                     # 51 项单元测试
python -m mgi.cli eval-kg     # NER F1=1.000 / RE F1=0.997（直接从语料评测）

# eval-sql 在已构建的图谱上评测，需先 build（或先跑一次 mgi pipeline）
python -m mgi.cli build
python -m mgi.cli eval-sql    # NL2SQL 100% (40/40)
```

本归档在 Windows + Python 3.11.4 下实测：**51 项测试全绿**，6 个 MCP 工具逐个调用成功，
图谱构建结果为 212 实体 / 351 关系，与上表指标一致。

更多技术细节见 [`docs/SOTA-指标与技术要点.md`](docs/SOTA-指标与技术要点.md)、
[`docs/npu-optimization.md`](docs/npu-optimization.md)、
[`docs/benchmark-ascend.md`](docs/benchmark-ascend.md)（昇腾 910B2C 真机基准）。

## 7. 说明

- `agent.json` 依据本仓库现有智能体的配置结构编写，用于在 Nexent 平台导入本智能体；
  其中 `model_id`/`model_name` 留空，导入时在平台选择可用大模型，`mcp_url` 按第 4.1 节填写。
- MCP 服务默认不带鉴权。使用 `--host 0.0.0.0` 会暴露到所有网卡，公网部署请置于
  反向代理/网关鉴权之后。
- 本作品面向技术演示与医疗知识管理，输出不构成医疗诊断建议。
- 语料与词典为公开医学常识整理，用于功能验证；不含任何真实患者数据。

## 8. 作者与许可证

- 作者：莫仁鹰（合肥工业大学）
- 许可证：[MIT License](LICENSE)
