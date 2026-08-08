# MediGraph Agent：医疗数据—知识—洞察智能体

> 队伍：**MaineCoon** ｜ 第八届 CCF 开源创新大赛 · ModelEngine 开源项目贡献赛
>
> 基于 Nexent、DataMate、MCP 与 A2A，把非结构化医疗文本一路做到可溯源知识图谱与
> 图谱驱动的 BI 洞察。四个智能体分工协作，全部工具由同一个 MCP 服务提供。

## 一、这是什么

医疗领域的数据分析常常卡在两头：原始病历是非结构化文本，直接喂给大模型问答无法溯源；
而做成知识图谱后，又难以支撑"哪个科室接诊最多"这类统计型分析。

MediGraph Agent 把这条链路打通成一个闭环：

```
非结构化医疗文本
   ↓  DataMate 五算子流水线 / MCP 算子（清洗 → 切块 → NER → RE → 三元组校验）
结构化三元组（逐条带置信度与来源）
   ↓  实体链接 + 本体校验 + 逐边溯源
可溯源医疗知识图谱
   ↓  GraphRAG 问答（证据不足时安全拒答）   ↓  图谱感知 NL2SQL / 图算法
可引用的医学问答                            统计洞察 + 可视化图表
```

### 四个智能体

| 智能体 | 配置文件 | 职责 | 使用工具 |
|---|---|---|---:|
| 医疗数据处理智能体 | `dataproc_agent.json` | 理解自然语言数据处理需求，编排算子完成清洗/抽取/建图，可调度 DataMate 五算子流水线 | 10 |
| 医疗知识图谱生成与问答智能体 | `kg_qa_agent.json` | 构建可溯源图谱并做 GraphRAG 问答，证据不足时拒答 | 7 |
| 图谱驱动分析与 BI 智能体 | `analysis_supervisor.json` | 复用图谱，自动在 SQL 统计与 GRAPH 关联分析间路由，产出图表与洞察 | 2 |
| **闭环协作总控智能体** | `collaboration_supervisor.json` | 统一调度上述三者，可通过 MCP 直接执行，也可把请求转交 A2A 子智能体 | 编排 |

只想跑单项能力就导入前三个之一；想看完整的数据→知识→洞察闭环，导入**协作总控智能体**。

### 核心特性

- **置信度路由抽取级联**：神经 GPLinker 为主力，毫秒级本地词典基线兜底，低置信难例再路由 LLM，
  三条路径共用同一套输出契约，缺 GPU 时自动降级而不是报错。
- **逐边溯源**：每条三元组都带 `confidence` 与 `source`，问答给出的每个结论都能追回原文。
- **安全拒答**：检索不到证据或证据置信度低于阈值时明确拒答，不顺着问题编造医学结论。
- **图谱感知 Schema-Linking**：借图谱实体类型消歧，让"脂肪肝"匹配到 `disease` 字段而不是科室或药品字段。
- **SQL 双层护栏**：AST 白名单 + SQLite authorizer，生成的 SQL 只读、限时、限行数。

## 二、目录结构

```text
MediGraphAgent/
├── README.md                      本文档
├── 智能体源文件/                   ← 导入 Nexent 的四个智能体配置
│   ├── dataproc_agent.json
│   ├── kg_qa_agent.json
│   ├── analysis_supervisor.json
│   ├── collaboration_supervisor.json
│   └── README.md                  各智能体的提示词与工具清单说明
├── mcp服务/                        ← 上述智能体依赖的 MCP 服务（17 个工具）
│   ├── mcp_server/                MCP 服务入口
│   ├── medigraph/                 算子、抽取级联、图谱、分析等核心实现
│   ├── config/                    配置解析
│   ├── integration/a2a/           A2A 服务（协作总控智能体用）
│   ├── data/
│   │   ├── models/                【知识库】词典抽取器、实体链接库、标定参数
│   │   │                          （词典抽取器以 .json.gz 交付，代码直接读取，无需解压）
│   │   ├── corpus/                示例医疗文档（60 篇）
│   │   └── demo_cases/            端到端演示语料
│   ├── outputs/
│   │   ├── graph.json             【知识库】演示医疗知识图谱（7,467 实体 / 26,784 关系）
│   │   └── analytics.db           【知识库】分析用关系库（600 就诊 / 887 处方 / 885 检验）
│   ├── scripts/                   启动与部署脚本
│   ├── requirements.txt           Python 依赖
│   ├── .env.example               配置模板（复制为 .env 后填写）
│   ├── compose.yaml / Dockerfile  容器化一键部署
├── skills/                        ← Nexent 技能包
│   ├── medical-kg-report.zip      直接上传到 Nexent 的技能包
│   ├── medical-kg-report源码/      技能源码（可修改后重新打包）
│   └── build_skill_zip.py         重新打包脚本
└── datamate算子包/                 ← DataMate 平台可上传的五个算子
    ├── text_clean.zip  chunker.zip  medical_ner.zip
    ├── medical_re.zip  triple_validator.zip
    └── 源码/                       算子源码与打包脚本
```

## 三、运行环境依赖

| 项 | 要求 | 说明 |
|---|---|---|
| Nexent 平台 | 已安装并可访问 | 参考 [Nexent 安装指导](https://modelengine-group.github.io/nexent/zh/quick-start/installation.html) |
| Python | 3.10 – 3.12 | MCP 服务运行环境 |
| 大语言模型 | OpenAI 兼容 API | SiliconFlow / DashScope / DeepSeek / 本地 vLLM 均可 |
| DataMate | 可选 | 仅在使用 `plan_datamate_pipeline`、`run_datamate_pipeline` 时需要 |
| GPU | 可选 | 无 GPU 时抽取自动回退词典/LLM 路径，功能不受影响 |
| Docker | 可选 | 用容器方式启动 MCP/A2A 时需要 |

**关于抽取模型的两点说明**：

1. 主力的自训练 GPLinker 神经权重约 1.4 GB，超出本仓库合理体积，未随包提供。
   **不影响开箱运行**——抽取级联会自动使用随包的词典快速路径或 LLM 路径。
   如需神经权重，见本文末尾"完整项目"一节。
2. 词典快速路径的产物以 **`data/models/fast_extractor.json.gz`**（2.3 MiB）交付，
   因为解压后的 23 MiB 超过本仓库单文件 10 MiB 的上限。**加载时自动读取压缩包，
   无需手工解压**；若同目录存在未压缩的 `fast_extractor.json` 则优先使用它。

## 四、快速开始

### 步骤 1：启动 MCP 服务

```powershell
cd mcp服务
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt

copy .env.example .env                 # Linux/macOS: cp .env.example .env
#  编辑 .env，至少填写一个模型的 API Key，例如：
#  LLM_PROVIDER=siliconflow
#  SILICONFLOW_API_KEY=sk-你的key

python mcp_server/server.py            # 默认监听 127.0.0.1:8011
```

看到服务启动后，MCP 地址为 `http://127.0.0.1:8011/sse`。

若 Nexent 跑在 Docker 里，容器内应使用 `http://host.docker.internal:8011/sse`
（四个智能体配置里已经预填这个地址）。

**容器方式**（可选，同时拉起 MCP 与 A2A）：

```powershell
docker compose up -d --build
```

### 步骤 2：启动 A2A 服务（仅协作总控智能体需要）

```powershell
python integration/a2a/a2a_server.py    # 默认监听 127.0.0.1:8100
```

Agent Card 地址：`http://127.0.0.1:8100/.well-known/agent.json`

### 步骤 3：在 Nexent 中上传技能

智能体配置页 → 技能 → 构建技能 → 上传 `skills/medical-kg-report.zip`。

### 步骤 4：导入智能体

智能体配置页 → 导入 → 选择 `智能体源文件/` 下的 JSON 文件 → 选择大语言模型 →
安装 MCP 工具（填入步骤 1 的 MCP 地址）→ 完成安装 → 发布。

建议导入顺序：先导入三个专才智能体，再导入协作总控智能体。

> 详细的平台操作图示见仓库根目录的 [操作指导.md](../../../操作指导.md)。

### 步骤 5（可选）：上传 DataMate 算子

在 DataMate 算子市场上传 `datamate算子包/` 下的五个 zip，
之后数据处理智能体即可通过 `run_datamate_pipeline` 调度真实的 DataMate 流水线。

## 五、使用示例

导入后可直接在 Nexent 对话框里试这些问题。

### 示例 1 · 数据处理智能体：从文本到图谱

```
请把下面这段病历清洗、切块，抽取医疗实体和关系，并校验三元组：

患者男性，65岁，因反复胸闷 3 个月入院。既往高血压病史 10 年，
长期口服苯磺酸氨氯地平。心电图示 ST 段压低，诊断为冠心病，
予阿司匹林肠溶片抗血小板治疗。
```

智能体会依次调用 `text_clean → chunker → medical_ner → medical_re → triple_validator`，
返回结构化三元组与各步骤的中间产物。

### 示例 2 · 知识图谱问答智能体：可溯源问答

```
基于 graph.json 回答：高血压可能并发哪些疾病？
请列出证据三元组、置信度和来源，不要补充图谱之外的医学结论。
```

返回结论 + 每条结论对应的三元组证据链。

**安全拒答验证**——问一个图谱里不存在的实体：

```
幻想性紫罗兰综合征有哪些典型症状？
```

智能体会明确回答图谱中没有该实体、证据为 0，拒绝作答，而不是顺着编造。

### 示例 3 · 分析 BI 智能体：统计与可视化

```
就诊人次最多的十种疾病是什么？用柱状图展示
```

自动路由到 SQL 路径，返回实际执行的 SQL、结果表和柱状图 HTML。换一个图关联型问题：

```
脂肪肝的并发症在知识图谱里是怎么连接的？画成关系图
```

自动路由到 GRAPH 路径，返回关系网络图与证据三元组。

### 示例 4 · 协作总控智能体：完整闭环

```
请完成一次完整的医疗数据闭环：把 data/demo_cases/ 下的文档处理成知识图谱，
就图谱内容做一次可溯源问答，再基于图谱做一次统计分析并出图。
```

总控智能体会依次调度数据处理、图谱问答与分析三个环节。

## 六、MCP 工具清单（17 个）

| 分类 | 工具 | 说明 |
|---|---|---|
| 数据接入 | `load_documents` | 多格式文档读取 |
| | `profile_data_quality` | 数据质量检查 |
| | `redact_medical_pii` | 医疗 PII 脱敏 |
| 算子 | `text_clean` | 文本清洗规范化 |
| | `chunker` | 语义感知切块 |
| | `medical_ner` | 医疗实体识别 |
| | `medical_re` | 医疗关系抽取 |
| | `triple_validator` | 三元组 schema/冲突校验 |
| | `link_medical_entities` | 实体链接与规范 ID |
| 图谱 | `build_medical_kg` | 端到端建图 |
| | `medical_kg_qa` | GraphRAG 可溯源问答 |
| | `inspect_medical_kg` | 图谱统计与审计 |
| 分析 | `analyze_medical_data` | SQL/GRAPH 双路由分析与出图 |
| | `inspect_analysis_assets` | 分析资产审计 |
| 平台 | `plan_datamate_pipeline` | 规划 DataMate 流水线 |
| | `run_datamate_pipeline` | 执行 DataMate 流水线 |
| 审计 | `inspect_extraction_models` | 抽取级联与模型路由审计 |

## 七、常见问题

**MCP 工具在 Nexent 里显示不可用**
检查 MCP 服务是否已启动、地址是否可达。Nexent 在 Docker 中运行时，
`127.0.0.1` 指向的是容器自身，必须用 `host.docker.internal`。

**抽取结果为空或报模型缺失**
说明既没有神经权重也没有配置 LLM。确认 `.env` 里至少填了一个可用的 API Key；
或确认 `data/models/fast_extractor.json` 存在（词典路径可完全离线工作）。

**问答一直拒答**
这是安全策略生效：图谱中没有相关证据。先用 `build_medical_kg` 建图，
或在问答时指定随包的演示图谱 `graph.json`。

**DataMate 相关工具报错**
这两个工具需要可访问的 DataMate 服务。不使用 DataMate 时，
数据处理智能体可以直接用 MCP 算子完成同样的流水线。

## 八、完整项目

本目录是为社区复现整理的**运行包**，包含四个智能体、MCP 服务、技能包、算子包与知识库。
完整工程（神经抽取训练代码与权重、昇腾 NPU Ascend C 融合算子、0.8B LoRA 编排模型、
315 项自动化测试、全部评测脚本与技术报告）见参赛仓库。

许可：源代码 MIT。随包的知识图谱与分析数据库为项目自产或合成数据；
第三方医疗数据集不随本包重新授权。

> 医疗声明：本项目为比赛与研究原型，不提供诊断、处方或医学建议。
