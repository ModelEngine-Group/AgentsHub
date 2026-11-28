# 质检数据分析师

> 面向汽车内饰缝纫质检的 Nexent 只读数据分析 Agent

## 简介

质检数据分析师是一个面向汽车内饰缝纫质检场景的只读数据 Agent。

它接收自然语言问题，调用质检数据工具，查询检测记录、双针/单针/孔径回归表和车型标准，输出合格率、趋势、车型/工位对比、异常线索和数据缺口。

适用岗位：质检主管、班组长、工艺工程师、质量数据员、SQE。

## 功能特性

- 检测记录查询与明细追溯
- OK/NG 数量、合格率统计
- 按日期查看合格率趋势
- 按车型、工位、任务类型拆分
- 双针、单针、孔径专项统计
- 车型质量标准查询
- 单次计算 d1/d2、直径、椭圆度
- 输出数据范围、统计表、异常点、数据缺口
- 只读，不修改检测记录和标准

## 技术架构

```text
Nexent Agent: quality_data_analyst
├─ Skill: quality-data-analysis
│  └─ 分析 SOP、工具选择顺序、输出规范
├─ Knowledge: 车型标准 / BOM / 工艺文档
└─ MCP: autodetect-mcp
   └─ readonly_server/server.py
      ├─ 18 个只读工具
      ├─ SQLite: detection_records
      │            stitch_inspection
      │            single_stitch_inspection
      │            hole_inspection
      └─ knowledge/*.json
```

| 层 | 作用 |
|---|---|
| Nexent Agent | 理解问题、调度工具、组织回答 |
| Skill | 约束分析流程和输出格式 |
| MCP | 提供只读数据工具 |
| SQLite | 存放检测记录和回归结果 |
| Knowledge | 存放车型标准、BOM、工艺说明 |

## 工作流程

1. 用户提出问题，例如“统计最近 7 天双针合格率”；
2. Agent 识别时间范围、任务类型、车型、工位等条件；
3. Skill 指导 Agent 选择合适的工具；
4. Agent 通过 MCP 查询 SQLite 或知识库；
5. Agent 汇总统计结果，输出数据范围、表格、异常点和数据缺口；
6. 需要人工判断的部分，明确标注“建议复核”或“数据不足”。

## 文件结构

```text
quality_data_analyst/
├─ agent.json
├─ README.md
├─ SHA256SUMS.txt
├─ skills/
│  ├─ quality-data-analysis.zip
│  └─ quality-data-analysis/SKILL.md
└─ mcps/
   ├─ autodetect-mcp.json
   └─ readonly_server/
      ├─ server.py
      ├─ requirements.txt
      ├─ Dockerfile
      ├─ docker-compose.yml
      ├─ config.example.json
      ├─ test_client.py
      ├─ data/
      │  ├─ schema.sql
      │  ├─ seed_demo.py
      │  └─ demo.db
      └─ knowledge/
         ├─ 通用.json
         └─ F520M.json
```

## 快速开始

### 1. 启动只读 MCP

```bash
pip install -r mcps/readonly_server/requirements.txt
python mcps/readonly_server/server.py --host 0.0.0.0 --port 5071
```

或：

```bash
cd mcps/readonly_server
docker compose up --build
```

### 2. 在 Nexent 注册 MCP

Nexent → MCP 管理 → 新增 Streamable HTTP：

```text
http://host.docker.internal:5071/mcp
```

刷新工具，确认 18 个只读工具。

### 3. 导入 Agent

Nexent → Agent 管理 → 导入 `agent.json`。

如果工具 ID 冲突，按 GUI 重新选择同一批工具；挂载 `quality-data-analysis` Skill；发布。

### 4. 验证

```bash
cd mcps/readonly_server
python test_client.py
```

预期：18 个工具，`get_stats` 返回总数 150。

## 工具清单

- 记录与统计：`query_records`、`get_record`、`get_stats`、`get_daily_trend`、`get_vehicle_stats`、`list_vehicles`
- 双针：`query_stitch_records`、`get_stitch_stats`
- 单针：`query_single_stitch_records`、`get_single_stitch_stats`、`query_single_stitch_standards`
- 孔径：`query_hole_records`、`get_hole_stats`、`query_hole_standards`
- 标准：`query_standards`
- 计算：`analyze_stitch`、`analyze_single_stitch`、`analyze_hole`

## 演示数据

`demo.db` 为合成数据：

- 180 条检测记录
- 60 条双针记录
- 45 条单针记录
- 45 条孔径记录

替换真实数据：

```bash
python server.py --db-path /path/to/detection.db --knowledge-dir /path/to/knowledge
```

表结构见 `data/schema.sql`。

## 示例问题

```text
统计最近 7 天双针和单针的合格率。
按车型统计最近 7 天的合格率，找出最低的车型。
看一下最近一周每天的合格率趋势。
F520M 的双针质量标准是多少？
孔径最近一周的椭圆度超差情况怎么样？
```

## 安全与边界

- 只读 MCP，无写工具
- 默认无鉴权，仅本机或受控内网
- `demo.db` 不含生产数据
- 不做图片视觉判定
- 不替代人工放行和停机决策

## 版本

- Agent：`quality_data_analyst`
- 版本：v2
- 模型：DeepSeek-Chat（可替换）
- 作者：fengzimu
