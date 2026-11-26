# 质检数据分析师

> 汽车内饰缝纫质检 · Nexent 只读数据分析 Agent

## 行业

汽车内饰缝纫质检覆盖座椅面套、门板、仪表台、扶手等部件。

| 检测任务 | 关键指标 | 示例标准 |
|---|---|---|
| 双针装饰线 | 线距、偏差 | 4.0 ± 1.0 mm |
| 单针偏移 | 单针距离、偏移 | 4.0 ± 1.0 mm |
| 孔径分析 | 直径、椭圆度 | 10.0 ± 0.8 mm，椭圆度 ≤ 0.3 mm |

典型使用岗位：质检主管、班组长、工艺工程师、质量数据员、SQE。

## 能做什么

- 检测记录查询
- OK/NG 数量与合格率统计
- 按日期看合格率趋势
- 按车型、工位、任务类型拆分
- 双针、单针、孔径专项统计
- 质量标准查询
- 单次计算 d1/d2、直径、椭圆度
- 输出数据范围、统计表、异常点、数据缺口

## 不做什么

- 不新增、修改、删除检测记录
- 不保存分析结果
- 不做图片视觉判定
- 不替代人工放行和停机决策

## 包结构

```text
quality_data_analyst/
├─ agent.json
├─ README.md
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

### 2. 注册 MCP

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

## 工具

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

## 安全

- 只读 MCP，无写工具
- 默认无鉴权，仅本机或受控内网
- `demo.db` 不含生产数据

## 版本

- Agent：`quality_data_analyst`
- 版本：v2
- 模型：DeepSeek-Chat（可替换）
- 作者：fengzimu
