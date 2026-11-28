# 只读质检数据 MCP

18 个只读/计算工具，无写工具。自带演示数据库、schema 和演示标准。

## 文件说明

| 文件/目录 | 作用 |
|---|---|
| `server.py` | 可执行 MCP 服务，注册 18 个只读工具，读取 SQLite 和 knowledge |
| `requirements.txt` | Python 依赖，目前只需要 `mcp` |
| `Dockerfile` | 可选容器镜像 |
| `docker-compose.yml` | 可选容器启动配置 |
| `config.example.json` | 参数示例，不会被程序自动读取；用于说明 DB、知识库、端口 |
| `test_client.py` | 冒烟测试：连接本机 MCP、列出工具、调用 `get_stats` |
| `data/schema.sql` | 演示库表结构 |
| `data/seed_demo.py` | 重新生成 `demo.db` |
| `data/demo.db` | 合成演示数据，不含生产数据 |
| `knowledge/通用.json` | 演示通用标准 |
| `knowledge/F520M.json` | 演示 F520M 标准 |
| `README.md` | 本说明 |

## 启动

```bash
pip install -r requirements.txt
python server.py --host 0.0.0.0 --port 5071
```

或：

```bash
docker compose up --build
```

默认：

```text
DB: data/demo.db
知识库: knowledge/
端点: http://127.0.0.1:5071/mcp
```

## 验证

```bash
python test_client.py
```

预期：18 个工具，`get_stats` 返回总数 150。

## Nexent 配置

```text
http://host.docker.internal:5071/mcp
```

Nexent 与 MCP 同 Docker 网络时，使用服务名。

## 工具

- 记录与统计：`query_records`、`get_record`、`get_stats`、`get_daily_trend`、`get_vehicle_stats`、`list_vehicles`
- 双针：`query_stitch_records`、`get_stitch_stats`
- 单针：`query_single_stitch_records`、`get_single_stitch_stats`、`query_single_stitch_standards`
- 孔径：`query_hole_records`、`get_hole_stats`、`query_hole_standards`
- 标准：`query_standards`
- 计算：`analyze_stitch`、`analyze_single_stitch`、`analyze_hole`

## 替换真实数据

```bash
python server.py --db-path /path/to/detection.db --knowledge-dir /path/to/knowledge
```

数据库表：

```text
detection_records
stitch_inspection
single_stitch_inspection
hole_inspection
users（可选）
```

表结构：`data/schema.sql`。

## 安全

只读、无鉴权、仅本机或受控内网；`demo.db` 为合成数据。
