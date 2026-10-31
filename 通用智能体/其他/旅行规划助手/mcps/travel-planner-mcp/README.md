# Travel Planner MCP Server

面向 NEXENT 旅行规划助手的自定义 MCP 工具服务。**只用 Python 标准库**，不需要安装依赖，Windows / macOS / Linux 均可直接运行。

## 1. 启动

```bash
# 方式 A：本地直接运行（HTTP，默认 0.0.0.0:8765）
cd mcp_server
python -m travel_mcp --transport http --port 8765

# 方式 B：Docker
docker compose up -d --build
```

健康检查：

```bash
curl http://127.0.0.1:8765/health
# {"ok": true, "server": "travel-planner-mcp", "version": "1.0.0", "tools": 6}
```

## 2. 在 NEXENT 中接入

NEXENT 与 MCP 服务不在同一容器网络时，URL 要写**宿主机可达地址**：

| NEXENT 部署方式 | 填写 URL |
| --- | --- |
| 本机 Docker 部署 | `http://host.docker.internal:8765/mcp` |
| 局域网服务器 | `http://<服务器IP>:8765/mcp` |
| 与 MCP 同一个 compose 网络 | `http://travel-mcp:8765/mcp` |

在「MCP 仓库 → 我的 MCP → 添加 MCP 服务」中选择「远程」，粘贴上面的 URL，点击「连通性校验」，通过后保存，再到智能体配置里勾选 6 个工具。

## 3. 工具清单

| 工具 | 作用 | 关键参数 |
| --- | --- | --- |
| `get_weather` | 未来 1-14 天天气（Open-Meteo） | `city`, `start_date`, `days` |
| `search_poi` | 景点/街区检索（本地精选库 + OSM 兜底） | `city`, `keyword`, `category`, `limit` |
| `plan_route` | 两点距离与耗时（OSRM + 估算兜底） | `origin`, `destination`, `mode`, `city` |
| `estimate_budget` | 分项预算与机票区间 | `city`, `days`, `travelers`, `style`, `origin_city` |
| `build_itinerary` | 按片区生成逐日行程骨架 | `city`, `days`, `pace`, `interests`, `must_see` |
| `get_local_essentials` | 货币/插头/紧急电话/交通卡 | `city` |

支持的离线城市：东京、曼谷、清迈、北京、上海、成都。

## 4. 测试

```bash
cd mcp_server
python -m unittest discover -v
```

16 个用例覆盖城市归一化、景点检索、预算、行程编排（含周一闭馆规避、整日项目单独成天）、路线和 MCP 协议（initialize / tools/list / tools/call / health）。

## 5. 协议说明

- `POST /mcp`：Streamable HTTP，直接返回 JSON-RPC 响应
- `GET /sse` + `POST /messages?session_id=...`：旧版 HTTP+SSE，兼容只支持 SSE 的客户端
- `--transport stdio`：stdio 模式，供容器化 MCP 客户端使用

示例：

```bash
curl -s http://127.0.0.1:8765/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"build_itinerary","arguments":{"city":"东京","days":3}}}'
```

## 6. 数据边界

`travel_mcp/data.py` 是演示用精选数据集：门票、开放时间、汇率、签证政策都可能变化，工具输出中已标注「需复核」。生产使用时请替换为官方 API（高德地图、飞常准、酒店供应商）或定期更新数据集。
