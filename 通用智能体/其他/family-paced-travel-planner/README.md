# 旅游规划助手

v1.0 为亲子、陪长辈和偏好慢节奏的旅行者生成中文行程。用户可以给出连续步行上限，助手会按主要景点列出步行与休息安排，汇总整团预算和雨天备选，并用网页搜索核实公开信息。

## 配置搜索服务

在导入前准备一个 Nexent 可访问的 Open-WebSearch 服务。[Open-WebSearch 项目与安装说明](https://github.com/Aas-ee/open-webSearch)

使用 Docker 部署 Nexent 时，可以让搜索容器加入 Nexent 的 Docker 网络。下例使用已验证的镜像和常见网络名 `nexent_network`。请按自己的部署替换网络名。已有可用的 Open-WebSearch 服务时，直接使用现有服务地址。

```powershell
docker run -d --name open-websearch --restart unless-stopped --network nexent_network -e MODE=http -e PORT=3000 -e DEFAULT_SEARCH_ENGINE=duckduckgo ghcr.io/aas-ee/open-web-search@sha256:748f1d083c9930db8e8b5f017834e1efa9ec77844dbed01baef2acf3ad47e81a
```

在 Nexent「选择工具 → MCP 配置」添加 Streamable HTTP 服务，服务名填写 `open-websearch`。如果使用上面的容器，地址填 `http://open-websearch:3000/mcp`。保存后测试 `search` 工具能否返回标题和链接。投稿配置中的 `mcp_info` 为空，导入时使用目标环境中已注册的服务。

## 导入与使用

1. 在 Nexent 配置一个可用的大语言模型。在 v2.8.2 上使用 DeepSeek Flash 验证。
2. 配置并测试上述搜索工具，再到「智能体开发」或「智能体仓库 → 我的智能体」选择「导入」，上传 `agent.json`。
3. 打开智能体配置，核对运行模型和 `open-websearch` 的 `search` 工具。按需重新选择模型和工具并保存。
4. 在「调试」中发送下方示例。确认结果后发布，可在聊天时选择该助手。

本配置使用聊天模型和网页搜索。无法从搜索来源确认的价格、步行时间和设施状况会标为估算或待核实。余票以官方预约平台的实时结果为准。

## 验证示例

已在 Nexent v2.8.2 调试，并验证导出的配置可在已配置 Open-WebSearch 的环境中重新导入，恢复聊天模型与搜索工具的绑定。

| 输入 | 检查结果 |
| --- | --- |
| 两位成人带一位长辈去苏州三天两晚，全团预算 3000 元，往返大交通另算。每天只安排一个主要景点，连续步行最多 15 分钟。 | 每天一个主要景点，步行段与休息点分列，附雨天备选。一次验证中五项预算合计 3000 元。 |
| 我们一家三口想周末出去玩，还没决定去哪里。 | 先询问出发地、预算和同行者的步行需求。 |
| 请联网查苏州博物馆本馆的官方预约入口，并提供来源链接。查不到实时余票就标为待核实。 | 实际调用 `search`，返回官网链接，区分搜索来源和待核实的实时信息。 |
