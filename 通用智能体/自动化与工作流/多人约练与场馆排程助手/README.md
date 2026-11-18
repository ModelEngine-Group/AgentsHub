# 多人约练与场馆排程助手

交集核对多人空闲时间、场馆开放时间、容量和费用，输出候选时段与日历文件。 处理流程见两项自定义 Skill，规则与数据口径见 `knowledge.md`。

## 本地运行与验证

Python 3.11+。在独立环境安装 `python -m pip install -r requirements.txt`。

```sh
python mcps/server.py --input examples/request.json
python -m unittest discover -s tests -v
python mcps/server.py --host 0.0.0.0 --port 8111 --data-root ./runtime-data
```

样例预期：找到1个共同可行时段，为北京时间19:30–20:30，场馆费用40元；这是待确认候选安排。

工具 `fitness_group_schedule(payload: str)` 接收完整 JSON 字符串，字段结构见 `examples/request.json`。时间类输入必须带 UTC 偏移；空值、冲突或不满足约束时返回明确状态。`save_artifacts: false` 可关闭文件保存，测试使用此选项。

启用文件保存后，响应 `artifacts` 提供带随机标识的下载链接。`FITNESS_PUBLIC_BASE` 或 `--public-base` 应设置成浏览器可访问的 MCP 根地址，`FITNESS_DATA_ROOT` 或 `--data-root` 设置数据目录。下载链接按持有链接访问，应仅在可信本机或私有网络部署，公开服务需要额外鉴权。文件由使用者自行保管和清理。

## 接入 Nexent

1. 启动此 MCP，在平台添加可访问的 `http://实际主机:8111/mcp` 并刷新工具。
2. 导入 `knowledge.md`，选择可用嵌入模型，记下实际索引名。
3. 导入 Nexent 实际导出的 `agent-export.zip`（或 `agent.json`），配置可用语言模型，核对自定义工具与两项自定义 Skill。
4. 将 `knowledge_base_search` 工具及 `search-knowledge-base` Skill 的索引名称映射到新知识库。导出包携带 Skill 依赖。
5. 发布后使用样例输入验收，再换成自己的数据。导出中的来源地址是演示环境地址，跨环境部署时需重新绑定自己的 MCP。

找到1个共同可行时段，为北京时间19:30–20:30，场馆费用40元；这是待确认候选安排。

数据与输出边界以知识库说明为准，输入示例为虚构数据。工具仅执行明确范围内的核对与计算。

`display_timezone` 为 IANA 时区名，默认 UTC，示例使用 Asia/Shanghai。答复直接引用工具计算的 display_time（含偏移与时区），start/end 为 UTC 原始值，ICS 也使用 UTC。price_cents 表示本次时段总费用。

工具也支持 `{"use_example":true}` 运行包内虚构示例，并明确返回 input_scope=bundled_example；这不能代替使用者的实际数据。较长 JSON 可放到数据根目录下本助手的 inputs 文件夹，再传 `{"input_json_filename":"request.json"}`。文件大小最多1MiB；不接受目录逃逸，也不接受把输入来源选择器与内联业务字段混合。
