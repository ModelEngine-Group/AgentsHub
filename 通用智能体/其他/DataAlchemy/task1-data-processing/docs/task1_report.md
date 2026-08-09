# 任务一：数据处理智能体 — 技术报告

***

## 1. 概述

任务一目标是基于 DataMate 平台 + Nexent 智能体框架，构建一个**数据处理ETL智能体**。用户只需在Nexent聊天界面输入自然语言指令（如"帮我清洗这份数据"），智能体即可自动编排算子完成数据加载→清洗→转换→导出的全流程。

### 设计原则

- **算子即函数**：每个算子只做一件事，可独立也可串联
- **零外部依赖**：纯Python实现，词典+规则引擎，无需LLM API
- **可复现**：相同的输入产生相同的输出

***

## 2. 架构设计

```
┌──────────────────────────────────────────────────────┐
│           Nexent 数据处理智能体                        │
│         用户输入 → 任务规划 → 工具调用 → 输出结果      │
└──────────────────────────┬───────────────────────────┘
                           │ MCP 协议
                           ▼
┌──────────────────────────────────────────────────────┐
│              MCP 服务器 (server.py)                    │
│    datamate_execute_operator → 调用指定算子           │
│    datamate_run_etl_pipeline → 一键ETL全流程           │
└───────────────────┬──────────────────────────────────┘
                    │ Python 直接调用
                    ▼
┌──────────────────────────────────────────────────────┐
│              DataMate 算子层                           │
│  ┌─────────┐  ┌──────────┐  ┌──────────────┐  ┌───┐ │
│  │ data_   │→│ data_    │→│ data_        │→│exp│ │
│  │ loader  │  │ cleaner  │  │ transformer  │  │   │ │
│  └─────────┘  └──────────┘  └──────────────┘  └───┘ │
└──────────────────────────────────────────────────────┘
```

### 三个层次

| 层             | 职责             | 关键文件                                                     |
| :------------ | :------------- | :------------------------------------------------------- |
| **Nexent智能体** | NL理解、任务规划、工具编排 | `nexent_config/data_processing_agent_nexent_import.json` |
| **MCP服务器**    | 工具注册、参数转发、结果返回 | `datamate_mcp_server/server.py`                          |
| **算子层**       | 实际数据处理逻辑       | `operators/下4个算子目录`                                      |

***

## 3. 核心算子

| 算子                    | 功能                 | 核心算法                                    |
| :-------------------- | :----------------- | :-------------------------------------- |
| **data\_loader**      | 读取 CSV/JSON/TXT 数据 | 自动检测文件格式→解析→标准化为统一数据字典                  |
| **data\_cleaner**     | 去重、空值填补、数据脱敏       | 哈希去重 → 缺失值统计 → 智能填补（均值/中位数/众数） → 敏感信息脱敏 |
| **data\_transformer** | 类型标准化、字段映射         | 数据类型推断 → 格式化 → 列名标准化                    |
| **data\_exporter**    | 输出为 CSV/JSON 文件    | 字典→DataFrame→序列化→写入文件                   |

### 3.1 data\_loader

```
输入：文件路径 → 自动检测格式
├── .csv  → pandas.read_csv + 编码猜测
├── .json → json.load
├── .txt  → 逐行读取
└── 其他  → error
输出：{"success": true, "data": [...], "columns": [...], "row_count": N}
```

### 3.2 data\_cleaner

关键特性：

- **去重**：基于行哈希的精确去重
- **空值填补**：数值列→均值填补，类别列→众数填补
- **脱敏**：支持身份证/手机号/姓名等敏感信息自动掩码

### 3.3 data\_transformer

- **类型推断**：自动识别 int/float/str/datetime
- **列名标准化**：驼峰→下划线，去除非ASCII字符

### 3.4 data\_exporter

- 支持 CSV（默认）和 JSON 两种输出格式
- 包含元数据统计信息（行数、列数、耗时）

***

## 4. MCP 工具

在 `server.py` 中注册的数据处理工具：

| 工具名                         | 功能       | 参数                              |
| :-------------------------- | :------- | :------------------------------ |
| `datamate_execute_operator` | 执行单个算子   | operator\_name, sample (参数dict) |
| `datamate_run_etl_pipeline` | 一键ETL全流程 | file\_path (数据文件路径)             |

### 一键ETL流程

```python
datamate_run_etl_pipeline(file_path="病例数据.csv")
# → 自动顺序执行：loader → cleaner → transformer → exporter
# → 输出：清洗后的文件 + 元数据报告
```

***

## 5. 智能体设计与Prompt

### 角色定位

> 你是一个数据处理专家，能够理解用户的数据处理需求，自动编排算子完成数据清洗全流程。

### Prompt三段式结构

| 段                      | 内容                    |
| :--------------------- | :-------------------- |
| **duty\_prompt**       | 角色定义 + 工具说明 + ETL流程讲解 |
| **constraint\_prompt** | 调用格式约束 + 参数规则 + 错误处理  |
| **few\_shots\_prompt** | "帮我清洗病例数据"完整示例        |

### 关键约束

- 必须使用 `<RUN>` 代码块调用MCP工具（禁止写Python代码）
- params中所有key必须用双引号
- 先调 `datamate_verify_connection` 确认MCP连通
- 简单数据用 `datamate_run_etl_pipeline` 一键搞定

***

## 6. 一键ETL脚本

除了MCP调用外，还提供了独立的**本地验证脚本**：

| 文件    | 路径                                          |
| :---- | :------------------------------------------ |
| 一键ETL | `OperatorTest/ETL_demo/run_etl_pipeline.py` |

```bash
cd D:\PythonProject\OperatorTest\ETL_demo
python run_etl_pipeline.py --input 病例数据.csv --output 清洗结果.csv
```

***

## 7. Nexent部署

| 步骤 | 操作                                                                           |
| :- | :--------------------------------------------------------------------------- |
| 1  | 启动MCP服务器：`python D:\PythonProject\ModelEngine\datamate_mcp_server\server.py` |
| 2  | 打开Nexent：`http://localhost:3000`                                             |
| 3  | 导入JSON：`connectors/nexent_config/data_processing_agent_nexent_import.json`   |
| 4  | 绑定MCP：智能体设置 → 添加 `http://host.docker.internal:8089`                          |

***

## 8. 测试结果

| 测试类别  | 场景              | 结果   |
| :---- | :-------------- | :--- |
| 数据读取  | CSV/JSON/TXT 格式 | ✅ 通过 |
| 数据清洗  | 去重/空值/脱敏        | ✅ 通过 |
| 数据转换  | 类型/格式/列名        | ✅ 通过 |
| 数据导出  | CSV/JSON 输出     | ✅ 通过 |
| 一键ETL | 全流程串联           | ✅ 通过 |

***

# 9. 创新点

1. **一键ETL**：单次MCP调用完成Loader→Cleaner→Transformer→Exporter全链路，用户无需关心内部编排
2. **双模式算子**：算子同时支持DataMate容器执行和本地Python直接导入，开发测试效率提升
3. **自动格式检测**：Loader自动识别CSV/JSON/TXT编码格式，用户无需指定
4. **智能脱敏**：正则+规则库自动识别身份证号、手机号等敏感信息并掩码

***

