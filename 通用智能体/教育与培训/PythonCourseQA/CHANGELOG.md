# CHANGELOG

本文件记录 python课程答疑小助手 的版本变更，版本号遵循语义化版本（MAJOR.MINOR.PATCH）。

## 1.0.0 — 2026-11-17

### 新增

- 首次提交至 AgentsHub（`通用智能体/教育与培训/PythonCourseQA`）。
- `agent.json`：Nexent 平台导出的完整智能体配置，包含系统提示词、三类内置工具、模型参数、欢迎语与 5 条示例问题。
- `skills/course-qa-assistant.zip`：自定义课程答疑 Skill，含 `SKILL.md` 工作流与 `example.md` 问答示例。
- `README.md`：功能说明、依赖清单、导入步骤、回归用例与局限说明。

### 脱敏

- `tenant_id` 置空；`author` 由个人邮箱替换为 `Community Contributor`。
- 确认不含 API Key、Token、租户密钥或外部服务地址。

### 待导入者配置

- 目标环境的大语言模型（原环境为 `大肥鱼V4` / `大肥鱼F1`）。
- `knowledge_base_search` 的 `index_names` 需替换为目标环境的课程知识库索引。
- 需先上传 Skill 包再导入智能体本体。
