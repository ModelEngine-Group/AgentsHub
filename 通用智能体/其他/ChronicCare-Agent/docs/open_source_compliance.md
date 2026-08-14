# 开源合规说明

## 1. 项目许可证

ChronicCare-Agent项目自研代码和文档采用MIT License，完整文本见根目录`LICENSE`。发布副本保留版权声明和MIT许可文本。

`NOTICE`说明项目与外部组件的集成关系；`THIRD_PARTY_NOTICES.md`记录直接运行依赖、版本、许可证、来源链接和非打包组件边界。

## 2. 外部平台与模型来源

| 组件 | 上游来源 | 当前集成基线 | 许可证/条款 | 集成方式 |
| --- | --- | --- | --- | --- |
| Nexent | [ModelEngine-Group/nexent](https://github.com/ModelEngine-Group/nexent) | `v2.2.0`，commit `a57538fb68e23d4e6b5439c3f13add014edbfd7e` | MIT | 外部Agent前端和编排平台；本项目提供MCP接入配置 |
| DataMate | [ModelEngine-Group/DataMate](https://github.com/ModelEngine-Group/DataMate) | commit `6136834ce00075f0a844e26dcc7fe1cc9e0d8dd9` | MIT | 外部mapper运行时；慢病算子增量源码位于`integrations/datamate/` |
| BGE模型 | [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) | `bge-small-zh-v1.5` | MIT | 部署方本地提供权重，交付包不包含模型文件 |
| DeepSeek API | [DeepSeek开放平台](https://platform.deepseek.com/) | 可选`deepseek-chat`接口 | 服务条款与隐私政策 | 仅在显式启用时生成候选SQL |
| Ascend驱动/CANN | [昇腾社区](https://www.hiascend.com/) | 由目标硬件环境确定 | 商业软件许可 | 目标环境提供，不属于本项目镜像 |

本次交付以适配器、MCP配置和独立算子目录集成上游项目，没有复制Nexent或DataMate完整源码。

## 3. Python依赖

运行依赖集中在`requirements.txt`并固定版本。组件许可证及PyPI/项目来源链接见`THIRD_PARTY_NOTICES.md`；硬件相关的PyTorch、`torch_npu`和Transformers由NPU环境按兼容矩阵提供。

## 4. 数据、远程模型与隐私边界

- 当前数据参考真实慢病随访业务的字段结构、业务关系和合理取值范围，采用程序化规则与固定随机种子合成，不包含可识别的真实患者身份信息，也不再分发任何原始真实数据。
- 数据不能被宣传为真实医疗机构生产数据、临床试验数据或流行病学证据。
- DeepSeek候选SQL功能默认关闭。启用后，仅发送评测问题以及白名单Schema、字段枚举、疾病中英文映射和统计口径约束；不发送整库记录、患者明细、API Key或服务器私有路径。
- 远程模型只能生成候选SQL；所有SQL仍需通过本地AST级SQL Guard并由只读SQLite执行。
- 模型权重、API Key、代理凭据和Ascend运行组件不包含在交付包中。

## 5. 免责声明

项目按MIT License“原样”提供，不提供适销性、特定用途适用性或不侵权保证。医疗相关输出仅用于数据处理、知识组织和辅助分析，不构成临床建议。
