# ChronicCare-Agent正式交付说明

- 发布版本：`v1.0.0`
- 交付日期：2026年8月10日

当前目录已完成部署与打包整理，包含`docs/`下的10份中文Markdown文档、1份技术报告PDF和42张经过复核的Nexent前端截图。

交付内容不包含：

- `.venv`及本地虚拟环境；
- `.git`元数据；
- 模型权重；
- 临时日志和瞬时运行产物。

运行及评测依赖统一固定在`requirements.txt`中。Ascend/CANN、PyTorch、`torch_npu`、Transformers、模型权重、Nexent和DataMate属于目标环境提供的外部集成依赖。
