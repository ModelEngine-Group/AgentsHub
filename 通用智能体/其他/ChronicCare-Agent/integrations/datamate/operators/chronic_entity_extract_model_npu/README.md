# chronic_entity_extract_model_npu

实体抽取模型型 NPU 增强算子，用于对实体抽取阶段的 BGE embedding 处理进行 NPU 加速评测。

说明：

- 该目录为交付包中的 NPU 增强算子交付源码。
- 主线 DataMate 流程仍由 DataMate 容器中的正式算子执行。
- 当评审环境缺少 NPU 运行时或本地 BGE 模型时，系统保留 CPU 主线可用，并可查看 `outputs/evaluation/npu_operator_benchmark_report.json` 中的已完成评测结果。
