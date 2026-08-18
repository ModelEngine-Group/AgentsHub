# ChronicCare NPU Readiness

- status: `success`
- npu_available: `False`
- backend: `cpu_fallback`
- fallback_enabled: `True`

## Recommended NPU Targets
- chronic_entity_extract
- chronic_relation_extract
- text_embedding
- open_nl2sql_model_inference

## Notes
- torch import failed: No module named 'torch'
- torch_npu import failed: No module named 'torch_npu'
- npu-smi info is not executable in the current runtime
- no Ascend device nodes are visible in the current runtime
- no local NPU model service detected on 18080/18081

安全声明：本项目仅用于慢病随访数据处理、知识组织和辅助分析，不提供临床诊断，不替代医生决策。
