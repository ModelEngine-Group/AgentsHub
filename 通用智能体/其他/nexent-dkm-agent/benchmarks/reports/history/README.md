# Ascend 910B2C 历史快照

本目录保存重要的 Ascend 910B2C benchmark 运行快照。父目录中的最新 JSON 可能被服务器复跑覆盖，但此处的文件应保留原始阶段标签。

命名规则：

```text
<task>_ascend_910b2c_<YYYYMMDD>_<stage>.json
```

当前阶段标签：

- `initial_runtime`：首次在服务器上验证 NPU 运行时可用的报告。
- `probe64`：首次 `npu.runtime_probe`，64×64 矩阵，5 次迭代。
- `probe256`：可配置探测，256×256 矩阵，20 次迭代。
- `device_adapter`：引入共享 `get_device()` 适配器后的复跑结果。
