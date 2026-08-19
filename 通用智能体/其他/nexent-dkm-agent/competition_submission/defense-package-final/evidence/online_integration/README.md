# 在线集成证据说明

本目录保存 Nexent、DataMate、Neo4j 与三套任务 API 的真实在线证据。

## 2026-07-02 在线集成（Windows + WSL）

集成验证按 L1 任务 API → L2 DataMate → L3 Nexent → L4 Neo4j 自下而上完成，`stack_status=ready`。层级表见 [在线集成文档](../../../../docs/online_integration.md) 与答辩材料 §6.3。

- `datamate-submit-20260702-final.json`：DataMate `catalog_summary(..., mode="submit")`，模板/任务均为已验证（`verified`）。
- `task2-neo4j-live-smoke-20260702-final.json`：Neo4j Bolt 冒烟测试 `passed=true`；读回节点/边数以 JSON 为准。
- `probe-20260702-final.json` / `prepare-20260702-final.json` / `datamate-readiness-20260702-final.json`：Nexent 与 DataMate 就绪检查汇总（Nexent 为 full 模式，经注册/登录获取 JWT 后探测）。
- `openapi-submit-no-token-rerun-20260702.json` / `agent-submit-no-token-rerun-20260702.json`：Nexent OpenAPI 与 DKM Agent 最终提交证据（已验证，`verified`；OpenAPI 工具目录 `tool_count=48`）。

## 2026-07-03 复验补充（本地复跑）

- `probe-20260703-fullstack.json`：JWT 刷新后全栈 probe，`stack_status=ready`，DataMate 3/3，Nexent OpenAPI 3 服务，三套任务 API 均 available。
- `datamate-submit-20260703-rerun.json`：`catalog_summary` 在线 submit 复验（template/task 均为 `verified`）。
- `../benchmarks/task1_datamate_submit.json`：任务一 pipeline submit benchmark（修复 dest 名冲突后 `passed=true`）。

## 2026-06-18 历史快照（已被后续复跑替代）

- 同名 `-20260618-` JSON 仍保留于源目录，答辩包默认不再打包。

## 2026-06-16 历史复验

- `service-reachability-20260616.json`：Neo4j=connected，DataMate=available，Nexent=available。
- `probe-20260616.json`：DKM 在线探测成功，Nexent/DataMate stack_status=ready，task1/task2/task3 API 均 available。
- `openapi-submit-20260616.json`：Nexent OpenAPI 导入/更新结果为 `status=verified`，工具目录刷新后 tool_count=47。
- `agent-submit-20260616.json`：DKM Agent 回查结果为 `status=verified`，agent_id=1，preexisting=true。
- `task2-neo4j-live-smoke-20260616.json`：Neo4j Bolt 连接、图谱写入读回、Cypher 查询和 KG QA 均通过。
- `datamate-readiness-20260616.json`：DataMate 健康检查与算子、模板、任务核心 API 探测通过（该轮未新建清洗任务）。
- `task-api-health-20260616.json`：task1/task2/task3 三套 API health 均返回 HTTP 200。

## 更早历史补充

- `openapi-submit-live.json` / `agent-submit-live.json`：2026-06-14 Nexent 首次导入与 Agent 创建证据。
- `datamate-live-probe-20260615.json`：2026-06-15 DataMate 只读复验证据。
- `probe-live.json`：2026-06-14 Nexent 只读探测证据。

## 边界说明

非 NPU 在线集成以 2026-07-02 JSON 为主；2026-07-03 复验 JSON 见上文。2026-06-18 / 2026-06-16 及更早 JSON 保留供历史查阅。2026-07-02 Neo4j 读回为 26/29，与 §3.3 默认 demo（4 条内置样例）同输入。NPU 硬件复验见 `../npu_summary.txt` 与 `../benchmarks/` 中的 Ascend 910B3 报告（2026-06-24 快照；历史 910B2C 数值见文档记录）。
