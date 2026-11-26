---
name: quality-data-analysis
description: 只读质检数据分析流程：记录查询、合格率/趋势/车型工位统计、双针/单针/孔径专项统计、标准与BOM检索。
tags:
- quality
- data-analysis
- read-only
---

# 质检数据分析 Skill

## 目的
只读分析汽车内饰质检数据：检测记录、合格率、趋势、车型/工位分布、双针/单针/孔径专项统计；涉及标准、BOM、工艺说明时优先检索 Nexent 知识库。

## 工具使用顺序
1. 先确认问题的时间范围、任务类型、车型、工位、结果筛选条件。
2. 概览/合格率：`get_stats`；按时间窗实时记录：如工具可用则 `get_detection_summary`，否则用 `query_records` 并说明精度限制。
3. 每日趋势：`get_daily_trend`。
4. 车型/工位：`get_vehicle_stats`、`list_vehicles`。
5. 明细：`query_records`、`get_record`；双针/单针/孔径分别用对应 `query_*_records` 与 `get_*_stats`。
6. 标准：`query_standards` / `query_single_stitch_standards` / `query_hole_standards`；标准数值以工具返回为准。
7. BOM/工艺/操作规程：先调用 `knowledge_base_search` 检索 Nexent 知识库；引用时说明来源文件。
8. 单次计算：`analyze_stitch` / `analyze_single_stitch` / `analyze_hole`，只计算不保存。

## 输出规范
- 先给一句话结论，再给数据范围、统计表、异常/风险、数据缺口。
- 表格按任务类型拆分；数值保留 2 位小数并带单位。
- 明确区分：总数 / 查询数 / 合格 / 不合格 / 合格率。
- 原因分析必须标注“推测”，不能把推测写成事实。
- 安全红线：双针/单针 d<1mm 立即停机；偏差>3mm 严重；孔径椭圆度>1.5mm 模具可能损坏。

## 禁止
- 不新增、修改、删除检测记录或标准。
- 不保存分析结果。
- 不编造记录 ID、图片路径、标准值或未检索到的知识。
- 工具无结果时说明“当前数据不足”，不要猜。