# 答辩材料包

## 阅读方式

1. 用浏览器打开 `competition_defense_document.html`（推荐，图片已内嵌）。
2. 如需编辑正文，修改 `docs/competition_defense_document.md`，再按 [答辩打包说明](../../docs/competition_defense_outline.md) 重采或仅重生成 HTML。

## 证据再生流程（维护者）

完整步骤见 `docs/competition_defense_outline.md` §重新采集证据：

1. `python demos/collect_competition_evidence.py` → `outputs/competition_evidence/$timestamp/`
2. `python demos/build_defense_pdf_package.py --source ... --output outputs/competition_evidence/defense-package-final`（勿直接 `--output competition_submission/...`）
3. 脚本自动同步到本目录；仅改正文时使用 `export_defense_pdf.py --sync-from docs/competition_defense_document.md`（见下方快速同步）

快速同步（仅改正文、证据未变）：

```powershell
python demos/export_defense_pdf.py `
  --source competition_submission/defense-package-final `
  --sync-from docs/competition_defense_document.md
```

## 证据时间线

- Windows 离线代码与最终回归：2026-07-03 证据包，pytest **437/437 passed**（`evidence/logs/pytest.txt`），ruff 全量通过。
- Nexent/DataMate/Neo4j 非 NPU 在线集成：2026-07-02 JSON 见 `evidence/online_integration/`；2026-07-03 全栈 probe 与 DataMate submit 复验 JSON 见该目录 README；2026-06-18 历史快照未打入本包（源目录 `outputs/competition_evidence/online-integration/`）；2026-06-16 及更早 JSON 在本目录。
- NPU：2026-06-24 Ascend 910B3 快照；插卡前无卡预配置 **737/737**，插卡后 **770/770**（NPU 专项 43/43），关系级 NPU P/R/F1=1.0（46/46，10 条标注病历；30 条 CPU 张量 145/145，NPU 30 条待 Ascend 复跑）；`cached_topk_labels` 99.95×、`cached_bincount_topk` 27.77×。

详细边界见 `evidence/online_integration/README.md` 和 `evidence/npu_summary.txt`。

## 目录

- `competition_defense_document.html` - 自包含答辩报告
- `competition_defense_document.md` - 答辩正文打包副本（编辑请改 `docs/` 源稿后同步）
- `evidence/screenshots/neo4j/` - Neo4j 查询结果 PNG
- `evidence/screenshots/task3/` - 任务三仪表盘截图
- `evidence/figures/` - 架构图与任务 SVG 图表
- `evidence/html/` - Neo4j 查询 HTML（`neo4j_query_evidence.html`）与任务三报告/仪表盘
- `evidence/benchmarks/` - 量化评测 JSON
- `evidence/online_integration/` - Nexent/DataMate/Neo4j 在线证据
- `evidence/logs/` - 演示程序 / pytest 终端摘录

来源证据包：`outputs/competition_evidence/20260703-105plus`
