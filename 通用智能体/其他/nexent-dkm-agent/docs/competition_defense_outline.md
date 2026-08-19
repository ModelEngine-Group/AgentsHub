# 答辩材料说明

口头答辩或 PPT 可参考本页索引；**阅读请使用自包含 HTML 答辩报告与证据包**。

## HTML 答辩报告（推荐）

- 正文源稿：`docs/competition_defense_document.md`
- 一键打包：`python demos/build_defense_pdf_package.py`
- 本地打包输出：`outputs/competition_evidence/defense-package-final/`
- **Git 提交目录**：`competition_submission/defense-package-final/`
- 浏览器打开：`competition_defense_document.html`（图片已 base64 内嵌，可离线阅读）

打包脚本会同步 `evidence/` 目录、生成 HTML，并自动复制到 `competition_submission/` 供仓库提交。Markdown 源稿仍保留，便于 diff 与二次编辑；**不再要求导出 PDF**。

## 证据包内容

| 目录 | 内容 |
| --- | --- |
| `competition_defense_document.html` | 自包含答辩报告（推荐提交/演示） |
| `competition_defense_document.md` | 答辩正文打包副本（源稿见 `docs/competition_defense_document.md`） |
| `evidence/screenshots/neo4j/` | Neo4j 四类 Cypher 查询结果 PNG |
| `evidence/screenshots/task3/` | 任务三 ECharts 仪表盘截图 |
| `evidence/figures/` | 架构图、闭环图、任务一/二/三 SVG 图表 |
| `evidence/html/` | 任务三报告/仪表盘 HTML；Neo4j 查询结果见 `neo4j_query_evidence.html` |
| `evidence/benchmarks/` | 抽取 / OOV / NL2SQL / Neo4j 冒烟测试 JSON |
| `evidence/logs/` | 各演示程序与 pytest 终端摘录 |
| `evidence/integration_probes/` | DataMate / Nexent / Neo4j 探测 |
| `evidence/online_integration/` | Nexent OpenAPI 导入、Agent 回查、DataMate 提交、Neo4j 冒烟测试 |
| `evidence/nexent_specs/` | 三项任务与 DKM 套件 Nexent agent spec |

## 重新采集证据

在 **Python 项目根目录** `nexent-dkm-agent/`（含 `demos/`、`tests/`、`docs/` 的目录）执行。若从 Git 仓库根进入，先 `cd nexent-dkm-agent`。

`--datamate-url none`、`--nexent-url none`、`--neo4j-uri none` 表示只复现**离线**主链路（集成报告为 `stack_status=offline`）。若环境已启动 Nexent/DataMate/Neo4j，可改为真实地址重新在线采证。分任务命令见 [任务一](task1_data_agent.md)、[任务二](task2_kg_agent.md)、[任务三](task3_analysis_agent.md)。

### 步骤 1：采集离线证据

```powershell
cd nexent-dkm-agent   # 若尚未在此目录

$timestamp = "20260703-105plus"   # 新采集时改为新时间戳；当前 Git 答辩包即此目录（见 `competition_submission/defense-package-final/manifest.json`）
$env:PYTEST_ADDOPTS = "--basetemp=$PWD\.pytest_basetemp_defense"
chcp 65001
$env:PYTHONUTF8 = "1"

python demos/collect_competition_evidence.py `
  --timestamp $timestamp `
  --datamate-url none `
  --nexent-url none `
  --neo4j-uri none `
  --include-pytest `
  --include-ruff
```

输出目录：`outputs/competition_evidence/$timestamp/`（含 `logs/pytest.txt`、演示日志、benchmark JSON 等）。

### 步骤 2：打包答辩材料

```powershell
python demos/build_defense_pdf_package.py `
  --source outputs/competition_evidence/$timestamp `
  --output outputs/competition_evidence/defense-package-final
```

**注意**：`--output` 必须指向 `outputs/competition_evidence/defense-package-final`，**不要**直接写到 `competition_submission/defense-package-final`（脚本会先清空目标目录）。打包完成后会自动**同步复制**到 `competition_submission/defense-package-final/` 供 Git 提交。

### 步骤 3（可选）：仅同步 Markdown 并重生成 HTML

证据目录已就绪、只改了 `docs/competition_defense_document.md` 时，**不要**直接 `cp` 源稿到答辩包（会保留 `../competition_submission/...` 图片路径）。应先用 `_package_markdown` 改写路径，再导出 HTML：

```powershell
python demos/export_defense_pdf.py `
  --source competition_submission/defense-package-final `
  --sync-from docs/competition_defense_document.md `
  --output competition_submission/defense-package-final/competition_defense_document.html
```

Neo4j 查询 PNG 与 HTML 与 `medical_kg.json`（26/29）不一致时，或任务三/ SVG 图表需更新时，可一键刷新全部答辩图片：

```powershell
python scripts/refresh_defense_images.py --captured-on 2026-07-03
```

等价分步（仅 Neo4j + 任务三 PNG）：

```powershell
python scripts/render_neo4j_screenshots.py --captured-on 20260703
python scripts/render_task3_screenshots.py --sync-from outputs/task3_evidence
python demos/generate_defense_figures.py --output-dir competition_submission/defense-package-final/evidence/figures ...
python demos/export_defense_pdf.py `
  --source competition_submission/defense-package-final `
  --sync-from docs/competition_defense_document.md
```

等价一行（不导出 HTML、只同步包内 Markdown）：

```powershell
python -c "from pathlib import Path; from demos.build_defense_pdf_package import _package_markdown; src=Path('docs/competition_defense_document.md'); dst=Path('competition_submission/defense-package-final/competition_defense_document.md'); dst.write_text(_package_markdown(src.read_text(encoding='utf-8')), encoding='utf-8')"
```

完整重打包（含证据同步）仍走步骤 1 → 2。

## 提交前检查

1. **pytest**（Windows 示例；Linux/Ascend 上同样执行，必要时指定可写 `--basetemp`）：

```powershell
cd nexent-dkm-agent
$env:PYTEST_ADDOPTS = "--basetemp=$PWD\.pytest_basetemp"
chcp 65001
$env:PYTHONUTF8 = "1"
python -m pytest -q
```

2. **ruff**：`python -m ruff check .`

3. **证据与答辩包一致**：pytest 数量变化时，按「步骤 1 → 步骤 2」重采并打包，使 `evidence/logs/pytest.txt` 与 HTML 一致（当前 Windows 主链路 **437/437**）。

## 相关文档

- [技术答辩材料](competition_defense_document.md)
- [架构说明](architecture.md)
- [NPU 优化说明](npu_optimization.md)（Ascend 服务器复测入口）
