# dki-agents 模块接口契约（CONTRACTS）

> 所有模块实现必须严格遵守本文件。修改契约需先改本文件。
> 代码注释/文档用中文；标识符用英文。Python 3.11；依赖尽量只用已装包
> （nexent 全家桶、pyyaml、jinja2、pandas、python-dotenv、pytest、requests、jieba、loguru）。
> 读文件一律 encoding="utf-8"。所有输出目录用 `dki.paths` 中的常量。

## 0. 已完成的基础层（可直接 import，不得改动契约）

### dki.paths
- `PROJECT_ROOT, DATA_DIR, RAW_DATA_DIR, GOLD_DATA_DIR, OUTPUT_DIR, TASK_DIR: Path`
- `LLM_API_KEY, LLM_MODEL_NAME, LLM_API_URL: str`（.env 加载）
- `find_datamate_runtime() -> Path | None`
- `ensure_dirs() -> None`

### dki.dmcompat（DataMate 集成开关层）
- `Mapper, Filter, Slicer, LLM, BaseOp, FileExporter, Fields, OPERATORS`
  （datamate 可用时为真实类，否则为 dki.compat.datamate_base 中的契约 shim）
- `DATAMATE_AVAILABLE: bool`
- `load_builtin_ops() -> bool`  # 注册 DataMate 内置 46 算子（ExtraSpaceCleaner 等）
- `register_dki_ops() -> None`  # import dki.operators 触发注册

### sample 数据契约（DataMate 同构）
```python
{"text": str, "data": b"", "fileName": str, "fileType": "txt", "fileId": str,
 "filePath": str, "fileSize": str, "export_path": str, "ext_params": dict,
 "target_type": None, "instance_id": str,
 # 失败时由框架写入：
 "execute_result": False, "failed_reason": {"op_name","error_code","reason"}}
```
- 算子间结构化数据传递：**写入 `sample["text"]`（JSON 字符串）**，同时镜像到
  `sample["ext_params"][<key>]`（本地便捷访问）。平台内只有 text 列可靠流转。

### dki.pipeline
- `LocalPipelineRunner(load_builtin_ops=True, register_dki_ops=True)`
  - `.run(process, samples, name="", export_path="", preloaded=None, retry=0, keep_failed=True) -> PipelineResult`
  - `process` 形如 `[{"HtmlTagCleaner": {}}, {"MedicalEntityExtractor": {"backend":"dict"}}]`
- `PipelineResult.report: TaskReport`；`.samples`；`.ok_samples`
- `TaskReport`: `task_id/status/n_input/n_output/n_failed/n_filtered/op_metrics/summary()/to_dict()`
- `TaskStatus`: PENDING/RUNNING/COMPLETED/PARTIAL_SUCCESS/STOPPED/FAILED
- `TASKS: TaskRegistry`（`.get(task_id) -> dict | None`，`.list_ids()`）
- `make_sample(file_path=None, text="", file_id="", export_path="")`
- `samples_from_dir(input_dir, pattern="*.txt", export_path="")`
- `dki.pipeline.config.load_pipeline_config(path) -> (process, meta)`

### dki.kg
- `ENTITY_TYPES`: Disease/Symptom/Drug/Exam/Treatment/Department（中文标签见 schema.py）
- `RELATION_TYPES`: HAS_SYMPTOM 临床表现 / TREATED_BY_DRUG 治疗药物 / TREATED_BY 治疗方式 /
  DIAGNOSED_BY 诊断检查 / HAS_COMPLICATION 并发症 / CONTRAINDICATED_FOR 用药禁忌 /
  BELONGS_TO_DEPT 所属科室（含定义域/值域约束）
- `validate_triple(head, head_type, relation, tail, tail_type) -> (bool, reason)`
- `schema_description() -> str`
- `KnowledgeGraph`: `add_triple(...)`, `neighbors(name, relation=None, direction="both")`,
  `subgraph(names, hops)`, `find_path(src,dst)`, `stats()`, `top_entities(etype,k)`,
  `to_echarts_graph()`, `save(path)/load(path)`, `from_triples(list)`

## 1. 数据与词典（owner: datagen 代理）

### dki/kg/dictionaries/medical_dict.json —— 医疗词典（唯一词典文件）
```json
{
  "diseases": {
    "高血压": {"aliases": ["高血压病", "原发性高血压"], "department": "心血管内科",
      "symptoms": ["头痛", "头晕", "心悸"], "drugs": ["硝苯地平", "缬沙坦"],
      "exams": ["血压测量", "动态血压监测"], "treatments": ["生活方式干预"],
      "complications": ["脑卒中", "冠心病", "肾功能不全"]},
    ...  // ≥40 种常见病，覆盖心血管/内分泌/呼吸/消化/神经等
  },
  "symptoms": {"头痛": {"aliases": ["头部疼痛"]}, ...},          // ≥60
  "drugs": {"硝苯地平": {"aliases": ["硝苯地平缓释片"], "contraindications": ["低血压"]}, ...}, // ≥50
  "exams": {"心电图": {"aliases": ["ECG", "EKG"]}, ...},          // ≥30
  "treatments": {"经皮冠状动脉介入治疗": {"aliases": ["PCI", "冠脉介入"]}, ...}, // ≥15
  "departments": ["心血管内科", "内分泌科", ...]                   // ≥12
}
```
- 词典即本体实例层：datagen 用它生成病历 + 金标准；抽取算子用它做词典匹配。
- 别名必须有一定覆盖（缩写/俗称/英文缩写），制造抽取归一化的真实难度。

### 数据生成 dki/datagen/generate.py
- `generate_corpus(n_docs=200, seed=42, out_dir=RAW_DATA_DIR, gold_dir=GOLD_DATA_DIR) -> dict`（统计信息）
- 产出：
  - `data/raw/emr_0001.txt ... emr_NNNN.txt`：中文电子病历（章节：基本信息/主诉/现病史/
    既往史/查体/辅助检查/诊断/治疗方案/医嘱），**注入噪声**：HTML 标签、全角字符、
    多余空格、不可见字符、重复语句、手机号/身份证号（PII）、少量繁体；症状/药物用
    别名随机替换；语句模板多样化（每种关系 ≥4 种表述模板，正则可抽 + 部分仅共现可抽）。
  - `data/raw/structured/patients.csv`：patient_id,name,gender,age,region
  - `data/raw/structured/visits.csv`：visit_id,patient_id,visit_date(YYYY-MM-DD,近24个月),
    department,disease,drug_prescribed,exam,cost(浮点),los_days(住院天数,可0)
    —— 与病历文档一一对应（visit_id == 文档 fileId == emr_NNNN）
  - `data/gold/entities/emr_NNNN.json`：`[{"name","type","span":[s,e],"mention"}]`（按清洗后文本可对不上 span 时允许 span=null，name/type 必准）
  - `data/gold/triples/emr_NNNN.json`：`[{"head","head_type","relation","tail","tail_type"}]`
  - `data/gold/corpus_meta.json`：每文档使用的疾病/科室等
- 全部确定性（random.Random(seed)）。金标准 name 一律用**规范名**（非别名）。

## 2. 医疗算子（owner: operators 代理）

每个算子包目录结构（DataMate 自定义算子规范）：
```
dki/operators/<pkg>/
├── __init__.py    # try: from datamate.core.base_op import OPERATORS
                   # except ImportError: from dki.compat.datamate_base import OPERATORS
                   # OPERATORS.register_module(module_name="<ClassName>",
                   #                           module_path="dki.operators.<pkg>.process")
├── metadata.yml   # DataMate 算子市场元数据（raw_id==类名，settings 定义参数与 defaultVal）
└── process.py     # class <ClassName>(Mapper): def execute(self, sample) ...
                   # 基类导入同样 try datamate / except dki.compat
```

### 2.1 MedicalTextCleaner(Mapper) —— 医疗文本清洗（组合算子）
- 参数 settings：`steps: list[str]`（默认 ["html","invisible","fullwidth","space","dedup_sent","traditional"]），`maskPii: bool=True`
- 行为：`read_file_first(sample)` 后按 steps 顺序清洗 text。
- **必须优先复用 DataMate 内置算子实现**：HtmlTagCleaner/InvisibleCharactersCleaner/
  FullWidthCharacterCleaner/ExtraSpaceCleaner/DuplicateSentencesFilter(句子去重的 mapper)/
  TraditionalChineseCleaner/AnonymizedPhoneNumber/AnonymizedIdNumber —— 通过
  `dki.dmcompat.load_builtin_ops()` + `resolve_operator(<名>)` 实例化子算子并调用其 execute；
  DataMate 不可用时回退到自带正则实现（功能等价、可测试）。
- 输出：清洗后的 text；`ext_params["clean_steps_applied"]`。

### 2.2 MedicalSectionSplitter(Mapper)
- 把病历 text 切成章节 dict：`{"主诉": "...", "现病史": "...", "诊断": "...", ...}`
- 输出：`sample["text"] = json.dumps({"sections": {...}, "raw_len": int}, ensure_ascii=False)`；
  镜像 `ext_params["sections"]`。
- 参数：`sections: list[str]`（默认章节名列表，可扩展）。

### 2.3 MedicalEntityExtractor(Mapper)
- 输入：上游 text（若为 JSON 且含 sections 则按章节抽取并记录 section 来源；否则整文抽取）
- backend 参数：`"dict"`（默认，AC 自动机词典匹配 + 别名归一 + 最长匹配去重叠）| `"llm"`（可选增强，读 dki.llm.chat）| `"bert"`（NPU 基准用，延迟导入 torch）
- **dict 后端必须复用 DataMate 的 AC 自动机工具**：`from datamate.common.utils.aho_corasick import ...`
  （datamate 可用时；shim 环境退回自研实现——放在 process.py 内部，接口一致）。
  先读该文件确认真实 API 再用。
- 输出 JSON（写回 text + ext_params["entities"]）：
```json
{"entities": [{"name":"高血压","type":"Disease","mention":"高血压病","span":[12,16],
               "section":"诊断","source":"dict","confidence":1.0}],
 "sections": {...原样传递...}}
```
- name=规范名（词典归一），mention=原文提及。span 相对**该章节文本**或整文（记录 `span_base`）。

### 2.4 MedicalRelationExtractor(Mapper)
- 输入：EntityExtractor 的输出 JSON。
- 两级策略：
  1) **模式规则**：同句内实体对 + 触发词（如 "表现为/伴有"→HAS_SYMPTOM；"给予/服用/口服…治疗"→TREATED_BY_DRUG；"并发/继发"→HAS_COMPLICATION；"查/示/提示"→DIAGNOSED_BY；"行…术/治疗"→TREATED_BY；"禁用/慎用"→CONTRAINDICATED_FOR），confidence=0.9；
  2) **章节先验共现**：诊断章节疾病 × 治疗方案章节药物 → TREATED_BY_DRUG（confidence=0.6）等；
- `useLlm: bool=False` 参数：对低置信关系用 LLM 复核（可选）。
- 输出 JSON：`{"entities":[...], "relations":[{"head","head_type","relation","tail","tail_type","confidence","evidence":"句子","rule":"pattern:表现为"}]}`

### 2.5 TripleBuilderValidator(Mapper)
- 输入：RelationExtractor 输出。
- 行为：关系→三元组；`dki.kg.validate_triple` 逐条校验（记录淘汰原因统计）；
  文档内去重合并；`minConfidence: float=0.55` 过滤。
- 输出 JSON：
```json
{"triples":[{"head","head_type","relation","tail","tail_type","confidence","evidence","source":"<fileId>"}],
 "validation":{"n_candidates":N,"n_valid":N,"n_rejected":N,"reject_reasons":{"...":n}}}
```
- 镜像 ext_params["triples"] / ext_params["validation"]。

### 2.6 metadata.yml 必填字段（对齐 datamate/runtime/ops/README.md 与 examples/test_operator）
`raw_id`(=类名)/name/description/language:python/vendor:dki/version:0.1.0/
modal/inputs:text/outputs:text/types:[cleaning]/runtime(cpu,memory)/settings(每个参数
name/description/type/defaultVal)。参照真实示例文件格式。

### 2.7 scripts/package_operators.py
- 把每个算子包打成 `dist/operators/<pkg>.zip`（metadata.yml 在压缩包根；__init__.py 中
  module_path 重写为 `ops.user.<pkg>.process`；process.py 的 import 保持 try/except 双路径）。
- `python scripts/package_operators.py` 直接可跑。

## 3. LLM 客户端（owner: agents 代理，供 operators/analysis 复用）

### dki/llm.py
- `chat(prompt: str, system: str = "", temperature=0.2, max_tokens=2048) -> str`
  - 用 openai SDK 调 `LLM_API_URL/LLM_API_KEY/LLM_MODEL_NAME`；未配置时 raise `LlmNotConfigured`
- `llm_available() -> bool`

## 4. Nexent 智能体（owner: agents 代理）

### dki/agents/runtime.py
- `build_observer(lang="zh") -> MessageObserver`
- `build_model(observer) -> OpenAIModel`（nexent.core.models.openai_llm.OpenAIModel；
  未配置 LLM 时 raise LlmNotConfigured）
- `ScriptedModel`：smolagents Model 子类，构造入参 `responses: list[str]`，每次调用弹出
  一条；必须与 CoreAgent 兼容（研究 nexent CoreAgent 对 model 的调用面：返回 ChatMessage、
  token 计数属性等；离线 Demo/测试用）。CoreAgent 要求动作代码用 `<code>...</code>` 包裹
  （见 core_agent.py parse_code_blobs）。
- `build_agent(name, description, tools, model, observer, max_steps=8, prompt_yaml=None) -> CoreAgent`
  - prompt_yaml: dki/agents/prompts/<name>.yaml，含 duty/constraint/few_shots 三段（Jinja2
    渲染进 system_prompt 模板 dki/agents/prompts/base_system_prompt.yaml —— 参考
    nexent backend/prompts/managed_system_prompt_template_zh.yaml 的结构自行精简，
    必须包含 <code> 动作格式说明与工具清单渲染）。
- `run_agent_sync(agent, task: str) -> str`（返回 final answer；过程消息经 observer 打印到
  终端：每条 get_cached_message 解析后友好展示）。

### dki/agents/tools/*.py —— smolagents Tool 子类（nexent 风格：name/description/inputs/output_type/forward + observer 可选注入）
- `ListOperatorsTool(name="list_operators")`：inputs {category?: string}；返回 JSON：
  本项目 5 算子（读各包 metadata.yml：名称/描述/参数）+ DataMate 内置算子名清单。
- `PreviewDataTool(name="preview_data")`：inputs {path: string, n?: integer}；返回目录文件列表
  或文件前 n 字符（限 RAW_DATA_DIR/OUTPUT_DIR 内，防越界）。
- `RunPipelineTool(name="run_pipeline")`：inputs {pipeline: string(JSON 数组，元素
  {"op":name,"params":{}}), input_dir: string, pattern?: string, export_path?: string,
  name?: string}；调 LocalPipelineRunner；返回 JSON {task_id, status, summary}。
- `TaskStatusTool(name="get_task_status")`：inputs {task_id}；TASKS.get → JSON（含逐算子指标
  与错误 top3，供智能体诊断/恢复）。
- `BuildKgTool(name="build_knowledge_graph")`：inputs {triples_dir 或 task_id?}；从任务二流水线
  输出样本聚合 triples（ext_params 或输出 jsonl）→ KnowledgeGraph → 保存
  OUTPUT_DIR/"kg"/"medical_kg.json"；返回 stats JSON。
- `KgQueryTool(name="kg_query")`：inputs {action: enum[stats,entity,neighbors,subgraph,path,top,by_relation], ...}；封装 KnowledgeGraph 查询；返回 JSON。
- `Nl2SqlTool(name="nl2sql_query")`：inputs {question}; 调 dki.analysis.nl2sql → {sql, rows, columns, engine}
- `ChartTool(name="make_chart")`：inputs {chart_type: enum[bar,line,pie,graph], title, data: string(JSON), description?}；把图表加入 dki.analysis.bi.ReportBuilder 当前会话；返回确认。
- `SaveReportTool(name="save_report")`：inputs {title, summary_markdown}；渲染 HTML 报告，返回路径。
- 工具 forward 内部禁止抛裸异常——捕获后返回结构化错误 JSON（智能体可读并自我修复）。

### 三个智能体构造器（dki/agents/data_agent.py / kg_agent.py / insight_agent.py）
- `build_data_processing_agent(model=None, observer=None) -> CoreAgent`
  工具：list_operators/preview_data/run_pipeline/get_task_status
- `build_kg_agent(...)`：+ build_knowledge_graph/kg_query
- `build_insight_agent(...)`：kg_query/nl2sql_query/make_chart/save_report
- 各自 prompts yaml：duty（职责与规划流程）、constraint（算子选择约束、失败恢复策略：
  查 task_status → 调整参数/剔除坏样本 → 重跑）、few_shots（1-2 个完整 <code> 示例）。

## 5. 分析层（owner: analysis 代理）

### dki/analysis/database.py
- `build_database(db_path=OUTPUT_DIR/"insight.db", visits_csv, patients_csv, kg: KnowledgeGraph|None) -> None`
  表：patients(patient_id TEXT PK, name, gender, age INT, region)
      visits(visit_id TEXT PK, patient_id, visit_date TEXT, department, disease, drug_prescribed, exam, cost REAL, los_days INT)
      kg_entities(name TEXT, type TEXT, mention_count INT)
      kg_triples(head, head_type, relation, tail, tail_type, confidence REAL, count INT)
- `get_schema_description() -> str`（建表语句 + 每列中文注释 + 样例行，供 NL2SQL 提示）
- `run_sql(sql, db_path=...) -> (columns, rows)`（只读连接；禁止非 SELECT）

### dki/analysis/nl2sql.py
- `class NL2SQL:` `.to_sql(question) -> dict {sql, engine("rule"|"llm"), confidence}`；
  `.query(question) -> {sql, columns, rows, engine}`
- 规则引擎：jieba 分词 + 词典/正则模板，覆盖：计数/去重计数、按科室/疾病/性别/地区/
  年龄段分组、Top-K、平均/总和（cost/los_days/age）、时间趋势（按月/年）、比较、
  图谱关系查询转 SQL（kg_triples）。规则不命中→ LLM（可用时）→ 都不行返回失败说明。
- LLM 引擎：schema + few-shots + 只输出 SQL 约束；执行报错回喂重试 ≤2 次。
- SQL 安全：仅 SELECT；表/列白名单校验。

### 评测（dki/analysis/nl2sql_eval.py + data/nl2sql_eval.jsonl）
- 评测集 ≥40 条：`{"id", "question", "gold_sql"}`（覆盖上述全部模板类型 + 8 条图谱类）
- `evaluate(engine="rule"|"llm"|"auto") -> {accuracy, n, details:[...]}`：执行 gold_sql 与
  预测 SQL，结果集规范化比较（列名无关、行序无关，除非 gold 含 ORDER BY）
- **验收线：rule 引擎在评测集 accuracy ≥ 0.85**（写进 tests 断言）。

### dki/analysis/metrics.py
- `standard_metrics(db_path) -> dict`：就诊量 Top10 疾病、科室分布、月度就诊趋势、
  平均费用 Top 科室、年龄段×疾病热点、KG 核心实体度数 Top10 等（每项 {title, columns, rows}）。

### dki/analysis/bi.py
- `class ReportBuilder:` `.add_chart(chart_type, title, data, description="")`（data 规范：
  bar/line/pie: {"categories":[...], "series":[{"name","values":[...]}]}；graph: to_echarts_graph() 输出）
  `.add_markdown(md)`；`.render(title, out_path=OUTPUT_DIR/"reports"/...) -> Path`
- 模板 dki/analysis/templates/report.html.j2：单文件 HTML，ECharts 用 CDN（
  https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js）+ `<!-- 离线提示注释 -->`；
  中文排版良好；图文交替。
- `build_default_report(db_path, kg, out_path) -> Path`：用 standard_metrics + KG 图 + 模板文字
  生成完整示例报告（不依赖 LLM）。

## 6. 基准与 NPU（owner: bench 代理）
- `dki/bench/operator_bench.py`：`run_bench(input_dir=RAW_DATA_DIR, ops=[...], repeat=3) -> csv+md`
  输出 OUTPUT_DIR/"bench"/operator_bench.csv + operator_bench.md（吞吐 docs/s、MB/s、
  p50/p95 延迟；含 DataMate 内置清洗算子与本项目 5 算子对比行）。
- `dki/bench/npu/bert_ner_infer.py`：BertNerBackend（transformers AutoModelForTokenClassification，
  device 参数 cpu/cuda/npu；npu 走 torch_npu 惰性导入 + 静态 shape 分桶 padding + jit 编译缓存说明）
- `dki/bench/npu/run_npu_bench.py`：CPU/NPU 吞吐延迟对比脚本（参数化 batch/seq_len/迭代数，
  输出 JSON+MD 到 OUTPUT_DIR/"bench"/npu/）。本机无 NPU 时 `--device cpu` 可完整跑通。
- `docs/npu_optimization.md` 由 bench 代理起草：优化思路（图编译缓存/静态分桶/绑核/混合精度）、
  与 DataMate 算子 accelerator="npu" 资源声明及 base_op.use_npu() 钩子的衔接、复跑步骤。

## 7. CLI（owner: 主控，代理不改 dki/cli.py 只保证公开 API 可调）
子命令：gen-data / task1 / task2 / task3 / bench / eval-nl2sql / demo-all / agent
（各代理保证：模块级 `main()` 或纯函数可被 CLI 一行调用。）

## 8. 测试约定
- tests/test_<module>.py；不依赖网络与 LLM（LLM 路径用 monkeypatch/跳过）；
- 涉及生成数据的测试用小参数（n_docs=8）+ tmp_path；
- 断言关键语义而非实现细节。
