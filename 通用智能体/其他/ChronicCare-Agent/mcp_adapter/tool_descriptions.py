from __future__ import annotations

from typing import Dict, List

from mcp_adapter.schemas import ToolDefinition, ToolSchema


def _desc(
    *,
    purpose: str,
    applicable: str,
    forbidden: str,
    input_format: str,
    output_format: str,
    examples: str,
    notes: str,
) -> str:
    return (
        f"用途：{purpose}\n"
        f"适用问题：{applicable}\n"
        f"禁止使用场景：{forbidden}\n"
        f"输入格式：{input_format}\n"
        f"输出格式：{output_format}\n"
        f"示例问题：{examples}\n"
        f"注意事项：{notes}"
    )


TOOL_DEFINITIONS: List[ToolDefinition] = [
    ToolDefinition(name="chroniccare_datamate_dag_plan", title="Plan Dynamic DataMate DAG", description=_desc(purpose="按目标生成动态 DataMate DAG 或 dry-run。", applicable="只清洗、只重建图谱、只刷新分析库、完整链路、NPU 增强计划。", forbidden="不要用它伪装真实执行。", input_format="goal、input_path、use_npu。", output_format="返回 Profile、节点依赖、跳过原因、资源和风险。", examples="如果只重建知识图谱，准备调用哪些算子？", notes="计划不修改业务产物。"), input_schema=ToolSchema(properties={"goal":{"type":"string"},"input_path":{"type":"string"},"use_npu":{"type":"boolean"}})),
    ToolDefinition(name="chroniccare_datamate_dag_run", title="Run Dynamic DataMate DAG", description=_desc(purpose="执行或 dry-run 动态 DAG。", applicable="按目标执行数据处理或验证计划。", forbidden="不要用旧报告冒充本次运行。", input_format="goal、use_npu、dry_run。", output_format="返回 run_id 和节点状态/hash。", examples="只刷新分析库并执行。", notes="dry_run 零写入。"), input_schema=ToolSchema(properties={"goal":{"type":"string"},"use_npu":{"type":"boolean"},"dry_run":{"type":"boolean"}})),
    ToolDefinition(name="chroniccare_datamate_dag_resume", title="Resume Dynamic DataMate DAG", description=_desc(purpose="从失败节点恢复 DAG。", applicable="修复条件后继续运行。", forbidden="输入或版本变化时不得复用失效缓存。", input_format="resume_run_id、resume_from、goal。", output_format="返回恢复后的节点状态。", examples="从三元组校验节点继续。", notes="必须提供 run_id。"), input_schema=ToolSchema(properties={"resume_run_id":{"type":"string"},"resume_from":{"type":"string"},"goal":{"type":"string"}},required=["resume_run_id"])),
    ToolDefinition(name="chroniccare_datamate_dag_status", title="Dynamic DAG Status", description=_desc(purpose="读取 DAG 状态或依赖图。", applicable="查询某 run_id 的节点和 DAG。", forbidden="不执行或修改任务。", input_format="run_id、include_dag。", output_format="运行状态或 DAG JSON。", examples="查看这次运行的失败节点。", notes="只读。"), input_schema=ToolSchema(properties={"run_id":{"type":"string"},"include_dag":{"type":"boolean"}},required=["run_id"])),
    ToolDefinition(
        name="chroniccare_health_check",
        title="ChronicCare Tool Server Health Check",
        description=_desc(
            purpose="检查 ChronicCare Tool Server 是否正常运行。",
            applicable="系统是否正常、服务是否可用、健康状态。",
            forbidden="不要用它回答患者数、疾病分布、图谱子图或 DataMate 处理结果。",
            input_format="无需参数。",
            output_format="返回 status、project、base_url、message。",
            examples="系统现在是否正常运行？",
            notes="回答时应明确说明结果来自 chroniccare_health_check。",
        ),
        input_schema=ToolSchema(),
    ),
    ToolDefinition(
        name="chroniccare_data_summary",
        title="ChronicCare System Data Summary",
        description=_desc(
            purpose="返回当前真实数据规模，包括患者、随访、检验、用药和图谱节点边。",
            applicable="当前数据规模是多少、现在有多少患者随访记录检验记录、系统主线规模。",
            forbidden="不要用它回答疾病分布、风险分层、指标均值、图谱关系或子图生成。",
            input_format="无需参数。",
            output_format="返回 patient_count、visit_count、lab_result_count、medication_record_count、node_count、edge_count、table、summary_text；回答时不要展示质量评分。",
            examples="当前数据规模是多少？；现在有多少患者、随访记录、检验记录？",
            notes="所有数值必须直接复用工具返回结果，不能用旧模板值。",
        ),
        input_schema=ToolSchema(),
    ),
    ToolDefinition(
        name="chroniccare_datamate_pipelines",
        title="ChronicCare DataMate Pipelines Overview",
        description=_desc(
            purpose="查看 ChronicCare 当前三条 DataMate CPU/通用主线 pipeline、11 个主线算子、逻辑分组和最近运行摘要。",
            applicable="三条 pipeline 是什么、11 个主线算子如何分组、当前 DataMate 主线如何拆分、现在 ChronicCare 支持哪些算子、现在 ChronicCare 支持哪些 CPU 算子。",
            forbidden="不要用它回答“哪些算子支持 NPU / NPU 算子有哪些 / NPU 加速覆盖哪些算子”；这类问题必须用 chroniccare_npu_supported_operators。",
            input_format="无需参数。",
            output_format="返回 pipelines、operator_count、latest_run、invocation_mode。",
            examples="当前 DataMate 主线怎么拆分？",
            notes="这 11 个是 DataMate 主线 CPU/通用算子，不等于 NPU 算子；回答 NPU 支持范围时禁止引用这里的 11 个算子。",
        ),
        input_schema=ToolSchema(),
    ),
    ToolDefinition(
        name="chroniccare_datamate_pipeline_run",
        title="ChronicCare DataMate Pipeline Run",
        description=_desc(
            purpose="真实触发慢病 DataMate CPU/通用全流程重跑，并返回本轮执行结果与清洗摘要。",
            applicable="请重新处理慢病数据、请重新执行 DataMate 算子链路、运行慢病原始数据的数据处理流程并返回清洗摘要。",
            forbidden="不要用它回答当前疾病有哪些、图谱节点边、知识图谱子图或指标均值。",
            input_format="可传 task_id、force、safe_run。",
            output_format="返回 run_id、summary、steps、pure_operator_duration_seconds、outer_elapsed_seconds、artifact_paths、pipeline_browser_url。",
            examples="请运行慢病原始数据的数据处理流程，并返回清洗摘要。",
            notes="用户说重新运行/重新执行时必须等待本轮执行 Observation，不能读取旧报告秒回；回答时要区分 11 个算子纯执行耗时和外层完整流程耗时。",
        ),
        input_schema=ToolSchema(
            properties={
                "task_id": {"type": "string", "description": "任务 id。"},
                "force": {"type": "boolean", "description": "是否强制重跑。"},
                "safe_run": {"type": "boolean", "description": "是否使用 safe-run。"},
            }
        ),
    ),
    ToolDefinition(
        name="chroniccare_npu_readiness",
        title="ChronicCare NPU Runtime Readiness",
        description=_desc(
            purpose="检测当前慢病项目是否具备 NPU runtime、torch_npu/CANN/npu-smi 或本地 NPU 模型服务能力。",
            applicable="NPU 是否可用、当前有没有 NPU 环境、NPU runtime 检查、是否会 fallback。",
            forbidden="不要用它回答疾病分布、图谱查询或执行 DataMate 主线。",
            input_format="无需参数。",
            output_format="返回 npu_available、backend、fallback_required、checks、supported_operators、报告路径。",
            examples="现在 NPU 能用吗？；检查一下 NPU runtime。",
            notes="若返回 fallback_required=true，最终回答必须明确说明没有声明 NPU 加速效果。",
        ),
        input_schema=ToolSchema(),
    ),
    ToolDefinition(
        name="chroniccare_npu_supported_operators",
        title="ChronicCare NPU Supported Operators",
        description=_desc(
            purpose="列出当前已接入 NPU 增强分支的 DataMate 算子。",
            applicable="哪些算子支持 NPU、NPU 算子有哪些、NPU 加速覆盖哪些算子、实体抽取和关系抽取是否支持 NPU。",
            forbidden="不要用它运行 benchmark 或回答真实业务分析结果。",
            input_format="无需参数。",
            output_format="返回 supported_operators、cpu_operator、project_wrapper、fallback_policy。",
            examples="哪些慢病算子已经支持 NPU？",
            notes="当前只能说 2 个 NPU 增强算子：chronic_entity_extract_model_npu、chronic_relation_extract_model_npu。严禁把 DataMate 主线 11 个 CPU/通用算子说成 NPU 算子。",
        ),
        input_schema=ToolSchema(),
    ),
    ToolDefinition(
        name="chroniccare_npu_operator_benchmark",
        title="ChronicCare NPU Operator Benchmark",
        description=_desc(
            purpose="运行实体抽取/关系抽取 NPU 增强分支 benchmark，并生成报告。",
            applicable="展示最近一次 NPU benchmark、NPU 加速效果如何、跑一下 NPU benchmark、实体抽取和关系抽取耗时对比、CPU BGE 抽样耗时与 NPU BGE 抽样耗时对比。",
            forbidden="不要用它回答患者统计、疾病分布、知识图谱子图；NPU 不可用时不要声称有加速比。",
            input_format="可传 use_npu、fallback、force_run；force_run 默认为 false，表示只读取最近一次缓存报告。",
            output_format="返回 runtime、operator_results、npu_comparison_rows、fallback_used、report_path、markdown_report_path。",
            examples="展示最近一次 NPU benchmark 结果；重新跑一次 NPU 算子 benchmark。",
            notes="用户只问最近一次/展示结果时不要重跑；只有用户明确说重新跑/跑一下时才传 force_run=true。每个算子使用四列表格：指标、CPU（2048条）、NPU（2048条）、NPU（全量），其中后三列是三个独立实测结果列。三组分别真实执行；CPU 与 NPU 正式测量前各预热一次且预热不计时；NPU 两组均为 batch 1024，并各自独立启停采样器。每列分别展示自身处理量、BGE实测耗时、吞吐量、平均单条延迟、资源、功耗和能耗，NPU 2048 条与全量禁止复用综合采样值；同样本加速比只比较 CPU/NPU 2048 条，NPU 全量列写不适用。默认不要展示 sidecar 路径。CPU 功耗/能耗未采集时必须写未采集。只有真实 NPU 分支成功时才可引用 speedup；fallback 场景必须说明未产生真实 NPU 加速数据。",
        ),
        input_schema=ToolSchema(
            properties={
                "use_npu": {"type": "boolean", "description": "是否尝试 NPU 分支。"},
                "fallback": {"type": "boolean", "description": "NPU 不可用时是否回退 CPU artifact。"},
                "force_run": {"type": "boolean", "description": "是否重新运行 benchmark；默认 false 只读取最近一次缓存报告。"},
            }
        ),
    ),
    ToolDefinition(
        name="chroniccare_datamate_pipeline_run_npu",
        title="ChronicCare DataMate Pipeline Run With NPU Enhancement",
        description=_desc(
            purpose="在保留 DataMate CPU 规则召回主线的前提下，用 NPU 运行实体候选 BGE 标准化、关系候选 BGE 重排/过滤增强分支。",
            applicable="运行 NPU 增强流水线、用 NPU 跑慢病数据处理、执行 DataMate 并启用 NPU、查看 CPU BGE 与 NPU BGE 耗时对比。",
            forbidden="不要用它回答普通疾病统计或只查询 pipeline 状态；不要把 fallback 结果说成 NPU 加速成功。",
            input_format="可传 task_id、force、safe_run、npu_targets、fallback。",
            output_format="返回 base_pipeline、npu_benchmark、npu_comparison_rows、fallback_used、report_path、markdown_report_path。",
            examples="请启用 NPU 跑一次慢病 DataMate pipeline。",
            notes="用户说“运行/执行/重新运行 NPU 全流程”时必须传 force=true 并等待本轮结果；只有用户明确说“展示/读取最近一次报告”时才读取缓存。回答必须引用 npu_comparison_rows；每个算子使用四列表格：指标、CPU（2048条）、NPU（2048条）、NPU（全量），其中后三列是三个独立实测结果列。三组分别真实执行；CPU 与 NPU 正式测量前各预热一次且预热不计时；NPU 两组均为 batch 1024，并各自独立启停采样器。每列分别展示自身处理量、BGE实测耗时、吞吐量、平均单条延迟、资源、功耗和能耗，NPU 2048 条与全量禁止复用综合采样值；同样本加速比只比较 CPU/NPU 2048 条，NPU 全量列写不适用。默认不要展示 sidecar 路径。CPU 功耗/能耗未采集时必须写未采集。",
        ),
        input_schema=ToolSchema(
            properties={
                "task_id": {"type": "string", "description": "任务 id。"},
                "force": {"type": "boolean", "description": "是否真实重跑 NPU 全流程；运行类请求默认 true。"},
                "safe_run": {"type": "boolean", "description": "是否使用 safe-run。"},
                "npu_targets": {"type": "array", "items": {"type": "string"}, "description": "NPU 增强算子列表。"},
                "fallback": {"type": "boolean", "description": "NPU 不可用时是否回退 CPU artifact。"},
            }
        ),
    ),
    ToolDefinition(
        name="chroniccare_datamate_pipeline_status",
        title="ChronicCare DataMate Pipeline Status",
        description=_desc(
            purpose="查询最近一次 DataMate 流水线状态、11 个算子状态和同步状态。",
            applicable="哪些算子执行成功了、11 个算子分别耗时多少、DataMate 处理结果同步了吗。",
            forbidden="不要用它代替 pipeline_run 执行流程，也不要回答疾病分布和图谱关系。",
            input_format="无需参数，默认查询 latest。",
            output_format="返回 run_id、steps、summary、sync_status、pure_operator_duration_seconds、outer_elapsed_seconds。",
            examples="11 个算子分别耗时多少？；DataMate 处理结果同步了吗？",
            notes="回答时不要空泛写“全部成功 100%”，要复述真实状态和耗时。",
        ),
        input_schema=ToolSchema(),
    ),
    ToolDefinition(
        name="chroniccare_datamate_pipeline_latest",
        title="ChronicCare DataMate Pipeline Latest",
        description=_desc(
            purpose="返回最近一次主线 DataMate pipeline 的轻量摘要。",
            applicable="最近一次主线结果、最近一次同步状态、最新运行摘要。",
            forbidden="不要用它回答疾病统计、风险分布或图谱子图。",
            input_format="无需参数。",
            output_format="返回 run_id、summary、artifact_paths、counts、metrics、check_report_path。",
            examples="最近一次 DataMate pipeline 状态如何？",
            notes="适合轻量摘要，不替代完整 run/status/report 工具。",
        ),
        input_schema=ToolSchema(),
    ),
    ToolDefinition(
        name="chroniccare_datamate_pipeline_report",
        title="ChronicCare DataMate Pipeline Report",
        description=_desc(
            purpose="读取最近一次 DataMate full pipeline 运行报告、同步报告和检查报告。",
            applicable="查看最新 pipeline 报告、检查报告、同步报告。",
            forbidden="不要用它回答疾病分布、图谱关系、患者指标分析。",
            input_format="无需参数。",
            output_format="返回 report_path、check_report_path、summary、artifact_paths。",
            examples="给我最新的 DataMate 运行报告。",
            notes="回答时优先展示报告入口或关键结论，不要编造额外状态。",
        ),
        input_schema=ToolSchema(),
    ),
    ToolDefinition(
        name="chroniccare_kg_summary",
        title="ChronicCare KG Summary",
        description=_desc(
            purpose="返回当前知识图谱节点边、实体关系类型分布和图谱入口。",
            applicable="当前知识图谱有多少节点和边、实体类型和关系类型、全局图谱概览。",
            forbidden="不要用它回答高血压的知识图谱子图、糖尿病的知识图谱子图、某群体图谱子图、疾病分布或 DataMate 流水线问题。",
            input_format="无需参数。",
            output_format="返回 node_count、edge_count、entity_type_total_count、relation_type_total_count、top_entity_types、top_relation_types、graph_url。",
            examples="当前知识图谱有多少节点和边？",
            notes="图谱节点边等规模必须来自真实图谱摘要，不展示图谱评分；graph_url 指向不加载全部节点的轻量概览页。如果用户在问某疾病/某群体知识图谱子图，必须改用 chroniccare_kg_subgraph_render。",
        ),
        input_schema=ToolSchema(),
    ),
    ToolDefinition(
        name="chroniccare_kg_entity_query",
        title="ChronicCare KG Entity Query",
        description=_desc(
            purpose="查询某疾病或实体关联的检查指标、药物、风险事件。",
            applicable="高血压关联哪些检查指标、糖尿病关联哪些药物、高血压关联哪些风险事件。",
            forbidden="不要用于疾病分布、风险等级分布、图谱子图页面生成。",
            input_format="传入 query。",
            output_format="返回 table、associated_indicators、associated_drugs、associated_risk_events、evidence、graph_url；table 中的覆盖患者数是该疾病群体内去重患者数，检验记录数/用药记录数/事件记录数是记录条数。",
            examples="高血压关联哪些检查指标？",
            notes="必须只按当前用户这一次输入的疾病名调用，不要继承上一轮疾病或共同患者条件；若 table 存在，最终回答必须优先展示 table；不要把记录数当患者数，不要复用上一轮问题的数字，不要编造 1200/1500 等模板值。",
        ),
        input_schema=ToolSchema(
            properties={"query": {"type": "string", "description": "自然语言图谱实体问题。"}},
            required=["query"],
        ),
    ),
    ToolDefinition(
        name="chroniccare_kg_relation_query",
        title="ChronicCare KG Relation Query",
        description=_desc(
            purpose="查询疾病组合、生活方式与指标异常之间的关系。",
            applicable="高血压和糖尿病共同关联哪些指标、高盐饮食和血压异常有什么关系。注意：单病种的药物/风险事件/检查指标问题必须使用 chroniccare_kg_entity_query。",
            forbidden="不要用于患者数量统计、疾病分布或 DataMate 清洗摘要。",
            input_format="传入 query。",
            output_format="返回 shared_indicators、shared_risk_events、table、evidence、graph_url；table 中的覆盖患者数是去重患者数，record_count/检验记录数/用药记录数是记录条数。",
            examples="高盐饮食和血压异常有什么关系？",
            notes="结果必须来自图谱或主线产物，不能自由联想；如果 table 存在，最终回答必须优先展示 table，不要把记录数当成患者数，不要自行生成 1500/1200 等模板值；不要因为上一轮问过共同关联，就把下一轮单病种问题改写成共同患者问题。",
        ),
        input_schema=ToolSchema(
            properties={"query": {"type": "string", "description": "自然语言图谱关系问题。"}},
            required=["query"],
        ),
    ),
    ToolDefinition(
        name="chroniccare_kg_patient_path_query",
        title="ChronicCare KG Patient Path Query",
        description=_desc(
            purpose="查询指定患者在图谱中的风险事件、随访计划和路径。",
            applicable="P0001 有哪些风险事件、某患者未来有哪些随访计划；验收问法“某个患者有哪些风险事件/某个患者未来有哪些随访计划”可用 P0001 作为示例患者。",
            forbidden="不要用它回答疾病分布和图谱概览；除“某个患者”验收占位外，没有明确患者 ID 时不要编造成果。",
            input_format="传入 patient_id。",
            output_format="返回 patient_id、risk_events、followup_plans、table、graph_url。",
            examples="P0001 未来有哪些随访计划？",
            notes="若用户写的是“某个患者”这类验收占位，可传 patient_id=P0001 并说明这是示例患者；正式查询应提示用户提供有效患者编号。",
        ),
        input_schema=ToolSchema(
            properties={"patient_id": {"type": "string", "description": "患者 id，例如 P0001。"}},
            required=["patient_id"],
        ),
    ),
    ToolDefinition(
        name="chroniccare_kg_subgraph_query",
        title="ChronicCare KG Subgraph Query",
        description=_desc(
            purpose="生成局部子图的结构化节点边结果，适合调试或 JSON 场景。",
            applicable="用户明确要求节点边 JSON、结构化子图数据。",
            forbidden="用户直接要图谱页面、子图可视化或打开 HTML 时不要用它，改用 chroniccare_kg_subgraph_render。",
            input_format="传入 query，可选 max_nodes。",
            output_format="返回 nodes、edges、html_url、graph_scope_explanation。",
            examples="只返回高血压子图的节点边 JSON。",
            notes="不要只拿结构化结果再自行拼接链接。",
        ),
        input_schema=ToolSchema(
            properties={
                "query": {"type": "string", "description": "自然语言子图问题。"},
                "disease": {"type": "string", "description": "兼容字段。若前端只传疾病名，如“高血压”，系统会自动补成“高血压的知识图谱子图”。"},
                "cohort_name": {"type": "string", "description": "兼容字段。若前端只传群体名，如“冠心病风险”，系统会自动补成对应的知识图谱子图请求。"},
                "subgraph_type": {"type": "string", "description": "兼容字段。若传入 risk，系统会优先按风险主题补全查询，如把“冠心病”补成“冠心病风险知识图谱子图”。"},
                "max_nodes": {"type": "integer", "description": "最大节点数。"},
            },
        ),
    ),
    ToolDefinition(
        name="chroniccare_kg_subgraph_render",
        title="ChronicCare KG Subgraph Render",
        description=_desc(
            purpose="根据当前问题实时生成知识图谱子图 HTML，并返回可访问 HTTP URL。",
            applicable="高血压的知识图谱子图、糖尿病知识图谱子图、任意疾病/风险/生活方式主题的知识图谱子图、生成高血压合并糖尿病群体的图谱子图、画出高盐饮食和血压异常之间的关系。",
            forbidden="不要用于系统能力说明、疾病人数分布、风险人数统计或 DataMate 流程摘要。",
            input_format="传入 query，可选 max_nodes。",
            output_format="返回 preview_url 与 preview_route_path（SVG 图片预览）、html_url/html_route_path/graph_url（交互式 HTML）、node_count、edge_count、seed_labels、graph_scope_explanation。",
            examples="高血压的知识图谱子图",
            notes="这是“某疾病/某主题知识图谱子图”这类问题的首选工具；最终回答必须先展示 preview_url 对应的 SVG 图片预览，再给 html_url/graph_url 交互式 HTML 链接。优先使用 preview_url/html_url 这类完整 HTTP URL；不要把 .html 当图片，不要输出相对路径或内部 outputs 路径，也不能只输出“点击查看”而没有真实 href。",
        ),
        input_schema=ToolSchema(
            properties={
                "query": {"type": "string", "description": "自然语言子图渲染问题。"},
                "disease": {"type": "string", "description": "兼容字段。若前端只传疾病名，如“高血压”，系统会自动补成“高血压的知识图谱子图”。"},
                "cohort_name": {"type": "string", "description": "兼容字段。若前端只传群体名，如“冠心病风险”，系统会自动补成对应的知识图谱子图请求。"},
                "subgraph_type": {"type": "string", "description": "兼容字段。若传入 risk，系统会优先按风险主题补全查询，如把“冠心病”补成“冠心病风险知识图谱子图”。"},
                "max_nodes": {"type": "integer", "description": "最大节点数。"},
            },
        ),
    ),
    ToolDefinition(
        name="chroniccare_analysis_query",
        title="ChronicCare Analysis Query",
        description=_desc(
            purpose="处理老的标准化分析问题接口，主要兼容固定问答。",
            applicable="固定指标问答兼容场景。",
            forbidden="不要优先用于疾病分布、风险分层、未来 N 天随访、高血压的知识图谱子图、糖尿病的知识图谱子图和图表可视化请求。",
            input_format="传入 question。",
            output_format="返回 matched_id、metric、table、summary_text、chart_url、report_url。",
            examples="高脂血症患者的 LDL-C 异常比例是多少？",
            notes="新接入前端时应优先考虑专用工具或 open_analysis_query 分支；如果问题包含图谱/子图/关系图，应改用 chroniccare_kg_subgraph_render。",
        ),
        input_schema=ToolSchema(
            properties={"question": {"type": "string", "description": "标准化分析问题。"}},
            required=["question"],
        ),
    ),
    ToolDefinition(
        name="chroniccare_disease_distribution",
        title="ChronicCare Disease Distribution",
        description=_desc(
            purpose="专门回答疾病分布、疾病类型、常见病、单病人数占比问题。",
            applicable="疾病分布、当前常见病有哪些、当前数据中有哪些疾病、高血压患者有多少、糖尿病患者有多少、高脂血症患者占比是多少。",
            forbidden="不得用于风险等级分布、高风险患者有多少、HbA1c 平均值、LDL-C 异常比例、图谱子图生成。",
            input_format="传入 question。",
            output_format="返回 disease_labels、table、metric、summary_text、chart_url、report_url、graph_url；单病问题的 metric.value 就是该病患者人数。",
            examples="当前常见病有哪些？；不同疾病患者人数分布是多少？",
            notes="若返回 disease_labels 或疾病表格，回答时必须列出全部疾病类型，不能只摘前 4-5 个。单病人数问题必须使用表格中的“患者人数”和 metric.value，禁止回答知识图谱患者数、节点数、DataMate 汇总口径或 3305 这类其它口径。",
        ),
        input_schema=ToolSchema(
            properties={"question": {"type": "string", "description": "疾病分布问题。"}},
            required=["question"],
        ),
    ),
    ToolDefinition(
        name="chroniccare_disease_combination_distribution",
        title="ChronicCare Disease Combination Distribution",
        description=_desc(
            purpose="专门回答疾病组合分布、共病人数统计问题。",
            applicable="不同疾病组合的人数分布是多少、多病共病患者有多少。",
            forbidden="不要用于单病人数分布、风险分层人数、指标均值或图谱关系。",
            input_format="传入 question。",
            output_format="返回 table、summary_text、chart_url、report_url。",
            examples="不同疾病组合的人数分布是多少？",
            notes="一个患者可同时属于多个慢病组合，回答时要保留组合语义。最终必须展示 Top 12 精确多病组合和“其他多病组合”汇总行，表格患者数合计必须等于多病共病患者总数；禁止混入单病行。",
        ),
        input_schema=ToolSchema(
            properties={"question": {"type": "string", "description": "疾病组合分布问题。"}},
            required=["question"],
        ),
    ),
    ToolDefinition(
        name="chroniccare_risk_level_distribution",
        title="ChronicCare Risk Level Distribution",
        description=_desc(
            purpose="专门回答高风险、中风险、低风险人数和风险分层分布。",
            applicable="不同风险等级患者人数分布是多少、高风险患者有多少。",
            forbidden="不得用于疾病分布、常见病、单病人数、未来 N 天需要随访的高风险患者人数、指标趋势或图谱子图。",
            input_format="传入 question。",
            output_format="返回 risk_distribution_rows、table、summary_text、chart_url、report_url。",
            examples="不同风险等级患者人数分布是多少？",
            notes="风险分布和疾病分布必须强区分，不能混答；“未来 N 天需要随访的高风险患者有多少”必须改用 chroniccare_followup_high_risk。禁止使用 120/360/720、1200 或低风险 60% 这类旧模板值。",
        ),
        input_schema=ToolSchema(
            properties={"question": {"type": "string", "description": "风险等级分布问题。"}},
            required=["question"],
        ),
    ),
    ToolDefinition(
        name="chroniccare_followup_high_risk",
        title="ChronicCare Future High Risk Followup",
        description=_desc(
            purpose="回答未来 1-200 天随访人数；未明确疾病时按验收口径统计高风险随访队列，并返回逐日趋势和图表。",
            applicable="未来 1 天/7 天/15 天/30 天/120 天/200 天随访人数、未来 N 天需要随访多少人、未来 N 天需要随访的高风险患者有多少、未来 N 天随访人数趋势图。",
            forbidden="不要用于疾病目录、风险等级总分布、图谱实体关系或 DataMate 运行问题。",
            input_format="传入 question。",
            output_format="返回 canonical_id、window_days、cohort_patient_count、trend_rows/daily_counts、risk_distribution_rows、charts、chart_url、report_url。",
            examples="未来 15 天随访人数？；未来 30 天需要随访的高风险患者有多少？；未来 60 天随访人数趋势图。",
            notes="必须严格使用用户问句里的真实天数，支持 1-200 天任意窗口；未写疾病名时不要转去 Open SQL 或风险总分布，必须用本工具。最终答案必须复述 Observation 的 cohort_patient_count 和逐日明细/图表 URL；禁止套用 1200、120 人或每天 1 人这类旧模板值。若问题明确包含糖尿病/高血压/高脂血症等疾病名并要求该疾病随访，才改用 chroniccare_open_sql_query。",
        ),
        input_schema=ToolSchema(
            properties={"question": {"type": "string", "description": "未来 N 天高风险随访问题。"}},
            required=["question"],
        ),
    ),
    ToolDefinition(
        name="chroniccare_cohort_disease_distribution",
        title="ChronicCare Cohort Disease Distribution",
        description=_desc(
            purpose="继承上一轮 cohort，上下文回答“他们/这些患者/该群体”的疾病类型或群体分布。",
            applicable="告诉我他们的疾病类型、这些患者主要有哪些慢病、该群体的风险等级分布如何。",
            forbidden="不要把‘他们’解释成全体患者；也不要用于无上下文的初始疾病分布问题。",
            input_format="传入 question。",
            output_format="返回 inherited_cohort、summary_text、table、chart_url、report_url。",
            examples="告诉我他们的疾病类型。",
            notes="若上一轮 cohort 是未来 30 天高风险随访患者，则必须沿用该 cohort。",
        ),
        input_schema=ToolSchema(
            properties={"question": {"type": "string", "description": "继承群体上下文的问题。"}},
            required=["question"],
        ),
    ),
    ToolDefinition(
        name="chroniccare_metric_query",
        title="ChronicCare Metric Query",
        description=_desc(
            purpose="历史兼容指标查询入口；当前内部会转发到 Open SQL Guard 后执行真实 SQL。",
            applicable="指标均值、异常率、人数、占比等问题；新问题优先直接使用 chroniccare_open_sql_query。",
            forbidden="不要用于纯疾病目录、风险总分布、图谱子图、开放能力说明。",
            input_format="传入 question。",
            output_format="返回 Open SQL 的 answer_markdown、table、sql、trace_id、chart_url/image_url。",
            examples="高脂血症患者的 LDL-C 异常比例是多少？",
            notes="为了避免旧模板误报，复杂指标应首选 chroniccare_open_sql_query；本工具仅作兼容别名。",
        ),
        input_schema=ToolSchema(
            properties={"question": {"type": "string", "description": "指标分析问题。"}},
            required=["question"],
        ),
    ),
    ToolDefinition(
        name="chroniccare_trend_query",
        title="ChronicCare Trend Query",
        description=_desc(
            purpose="历史兼容趋势入口；当前内部会转发到 Open SQL Guard 后执行真实 SQL。",
            applicable="最近 6 个月 HbA1c 异常人数趋势如何、最近 6 个月血压异常人数趋势如何、高血压患者最近半年的血压趋势如何。",
            forbidden="不要用于疾病目录、风险人数总分布、图谱子图或 DataMate 运行问题。",
            input_format="传入 question。",
            output_format="返回 Open SQL 的 answer_markdown、trend_rows、chart_url/image_url、sql、trace_id。",
            examples="最近 6 个月 HbA1c 异常人数趋势如何？",
            notes="新趋势问题优先直接使用 chroniccare_open_sql_query；不要输出旧模板常数。",
        ),
        input_schema=ToolSchema(
            properties={"question": {"type": "string", "description": "趋势分析问题。"}},
            required=["question"],
        ),
    ),
    ToolDefinition(
        name="chroniccare_graph_driven_analysis",
        title="ChronicCare Graph-driven Analysis",
        description=_desc(
            purpose="基于图谱先定位群体，再做综合分析并返回图表、报告、图谱入口。",
            applicable="基于图谱分析高血压合并糖尿病患者的 HbA1c、血压和 LDL-C 异常情况。",
            forbidden="不要替代单纯疾病分布、单个指标均值或单纯图谱子图生成。",
            input_format="传入 question。",
            output_format="返回 graph_url、chart_url、report_url、cohort_table_url、summary_text。",
            examples="请基于图谱驱动分析高血压合并糖尿病患者的 HbA1c、血压和 LDL-C 异常情况。",
            notes="只能复用工具真实返回的链接，不能自行扩写不存在的图表入口。",
        ),
        input_schema=ToolSchema(
            properties={"question": {"type": "string", "description": "图谱驱动分析问题。"}},
            required=["question"],
        ),
    ),
    ToolDefinition(
        name="chroniccare_open_analysis_query",
        title="ChronicCare Open Analysis Query",
        description=_desc(
            purpose="开放式分析总入口，用于同义改写、规则路由、安全兜底和兼容旧调用。",
            applicable="口语化自然语言分析、需要动态改写和安全 fallback 的问题。",
            forbidden="已有明确专用工具时不要优先选它，例如疾病分布、风险分布、未来 N 天高风险随访、受控指标查询、趋势查询。",
            input_format="传入 question。",
            output_format="返回 canonical_id、summary_text、table、charts、graph_url、chart_url、report_url、rule_pipeline。",
            examples="当前慢病类型分布如何？",
            notes="它是总入口，不应替代前端已明确定义的强路由专用工具。",
        ),
        input_schema=ToolSchema(
            properties={"question": {"type": "string", "description": "开放式自然语言分析问题。"}},
            required=["question"],
        ),
    ),
    ToolDefinition(
        name="chroniccare_open_sql_query",
        title="ChronicCare Open SQL Query",
        description=_desc(
            purpose="受控开放式 NL2SQL 查询入口，用于固定 A-G 工具未覆盖的慢病指标统计、均值、异常率、趋势和分布问题。",
            applicable="高血压和糖尿病都有的人糖化平均是多少、肥胖患者最近半年 BMI 趋势、高尿酸患者尿酸异常比例、用降压药患者血压控制情况。",
            forbidden="不要用它回答疾病分布、风险等级分布、未来 N 天高风险随访、图谱子图、DataMate pipeline、NPU benchmark、系统状态、图谱规模；这些应优先走专用工具。",
            input_format="传入 question，可选 prefer_llm、force_llm、allow_chart。",
            output_format="返回 stage、intent、sql_safe、sql、answer_markdown、table、chart_url、trace_id、llm_status、force_llm、safety_note。",
            examples="高血压和糖尿病都有的人，糖化平均是多少？",
            notes="回答必须基于 answer_markdown，不得自行改写数值；SQL Guard 不通过或 unsupported 时必须如实说明。",
        ),
        input_schema=ToolSchema(
            properties={
                "question": {"type": "string", "description": "开放式慢病统计问题。"},
                "prefer_llm": {"type": "boolean", "description": "模板未覆盖时是否允许 LLM SQL candidate。"},
                "force_llm": {"type": "boolean", "description": "是否优先尝试二阶段大模型生成 SQL；仍会经过 SQL Guard，不可执行危险 SQL。"},
                "allow_chart": {"type": "boolean", "description": "趋势/分布类是否生成图表。"},
            },
            required=["question"],
        ),
    ),
    ToolDefinition(
        name="chroniccare_open_sql_schema",
        title="ChronicCare Open SQL Schema",
        description=_desc(
            purpose="查看 Open SQL 允许访问的 SQLite schema catalog、字段白名单和 join 白名单。",
            applicable="当前开放式 SQL 能访问哪些表和字段、SQL Guard 白名单是什么。",
            forbidden="不要用它回答具体统计结果。",
            input_format="无参数。",
            output_format="返回 tables、fields、joins、safety_note。",
            examples="开放式 SQL 支持哪些表？",
            notes="只用于解释 Open SQL schema 和 SQL Guard 白名单，不返回业务统计数值。",
        ),
        input_schema=ToolSchema(properties={}),
    ),
    ToolDefinition(
        name="chroniccare_open_sql_eval",
        title="ChronicCare Open SQL Eval",
        description=_desc(
            purpose="运行或读取 Open SQL 阶段 1/阶段 2 评测报告。",
            applicable="开放式 SQL 准确率是多少、NL2SQL 是否达到 85%、Open SQL 评测结果。",
            forbidden="不要用它回答单个业务统计问题。",
            input_format="无参数。",
            output_format="返回 total_questions、sql_executable_rate、result_success_rate、template_stage_success_rate、llm_candidate_stage_success_rate。",
            examples="开放式 SQL 评测结果如何？",
            notes="用于比赛评测说明；不要把评测报告当作单个统计问题答案。",
        ),
        input_schema=ToolSchema(properties={}),
    ),
    ToolDefinition(
        name="chroniccare_open_sql_examples",
        title="ChronicCare Open SQL Examples",
        description=_desc(
            purpose="列出 Open SQL 支持的示例问题和能力范围。",
            applicable="开放式 SQL 可以问哪些问题、给我一些 NL2SQL 示例。",
            forbidden="不要用它回答具体统计结果。",
            input_format="无参数。",
            output_format="返回 examples、supported_intents、llm_status。",
            examples="Open SQL 有哪些示例问题？",
            notes="用于展示能力范围；用户问具体数值时应改用 chroniccare_open_sql_query。",
        ),
        input_schema=ToolSchema(properties={}),
    ),
    ToolDefinition(
        name="chroniccare_agent_run",
        title="ChronicCare Agent Run",
        description=_desc(
            purpose="执行多步综合总结型任务，输出完整 agent trace 和汇总结论。",
            applicable="请总结当前系统、完整端到端流程、综合分析说明。",
            forbidden="不要优先处理单个疾病分布、单个图谱子图、单个未来 N 天随访人数问题。",
            input_format="传入 user_goal。",
            output_format="返回 run_id、plan、tool_results、final_answer、trace_path。",
            examples="请总结当前慢病系统的整体能力。",
            notes="单问题场景应优先使用更专用的工具。",
        ),
        input_schema=ToolSchema(
            properties={"user_goal": {"type": "string", "description": "用户目标。"}},
            required=["user_goal"],
        ),
    ),
    ToolDefinition(
        name="chroniccare_report_summary",
        title="ChronicCare Report Summary",
        description=_desc(
            purpose="返回当前图表、报告、图谱 HTML 等可访问入口。",
            applicable="当前有哪些图表和报告入口、全局图谱概览在哪里看、报告入口是什么。",
            forbidden="不要用于疾病分布、指标均值、未来 N 天随访人数、高血压的知识图谱子图、糖尿病的知识图谱子图或特定图谱子图生成。",
            input_format="无需参数。",
            output_format="返回 report_url、chart_index_url、graph_url、charts、entry_guide。",
            examples="当前有哪些图表和报告入口？",
            notes="正文应优先展示 HTTP URL，不要只给内部路径；如果用户是在问某疾病/某群体知识图谱子图，不要用它，改用 chroniccare_kg_subgraph_render。",
        ),
        input_schema=ToolSchema(),
    ),
    ToolDefinition(
        name="chroniccare_trace_summary",
        title="ChronicCare Trace Summary",
        description=_desc(
            purpose="查询 MCP Adapter 最近工具调用审计记录。",
            applicable="最近工具调用情况、成功率、调用摘要。",
            forbidden="不要用它回答患者数量、疾病分布或图谱关系。",
            input_format="无需参数。",
            output_format="返回 total_calls、success_rate、tool_counts、recent_failures。",
            examples="最近工具调用情况如何？",
            notes="仅用于审计与追踪，不用于医疗结论。",
        ),
        input_schema=ToolSchema(),
    ),
]


def get_tool_map() -> Dict[str, ToolDefinition]:
    return {item.name: item for item in TOOL_DEFINITIONS}
