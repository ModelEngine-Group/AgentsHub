WITH tool_seed(name, description, inputs) AS (
    VALUES
    (
        'chroniccare_npu_readiness',
        '用途：检测当前慢病项目是否具备 NPU runtime、torch_npu/CANN/npu-smi 或本地 NPU 模型服务能力。适用问题：NPU 是否可用、当前有没有 NPU 环境、NPU runtime 检查。禁止用于疾病分布或图谱查询。',
        '{}'
    ),
    (
        'chroniccare_npu_supported_operators',
        '用途：列出当前已接入 NPU 增强分支的 DataMate 算子。当前 NPU 增强只支持 2 个算子：chronic_entity_extract_model_npu、chronic_relation_extract_model_npu。严禁把 DataMate 主线 11 个 CPU/通用算子说成 NPU 算子。',
        '{}'
    ),
    (
        'chroniccare_npu_operator_benchmark',
        '用途：默认读取最近一次 NPU benchmark 缓存报告；只有 force_run=true 时才重新运行。返回 CPU 与 NPU 对照指标。回答时每个算子使用纵向表格，列为“指标 / CPU / NPU / 对比说明”，包含处理量、CPU 规则耗时、BGE 抽样耗时、抽样吞吐量、平均单条延迟、抽样加速比、全量 BGE 耗时、资源/能耗采集状态。默认不要展示 sidecar 路径。适用问题：展示最近一次 NPU benchmark、NPU 加速效果、实体抽取关系抽取 BGE 耗时对比。',
        '{''use_npu'': {''type'': ''boolean'', ''description'': ''是否尝试 NPU 分支。''}, ''fallback'': {''type'': ''boolean'', ''description'': ''NPU 不可用时是否回退。''}, ''force_run'': {''type'': ''boolean'', ''description'': ''是否重新运行 benchmark；默认 false 只读取最近一次缓存报告。''}}'
    ),
    (
        'chroniccare_datamate_pipeline_run_npu',
        '用途：启用 NPU 重新运行 ChronicCare DataMate 全流程，覆盖实体候选 BGE 标准化、关系候选 BGE 重排/过滤。返回 npu_comparison_rows、CPU 与 NPU 对照指标和 report_path。回答时每个算子使用纵向表格，列为“指标 / CPU / NPU / 对比说明”，包含处理量、CPU 规则耗时、BGE 抽样耗时、抽样吞吐量、平均单条延迟、抽样加速比、全量 BGE 耗时、资源/能耗采集状态。默认不要展示 sidecar 路径。',
        '{''task_id'': {''type'': ''string'', ''description'': ''任务 id。''}, ''force'': {''type'': ''boolean'', ''description'': ''是否强制重跑；运行全流程场景默认 true。''}, ''safe_run'': {''type'': ''boolean'', ''description'': ''是否使用 safe-run。''}, ''npu_targets'': {''type'': ''array'', ''items'': {''type'': ''string''}, ''description'': ''NPU 增强算子列表。''}, ''fallback'': {''type'': ''boolean'', ''description'': ''NPU 不可用时是否回退。''}}'
    )
),
upserted AS (
    INSERT INTO nexent.ag_tool_info_t (
        name,
        origin_name,
        class_name,
        description,
        source,
        author,
        usage,
        params,
        inputs,
        output_type,
        category,
        is_available,
        created_by,
        updated_by,
        delete_flag
    )
    SELECT
        name,
        name,
        name,
        description,
        'mcp',
        'tenant_id',
        'ChronicCare MCP Adapter',
        '[]'::json,
        inputs,
        'string',
        '',
        TRUE,
        'codex',
        'codex',
        'N'
    FROM tool_seed seed
    WHERE NOT EXISTS (
        SELECT 1
        FROM nexent.ag_tool_info_t info
        WHERE info.name = seed.name
          AND info.delete_flag = 'N'
    )
    RETURNING tool_id
),
updated AS (
    UPDATE nexent.ag_tool_info_t info
    SET
        description = seed.description,
        source = 'mcp',
        author = 'tenant_id',
        usage = 'ChronicCare MCP Adapter',
        params = '[]'::json,
        inputs = seed.inputs,
        output_type = 'string',
        is_available = TRUE,
        updated_by = 'codex',
        delete_flag = 'N'
    FROM tool_seed seed
    WHERE info.name = seed.name
    RETURNING info.tool_id
),
targets AS (
    SELECT agent_id, version_no
    FROM nexent.ag_tenant_agent_t
    WHERE enabled = true
      AND (
        lower(coalesce(name, '')) like '%chroniccare%'
        OR lower(coalesce(display_name, '')) like '%chroniccare%'
        OR lower(coalesce(display_name, '')) like '%慢病%'
        OR lower(coalesce(name, '')) like 'cc_agent%'
      )
),
missing_instances AS (
    SELECT nextval('nexent.ag_tool_instance_t_tool_instance_id_seq') AS tool_instance_id,
           info.tool_id,
           targets.agent_id,
           targets.version_no
    FROM nexent.ag_tool_info_t info
    CROSS JOIN targets
    WHERE info.name IN (
        'chroniccare_npu_readiness',
        'chroniccare_npu_supported_operators',
        'chroniccare_npu_operator_benchmark',
        'chroniccare_datamate_pipeline_run_npu'
    )
      AND NOT EXISTS (
        SELECT 1
        FROM nexent.ag_tool_instance_t instance
        WHERE instance.tool_id = info.tool_id
          AND instance.agent_id = targets.agent_id
          AND instance.version_no = targets.version_no
          AND instance.delete_flag = 'N'
      )
)
INSERT INTO nexent.ag_tool_instance_t (
    tool_instance_id,
    tool_id,
    agent_id,
    params,
    user_id,
    tenant_id,
    enabled,
    version_no,
    created_by,
    updated_by,
    delete_flag
)
SELECT
    tool_instance_id,
    tool_id,
    agent_id,
    '{}'::json,
    'user_id',
    'tenant_id',
    TRUE,
    version_no,
    'codex',
    'codex',
    'N'
FROM missing_instances;

WITH targets AS (
    SELECT agent_id, version_no
    FROM nexent.ag_tenant_agent_t
    WHERE enabled = true
      AND (
        lower(coalesce(name, '')) like '%chroniccare%'
        OR lower(coalesce(display_name, '')) like '%chroniccare%'
        OR lower(coalesce(display_name, '')) like '%慢病%'
        OR lower(coalesce(name, '')) like 'cc_agent%'
      )
),
missing_instances AS (
    SELECT nextval('nexent.ag_tool_instance_t_tool_instance_id_seq') AS tool_instance_id,
           info.tool_id,
           targets.agent_id,
           targets.version_no
    FROM nexent.ag_tool_info_t info
    CROSS JOIN targets
    WHERE info.name IN (
        'chroniccare_npu_readiness',
        'chroniccare_npu_supported_operators',
        'chroniccare_npu_operator_benchmark',
        'chroniccare_datamate_pipeline_run_npu'
    )
      AND info.delete_flag = 'N'
      AND NOT EXISTS (
        SELECT 1
        FROM nexent.ag_tool_instance_t instance
        WHERE instance.tool_id = info.tool_id
          AND instance.agent_id = targets.agent_id
          AND instance.version_no = targets.version_no
          AND instance.delete_flag = 'N'
      )
)
INSERT INTO nexent.ag_tool_instance_t (
    tool_instance_id,
    tool_id,
    agent_id,
    params,
    user_id,
    tenant_id,
    enabled,
    version_no,
    created_by,
    updated_by,
    delete_flag
)
SELECT
    tool_instance_id,
    tool_id,
    agent_id,
    '{}'::json,
    'user_id',
    'tenant_id',
    TRUE,
    version_no,
    'codex',
    'codex',
    'N'
FROM missing_instances;

UPDATE nexent.ag_tool_instance_t instance
SET enabled = TRUE,
    delete_flag = 'N',
    updated_by = 'codex'
FROM nexent.ag_tool_info_t info
WHERE instance.tool_id = info.tool_id
  AND info.name IN (
      'chroniccare_npu_readiness',
      'chroniccare_npu_supported_operators',
      'chroniccare_npu_operator_benchmark',
      'chroniccare_datamate_pipeline_run_npu'
  );
