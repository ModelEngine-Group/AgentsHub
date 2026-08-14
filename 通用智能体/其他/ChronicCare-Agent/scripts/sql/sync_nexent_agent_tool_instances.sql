WITH missing_tools AS (
    SELECT ti.tool_id
    FROM nexent.ag_tool_info_t ti
    WHERE ti.name LIKE 'chroniccare_%'
      AND NOT EXISTS (
          SELECT 1
          FROM nexent.ag_tool_instance_t ati
          WHERE ati.agent_id = 1
            AND ati.version_no = 1
            AND ati.tool_id = ti.tool_id
            AND ati.delete_flag = 'N'
      )
),
allocated_ids AS (
    SELECT nextval('nexent.ag_tool_instance_t_tool_instance_id_seq') AS tool_instance_id, tool_id
    FROM missing_tools
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
    allocated_ids.tool_instance_id,
    allocated_ids.tool_id,
    1,
    '{}'::json,
    'user_id',
    'tenant_id',
    TRUE,
    versions.version_no,
    'codex',
    'codex',
    'N'
FROM allocated_ids
CROSS JOIN (VALUES (0), (1)) AS versions(version_no);
