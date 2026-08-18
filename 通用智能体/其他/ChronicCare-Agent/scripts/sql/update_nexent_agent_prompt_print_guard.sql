UPDATE nexent.ag_tenant_agent_t
SET
    duty_prompt = CASE
        WHEN POSITION('每次工具调用后，必须立刻 `print(result)`。' IN COALESCE(duty_prompt, '')) > 0 THEN duty_prompt
        ELSE duty_prompt || E'\n\n【Nexent 工具调用硬约束】\n1. 每次工具调用后，必须立刻 `print(result)`。\n2. 禁止只写 `result = chroniccare_xxx(...)` 而不打印结果。\n3. 对知识图谱子图问题，推荐固定写法：\nresult = chroniccare_kg_subgraph_render(disease="冠心病风险")\nprint(result)'
    END,
    constraint_prompt = CASE
        WHEN POSITION('否则可能触发 Nexent 内部 warning 分支' IN COALESCE(constraint_prompt, '')) > 0 THEN constraint_prompt
        ELSE constraint_prompt || E'\n11. 在 Nexent 代码执行环境中，每次工具调用后必须 `print(result)`，否则可能触发 Nexent 内部 warning 分支并导致前端报 `Error in interaction: WARNING`。\n12. 对 `chroniccare_kg_subgraph_render`、`chroniccare_open_sql_query`、`chroniccare_report_summary`、`chroniccare_datamate_pipeline_run` 等工具都适用上一条，不允许省略 `print(result)`。'
    END,
    few_shots_prompt = CASE
        WHEN POSITION('用户：给我冠心病风险知识图谱子图' IN COALESCE(few_shots_prompt, '')) > 0 THEN few_shots_prompt
        ELSE few_shots_prompt || E'\n\n示例5：\n用户：给我冠心病风险知识图谱子图\n正确做法：\n- 先调用 chroniccare_kg_subgraph_render\n- 代码示例：`result = chroniccare_kg_subgraph_render(disease="冠心病风险")` 然后立刻 `print(result)`\n- 不要只写 `result = chroniccare_kg_subgraph_render(disease="冠心病风险")`\n- 必须根据 Observation 中真实返回的 graph_url / html_url 回答'
    END
WHERE agent_id = 1
  AND version_no = 1;
