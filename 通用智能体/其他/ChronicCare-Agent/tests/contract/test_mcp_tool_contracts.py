from mcp_adapter.schemas import MCPRequest, MCPResponse, ToolCallRequest, ToolDefinition, ToolSchema
from mcp_adapter.tool_descriptions import TOOL_DEFINITIONS, get_tool_map


def test_tool_names_are_unique() -> None:
    names = [item.name for item in TOOL_DEFINITIONS]
    assert len(names) == len(set(names))


def test_tool_map_contains_every_definition() -> None:
    tool_map = get_tool_map()
    assert set(tool_map) == {item.name for item in TOOL_DEFINITIONS}


def test_tool_descriptions_contain_contract_sections() -> None:
    required_sections = (
        "用途：",
        "适用问题：",
        "禁止使用场景：",
        "输入格式：",
        "输出格式：",
        "示例问题：",
        "注意事项：",
    )
    for tool in TOOL_DEFINITIONS:
        assert all(section in tool.description for section in required_sections), tool.name


def test_required_schema_is_declared_for_dag_resume() -> None:
    schema = get_tool_map()["chroniccare_datamate_dag_resume"].input_schema
    assert schema.required == ["resume_run_id"]


def test_pydantic_contract_defaults() -> None:
    request = MCPRequest(method="tools/list")
    response = MCPResponse(id=request.id, result={"tools": []})
    call = ToolCallRequest(name="chroniccare_health_check")
    assert request.jsonrpc == "2.0"
    assert response.error is None
    assert call.arguments == {}


def test_tool_definition_round_trip() -> None:
    definition = ToolDefinition(
        name="demo",
        title="Demo",
        description="Demo contract",
        input_schema=ToolSchema(properties={"query": {"type": "string"}}, required=["query"]),
    )
    assert ToolDefinition.model_validate(definition.model_dump()) == definition
