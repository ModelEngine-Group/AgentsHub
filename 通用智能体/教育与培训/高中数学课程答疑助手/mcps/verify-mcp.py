import asyncio
import json

from fastmcp import Client


async def main():
    async with Client("http://nexent-math-course:8765/mcp") as client:
        names = {tool.name for tool in await client.list_tools()}
        assert names == {"quadratic_check", "fraction_check"}, names
        quadratic_result = await client.call_tool("quadratic_check", {"a": 1, "b": -5, "c": 6})
        fraction_result = await client.call_tool("fraction_check", {
            "numerator_a": 3, "denominator_a": 5, "numerator_b": 2,
            "denominator_b": 4, "operation": "multiply"
        })
        quadratic = quadratic_result.data or json.loads(quadratic_result.content[0].text)
        fraction = fraction_result.data or json.loads(fraction_result.content[0].text)
        assert quadratic["roots"] == ["2", "3"] and quadratic["vertex_y"] == "-1/4", quadratic
        assert fraction["exact"] == "3/10", fraction
        print({"quadratic": quadratic, "fraction": fraction})


asyncio.run(main())
