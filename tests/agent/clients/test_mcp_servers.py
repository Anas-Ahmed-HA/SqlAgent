"""MCPServers: the SQL server required, the charts server optional, one interface."""

from __future__ import annotations

import asyncio

import pytest

from agent.clients.mcp_servers import MCPServers
from agent.errors import MCPConnectionException
from tests.agent.fakes import FakeMCP, ok

CHART_TOOLS = ["create_bar_chart", "create_dashboard"]


def servers(*, charts_down: bool = False, chart_tools: list[str] | None = None) -> tuple[MCPServers, FakeMCP, FakeMCP]:
    sql = FakeMCP(instructions="Call list_databases first.")
    charts = FakeMCP(
        {"create_bar_chart": ok("{\"chartType\": \"bar\"}")},
        tools=chart_tools or CHART_TOOLS,
        instructions="Pass run_select's result as data.",
        unreachable=charts_down,
        name="charts",
    )
    return MCPServers(sql, [charts]), sql, charts


def names(tools: list[dict]) -> list[str]:
    return [t["function"]["name"] for t in tools]


def test_tools_are_listed_in_server_order_and_calls_go_where_they_were_listed():
    async def scenario():
        mcp, sql, charts = servers()
        await mcp.ping()
        tools = await mcp.get_tools_for_openai()
        assert names(tools) == [*sql.tool_list, *CHART_TOOLS]
        assert mcp.server_of("run_select") == "sql" and mcp.server_of("create_bar_chart") == "charts"
        await mcp.call_tool_structured("run_select", {"sql": "SELECT 1"})
        await mcp.call_tool_structured("create_bar_chart", {"x": "Dept"})
        assert sql.calls == [("run_select", {"sql": "SELECT 1"})]
        assert charts.calls == [("create_bar_chart", {"x": "Dept"})]

    asyncio.run(scenario())


def test_a_name_listed_twice_stays_with_the_sql_server():
    async def scenario():
        mcp, sql, charts = servers(chart_tools=["run_select", "create_bar_chart"])
        tools = await mcp.get_tools_for_openai()
        assert names(tools).count("run_select") == 1
        await mcp.call_tool_structured("run_select", {})
        assert sql.calls and not charts.calls

    asyncio.run(scenario())


def test_a_charts_server_that_is_down_leaves_only_the_sql_tools():
    async def scenario():
        mcp, sql, _ = servers(charts_down=True)
        await mcp.ping()  # does not raise
        assert names(await mcp.get_tools_for_openai()) == sql.tool_list
        assert mcp.instructions == "Call list_databases first."

    asyncio.run(scenario())


def test_an_unreachable_sql_server_still_fails_the_turn():
    async def scenario():
        mcp, sql, _ = servers()
        sql.unreachable = True
        with pytest.raises(MCPConnectionException):
            await mcp.ping()

    asyncio.run(scenario())


def test_a_charts_server_that_breaks_mid_call_answers_with_an_error():
    async def scenario():
        mcp, _, charts = servers()
        await mcp.get_tools_for_openai()
        charts.break_on_call = 1
        result = await mcp.call_tool_structured("create_bar_chart", {})
        assert result.is_error and "charts server is not reachable" in result.text

    asyncio.run(scenario())


def test_instructions_name_each_server_and_leave_out_the_hidden_ones():
    mcp, _, _ = servers()
    assert mcp.instructions == (
        "[sql server]\nCall list_databases first.\n\n[charts server]\nPass run_select's result as data."
    )
    assert mcp.instructions_of({"charts"}) == "Call list_databases first."
