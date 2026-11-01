"""Charts in the loop: the charts server's pages go under the reply as `chart` blocks."""

from __future__ import annotations

import asyncio
import json

from agent.clients.mcp_servers import MCPServers
from agent.clients.mcp_session import MCPToolResult
from agent.memory.history import validate_message_sequence
from agent.prompts.templates import CHARTS_NOTE, NO_CHARTS_NOTE
from agent_cards import MAX_CHARTS_PER_TURN
from tests.agent.fakes import FakeAzure, FakeMCP, make_loop, text_turn, tool_turn

PAGE = "<!DOCTYPE html><html><body><div id='chart'></div></body></html>"
SUMMARY = {"chartType": "bar", "format": "html", "title": "Staff", "traces": 1, "points": 3, "truncated": False}
DATA = {"columns": ["Dept", "Staff"], "rows": [["IT", 12], ["HR", 7]]}


def chart(summary: dict | None = None, *attachments: dict) -> MCPToolResult:
    return MCPToolResult(
        text=json.dumps(summary or SUMMARY),
        attachments=list(attachments) or [{"kind": "resource", "uri": "plotly://charts/bar.html", "mime_type": "text/html", "text": PAGE}],
    )


def loop(azure: FakeAzure, result: MCPToolResult | None = None):
    charts = FakeMCP(
        {"create_bar_chart": result or chart(), "create_dashboard": result or chart()},
        tools=["create_bar_chart", "create_dashboard"],
        instructions="Pass run_select's result as data.",
        name="charts",
    )
    return make_loop(azure, MCPServers(FakeMCP(), [charts]))


def run(azure, result=None, **kwargs):
    return asyncio.run(loop(azure, result).run(user_message="Staff per department?", session_id="HR-1|s", **kwargs))


def offered(call) -> set[str]:
    return {t["function"]["name"] for t in call["tools"]}


def system_text(call) -> str:
    return "\n".join(m["content"] for m in call["messages"] if m["role"] == "system")


def draw(call_id: str = "c1", tool: str = "create_bar_chart"):
    return tool_turn((call_id, tool, {"data": DATA, "x": "Dept", "y": ["Staff"]}))


def test_a_client_that_draws_no_charts_is_offered_no_chart_tools():
    azure = FakeAzure([text_turn("IT has 12.")])
    run(azure, card_types=["stats"])
    assert not offered(azure.calls[0]) & {"create_bar_chart", "create_dashboard"}
    assert CHARTS_NOTE not in system_text(azure.calls[0])
    assert "Pass run_select's result" not in system_text(azure.calls[0])
    # The model is told why, so it does not answer as if charts did not exist.
    assert NO_CHARTS_NOTE in system_text(azure.calls[0])


def test_a_drawn_chart_goes_under_the_reply_and_its_page_never_reaches_the_model():
    azure = FakeAzure([draw(), text_turn("IT has the most staff.")])
    state = run(azure, card_types=["stats", "chart"])
    assert {"create_bar_chart", "create_dashboard"} <= offered(azure.calls[0])
    assert CHARTS_NOTE in system_text(azure.calls[0])
    assert NO_CHARTS_NOTE not in system_text(azure.calls[0])
    assert state["charts"] == [{"type": "chart", "title": "Staff", "html": PAGE}]
    tool_message = next(m for m in state["messages"] if m["role"] == "tool")
    assert PAGE not in tool_message["content"]
    assert "Shown to the user under your reply" in tool_message["content"]
    assert json.loads(tool_message["content"].splitlines()[0]) == SUMMARY
    assert validate_message_sequence(state["messages"]) == []


def test_a_dashboard_gets_a_taller_frame():
    dashboard = {"title": "Overview", "format": "html", "items": 3, "charts": 2, "points": 9, "truncated": False}
    state = run(FakeAzure([draw(tool="create_dashboard"), text_turn("Done.")]), chart(dashboard), card_types=["chart"])
    assert state["charts"][0]["height"] == 900


def test_an_image_is_not_shown_and_the_model_is_told_to_use_html():
    image = chart(None, {"kind": "image", "mime_type": "image/png", "data": "iVBORw0KGgo="})
    state = run(FakeAzure([draw(), text_turn("IT has 12.")]), image, card_types=["chart"])
    assert state["charts"] == []
    tool_message = next(m for m in state["messages"] if m["role"] == "tool")
    assert "format 'html'" in tool_message["content"] and "iVBORw0KGgo" not in tool_message["content"]


def test_a_reply_carries_a_limited_number_of_charts():
    calls = [draw(f"c{i}") for i in range(MAX_CHARTS_PER_TURN + 1)]
    state = run(FakeAzure([*calls, text_turn("Done.")]), card_types=["chart"])
    assert len(state["charts"]) == MAX_CHARTS_PER_TURN
    last = [m for m in state["messages"] if m["role"] == "tool"][-1]
    assert "create_dashboard" in last["content"]
