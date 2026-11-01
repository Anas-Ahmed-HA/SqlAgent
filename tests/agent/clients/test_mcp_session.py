"""MCPSession over real streamable HTTP, against an in-process MCP server.

The server is the SDK's own FastMCP on a free local port, so these tests
cover the transport, the configured headers, structured errors, reconnecting
and the unreachable case without the C# SQL server.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time

import pytest
import uvicorn
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import EmbeddedResource, TextContent, TextResourceContents

from agent.clients.mcp_session import MCPSession
from agent.errors import MCPConnectionException


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="module")
def server_url():
    server = FastMCP("test", instructions="Call echo first.", stateless_http=True)

    @server.tool()
    def echo(text: str) -> str:
        """Echo the text."""
        return text

    @server.tool()
    def api_key(ctx: Context) -> str:
        """The X-Api-Key header the client sent."""
        return ctx.request_context.request.headers.get("x-api-key", "")

    @server.tool()
    def fails() -> str:
        """Always fails."""
        raise ValueError("bad query")

    @server.tool()
    def chart() -> list:
        """A summary, then the page: the shape of the charts server's results."""
        page = TextResourceContents(uri="plotly://charts/bar.html", mimeType="text/html", text="<html></html>")
        return [TextContent(type="text", text='{"chartType": "bar"}'), EmbeddedResource(type="resource", resource=page)]

    port = _free_port()
    uv = uvicorn.Server(uvicorn.Config(server.streamable_http_app(), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not uv.started:
        assert time.monotonic() < deadline, "the test MCP server did not start"
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/mcp"
    uv.should_exit = True
    thread.join(timeout=10)


def test_tools_instructions_and_calls_over_http(server_url) -> None:
    async def scenario():
        mcp = MCPSession(server_url, headers={"X-Api-Key": "k-123"})
        try:
            tools = await mcp.get_tools_for_openai()
            echoed = await mcp.call_tool_structured("echo", {"text": "hello"})
            key = await mcp.call_tool_structured("api_key", {})
            failed = await mcp.call_tool_structured("fails", {})
            await mcp.ping()
            return mcp.instructions, tools, echoed, key, failed
        finally:
            await mcp.close()

    instructions, tools, echoed, key, failed = asyncio.run(scenario())
    assert instructions == "Call echo first."
    assert [t["function"]["name"] for t in tools] == ["echo", "api_key", "fails", "chart"]
    assert tools[0]["function"]["parameters"]["properties"]["text"]["type"] == "string"
    assert echoed.success and echoed.text == "hello"
    # Configured headers reach the server on every request.
    assert key.text == "k-123"
    assert failed.is_error and "bad query" in failed.text


def test_an_embedded_page_comes_back_as_an_attachment_not_as_text(server_url) -> None:
    async def scenario():
        mcp = MCPSession(server_url)
        try:
            return await mcp.call_tool_structured("chart", {})
        finally:
            await mcp.close()

    result = asyncio.run(scenario())
    assert result.text == '{"chartType": "bar"}'
    assert result.attachments == [
        {"kind": "resource", "uri": "plotly://charts/bar.html", "mime_type": "text/html", "text": "<html></html>"}
    ]


def test_an_unreachable_server_is_a_connection_error_naming_the_url() -> None:
    url = f"http://127.0.0.1:{_free_port()}/mcp"

    async def scenario():
        mcp = MCPSession(url, connect_timeout_seconds=2)
        try:
            await mcp.get_tools_for_openai()
        finally:
            await mcp.close()

    with pytest.raises(MCPConnectionException) as raised:
        asyncio.run(scenario())
    assert url in str(raised.value)


def test_a_broken_session_is_reopened_before_the_next_call(server_url) -> None:
    async def scenario():
        mcp = MCPSession(server_url)
        try:
            await mcp.connect()
            mcp._mark_dead("simulated drop")
            assert not mcp.connected
            return await mcp.call_tool_structured("echo", {"text": "again"})
        finally:
            await mcp.close()

    assert asyncio.run(scenario()).text == "again"
