"""Status adapters: liveness and tool inventory.

Thin HTTP adapters only. Turn coordination lives in ``agent.host.turns`` and
``agent.host.bootstrap``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from agent.errors import MCPConnectionException
from shared.api_contract import CAPABILITIES, Routes
from shared.api_schemas import HealthResponse, ToolsResponse

__all__ = ["router", "health", "tools"]

logger = logging.getLogger("sql_agent.api")

router = APIRouter()


@router.get(Routes.HEALTH, response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness, and whether the MCP server is connected.

    The server is pinged (and a stale session reopened), so the report says
    whether it can be reached now, not only whether it could at startup.
    """
    runtime = request.app.state.runtime
    mcp = runtime.mcp
    common = {
        "capabilities": sorted(CAPABILITIES),
        "creds_source": runtime.creds_source,
        "creds_error": runtime.creds_error,
    }
    if mcp is None:
        return HealthResponse(status="starting", mcp_session=False, **common)
    try:
        await mcp.ping()
    except MCPConnectionException as exc:
        return HealthResponse(
            status="degraded", mcp_session=False, mcp_url=mcp.url, detail=str(exc), **common
        )
    return HealthResponse(status="ok", mcp_session=True, mcp_url=mcp.url, **common)


@router.get(Routes.TOOLS, response_model=ToolsResponse)
async def tools(request: Request) -> ToolsResponse:
    """The MCP server's tools, as the model is offered them."""
    mcp = request.app.state.runtime.mcp
    try:
        tools_list = await mcp.get_tools_for_openai() if mcp is not None else None
    except MCPConnectionException as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if tools_list is None:
        raise HTTPException(status_code=503, detail="MCP session is not initialized")
    return ToolsResponse(count=len(tools_list), tools=tools_list)
