"""
agent.host.app

FastAPI service for the SQL agent: chat turns over HTTP, with live events
(thinking, tool status, questions, steers) on a per-turn WebSocket.

Run directly:
  uvicorn agent.host.app:app --host 127.0.0.1 --port 8321
(or `python main.py`, which delegates here).

The MCP server is reached by URL (`MCP_URL`), so nothing is spawned here.
Startup does not fail when it is unreachable: `/health` reports it, and each
chat turn tries to connect again.
"""

import logging  # noqa: E402
import os  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

from dotenv import load_dotenv  # noqa: E402
from fastapi import FastAPI  # noqa: E402

# shared.settings freezes endpoint constants at import time, so .env must be
# created before that import (see shared.env).
from shared.env import ensure_env_early  # noqa: E402

APP_DIR, ENV_PATH, ENV_STATUS = ensure_env_early()
ENV_LOADED = load_dotenv(ENV_PATH)

from agent.api.chat import router as chat_router  # noqa: E402
from agent.api.page import router as page_router  # noqa: E402
from agent.api.skills import router as skills_router  # noqa: E402
from agent.api.status import router as status_router  # noqa: E402
from agent.clients.mcp_session import MCPSession  # noqa: E402
from agent.config import MCPServerSettings  # noqa: E402
from agent.errors import MCPConnectionException  # noqa: E402
from agent.host.runner import run as run_server  # noqa: E402
from agent.host.runtime import Runtime  # noqa: E402
from agent.observability.logging_setup import configure_logging  # noqa: E402
from shared.settings import AIAPI_BASE_URL  # noqa: E402

configure_logging(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    json_output=os.environ.get("LOG_JSON", "").lower() in {"1", "true", "yes"},
)
logger = logging.getLogger("sql_agent.service")

logger.info(
    "env.loaded",
    extra={
        "app_dir": str(APP_DIR),
        "env_path": str(ENV_PATH),
        "loaded": ENV_LOADED,
        "env_status": ENV_STATUS,
        "aiapi_base_url": AIAPI_BASE_URL,
    },
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """The MCP connection is opened here; the Azure side is built on the first
    authenticated chat (token passthrough, agent/host/bootstrap.py), so missing
    credentials or an unreachable server never abort startup.

    Runtime state lives in exactly one place: `app.state.runtime`.
    """
    settings = MCPServerSettings.from_env()
    mcp = MCPSession(
        settings.url,
        headers=settings.headers,
        name="sql",
        connect_timeout_seconds=settings.connect_timeout_seconds,
        call_timeout_seconds=settings.call_timeout_seconds,
    )
    runtime = Runtime(mcp=mcp)
    app.state.runtime = runtime

    logger.info("startup.connecting", extra={"url": settings.url})
    try:
        await mcp.connect()
        await mcp.get_tools_for_openai()
    except MCPConnectionException:
        logger.error("startup.mcp_unavailable", exc_info=True)
    else:
        logger.info("startup.complete")

    try:
        yield
    finally:
        logger.info("shutdown.closing")
        await mcp.close()
        if runtime.azure is not None:
            await runtime.azure.aclose()


app = FastAPI(title="SQL Agent", lifespan=lifespan)
app.include_router(status_router)
app.include_router(skills_router)
app.include_router(chat_router)
app.include_router(page_router)


if __name__ == "__main__":
    run_server(app)
