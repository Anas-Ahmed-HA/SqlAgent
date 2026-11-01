"""
Several MCP servers behind the one interface the loop uses.

The first server (the SQL server) is required: when it cannot be reached the
turn ends, as before. The others (the charts server) add tools when they are
up and are left out when they are not, so the agent still answers in text:

1. **Routing.** Each tool goes to the server that listed it. A name listed
   twice stays with the first server, so an extra server can never take over
   a SQL tool.
2. **Optional servers never block a turn.** They are pinged beside the
   required one, with their own short connect deadline, and after a failure
   they are not tried again for ``retry_after_seconds``.
3. **A broken optional server is a tool error, not the end of the turn:** the
   call is answered with what happened, and the model carries on without it.

Each server keeps its own :class:`MCPSession` rules (bounded connect,
reconnect but never replay).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from agent.clients.mcp_session import MCPSession, MCPToolResult
from agent.errors import MCPConnectionException

__all__ = ["MCPServers"]

logger = logging.getLogger("sql_agent.mcp")


class MCPServers:
    """The required server first, then the optional ones, as one session."""

    def __init__(
        self,
        required: MCPSession,
        optional: list[MCPSession] | None = None,
        *,
        retry_after_seconds: float = 60.0,
    ) -> None:
        self.required = required
        self.optional = list(optional or [])
        self.retry_after_seconds = retry_after_seconds
        #: When each optional server may be tried again after a failure.
        self._retry_at: dict[str, float] = {}
        self._routes: dict[str, MCPSession] = {}
        self._tools_cache: list[dict] = []

    # -- what the loop and the routes read --------------------------------

    @property
    def url(self) -> str:
        return self.required.url

    @property
    def name(self) -> str:
        return self.required.name

    @property
    def connected(self) -> bool:
        return self.required.connected

    @property
    def last_lifecycle_error(self) -> str:
        return self.required.last_lifecycle_error

    @property
    def instructions(self) -> str:
        return self.instructions_of()

    def instructions_of(self, exclude: frozenset[str] | set[str] = frozenset()) -> str:
        """Each connected server's instructions, but those named in ``exclude``
        (servers whose tools this turn does not offer). The required server's
        alone when no other is left, so a single-server prompt is unchanged."""
        extra = [s for s in self.optional if s.connected and s.instructions.strip() and s.name not in exclude]
        if not extra:
            return self.required.instructions
        sections = [self.required, *extra]
        return "\n\n".join(
            f"[{s.name} server]\n{s.instructions.strip()}" for s in sections if s.instructions.strip()
        )

    def server_of(self, tool: str) -> str | None:
        """The name of the server a tool goes to, or ``None`` when none listed it."""
        session = self._routes.get(tool)
        return session.name if session is not None else None

    def tool_names(self) -> list[str]:
        return [entry["function"]["name"] for entry in self._tools_cache]

    # -- lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        await asyncio.gather(*(self._try(s, s.connect) for s in self.optional))
        await self.required.connect()

    async def close(self) -> None:
        for session in [self.required, *self.optional]:
            await session.close()

    async def ensure_ready(self) -> None:
        await self.required.ensure_ready()

    async def ping(self) -> None:
        """The required server must answer; the others are checked beside it."""
        results = await asyncio.gather(
            self.required.ping(),
            *(self._try(s, s.ping) for s in self.optional),
            return_exceptions=True,
        )
        if isinstance(results[0], BaseException):
            raise results[0]

    async def _try(self, session: MCPSession, step) -> bool:
        """Run ``step`` for an optional server; ``False`` (logged) when it fails
        or is waiting out an earlier failure."""
        if time.monotonic() < self._retry_at.get(session.name, 0.0):
            return False
        try:
            await step()
        except MCPConnectionException as exc:
            self._retry_at[session.name] = time.monotonic() + self.retry_after_seconds
            logger.warning("mcp.optional_unavailable", extra={"server": session.name, "detail": str(exc)})
            return False
        self._retry_at.pop(session.name, None)
        return True

    # -- tool catalogue ----------------------------------------------------

    async def get_tools_for_openai(self, refresh: bool = False) -> list[dict]:
        """The required server's tools, then each reachable optional server's."""
        listed = [(self.required, await self.required.get_tools_for_openai(refresh=refresh))]
        for session in self.optional:
            if not session.connected:
                continue
            tools: list[dict] = []

            async def load(session: MCPSession = session) -> None:
                tools.extend(await session.get_tools_for_openai(refresh=refresh))

            if await self._try(session, load):
                listed.append((session, tools))

        routes: dict[str, MCPSession] = {}
        merged: list[dict] = []
        for session, tools in listed:
            for tool in tools:
                name = tool["function"]["name"]
                if name in routes:
                    logger.warning(
                        "mcp.tool_name_taken",
                        extra={"tool": name, "server": session.name, "kept": routes[name].name},
                    )
                    continue
                routes[name] = session
                merged.append(tool)
        self._routes = routes
        self._tools_cache = merged
        return merged

    # -- execution ---------------------------------------------------------

    async def call_tool_structured(self, name: str, arguments: dict[str, Any]) -> MCPToolResult:
        """Run the call on the server that listed the tool. A name nobody
        listed goes to the required server, which answers it with its own error."""
        session = self._routes.get(name, self.required)
        if session is self.required:
            return await session.call_tool_structured(name, arguments)
        try:
            return await session.call_tool_structured(name, arguments)
        except MCPConnectionException as exc:
            self._retry_at[session.name] = time.monotonic() + self.retry_after_seconds
            logger.warning("mcp.optional_call_failed", extra={"server": session.name, "tool": name})
            return MCPToolResult(
                text=(
                    f"NOT RUN -- the {session.name} server is not reachable ({exc}). "
                    "Answer without it."
                ),
                is_error=True,
                tool_name=name,
            )
