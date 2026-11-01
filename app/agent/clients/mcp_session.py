"""
The client side of one MCP server reached by URL (streamable HTTP).

Owns the HTTP connection and the MCP session over it:

1. **Bounded connect.** Opening the connection and MCP initialize each have a
   deadline; an unreachable server fails fast instead of hanging the request.
2. **Reconnect, never replay.** A failed call marks the session dead and
   clears the caches. A fresh session is opened before a *later* call; a call
   in flight when the connection broke is never repeated.
3. **Structured results.** MCP ``isError`` is kept beside the text, and the
   result's ``structuredContent`` comes back as data, so callers judge success
   without parsing text. Embedded pages and images (a chart tool's HTML) come
   back as ``attachments``, never in the text: they are for the user, and
   would only fill the model's context.
4. **The catalogue.** Tools and the server's ``instructions``, read once per
   session.

``headers`` go on every HTTP request (an API key, for example). They come
from configuration, never from the model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from agent.errors import MCPConnectionException, ToolTimeoutException

__all__ = ["MCPSession", "MCPToolResult"]

logger = logging.getLogger("sql_agent.mcp")


@dataclass
class MCPToolResult:
    """Structured outcome of one MCP tool call.

    ``text`` renders into the OpenAI ``tool`` message; ``is_error`` (from MCP)
    is the success signal.
    """

    text: str
    is_error: bool = False
    tool_name: str = ""
    elapsed_ms: int = 0
    #: The result's ``structuredContent``, when the server sent one.
    structured: dict[str, Any] | None = None
    #: Embedded resources and images, in order: ``{"kind": "resource", "uri",
    #: "mime_type", "text"}`` or ``{"kind": "image", "mime_type", "data"}``.
    attachments: list[dict[str, Any]] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.is_error


class MCPSession:
    """Owns one connection to an MCP server at ``url``."""

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        name: str = "mcp",
        connect_timeout_seconds: float = 30.0,
        call_timeout_seconds: float = 120.0,
    ) -> None:
        self.url = url
        self.headers = dict(headers or {})
        self.name = name
        self.connect_timeout_seconds = connect_timeout_seconds
        self.call_timeout_seconds = call_timeout_seconds

        self.session: ClientSession | None = None
        #: The server's ``instructions`` from initialization: how to use its
        #: tools. Empty until connected, or when the server sends none.
        self.instructions: str = ""
        self._lifecycle_lock = asyncio.Lock()
        self._dead = False
        self._last_lifecycle_error = ""
        self._tools_cache: list[dict] = []

        # The SDK's transport owns an anyio task group. Enter and exit it in
        # one dedicated task; request tasks only signal it to stop.
        self._session_task: asyncio.Task | None = None
        self._session_stop: asyncio.Event | None = None

    # -- lifecycle ---------------------------------------------------------

    async def _close_stack(self) -> None:
        task = self._session_task
        stop = self._session_stop
        if task is None:
            self.session = None
            return
        if stop is not None:
            stop.set()
        try:
            await task
        except BaseException:  # noqa: BLE001 - shutdown must detach cleanly
            logger.debug("mcp.close_stack_error", exc_info=True)
        finally:
            if self._session_task is task:
                self._session_task = None
                self._session_stop = None
            self.session = None

    async def _run_session(self, ready: asyncio.Future[None]) -> None:
        """Own the HTTP client, transport and session from entry to teardown."""
        stack = AsyncExitStack()
        stop = self._session_stop
        assert stop is not None
        try:
            client = await stack.enter_async_context(
                httpx.AsyncClient(
                    headers=self.headers,
                    timeout=httpx.Timeout(self.connect_timeout_seconds, read=self.call_timeout_seconds),
                    follow_redirects=True,
                )
            )
            async with asyncio.timeout(self.connect_timeout_seconds):
                read_stream, write_stream, _ = await stack.enter_async_context(
                    streamable_http_client(self.url, http_client=client)
                )
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            async with asyncio.timeout(self.connect_timeout_seconds):
                initialized = await session.initialize()
            self.instructions = getattr(initialized, "instructions", None) or ""
            self.session = session
            if not ready.done():
                ready.set_result(None)
            await stop.wait()
        except BaseException as exc:  # noqa: BLE001 - reported to connect()
            if not ready.done():
                ready.set_exception(exc)
        finally:
            self.session = None
            try:
                await stack.aclose()
            except BaseException:  # noqa: BLE001 - shutdown must detach cleanly
                logger.debug("mcp.close_stack_error", exc_info=True)

    def _mark_dead(self, reason: str) -> None:
        self._dead = True
        self._last_lifecycle_error = reason
        self.session = None
        self._tools_cache = []

    async def connect(self) -> None:
        async with self._lifecycle_lock:
            # Callers queued behind a successful connect must not tear down
            # its live session.
            if self.connected:
                return
            await self._close_stack()
            self._tools_cache = []
            self._dead = False
            ready = asyncio.get_running_loop().create_future()
            self._session_stop = asyncio.Event()
            self._session_task = asyncio.create_task(
                self._run_session(ready), name=f"{self.name}-mcp-session"
            )
            try:
                await ready
            except BaseException as exc:  # noqa: BLE001 - re-raised as a typed error
                await self._close_stack()
                task = asyncio.current_task()
                if task is not None and task.cancelling():
                    raise  # this turn was stopped; not a connection failure
                # The SDK logs a refused connection and leaves initialize
                # waiting, so an unreachable server shows up as the deadline.
                silent = isinstance(exc, (asyncio.TimeoutError, TimeoutError, asyncio.CancelledError))
                reason = (
                    f"no answer within {self.connect_timeout_seconds:.0f}s (is the server running?)"
                    if silent else _describe(exc)
                )
                self._mark_dead(reason)
                raise MCPConnectionException(
                    f"Could not connect to the MCP server at {self.url}: {reason}"
                ) from exc
            logger.info("mcp.connected", extra={"server": self.name, "url": self.url})

    async def close(self) -> None:
        await self._close_stack()
        self._tools_cache = []
        self._dead = False
        logger.info("mcp.closed", extra={"server": self.name})

    @property
    def connected(self) -> bool:
        return self.session is not None and not self._dead

    @property
    def last_lifecycle_error(self) -> str:
        return self._last_lifecycle_error

    def _require_session(self) -> ClientSession:
        if self.session is None or self._dead:
            raise MCPConnectionException(
                f"The MCP server at {self.url} is not connected."
                + (f" Last error: {self._last_lifecycle_error}." if self._last_lifecycle_error else "")
            )
        return self.session

    async def ensure_ready(self) -> None:
        """Open a fresh session if the previous one broke. Used before a new
        call, never to replay an in-flight one."""
        if not self.connected:
            await self.connect()

    async def ping(self) -> None:
        """Check that the server answers now, reopening the session once if not.

        Over HTTP a session only finds out the server went away on its next
        request, so this runs before each turn: a server that restarted
        between turns gets a fresh session instead of a failed first call.
        """
        await self.ensure_ready()
        try:
            await asyncio.wait_for(self._require_session().send_ping(), timeout=self.connect_timeout_seconds)
            return
        except Exception as exc:  # noqa: BLE001 - stale session or server gone
            self._mark_dead(f"ping failed: {_describe(exc)}")
            await self._close_stack()
        await self.connect()

    # -- tool catalogue ----------------------------------------------------

    async def get_tools_for_openai(self, refresh: bool = False) -> list[dict]:
        """Tool definitions in Azure function-calling shape, in the server's order (cached)."""
        if self._tools_cache and not refresh:
            return self._tools_cache
        await self.ensure_ready()
        try:
            listed = await self._require_session().list_tools()
        except MCPConnectionException:
            raise
        except Exception as exc:  # noqa: BLE001 - broken connection
            self._mark_dead(f"list_tools failed: {exc}")
            await self._close_stack()
            raise MCPConnectionException(f"The MCP server is unreachable (list_tools): {exc}") from exc
        self._tools_cache = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                },
            }
            for tool in listed.tools
        ]
        logger.info("mcp.tools.loaded", extra={"server": self.name, "count": len(self._tools_cache)})
        return self._tools_cache

    def tool_names(self) -> list[str]:
        """The listed tools' names, in the server's order."""
        return [entry["function"]["name"] for entry in self._tools_cache]

    def server_of(self, tool: str) -> str | None:
        """This server's name when it listed ``tool``."""
        return self.name if tool in self.tool_names() else None

    # -- execution ---------------------------------------------------------

    async def call_tool_structured(self, name: str, arguments: dict[str, Any]) -> MCPToolResult:
        """Execute one tool call, preserving MCP ``isError``.

        Never replays after a disconnect: a failure mid-call marks the session
        dead and raises; the next call reconnects first.
        """
        await self.ensure_ready()
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self._require_session().call_tool(name, arguments),
                timeout=self.call_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            self._mark_dead(f"call_tool {name} timed out")
            await self._close_stack()
            raise ToolTimeoutException(
                f"`{name}` did not return within {self.call_timeout_seconds:.0f}s."
            ) from exc
        except MCPConnectionException:
            raise
        except Exception as exc:  # noqa: BLE001 - broken connection
            self._mark_dead(f"call_tool {name} failed: {exc}")
            await self._close_stack()
            raise MCPConnectionException(
                f"The connection to the MCP server broke during `{name}`: {exc}. It was not replayed."
            ) from exc
        finally:
            logger.info(
                "mcp.tool.call",
                extra={
                    "server": self.name,
                    "tool": name,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000),
                },
            )
        return self._structured(name, result, elapsed_ms=round((time.perf_counter() - started) * 1000))

    @staticmethod
    def _structured(name: str, result: Any, *, elapsed_ms: int) -> MCPToolResult:
        parts: list[str] = []
        attachments: list[dict[str, Any]] = []
        for block in result.content:
            kind = getattr(block, "type", None)
            resource = getattr(block, "resource", None)
            if kind == "text":
                parts.append(block.text)
            elif kind == "resource" and isinstance(getattr(resource, "text", None), str):
                attachments.append({
                    "kind": "resource",
                    "uri": str(resource.uri),
                    "mime_type": resource.mimeType or "",
                    "text": resource.text,
                })
            elif kind == "image":
                attachments.append({"kind": "image", "mime_type": block.mimeType or "", "data": block.data})
            else:
                parts.append(json.dumps(block.model_dump(), default=str))
        structured = getattr(result, "structuredContent", None)
        return MCPToolResult(
            text="\n".join(parts) if parts else "(no content returned)",
            is_error=bool(getattr(result, "isError", False)),
            tool_name=name,
            elapsed_ms=elapsed_ms,
            structured=structured if isinstance(structured, dict) else None,
            attachments=attachments,
        )


def _describe(exc: BaseException) -> str:
    """The innermost useful message: the transport wraps failures in task groups."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return str(exc) or type(exc).__name__
