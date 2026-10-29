"""Shared FastAPI runtime state.

The lifespan in ``agent.host.app`` builds exactly one of these and stores it as
``app.state.runtime``. Routes in ``agent.api`` and the lazy builder in
``agent.host.bootstrap`` read and mutate that instance directly -- there is
no other ``app.state`` attribute and no ``getattr`` fallback.

Fields: the MCP session, the lazily built Azure/agent stack, credential
status flags, the last gateway failure (message only, with its timestamp, so a
recent failure can short-circuit queued requests during the cooldown), and
the lock serialising stack rebuilds/resets.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from agent.clients.azure import AzureChatService
    from agent.clients.mcp_session import MCPSession
    from agent.loop import AgentLoop
    from agent.memory.store import MemoryStore

__all__ = ["Runtime"]


@dataclass
class Runtime:
    """Mutable holder for the single live agent stack."""

    mcp: MCPSession | Any | None = None
    """MCP session to the server at `MCP_URL`, opened during startup."""

    azure: AzureChatService | Any | None = None
    """Azure chat client, built lazily on the first authenticated chat."""

    memory: MemoryStore | Any | None = None
    """In-process session history, preserved across credential resets."""

    agent: AgentLoop | Any | None = None
    """The agent loop; ``None`` until the first successful lazy build."""

    creds_source: str = "lazy"
    """``"lazy"`` before the first fetch, ``"aiapi-gateway"`` once built."""

    creds_error: str = ""
    """Human-readable reason for the last credential failure, if any."""

    creds_stale: bool = False
    """Set by the Azure auth-failure hook; next request rebuilds the stack."""

    creds_token_fingerprint: str | None = None
    """Fingerprint of the login token used to build the cached stack."""

    gateway_failure: str | None = None
    """Last gateway failure message (message only, never the exception)."""

    gateway_failure_at: float | None = None
    """``time.monotonic()`` timestamp of the last gateway failure."""

    gateway_failure_token_fingerprint: str | None = None
    """Fingerprint of the token that produced ``gateway_failure``."""

    build_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    """Serialises lazy builds (``ensure_ready``) against resets."""

    active_turns: set[asyncio.Task] = field(default_factory=set)
    """Chat turns currently using the cached agent stack."""

    turn_condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    """Wakes lifecycle operations when active chat turns finish."""

    async def begin_turn(self) -> None:
        """Register the current request before it can use the agent stack."""
        task = asyncio.current_task()
        if task is None:
            return
        async with self.turn_condition:
            self.active_turns.add(task)

    async def end_turn(self) -> None:
        """Release the current request's lease on the cached stack."""
        task = asyncio.current_task()
        async with self.turn_condition:
            if task is not None:
                self.active_turns.discard(task)
            self.turn_condition.notify_all()

    async def wait_for_turns_idle(self) -> None:
        """Wait until no other request is using the cached agent stack."""
        current = asyncio.current_task()
        async with self.turn_condition:
            while any(task is not current for task in self.active_turns):
                await self.turn_condition.wait()

    def current_turn_is_active(self) -> bool:
        """Return whether the current task owns a stack lease."""
        task = asyncio.current_task()
        return task is not None and task in self.active_turns
