"""The local-tool interface: tools the model calls that the agent answers itself.

A local tool runs in this process instead of on the MCP server: ``ask_user``
waits for the user's answers, ``update_tasks`` changes the turn's task list.
Each one declares, in its own module:

* which call names it handles (:meth:`LocalTool.handles`),
* the definitions the loop may offer this step (:meth:`LocalTool.definitions`),
* how to run it (:meth:`LocalTool.run`).

A run returns a :class:`LocalResult`; it never raises for a bad call.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["LocalCall", "LocalResult", "LocalTool", "LocalToolRegistry"]

logger = logging.getLogger("sql_agent.tools")


@dataclass(frozen=True)
class LocalCall:
    """One call of a local tool, with the turn state it runs in."""

    name: str
    arguments: dict[str, Any]
    call_id: str
    state: Mapping[str, Any]


@dataclass
class LocalResult:
    """What a local tool answered.

    ``text`` is the tool message the model reads. ``update`` is merged into the
    turn state (for example the task list, or a later deadline); it must only
    carry keys the tool owns.
    """

    text: str
    is_error: bool = False
    #: ``validation`` for a call the model can fix by asking differently,
    #: ``runtime`` for everything else. Only read when ``is_error``.
    error_kind: str | None = None
    data: Any = None
    update: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def validation_error(cls, text: str) -> LocalResult:
        return cls(text=text, is_error=True, error_kind="validation")

    @classmethod
    def runtime_error(cls, text: str) -> LocalResult:
        return cls(text=text, is_error=True, error_kind="runtime")


class LocalTool:
    """Base class for a tool the agent answers itself."""

    def handles(self, name: str) -> bool:
        """Whether a call named ``name`` belongs to this tool."""
        raise NotImplementedError

    def definitions(self, state: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Definitions the loop offers this step; none by default."""
        return []

    async def run(self, call: LocalCall) -> LocalResult:
        raise NotImplementedError


class LocalToolRegistry:
    """The local tools, in the order they are offered.

    The order is the order of the definitions in every model request, so it
    must stay stable: the tool list is part of the cached prompt prefix.
    """

    def __init__(self, tools: list[LocalTool]) -> None:
        self._tools = list(tools)

    def find(self, name: str) -> LocalTool | None:
        """The tool handling ``name``, or None for a call that goes to the MCP server."""
        for tool in self._tools:
            if tool.handles(name):
                return tool
        return None

    def definitions(self, state: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Every definition the local tools offer this step, in registry order."""
        offered: list[dict[str, Any]] = []
        for tool in self._tools:
            offered.extend(tool.definitions(state) or [])
        return offered
