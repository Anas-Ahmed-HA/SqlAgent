"""
Test doubles.

Every external call goes through `AzureChatService` and `MCPSession`, so the
loop can be driven end to end with no Azure account and no MCP server.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agent.clients.mcp_session import MCPToolResult
from agent.config import AgentConfig
from agent.errors import MCPConnectionException
from agent.loop import AgentLoop
from agent.memory.store import MemoryStore

# -- Azure ------------------------------------------------------------------


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction
    type: str = "function"


@dataclass
class FakeMessage:
    content: str | None = None
    tool_calls: list[FakeToolCall] | None = None


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeCompletion:
    choices: list[FakeChoice]
    usage: Any = None


def text_turn(content: str) -> FakeCompletion:
    return FakeCompletion([FakeChoice(FakeMessage(content=content))])


def tool_turn(*calls: tuple[str, str, dict], content: str | None = None) -> FakeCompletion:
    """`tool_turn(("call_1", "run_select", {"database": "X", "sql": "SELECT 1"}))`"""
    return FakeCompletion(
        [
            FakeChoice(
                FakeMessage(
                    content=content,
                    tool_calls=[
                        FakeToolCall(
                            id=call_id,
                            function=FakeFunction(name=name, arguments=json.dumps(args)),
                        )
                        for call_id, name, args in calls
                    ],
                )
            )
        ]
    )


class FakeAzure:
    """Replays a scripted sequence of completions.

    A script entry may also be a callable taking the call's kwargs, for a test
    that acts while the model "thinks" (sending a steer, say).
    """

    def __init__(
        self,
        turns: list[FakeCompletion | Callable[[dict], FakeCompletion]] | None = None,
        skill_result: str = "{\"name\": \"Test skill\", \"description\": \"A test skill.\", \"instructions\": [\"Follow the test rule.\"]}",
    ) -> None:
        self.turns = list(turns or [])
        self.skill_result = skill_result
        self.calls: list[dict] = []
        self.text_calls: list[tuple[str, str]] = []

    async def complete(self, **kwargs: Any) -> FakeCompletion:
        self.calls.append(kwargs)
        if not self.turns:
            return text_turn("(fake ran out of scripted turns)")
        turn = self.turns.pop(0)
        return turn(kwargs) if callable(turn) else turn

    async def complete_text(self, system_prompt: str, user_content: str) -> str:
        self.text_calls.append((system_prompt, user_content))
        return self.skill_result

    async def aclose(self) -> None:
        return None


# -- MCP ----------------------------------------------------------------------

SQL_TOOLS = ["list_databases", "get_business_rules", "list_objects", "describe_object", "run_select"]


def ok(text: str) -> MCPToolResult:
    return MCPToolResult(text=text)


def tool_error(code: str, message: str, fix: str = "Fix it.") -> MCPToolResult:
    """The SQL MCP server's error shape: JSON with code, message, fix, retryable."""
    payload = {"code": code, "message": message, "fix": fix, "retryable": False}
    return MCPToolResult(text=json.dumps(payload), is_error=True)


class FakeMCP:
    """An MCP session whose tools answer from a script.

    ``results`` maps a tool name to a list of results (used in order; the
    last one repeats) or to a callable taking the arguments. ``unreachable``
    makes every call fail as if the server were down.
    """

    def __init__(
        self,
        results: dict[str, Any] | None = None,
        *,
        tools: list[str] | None = None,
        instructions: str = "Call list_databases first.",
        unreachable: bool = False,
    ) -> None:
        self.url = "http://mcp.test/mcp"
        self.instructions = instructions
        self.results = dict(results or {})
        self.tool_list = list(tools or SQL_TOOLS)
        self.unreachable = unreachable
        self.calls: list[tuple[str, dict]] = []
        #: Raise a connection failure on this call number (1-based).
        self.break_on_call: int | None = None

    @property
    def connected(self) -> bool:
        return not self.unreachable

    async def ensure_ready(self) -> None:
        if self.unreachable:
            raise MCPConnectionException(f"Could not connect to the MCP server at {self.url}: refused")

    async def ping(self) -> None:
        await self.ensure_ready()

    async def get_tools_for_openai(self, refresh: bool = False) -> list[dict]:
        await self.ensure_ready()
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"The {name} tool.",
                    "parameters": {"type": "object", "properties": {"database": {"type": "string"}}},
                },
            }
            for name in self.tool_list
        ]

    async def call_tool_structured(self, name: str, arguments: dict) -> MCPToolResult:
        await self.ensure_ready()
        self.calls.append((name, arguments))
        if self.break_on_call is not None and len(self.calls) == self.break_on_call:
            self.unreachable = True
            raise MCPConnectionException(f"The connection to the MCP server broke during `{name}`.")
        script = self.results.get(name)
        if callable(script):
            return script(arguments)
        if isinstance(script, list) and script:
            return script.pop(0) if len(script) > 1 else script[0]
        if isinstance(script, MCPToolResult):
            return script
        return ok(f"{name} ok")


def make_loop(azure: FakeAzure, mcp: FakeMCP | None = None, **config: Any) -> AgentLoop:
    return AgentLoop(
        mcp=mcp or FakeMCP(),
        azure=azure,
        memory=MemoryStore(),
        config=AgentConfig(**config),
    )
