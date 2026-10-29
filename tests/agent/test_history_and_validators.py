"""
Unit tests for the pieces that fail silently.

The truncation tests are the important ones: the original slice-based cap
produced message lists Azure rejects, and the failure surfaced one request
later, in a different session, as an opaque 400.
"""

from __future__ import annotations

import asyncio

import pytest

from agent.memory.history import (
    strip_for_persistence,
    truncate_messages,
    validate_message_sequence,
)
from agent.memory.store import MemoryStore


def tool_turn_messages(index: int) -> list[dict]:
    """One complete turn: user -> assistant(tool_calls) -> tool -> assistant."""
    return [
        {"role": "user", "content": f"request {index}"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{index}",
                    "type": "function",
                    "function": {"name": "send_code_to_revit", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": f"call_{index}", "content": "result"},
        {"role": "assistant", "content": f"answer {index}"},
    ]


# -- truncation -------------------------------------------------------------


@pytest.mark.parametrize("cap", list(range(1, 17)))
def test_truncation_never_produces_an_invalid_sequence(cap: int):
    """The property that matters: for *every* cap, the result is something
    Azure will accept. A naive tail slice fails this for most caps."""
    messages = [{"role": "system", "content": "rules"}]
    for i in range(4):
        messages += tool_turn_messages(i)

    result = truncate_messages(messages, cap)
    assert validate_message_sequence(result) == [], f"cap={cap} produced an invalid list"


def test_naive_slice_would_have_been_invalid():
    """Documents the bug this replaces, so a future 'simplification' back to
    a slice fails the suite instead of production."""
    messages = []
    for i in range(4):
        messages += tool_turn_messages(i)
    naive = messages[-2:]  # cuts between assistant(tool_calls) and its result
    assert validate_message_sequence(naive) != []


def test_truncation_pins_leading_system_messages():
    messages = [{"role": "system", "content": "rules"}] + tool_turn_messages(0) * 3
    result = truncate_messages(messages, 3)
    assert result[0] == {"role": "system", "content": "rules"}


def test_truncation_is_a_noop_when_under_the_cap():
    messages = tool_turn_messages(0)
    assert truncate_messages(messages, 100) == messages


def test_strip_for_persistence_drops_ephemeral_system_messages():
    messages = [
        {"role": "system", "content": "SEED"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "retry guidance: fix your snippet"},
        {"role": "assistant", "content": "done"},
    ]
    result = strip_for_persistence(messages)
    assert [m["role"] for m in result] == ["user", "assistant"]


def test_strip_for_persistence_keeps_messages_with_empty_content():
    """The seed is empty when Revit never connected: an assistant step with
    content "" (and its tool results) must still be stored."""
    call = {"id": "call_1", "type": "function", "function": {"name": "excel_read_table", "arguments": "{}"}}
    messages = [
        {"role": "user", "content": "sum the areas"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "table"},
        {"role": "assistant", "content": "42 m2"},
    ]
    assert strip_for_persistence(messages) == messages


def test_strip_for_persistence_drops_orphaned_tool_calls():
    """An aborted run leaves an assistant turn whose tool calls were never
    answered. Persisting it would break the next request."""
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "working on it",
            "tool_calls": [
                {"id": "call_x", "type": "function",
                 "function": {"name": "t", "arguments": "{}"}}
            ],
        },
    ]
    result = strip_for_persistence(messages)
    assert validate_message_sequence(result) == []
    assert result[-1] == {"role": "assistant", "content": "working on it"}


# -- outcome parsing --------------------------------------------------------
#
# The MCP server reads its diagnostic wrapper (tests/mcps/revit_mcp/test_csharp.py): a
# snippet that threw arrives as a failure, and errors the snippet caught
# itself arrive as `swallowed` beside a clean `result`. The agent only reads
# that structure; it never searches text.



def test_sessions_are_isolated():
    """The original kept `learned_lessons` on the agent singleton, so one
    engineer's compile error was replayed into everyone else's prompts."""
    store = MemoryStore()
    asyncio.run(store.save("a", [], ["lesson A"], []))
    asyncio.run(store.save("b", [], ["lesson B"], []))
    assert asyncio.run(store.load("a")).lessons == ["lesson A"]
    assert asyncio.run(store.load("b")).lessons == ["lesson B"]


def test_sessions_are_evicted_lru():
    store = MemoryStore(max_sessions=3)
    for i in range(5):
        asyncio.run(store.save(f"s{i}", [], [], []))
    assert len(store._sessions) == 3


def test_loaded_memory_is_a_copy():
    store = MemoryStore()
    asyncio.run(store.save("a", [{"role": "user", "content": "x"}], [], []))
    loaded = asyncio.run(store.load("a"))
    loaded.messages.append({"role": "user", "content": "mutation"})
    assert len(asyncio.run(store.load("a")).messages) == 1
