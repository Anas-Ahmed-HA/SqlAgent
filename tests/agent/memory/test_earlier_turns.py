"""Earlier turns: their tool work is cut down once per turn, never the last N.

A long conversation is mostly the tool work of turns the user has moved on
from. `trim_earlier_turns` shortens it while keeping every tool call and
result, and it must give the same answer on every call of a turn (the
prompt prefix Azure caches) and when run again on its own output (a
persisted history).
"""

from __future__ import annotations

import asyncio
import json

from agent.config import AgentConfig
from agent.memory.context_budget import (
    SHORTENED_OUTPUT_CHARS,
    estimate_tokens,
    trim_earlier_turns,
)
from agent.memory.history import validate_message_sequence
from tests.agent.fakes import FakeAzure, FakeMCP, text_turn, tool_turn

COMPILE_ERROR = (
    "Code execution failed (compilation): (4,18): error CS1503: Argument 1: "
    "cannot convert from 'Autodesk.Revit.DB.UV' to 'Autodesk.Revit.DB.XYZ'"
)
from tests.agent.fakes import make_loop


def build(azure, mcp, **config):
    return make_loop(azure, mcp, **config)


def run(loop, message, history=None):
    return asyncio.run(loop.run(user_message=message, history=history))



def _call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _step(call_id: str, name: str, arguments: dict, result: str) -> list[dict]:
    return [
        {"role": "assistant", "content": None, "tool_calls": [_call(call_id, name, arguments)]},
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


def _success(payload_chars: int) -> str:
    return "Code execution completed successfully.\nResult:\n" + "x" * payload_chars


def turn(i: int) -> list[dict]:
    """A realistic turn: a failed snippet, two good ones, a big read, an answer."""
    p = f"t{i}"
    return [
        {"role": "user", "content": f"request {i}"},
        *_step(f"{p}_read", "ai_element_filter", {"category": "Walls"}, "e" * 18_400),
        *_step(f"{p}_c1", "send_code_to_revit", {"code": f"// {p} first try\nvar uv = new UV();"},
               COMPILE_ERROR),
        *_step(f"{p}_c2", "send_code_to_revit", {"code": f"// {p} second\nreturn 1;"},
               _success(50)),
        *_step(f"{p}_c3", "send_code_to_revit",
               {"code": f"// {p} final\nreturn walls;", "transactionMode": "auto"},
               _success(5_000)),
        {"role": "assistant", "content": f"answer {i}"},
    ]


def conversation(turns: int) -> list[dict]:
    return [m for i in range(turns) for m in turn(i)]


def _code(message: dict, index: int = 0) -> str:
    return json.loads(message["tool_calls"][index]["function"]["arguments"])["code"]


def _calls(messages: list[dict], call_id: str) -> dict:
    return next(
        c for m in messages for c in m.get("tool_calls") or () if c["id"] == call_id
    )


def _result(messages: list[dict], call_id: str) -> str:
    return next(m["content"] for m in messages if m.get("tool_call_id") == call_id)


# -- turn boundaries ---------------------------------------------------------


def test_the_last_three_turns_are_left_exactly_as_they_were():
    messages = conversation(5)
    trimmed = trim_earlier_turns(messages, 3)

    first_kept = next(
        i for i, m in enumerate(messages) if m.get("content") == "request 2"
    )
    assert trimmed[first_kept:] == messages[first_kept:]
    # Turns 0 and 1 are trimmed: their big outputs are no longer whole.
    for p in ("t0", "t1"):
        assert len(_result(trimmed, f"{p}_read")) < 18_400
    for p in ("t2", "t3", "t4"):
        assert _result(trimmed, f"{p}_read") == "e" * 18_400


def test_three_turns_or_fewer_are_not_touched():
    messages = conversation(3)
    assert trim_earlier_turns(messages, 3) == messages
    assert trim_earlier_turns([], 3) == []


def test_messages_before_the_first_user_message_belong_to_the_oldest_turn():
    stray = _step("s0", "get_current_view_info", {}, "v" * 9_000)
    messages = [*stray, *conversation(4)]
    trimmed = trim_earlier_turns(messages, 3)

    assert _result(trimmed, "s0").startswith("v" * SHORTENED_OUTPUT_CHARS)
    assert "[shortened: 9,000 chars from an earlier turn." in _result(trimmed, "s0")


def test_full_turns_zero_trims_every_earlier_turn():
    trimmed = trim_earlier_turns(conversation(2), 0)
    assert all(len(_result(trimmed, f"t{i}_read")) < 18_400 for i in range(2))


# -- what a trimmed turn keeps -----------------------------------------------


def test_old_tool_outputs_are_cut_with_a_note_giving_the_original_length():
    trimmed = trim_earlier_turns(conversation(4), 3)
    content = _result(trimmed, "t0_read")

    assert content.startswith("e" * SHORTENED_OUTPUT_CHARS)
    assert content[SHORTENED_OUTPUT_CHARS:] == (
        "\n[shortened: 18,400 chars from an earlier turn. "
        "Run the query again if you need this.]"
    )
    # Short outputs are left alone.
    assert _result(trimmed, "t0_c1") == COMPILE_ERROR


# -- validity and determinism ------------------------------------------------


def test_the_trimmed_sequence_is_still_valid_and_keeps_every_call():
    messages = conversation(6)
    trimmed = trim_earlier_turns(messages, 3)

    assert validate_message_sequence(messages) == []
    assert validate_message_sequence(trimmed) == []
    assert [m["role"] for m in trimmed] == [m["role"] for m in messages]
    assert [m.get("tool_call_id") for m in trimmed] == [m.get("tool_call_id") for m in messages]
    assert [
        [c["id"] for c in m.get("tool_calls") or ()] for m in trimmed
    ] == [[c["id"] for c in m.get("tool_calls") or ()] for m in messages]


def test_trimming_is_deterministic_and_idempotent():
    messages = conversation(6)
    once = trim_earlier_turns(messages, 3)

    assert json.dumps(trim_earlier_turns(conversation(6), 3)) == json.dumps(once)
    assert trim_earlier_turns(once, 3) == once, (
        "a persisted, already trimmed history must come back unchanged"
    )
    # The input is not modified in place.
    assert messages == conversation(6)


def test_a_turn_aging_out_later_does_not_change_turns_trimmed_before():
    """Turn 0 trimmed on the 5th turn and again on the 6th looks the same, so
    the cached prefix only grows."""
    four = trim_earlier_turns(conversation(4), 3)
    five = trim_earlier_turns([*four, *turn(4)], 3)
    assert five[: len(turn(0))] == four[: len(turn(0))]


def test_trimming_saves_most_of_a_long_conversation():
    messages = conversation(11)
    before = estimate_tokens(messages)
    after = estimate_tokens(trim_earlier_turns(messages, 3))
    assert after < before * 0.6


# -- inside the loop -------------------------------------------------------------


def test_every_model_call_of_a_turn_sends_the_same_trimmed_history():
    history = conversation(5)
    azure = FakeAzure(
        [
            tool_turn(("n1", "get_current_view_info", {})),
            tool_turn(("n2", "get_current_view_info", {})),
            text_turn("Done."),
        ]
    )
    mcp = FakeMCP()
    workflow = build(azure, mcp)
    asyncio.run(workflow.run("next request", history=history))

    planner_calls = [c for c in azure.calls if c.get("tools")]
    assert len(planner_calls) >= 2
    first = planner_calls[0]["messages"]
    sent_read = next(m for m in first if m.get("tool_call_id") == "t0_read")["content"]
    assert "[shortened: 18,400 chars from an earlier turn." in sent_read
    assert next(m for m in first if m.get("tool_call_id") == "t4_read")["content"] == "e" * 18_400
    assert validate_message_sequence(first) == []

    request = next(i for i, m in enumerate(first) if m.get("content") == "next request")
    for call in planner_calls[1:]:
        assert call["messages"][: request + 1] == first[: request + 1], (
            "the prompt prefix must not change between calls of a turn"
        )


def test_full_turns_is_configurable(monkeypatch):
    monkeypatch.setenv("AGENT_FULL_TURNS", "5")
    assert AgentConfig.from_env().full_turns == 5
    assert AgentConfig().full_turns == 3
