"""A steer is part of the turn it steers, not the start of a new turn.

Every place that finds where a turn starts by its user message skips steers:
the context budget, the trimming of earlier turns, the turn trace and the
The steer mark never reaches the model API.
"""

from __future__ import annotations

from agent.memory.context_budget import fit_context, trim_earlier_turns
from agent.memory.history import (
    STEER_KEY,
    is_steer,
    provider_messages,
    sanitize_client_history,
    validate_message_sequence,
)
from agent.observability.turn_trace import build_turn_trace, turn_step_count

SYSTEM = {"role": "system", "content": "rules"}
STEER = {"role": "user", "content": "STEER: make them 3m", STEER_KEY: True}


def step(n: int, size: int = 10) -> list[dict]:
    return [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{n}", "type": "function", "function": {"name": "get_x", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": f"c{n}", "content": "x" * size},
    ]


def turn_with_steer() -> list[dict]:
    return [SYSTEM, {"role": "user", "content": "create walls"}, *step(1), STEER, *step(2)]


def test_the_steer_mark_is_removed_before_the_model_api_sees_it():
    messages = turn_with_steer()
    sent = provider_messages(messages)
    assert all(STEER_KEY not in m for m in sent)
    assert is_steer(messages[4]), "the turn's own messages keep the mark"
    plain = [SYSTEM]
    assert provider_messages(plain) is plain


def test_the_turn_step_count_and_trace_include_the_steps_before_a_steer():
    messages = turn_with_steer()
    assert turn_step_count(messages) == 2
    trace = build_turn_trace(messages, {}, "t1")
    assert [span["id"] for span in trace["spans"]] == ["c1", "c2"]


def test_the_context_budget_keeps_the_request_as_anchor_and_keeps_steers():
    messages = [
        SYSTEM, {"role": "user", "content": "create walls"},
        *step(1, 4000), STEER, *step(2, 4000), *step(3, 400),
    ]
    fitted, plan = fit_context(messages, token_budget=600, max_messages=400)
    assert plan["turn_drop"] > 0, "old steps of this turn were dropped"
    assert fitted[1]["content"] == "create walls", "the request is still the anchor"
    assert STEER in fitted, "the steer survives the steps around it being dropped"
    assert validate_message_sequence(fitted) == []
    assert fitted[-2]["tool_calls"][0]["id"] == "c3"


def test_trimming_earlier_turns_does_not_split_a_turn_at_its_steer():
    earlier = [{"role": "user", "content": "create walls"}, *step(1, 5000), STEER, *step(2, 5000),
               {"role": "assistant", "content": "done"}]
    latest = [{"role": "user", "content": "now doors"}, {"role": "assistant", "content": "ok"}]
    trimmed = trim_earlier_turns([*earlier, *latest], full_turns=1)
    # The whole earlier turn, both sides of the steer, is trimmed as one.
    outputs = [m["content"] for m in trimmed if m.get("role") == "tool"]
    assert all(len(text) < 2500 for text in outputs)
    kept = trim_earlier_turns([*earlier, *latest], full_turns=2)
    assert [m["content"] for m in kept if m.get("role") == "tool"] == ["x" * 5000] * 2


def test_a_replayed_steer_keeps_its_mark_through_client_history_checks():
    cleaned = sanitize_client_history([
        {"role": "user", "content": "create walls"},
        {"role": "user", "content": "STEER: make them 3m", STEER_KEY: True},
        {"role": "user", "content": "plain", STEER_KEY: "yes"},
    ])
    assert [is_steer(m) for m in cleaned] == [False, True, False]
