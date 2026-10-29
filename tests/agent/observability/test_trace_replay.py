"""Trace replay: earlier turns' tool calls come back with their replies."""

from __future__ import annotations

import asyncio

import json

from shared.api_schemas import ChatRequest
from agent.memory.context_budget import estimate_tokens
from agent.memory.history import validate_message_sequence
from agent.prompts.templates import EARLIER_TOOL_RESULTS_NOTE
from agent.observability.turn_trace import build_turn_trace, expand_history_traces, trace_messages
from tests.agent.fakes import FakeAzure, FakeMCP, text_turn
from tests.agent.fakes import make_loop


def build(azure, mcp, **config):
    return make_loop(azure, mcp, **config)


def run(loop, message, history=None):
    return asyncio.run(loop.run(user_message=message, history=history))



def _call(call_id, name, args):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _stored_trace():
    """A trace as the agent returned it and the desktop stored it."""
    messages = [
        {"role": "user", "content": "list the walls on Level 2"},
        {"role": "assistant", "content": None, "tool_calls": [
            _call("a", "get_elements_by_category", {"category": "Walls", "ui_status": "Finding the walls"}),
            _call("b", "get_current_view_info", {}),
        ]},
        {"role": "tool", "tool_call_id": "a", "content": "ids: 1201, 1202"},
        {"role": "tool", "tool_call_id": "b", "content": "Level 2 plan"},
        {"role": "assistant", "content": None, "tool_calls": [_call("c", "create_level", {"elevation": 3})]},
        {"role": "assistant", "content": "Level 2 has walls 1201 and 1202."},
    ]
    trace = build_turn_trace(messages, {"a": {"outcome": "ok"}, "b": {"outcome": "ok"}}, "turn-1")
    return json.loads(json.dumps(trace))  # what comes back over the wire


def test_a_stored_trace_becomes_the_tool_calls_it_records():
    replayed = trace_messages(_stored_trace())

    assert [m["role"] for m in replayed] == ["assistant", "tool", "tool"]
    calls = replayed[0]["tool_calls"]
    assert [c["id"] for c in calls] == ["a", "b"], "parallel calls stay one step"
    assert json.loads(calls[0]["function"]["arguments"]) == {"category": "Walls"}, "no ui_status"
    assert replayed[1] == {"role": "tool", "tool_call_id": "a", "content": "ids: 1201, 1202"}
    # `c` never ran: it has no result to replay, so its call is left out too.
    assert all(c["id"] != "c" for m in replayed for c in m.get("tool_calls") or [])
    assert validate_message_sequence(replayed) == []


def test_untrusted_spans_are_skipped_not_trusted():
    good = {"id": "ok", "step": 1, "tool": "get_current_view_info", "input": {}, "output": "fine"}
    trace = {"v": 1, "spans": [
        good,
        {**good, "id": "bad-name", "tool": "drop table; --"},
        {**good, "id": "", "tool": "get_current_view_info"},
        {**good, "id": "x" * 500},
        {**good, "id": "no-output", "output": None},
        {**good, "id": "odd-input", "input": 42},
        "not a span",
    ]}
    replayed = trace_messages(trace)
    ids = [c["id"] for c in replayed[0]["tool_calls"]]
    assert ids == ["ok", "odd-input"]
    assert replayed[0]["tool_calls"][1]["function"]["arguments"] == "{}"
    assert trace_messages(None) == [] and trace_messages({"spans": "x"}) == []


def test_replayed_arguments_are_always_a_json_object():
    """A provider that parses past arguments must never see broken JSON."""
    truncated = '{"code": "very long… [truncated]'
    trace = {"spans": [
        {"id": "t", "step": 1, "tool": "send_code_to_revit", "input": truncated, "output": "ok"},
        {"id": "j", "step": 1, "tool": "send_code_to_revit", "input": '{"code": "x"}', "output": "ok"},
        {"id": "l", "step": 1, "tool": "send_code_to_revit", "input": "[1, 2]", "output": "ok"},
    ]}
    calls = {c["id"]: json.loads(c["function"]["arguments"]) for c in trace_messages(trace)[0]["tool_calls"]}
    assert calls["t"] == {"truncated_input": truncated}, "the cut text is kept, wrapped"
    assert calls["j"] == {"code": "x"}, "valid stored JSON is replayed as-is"
    assert calls["l"] == {"truncated_input": "[1, 2]"}


def test_tool_calls_go_back_before_the_reply_they_led_to():
    history = [
        {"role": "user", "content": "list the walls"},
        {"role": "assistant", "content": "Walls 1201 and 1202.", "trace": _stored_trace()},
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "You're welcome."},
    ]
    expanded = expand_history_traces(history)

    roles = [m["role"] for m in expanded]
    assert roles == ["user", "assistant", "tool", "tool", "assistant", "user", "assistant"]
    assert expanded[4] == {"role": "assistant", "content": "Walls 1201 and 1202."}
    assert "trace" in history[1], "the caller's history is not modified"


def test_a_follow_up_turn_sees_the_earlier_results_and_the_staleness_note():
    azure = FakeAzure([text_turn("Tagged 1201 and 1202.")])
    history = [
        {"role": "user", "content": "list the walls on Level 2"},
        {"role": "assistant", "content": "Level 2 has walls 1201 and 1202.", "trace": _stored_trace()},
    ]
    run(build(azure, FakeMCP()), "now tag those walls", history=history)

    sent = azure.calls[0]["messages"]
    assert validate_message_sequence(sent) == []
    contents = [m.get("content") for m in sent]
    answer = contents.index("Level 2 has walls 1201 and 1202.")
    assert {"role": "tool", "tool_call_id": "a", "content": "ids: 1201, 1202"} in sent[:answer]
    # The note sits with this request, just before the user's message.
    note = contents.index(EARLIER_TOOL_RESULTS_NOTE)
    assert note > answer and sent[-1] == {"role": "user", "content": "now tag those walls"}


def test_no_earlier_tool_results_means_no_staleness_note():
    azure = FakeAzure([text_turn("Hi.")])
    run(build(azure, FakeMCP()), "hello", history=[
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello."}])
    assert all(m.get("content") != EARLIER_TOOL_RESULTS_NOTE for m in azure.calls[0]["messages"])


def test_large_replayed_results_are_compacted_like_any_history():
    big = {"spans": [{"id": f"r{i}", "step": 1, "tool": "get_elements_by_category",
                      "input": {}, "output": "x" * 40_000} for i in range(3)]}
    history = []
    for i in range(6):
        history += [{"role": "user", "content": f"question {i}"},
                    {"role": "assistant", "content": f"answer {i}",
                     "trace": {"spans": [{**s, "id": f"{s['id']}-{i}"} for s in big["spans"]]}}]
    azure = FakeAzure([text_turn("Done.")])
    run(build(azure, FakeMCP(), context_token_budget=60_000), "and now?", history=history)

    sent = azure.calls[0]["messages"]
    assert estimate_tokens(sent) <= 60_000
    assert validate_message_sequence(sent) == []
    assert sent[-1] == {"role": "user", "content": "and now?"}


def test_the_api_accepts_a_trace_on_history_messages():
    request = ChatRequest.model_validate({"message": "go", "history": [
        {"role": "assistant", "content": "done", "trace": _stored_trace()}]})
    assert request.history[0].trace["turn_id"] == "turn-1"
