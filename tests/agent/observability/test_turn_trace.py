"""Turn trace and ui_status: what a client stores with each reply."""

from __future__ import annotations

import asyncio

import json

from agent.observability.turn_trace import (
    MAX_TRACE_BYTES,
    build_turn_trace,
    fallback_ui_status,
    split_ui_status,
    with_ui_status,
)
from tests.agent.fakes import FakeAzure, FakeMCP, text_turn, tool_error, tool_turn
from tests.agent.fakes import make_loop


def build(azure, mcp, **config):
    return make_loop(azure, mcp, **config)


def run(loop, message="Count laptops.", history=None):
    return asyncio.run(loop.run(user_message=message, history=history))


def test_ui_status_is_added_to_copies_and_never_to_the_real_schemas():
    original = [{"type": "function", "function": {
        "name": "create_level", "parameters": {"type": "object", "properties": {"elevation": {"type": "number"}}}}}]
    snapshot = json.dumps(original)

    extended = with_ui_status(original)

    assert "ui_status" in extended[0]["function"]["parameters"]["properties"]
    assert "elevation" in extended[0]["function"]["parameters"]["properties"]
    # The MCP client caches its definitions: they must stay what the server accepts.
    assert json.dumps(original) == snapshot


def test_split_ui_status_cleans_the_text_and_leaves_real_arguments():
    args, status = split_ui_status({"code": "x", "ui_status": "  Creating the **wall**\n on `Level 1` "})
    assert args == {"code": "x"}
    assert status == "Creating the wall on Level 1"

    args, status = split_ui_status({"code": "x", "ui_status": "y" * 200})
    assert len(status) == 80 and status.endswith("…")

    assert split_ui_status({"code": "x", "ui_status": "   "}) == ({"code": "x"}, None)
    assert split_ui_status({"code": "x"}) == ({"code": "x"}, None)
    assert split_ui_status("not a dict") == ("not a dict", None)


def test_fallback_statuses_by_tool_name():
    assert fallback_ui_status("run_select") == "Running a query"
    assert fallback_ui_status("get_business_rules") == "Reading the business rules"
    assert fallback_ui_status("ask_user") == "Asking you a question"
    assert fallback_ui_status("some_new_tool") == "Reading the data"


def _call(call_id, name, args):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def test_trace_covers_only_this_turn_and_groups_parallel_calls_by_step():
    messages = [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": None, "tool_calls": [_call("old", "get_selected_elements", {})]},
        {"role": "tool", "tool_call_id": "old", "content": "old result"},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": "now"},
        {"role": "assistant", "content": None, "tool_calls": [
            _call("a", "get_current_view_info", {"ui_status": "Looking at the open view"}),
            _call("b", "run_select", {"database": "Apps", "sql": "SELECT 1"}),
        ]},
        {"role": "tool", "tool_call_id": "a", "content": "Level 1"},
        {"role": "tool", "tool_call_id": "b", "content": "60 minutes"},
        {"role": "assistant", "content": None, "tool_calls": [_call("c", "create_level", {"elevation": 3})]},
        {"role": "assistant", "content": "done"},
    ]
    verdicts = {"a": {"outcome": "ok", "duration_ms": 12, "ui_status": "Looking at the open view"},
                "b": {"outcome": "error", "error_kind": "runtime"}}

    trace = build_turn_trace(messages, verdicts, "turn-9")

    assert trace["turn_id"] == "turn-9" and "truncated" not in trace
    spans = {s["id"]: s for s in trace["spans"]}
    assert set(spans) == {"a", "b", "c"}  # nothing from the earlier turn
    assert spans["a"]["step"] == spans["b"]["step"] == 1 and spans["c"]["step"] == 2
    assert spans["a"]["ui_status"] == "Looking at the open view"
    assert spans["a"]["input"] == {}  # ui_status is not part of the stored input
    assert spans["b"]["ui_status"] == "Running a query"
    assert spans["b"]["outcome"] == "error" and spans["b"]["error_kind"] == "runtime"
    # Never answered (the turn stopped first): recorded, not dropped.
    assert spans["c"]["outcome"] == "not_run" and spans["c"]["output"] is None


def test_with_a_task_list_each_span_names_the_item_it_worked_on():
    def step(*calls):
        return {"role": "assistant", "content": None, "tool_calls": list(calls)}

    def update(call_id, *updates):
        return _call(call_id, "update_tasks", {"updates": [dict(zip(("id", "status"), u)) for u in updates]})

    messages = [
        {"role": "user", "content": "get the doors on every level"},
        step(_call("a", "get_levels", {})),
        # The steer came after step 1: item 2.
        {"role": "user", "content": "only L1", "steer": True},
        step(update("u1", (1, "dropped"), (2, "in_progress")), _call("b", "get_doors", {})),
        step(_call("c", "get_doors", {"level": "L1"})),
        step(update("u2", (2, "done")), _call("d", "check_links", {})),
        # A failed update starts nothing.
        step(update("u3", (3, "in_progress")), _call("e", "get_doors", {})),
        {"role": "assistant", "content": "35 doors"},
    ]
    steers = [{"id": "s1", "text": "only L1", "after_step": 1, "reply": None}]
    tasks = [{"id": 1, "status": "dropped"}, {"id": 2, "status": "done"}]

    trace = build_turn_trace(messages, {"u3": {"outcome": "error"}}, "t", steers=steers, tasks=tasks)

    owners = {s["id"]: s["task"] for s in trace["spans"]}
    assert owners == {"a": 1, "u1": 2, "b": 2, "c": 2, "u2": 2, "d": 2, "u3": 2, "e": 2}
    # Without a task list, the spans name no item.
    assert all("task" not in s for s in build_turn_trace(messages, {}, "t")["spans"])


def test_no_tool_calls_means_no_trace():
    assert build_turn_trace([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}], {}, "t") is None


def test_oversized_outputs_are_cut_to_fit_one_megabyte():
    huge = "x" * (3 * MAX_TRACE_BYTES)
    messages = [
        {"role": "user", "content": "dump everything"},
        {"role": "assistant", "content": None, "tool_calls": [
            _call("big", "get_current_view_elements", {}), _call("small", "get_current_view_info", {})]},
        {"role": "tool", "tool_call_id": "big", "content": huge},
        {"role": "tool", "tool_call_id": "small", "content": "Level 1"},
    ]
    trace = build_turn_trace(messages, {}, "t")

    assert len(json.dumps(trace, ensure_ascii=False).encode("utf-8")) <= MAX_TRACE_BYTES
    assert trace["truncated"] is True
    spans = {s["id"]: s for s in trace["spans"]}
    assert "characters omitted" in spans["big"]["output"]
    assert spans["small"]["output"] == "Level 1"  # small fields are left alone


def test_each_dispatched_call_streams_the_status_the_trace_stores():
    """The live status a running turn publishes is the one saved with the reply,
    including the fixed fallback when the model wrote none."""
    from agent.host.turn_events import turn_events

    azure = FakeAzure([
        tool_turn(("call_1", "run_select", {"database": "Apps", "sql": "SELECT x", "ui_status": "Counting laptops"})),
        tool_turn(("call_2", "run_select", {"database": "Apps", "sql": "SELECT y"})),
        text_turn("It failed."),
    ])
    mcp = FakeMCP({"run_select": tool_error("query_failed", "Invalid column name 'x'.")})
    queue, _ = turn_events.subscribe("HR-LIVE", "live-turn")
    try:
        with turn_events.bind("HR-LIVE", "live-turn"):
            final = run(build(azure, mcp))
    finally:
        turn_events.unsubscribe("HR-LIVE", "live-turn", queue)

    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    statuses = [(e["call_id"], e["text"]) for e in events if e["type"] == "status"]
    trace = build_turn_trace(final["messages"], final["tool_results"], "t")
    assert statuses == [(s["id"], s["ui_status"]) for s in trace["spans"]]
    assert statuses[1][1] == fallback_ui_status("run_select")
    assert events[-1] == {"type": "done"}


def test_the_models_own_time_is_timed_per_step_streamed_and_saved():
    """Most of a turn is the model planning, which is in no tool call: each
    model call is timed (one per step, then one for the answer), announced
    live as "thinking", and saved in the trace as thinking_ms."""
    from agent.host.turn_events import turn_events

    azure = FakeAzure([
        tool_turn(("call_1", "run_select", {"database": "Apps", "sql": "SELECT 1", "ui_status": "Counting"})),
        tool_turn(("call_2", "run_select", {"database": "Apps", "sql": "SELECT 2", "ui_status": "Checking it"})),
        text_turn("Done."),
    ])
    queue, _ = turn_events.subscribe("HR-THINK", "think-turn")
    try:
        with turn_events.bind("HR-THINK", "think-turn"):
            final = run(build(azure, FakeMCP()))
    finally:
        turn_events.unsubscribe("HR-THINK", "think-turn", queue)

    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    assert [e["type"] for e in events] == ["thinking", "status", "thinking", "status", "thinking", "done"]

    assert len(final["planning_ms"]) == 3
    assert all(isinstance(ms, int) and ms >= 0 for ms in final["planning_ms"])
    trace = build_turn_trace(final["messages"], final["tool_results"], "t", final["planning_ms"])
    assert trace["thinking_ms"] == final["planning_ms"]
    # Without the timings (an older caller) the trace is as before.
    assert "thinking_ms" not in build_turn_trace(final["messages"], final["tool_results"], "t")
