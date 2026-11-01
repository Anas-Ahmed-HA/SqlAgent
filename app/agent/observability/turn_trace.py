"""Per-turn tool trace: what the agent did, in a shape a client can store.

Two concerns live here because they meet in the same record:

``ui_status``
    A short, user-facing description of a tool call ("Checking the fire
    ratings in section 8"). The model writes it as an optional extra
    argument on every tool call, so it costs no extra model round trip.
    :func:`with_ui_status` adds that argument to the tool definitions sent to
    the model, and :func:`split_ui_status` removes it again before the call is
    fingerprinted or executed. The MCP server never sees it. When the model
    omits it, :func:`fallback_ui_status` supplies a fixed one.

Turn trace
    :func:`build_turn_trace` turns one turn's messages into span-like records
    (``id, step, tool, ui_status, input, output, outcome, error_kind,
    duration_ms``). It is built once, from the messages, after the turn ends,
    so no execution path can be missed. The verdict for each call comes from
    the ``tool_results`` record the loop fills in.

A client stores the trace verbatim and never needs to understand it.

Replay
    A client sends each stored trace back on the assistant message it
    belongs to. :func:`expand_history_traces` rebuilds the assistant
    tool-call and tool-result messages from it, so a later turn sees what
    earlier turns read (object names, errors), not just the final answer. The context budget shortens or drops them like any other history.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from agent.memory.history import STEER_KEY, is_turn_start
from agent.prompts.templates import build_steer_message
from agent.tools.tasks import UPDATE_TASKS_TOOL_NAME

__all__ = [
    "MAX_TRACE_BYTES",
    "UI_STATUS_ARG",
    "build_turn_trace",
    "expand_history_traces",
    "fallback_ui_status",
    "split_ui_status",
    "trace_messages",
    "turn_step_count",
    "with_ui_status",
]

#: Argument name added to every tool. Namespaced so it can never shadow a real
#: tool parameter (a tool might have a plain `status`).
UI_STATUS_ARG = "ui_status"
_UI_STATUS_MAX_CHARS = 80
_UI_STATUS_DESCRIPTION = (
    "Optional. One short plain sentence, in the user's language, telling the "
    "user what this call is doing, e.g. 'Counting laptops per company'. "
    f"At most {_UI_STATUS_MAX_CHARS} characters, no markdown."
)

#: Ceiling for one message's stored trace. The conversation service has no
#: size limit, but every trace is returned each time the conversation is
#: opened, so one huge result must not make a conversation slow to load.
MAX_TRACE_BYTES = 1024 * 1024
_TRUNCATION_NOTE = "\n[truncated: {omitted} characters omitted to keep the stored trace under 1 MB]"

_FIXED_UI_STATUS = {
    "list_databases": "Looking up the databases",
    "get_business_rules": "Reading the business rules",
    "list_objects": "Listing the tables and views",
    "describe_object": "Reading the columns",
    "run_select": "Running a query",
    "ask_user": "Asking you a question",
    "update_tasks": "Updating the task list",
    "show_cards": "Preparing the cards",
    "ask_with_card": "Preparing a card for you",
    "create_dashboard": "Building a dashboard",
}


def fallback_ui_status(tool_name: str) -> str:
    """Fixed status for a call whose model-written one is missing or unusable."""
    if tool_name not in _FIXED_UI_STATUS and tool_name.startswith("create_"):
        return "Drawing a chart"  # the charts server's tools
    return _FIXED_UI_STATUS.get(tool_name, "Reading the data")


def with_ui_status(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copies of the tool definitions with the optional ``ui_status`` argument.

    Copies, because the MCP client caches its definitions and the real schemas
    must stay exactly what the server's tools accept.
    """
    extended: list[dict[str, Any]] = []
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            extended.append(tool)
            continue
        tool = copy.deepcopy(tool)
        parameters = tool["function"].get("parameters")
        if not isinstance(parameters, dict):
            parameters = {"type": "object", "properties": {}}
            tool["function"]["parameters"] = parameters
        properties = parameters.setdefault("properties", {})
        if isinstance(properties, dict) and UI_STATUS_ARG not in properties:
            properties[UI_STATUS_ARG] = {"type": "string", "description": _UI_STATUS_DESCRIPTION}
        extended.append(tool)
    return extended


def _clean_ui_status(value: Any) -> str | None:
    """One plain line, or ``None`` when the model's text is not usable."""
    if not isinstance(value, str):
        return None
    text = re.sub(r"[`*_#>\[\]]", "", value)
    text = " ".join(text.split())
    if not text:
        return None
    if len(text) > _UI_STATUS_MAX_CHARS:
        text = text[: _UI_STATUS_MAX_CHARS - 1].rstrip() + "…"
    return text


def split_ui_status(arguments: Any) -> tuple[Any, str | None]:
    """``(arguments without ui_status, cleaned ui_status or None)``.

    Every consumer of tool arguments (the repeat-call fingerprint, document
    prefetch, execution, tracing) must see the arguments without it: a
    reworded status would otherwise make a failed call look new.
    """
    if not isinstance(arguments, dict) or UI_STATUS_ARG not in arguments:
        return arguments, None
    cleaned = dict(arguments)
    status = _clean_ui_status(cleaned.pop(UI_STATUS_ARG))
    return cleaned, status


def _turn_slice(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Messages produced by this turn: everything after the message it started
    with (the user's steers are part of the turn)."""
    for index in range(len(messages) - 1, -1, -1):
        if is_turn_start(messages[index]):
            return messages[index + 1:]
    return list(messages)


def turn_step_count(messages: list[dict[str, Any]]) -> int:
    """How many tool-call steps the turn in ``messages`` has made so far."""
    return sum(
        1 for m in _turn_slice(messages or [])
        if m.get("role") == "assistant" and m.get("tool_calls")
    )


def build_turn_trace(
    messages: list[dict[str, Any]],
    tool_results: dict[str, dict[str, Any]] | None,
    turn_id: str,
    planning_ms: list[int] | None = None,
    *,
    steers: list[dict[str, Any]] | None = None,
    tasks: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """The turn's tool calls as stored spans, or ``None`` when it made none
    and took no steers.

    ``planning_ms`` (the model's time before each step, then before the
    answer) is stored as ``thinking_ms``, so a client can show the time
    that is not in any tool call. The steers the turn took (``id``, ``text``,
    ``after_step``, ``reply``) and its final task list are stored as
    ``steers`` and ``tasks``, so a reopened conversation shows them and a
    later turn replays the steers where they came (:func:`trace_messages`).
    With a task list, each span names the item it worked on in ``task``
    (:func:`_step_tasks`), so a client can group the steps by item.
    """
    turn = _turn_slice(messages or [])
    results_by_call = {
        m.get("tool_call_id"): m.get("content")
        for m in turn
        if m.get("role") == "tool" and m.get("tool_call_id")
    }
    verdicts = tool_results or {}
    spans: list[dict[str, Any]] = []
    step_tasks = _step_tasks(turn, verdicts, steers) if tasks else {}
    step = 0
    for message in turn:
        calls = message.get("tool_calls") if message.get("role") == "assistant" else None
        if not calls:
            continue
        step += 1
        for call in calls:
            function = call.get("function") or {}
            name = function.get("name") or ""
            try:
                parsed = json.loads(function.get("arguments") or "{}")
            except (TypeError, ValueError):
                parsed = function.get("arguments")
            arguments, written_status = split_ui_status(parsed)
            verdict = verdicts.get(call.get("id"), {})
            output = results_by_call.get(call.get("id"))
            spans.append({
                "id": call.get("id"),
                "step": step,
                "tool": name,
                "ui_status": verdict.get("ui_status") or written_status or fallback_ui_status(name),
                "input": arguments,
                "output": output if isinstance(output, str) else None,
                "outcome": verdict.get("outcome") or ("unknown" if output is not None else "not_run"),
                "error_kind": verdict.get("error_kind"),
                "duration_ms": verdict.get("duration_ms"),
                **({"interaction": verdict["interaction"]} if verdict.get("interaction") else {}),
                **({"task": step_tasks[step]} if step in step_tasks else {}),
            })
    if not spans and not steers:
        return None
    trace: dict[str, Any] = {"v": 1, "turn_id": turn_id, "spans": spans}
    thinking = [int(ms) for ms in planning_ms or [] if isinstance(ms, (int, float)) and ms >= 0]
    if thinking:
        trace["thinking_ms"] = thinking
    if steers:
        trace["steers"] = [
            {key: steer.get(key) for key in ("id", "text", "after_step", "reply")}
            for steer in steers
        ]
    if tasks:
        trace["tasks"] = [dict(task) for task in tasks]
    return _fit(trace, MAX_TRACE_BYTES)


_FINISHED_TASK = frozenset({"done", "dropped"})


def _step_tasks(
    turn: list[dict[str, Any]],
    verdicts: dict[str, dict[str, Any]],
    steers: list[dict[str, Any]] | None,
) -> dict[int, int]:
    """The task-list item each tool-call step worked on, by step number.

    Replayed from the step's own ``update_tasks`` calls: the request (item 1)
    until the model starts another item. A step that starts an item belongs to
    it (the newest one it starts); once the current item is finished, the
    next steps belong to the first item still unfinished, if any. Item ``n``
    (``n`` > 1) is the steer stored at ``n - 2``, on the list from the step
    after the one it came after.
    """
    arrivals = [
        item.get("after_step") if isinstance(item.get("after_step"), int) else 0
        for item in steers or () if isinstance(item, dict)
    ]
    statuses: dict[int, str] = {1: "in_progress"}
    current, step, owners = 1, 0, {}
    for message in turn:
        calls = message.get("tool_calls") if message.get("role") == "assistant" else None
        if not calls:
            continue
        step += 1
        for index, after in enumerate(arrivals):
            if after < step:
                statuses.setdefault(index + 2, "open")
        changes: dict[int, str] = {}
        for call in calls:
            function = call.get("function") or {}
            if function.get("name") != UPDATE_TASKS_TOOL_NAME:
                continue
            if verdicts.get(call.get("id"), {}).get("outcome") in {"error", "blocked", "not_run"}:
                continue
            try:
                updates = json.loads(function.get("arguments") or "{}").get("updates")
            except (AttributeError, TypeError, ValueError):
                continue
            for update in updates if isinstance(updates, list) else ():
                task_id = update.get("id") if isinstance(update, dict) else None
                if isinstance(task_id, str) and task_id.strip().isdigit():
                    task_id = int(task_id)
                if isinstance(task_id, int) and task_id in statuses:
                    changes[task_id] = str(update.get("status"))
        started = [task_id for task_id, status in changes.items() if status == "in_progress"]
        if started:
            current = max(started)
        owners[step] = current
        statuses.update(changes)
        if statuses.get(current) in _FINISHED_TASK:
            left = sorted(task_id for task_id, status in statuses.items() if status not in _FINISHED_TASK)
            current = left[0] if left else current
    return owners


def _size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def _fit(trace: dict[str, Any], limit: int) -> dict[str, Any]:
    """Shorten the largest inputs/outputs until the trace fits ``limit`` bytes.

    Normal turns are untouched. When one field is huge (a full-model element
    dump), only the biggest fields are cut -- to the same length -- and each
    cut says how much was omitted.
    """
    if _size(trace) <= limit:
        return trace
    spans = trace["spans"]
    # Inputs are dicts; a cut one is stored as its truncated JSON text.
    fields: list[tuple[int, str, str]] = []
    for index, span in enumerate(spans):
        for key in ("output", "input"):
            value = span.get(key)
            if value is None:
                continue
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            fields.append((index, key, text))
    # Largest per-field length that fits: binary search on a common cap.
    low, high = 0, max((len(text) for _, _, text in fields), default=0)
    best: dict[str, Any] = trace
    while low <= high:
        cap = (low + high) // 2
        candidate = copy.deepcopy(trace)
        # Part of the stored size, so it is counted while fitting.
        candidate["truncated"] = True
        for index, key, text in fields:
            if len(text) > cap:
                omitted = len(text) - cap
                candidate["spans"][index][key] = text[:cap] + _TRUNCATION_NOTE.format(omitted=omitted)
        if _size(candidate) <= limit:
            best, low = candidate, cap + 1
        else:
            high = cap - 1
    return best


#: What Azure accepts as a function name; anything else would get the whole
#: request rejected, so such a span is not replayed.
_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_CALL_ID_CHARS = 200


def _replay_arguments(stored: Any) -> str:
    """Stored input as a JSON object string, which every provider accepts.

    An input cut to fit the 1 MB trace is stored as its truncated JSON text,
    which no longer parses; a provider that parses past arguments would reject
    the whole request, so that text is wrapped in an object instead.
    """
    if isinstance(stored, dict):
        return json.dumps(stored, ensure_ascii=False)
    if isinstance(stored, str):
        try:
            if isinstance(json.loads(stored), dict):
                return stored
        except ValueError:
            pass
        return json.dumps({"truncated_input": stored}, ensure_ascii=False)
    return "{}"


def trace_messages(trace: Any) -> list[dict[str, Any]]:
    """The assistant tool-call and tool-result messages a stored trace records.

    The trace comes back from the client, so it is validated, not trusted:
    spans with a bad id or tool name, or with no stored output, are skipped.
    One assistant message per step, each followed by its results. The user's
    steers go back where the model read them, as one steer message per
    delivery; the model's reply to them is the text of the step that follows
    (or an assistant message of its own when no step followed).
    """
    if not isinstance(trace, dict) or not isinstance(trace.get("spans"), list):
        return []
    steps: dict[int, list[dict[str, Any]]] = {}
    for span in trace["spans"]:
        if not isinstance(span, dict):
            continue
        call_id, name, output = span.get("id"), span.get("tool"), span.get("output")
        if not (isinstance(call_id, str) and 0 < len(call_id) <= _MAX_CALL_ID_CHARS):
            continue
        if not (isinstance(name, str) and _TOOL_NAME.match(name)):
            continue
        if not isinstance(output, str):
            continue
        step = span.get("step")
        steps.setdefault(step if isinstance(step, int) else 0, []).append(span)
    deliveries = _steer_deliveries(trace.get("steers"))
    messages: list[dict[str, Any]] = []

    def steer(delivery: tuple[int, list[str], str | None]) -> str | None:
        messages.append({"role": "user", "content": build_steer_message(delivery[1]), STEER_KEY: True})
        return delivery[2]

    for step in sorted(steps):
        reply = None
        while deliveries and deliveries[0][0] < step:
            reply = steer(deliveries.pop(0))
        spans = steps[step]
        calls = [
            {"id": span["id"], "type": "function",
             "function": {"name": span["tool"], "arguments": _replay_arguments(span.get("input"))}}
            for span in spans
        ]
        messages.append({"role": "assistant", "content": reply, "tool_calls": calls})
        messages.extend({"role": "tool", "tool_call_id": span["id"], "content": span["output"]}
                        for span in spans)
    for delivery in deliveries:
        reply = steer(delivery)
        if reply:
            messages.append({"role": "assistant", "content": reply})
    return messages


def _steer_deliveries(stored: Any) -> list[tuple[int, list[str], str | None]]:
    """Stored steers as ``(after_step, texts, reply)``, one per delivery, in order.

    Steers the model read together share ``after_step``; the reply is on the
    last of them. Entries without text are skipped.
    """
    deliveries: list[tuple[int, list[str], str | None]] = []
    for item in stored if isinstance(stored, list) else ():
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        after = item.get("after_step")
        after = after if isinstance(after, int) and not isinstance(after, bool) and after >= 0 else 0
        reply = item.get("reply") if isinstance(item.get("reply"), str) and item["reply"].strip() else None
        if deliveries and deliveries[-1][0] == after:
            _, texts, earlier = deliveries[-1]
            deliveries[-1] = (after, [*texts, text], reply or earlier)
        else:
            deliveries.append((after, [text], reply))
    deliveries.sort(key=lambda delivery: delivery[0])
    return deliveries


def expand_history_traces(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Client history with each assistant message's ``trace`` replayed before it.

    The tool calls came before the answer, so that is where they go back.
    The ``trace`` key itself is removed; messages without one are unchanged.
    """
    expanded: list[dict[str, Any]] = []
    for message in history or []:
        if isinstance(message, dict) and message.get("role") == "assistant" and "trace" in message:
            message = dict(message)
            expanded.extend(trace_messages(message.pop("trace")))
        expanded.append(message)
    return expanded
