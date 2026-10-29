"""Fitting the planner's context into a token budget.

WHY THIS MODULE EXISTS
----------------------
The planner used to send the last 40 *messages*. That cap ignored size (a
20-word reply and a 200 KB element dump counted the same), and because it
only kept the system messages at the very start, a long tool-heavy turn
pushed out the user's own request, the document context and the equipped
skills -- the model kept working without seeing what it was asked to do.

The planner's message list has four parts::

    [harness rules] [earlier turns] [this turn's anchor] [this turn's tool work]
     leading system   history          per-request system    assistant tool calls,
     messages                          messages + the user   tool results, retry
                                       message               guidance

The harness rules and the anchor are always kept. When the context is over
budget it is reduced in this order:

1. shorten the oldest tool outputs (the latest step's outputs stay whole);
2. drop the oldest earlier turns, whole turns only;
3. drop the oldest steps of this turn, with a note saying so;
4. only when the rules, the anchor and the latest step are still over the
   budget: shorten the latest step's outputs too, longest first.

The user's steers (agent/host/steers.py) are part of this turn's tool work,
not the start of a turn, and step 3 keeps them: a correction must not vanish
while the work it corrects goes on.

Whatever the budget, no tool output longer than :data:`MAX_OUTPUT_CHARS` is
sent: the API refuses a single message that long.

Prompt caching
--------------
Every planner call resends the whole context. Providers that cache repeated
prompt prefixes only help if that prefix stays the same between calls, so
trimming must not shift a little on every call. The reductions above are
recorded in a *plan* that only ever grows within a turn, and when the budget
is crossed the context is cut down to a low watermark (50% of the budget),
not to just under it. Most calls then reuse the previous plan unchanged.

Sizes are estimated (characters / 4): close enough for a limit, and free.
The budget covers the messages only; tool definitions and the model's answer
come on top, which is why it is set well below the model's context window.

Earlier turns
-------------
The budget rarely triggers, yet most of a long conversation is the tool work
of turns the user has moved on from: old query results, failed queries
included. So :func:`trim_earlier_turns` runs once per turn, when the
conversation is loaded, on every turn before the last ``full_turns``: tool
outputs are cut to :data:`SHORTENED_OUTPUT_CHARS`. Every tool call and tool
result stays, so the sequence remains valid. The result depends on the messages alone and trimming it again
changes nothing, so the prefix is the same for every call of a turn.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agent.memory.history import find_safe_cut_points, is_steer, is_turn_start

__all__ = [
    "LOW_WATERMARK",
    "MAX_OUTPUT_CHARS",
    "describe_plan",
    "estimate_message_tokens",
    "estimate_tokens",
    "fit_context",
    "trim_earlier_turns",
]

Message = dict[str, Any]

CHARS_PER_TOKEN = 4
#: Per-message framing (role, separators) the character count does not see.
MESSAGE_OVERHEAD_TOKENS = 4
#: When trimming is needed, trim to this share of the budget, so the next
#: few calls fit without changing the plan (and the cached prefix).
LOW_WATERMARK = 0.5
#: What a shortened tool output keeps.
SHORTENED_OUTPUT_CHARS = 2000
#: The longest tool output ever sent: the API refuses any one string over
#: 1,048,576 characters (``string_above_max_length``), whatever the budget.
MAX_OUTPUT_CHARS = 1_000_000
_SHORTENED_NOTE = (
    "\n[shortened to fit the context budget: {omitted} characters omitted. "
    "Re-run the tool if you need the full result again.]"
)
_OMITTED_STEPS_NOTE = (
    "Context note: {count} earlier messages from this task were omitted to fit "
    "the context budget. Their results were already used; re-run a read tool "
    "if you need them again."
)


_EARLIER_TURN_NOTE = (
    "\n[shortened: {total:,} chars from an earlier turn. "
    "Run the query again if you need this.]"
)
#: Recognises an output :func:`trim_earlier_turns` already shortened, so a
#: persisted history is never shortened twice (which would change it).
_EARLIER_TURN_NOTE_RE = re.compile(
    r"\n\[shortened: [\d,]+ chars from an earlier turn\. "
    r"Run the query again if you need this\.\]$"
)


def estimate_message_tokens(message: Message) -> int:
    chars = 0
    content = message.get("content")
    if isinstance(content, str):
        chars += len(content)
    tool_calls = message.get("tool_calls")
    if tool_calls:
        chars += len(json.dumps(tool_calls, ensure_ascii=False, default=str))
    return chars // CHARS_PER_TOKEN + MESSAGE_OVERHEAD_TOKENS


def estimate_tokens(messages: list[Message]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)


class _Segments:
    def __init__(self, messages: list[Message]) -> None:
        pinned_end = 0
        while pinned_end < len(messages) and messages[pinned_end].get("role") == "system":
            pinned_end += 1
        user_index = next(
            (i for i in range(len(messages) - 1, pinned_end - 1, -1)
             if is_turn_start(messages[i])),
            None,
        )
        self.valid = user_index is not None
        if not self.valid:
            self.pinned, self.history, self.anchor, self.turn = messages, [], [], []
            return
        anchor_start = user_index
        while anchor_start > pinned_end and messages[anchor_start - 1].get("role") == "system":
            anchor_start -= 1
        self.pinned = messages[:pinned_end]
        self.history = messages[pinned_end:anchor_start]
        self.anchor = messages[anchor_start:user_index + 1]
        self.turn = messages[user_index + 1:]

    def history_cuts(self) -> list[int]:
        """Whole-turn boundaries: dropping up to one keeps every turn intact."""
        cuts = [i for i, m in enumerate(self.history) if is_turn_start(m) and i > 0]
        return [*cuts, len(self.history)]

    def _latest_step_start(self) -> int | None:
        for index in range(len(self.turn) - 1, -1, -1):
            message = self.turn[index]
            if message.get("role") == "assistant" and message.get("tool_calls"):
                return index
        return None

    def turn_cuts(self) -> list[int]:
        """Step boundaries in this turn, never past the latest tool-call step.

        Bounded by the newest assistant tool-call message rather than by the
        last safe point: retry guidance appended after a failed call is its
        own safe point, and cutting there would drop the step it is about.
        """
        latest = self._latest_step_start()
        if latest is None:
            return []
        return [p for p in find_safe_cut_points(self.turn) if 0 < p <= latest]

    def latest_step_call_ids(self) -> set[str]:
        """Tool results of the newest assistant tool-call message: kept whole."""
        latest = self._latest_step_start()
        if latest is None:
            return set()
        return {c.get("id") for c in self.turn[latest]["tool_calls"] if c.get("id")}


def _empty_plan() -> dict[str, Any]:
    return {"history_drop": 0, "turn_drop": 0, "shortened": []}


def _normalise(plan: dict[str, Any] | None, segments: _Segments) -> dict[str, Any]:
    """A plan from an earlier call, clamped to cuts that are still valid."""
    result = _empty_plan()
    if not isinstance(plan, dict):
        return result
    history_drop = plan.get("history_drop", 0)
    if history_drop in segments.history_cuts():
        result["history_drop"] = history_drop
    turn_drop = plan.get("turn_drop", 0)
    if turn_drop in segments.turn_cuts():
        result["turn_drop"] = turn_drop
    shortened = plan.get("shortened")
    if isinstance(shortened, list):
        result["shortened"] = [s for s in shortened if isinstance(s, str)]
    return result


def _shorten(message: Message, limit: int = SHORTENED_OUTPUT_CHARS) -> Message:
    content = message.get("content")
    if not isinstance(content, str) or len(content) <= limit:
        return message
    omitted = len(content) - limit
    return {**message, "content": content[:limit] + _SHORTENED_NOTE.format(omitted=omitted)}


def _apply(segments: _Segments, plan: dict[str, Any]) -> list[Message]:
    shortened = set(plan["shortened"])

    def keep(message: Message) -> Message:
        if message.get("role") != "tool":
            return message
        if message.get("tool_call_id") in shortened:
            return _shorten(message)
        return _shorten(message, MAX_OUTPUT_CHARS)

    turn = segments.turn[plan["turn_drop"]:]
    # A cut point is never inside a tool-call group, so the steers of the
    # dropped steps can stand together after the note.
    steers = [m for m in segments.turn[:plan["turn_drop"]] if is_steer(m)]
    note: list[Message] = []
    count = plan["turn_drop"] - len(steers)
    if count > 0:
        note = [{"role": "system", "content": _OMITTED_STEPS_NOTE.format(count=count)}]
    return [
        *segments.pinned,
        *(keep(m) for m in segments.history[plan["history_drop"]:]),
        *segments.anchor,
        *note,
        *steers,
        *(keep(m) for m in turn),
    ]


def _fits(messages: list[Message], token_limit: int, max_messages: int) -> bool:
    return estimate_tokens(messages) <= token_limit and len(messages) <= max_messages


def describe_plan(plan: dict[str, Any] | None) -> dict[str, int]:
    """What a plan removed, for showing the user (the context ring)."""
    plan = plan or _empty_plan()
    return {
        "shortened_outputs": len(plan.get("shortened") or []),
        "dropped_history_messages": int(plan.get("history_drop") or 0),
        "dropped_turn_messages": int(plan.get("turn_drop") or 0),
    }


def fit_context(
    messages: list[Message],
    *,
    token_budget: int,
    max_messages: int,
    plan: dict[str, Any] | None = None,
    target: int | None = None,
) -> tuple[list[Message], dict[str, Any]]:
    """``(messages to send, plan to keep for the next call of this turn)``.

    ``plan`` is the previous call's result; pass ``None`` at the start of a
    turn. When over ``token_budget``, the context is cut to ``target``
    (default: :data:`LOW_WATERMARK` of the budget). The returned list is
    always a well-formed tool-call sequence.
    """
    segments = _Segments(list(messages))
    if not segments.valid:
        return list(messages), _empty_plan()
    plan = _normalise(plan, segments)
    fitted = _apply(segments, plan)
    if _fits(fitted, token_budget, max_messages):
        return fitted, plan

    target = max(1, int(token_budget * LOW_WATERMARK) if target is None else target)

    # 1. Shorten old tool outputs, oldest first. Only size helps here.
    latest = segments.latest_step_call_ids()
    candidates = [
        m.get("tool_call_id")
        for m in [*segments.history[plan["history_drop"]:], *segments.turn[plan["turn_drop"]:]]
        if m.get("role") == "tool"
        and m.get("tool_call_id") not in latest
        and m.get("tool_call_id") not in plan["shortened"]
        and len(m.get("content") or "") > SHORTENED_OUTPUT_CHARS
        # Already cut by `trim_earlier_turns`: nothing left to gain.
        and not _EARLIER_TURN_NOTE_RE.search(m.get("content") or "")
    ]
    for call_id in candidates:
        if estimate_tokens(fitted) <= target:
            break
        plan["shortened"].append(call_id)
        fitted = _apply(segments, plan)
    if _fits(fitted, target, max_messages):
        return fitted, plan

    # 2. Drop the oldest earlier turns, whole turns only.
    for cut in segments.history_cuts():
        if cut <= plan["history_drop"]:
            continue
        plan["history_drop"] = cut
        fitted = _apply(segments, plan)
        if _fits(fitted, target, max_messages):
            return fitted, plan

    # 3. Drop the oldest steps of this turn; the latest step always stays.
    for cut in segments.turn_cuts():
        if cut <= plan["turn_drop"]:
            continue
        plan["turn_drop"] = cut
        fitted = _apply(segments, plan)
        if _fits(fitted, target, max_messages):
            return fitted, plan

    # 4. Harness rules + anchor + latest step alone are over the budget: the
    # latest step's outputs are shortened too, longest first, rather than
    # sending a context the model refuses (and refuses again on the retry).
    if not _fits(fitted, token_budget, max_messages):
        outputs = sorted(
            (
                m for m in segments.turn[plan["turn_drop"]:]
                if m.get("role") == "tool"
                and m.get("tool_call_id") in latest
                and m.get("tool_call_id") not in plan["shortened"]
                and len(m.get("content") or "") > SHORTENED_OUTPUT_CHARS
            ),
            key=lambda m: -len(m.get("content") or ""),
        )
        for message in outputs:
            if estimate_tokens(fitted) <= target:
                break
            plan["shortened"].append(message.get("tool_call_id"))
            fitted = _apply(segments, plan)

    # Still over: send what is left; the planner's context-length retry is
    # the last line of defence.
    return fitted, plan


# -- earlier turns ------------------------------------------------------------


def _shorten_earlier(message: Message, limit: int) -> Message:
    content = message.get("content")
    if not isinstance(content, str) or len(content) <= limit:
        return message
    if _EARLIER_TURN_NOTE_RE.search(content):
        return message
    return {
        **message,
        "content": content[:limit] + _EARLIER_TURN_NOTE.format(total=len(content)),
    }


def trim_earlier_turns(
    messages: list[Message], full_turns: int, *, limit: int = SHORTENED_OUTPUT_CHARS
) -> list[Message]:
    """``messages`` with the tool outputs of every turn before the last ``full_turns`` shortened.

    A turn starts at a user message that is not a steer; ``messages`` holds
    earlier turns only (the current one is added after this runs). Each tool
    output over ``limit`` characters is cut to ``limit`` with a note giving
    its length. No message is removed or reordered, so a valid sequence stays
    valid.

    Pure and idempotent: the same input always gives the same output, and
    trimming the output again returns it unchanged.
    """
    starts = [i for i, m in enumerate(messages) if is_turn_start(m)]
    if full_turns <= 0:
        keep_from = len(messages)
    elif len(starts) > full_turns:
        keep_from = starts[-full_turns]
    else:
        return list(messages)
    earlier = [
        _shorten_earlier(m, limit) if m.get("role") == "tool" else m
        for m in messages[:keep_from]
    ]
    return [*earlier, *messages[keep_from:]]
