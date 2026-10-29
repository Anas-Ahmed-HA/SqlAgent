"""
Conversation-history handling.

WHY THIS MODULE EXISTS
----------------------
The original service capped history with a plain slice:

    _session_histories[session_id] = result_messages[-MAX_HISTORY_MESSAGES:]

That is a latent production bug. An Azure chat completion is rejected with a
400 when a `tool` message has no preceding `assistant` message carrying the
matching `tool_call_id`, and equally when an `assistant` message declares
`tool_calls` that never receive responses. A blind tail slice cuts through the
middle of a tool-call group roughly whenever the last turn used tools -- which,
in a tool-using agent, is most turns. The failure is delayed to the *next* request
in the session, which is exactly the kind of bug that is painful to trace.

Truncation here therefore only ever cuts at a **turn boundary**: an index at
which no tool call is outstanding.

STEERS
------
A note the user sends while a turn runs (agent/host/steers.py) joins the
messages as a ``user`` message, so the model gives it the user's authority,
marked with :data:`STEER_KEY`. It is part of the turn it steers, not the start
of a new one: code that looks for where a turn starts skips it
(:func:`is_turn_start`). The mark is ours; :func:`provider_messages` removes
it before a request is sent.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "STEER_KEY",
    "extract_last_assistant_text",
    "find_safe_cut_points",
    "is_steer",
    "is_turn_start",
    "provider_messages",
    "sanitize_client_history",
    "strip_for_persistence",
    "truncate_messages",
    "validate_message_sequence",
]

Message = dict[str, Any]

#: Marks a ``user`` message that steers the running turn.
STEER_KEY = "steer"


def is_steer(message: Any) -> bool:
    """Whether ``message`` is a steer: a user's note to the running turn."""
    return (
        isinstance(message, dict)
        and message.get("role") == "user"
        and message.get(STEER_KEY) is True
    )


def is_turn_start(message: Any) -> bool:
    """Whether ``message`` starts a turn: a user message that is not a steer."""
    return isinstance(message, dict) and message.get("role") == "user" and not is_steer(message)


def provider_messages(messages: list[Message] | None) -> list[Message] | None:
    """``messages`` as the model API accepts them: without our steer mark."""
    if not messages or not any(isinstance(m, dict) and STEER_KEY in m for m in messages):
        return messages
    return [
        {k: v for k, v in m.items() if k != STEER_KEY} if isinstance(m, dict) else m
        for m in messages
    ]


def find_safe_cut_points(messages: list[Message]) -> list[int]:
    """Indices `i` such that `messages[i:]` is a well-formed conversation.

    An index is safe when every tool call issued before it has already been
    answered, and the message at that index is not itself a tool result.
    """
    outstanding: set[str] = set()
    points: list[int] = []

    for i, message in enumerate(messages):
        role = message.get("role")
        if not outstanding and role != "tool":
            points.append(i)

        if role == "assistant" and message.get("tool_calls"):
            for call in message["tool_calls"]:
                call_id = call.get("id")
                if call_id:
                    outstanding.add(call_id)
        elif role == "tool":
            outstanding.discard(message.get("tool_call_id"))

    if not outstanding:
        points.append(len(messages))
    return points


def truncate_messages(messages: list[Message], max_messages: int) -> list[Message]:
    """Trims to roughly `max_messages`, never splitting a tool-call group.

    Leading system messages are pinned: they carry the harness rules, and
    dropping them is how a model "forgets" that a transaction is already open.
    """
    if max_messages <= 0 or len(messages) <= max_messages:
        return list(messages)

    pinned: list[Message] = []
    rest = list(messages)
    while rest and rest[0].get("role") == "system":
        pinned.append(rest.pop(0))

    budget = max(max_messages - len(pinned), 1)
    if len(rest) <= budget:
        return pinned + rest

    cut_points = find_safe_cut_points(rest)
    chosen: int | None = None
    for point in cut_points:
        if len(rest) - point <= budget:
            chosen = point
            break

    if chosen is None:
        # No boundary small enough. Keep the tightest valid tail rather than
        # emitting a malformed sequence -- correctness beats the size target.
        chosen = cut_points[-1] if cut_points else len(rest)

    return pinned + rest[chosen:]


def strip_for_persistence(messages: list[Message]) -> list[Message]:
    """Prepares a message list for storage between requests.

    Drops every `system` message: the ephemeral ones the loop injects
    mid-run (retry guidance, replayed lessons, swallowed-error notices)
    describe one execution attempt, and replaying them on the next request
    wastes tokens and tells the model to fix a call that no longer exists.
    The system prompt goes too -- the loop re-adds it each turn, so
    keeping it would duplicate it once per turn. Only the role decides: a
    user or assistant message is never dropped for its content.
    """
    kept = [m for m in messages if m.get("role") != "system"]
    # Removing messages can orphan tool results, so re-normalise.
    return _drop_orphans(kept)


def _drop_orphans(messages: list[Message]) -> list[Message]:
    """Removes tool results with no matching call and calls with no results."""
    answered: set[str] = {
        m["tool_call_id"]
        for m in messages
        if m.get("role") == "tool" and m.get("tool_call_id")
    }
    issued: set[str] = {
        call.get("id")
        for m in messages
        if m.get("role") == "assistant" and m.get("tool_calls")
        for call in m["tool_calls"]
    }

    out: list[Message] = []
    for message in messages:
        role = message.get("role")
        if role == "tool" and message.get("tool_call_id") not in issued:
            continue
        if role == "assistant" and message.get("tool_calls"):
            surviving = [c for c in message["tool_calls"] if c.get("id") in answered]
            if not surviving:
                # An assistant turn whose calls all went unanswered is only
                # useful for its text, if it had any.
                if message.get("content"):
                    out.append({"role": "assistant", "content": message["content"]})
                continue
            message = {**message, "tool_calls": surviving}
        out.append(message)
    return out


def sanitize_client_history(messages: list[Message] | None) -> list[Message]:
    """Validate untrusted client-supplied history before use.

    Keeps only well-formed user/assistant/tool messages with the fields the
    loop needs, then drops orphaned tool calls/results so a malformed
    client sequence cannot poison the next Azure call or become reusable
    executable history.
    """
    if not messages:
        return []
    cleaned: list[Message] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                entry = {"role": "user", "content": content}
                # Replayed from a stored trace: still part of its turn.
                if is_steer(message):
                    entry[STEER_KEY] = True
                cleaned.append(entry)
        elif role == "assistant":
            content = message.get("content")
            tool_calls = message.get("tool_calls")
            entry: Message = {"role": "assistant"}
            if isinstance(content, str) and content:
                entry["content"] = content
            if isinstance(tool_calls, list):
                calls = [
                    c for c in tool_calls
                    if isinstance(c, dict) and c.get("id")
                ]
                if calls:
                    entry["tool_calls"] = calls
            if "content" in entry or "tool_calls" in entry:
                cleaned.append(entry)
        elif role == "tool":
            if message.get("tool_call_id") and isinstance(message.get("content"), str):
                cleaned.append({
                    "role": "tool",
                    "tool_call_id": message["tool_call_id"],
                    "content": message["content"],
                })
        # System messages are server-owned; never accept them from clients.
    return _drop_orphans(cleaned)


def validate_message_sequence(messages: list[Message]) -> list[str]:
    """Returns human-readable problems, or an empty list if Azure will accept
    the sequence. Used in tests and in the `/health` self-check."""
    problems: list[str] = []
    outstanding: dict[str, int] = {}

    for i, message in enumerate(messages):
        role = message.get("role")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in outstanding:
                problems.append(
                    f"message {i}: tool result for unknown tool_call_id {call_id!r}"
                )
            else:
                del outstanding[call_id]
        else:
            if outstanding:
                problems.append(
                    f"message {i}: {role} message before every tool_call of "
                    "the previous assistant message was answered"
                )
            if role == "assistant" and message.get("tool_calls"):
                for call in message["tool_calls"]:
                    outstanding[call["id"]] = i

    for call_id, index in outstanding.items():
        problems.append(f"message {index}: tool_call {call_id!r} never answered")
    return problems


def extract_last_assistant_text(messages: list[Message]) -> str | None:
    """The final natural-language answer, ignoring pure tool-call turns."""
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return message["content"]
    return None
