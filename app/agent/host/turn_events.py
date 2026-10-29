"""Live status events for chat turns that are still running.

The ``ui_status`` the model writes on each tool call is stored once, in the
turn trace, after the turn ends. While it runs, the same text is published
here so a client can show it as it happens, over the
``/chat/events/{turn_id}`` WebSocket.

Events are keyed by caller (HR code) as well as by the client-chosen turn id,
so one user can never follow another's turn. Each turn keeps a short backlog:
a subscriber that connects a moment late (or after the turn ended) still gets
every event, then the closing ``done``. Finished turns expire after a TTL.
Single-process state: the agent runs as one service process.
The closing ``done`` carries the turn's result fields set with
:func:`set_turn_result` (which skills it used and dropped).

The loop does not know the caller or turn id; :meth:`TurnEvents.bind`
puts them in a context variable around the loop run, and
:func:`publish_status` reads it. Outside a bound turn (tests, scripts) it does
nothing.
"""

from __future__ import annotations

import asyncio
import contextvars
import time
from contextlib import contextmanager
from typing import Any, Iterator

from shared.api_contract import (
    EVENT_DONE, EVENT_MESSAGE, EVENT_STATUS, EVENT_TASKS, EVENT_THINKING,
)

__all__ = [
    "TurnEvents", "current_turn_key", "publish_message", "publish_status", "publish_tasks",
    "publish_thinking", "set_turn_result", "turn_events",
]

#: How long a finished turn's events stay readable.
_TTL_SECONDS = 120.0
#: A client-chosen turn id is an opaque id; anything longer is not one of ours.
_MAX_TURN_ID_CHARS = 100
#: Backlog ceiling per turn. A turn makes at most a few dozen tool calls; this
#: only bounds memory if something loops.
_MAX_BACKLOG = 500

_Key = tuple[str, str]

_current_turn: contextvars.ContextVar[_Key | None] = contextvars.ContextVar(
    "sql_agent_turn_events", default=None
)


class _Turn:
    __slots__ = ("events", "started", "done", "updated", "subscribers", "result")

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        #: Extra fields for the closing ``done`` event.
        self.result: dict[str, Any] = {}
        self.started = False
        self.done = False
        self.updated = time.monotonic()
        self.subscribers: set[asyncio.Queue] = set()


class TurnEvents:
    def __init__(self, ttl_seconds: float = _TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._turns: dict[_Key, _Turn] = {}

    @staticmethod
    def key(owner: str, turn_id: str | None) -> _Key | None:
        if not isinstance(turn_id, str):
            return None
        turn_id = turn_id.strip()
        if not 0 < len(turn_id) <= _MAX_TURN_ID_CHARS:
            return None
        return (str(owner), turn_id)

    def _prune(self, now: float) -> None:
        # A running turn is never pruned; a finished one, or one subscribed
        # to but never started, goes once nobody has touched it for the TTL.
        expired = [
            key for key, turn in self._turns.items()
            if (turn.done or not turn.started) and not turn.subscribers
            and now - turn.updated >= self._ttl
        ]
        for key in expired:
            del self._turns[key]

    def _turn(self, key: _Key) -> _Turn:
        self._prune(time.monotonic())
        turn = self._turns.get(key)
        if turn is None:
            turn = self._turns[key] = _Turn()
        return turn

    @contextmanager
    def bind(self, owner: str, turn_id: str | None) -> Iterator[None]:
        """Route :func:`publish_status` calls in this context to the turn.

        Starts a fresh backlog (a reused turn id must not replay an old
        turn) and always closes the turn with ``done`` on the way out.
        """
        key = self.key(owner, turn_id)
        if key is None:
            yield
            return
        existing = self._turns.get(key)
        subscribers = existing.subscribers if existing else set()
        turn = self._turns[key] = _Turn()
        turn.started = True
        turn.subscribers = subscribers
        token = _current_turn.set(key)
        try:
            yield
        finally:
            _current_turn.reset(token)
            self.publish(key, {**turn.result, "type": EVENT_DONE})
            turn.done = True

    def publish(self, key: _Key, event: dict[str, Any]) -> None:
        turn = self._turn(key)
        if turn.done:
            return
        if len(turn.events) < _MAX_BACKLOG:
            turn.events.append(event)
        turn.updated = time.monotonic()
        for queue in turn.subscribers:
            queue.put_nowait(event)

    def subscribe(self, owner: str, turn_id: str | None) -> tuple[asyncio.Queue, list[dict]] | None:
        """``(queue of new events, backlog so far)``, or ``None`` for a bad id.

        A turn that has not started yet is created empty, so a client may
        subscribe before it sends the turn.
        """
        key = self.key(owner, turn_id)
        if key is None:
            return None
        turn = self._turn(key)
        queue: asyncio.Queue = asyncio.Queue()
        turn.subscribers.add(queue)
        return queue, list(turn.events)

    def has_subscribers(self, key: _Key) -> bool:
        """Whether any client is following the turn right now."""
        turn = self._turns.get(key)
        return turn is not None and bool(turn.subscribers)

    def unsubscribe(self, owner: str, turn_id: str | None, queue: asyncio.Queue) -> None:
        key = self.key(owner, turn_id)
        turn = self._turns.get(key) if key else None
        if turn is not None:
            turn.subscribers.discard(queue)
            turn.updated = time.monotonic()


#: Process-wide registry shared by the chat endpoint and the events socket.
turn_events = TurnEvents()


def current_turn_key() -> _Key | None:
    """``(owner, turn_id)`` of the turn bound in this context, or None."""
    return _current_turn.get()


def set_turn_result(**fields: Any) -> None:
    """Fields the bound turn's closing ``done`` event carries (JSON values)."""
    key = _current_turn.get()
    turn = turn_events._turns.get(key) if key is not None else None
    if turn is not None:
        turn.result.update(fields)


def publish_thinking() -> None:
    """Publish that the model is planning (choosing tools or writing the answer)."""
    key = _current_turn.get()
    if key is None:
        return
    turn_events.publish(key, {"type": EVENT_THINKING, "at": time.time()})


def publish_status(text: str, *, tool: str, call_id: str) -> None:
    """Publish one tool call's user-facing status to the bound turn, if any."""
    key = _current_turn.get()
    if key is None:
        return
    turn_events.publish(key, {
        "type": EVENT_STATUS, "text": text, "tool": tool, "call_id": call_id,
        "at": time.time(),
    })


def publish_message(text: str, *, reply_to: str | None) -> None:
    """Publish the assistant's reply to a steer to the bound turn, if any."""
    key = _current_turn.get()
    if key is None:
        return
    turn_events.publish(key, {
        "type": EVENT_MESSAGE, "text": text, "reply_to": reply_to, "at": time.time(),
    })


def publish_tasks(tasks: list[dict[str, Any]]) -> None:
    """Publish the turn's task list (agent.tools.tasks) to the bound turn, if any."""
    key = _current_turn.get()
    if key is None:
        return
    turn_events.publish(key, {
        "type": EVENT_TASKS, "tasks": [dict(task) for task in tasks], "at": time.time(),
    })
