"""Steers: notes the user sends to a turn while it runs.

A steer corrects, redirects or adds to the work the assistant is doing ("make
it by company", "after you are done also count the monitors"). The client sends it as
``CLIENT_STEER`` on the turn's event socket; :meth:`Steers.submit` queues it
and publishes ``EVENT_STEER`` with status ``queued``, so every window following
the turn shows it.

The model cannot read a steer in the middle of a step: the API accepts nothing
between an assistant's tool calls and their answers. The planner takes the
queued steers at the start of its next step (:meth:`Steers.take`), once the
calls it is waiting for are answered, and they are published as ``applied``.
Until then the user may take one back (``CLIENT_STEER_CANCEL``,
:meth:`Steers.cancel`), published as ``cancelled``.
Steers still queued when the turn ends (it was answering, wrapping up or was
stopped) are published as ``missed``; the client sends them as the next turn.

Only a turn whose client follows it (``ChatRequest.interactive``, bound with
:meth:`Steers.bind`) takes steers. They are keyed by the caller (HR code) and
turn id like the events themselves, so one user can never steer another's
turn. Single-process state, like the event registry.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from agent.host.turn_events import TurnEvents, current_turn_key, turn_events
from shared.api_contract import CLIENT_STEER, CLIENT_STEER_CANCEL, EVENT_STEER, MAX_STEER_CHARS

__all__ = ["MAX_STEER_CHARS", "MAX_STEERS_PER_TURN", "Steers", "steers"]

logger = logging.getLogger("sql_agent.api")

#: Steers one turn takes; more are missed and wait for the next turn.
MAX_STEERS_PER_TURN = 10
_MAX_ID_CHARS = 64


@dataclass
class _Turn:
    pending: list[dict[str, str]] = field(default_factory=list)
    #: Every id sent, cancelled or not, so a resent steer is ignored.
    ids: set[str] = field(default_factory=set)
    #: Steers the model has read; with ``pending``, what counts to the limit.
    taken: int = 0


class Steers:
    def __init__(self, events: TurnEvents | None = None) -> None:
        self._events = events or turn_events
        self._open: dict[tuple[str, str], _Turn] = {}

    @contextmanager
    def bind(self, enabled: bool) -> Iterator[None]:
        """Take steers for the turn bound in this context while it runs.

        Enter inside :meth:`TurnEvents.bind`, so the steers still queued are
        published as missed before the turn's closing ``done``.
        """
        key = current_turn_key()
        if not enabled or key is None:
            yield
            return
        self._open[key] = _Turn()
        try:
            yield
        finally:
            turn = self._open.pop(key, None)
            for steer in turn.pending if turn else ():
                self._publish(key, steer["id"], "missed", reason="ended")
            if turn and turn.pending:
                logger.info("steers.missed", extra={"steers": len(turn.pending)})

    def submit(self, owner: str, turn_id: str | None, message: Any) -> bool:
        """Queue a client's ``CLIENT_STEER``; False when it is not taken.

        Ignored (False, nothing published): another message type, a bad id or
        text, a turn that takes no steers, or an id already sent. Missed
        (False, published): the turn has taken as many steers as it may.
        """
        if not isinstance(message, dict) or message.get("type") != CLIENT_STEER:
            return False
        steer_id, text = message.get("id"), message.get("text")
        if not isinstance(steer_id, str) or not 0 < len(steer_id) <= _MAX_ID_CHARS:
            return False
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_STEER_CHARS:
            return False
        key = TurnEvents.key(owner, turn_id)
        turn = self._open.get(key) if key else None
        if turn is None or steer_id in turn.ids:
            return False
        turn.ids.add(steer_id)
        if turn.taken + len(turn.pending) >= MAX_STEERS_PER_TURN:
            self._publish(key, steer_id, "missed", reason="limit")
            return False
        text = text.strip()
        turn.pending.append({"id": steer_id, "text": text})
        self._publish(key, steer_id, "queued", text=text)
        logger.info("steers.queued", extra={"chars": len(text)})
        return True

    def cancel(self, owner: str, turn_id: str | None, message: Any) -> bool:
        """Take back a client's queued steer (``CLIENT_STEER_CANCEL``).

        False, and nothing published, when the steer is not waiting: the
        model already read it, or there is no such steer.
        """
        if not isinstance(message, dict) or message.get("type") != CLIENT_STEER_CANCEL:
            return False
        key = TurnEvents.key(owner, turn_id)
        turn = self._open.get(key) if key else None
        steer_id = message.get("id")
        if turn is None or not any(steer["id"] == steer_id for steer in turn.pending):
            return False
        turn.pending = [steer for steer in turn.pending if steer["id"] != steer_id]
        self._publish(key, steer_id, "cancelled")
        logger.info("steers.cancelled")
        return True

    def pending(self) -> bool:
        """Whether the turn bound in this context has steers waiting."""
        turn = self._open.get(current_turn_key())
        return bool(turn and turn.pending)

    def take(self) -> list[dict[str, str]]:
        """The steers waiting for the bound turn, oldest first, now applied."""
        key = current_turn_key()
        turn = self._open.get(key)
        if not turn or not turn.pending:
            return []
        taken, turn.pending = turn.pending, []
        turn.taken += len(taken)
        for steer in taken:
            self._publish(key, steer["id"], "applied")
        return taken

    def _publish(self, key: tuple[str, str], steer_id: str, status: str, **extra: Any) -> None:
        self._events.publish(key, {"type": EVENT_STEER, "id": steer_id, "status": status, **extra})


#: Process-wide registry shared by the chat turn and the events socket.
steers = Steers()
