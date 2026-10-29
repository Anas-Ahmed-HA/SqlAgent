"""Questions the assistant asks the user while a turn runs.

The ``ask_user`` tool (:mod:`agent.tools.questions`) calls :meth:`Interactions.ask`
with one or more questions. That publishes an ``EVENT_QUESTION`` on the turn's
event socket (:mod:`agent.host.turn_events`) and waits. The client shows the
questions and sends ``CLIENT_ANSWER`` back on the same socket, answering them
all at once; :meth:`Interactions.answer` checks it against the open ask and
wakes the waiting tool. Whatever ends the wait (answers, a skip, the time
limit, the app going away, Stop) is published as ``EVENT_QUESTION_CLOSED``, so
every window following the turn closes the ask.

Only a turn whose client said it answers questions (``ChatRequest.interactive``,
bound here with :meth:`Interactions.bind`) is asked anything; any other turn
gets ``unavailable`` straight away. Answers are keyed by the caller (HR code)
and turn id like the events themselves, so one user can never answer
another's questions. Single-process state, like the event registry.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator
from uuid import uuid4

from agent.host.turn_events import TurnEvents, current_turn_key, turn_events
from shared.api_contract import CLIENT_ANSWER, EVENT_QUESTION, EVENT_QUESTION_CLOSED

__all__ = ["Answer", "AskOutcome", "Interactions", "interactions"]

logger = logging.getLogger("sql_agent.api")

#: The most text a client's own answer to one question may carry.
MAX_OTHER_CHARS = 1000
#: How often a waiting ask checks whether the app still follows the turn.
_POLL_SECONDS = 1.0

_interactive: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "sql_agent_interactive", default=False
)


@dataclass
class Answer:
    """The user's answer to one question: picked labels and/or their own text."""

    choices: list[str] = field(default_factory=list)
    other: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"choices": list(self.choices), "other": self.other}


@dataclass
class AskOutcome:
    """How an ask ended.

    ``reason`` is ``answered``, ``skipped``, ``timeout``, ``disconnected``,
    ``stopped`` or ``unavailable`` (the turn's client does not answer
    questions). ``answers`` holds one :class:`Answer` per question when
    answered. ``waited_seconds`` is the time spent waiting, which the caller
    gives back to the turn's deadline.
    """

    reason: str
    answers: list[Answer] = field(default_factory=list)
    waited_seconds: float = 0.0

    @property
    def answered(self) -> bool:
        return self.reason == "answered"

    def answers_payload(self) -> list[dict[str, Any]] | None:
        return [answer.as_dict() for answer in self.answers] if self.answered else None


@dataclass
class _Open:
    future: asyncio.Future
    questions: list[dict[str, Any]]


class Interactions:
    def __init__(self, events: TurnEvents | None = None) -> None:
        self._events = events or turn_events
        self._open: dict[tuple[tuple[str, str], str], _Open] = {}

    @contextmanager
    def bind(self, interactive: bool) -> Iterator[None]:
        """Whether turns run in this context may ask (their client answers)."""
        token = _interactive.set(bool(interactive))
        try:
            yield
        finally:
            _interactive.reset(token)

    def available(self) -> bool:
        """Whether the turn bound in this context can be asked anything."""
        return _interactive.get() and current_turn_key() is not None

    async def ask(
        self,
        questions: list[dict[str, Any]],
        *,
        wait_seconds: float,
        reconnect_seconds: float,
    ) -> AskOutcome:
        """Publish the questions and wait for the outcome.

        Each question is ``{"question", "options", "multi_select",
        "allow_other"}``, already checked by the caller. Raises only when the
        turn itself is cancelled (Stop), after closing the ask.
        """
        key = current_turn_key()
        if not _interactive.get() or key is None:
            return AskOutcome("unavailable")
        ask_id = uuid4().hex[:12]
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._open[(key, ask_id)] = _Open(future, questions)
        self._events.publish(key, {
            "type": EVENT_QUESTION, "id": ask_id, "questions": questions,
            "expires_at": time.time() + wait_seconds,
        })
        started = time.monotonic()
        outcome: AskOutcome | None = None
        try:
            outcome = await self._wait(key, future, started, wait_seconds, reconnect_seconds)
            return outcome
        finally:
            self._open.pop((key, ask_id), None)
            closed = outcome or AskOutcome("stopped")
            closed.waited_seconds = time.monotonic() - started
            self._events.publish(key, {
                "type": EVENT_QUESTION_CLOSED, "id": ask_id,
                "reason": closed.reason, "answers": closed.answers_payload(),
            })
            logger.info("questions.closed", extra={"reason": closed.reason, "questions": len(questions)})

    async def _wait(
        self, key: tuple[str, str], future: asyncio.Future, started: float,
        wait_seconds: float, reconnect_seconds: float,
    ) -> AskOutcome:
        absent_since: float | None = None
        while True:
            remaining = wait_seconds - (time.monotonic() - started)
            if remaining <= 0:
                return AskOutcome("timeout")
            await asyncio.wait({future}, timeout=min(_POLL_SECONDS, remaining))
            if future.done():
                return future.result()
            # The app may reconnect after a dropped socket; only a lasting
            # absence closes the ask.
            now = time.monotonic()
            if self._events.has_subscribers(key):
                absent_since = None
            elif absent_since is None:
                absent_since = now
            elif now - absent_since >= reconnect_seconds:
                return AskOutcome("disconnected")

    def answer(self, owner: str, turn_id: str | None, message: Any) -> bool:
        """Apply a client's ``CLIENT_ANSWER``; False when it answers nothing.

        Ignored (False): another message type, an unknown or already closed
        ask, or answers that do not fit the questions (see :func:`_answer_for`).
        """
        if not isinstance(message, dict) or message.get("type") != CLIENT_ANSWER:
            return False
        key = TurnEvents.key(owner, turn_id)
        ask_id = message.get("id")
        pending = self._open.get((key, ask_id)) if key and isinstance(ask_id, str) else None
        if pending is None or pending.future.done():
            return False
        if message.get("skipped") is True:
            pending.future.set_result(AskOutcome("skipped"))
            return True
        raw = message.get("answers")
        if not isinstance(raw, list) or len(raw) != len(pending.questions):
            return False
        answers = [_answer_for(question, item) for question, item in zip(pending.questions, raw)]
        if any(answer is None for answer in answers):
            return False
        pending.future.set_result(AskOutcome("answered", answers))
        return True


def _answer_for(question: dict[str, Any], item: Any) -> Answer | None:
    """The answer to ``question`` in ``item``, or None when it does not fit:
    choices that are not the question's options, more than one choice for a
    single-choice question, own text where none is allowed, or neither."""
    if not isinstance(item, dict):
        return None
    labels = [option["label"] for option in question["options"]]
    raw = item.get("choices")
    if raw is not None and not isinstance(raw, list):
        return None
    choices: list[str] = []
    for choice in raw or ():
        if not isinstance(choice, str) or choice not in labels:
            return None
        if choice not in choices:
            choices.append(choice)
    if len(choices) > 1 and not question.get("multi_select"):
        return None
    other = item.get("other")
    if other is not None:
        if not isinstance(other, str) or not question.get("allow_other"):
            return None
        other = other.strip()[:MAX_OTHER_CHARS] or None
    if not choices and other is None:
        return None
    return Answer(choices, other)


#: Process-wide registry shared by the ask_user tool and the events socket.
interactions = Interactions()
