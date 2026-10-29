"""The HTTP contract between a client and the agent.

Route paths, the events streamed while a turn runs, the capabilities the
agent reports, and how a client recognises a compatible backend. Request and
response bodies are in :mod:`shared.api_schemas`.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CAPABILITIES",
    "CAPABILITY_EVENT_TICKETS",
    "CAPABILITY_QUESTIONS",
    "CLIENT_ANSWER",
    "CAPABILITY_SKILLS",
    "CAPABILITY_STEERS",
    "CLIENT_STEER",
    "CLIENT_STEER_CANCEL",
    "CONTRACT_VERSION",
    "EVENT_DONE",
    "EVENT_MESSAGE",
    "EVENT_QUESTION",
    "EVENT_QUESTION_CLOSED",
    "EVENT_STATUS",
    "EVENT_STEER",
    "EVENT_TASKS",
    "EVENT_THINKING",
    "HEALTH_STATUSES",
    "MAX_STEER_CHARS",
    "Routes",
    "health_is_compatible",
]

#: Raised when a route or a body changes incompatibly. ``/health`` reports
#: it, and a client only talks to a backend reporting the same version.
CONTRACT_VERSION = 1


class Routes:
    """Every path a client calls. Paths with ``{...}`` are templates."""

    HEALTH = "/health"
    TOOLS = "/tools"

    CHAT = "/chat"
    CHAT_EVENTS = "/chat/events/{turn_id}"
    #: A single-use ticket that opens :attr:`CHAT_EVENTS` for one turn, for a
    #: client that cannot set headers on a WebSocket (a browser). Body
    #: ``{"turn_id"}``; answer ``{"ticket", "expires_in"}`` (seconds). The
    #: socket is then opened with ``?ticket=...`` and no ``Authorization``.
    CHAT_TICKET = "/chat/ticket"
    CHAT_STOP = "/chat/stop"

    #: The AI rewrite of a user's wording into a skill. Skills themselves
    #: live in the cloud skills API, which the client calls directly; a turn
    #: brings its equipped skills in the chat request's ``skills``.
    SKILL_PROPOSE = "/skills/propose"


# -- Events on CHAT_EVENTS ----------------------------------------------------
# One JSON object per message, ``{"type": ...}``. EVENT_DONE is always last,
# then the socket closes.

#: The model is planning its next step. ``{"type", "at"}`` (epoch seconds).
EVENT_THINKING = "thinking"
#: A step started. ``{"type", "text", "tool", "call_id"}``; ``text`` is the
#: user-facing description of the step.
EVENT_STATUS = "status"
#: The turn ended. ``{"type", "skills_used", "skills_dropped"}``, the last
#: two as in ``ChatResponse``.
EVENT_DONE = "done"
#: The assistant asks the user one or more questions and waits (only in a turn
#: sent with ``interactive``). ``{"type", "id", "questions", "expires_at"}``:
#: each question is ``{"question", "options", "multi_select", "allow_other"}``
#: with ``options`` a list of ``{"label", "description", "recommended"}``.
#: ``expires_at`` is epoch
#: seconds. Answer them all at once with :data:`CLIENT_ANSWER` on the
#: same socket.
EVENT_QUESTION = "question"
#: An ask is closed. ``{"type", "id", "reason", "answers"}``: ``reason`` is
#: ``answered``, ``skipped``, ``timeout``, ``disconnected`` (the app stopped
#: following the turn) or ``stopped`` (the turn was stopped); ``answers`` is
#: one ``{"choices", "other"}`` per question when answered, else null. Sent to
#: every window following the turn, so an ask answered in one closes in the
#: others.
EVENT_QUESTION_CLOSED = "question_closed"
#: A steer the user sent (:data:`CLIENT_STEER`) changed state.
#: ``{"type", "id", "status"}``: ``status`` is ``queued`` (accepted; this
#: event also carries ``text``, so every window following the turn shows it),
#: ``applied`` (the model has it), ``cancelled`` (the user took it back
#: with :data:`CLIENT_STEER_CANCEL` before the model read it), or ``missed``
#: (the turn cannot take it: it was ending, stopping or had taken as many
#: steers as it may; ``reason`` says which). A steer still not ``applied`` when :data:`EVENT_DONE` arrives
#: was missed too. Send a missed steer as the next turn's message.
EVENT_STEER = "steer"
#: The assistant's reply to a steer, written while it carries on with the
#: work. ``{"type", "text", "reply_to", "at"}``: ``reply_to`` is the id of the
#: last steer it answers. The assistant writes no other text mid-turn.
EVENT_MESSAGE = "message"
#: The turn's task list changed. ``{"type", "tasks", "at"}``: each task is
#: ``{"id", "text", "source", "status", "note"}``, and ``title`` (a few words
#: naming it) once the model gives it one; ``source`` ``request`` or
#: ``steer``, ``status`` ``open``, ``in_progress``, ``done`` or ``dropped``.
#: A turn has a task list only once the user steers it. Each span of the
#: stored trace then names its item in ``task``.
EVENT_TASKS = "tasks"

# -- Messages a client sends on CHAT_EVENTS -------------------------------------

#: The user's answers to an open ask: ``{"type", "id", "answers"}`` with one
#: ``{"choices", "other"}`` per question, in order (``choices`` the picked
#: option labels, ``other`` free text or null); or ``{"type", "id",
#: "skipped": true}``. Anything else is ignored.
CLIENT_ANSWER = "answer"
#: A note from the user to the turn while it runs, correcting, redirecting or
#: adding to the work: ``{"type", "id", "text"}``, ``id`` chosen by the client
#: (at most 64 characters), ``text`` at most :data:`MAX_STEER_CHARS`. Only a
#: turn sent with ``interactive`` takes steers. The model reads it once the
#: tool calls it is waiting for are answered; see :data:`EVENT_STEER` for what
#: follows.
CLIENT_STEER = "steer"
#: Takes back a steer the model has not read yet: ``{"type", "id"}``. The
#: turn confirms with :data:`EVENT_STEER` ``cancelled``; a steer already
#: ``applied`` stays applied, and nothing is sent.
CLIENT_STEER_CANCEL = "steer_cancel"
#: The longest steer: a correction or an added task, not a document.
MAX_STEER_CHARS = 2000


# -- Capabilities in /health ----------------------------------------------------

#: Accepts ``skills`` in the chat request and serves :attr:`Routes.SKILL_PROPOSE`.
CAPABILITY_SKILLS = "skills"
#: Questions mid-turn: :data:`EVENT_QUESTION` and :data:`CLIENT_ANSWER` on
#: ``CHAT_EVENTS`` for a turn sent with ``interactive``.
CAPABILITY_QUESTIONS = "questions"
#: Steers mid-turn: :data:`CLIENT_STEER`, :data:`CLIENT_STEER_CANCEL`, :data:`EVENT_STEER`,
#: :data:`EVENT_MESSAGE` and :data:`EVENT_TASKS` on ``CHAT_EVENTS`` for a turn
#: sent with ``interactive``.
CAPABILITY_STEERS = "steers"
#: :attr:`Routes.CHAT_TICKET` is served and ``CHAT_EVENTS`` accepts its
#: tickets. A client uses a ticket when this is listed, and the
#: ``Authorization`` header otherwise.
CAPABILITY_EVENT_TICKETS = "event_tickets"
CAPABILITIES = frozenset({
    CAPABILITY_SKILLS, CAPABILITY_QUESTIONS, CAPABILITY_STEERS, CAPABILITY_EVENT_TICKETS,
})

#: ``/health`` ``status`` values. ``starting`` and ``degraded`` still serve
#: chat requests; they explain why an answer may fail.
HEALTH_STATUSES = frozenset({"ok", "degraded", "starting"})


def health_is_compatible(body: Any) -> bool:
    """Whether a ``/health`` body comes from a backend this client can use."""
    return (
        isinstance(body, dict)
        and body.get("contract") == CONTRACT_VERSION
        and body.get("status") in HEALTH_STATUSES
        and isinstance(body.get("capabilities"), list)
    )
