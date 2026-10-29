"""Questions the assistant asks mid-turn: the registry and the socket.

An ask is published on the turn's event socket and answered there, by the
turn's owner only, with answers that fit the questions. Whatever ends it is
published as ``question_closed``.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from agent.host import interactions as interactions_module
from agent.host.interactions import Interactions, interactions
from agent.host.turn_events import TurnEvents
from tests.agent.api.test_api import auth_headers
from tests.agent.host.test_chat_stop import _client_with
from tests.agent.host.test_turn_events import _auth_env, _drain  # noqa: F401 - fixture

QUESTIONS = [
    {"question": "Which level?", "multi_select": False, "allow_other": False,
     "options": [{"label": "Level 1", "description": ""}, {"label": "Level 2", "description": ""}]},
    {"question": "Which types?", "multi_select": True, "allow_other": True,
     "options": [{"label": "Exterior", "description": ""}, {"label": "Interior", "description": ""}]},
]
GOOD = {"type": "answer", "answers": [{"choices": ["Level 2"], "other": None},
                                      {"choices": ["Exterior", "Interior"], "other": "and curtain walls"}]}


@pytest.fixture(autouse=True)
def _fast_polling(monkeypatch):
    monkeypatch.setattr(interactions_module, "_POLL_SECONDS", 0.01)


async def _open_ask(events: TurnEvents, registry: Interactions, *, subscribe=True, **limits):
    """Start an ask in a bound, interactive turn; return (task, queue, ask id)."""
    queue = events.subscribe("HR-1", "t1")[0] if subscribe else None
    ready = asyncio.Event()
    holder: dict = {}

    async def run():
        with events.bind("HR-1", "t1"), registry.bind(True):
            ready.set()
            return await registry.ask(QUESTIONS, wait_seconds=limits.get("wait", 5),
                                      reconnect_seconds=limits.get("reconnect", 5))

    task = asyncio.ensure_future(run())
    await ready.wait()
    await asyncio.sleep(0)
    if queue is not None:
        question = (await asyncio.wait_for(queue.get(), 1))
        assert question["type"] == "question" and question["questions"] == QUESTIONS
        holder["id"] = question["id"]
    return task, queue, holder.get("id")


def test_an_answer_that_fits_closes_the_ask() -> None:
    async def scenario():
        events = TurnEvents()
        registry = Interactions(events)
        task, queue, ask_id = await _open_ask(events, registry)
        accepted = registry.answer("HR-1", "t1", {**GOOD, "id": ask_id})
        outcome = await task
        return accepted, outcome, _drain(queue)

    accepted, outcome, events = asyncio.run(scenario())
    assert accepted and outcome.reason == "answered"
    assert [a.as_dict() for a in outcome.answers] == [
        {"choices": ["Level 2"], "other": None},
        {"choices": ["Exterior", "Interior"], "other": "and curtain walls"},
    ]
    closed = [e for e in events if e["type"] == "question_closed"]
    assert closed[0]["reason"] == "answered" and closed[0]["answers"][0]["choices"] == ["Level 2"]


@pytest.mark.parametrize("bad", [
    {"answers": [{"choices": ["Level 9"]}, {"choices": ["Exterior"]}]},           # not an option
    {"answers": [{"choices": ["Level 1", "Level 2"]}, {"choices": ["Exterior"]}]},  # two on single choice
    {"answers": [{"choices": [], "other": "Level 3"}, {"choices": ["Exterior"]}]},  # text not allowed
    {"answers": [{"choices": []}, {"choices": ["Exterior"]}]},                     # nothing
    {"answers": [{"choices": ["Level 1"]}]},                                        # one answer short
    {"answers": "Level 1"},
    {"skipped": "yes"},
])
def test_answers_that_do_not_fit_are_ignored(bad) -> None:
    async def scenario():
        events = TurnEvents()
        registry = Interactions(events)
        task, _queue, ask_id = await _open_ask(events, registry)
        refused = registry.answer("HR-1", "t1", {"type": "answer", "id": ask_id, **bad})
        still_open = not task.done()
        registry.answer("HR-1", "t1", {**GOOD, "id": ask_id})
        await task
        return refused, still_open

    refused, still_open = asyncio.run(scenario())
    assert refused is False and still_open


def test_only_the_owner_of_the_turn_can_answer() -> None:
    async def scenario():
        events = TurnEvents()
        registry = Interactions(events)
        task, _queue, ask_id = await _open_ask(events, registry)
        other_user = registry.answer("HR-2", "t1", {**GOOD, "id": ask_id})
        other_turn = registry.answer("HR-1", "t2", {**GOOD, "id": ask_id})
        unknown_ask = registry.answer("HR-1", "t1", {**GOOD, "id": "nope"})
        registry.answer("HR-1", "t1", {"type": "answer", "id": ask_id, "skipped": True})
        outcome = await task
        late = registry.answer("HR-1", "t1", {**GOOD, "id": ask_id})
        return other_user, other_turn, unknown_ask, outcome.reason, late

    assert asyncio.run(scenario()) == (False, False, False, "skipped", False)


def test_an_unanswered_ask_times_out() -> None:
    async def scenario():
        events = TurnEvents()
        registry = Interactions(events)
        task, queue, _ = await _open_ask(events, registry, wait=0.05)
        outcome = await task
        return outcome, _drain(queue)

    outcome, events = asyncio.run(scenario())
    assert outcome.reason == "timeout" and outcome.waited_seconds >= 0.05
    closed = [e for e in events if e["type"] == "question_closed"]
    assert closed == [{"type": "question_closed", "id": closed[0]["id"], "reason": "timeout", "answers": None}]


def test_an_ask_closes_when_the_app_stops_following_the_turn() -> None:
    async def scenario():
        events = TurnEvents()
        registry = Interactions(events)
        task, _, _ = await _open_ask(events, registry, subscribe=False, reconnect=0.03)
        return await task

    assert asyncio.run(scenario()).reason == "disconnected"


def test_stopping_the_turn_closes_the_ask() -> None:
    async def scenario():
        events = TurnEvents()
        registry = Interactions(events)
        task, queue, _ = await _open_ask(events, registry)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return _drain(queue)

    events = asyncio.run(scenario())
    assert [e for e in events if e["type"] == "question_closed"][0]["reason"] == "stopped"


def test_a_turn_without_an_answering_client_is_never_asked() -> None:
    async def scenario():
        events = TurnEvents()
        registry = Interactions(events)
        unbound = await registry.ask(QUESTIONS, wait_seconds=5, reconnect_seconds=5)
        with events.bind("HR-1", "t1"), registry.bind(False):
            not_interactive = await registry.ask(QUESTIONS, wait_seconds=5, reconnect_seconds=5)
        return unbound.reason, not_interactive.reason

    assert asyncio.run(scenario()) == ("unavailable", "unavailable")


class AskingWorkflow:
    """Asks the user through the real registry the way ask_user does, then
    answers with what they chose."""

    def __init__(self) -> None:
        self.interactive = None

    async def run(self, **kwargs):
        self.interactive = kwargs.get("interactive")
        outcome = await interactions.ask(QUESTIONS[:1], wait_seconds=10, reconnect_seconds=5)
        reply = outcome.answers[0].choices[0] if outcome.answered else outcome.reason
        return {"messages": [], "final_response": reply, "raw_reply": reply}


def test_a_question_is_answered_on_the_turn_socket() -> None:
    workflow = AskingWorkflow()
    result: dict = {}
    with _client_with(workflow) as client:
        with client.websocket_connect("/chat/events/turn-q", headers=auth_headers()) as ws:
            chat = threading.Thread(target=lambda: result.update(response=client.post(
                "/chat", json={"message": "add walls", "turn_id": "turn-q", "interactive": True},
                headers=auth_headers())), daemon=True)
            chat.start()
            question = ws.receive_json()
            ws.send_json({"type": "answer", "id": question["id"],
                          "answers": [{"choices": ["Level 2"], "other": None}]})
            closed = ws.receive_json()
            done = ws.receive_json()
            chat.join(30)
    assert question["type"] == "question" and question["questions"][0]["question"] == "Which level?"
    assert closed["type"] == "question_closed" and closed["reason"] == "answered"
    assert done["type"] == "done"
    assert workflow.interactive is True
    assert result["response"].json()["reply"] == "Level 2"


def test_a_turn_sent_without_interactive_is_not_asked() -> None:
    workflow = AskingWorkflow()
    with _client_with(workflow) as client:
        response = client.post("/chat", json={"message": "add walls", "turn_id": "turn-n"},
                               headers=auth_headers())
    assert workflow.interactive is False
    assert response.json()["reply"] == "unavailable"
