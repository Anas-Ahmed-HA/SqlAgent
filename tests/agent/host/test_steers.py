"""Steers: notes the user sends to a running turn, the registry and the socket.

A steer is taken only for a running turn of its sender that follows the
socket, is published as ``queued`` (with its text), ``applied`` when the
planner takes it, and ``missed`` when the turn ends first or has taken as
many as it may.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from agent.host.steers import MAX_STEER_CHARS, MAX_STEERS_PER_TURN, Steers, steers
from agent.host.turn_events import TurnEvents
from tests.agent.api.test_api import auth_headers
from tests.agent.host.test_chat_stop import _client_with
from tests.agent.host.test_turn_events import _auth_env, _drain  # noqa: F401 - fixture


def steer(steer_id: str = "s1", text: str = "make them 3m") -> dict:
    return {"type": "steer", "id": steer_id, "text": text}


def test_a_steer_is_queued_then_taken_and_applied() -> None:
    async def scenario():
        events = TurnEvents()
        registry = Steers(events)
        queue, _ = events.subscribe("HR-1", "t1")
        with events.bind("HR-1", "t1"), registry.bind(True):
            assert registry.submit("HR-1", "t1", steer(text="  make them 3m  ")) is True
            assert registry.pending() is True
            taken = registry.take()
            again = registry.take()
            pending_after = registry.pending()
        return taken, again, pending_after, _drain(queue)

    taken, again, pending_after, published = asyncio.run(scenario())
    assert taken == [{"id": "s1", "text": "make them 3m"}]
    assert again == [] and pending_after is False
    assert published[0] == {"type": "steer", "id": "s1", "status": "queued", "text": "make them 3m"}
    assert published[1] == {"type": "steer", "id": "s1", "status": "applied"}
    assert published[-1]["type"] == "done"


@pytest.mark.parametrize("bad", [
    {"type": "answer", "id": "s1", "text": "x"},
    {"type": "steer", "id": "", "text": "x"},
    {"type": "steer", "id": "x" * 65, "text": "x"},
    {"type": "steer", "id": 7, "text": "x"},
    {"type": "steer", "id": "s1", "text": "   "},
    {"type": "steer", "id": "s1", "text": "x" * (MAX_STEER_CHARS + 1)},
    {"type": "steer", "id": "s1"},
    "steer",
])
def test_malformed_steers_are_ignored(bad) -> None:
    async def scenario():
        events = TurnEvents()
        registry = Steers(events)
        with events.bind("HR-1", "t1"), registry.bind(True):
            return registry.submit("HR-1", "t1", bad), registry.take()

    assert asyncio.run(scenario()) == (False, [])


def test_only_the_owner_of_a_running_interactive_turn_can_steer_it() -> None:
    async def scenario():
        events = TurnEvents()
        registry = Steers(events)
        before = registry.submit("HR-1", "t1", steer())
        with events.bind("HR-1", "t1"), registry.bind(True):
            other_owner = registry.submit("HR-2", "t1", steer("s2"))
            other_turn = registry.submit("HR-1", "t2", steer("s3"))
            mine = registry.submit("HR-1", "t1", steer("s4"))
            repeated = registry.submit("HR-1", "t1", steer("s4", "again"))
            taken = registry.take()
        after = registry.submit("HR-1", "t1", steer("s5"))
        with events.bind("HR-1", "t3"), registry.bind(False):
            not_interactive = registry.submit("HR-1", "t3", steer("s6"))
        return before, other_owner, other_turn, mine, repeated, taken, after, not_interactive

    before, other_owner, other_turn, mine, repeated, taken, after, not_interactive = asyncio.run(scenario())
    assert (before, other_owner, other_turn, after, not_interactive) == (False,) * 5
    assert mine is True and repeated is False
    assert taken == [{"id": "s4", "text": "make them 3m"}]


def test_a_turn_takes_at_most_so_many_steers() -> None:
    async def scenario():
        events = TurnEvents()
        registry = Steers(events)
        queue, _ = events.subscribe("HR-1", "t1")
        with events.bind("HR-1", "t1"), registry.bind(True):
            results = [registry.submit("HR-1", "t1", steer(f"s{n}", f"note {n}"))
                       for n in range(MAX_STEERS_PER_TURN + 1)]
            taken = registry.take()
        return results, taken, _drain(queue)

    results, taken, published = asyncio.run(scenario())
    assert results == [True] * MAX_STEERS_PER_TURN + [False]
    assert len(taken) == MAX_STEERS_PER_TURN
    last = f"s{MAX_STEERS_PER_TURN}"
    assert {"type": "steer", "id": last, "status": "missed", "reason": "limit"} in published


def test_steers_still_queued_when_the_turn_ends_are_missed_before_done() -> None:
    async def scenario():
        events = TurnEvents()
        registry = Steers(events)
        queue, _ = events.subscribe("HR-1", "t1")
        with events.bind("HR-1", "t1"), registry.bind(True):
            registry.submit("HR-1", "t1", steer())
        return _drain(queue)

    published = asyncio.run(scenario())
    assert [e["type"] for e in published] == ["steer", "steer", "done"]
    assert published[1] == {"type": "steer", "id": "s1", "status": "missed", "reason": "ended"}


class SteeredWorkflow:
    """Waits for a steer through the real registry, the way the planner takes
    it, then answers with its text."""

    def __init__(self) -> None:
        self.running = threading.Event()

    async def run(self, **kwargs):
        self.running.set()
        for _ in range(500):
            if steers.pending():
                break
            await asyncio.sleep(0.01)
        taken = steers.take()
        reply = taken[0]["text"] if taken else "no steer"
        return {"messages": [], "final_response": reply, "raw_reply": reply}


def test_a_steer_is_sent_on_the_turn_socket() -> None:
    workflow = SteeredWorkflow()
    result: dict = {}
    with _client_with(workflow) as client:
        with client.websocket_connect("/chat/events/turn-s", headers=auth_headers()) as ws:
            chat = threading.Thread(target=lambda: result.update(response=client.post(
                "/chat", json={"message": "add walls", "turn_id": "turn-s", "interactive": True},
                headers=auth_headers())), daemon=True)
            chat.start()
            # A turn takes steers only while it runs.
            assert workflow.running.wait(30)
            ws.send_json(steer("s1", "make them 3m"))
            queued = ws.receive_json()
            applied = ws.receive_json()
            done = ws.receive_json()
            chat.join(30)
    assert queued == {"type": "steer", "id": "s1", "status": "queued", "text": "make them 3m"}
    assert applied == {"type": "steer", "id": "s1", "status": "applied"}
    assert done["type"] == "done"
    assert result["response"].json()["reply"] == "make them 3m"


def test_a_queued_steer_can_be_cancelled_until_it_is_read() -> None:
    async def scenario():
        events = TurnEvents()
        registry = Steers(events)
        queue, _ = events.subscribe("HR-1", "t1")
        with events.bind("HR-1", "t1"), registry.bind(True):
            registry.submit("HR-1", "t1", steer("s1", "only L1"))
            registry.submit("HR-1", "t1", steer("s2", "and tag them"))
            other_user = registry.cancel("HR-2", "t1", {"type": "steer_cancel", "id": "s1"})
            cancelled = registry.cancel("HR-1", "t1", {"type": "steer_cancel", "id": "s1"})
            again = registry.cancel("HR-1", "t1", {"type": "steer_cancel", "id": "s1"})
            wrong_type = registry.cancel("HR-1", "t1", {"type": "steer", "id": "s2"})
            taken = registry.take()
            too_late = registry.cancel("HR-1", "t1", {"type": "steer_cancel", "id": "s2"})
        return other_user, cancelled, again, wrong_type, taken, too_late, _drain(queue)

    other_user, cancelled, again, wrong_type, taken, too_late, published = asyncio.run(scenario())
    assert (other_user, cancelled, again, wrong_type, too_late) == (False, True, False, False, False)
    assert taken == [{"id": "s2", "text": "and tag them"}]
    statuses = [(e["id"], e["status"]) for e in published if e["type"] == "steer"]
    assert statuses == [("s1", "queued"), ("s2", "queued"), ("s1", "cancelled"), ("s2", "applied")]


def test_a_cancelled_steer_does_not_count_to_the_limit() -> None:
    async def scenario():
        events = TurnEvents()
        registry = Steers(events)
        with events.bind("HR-1", "t1"), registry.bind(True):
            for n in range(MAX_STEERS_PER_TURN):
                registry.submit("HR-1", "t1", steer(f"c{n}", "typo"))
                registry.cancel("HR-1", "t1", {"type": "steer_cancel", "id": f"c{n}"})
            after_cancels = registry.submit("HR-1", "t1", steer("s1", "make them 3m"))
            resent = registry.submit("HR-1", "t1", steer("c0", "typo"))
            taken = registry.take()
            more = [registry.submit("HR-1", "t1", steer(f"s{n}", f"note {n}"))
                    for n in range(2, MAX_STEERS_PER_TURN + 2)]
        return after_cancels, resent, taken, more

    after_cancels, resent, taken, more = asyncio.run(scenario())
    assert (after_cancels, resent) == (True, False)
    assert taken == [{"id": "s1", "text": "make them 3m"}]
    # Steers the model already read still count: one taken plus nine more.
    assert more == [True] * (MAX_STEERS_PER_TURN - 1) + [False]
