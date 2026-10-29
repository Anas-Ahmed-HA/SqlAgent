"""Live turn status: the registry and the ``/chat/events`` WebSocket.

Each tool call's ``ui_status`` is published while the turn runs, to the
caller who owns the turn only, and the socket closes with ``done``.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from starlette.websockets import WebSocketDisconnect

from agent.host.turn_events import TurnEvents, publish_status, turn_events
from tests.agent.api.test_api import auth_headers
from tests.agent.host.test_chat_stop import _client_with


@pytest.fixture(autouse=True)
def _auth_env(monkeypatch):
    monkeypatch.setenv("JWT_HR_CODE_CLAIM", "UserHRCode")

    async def verified(claims, token, settings=None):
        return claims

    monkeypatch.setattr("agent.api.auth.verify_employee_identity", verified)


def _drain(queue: asyncio.Queue) -> list[dict]:
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


def test_bound_turn_publishes_to_its_owner_only() -> None:
    async def scenario():
        events = TurnEvents()
        mine, _ = events.subscribe("HR-1", "t1")
        theirs, _ = events.subscribe("HR-2", "t1")
        with events.bind("HR-1", "t1"):
            events.publish(("HR-1", "t1"), {"type": "status", "text": "Reading walls"})
        return _drain(mine), _drain(theirs)

    mine, theirs = asyncio.run(scenario())
    assert [e["type"] for e in mine] == ["status", "done"]
    assert theirs == []


def test_late_subscriber_gets_the_backlog_and_reused_ids_start_fresh() -> None:
    async def scenario():
        events = TurnEvents()
        with events.bind("HR-1", "t1"):
            events.publish(("HR-1", "t1"), {"type": "status", "text": "old"})
        _, late = events.subscribe("HR-1", "t1")
        with events.bind("HR-1", "t1"):
            _, fresh = events.subscribe("HR-1", "t1")
        return late, fresh

    late, fresh = asyncio.run(scenario())
    assert [e.get("text") for e in late] == ["old", None]
    assert fresh == []


def test_bad_turn_ids_and_unbound_publishes_are_ignored() -> None:
    assert TurnEvents().subscribe("HR-1", "") is None
    assert TurnEvents().subscribe("HR-1", "x" * 101) is None
    publish_status("Nothing listens", tool="get_walls", call_id="c1")  # no error


class StatusWorkflow:
    """Publishes two statuses the way dispatch_tool does, then answers."""

    def __init__(self) -> None:
        self.go = threading.Event()

    async def run(self, **kwargs):
        await asyncio.to_thread(self.go.wait, 30)
        publish_status("Reading the walls", tool="get_walls", call_id="c1")
        publish_status("Updating the doors", tool="set_doors", call_id="c2")
        return {"messages": [], "final_response": "done", "raw_reply": "done"}


def test_socket_streams_statuses_then_done() -> None:
    workflow = StatusWorkflow()
    with _client_with(workflow) as client:
        with client.websocket_connect("/chat/events/turn-1", headers=auth_headers()) as ws:
            chat = threading.Thread(target=lambda: client.post(
                "/chat", json={"message": "hi", "turn_id": "turn-1"},
                headers=auth_headers()), daemon=True)
            chat.start()
            workflow.go.set()
            received = [ws.receive_json() for _ in range(3)]
            chat.join(30)
    assert [(e["type"], e.get("text")) for e in received] == [
        ("status", "Reading the walls"), ("status", "Updating the doors"), ("done", None),
    ]
    assert received[0]["tool"] == "get_walls"


def test_socket_requires_the_bearer_token() -> None:
    with _client_with(StatusWorkflow()) as client:
        with pytest.raises(WebSocketDisconnect) as refused:
            with client.websocket_connect("/chat/events/turn-2"):
                pass
    assert refused.value.code == 1008


def test_a_token_offered_as_a_subprotocol_is_not_accepted() -> None:
    """Browsers open the socket with a ticket (``test_event_tickets``); the
    token itself never travels on the socket."""
    token = auth_headers()["Authorization"].split(" ", 1)[1]
    with _client_with(StatusWorkflow()) as client:
        with pytest.raises(WebSocketDisconnect) as refused:
            with client.websocket_connect("/chat/events/turn-5", subprotocols=["bearer", token]):
                pass
    assert refused.value.code == 1008


def test_another_user_cannot_follow_the_turn() -> None:
    workflow = StatusWorkflow()
    with _client_with(workflow) as client:
        with client.websocket_connect(
            "/chat/events/turn-3", headers=auth_headers("HR-OTHER")
        ) as ws:
            workflow.go.set()
            client.post("/chat", json={"message": "hi", "turn_id": "turn-3"},
                        headers=auth_headers())
            # The owner's turn finished; the other user's subscription saw
            # nothing and is still waiting. Closing ends it cleanly.
            ws.close()
    key = turn_events.key("HR-OTHER", "turn-3")
    assert key in turn_events._turns and turn_events._turns[key].events == []


def test_a_turn_refused_before_it_runs_still_ends_with_done(monkeypatch) -> None:
    """A subscriber must not wait forever on a turn that failed before the
    loop (here: the gateway refused the credentials, a 503)."""
    from agent.clients.aiapi_credentials import GatewayCredentialsError

    async def refused_credentials(token):
        raise GatewayCredentialsError("the gateway rejected the login token")

    monkeypatch.setattr("agent.host.bootstrap.load_credentials_from_gateway", refused_credentials)
    with _client_with(StatusWorkflow()) as client:
        client.app.state.runtime.agent = None
        with client.websocket_connect("/chat/events/turn-4", headers=auth_headers()) as ws:
            refused = client.post(
                "/chat", json={"message": "hi", "turn_id": "turn-4"}, headers=auth_headers(),
            )
            event = ws.receive_json()
    assert refused.status_code == 503
    assert event["type"] == "done"
