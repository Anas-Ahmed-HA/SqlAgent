"""Single-use tickets for the ``/chat/events`` WebSocket, and its Origin check.

A browser cannot set headers on a WebSocket, so it opens the socket with a
ticket it got over HTTP. A ticket opens one socket, for one turn of one
caller, within its TTL. A web page may open the socket only from this
service's own origin or an allowed one; a client that sends no Origin (a
script) is unaffected.
"""

from __future__ import annotations

import threading

import pytest
from starlette.websockets import WebSocketDisconnect

from agent.api.origins import ALLOWED_ORIGINS_ENV, origin_allowed
from agent.host.event_tickets import EventTickets
from shared.api_contract import Routes
from tests.agent.api.test_api import auth_headers
from tests.agent.host.test_chat_stop import _client_with
from tests.agent.host.test_turn_events import StatusWorkflow


@pytest.fixture(autouse=True)
def _auth_env(monkeypatch):
    monkeypatch.setenv("JWT_HR_CODE_CLAIM", "UserHRCode")
    monkeypatch.delenv(ALLOWED_ORIGINS_ENV, raising=False)

    async def verified(claims, token, settings=None):
        return claims

    monkeypatch.setattr("agent.api.auth.verify_employee_identity", verified)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# -- the store ---------------------------------------------------------------


def test_a_ticket_opens_its_turn_once() -> None:
    tickets = EventTickets()
    ticket = tickets.issue("HR-1", "turn-1")
    assert tickets.redeem(ticket, "turn-1") == "HR-1"
    assert tickets.redeem(ticket, "turn-1") is None


def test_a_ticket_opens_only_the_turn_it_was_issued_for() -> None:
    tickets = EventTickets()
    ticket = tickets.issue("HR-1", "turn-1")
    assert tickets.redeem(ticket, "turn-2") is None
    # The wrong attempt spent it: it cannot be tried again on the right turn.
    assert tickets.redeem(ticket, "turn-1") is None


def test_a_ticket_expires() -> None:
    clock = Clock()
    tickets = EventTickets(ttl_seconds=30, clock=clock)
    late, in_time = tickets.issue("HR-1", "t"), tickets.issue("HR-1", "t")
    clock.now += 29.9
    assert tickets.redeem(in_time, "t") == "HR-1"
    clock.now += 0.1
    assert tickets.redeem(late, "t") is None


def test_bad_turn_ids_and_unknown_tickets_are_refused() -> None:
    tickets = EventTickets()
    assert tickets.issue("HR-1", "") is None
    assert tickets.issue("HR-1", "x" * 101) is None
    assert tickets.redeem("not-a-ticket", "t") is None
    assert tickets.redeem(None, "t") is None
    assert tickets.redeem("", "t") is None


def test_tickets_are_random_and_expired_ones_are_dropped() -> None:
    clock = Clock()
    tickets = EventTickets(ttl_seconds=30, clock=clock)
    first, second = tickets.issue("HR-1", "t"), tickets.issue("HR-1", "t")
    assert first != second and len(first) >= 40
    clock.now += 31
    tickets.issue("HR-1", "t")
    assert len(tickets._tickets) == 1


# -- the Origin check ----------------------------------------------------------


def test_no_origin_is_not_a_browser_and_passes() -> None:
    assert origin_allowed(None) is True


def test_without_a_pattern_no_web_page_may_open_the_socket() -> None:
    assert origin_allowed("https://chat.example.com") is False
    assert origin_allowed("https://chat.example.com", "agent.example.com") is False


def test_a_page_this_service_served_may_open_the_socket() -> None:
    """The test chat page at ``/``: same host and port, http or https (a proxy
    may end TLS in front of the service)."""
    assert origin_allowed("http://127.0.0.1:8321", "127.0.0.1:8321") is True
    assert origin_allowed("https://Agent.Example.com", "agent.example.com") is True
    assert origin_allowed("http://127.0.0.1:9999", "127.0.0.1:8321") is False
    assert origin_allowed("null", "127.0.0.1:8321") is False
    assert origin_allowed("file://127.0.0.1:8321", "127.0.0.1:8321") is False


def test_the_pattern_must_match_the_whole_origin(monkeypatch) -> None:
    monkeypatch.setenv(ALLOWED_ORIGINS_ENV, r"https://([a-z0-9-]+\.)*example\.com")
    assert origin_allowed("https://example.com") is True
    assert origin_allowed("https://chat.example.com") is True
    assert origin_allowed("https://example.com.evil.test") is False
    assert origin_allowed("https://evilexample.com") is False
    assert origin_allowed("http://chat.example.com") is False


def test_a_broken_pattern_allows_no_web_page(monkeypatch) -> None:
    monkeypatch.setenv(ALLOWED_ORIGINS_ENV, "https://(")
    assert origin_allowed("https://chat.example.com") is False


# -- the route and the socket ----------------------------------------------------


def _ticket(client, turn_id: str, hr_code: str = "HR-TEST") -> str:
    response = client.post(Routes.CHAT_TICKET, json={"turn_id": turn_id},
                           headers=auth_headers(hr_code))
    assert response.status_code == 200
    body = response.json()
    assert body["expires_in"] == 30
    return body["ticket"]


def test_the_ticket_route_needs_the_bearer_token() -> None:
    with _client_with(StatusWorkflow()) as client:
        assert client.post(Routes.CHAT_TICKET, json={"turn_id": "t"}).status_code in (401, 403)
        assert client.post(Routes.CHAT_TICKET, json={"turn_id": ""},
                           headers=auth_headers()).status_code == 422


def test_a_ticket_opens_the_socket_and_events_stream() -> None:
    workflow = StatusWorkflow()
    with _client_with(workflow) as client:
        ticket = _ticket(client, "turn-t1")
        with client.websocket_connect(f"/chat/events/turn-t1?ticket={ticket}") as ws:
            chat = threading.Thread(target=lambda: client.post(
                "/chat", json={"message": "hi", "turn_id": "turn-t1"},
                headers=auth_headers()), daemon=True)
            chat.start()
            workflow.go.set()
            received = [ws.receive_json() for _ in range(3)]
            chat.join(30)
    assert [e["type"] for e in received] == ["status", "status", "done"]


def test_a_used_ticket_does_not_open_a_second_socket() -> None:
    with _client_with(StatusWorkflow()) as client:
        ticket = _ticket(client, "turn-t2")
        with client.websocket_connect(f"/chat/events/turn-t2?ticket={ticket}") as ws:
            ws.close()
        with pytest.raises(WebSocketDisconnect) as refused:
            with client.websocket_connect(f"/chat/events/turn-t2?ticket={ticket}"):
                pass
    assert refused.value.code == 1008


def test_a_ticket_for_another_turn_is_refused() -> None:
    with _client_with(StatusWorkflow()) as client:
        ticket = _ticket(client, "turn-t3")
        with pytest.raises(WebSocketDisconnect) as refused:
            with client.websocket_connect(f"/chat/events/turn-other?ticket={ticket}"):
                pass
    assert refused.value.code == 1008


def test_a_bad_ticket_is_refused_even_with_a_valid_header() -> None:
    with _client_with(StatusWorkflow()) as client:
        with pytest.raises(WebSocketDisconnect) as refused:
            with client.websocket_connect("/chat/events/turn-t4?ticket=forged",
                                          headers=auth_headers()):
                pass
    assert refused.value.code == 1008


def test_a_ticket_follows_only_its_owners_turn() -> None:
    """Another user's ticket for the same turn id subscribes to their own
    (empty) turn, never the owner's."""
    workflow = StatusWorkflow()
    with _client_with(workflow) as client:
        ticket = _ticket(client, "turn-t5", hr_code="HR-OTHER")
        with client.websocket_connect(f"/chat/events/turn-t5?ticket={ticket}") as ws:
            workflow.go.set()
            client.post("/chat", json={"message": "hi", "turn_id": "turn-t5"},
                        headers=auth_headers())
            ws.close()
    from agent.host.turn_events import turn_events

    key = turn_events.key("HR-OTHER", "turn-t5")
    assert turn_events._turns[key].events == []


def test_a_page_from_another_origin_cannot_open_the_socket(monkeypatch) -> None:
    monkeypatch.setenv(ALLOWED_ORIGINS_ENV, r"https://chat\.example\.com")
    with _client_with(StatusWorkflow()) as client:
        ticket = _ticket(client, "turn-t6")
        with pytest.raises(WebSocketDisconnect) as refused:
            with client.websocket_connect(f"/chat/events/turn-t6?ticket={ticket}",
                                          headers={"Origin": "https://evil.test"}):
                pass
        assert refused.value.code == 1008
        # The refused handshake did not spend the ticket on the origin check;
        # an allowed page can still use it.
        with client.websocket_connect(f"/chat/events/turn-t6?ticket={ticket}",
                                      headers={"Origin": "https://chat.example.com"}) as ws:
            ws.close()


def test_the_origin_check_applies_to_the_bearer_header_too() -> None:
    with _client_with(StatusWorkflow()) as client:
        with pytest.raises(WebSocketDisconnect) as refused:
            with client.websocket_connect(
                "/chat/events/turn-t7",
                headers={**auth_headers(), "Origin": "https://evil.test"},
            ):
                pass
    assert refused.value.code == 1008


def test_the_served_chat_page_opens_the_socket_without_any_setting() -> None:
    # The test client sends Host: testserver.
    with _client_with(StatusWorkflow()) as client:
        ticket = _ticket(client, "turn-t8")
        with client.websocket_connect(f"/chat/events/turn-t8?ticket={ticket}",
                                      headers={"Origin": "http://testserver"}) as ws:
            ws.close()
