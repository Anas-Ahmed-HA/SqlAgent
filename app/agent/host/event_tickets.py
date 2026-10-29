"""Single-use tickets that open a turn's ``/chat/events`` WebSocket.

A browser cannot set an ``Authorization`` header on a WebSocket. So a client
asks for a ticket over HTTP, where the header works, and opens the socket
with ``?ticket=...`` instead. A ticket is random, opens one socket for one
turn of one caller, and only within a few seconds of being issued; it is
deleted when used. So the long-lived bearer token never travels on the
socket's URL or headers, and a ticket that ends up in a log is already
spent.

Each reconnect needs a new ticket. Single-process state, like the turns it
opens (:mod:`agent.host.turn_events`).
"""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Callable

from agent.host.turn_events import TurnEvents

__all__ = ["EventTickets", "TICKET_TTL_SECONDS", "event_tickets"]

#: How long a ticket can be used. The client opens the socket right after it
#: gets one, so this only has to cover a slow network.
TICKET_TTL_SECONDS = 30
#: Tickets not yet used. Each lives at most the TTL; this only bounds memory
#: if a client asks for tickets and never uses them.
_MAX_TICKETS = 10_000


class EventTickets:
    def __init__(self, ttl_seconds: float = TICKET_TTL_SECONDS,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._ttl = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        #: ticket -> (owner, turn id, expires at)
        self._tickets: dict[str, tuple[str, str, float]] = {}

    def issue(self, owner: str, turn_id: str) -> str | None:
        """A new ticket for ``owner``'s turn, or None for a bad turn id."""
        key = TurnEvents.key(owner, turn_id)
        if key is None:
            return None
        ticket = secrets.token_urlsafe(32)
        now = self._clock()
        with self._lock:
            self._prune(now)
            while len(self._tickets) >= _MAX_TICKETS:
                del self._tickets[next(iter(self._tickets))]
            self._tickets[ticket] = (key[0], key[1], now + self._ttl)
        return ticket

    def redeem(self, ticket: str | None, turn_id: str) -> str | None:
        """The owner a ticket was issued to, if it opens ``turn_id`` now.

        A ticket is spent by any attempt, so it cannot be tried twice.
        """
        if not isinstance(ticket, str) or not ticket:
            return None
        with self._lock:
            entry = self._tickets.pop(ticket, None)
        if entry is None:
            return None
        owner, issued_for, expires_at = entry
        key = TurnEvents.key(owner, turn_id)
        if key is None or key[1] != issued_for or self._clock() >= expires_at:
            return None
        return owner

    def _prune(self, now: float) -> None:
        expired = [t for t, (_, _, expires_at) in self._tickets.items() if expires_at <= now]
        for ticket in expired:
            del self._tickets[ticket]


event_tickets = EventTickets()
