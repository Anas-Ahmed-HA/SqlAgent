"""Chat adapters: run and stop turns.

Thin HTTP adapters only. Turn coordination lives in ``agent.host.turns``.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response, WebSocket, WebSocketDisconnect

from agent.api.auth import TokenClaims, require_token
from agent.api.origins import origin_allowed
from agent.host.event_tickets import TICKET_TTL_SECONDS, event_tickets
from agent.host.interactions import interactions
from agent.host.steers import steers
from agent.host.turn_events import turn_events
from agent.host.turns import coordinator as chat_coordinator
from agent.host.turns import run_chat
from shared.api_contract import CLIENT_STEER, CLIENT_STEER_CANCEL, EVENT_DONE, Routes
from shared.api_schemas import (
    ChatRequest,
    ChatResponse,
    ChatStopRequest,
    EventTicketRequest,
    EventTicketResponse,
)

__all__ = ["router", "chat", "chat_events", "chat_ticket", "stop_chat"]

logger = logging.getLogger("sql_agent.api")

router = APIRouter()

#: An answer is a few labels and at most a short paragraph.
_MAX_CLIENT_MESSAGE_CHARS = 8000


@router.post(Routes.CHAT, response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    response: Response,
    claims: TokenClaims = Depends(require_token),
) -> ChatResponse:
    """Run one authenticated chat turn and return its JSON response."""
    result = await run_chat(payload, request, claims)
    response.headers["X-Turn-Id"] = result.turn_id
    return result


@router.post(Routes.CHAT_TICKET, response_model=EventTicketResponse)
async def chat_ticket(
    payload: EventTicketRequest,
    claims: TokenClaims = Depends(require_token),
) -> EventTicketResponse:
    """A single-use ticket that opens the events socket of one of the
    caller's turns (``agent.host.event_tickets``)."""
    ticket = event_tickets.issue(str(claims.hr_code), payload.turn_id)
    if ticket is None:
        raise HTTPException(status_code=422, detail="Not a valid turn id.")
    return EventTicketResponse(ticket=ticket, expires_in=TICKET_TTL_SECONDS)


async def _events_owner(websocket: WebSocket, turn_id: str) -> str | None:
    """Who opens this socket: a ticket's owner, or the bearer token's.

    A ticket in the URL is the only credential looked at, so a bad one is
    refused rather than retried with the header.
    """
    ticket = websocket.query_params.get("ticket")
    if ticket is not None:
        return event_tickets.redeem(ticket, turn_id)
    try:
        claims = await require_token(websocket.headers.get("authorization"))
    except HTTPException as exc:
        logger.info("chat.events_refused", extra={"status": exc.status_code})
        return None
    return str(claims.hr_code)


@router.websocket(Routes.CHAT_EVENTS)
async def chat_events(websocket: WebSocket, turn_id: str) -> None:
    """Live status events for one of the caller's turns, until it ends.

    Opened with a ticket from ``CHAT_TICKET`` (``?ticket=...``), or with the
    same bearer token as ``/chat``. A browser page may open it only from an
    allowed origin (``agent.api.origins``). The client may connect before it
    sends the turn (it chooses the turn id) or a moment after: events
    published so far are replayed first. Each message is one JSON event; ``{"type": "done"}`` is the last, then the socket closes.

    The client answers the assistant's questions on the same socket
    (``CLIENT_ANSWER``); the answer counts only for an open question of this
    caller's turn (:meth:`Interactions.answer`). It steers the turn there too
    (``CLIENT_STEER``, :meth:`Steers.submit`), and takes a steer back
    (``CLIENT_STEER_CANCEL``, :meth:`Steers.cancel`).
    """
    # Closing before accept answers the handshake with 403.
    if not origin_allowed(websocket.headers.get("origin"), websocket.headers.get("host")):
        logger.info("chat.events_refused", extra={"reason": "origin"})
        await websocket.close(code=1008)
        return
    owner = await _events_owner(websocket, turn_id)
    if owner is None:
        await websocket.close(code=1008)
        return
    subscription = turn_events.subscribe(owner, turn_id)
    if subscription is None:
        await websocket.close(code=1008)
        return
    queue, backlog = subscription
    await websocket.accept()

    async def next_event() -> dict:
        return backlog.pop(0) if backlog else await queue.get()

    async def receive_messages() -> None:
        # Ends when the client goes away, so a socket for a turn that never
        # runs does not wait forever.
        while True:
            try:
                message = await websocket.receive()
            except (WebSocketDisconnect, RuntimeError):
                return
            if message.get("type") == "websocket.disconnect":
                return
            text = message.get("text")
            if not text or len(text) > _MAX_CLIENT_MESSAGE_CHARS:
                continue
            try:
                data = json.loads(text)
            except ValueError:
                continue
            if isinstance(data, dict) and data.get("type") == CLIENT_STEER:
                if not steers.submit(owner, turn_id, data):
                    logger.info("steers.not_taken")
            elif isinstance(data, dict) and data.get("type") == CLIENT_STEER_CANCEL:
                if not steers.cancel(owner, turn_id, data):
                    logger.info("steers.cancel_too_late")
            elif not interactions.answer(owner, turn_id, data):
                logger.info("questions.answer_ignored")

    receiver = asyncio.ensure_future(receive_messages())
    try:
        while True:
            getter = asyncio.ensure_future(next_event())
            await asyncio.wait({getter, receiver}, return_when=asyncio.FIRST_COMPLETED)
            if not getter.done():
                getter.cancel()
                return
            event = getter.result()
            await websocket.send_json(event)
            if event.get("type") == EVENT_DONE:
                await websocket.close()
                return
    except (WebSocketDisconnect, RuntimeError):
        return
    finally:
        receiver.cancel()
        turn_events.unsubscribe(owner, turn_id, queue)


@router.post(Routes.CHAT_STOP)
async def stop_chat(
    payload: ChatStopRequest,
    request: Request,
    claims: TokenClaims = Depends(require_token),
) -> dict:
    """Cancel the in-flight chat turn for this session, if any.

    A client's stop button calls this while its send request is still
    pending. Cancelling the handler task unwinds the loop; the pending
    request then answers 499. The MCP session itself is untouched,
    so the next turn starts normally.
    """
    del request
    key = claims.memory_scope(payload.session_id)
    if not await chat_coordinator.request_stop(key, payload.turn_id):
        return {"stopped": False}
    logger.info("chat.stop_requested", extra={"session_id": key})
    return {"stopped": True}
