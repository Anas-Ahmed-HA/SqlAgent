"""Chat-turn coordination for the agent host.

Owns the in-flight chat turn registry (one tracked asyncio task per
HR-scoped session) and the shared chat execution adapter used by the
`/chat` endpoint. HTTP routes stay thin:
they authenticate, then delegate here.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from uuid import uuid4

from fastapi import HTTPException, Request

from agent.api.auth import TokenClaims, auth_log_fields
from agent.config import AgentConfig
from shared.api_schemas import ChatRequest, ChatResponse, SkippedSkill
from agent.errors import AgentException, CredentialsStaleException
from agent.host.bootstrap import NotReadyError, ensure_ready, reset_ready
from agent.host.interactions import interactions
from agent.host.steers import steers
from agent.host.turn_events import set_turn_result, turn_events
from agent.clients.aiapi_credentials import GatewayCredentialsError
from agent.observability.turn_trace import build_turn_trace
from skills_hub.budget import fit_to_budget
from skills_hub.safety import skill_text_is_safe

__all__ = [
    "ChatCoordinator",
    "client_session_id",
    "coordinator",
    "run_chat",
]

logger = logging.getLogger("sql_agent.api")


@dataclass
class _QueuedTurn:
    key: str
    turn_id: str
    task: asyncio.Task


class ChatCoordinator:
    """Tracks FIFO turns for each HR-scoped conversation.

    A queued turn can be cancelled before it starts. An active turn can be
    cancelled without affecting another conversation, and identity-safe
    cleanup prevents an older request from removing a newer queue entry.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task] = {}
        self._queues: dict[str, list[_QueuedTurn]] = {}
        self._turns: dict[asyncio.Task, _QueuedTurn] = {}
        self._condition = asyncio.Condition()

    async def track(self, key: str, turn_id: str | None = None) -> asyncio.Task | None:
        """Queue the current task and wait until it owns the conversation."""
        task = asyncio.current_task()
        if task is None:
            return None
        turn = _QueuedTurn(key, turn_id or uuid4().hex, task)
        async with self._condition:
            queue = self._queues.setdefault(key, [])
            queue.append(turn)
            self._turns[task] = turn
            try:
                while queue[0] is not turn:
                    await self._condition.wait()
                self._tasks[key] = task
            except asyncio.CancelledError:
                if turn in queue:
                    queue.remove(turn)
                self._turns.pop(task, None)
                if not queue:
                    self._queues.pop(key, None)
                self._condition.notify_all()
                raise
        return task

    async def release(self, key: str, task: asyncio.Task | None = None) -> None:
        """Remove `key` only when it still points at `task`.

        Defaults to the current task, which is what `run_chat` passes
        implicitly from its `finally` block.
        """
        if task is None:
            task = asyncio.current_task()
        async with self._condition:
            turn = self._turns.pop(task, None) if task is not None else None
            queue = self._queues.get(key, [])
            if turn in queue:
                queue.remove(turn)
            if self._tasks.get(key) is task:
                self._tasks.pop(key, None)
            if not queue:
                self._queues.pop(key, None)
            self._condition.notify_all()

    async def current(self, key: str) -> asyncio.Task | None:
        """Return the tracked turn for `key` without removing it."""
        async with self._condition:
            return self._tasks.get(key)

    async def request_stop(self, key: str, turn_id: str | None = None) -> bool:
        """Cancel the active or selected queued turn for `key`."""
        async with self._condition:
            queue = self._queues.get(key, [])
            turn = next(
                (item for item in queue if turn_id is None or item.turn_id == turn_id),
                None,
            )
            if turn is None or turn.task.done():
                return False
            turn.task.cancel()
            return True


#: Shared turn registry used by the HTTP chat endpoints.
coordinator = ChatCoordinator()


def client_session_id(value: str | None) -> str:
    """Client-visible session id, usable as-is with POST /chat/stop.

    Mirrors the suffix normalization in `TokenClaims.memory_scope` (strip,
    `|` -> `_`, bounded) without the `hr_code` prefix: the stop endpoint
    applies `memory_scope` itself, so returning the scoped `hr|suffix` key
    here would double-scope on the next call and miss the in-flight turn.
    """
    from shared.identity import normalize_session_suffix

    return normalize_session_suffix(value)


def _agent(request: Request):
    runtime = request.app.state.runtime
    if runtime.agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized yet")
    return runtime.agent


def _turn_skills(payload: ChatRequest) -> tuple[list[dict], list[SkippedSkill]]:
    """The request's skills this turn may use, and those it may not.

    The client sends the user's equipped skills with every turn. Their text
    is untrusted (a shared skill is written by someone else), so each one is
    checked again here: unsafe skills are dropped, then the rest are kept in
    order until the equipped limits are reached.
    """
    config = AgentConfig()
    safe, dropped = [], []
    for skill in payload.skills or []:
        if skill_text_is_safe(skill.name, skill.description, skill.instructions):
            safe.append(skill)
        else:
            dropped.append(SkippedSkill(id=skill.id, reason="unsafe"))
    if dropped:
        logger.warning("skills.unsafe_dropped",
                       extra={"skill_ids": [str(item.id) for item in dropped]})
    kept, over = fit_to_budget(
        safe, config.max_equipped_skills, config.max_equipped_skill_words)
    if over:
        logger.warning("skills.over_budget",
                       extra={"skill_ids": [str(skill.id) for skill in over]})
        dropped.extend(SkippedSkill(id=skill.id, reason="over_budget") for skill in over)
    return [skill.model_dump(mode="json") for skill in kept], dropped


async def run_chat(
    payload: ChatRequest,
    request: Request,
    claims: TokenClaims,
    turns: ChatCoordinator | None = None,
) -> ChatResponse:
    """Body of the chat endpoint: queue the turn, build the stack, run the loop."""
    active = turns if turns is not None else coordinator
    bearer_token = None
    authorization = request.headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        bearer_token = authorization[7:].strip()

    runtime = request.app.state.runtime
    lazy_start_required = runtime.agent is None or runtime.creds_stale
    session_id = claims.memory_scope(payload.session_id)
    hr_code = claims.hr_code
    turn_id = payload.turn_id or uuid4().hex
    history = (
        [m.model_dump(exclude_none=True) for m in payload.history]
        if payload.history
        else None
    )
    # Bound around the whole turn, not only the loop: a turn that fails
    # before it (stopped while queued, not ready) still
    # ends with `done` for a client that subscribed first.
    skills, skills_dropped = _turn_skills(payload)
    skills_used = [skill["id"] for skill in skills]
    with (
        turn_events.bind(hr_code, turn_id),
        interactions.bind(payload.interactive),
        steers.bind(payload.interactive),
    ):
        set_turn_result(
            skills_used=skills_used,
            skills_dropped=[item.model_dump(mode="json") for item in skills_dropped],
        )
        turn = None
        leased = False

        try:
            # Track first so turns for one conversation are FIFO. `track` returns
            # only when this request owns the conversation and may use the stack.
            turn = await active.track(session_id, turn_id)
            await ensure_ready(request.app, bearer_token)
            agent = _agent(request)
            await runtime.begin_turn()
            leased = True

            if lazy_start_required:
                logger.info(
                    "chat.authenticated_after_lazy_start",
                    extra={
                        **auth_log_fields(bearer_token, claims),
                        "hr_code": claims.hr_code,
                        "email": claims.email or "-",
                    },
                )

            final = await agent.run(
                user_message=payload.message,
                session_id=session_id,
                history=history,
                hr_code=hr_code,
                equipped_skills=skills,
                interactive=payload.interactive,
            )
        except NotReadyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except GatewayCredentialsError as exc:
            runtime.creds_error = str(exc)
            logger.error("chat.creds_unavailable", exc_info=True)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None and hasattr(task, "uncancel"):
                task.uncancel()
            logger.info("chat.stopped", extra={"session_id": session_id, "turn_id": turn_id})
            raise HTTPException(status_code=499, detail="Stopped by user") from None
        except CredentialsStaleException as exc:
            logger.error("chat.credentials_stale", exc_info=True)
            await reset_ready(request.app)
            raise HTTPException(
                status_code=503,
                detail=(
                    "Azure rejected the cached credentials -- please retry; your "
                    "login token will refresh the credentials."
                ),
            ) from exc
        except AgentException as exc:
            logger.error("chat.dependency_failure", exc_info=True)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("chat.failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if turn is not None:
                await active.release(session_id, turn)
            if leased:
                await runtime.end_turn()

    if lazy_start_required:
        logger.info(
            "chat.first_authenticated_complete",
            extra={
                **auth_log_fields(bearer_token, claims),
                "hr_code": claims.hr_code,
                "email": claims.email or "-",
                "session_id": session_id,
                "turn_id": turn_id,
            },
        )

    return ChatResponse(
        messages=final.get("messages", []),
        reply=final.get("final_response"),
        raw_reply=final.get("raw_reply"),
        session_id=client_session_id(payload.session_id),
        turn_id=turn_id,
        context=final.get("context_usage"),
        trace=build_turn_trace(
            final.get("messages", []), final.get("tool_results"), turn_id,
            final.get("planning_ms"), steers=final.get("steers"), tasks=final.get("tasks"),
        ),
        skills_used=skills_used,
        skills_dropped=skills_dropped,
    )
