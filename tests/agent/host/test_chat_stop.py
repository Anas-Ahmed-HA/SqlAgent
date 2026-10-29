"""
Stop-button contract: POST /chat/stop cancels the in-flight turn for
the caller's memory-scoped session, and the pending request answers 499.

A stub workflow blocks until released, so the test cancels deterministically
instead of racing a real model round-trip. The MCP session itself is
untouched by a stop -- only the current turn ends.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.api.chat import router as chat_router
from agent.api.skills import router as skills_router
from agent.api.status import router as status_router
from agent.host.turns import ChatCoordinator
from agent.host.turns import coordinator as chat_coordinator
from agent.memory.store import MemoryStore
from agent.host.runtime import Runtime
from tests.agent.fakes import FakeAzure, FakeMCP
from tests.agent.api.test_api import auth_headers


@pytest.fixture(autouse=True)
def _auth_env(monkeypatch):
    monkeypatch.setenv("JWT_HR_CODE_CLAIM", "UserHRCode")

    async def verified(claims, token, settings=None):
        return claims

    monkeypatch.setattr("agent.api.auth.verify_employee_identity", verified)
    yield


class BlockingWorkflow:
    """Stays inside run() until released; records entries for assertions."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.calls = 0
        self._release: asyncio.Event | None = None

    async def run(self, **kwargs):
        self.calls += 1
        self._release = asyncio.Event()
        self.started.set()
        # Bounded so a failed cancel fails the test instead of hanging it.
        await asyncio.wait_for(self._release.wait(), timeout=60)
        return {"messages": [], "reply": "released", "raw_reply": "released"}


def _client_with(workflow) -> TestClient:
    azure, mcp = FakeAzure(), FakeMCP()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.runtime = Runtime(
            mcp=mcp,
            azure=azure,
            memory=MemoryStore(),
            agent=workflow,
            creds_source="env",
        )
        yield

    application = FastAPI(lifespan=lifespan)
    application.include_router(status_router)
    application.include_router(skills_router)
    application.include_router(chat_router)
    return TestClient(application)


def test_stop_cancels_inflight_chat() -> None:
    workflow = BlockingWorkflow()
    results: dict = {}
    with _client_with(workflow) as client:
        chat_thread = threading.Thread(
            target=lambda: results.setdefault(
                "chat",
                client.post(
                    "/chat",
                    json={"message": "hi"},
                    headers=auth_headers(),
                ),
            ),
            daemon=True,
        )
        chat_thread.start()
        assert workflow.started.wait(30), "chat turn never started"
        stopped = client.post(
            "/chat/stop", json={}, headers=auth_headers()
        ).json()
        assert stopped == {"stopped": True}
        chat_thread.join(30)
        assert not chat_thread.is_alive(), "chat request did not finish"
        assert results["chat"].status_code == 499
        assert results["chat"].json()["detail"] == "Stopped by user"
        assert workflow.calls == 1
        assert asyncio.run(chat_coordinator.current("HR-TEST|default")) is None


def test_stop_without_active_turn_reports_false() -> None:
    workflow = BlockingWorkflow()
    with _client_with(workflow) as client:
        assert client.post(
            "/chat/stop", json={}, headers=auth_headers()
        ).json() == {"stopped": False}
        assert workflow.calls == 0


def test_same_conversation_turns_are_fifo() -> None:
    """A queued turn cannot become active before the older turn releases."""

    async def scenario() -> None:
        turns = ChatCoordinator()
        entered: list[str] = []
        release_older = asyncio.Event()
        release_newer = asyncio.Event()
        newer_started = asyncio.Event()

        async def run_turn(name: str, release: asyncio.Event) -> None:
            task = await turns.track("HR-1|default", name)
            try:
                entered.append(name)
                if name == "newer":
                    newer_started.set()
                await release.wait()
            finally:
                await turns.release("HR-1|default", task)

        older = asyncio.create_task(run_turn("older", release_older))
        await asyncio.sleep(0)
        newer = asyncio.create_task(run_turn("newer", release_newer))
        await asyncio.sleep(0)
        assert entered == ["older"]
        assert await turns.current("HR-1|default") is older

        release_older.set()
        await asyncio.wait_for(newer_started.wait(), timeout=1)
        assert entered == ["older", "newer"]
        assert await turns.current("HR-1|default") is newer

        release_newer.set()
        await older
        await newer
        assert await turns.current("HR-1|default") is None

    asyncio.run(scenario())


def test_stop_can_cancel_a_selected_queued_turn() -> None:
    """A selected queued turn is cancelled without affecting the active one."""

    async def scenario() -> None:
        turns = ChatCoordinator()
        entered: list[str] = []
        release_older = asyncio.Event()

        async def make_turn(name: str) -> None:
            task = await turns.track("HR-1|default", name)
            try:
                entered.append(name)
                await release_older.wait()
            finally:
                await turns.release("HR-1|default", task)

        older_task = asyncio.create_task(make_turn("older"))
        await asyncio.sleep(0)
        newer_task = asyncio.create_task(make_turn("newer"))
        await asyncio.sleep(0)
        assert entered == ["older"]
        assert await turns.current("HR-1|default") is older_task

        assert await turns.request_stop("HR-1|default", "newer") is True
        await asyncio.sleep(0)
        assert entered == ["older"]
        assert await turns.current("HR-1|default") is older_task

        release_older.set()
        await older_task
        with pytest.raises(asyncio.CancelledError):
            await newer_task
        assert await turns.current("HR-1|default") is None

    asyncio.run(scenario())


def test_stop_after_turn_finished_reports_false() -> None:
    """A completed turn is no longer a stop target."""

    async def scenario() -> None:
        turns = ChatCoordinator()

        async def register() -> asyncio.Task | None:
            return await turns.track("HR-1|default")

        turn = await asyncio.create_task(register())
        await turns.release("HR-1|default", turn)
        assert await turns.request_stop("HR-1|default") is False

    asyncio.run(scenario())
