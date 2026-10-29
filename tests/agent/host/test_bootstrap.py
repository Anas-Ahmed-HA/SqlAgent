"""Focused tests for lazy agent-stack lifecycle."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agent.config import AzureSettings
from agent.host import bootstrap
from agent.host.bootstrap import ensure_ready, reset_ready
from agent.host.runtime import Runtime
from agent.loop import AgentLoop
from agent.memory.store import MemoryStore
from agent.clients.aiapi_credentials import GatewayModelConfig


class _Closable:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


CHAT_CFG = GatewayModelConfig(
    model_name="chat",
    calling_model_name=None,
    base_url="https://chat.example",
    api_key="chat-key",
    api_format="openai",
    embedding_dimensions=None,
)


def _fake_gateway(monkeypatch, azure) -> None:
    async def load_credentials(_token):
        return CHAT_CFG

    monkeypatch.setattr(bootstrap, "load_credentials_from_gateway", load_credentials)
    monkeypatch.setattr(
        bootstrap, "to_azure_settings",
        lambda _cfg, *, api_version: AzureSettings("https://chat.example/", "chat-key", "chat", api_version),
    )
    monkeypatch.setattr(bootstrap, "AzureChatService", lambda settings, *, on_auth_failure: azure)


def test_reset_closes_stale_stack_and_preserves_conversation_memory():
    azure = _Closable()
    memory = MemoryStore()
    app = SimpleNamespace(
        state=SimpleNamespace(
            runtime=Runtime(
                agent=object(),
                azure=azure,
                memory=memory,
                creds_stale=True,
                creds_source="aiapi-gateway",
            )
        )
    )

    asyncio.run(reset_ready(app))

    assert azure.closed
    assert app.state.runtime.agent is None
    assert app.state.runtime.memory is memory


def test_ensure_ready_builds_the_loop_on_the_first_chat(monkeypatch):
    azure = _Closable()
    _fake_gateway(monkeypatch, azure)
    mcp = SimpleNamespace(connected=True)
    runtime = Runtime(mcp=mcp)

    asyncio.run(ensure_ready(SimpleNamespace(state=SimpleNamespace(runtime=runtime)), "user-token"))

    assert isinstance(runtime.agent, AgentLoop)
    assert runtime.agent.mcp is mcp and runtime.agent.azure is azure
    assert runtime.agent.memory is runtime.memory
    assert runtime.creds_source == "aiapi-gateway"


def test_ensure_ready_closes_stale_stack_before_rebuild(monkeypatch):
    old_azure = _Closable()
    new_azure = _Closable()
    _fake_gateway(monkeypatch, new_azure)
    memory = MemoryStore()
    runtime = Runtime(
        mcp=SimpleNamespace(connected=True),
        agent=object(),
        azure=old_azure,
        memory=memory,
        creds_stale=True,
    )

    asyncio.run(ensure_ready(SimpleNamespace(state=SimpleNamespace(runtime=runtime)), "user-token"))

    assert old_azure.closed
    assert runtime.azure is new_azure
    assert isinstance(runtime.agent, AgentLoop)
    # Conversations survive a credential rotation.
    assert runtime.memory is memory
    assert runtime.creds_stale is False


def test_reset_defers_client_teardown_until_active_turn_finishes():
    azure = _Closable()
    runtime = Runtime(agent=object(), azure=azure, creds_source="aiapi-gateway")
    app = SimpleNamespace(state=SimpleNamespace(runtime=runtime))

    async def scenario() -> None:
        await runtime.begin_turn()
        await reset_ready(app)
        assert runtime.creds_stale is True
        assert runtime.agent is not None
        assert not azure.closed

        await runtime.end_turn()
        await reset_ready(app)

    asyncio.run(scenario())
    assert runtime.agent is None
    assert azure.closed
