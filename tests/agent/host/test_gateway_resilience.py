"""Focused tests for gateway failure coalescing and parallel model fetches."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent.host import bootstrap
from agent.host.runtime import Runtime
from agent.clients import aiapi_credentials
from agent.clients.aiapi_credentials import (
    GatewayCredentialsError,
    GatewayModelConfig,
)


def _app_for_bootstrap() -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(
            runtime=Runtime(
                mcp=SimpleNamespace(connected=True),
            )
        )
    )


def test_concurrent_bootstrap_failures_share_one_gateway_attempt(monkeypatch):
    app = _app_for_bootstrap()
    calls = 0

    async def failing_loader(_bearer_token):
        nonlocal calls
        calls += 1
        # Leave the lock briefly so all callers queue behind the leader.
        await asyncio.sleep(0.01)
        raise GatewayCredentialsError("gateway unavailable")

    monkeypatch.setattr(bootstrap, "load_credentials_from_gateway", failing_loader)

    async def run():
        return await asyncio.gather(
            *(bootstrap.ensure_ready(app, "user-token") for _ in range(5)),
            return_exceptions=True,
        )

    results = asyncio.run(run())

    assert calls == 1
    assert all(isinstance(result, GatewayCredentialsError) for result in results)
    assert {str(result) for result in results} == {"gateway unavailable"}


def test_gateway_failure_cache_expires(monkeypatch):
    app = _app_for_bootstrap()
    calls = 0
    now = 100.0

    async def failing_loader(_bearer_token):
        nonlocal calls
        calls += 1
        raise GatewayCredentialsError("gateway unavailable")

    monkeypatch.setattr(bootstrap, "load_credentials_from_gateway", failing_loader)
    monkeypatch.setattr(bootstrap, "GATEWAY_FAILURE_COOLDOWN_SECONDS", 15.0)
    monkeypatch.setattr(bootstrap.time, "monotonic", lambda: now)

    async def run():
        nonlocal now
        with pytest.raises(GatewayCredentialsError):
            await bootstrap.ensure_ready(app, "user-token")
        now += 16.0
        with pytest.raises(GatewayCredentialsError):
            await bootstrap.ensure_ready(app, "user-token")

    asyncio.run(run())

    assert calls == 2


def test_new_login_token_does_not_reuse_cached_agent_stack(monkeypatch):
    app = _app_for_bootstrap()
    app.state.runtime.agent = object()
    app.state.runtime.creds_source = "aiapi-gateway"
    app.state.runtime.creds_token_fingerprint = bootstrap.cache_fingerprint("old-token")
    calls = 0

    async def close_stack(runtime):
        runtime.agent = None

    async def failing_loader(token):
        nonlocal calls
        calls += 1
        assert token == "new-token"
        raise GatewayCredentialsError("fresh token was used")

    monkeypatch.setattr(bootstrap, "_close_stack", close_stack)
    monkeypatch.setattr(bootstrap, "load_credentials_from_gateway", failing_loader)

    with pytest.raises(GatewayCredentialsError, match="fresh token was used"):
        asyncio.run(bootstrap.ensure_ready(app, "new-token"))

    assert calls == 1


def test_gateway_fetches_only_the_chat_model_with_the_users_token(monkeypatch):
    chat_cfg = GatewayModelConfig(
        model_name="chat",
        calling_model_name=None,
        base_url="https://chat.example",
        api_key="chat-key",
        api_format="openai",
        embedding_dimensions=None,
    )
    aliases_seen: list[str] = []

    async def fake_fetch(_client, *, model_alias, bearer_token, gateway_base_url):
        assert bearer_token == "user-token"
        assert gateway_base_url == "https://gateway.example"
        aliases_seen.append(model_alias)
        return chat_cfg

    monkeypatch.setattr(aiapi_credentials, "fetch_model_config", fake_fetch)
    monkeypatch.setenv("AIAPI_BASE_URL", "https://gateway.example")
    monkeypatch.setenv("AIAPI_CHAT_MODEL", "chat-alias")

    result = asyncio.run(aiapi_credentials.load_credentials_from_gateway("user-token"))

    assert result == chat_cfg
    assert aliases_seen == ["chat-alias"]
