"""Regression tests for credential-refresh error classification."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from agent.api.auth import TokenClaims
from agent.host import turns as chat_module
from shared.api_schemas import ChatRequest
from agent.errors import CredentialsStaleException, MCPConnectionException
from agent.host.runtime import Runtime


def _request(workflow, *, stale=False):
    app = SimpleNamespace(
        state=SimpleNamespace(
            runtime=Runtime(
                agent=workflow,
                creds_stale=stale,
                creds_error="",
            )
        )
    )
    return SimpleNamespace(
        app=app,
        headers={"Authorization": "Bearer token"},
    )


CLAIMS = TokenClaims(hr_code="HR-1", email="engineer@example.com")


class _Raises:
    def __init__(self, error):
        self.error = error
        self.app = None

    async def run(self, **kwargs):
        if self.app is not None:
            self.app.state.runtime.creds_stale = True
        raise self.error


def test_only_typed_stale_error_uses_refresh_path(monkeypatch, caplog):
    reset_calls = []

    async def fake_reset(app):
        reset_calls.append(app)

    monkeypatch.setattr(chat_module, "reset_ready", fake_reset)
    workflow = _Raises(CredentialsStaleException("auth failed"))
    request = _request(workflow)
    workflow.app = request.app

    with pytest.raises(HTTPException) as raised:
        asyncio.run(chat_module.run_chat(ChatRequest(message="hi"), request, CLAIMS))

    assert raised.value.status_code == 503
    assert "Azure rejected" in raised.value.detail
    assert reset_calls == [request.app]
    assert "chat.credentials_stale" in caplog.text


def test_an_unrelated_error_is_not_misreported_as_credentials(caplog):
    workflow = _Raises(MCPConnectionException("the MCP server is down"))
    request = _request(workflow)
    workflow.app = request.app

    with pytest.raises(HTTPException) as raised:
        asyncio.run(chat_module.run_chat(ChatRequest(message="hi"), request, CLAIMS))

    assert raised.value.status_code == 503
    assert raised.value.detail == "the MCP server is down"
    assert "Azure rejected" not in raised.value.detail
    assert "chat.dependency_failure" in caplog.text
