"""
Tests for the credentials gateway integration (agent/clients/aiapi_credentials.py).

Hermetic by design: no network. The scrambled gateway payload is rebuilt with
the inverse of the decode scheme, and the HTTP calls run through
`httpx.MockTransport`.
"""

from __future__ import annotations

import asyncio
import base64
import json

import httpx
import pytest

from agent.clients.aiapi_credentials import (
    GatewayCredentialsError,
    decode_model_response,
    fetch_model_config,
    to_azure_settings,
)

SAMPLE_TOKEN = "the-user-s-gateway-login-token"

CHAT_PAYLOAD = {
    "modelName": "gpt-5.6-luna",
    "callingModelName": "LLM_MCP",
    "baseUrl": "https://example-resource.openai.azure.com/openai/v1",
    "apiKey": "test-chat-key",
    "forDeveloper": False,
    "apiFormat": "openai",
    "embeddingDimensions": None,
}

EMBEDDING_PAYLOAD = {
    "modelName": "text-embedding-ada-002",
    "callingModelName": "Embedding",
    "baseUrl": "https://example-resource.services.ai.azure.com",
    "apiKey": "test-embed-key",
    "forDeveloper": False,
    "apiFormat": None,
    "embeddingDimensions": 1536,
}


def scramble(payload: dict) -> str:
    """Inverse of the gateway's scramble: JSON -> base64 -> reverse -> pad 3+3."""
    encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    return "hzq" + encoded[::-1] + "kwt"


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# -- decode ------------------------------------------------------------------


def test_decode_roundtrips_the_verified_scheme():
    cfg = decode_model_response(scramble(CHAT_PAYLOAD))
    assert cfg.model_name == "gpt-5.6-luna"
    assert cfg.calling_model_name == "LLM_MCP"
    assert cfg.base_url == "https://example-resource.openai.azure.com/openai/v1"
    assert cfg.api_key == "test-chat-key"
    assert cfg.api_format == "openai"
    assert cfg.embedding_dimensions is None


def test_decode_handles_the_embedding_payload():
    cfg = decode_model_response(scramble(EMBEDDING_PAYLOAD))
    assert cfg.model_name == "text-embedding-ada-002"
    assert cfg.embedding_dimensions == 1536


def test_decode_rejects_a_too_short_body():
    with pytest.raises(GatewayCredentialsError, match="too short"):
        decode_model_response("hzqkwt")


def test_decode_rejects_a_garbled_body():
    with pytest.raises(GatewayCredentialsError, match="could not decode"):
        decode_model_response("hzq!!!not-base64!!!kwt")


def test_decode_rejects_a_payload_missing_required_keys():
    with pytest.raises(GatewayCredentialsError, match="missing"):
        decode_model_response(scramble({"modelName": "x"}))


# -- translations ------------------------------------------------------------


def test_to_azure_settings_strips_the_openai_v1_suffix():
    cfg = decode_model_response(scramble(CHAT_PAYLOAD))
    settings = to_azure_settings(cfg, api_version="2025-03-01-preview")
    assert settings.endpoint == "https://example-resource.openai.azure.com/"
    assert settings.deployment == "gpt-5.6-luna"
    assert settings.api_key == "test-chat-key"
    assert settings.api_version == "2025-03-01-preview"

def test_to_azure_settings_keeps_a_deployments_style_base_url():
    cfg = decode_model_response(scramble(EMBEDDING_PAYLOAD))
    settings = to_azure_settings(cfg, api_version="2024-08-01-preview")
    assert settings.endpoint == "https://example-resource.services.ai.azure.com/"


# -- fetch -------------------------------------------------------------------


def _route(handler):
    def run(model_alias: str):
        async def go():
            async with _client(handler) as client:
                return await fetch_model_config(
                    client, model_alias=model_alias, bearer_token=SAMPLE_TOKEN
                )

        return asyncio.run(go())

    return run


def test_fetch_returns_the_decoded_chat_config():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/apps/models/LLM_MCP"
        auth = request.headers["Authorization"]
        assert auth == f"Bearer {SAMPLE_TOKEN}"
        return httpx.Response(200, text=scramble(CHAT_PAYLOAD))

    cfg = _route(handler)("LLM_MCP")
    assert cfg.model_name == "gpt-5.6-luna"


def test_fetch_unknown_model_is_an_actionable_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "No Models row for 'nope' on this route"})

    with pytest.raises(GatewayCredentialsError, match="no model 'nope'"):
        _route(handler)("nope")


def test_fetch_auth_failure_is_an_actionable_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    with pytest.raises(GatewayCredentialsError, match="rejected the login token"):
        _route(handler)("LLM_MCP")


def test_fetch_malformed_body_is_an_error_not_a_crash():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="hzqgarbledkwt")

    with pytest.raises(GatewayCredentialsError):
        _route(handler)("LLM_MCP")


def test_fetch_network_failure_is_wrapped():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(GatewayCredentialsError, match="could not reach"):
        _route(handler)("LLM_MCP")


def test_fetch_retries_transient_gateway_failures(monkeypatch):
    calls = 0
    delays = []

    async def fake_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr("agent.clients.aiapi_credentials.asyncio.sleep", fake_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(503, text="temporarily unavailable")
        return httpx.Response(200, text=scramble(CHAT_PAYLOAD))

    cfg = _route(handler)("LLM_MCP")
    assert cfg.model_name == "gpt-5.6-luna"
    assert calls == 3
    assert delays == [0.5, 1.0]


def test_fetch_does_not_retry_authentication_failures():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="unauthorized")

    with pytest.raises(GatewayCredentialsError, match="rejected the login token"):
        _route(handler)("LLM_MCP")
    assert calls == 1


def test_fetch_without_a_token_is_an_actionable_error():
    """Token passthrough means no token on the request = no credentials fetch."""
    async def go():
        async with _client(lambda request: None) as client:  # pragma: no cover
            return await fetch_model_config(
                client, model_alias="LLM_MCP", bearer_token=""
            )

    with pytest.raises(GatewayCredentialsError, match="no bearer token"):
        asyncio.run(go())
