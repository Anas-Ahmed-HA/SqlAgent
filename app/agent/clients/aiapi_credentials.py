"""
Credentials from the Hassan Allam AI gateway (aiapi.hassanallam.com).

The service ships with ZERO Azure secrets and ZERO gateway signing power.
The caller's own gateway-issued login token (the same Bearer token the
Flutter app presents) is forwarded verbatim to `GET /apps/models/{name}`:

* `LLM_MCP`   -> the chat model config (currently gpt-5.6-luna)

Because the token is forwarded and never minted, this process cannot
impersonate anyone: it can only act as the user who is already logged in.
The response body is a scrambled string, not plain base64. Verified decode
scheme against the live gateway:

    1. strip the first 3 and the last 3 characters
    2. reverse the remaining string
    3. base64-decode, then UTF-8 decode -> JSON with modelName,
       callingModelName, baseUrl, apiKey, forDeveloper, apiFormat

URL-surface translation
-----------------------
The gateway reports the OpenAI-compatible surface (`baseUrl` ending in
`/openai/v1`, `apiFormat: "openai"`). This service talks to Azure through
the classic deployments API (`AsyncAzureOpenAI`). Both surfaces accept the
same key and the same model name -- verified live against
`/openai/deployments/gpt-5.6-luna/chat/completions` on two api-versions -- so
the chat translation is: strip a trailing `/openai/v1` from `baseUrl`, and use
`modelName` as the deployment name.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from dataclasses import dataclass

import httpx

from shared.settings import AIAPI_BASE_URL
from agent.config import (
    AIAPI_CHAT_MODEL,
    AIAPI_TIMEOUT_SECONDS,
    AzureSettings,
)

__all__ = [
    "DEFAULT_CHAT_MODEL_ALIAS",
    "DEFAULT_GATEWAY_BASE_URL",
    "GatewayCredentialsError",
    "GatewayModelConfig",
    "decode_model_response",
    "fetch_model_config",
    "load_credentials_from_gateway",
    "to_azure_settings",
]

logger = logging.getLogger("sql_agent.aiapi")

DEFAULT_GATEWAY_BASE_URL = AIAPI_BASE_URL
DEFAULT_CHAT_MODEL_ALIAS = AIAPI_CHAT_MODEL
GATEWAY_TIMEOUT_SECONDS = AIAPI_TIMEOUT_SECONDS
# Retry policy for the bootstrap credential fetch. Fixed constants, not config:
# this runs once at startup and is never tuned per deployment.
GATEWAY_RETRY_ATTEMPTS = 3
GATEWAY_RETRY_BACKOFF_SECONDS = 0.5

#: The gateway's chat `baseUrl` points at the OpenAI-compatible surface; this
#: service speaks the deployments API against the bare resource host.
_OPENAI_V1_SUFFIX = "/openai/v1"


class GatewayCredentialsError(RuntimeError):
    """The credentials gateway could not serve usable model config."""


@dataclass(frozen=True)
class GatewayModelConfig:
    """One decoded `/apps/models/{name}` payload."""

    model_name: str
    calling_model_name: str | None
    base_url: str
    api_key: str
    api_format: str | None
    embedding_dimensions: int | None

    @classmethod
    def from_payload(cls, payload: dict) -> "GatewayModelConfig":
        missing = [k for k in ("modelName", "baseUrl", "apiKey") if not payload.get(k)]
        if missing:
            raise GatewayCredentialsError(
                f"decoded gateway payload is missing {missing}; keys={sorted(payload)}"
            )
        dims = payload.get("embeddingDimensions")
        return cls(
            model_name=payload["modelName"],
            calling_model_name=payload.get("callingModelName"),
            base_url=payload["baseUrl"].rstrip("/"),
            api_key=payload["apiKey"],
            api_format=payload.get("apiFormat"),
            embedding_dimensions=int(dims) if dims else None,
        )


def decode_model_response(raw: str) -> GatewayModelConfig:
    """Decode the scrambled `/apps/models/{name}` body (see module docstring)."""
    if len(raw) < 7:
        raise GatewayCredentialsError(
            f"gateway response too short to decode ({len(raw)} chars)"
        )
    try:
        payload = json.loads(base64.b64decode(raw[3:-3][::-1]).decode("utf-8"))
    except ValueError as exc:
        # binascii.Error, UnicodeDecodeError and JSONDecodeError all subclass
        # ValueError, so this covers every malformed-body case.
        raise GatewayCredentialsError(
            f"could not decode gateway response: {exc}"
        ) from exc
    return GatewayModelConfig.from_payload(payload)

async def fetch_model_config(
    client: httpx.AsyncClient,
    *,
    model_alias: str,
    bearer_token: str,
    gateway_base_url: str = DEFAULT_GATEWAY_BASE_URL,
) -> GatewayModelConfig:
    """Fetch + decode one model config, with actionable error messages.

    `bearer_token` is the CALLER's own gateway-issued token, forwarded
    verbatim (token passthrough). This module never mints tokens in
    production never creates gateway tokens; it forwards the caller's token.
    """
    if not bearer_token:
        raise GatewayCredentialsError(
            "no bearer token available -- the request must carry the "
            "user's `Authorization: Bearer <token>` header."
        )
    url = f"{gateway_base_url.rstrip('/')}/apps/models/{model_alias}"
    for attempt in range(GATEWAY_RETRY_ATTEMPTS):
        try:
            resp = await client.get(
                url, headers={"Authorization": f"Bearer {bearer_token}"}
            )
        except httpx.HTTPError as exc:
            if attempt == GATEWAY_RETRY_ATTEMPTS - 1:
                raise GatewayCredentialsError(
                    f"could not reach the credentials gateway ({url}): {exc}"
                ) from exc
            delay = GATEWAY_RETRY_BACKOFF_SECONDS * (2**attempt)
            logger.warning(
                "aiapi.retry",
                extra={"model_alias": model_alias, "attempt": attempt + 1, "delay": delay},
            )
            await asyncio.sleep(delay)
            continue

        if resp.status_code not in (408, 429) and not 500 <= resp.status_code < 600:
            break
        if attempt == GATEWAY_RETRY_ATTEMPTS - 1:
            break
        delay = GATEWAY_RETRY_BACKOFF_SECONDS * (2**attempt)
        logger.warning(
            "aiapi.retry",
            extra={"model_alias": model_alias, "attempt": attempt + 1, "delay": delay},
        )
        await asyncio.sleep(delay)

    if resp.status_code == 404:
        raise GatewayCredentialsError(
            f"credentials gateway has no model '{model_alias}' "
            f"({url}). Check AIAPI_CHAT_MODEL."
        )
    if resp.status_code in (401, 403):
        raise GatewayCredentialsError(
            f"the gateway rejected the login token for '{model_alias}' "
            f"({resp.status_code}). Your login may have expired -- "
            "re-login in the app and try again."
        )
    if resp.status_code != 200:
        raise GatewayCredentialsError(
            f"credentials gateway returned {resp.status_code} for "
            f"'{model_alias}': {resp.text[:200]}"
        )
    return decode_model_response(resp.text.strip().strip('"'))


def to_azure_settings(cfg: GatewayModelConfig, api_version: str) -> AzureSettings:
    """Translate a chat model config into the deployments-API settings that
    `AzureChatService` consumes."""
    base = cfg.base_url
    if base.endswith(_OPENAI_V1_SUFFIX):
        base = base[: -len(_OPENAI_V1_SUFFIX)]
    endpoint = base if base.endswith("/") else base + "/"
    return AzureSettings(
        endpoint=endpoint,
        api_key=cfg.api_key,
        deployment=cfg.model_name,
        api_version=api_version,
    )


async def load_credentials_from_gateway(bearer_token: str | None) -> GatewayModelConfig:
    """Request-path entry point used by the lazy bootstrap: the chat model's config.

    `bearer_token` is the caller's own gateway-issued login token, forwarded
    verbatim -- this process never mints gateway tokens. Raises
    `GatewayCredentialsError` on any failure; the caller turns that into a
    503 with the actionable detail.
    """
    if not bearer_token:
        raise GatewayCredentialsError(
            "no bearer token on the request -- log in first; the agent "
            "fetches Azure credentials with the user's own login token."
        )
    # Read overrides at request time so tests and long-running processes can
    # change deployment settings without reloading this module.
    gateway_base_url = os.environ.get("AIAPI_BASE_URL") or DEFAULT_GATEWAY_BASE_URL
    chat_alias = os.environ.get("AIAPI_CHAT_MODEL") or DEFAULT_CHAT_MODEL_ALIAS

    async with httpx.AsyncClient(timeout=GATEWAY_TIMEOUT_SECONDS) as client:
        chat_cfg = await fetch_model_config(
            client,
            model_alias=chat_alias,
            bearer_token=bearer_token,
            gateway_base_url=gateway_base_url,
        )

    # Deliberately no keys in the log line.
    logger.info(
        "aiapi.credentials.fetched",
        extra={
            "chat_alias": chat_alias,
            "chat_model": chat_cfg.model_name,
            "chat_endpoint": to_azure_settings(chat_cfg, "v").endpoint,
        },
    )
    return chat_cfg
