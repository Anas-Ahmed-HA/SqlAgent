"""
Agent-host lifecycle: lazy construction of the agent stack.

The stack is built on the first authenticated request because Azure
credentials come from the gateway with the caller's own login token (token
passthrough). Startup cannot fail on missing credentials; the first chat
request surfaces gateway problems as 503.
"""

from __future__ import annotations

import logging
import os
import time

from shared.identity import cache_fingerprint

from agent.clients.aiapi_credentials import (
    GatewayCredentialsError,
    load_credentials_from_gateway,
    to_azure_settings,
)
from agent.clients.azure import AzureChatService
from agent.config import AgentConfig
from agent.loop import AgentLoop
from agent.memory.store import MemoryStore

__all__ = ["NotReadyError", "ensure_ready", "reset_ready"]

logger = logging.getLogger("sql_agent.bootstrap")


# A failed gateway build should not make every queued request repeat the same
# expensive fetch immediately. Keep the window short so a recovered gateway
# is retried without requiring a process restart.
GATEWAY_FAILURE_COOLDOWN_SECONDS = 15.0


class NotReadyError(RuntimeError):
    """The agent stack cannot be built yet (no token, no MCP session)."""


def _clear_gateway_failure(app) -> None:
    runtime = app.state.runtime
    runtime.gateway_failure = None
    runtime.gateway_failure_at = None
    runtime.gateway_failure_token_fingerprint = None


def _cached_gateway_failure(
    app, bearer_token: str | None
) -> GatewayCredentialsError | None:
    """Return a recent gateway failure, expiring it after the cooldown."""
    runtime = app.state.runtime
    failure = runtime.gateway_failure
    failed_at = runtime.gateway_failure_at
    if not isinstance(failure, str) or not failure or not isinstance(
        failed_at, (int, float)
    ):
        return None
    if runtime.gateway_failure_token_fingerprint != cache_fingerprint(bearer_token):
        _clear_gateway_failure(app)
        return None
    if time.monotonic() - failed_at < GATEWAY_FAILURE_COOLDOWN_SECONDS:
        # Keep only the message in runtime state; retaining the exception would
        # retain its traceback and the request frame (including the token).
        return GatewayCredentialsError(failure)
    _clear_gateway_failure(app)
    return None


def _remember_gateway_failure(
    app, failure: GatewayCredentialsError, bearer_token: str | None
) -> None:
    runtime = app.state.runtime
    runtime.gateway_failure = str(failure)
    runtime.gateway_failure_at = time.monotonic()
    runtime.gateway_failure_token_fingerprint = cache_fingerprint(bearer_token)
    runtime.creds_error = str(failure)


async def _close_stack(runtime) -> None:
    """Detach and close the current Azure stack.

    The caller must hold ``runtime.build_lock``. References are cleared before
    awaiting teardown so a queued rebuild cannot observe a half-closed stack.
    """
    azure = runtime.azure
    runtime.agent = None
    runtime.azure = None
    if azure is not None:
        try:
            await azure.aclose()
        except Exception:  # noqa: BLE001
            logger.exception("bootstrap.azure_close_failed")


async def ensure_ready(app, bearer_token: str | None) -> None:
    """Build the agent stack on first use; no-op once built.

    Raises `NotReadyError` when the stack cannot be built *yet* (no token on
    the request, MCP not connected) and `GatewayCredentialsError` when the
    gateway rejects the credentials fetch -- callers map both to 503.
    """
    runtime = app.state.runtime
    token_fingerprint = cache_fingerprint(bearer_token)
    if (
        runtime.agent is not None
        and not runtime.creds_stale
        and (
            runtime.creds_source != "aiapi-gateway"
            or runtime.creds_token_fingerprint == token_fingerprint
        )
    ):
        return

    if runtime.agent is not None and (
        runtime.creds_stale
        or runtime.creds_token_fingerprint != token_fingerprint
    ):
        # Do not wait while holding build_lock. An active turn may be the
        # task that will mark the stack stale and release its lease.
        await runtime.wait_for_turns_idle()

    cached_failure = _cached_gateway_failure(app, bearer_token)
    if cached_failure is not None:
        raise cached_failure

    async with runtime.build_lock:
        if (
            runtime.agent is not None
            and not runtime.creds_stale
            and (
                runtime.creds_source != "aiapi-gateway"
                or runtime.creds_token_fingerprint == token_fingerprint
            )
        ):
            return

        # Logout/login changes the gateway token while this process remains
        # alive. Do not reuse Azure credentials fetched for another session.
        if runtime.agent is not None and runtime.creds_token_fingerprint != token_fingerprint:
            await _close_stack(runtime)
            runtime.creds_stale = False

        # Requests that queued behind the leader see its failure instead of
        # starting another pair of gateway calls.
        if runtime.creds_stale:
            await _close_stack(runtime)

        cached_failure = _cached_gateway_failure(app, bearer_token)
        if cached_failure is not None:
            raise cached_failure

        mcp = runtime.mcp
        if mcp is None:
            raise NotReadyError("The MCP session is not initialized.")

        try:
            chat_cfg = await load_credentials_from_gateway(bearer_token)
        except GatewayCredentialsError as exc:
            _remember_gateway_failure(app, exc, bearer_token)
            raise

        api_version = os.environ.get(
            "AZURE_OPENAI_API_VERSION", "2024-08-01-preview"
        )
        def _mark_stale() -> None:
            runtime.creds_stale = True
            runtime.creds_error = (
                "Azure rejected the cached credentials; the configuration was "
                "cleared and will be re-fetched with your login token on the next "
                "request."
            )
            logger.warning("bootstrap.creds_stale")

        azure_settings = to_azure_settings(chat_cfg, api_version=api_version)
        azure = AzureChatService(azure_settings, on_auth_failure=_mark_stale)

        if runtime.memory is None:
            runtime.memory = MemoryStore()

        runtime.azure = azure
        runtime.agent = AgentLoop(
            mcp=mcp, azure=azure, memory=runtime.memory, config=AgentConfig.from_env()
        )
        runtime.creds_source = "aiapi-gateway"
        runtime.creds_token_fingerprint = token_fingerprint
        runtime.creds_error = ""
        runtime.creds_stale = False
        _clear_gateway_failure(app)
        logger.info("bootstrap.ready")


async def reset_ready(app) -> None:
    """Tear the built stack down so the next request rebuilds it.

    Used when Azure rejects the cached credentials (gateway-side rotation):
    the next authenticated request fetches fresh credentials with its own
    token.
    """
    runtime = app.state.runtime
    # An active turn may discover credential rotation while still using
    # the cached clients. Mark the stack stale and let its finally block
    # release the lease; the next request performs the close.
    if runtime.current_turn_is_active():
        runtime.creds_stale = True
        runtime.creds_source = "lazy"
        _clear_gateway_failure(app)
        logger.info("bootstrap.reset_deferred")
        return
    await runtime.wait_for_turns_idle()
    async with runtime.build_lock:
        if runtime.current_turn_is_active():
            runtime.creds_stale = True
            runtime.creds_source = "lazy"
            _clear_gateway_failure(app)
            logger.info("bootstrap.reset_deferred")
            return
        await _close_stack(runtime)
        runtime.creds_stale = False
        runtime.creds_source = "lazy"
        _clear_gateway_failure(app)
        logger.info("bootstrap.reset")
