"""
Azure OpenAI access.

Uses `AsyncAzureOpenAI` so a completion never stalls the event loop or takes a
thread-pool slot. `max_retries=3` handles transient 429/5xx inside the SDK; it
is unrelated to the loop's step limit.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from openai import AsyncAzureOpenAI
from openai import AuthenticationError as OpenAIAuthenticationError

from agent.config import AzureSettings
from agent.errors import ContextTooLongException, CredentialsStaleException, PlanningException
from agent.memory.history import provider_messages

__all__ = ["AzureChatService", "is_context_too_long"]


def is_context_too_long(exc: BaseException) -> bool:
    """Whether the model rejected the request for exceeding its context.

    OpenAI-compatible endpoints report it as a 400 with code
    ``context_length_exceeded``, or ``string_above_max_length`` when one
    message is longer than the API takes; the message text is the fallback
    for gateways that only forward the text.
    """
    if getattr(exc, "code", None) in _TOO_LONG_CODES:
        return True
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error") if isinstance(body.get("error"), dict) else body
        if error.get("code") in _TOO_LONG_CODES:
            return True
    text = str(exc).lower()
    return any(code in text for code in _TOO_LONG_CODES) or "maximum context length" in text


_TOO_LONG_CODES = ("context_length_exceeded", "string_above_max_length")

logger = logging.getLogger("sql_agent.azure")


class AzureChatService:
    """Thin, testable seam over the Azure chat-completions endpoint.

    The loop talks to Azure only through this class, so tests inject a fake
    here instead of monkeypatching the SDK.

    `on_auth_failure` is invoked when Azure rejects the configured credentials
    (gateway-side rotation): the bootstrap clears the cached stack so the next
    request re-fetches with the caller's fresh login token.
    """

    def __init__(
        self, settings: AzureSettings, *, on_auth_failure: Callable[[], None] | None = None
    ) -> None:
        self._deployment = settings.deployment
        self._on_auth_failure = on_auth_failure
        self._client = AsyncAzureOpenAI(
            azure_endpoint=settings.endpoint,
            api_key=settings.api_key,
            api_version=settings.api_version,
            max_retries=3,
        )

    async def complete(self, **kwargs: Any) -> Any:
        """One chat completion, with timing and usage logged."""
        started = time.perf_counter()
        if "messages" in kwargs:
            kwargs["messages"] = provider_messages(kwargs["messages"])
        try:
            response = await self._client.chat.completions.create(
                model=self._deployment, **kwargs
            )
        except OpenAIAuthenticationError as exc:
            # Gateway-side credential rotation: mark the cached stack stale so
            # the next request re-fetches with the caller's fresh login token.
            if self._on_auth_failure:
                self._on_auth_failure()
            logger.error(
                "azure.completion.auth_failed",
                extra={"elapsed_ms": round((time.perf_counter() - started) * 1000)},
                exc_info=True,
            )
            raise CredentialsStaleException(
                f"Azure OpenAI authentication failed: {exc}. The credentials are "
                "stale -- re-login and retry, or the cached config will be refreshed."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed error
            logger.error(
                "azure.completion.failed",
                extra={"elapsed_ms": round((time.perf_counter() - started) * 1000)},
                exc_info=True,
            )
            if is_context_too_long(exc):
                # Typed so the loop can retry once with a smaller context
                # instead of failing the turn.
                raise ContextTooLongException(
                    f"Azure OpenAI rejected the context as too long: {exc}"
                ) from exc
            raise PlanningException(f"Azure OpenAI call failed: {exc}") from exc

        usage = getattr(response, "usage", None)
        logger.info(
            "azure.completion",
            extra={
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            },
        )
        return response

    async def complete_text(self, system_prompt: str, user_content: str) -> str:
        """Single-shot helper for the skill rewrite."""
        response = await self.complete(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        )
        return response.choices[0].message.content or ""

    async def aclose(self) -> None:
        await self._client.close()
