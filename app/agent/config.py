"""Agent, model and MCP server settings.

Service endpoints and the local API address live in :mod:`shared.settings`,
which loads ``.env`` on import, so every value here sees the environment.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from shared.settings import env_float as _env_float
from shared.settings import env_int as _env_int
from shared.settings import MAX_EQUIPPED_SKILL_WORDS, MAX_EQUIPPED_SKILLS

AIAPI_CHAT_MODEL = os.environ.get("AIAPI_CHAT_MODEL", "LLM_MCP")
AIAPI_TIMEOUT_SECONDS = _env_float("AIAPI_TIMEOUT_SECONDS", 30.0)


@dataclass
class AgentConfig:
    """Limits of one turn and of the context sent to the model."""

    #: Model calls per turn. Each call may run several tools; a turn that
    #: reaches this writes its answer with what it has.
    max_steps: int = 25

    #: Wall-clock ceiling for one turn. Time spent waiting for the user to
    #: answer a question is given back.
    max_request_seconds: float = 600.0

    #: Estimated tokens of messages sent to the model per call (see
    #: agent/memory/context_budget.py). Every call in a turn resends the whole
    #: context, so size is paid again per step.
    context_token_budget: int = 200_000

    #: Safety cap on messages per call. Size (above) is the real limit; this
    #: only bounds a pathological count of tiny messages.
    max_context_messages: int = 400

    #: Messages persisted per session between requests.
    max_history_messages: int = 40

    #: Earlier turns sent whole. Older turns have their tool outputs shortened
    #: (agent/memory/context_budget.py, `trim_earlier_turns`).
    full_turns: int = 3

    #: How many of a turn's skills, and how many words in them, reach Azure.
    max_equipped_skills: int = MAX_EQUIPPED_SKILLS
    max_equipped_skill_words: int = MAX_EQUIPPED_SKILL_WORDS

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            max_steps=_env_int("AGENT_MAX_STEPS", cls.max_steps),
            max_request_seconds=_env_float("AGENT_MAX_REQUEST_SECONDS", cls.max_request_seconds),
            context_token_budget=_env_int("AGENT_CONTEXT_TOKEN_BUDGET", cls.context_token_budget),
            max_context_messages=_env_int("AGENT_MAX_CONTEXT_MESSAGES", cls.max_context_messages),
            max_history_messages=_env_int("AGENT_MAX_HISTORY_MESSAGES", cls.max_history_messages),
            full_turns=_env_int("AGENT_FULL_TURNS", cls.full_turns),
        )


@dataclass
class MCPServerSettings:
    """The MCP server the agent uses, reached by URL.

    ``headers`` are sent on every request, e.g. ``{"X-Api-Key": "..."}``.
    They are configuration, never something the model can set.
    """

    url: str = "http://localhost:5100/mcp"
    headers: dict[str, str] = field(default_factory=dict)
    connect_timeout_seconds: float = 30.0
    call_timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls) -> "MCPServerSettings":
        raw_headers = os.environ.get("MCP_HEADERS", "").strip()
        headers = json.loads(raw_headers) if raw_headers else {}
        if not isinstance(headers, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
        ):
            raise ValueError('MCP_HEADERS must be a JSON object of strings, e.g. {"X-Api-Key": "..."}')
        return cls(
            url=os.environ.get("MCP_URL", cls.url).strip(),
            headers=headers,
            connect_timeout_seconds=_env_float("MCP_CONNECT_TIMEOUT_SECONDS", cls.connect_timeout_seconds),
            call_timeout_seconds=_env_float("MCP_CALL_TIMEOUT_SECONDS", cls.call_timeout_seconds),
        )


@dataclass
class AzureSettings:
    endpoint: str
    api_key: str
    deployment: str
    api_version: str = "2024-08-01-preview"
