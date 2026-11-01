"""Agent, model and MCP server settings.

Service endpoints and the local API address live in :mod:`shared.settings`,
which loads ``.env`` on import, so every value here sees the environment.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

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


#: The name the charts server goes by: its tools are offered only to a client
#: that draws charts.
CHARTS_SERVER = "charts"


#: The required server's name; an extra server may not take it.
SQL_SERVER = "sql"
#: A server's name: it routes nothing, but shows in logs and in the prompt.
_SERVER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,39}$")
#: Extra servers are optional, so a turn never waits long for one.
_EXTRA_CONNECT_SECONDS = 5.0
_EXTRA_CALL_SECONDS = 60.0


@dataclass
class MCPServerSettings:
    """An MCP server the agent uses, reached by URL.

    ``headers`` are sent on every request, e.g. ``{"X-Api-Key": "..."}``.
    They are configuration, never something the model can set.
    """

    url: str = "http://localhost:5100/mcp"
    headers: dict[str, str] = field(default_factory=dict)
    connect_timeout_seconds: float = 30.0
    call_timeout_seconds: float = 120.0
    name: str = SQL_SERVER

    @classmethod
    def from_env(cls) -> "MCPServerSettings":
        """The required SQL server (``MCP_URL``)."""
        return cls(
            url=os.environ.get("MCP_URL", cls.url).strip(),
            headers=_headers(os.environ.get("MCP_HEADERS", ""), "MCP_HEADERS"),
            connect_timeout_seconds=_env_float("MCP_CONNECT_TIMEOUT_SECONDS", cls.connect_timeout_seconds),
            call_timeout_seconds=_env_float("MCP_CALL_TIMEOUT_SECONDS", cls.call_timeout_seconds),
        )

    @classmethod
    def extras_from_env(cls) -> list["MCPServerSettings"]:
        """The optional servers, in the order their tools are offered.

        ``MCP_EXTRA_SERVERS`` is a JSON object of servers by name::

            {"charts": {"url": "http://localhost:5200/mcp"},
             "docs": {"url": "...", "headers": {"X-Api-Key": "..."},
                      "connect_timeout_seconds": 5, "call_timeout_seconds": 60}}

        ``CHARTS_MCP_URL`` (with ``CHARTS_MCP_HEADERS`` and its timeouts) is a
        shortcut for the ``charts`` entry, used when that object has none. A
        mistake raises at startup, naming the setting to fix.
        """
        raw = os.environ.get("MCP_EXTRA_SERVERS", "").strip()
        try:
            entries = json.loads(raw) if raw else {}
        except ValueError as exc:
            raise ValueError(f"MCP_EXTRA_SERVERS is not valid JSON: {exc}") from exc
        if not isinstance(entries, dict):
            raise ValueError('MCP_EXTRA_SERVERS must be a JSON object of servers by name, e.g. {"charts": {"url": "..."}}')
        servers = [cls._extra(name, entry) for name, entry in entries.items()]
        charts_url = os.environ.get("CHARTS_MCP_URL", "").strip()
        if charts_url and CHARTS_SERVER not in entries:
            servers.append(cls(
                url=charts_url,
                headers=_headers(os.environ.get("CHARTS_MCP_HEADERS", ""), "CHARTS_MCP_HEADERS"),
                connect_timeout_seconds=_env_float("CHARTS_MCP_CONNECT_TIMEOUT_SECONDS", _EXTRA_CONNECT_SECONDS),
                call_timeout_seconds=_env_float("CHARTS_MCP_CALL_TIMEOUT_SECONDS", _EXTRA_CALL_SECONDS),
                name=CHARTS_SERVER,
            ))
        return servers

    @classmethod
    def _extra(cls, name: Any, entry: Any) -> "MCPServerSettings":
        where = f"MCP_EXTRA_SERVERS[{json.dumps(name)}]"
        if not isinstance(name, str) or not _SERVER_NAME.match(name):
            raise ValueError(f"{where}: a name is a letter, then letters, digits, - or _ (at most 40)")
        if name == SQL_SERVER:
            raise ValueError(f"{where}: `{SQL_SERVER}` is the required server, set by MCP_URL")
        if not isinstance(entry, dict):
            raise ValueError(f'{where} must be an object, e.g. {{"url": "http://localhost:5200/mcp"}}')
        url = entry.get("url")
        if not isinstance(url, str) or not url.strip().lower().startswith(("http://", "https://")):
            raise ValueError(f"{where}.url must be the server's http(s) MCP endpoint")
        unknown = set(entry) - {"url", "headers", "connect_timeout_seconds", "call_timeout_seconds"}
        if unknown:
            raise ValueError(f"{where}: unknown settings {sorted(unknown)}")
        return cls(
            url=url.strip(),
            headers=_headers(entry.get("headers") or {}, f"{where}.headers"),
            connect_timeout_seconds=_seconds(entry, "connect_timeout_seconds", _EXTRA_CONNECT_SECONDS, where),
            call_timeout_seconds=_seconds(entry, "call_timeout_seconds", _EXTRA_CALL_SECONDS, where),
            name=name,
        )


def _headers(raw: Any, where: str) -> dict[str, str]:
    """Headers from a JSON string (an environment variable) or an object."""
    headers = raw
    if isinstance(raw, str):
        try:
            headers = json.loads(raw) if raw.strip() else {}
        except ValueError:
            headers = None
    if not isinstance(headers, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
    ):
        raise ValueError(f'{where} must be a JSON object of strings, e.g. {{"X-Api-Key": "..."}}')
    return headers


def _seconds(entry: dict[str, Any], key: str, default: float, where: str) -> float:
    value = entry.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{where}.{key} must be a number of seconds above 0")
    return float(value)


@dataclass
class AzureSettings:
    endpoint: str
    api_key: str
    deployment: str
    api_version: str = "2024-08-01-preview"
