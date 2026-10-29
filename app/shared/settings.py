"""Settings read across the service: endpoints, the local API address, and
the limits on questions and skills.

Agent-only settings (models, the MCP server, turn limits) live in
``agent.config``.

``.env`` is loaded on import, so call :func:`shared.env.ensure_env_early`
first when the file may still need creating: the constants below are frozen
at import time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "AIAPI_BASE_URL",
    "EMPLOYEE_API_URL",
    "LOCAL_API_HOST",
    "LOCAL_API_PORT",
    "MAX_ASKS_PER_TURN",
    "MAX_EQUIPPED_SKILL_WORDS",
    "MAX_EQUIPPED_SKILLS",
    "MAX_QUESTIONS_PER_ASK",
    "MAX_QUESTIONS_PER_TURN",
    "QUESTION_RECONNECT_SECONDS",
    "QUESTION_WAIT_SECONDS",
    "SHARED_API_BASE_URL",
    "env_bool",
    "env_float",
    "env_int",
    "resolve_app_dir",
]


def resolve_app_dir() -> Path:
    """Where runtime state (``.env``, logs) lives: the repository root."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parents[2]


def _load_config_environment() -> None:
    """Load the external environment before any constant below is read."""
    try:
        from dotenv import load_dotenv

        load_dotenv(resolve_app_dir() / ".env")
    except Exception:
        pass


_load_config_environment()


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    try:
        return float(raw) if raw is not None else default
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


LOCAL_API_HOST = os.environ.get("AGENT_HOST", "127.0.0.1") or "127.0.0.1"
LOCAL_API_PORT = env_int("AGENT_PORT", 8321)
SHARED_API_BASE_URL = os.environ.get("SHARED_API_URL", "https://api.hassanallam.com")
AIAPI_BASE_URL = os.environ.get("AIAPI_BASE_URL", "https://aiapi.hassanallam.com")
EMPLOYEE_API_URL = os.environ.get(
    "EMPLOYEE_API_URL",
    f"{SHARED_API_BASE_URL}/api/portal/Employee_Basic_Info/GetByUserhrcode",
)

# User skills. The client checks these before saving to the skills API; the
# agent applies them again to every turn's skills.
MAX_EQUIPPED_SKILLS = env_int("AGENT_MAX_EQUIPPED_SKILLS", 16)
MAX_EQUIPPED_SKILL_WORDS = env_int("AGENT_MAX_EQUIPPED_SKILL_WORDS", 4000)

# Questions the assistant asks mid-turn (the ask_user tool). One ask holds 1 to
# MAX_QUESTIONS_PER_ASK questions, answered together; a turn may ask at most
# MAX_ASKS_PER_TURN times and MAX_QUESTIONS_PER_TURN questions in all. Time
# spent waiting for an answer is not counted against the turn's deadline, so a
# chat request may run that much longer.
QUESTION_WAIT_SECONDS = env_float("AGENT_QUESTION_WAIT_SECONDS", 600.0)
MAX_QUESTIONS_PER_ASK = env_int("AGENT_MAX_QUESTIONS_PER_ASK", 4)
MAX_ASKS_PER_TURN = env_int("AGENT_MAX_ASKS_PER_TURN", 3)
MAX_QUESTIONS_PER_TURN = env_int("AGENT_MAX_QUESTIONS_PER_TURN", 8)
#: How long an open question survives the client's event socket going away, so
#: a dropped connection can come back before the question is closed unanswered.
QUESTION_RECONNECT_SECONDS = env_float("AGENT_QUESTION_RECONNECT_SECONDS", 30.0)
