"""Bounded, in-process conversation history and lessons, keyed by session."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

__all__ = ["MemoryStore", "SessionMemory"]


@dataclass
class SessionMemory:
    """Everything remembered about one conversation."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    successful_tools: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)



class MemoryStore:
    """Async-safe session storage with least-recently-used eviction."""

    def __init__(
        self,
        max_sessions: int = 500,
        max_lessons_per_session: int = 32,
    ) -> None:
        self._sessions: OrderedDict[str, SessionMemory] = OrderedDict()
        self._lock = asyncio.Lock()
        self._max_sessions = max_sessions
        self._max_lessons = max_lessons_per_session

    # -- short-term --------------------------------------------------------

    async def load(self, session_id: str) -> SessionMemory:
        async with self._lock:
            memory = self._sessions.get(session_id)
            if memory is None:
                return SessionMemory()
            self._sessions.move_to_end(session_id)
            # Hand back copies: the caller mutates its turn state, and a
            # shared mutable list would let one request corrupt another's view.
            return SessionMemory(
                messages=list(memory.messages),
                lessons=list(memory.lessons),
                successful_tools=list(memory.successful_tools),
                updated_at=memory.updated_at,
            )

    async def save(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        lessons: list[str],
        successful_tools: list[str],
    ) -> None:
        async with self._lock:
            memory = SessionMemory(
                messages=list(messages),
                lessons=list(lessons)[-self._max_lessons :],
                successful_tools=list(successful_tools)[-32:],
            )
            self._sessions[session_id] = memory
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)
