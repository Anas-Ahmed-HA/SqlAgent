"""Stable fingerprints for tool calls."""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = ["failure_signature"]


def failure_signature(tool_name: str, arguments: dict[str, Any]) -> str:
    """Identifies a tool call by name plus a digest of its arguments.

    Used to detect a model re-emitting an attempt that has already failed in
    this request. Sorting keys makes the digest independent of the order the
    model happened to serialise its JSON in.
    """
    payload = json.dumps(arguments, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{tool_name}:{digest}"
