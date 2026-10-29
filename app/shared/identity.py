"""Shared identity rules (stdlib only).

Single source for HR-scoped conversation keys, JWT claim lookup, expiry
parsing, token fingerprints, and employee-record matching. Backend memory,
and chat coordination must agree:

- ``conversation_id`` is the logical conversation (``"default"`` for now).
- ``memory_scope`` is ``"<hr_code>|<suffix>"`` isolating tenants.
- Client suffixes are stripped, ``|``-sanitized, and bounded so a crafted
  ``session_id`` cannot escape its HR namespace.
- Email is never part of the memory key.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "DEFAULT_CONVERSATION_ID",
    "normalize_session_suffix",
    "conversation_key",
    "find_claim",
    "parse_expiry",
    "token_fingerprint",
    "cache_fingerprint",
    "normalize_employee_records",
    "employee_hr_matches",
]

DEFAULT_CONVERSATION_ID = "default"


def normalize_session_suffix(value: str | None) -> str:
    """Client-visible suffix usable as-is with the stop endpoint."""
    suffix = (value or DEFAULT_CONVERSATION_ID).strip() or DEFAULT_CONVERSATION_ID
    return suffix.replace("|", "_")[:120]


def conversation_key(hr_code: str, suffix: str | None) -> str:
    """HR-scoped memory key. Email is never part of the key."""
    return f"{hr_code}|{normalize_session_suffix(suffix)}"


def find_claim(payload: dict[str, Any], name: str) -> Any:
    """Case-insensitive claim lookup."""
    if name in payload:
        return payload[name]
    lowered = name.lower()
    for key, value in payload.items():
        if key.lower() == lowered:
            return value
    return None


#: .NET `DateTime.ToString("o")` emits seven fractional digits
#: ("2026-08-01T10:00:00.1234567Z"). `datetime.fromisoformat` accepts three or
#: six and raises on seven, so the tail is trimmed before parsing.
_ISO_FRACTION_RE = re.compile(r"\.(\d+)")


def parse_expiry(value: Any) -> datetime | None:
    """Accept epoch seconds or ISO-8601; UTC-aware or None when unreadable."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.isdigit():
        try:
            return datetime.fromtimestamp(float(text), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    match = _ISO_FRACTION_RE.search(text)
    if match and len(match.group(1)) > 6:
        text = text[: match.start(1) + 6] + text[match.end(1):]
    try:
        parsed = datetime.fromisoformat(text)
    except (OverflowError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def token_fingerprint(token: str, length: int = 10) -> str:
    """Stable, non-reversible log handle for a bearer token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:length]


def cache_fingerprint(token: str | None) -> str | None:
    """Bind cached credentials to a login without retaining the raw token."""
    if not token:
        return None
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]


def normalize_employee_records(body: Any) -> list[dict[str, Any]]:
    """Normalize the employee lookup envelope to a record list."""
    records = body.get("data", []) if isinstance(body, dict) else []
    if isinstance(records, dict):
        records = [records]
    return [r for r in records if isinstance(r, dict)]


def employee_hr_matches(records: list[dict[str, Any]], hr_code: str) -> bool:
    """True when the first record's HR code matches ``hr_code``."""
    if not records:
        return False
    employee_hr = next(
        (v for k, v in records[0].items() if k.lower() in {"user_hr_code", "userhrcode"}),
        None,
    )
    return str(employee_hr).strip() == hr_code
