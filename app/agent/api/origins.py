"""Which web pages may open this service's WebSockets.

CORS does not apply to WebSockets: any page can open one to any server, and
the browser sends that page's ``Origin`` with the handshake. So the socket
checks it itself. A client that is not a browser (a script, another service)
sends no ``Origin`` and is not affected; it still needs a ticket or the
bearer token.

A page this service served itself (the test chat page at ``/``) is always
allowed: its ``Origin`` names the same host and port the handshake was sent
to. ``AGENT_ALLOWED_ORIGINS_REGEX`` names other allowed origins, matched
against the whole ``Origin`` value, e.g. ``https://([a-z0-9-]+\\.)*example\\.com``.
Unset, no other page may open a socket. Read on every call, so a change
needs no restart and tests can set it.
"""

from __future__ import annotations

import logging
import os
import re
from urllib.parse import urlsplit

__all__ = ["ALLOWED_ORIGINS_ENV", "origin_allowed"]

ALLOWED_ORIGINS_ENV = "AGENT_ALLOWED_ORIGINS_REGEX"

logger = logging.getLogger("sql_agent.api")


def origin_allowed(origin: str | None, host: str | None = None) -> bool:
    """Whether a WebSocket handshake carrying ``origin`` may go on.

    ``host`` is the handshake's ``Host`` header. A forged ``Host`` gains a page
    nothing: it still needs a ticket, which only the bearer token gets.
    """
    if origin is None:
        return True
    origin = origin.strip()
    if host and _same_host(origin, host):
        return True
    pattern = (os.environ.get(ALLOWED_ORIGINS_ENV) or "").strip()
    if not pattern:
        return False
    try:
        return re.fullmatch(pattern, origin) is not None
    except re.error:
        logger.warning("origins.bad_pattern", extra={"env": ALLOWED_ORIGINS_ENV})
        return False


def _same_host(origin: str, host: str) -> bool:
    # The scheme is left out: behind a proxy that ends TLS the page is https
    # while the socket arrives here as plain ws.
    try:
        parts = urlsplit(origin)
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and bool(parts.netloc) and parts.netloc.lower() == host.strip().lower()
