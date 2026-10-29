"""Direct-process runner for the FastAPI application."""

from __future__ import annotations

import socket
import sys
from typing import Any

from shared.settings import LOCAL_API_HOST, LOCAL_API_PORT


def run(app: Any, *, host: str = LOCAL_API_HOST, port: int = LOCAL_API_PORT) -> None:
    """Run uvicorn, first saying plainly when the port is already taken."""
    import uvicorn

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host, port))
    except OSError as exc:
        sys.exit(
            f"Port {port} on {host} could not be bound: {exc}\n"
            "Another instance may be running; set AGENT_PORT to use another port."
        )
    finally:
        probe.close()

    try:
        uvicorn.run(app, host=host, port=port)
    except KeyboardInterrupt:
        pass
