"""
Structured logging.

The brief asks to log node execution, planner decisions, tool selection,
validation, retries, timings, reflection and the final response -- while
avoiding excessive logging. Those pull in opposite directions if every event is
a formatted sentence, so events here are *structured*: a short dotted event
name plus fields. One line per event, greppable by `event`, and cheap enough
that leaving it on in production is reasonable.

`request_id` is attached to every record via a `contextvars`-backed filter, so
one request's trace can be isolated from concurrent traffic without threading a
logger object through fourteen node signatures.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

__all__ = ["configure_logging", "node_timer", "request_context"]

_request_id: ContextVar[str] = ContextVar("request_id", default="-")

#: Attributes present on every LogRecord; anything else is a caller-supplied
#: field and gets serialised into the structured payload.
_STANDARD = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"asctime", "message", "taskName"}


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


class KeyValueFormatter(logging.Formatter):
    """`2026-08-01T10:00:00 INFO sql_agent.loop turn.end req=abc key=value`

    Chosen over pure JSON because these logs are read live in a terminal on the
    developer machine, where JSON is unreadable. Set `json_output=True` for a
    log shipper.
    """

    def __init__(self, json_output: bool = False) -> None:
        super().__init__()
        self.json_output = json_output

    def format(self, record: logging.LogRecord) -> str:
        extras: dict[str, Any] = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD and key != "request_id"
        }
        message = record.getMessage()

        if self.json_output:
            payload = {
                "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                "level": record.levelname,
                "logger": record.name,
                "event": message,
                "request_id": getattr(record, "request_id", "-"),
                **extras,
            }
            if record.exc_info:
                payload["exc"] = self.formatException(record.exc_info)
            return json.dumps(payload, default=str)

        fields = " ".join(f"{k}={_render(v)}" for k, v in extras.items())
        line = (
            f"{self.formatTime(record, '%H:%M:%S')} "
            f"{record.levelname:<7} {record.name:<26} {message}"
            f" req={getattr(record, 'request_id', '-')}"
        )
        if fields:
            line += " " + fields
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def _render(value: Any) -> str:
    text = str(value)
    if len(text) > 160:
        text = text[:157] + "..."
    return f'"{text}"' if " " in text else text


def configure_logging(level: int = logging.INFO, json_output: bool = False) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(KeyValueFormatter(json_output=json_output))
    handler.addFilter(_RequestIdFilter())

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)

    # These are chatty at INFO and say nothing about our workflow.
    for noisy in ("httpx", "httpcore", "openai._base_client", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@contextmanager
def request_context(request_id: str):
    token = _request_id.set(request_id)
    try:
        yield
    finally:
        _request_id.reset(token)


@contextmanager
def node_timer(logger: logging.Logger, node: str, **fields: Any):
    """Times a node and logs one line on the way out.

    Entry is deliberately not logged: with fourteen nodes per tool call,
    entry+exit doubles log volume to say nothing an exit line does not.
    """
    started = time.perf_counter()
    try:
        yield
    finally:
        logger.info(
            "node.done",
            extra={
                "node": node,
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
                **fields,
            },
        )
