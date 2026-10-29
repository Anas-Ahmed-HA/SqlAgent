"""
Error taxonomy for the SQL agent.

Two kinds matter to the loop: a tool error the model can fix (bad SQL, an
unknown column), which goes back to it as the tool message, and an
environmental failure (the MCP server is unreachable), which ends the turn
with a concrete message instead of more model calls.
"""

from __future__ import annotations

from enum import Enum


class ErrorCategory(str, Enum):
    MCP_CONNECTION = "MCPConnectionError"
    TOOL_EXECUTION = "ToolExecutionError"
    TIMEOUT = "TimeoutError"
    PLANNING = "PlanningError"
    UNEXPECTED = "UnexpectedError"


class AgentException(Exception):
    category: ErrorCategory = ErrorCategory.UNEXPECTED


class MCPConnectionException(AgentException):
    category = ErrorCategory.MCP_CONNECTION


class ToolTimeoutException(AgentException):
    """A tool call exceeded its deadline without a result."""

    category = ErrorCategory.TIMEOUT


class PlanningException(AgentException):
    category = ErrorCategory.PLANNING


class ContextTooLongException(PlanningException):
    """The model rejected the request as larger than its context window."""


class CredentialsStaleException(PlanningException):
    """Azure rejected the cached gateway-issued credentials."""
