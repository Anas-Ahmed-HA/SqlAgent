"""Pydantic models for the public HTTP request and response shapes."""

from __future__ import annotations

from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from shared.api_contract import CONTRACT_VERSION

__all__ = [
    "ChatMessage", "ChatRequest", "ChatResponse", "ChatStopRequest", "EventTicketRequest",
    "EventTicketResponse", "HealthResponse",
    "ToolsResponse", "SkillProposalRequest", "SkillProposalResponse", "SkippedSkill",
    "TurnSkillPayload",
]


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    #: On assistant messages: the turn trace returned with that reply
    #: (`ChatResponse.trace`). Its tool calls are replayed before the reply.
    trace: dict | None = None


class TurnSkillPayload(BaseModel):
    """One equipped skill sent with a chat turn (``skills_hub.TurnSkill``).

    The limits match the cloud skills API's own.
    """

    id: UUID
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=500)
    instructions: list[Annotated[str, Field(min_length=1, max_length=1000)]] = Field(
        min_length=1, max_length=20
    )
    version: int = Field(ge=1)


class SkippedSkill(BaseModel):
    """A skill the turn brought but did not use, and why."""

    id: UUID
    reason: Literal["unsafe", "over_budget"]


class ChatRequest(BaseModel):
    message: str
    #: Earlier turns, when the client keeps the conversation. Without it the
    #: server's memory of the session is used.
    history: list[ChatMessage] | None = None
    session_id: str | None = None
    turn_id: str | None = None
    #: The user's equipped skills for this turn, in order. The client owns
    #: skills (the cloud skills API) and sends them with every turn; the agent
    #: keeps none. Missing or empty means no skills.
    skills: list[TurnSkillPayload] | None = Field(default=None, max_length=50)
    #: The client follows the turn on CHAT_EVENTS and answers the assistant's
    #: questions there (CAPABILITY_QUESTIONS). Without it the assistant never
    #: asks: nobody could answer.
    interactive: bool = False


class EventTicketRequest(BaseModel):
    turn_id: str = Field(min_length=1, max_length=100)
    """The turn whose events the ticket opens."""


class EventTicketResponse(BaseModel):
    ticket: str
    """Opens ``/chat/events/{turn_id}?ticket=...`` once, for that turn only."""
    expires_in: int
    """Seconds left to open the socket with it."""


class ChatStopRequest(BaseModel):
    session_id: str | None = None
    turn_id: str | None = None


class ChatResponse(BaseModel):
    messages: list[dict]
    reply: str | None = None
    """The assistant's answer."""
    raw_reply: str | None = None
    """Same as `reply`; kept for clients of the original contract."""
    session_id: str = "default"
    """Echo of the client session id, usable as-is with `/chat/stop`."""
    turn_id: str = ""
    """Identifier for this queued turn, usable with `/chat/stop`."""
    context: dict | None = None
    """How full the model's context was on the turn's last call:
    `used_tokens`, `budget_tokens` (AGENT_CONTEXT_TOKEN_BUDGET) and what
    compaction removed (`shortened_outputs`, `dropped_history_messages`,
    `dropped_turn_messages`). Null when the turn made no model call."""
    trace: dict | None = None
    """This turn's tool calls as spans (`v`, `turn_id`, `spans`), or null when
    it made none. Opaque to clients: a client stores it with the reply and
    sends it back on that message in `history`."""
    skills_used: list[UUID] = Field(default_factory=list)
    """Ids of the request's skills this turn put in front of the model, in order."""
    skills_dropped: list[SkippedSkill] = Field(default_factory=list)
    """The request's skills left out: unsafe text, or past the equipped limits."""


class HealthResponse(BaseModel):
    status: str
    """One of ``shared.api_contract.HEALTH_STATUSES``."""
    #: ``shared.api_contract.CONTRACT_VERSION`` of the routes and bodies this
    #: backend serves; a client uses only a backend reporting its own version.
    contract: int = CONTRACT_VERSION
    #: Optional features this backend serves (``shared.api_contract.CAPABILITIES``).
    capabilities: list[str] = Field(default_factory=list)
    mcp_session: bool
    """Whether the MCP server at ``mcp_url`` is connected."""
    mcp_url: str = ""
    detail: str = ""
    #: Where the Azure credentials came from: "lazy" (not yet fetched) or
    #: "aiapi-gateway" (fetched on the first chat with the user's token).
    creds_source: str = "lazy"
    #: When the last credentials fetch failed, why.
    creds_error: str = ""


class ToolsResponse(BaseModel):
    count: int
    #: The MCP server's tools, in its order.
    tools: list[dict[str, Any]] = Field(default_factory=list)


class SkillProposalRequest(BaseModel):
    source_text: str = Field(min_length=1, max_length=8000)


class SkillProposalResponse(BaseModel):
    #: ``name``, ``description`` and ``instructions``, or for unclear input
    #: ``unclear`` (true) and ``clarification``. Nothing is kept: the client
    #: saves the proposal to the skills API itself.
    proposal: dict[str, Any]
