"""``show_cards`` and ``ask_with_card``: the shared cards (``agent_cards``) in this loop.

The package owns the cards, their tool definitions and checks, and the prompt
rules; this module only fits them to :class:`LocalTool`. A turn's card types
are this agent's that the client draws (``ChatRequest.cards``), in
``state["card_types"]``; the cards shown so far are ``state["cards"]`` and
go out as ``ChatResponse.blocks``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent_cards import TOOL_NAMES, run_tool, tool_definitions

from agent.tools.base import LocalCall, LocalResult, LocalTool

__all__ = ["SQL_AGENT_CARD_TYPES", "CardsTool"]

#: The cards that fit answers about company data. No emails, files or
#: pictures: the data service has none. No confirm: nothing here changes data.
#: A form gathers a report's parameters (a period, a department).
SQL_AGENT_CARD_TYPES = ("people", "items", "events", "stats", "links", "timeline", "date", "callout", "form")


class CardsTool(LocalTool):
    def handles(self, name: str) -> bool:
        return name in TOOL_NAMES

    def definitions(self, state: Mapping[str, Any]) -> list[dict[str, Any]]:
        return tool_definitions(state.get("card_types") or ())

    async def run(self, call: LocalCall) -> LocalResult:
        shown = list(call.state.get("cards") or ())
        outcome = run_tool(call.name, call.arguments, types=call.state.get("card_types") or (), shown=shown)
        if outcome.is_error:
            return LocalResult.validation_error(outcome.text)
        return LocalResult(text=outcome.text, update={"cards": [*shown, *outcome.cards]})
