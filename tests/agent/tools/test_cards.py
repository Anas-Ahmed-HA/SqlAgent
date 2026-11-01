"""The shared cards in this loop (agent/tools/cards.py), end to end over the fakes."""

from __future__ import annotations

import asyncio

from agent.memory.history import validate_message_sequence
from agent.observability.turn_trace import build_turn_trace
from tests.agent.fakes import FakeAzure, make_loop, text_turn, tool_turn

WEBSITE_CARDS = ["people", "items", "emails", "events", "files", "images", "stats", "links",
                 "timeline", "date", "callout", "form", "confirm"]
STATS = {"type": "stats", "title": "Laptops", "items": [{"label": "In use", "value": 3679, "delta": "+4%"}]}


def run(azure, **kwargs):
    return asyncio.run(make_loop(azure).run(user_message="How many laptops?", session_id="HR-1|s", **kwargs))


def offered(call) -> set[str]:
    return {t["function"]["name"] for t in call["tools"]}


def system_text(call) -> str:
    return "\n".join(m["content"] for m in call["messages"] if m["role"] == "system")


def test_a_client_that_draws_no_cards_gets_no_card_tools():
    azure = FakeAzure([text_turn("3,679.")])
    state = run(azure)
    assert not offered(azure.calls[0]) & {"show_cards", "ask_with_card"}
    assert "CARDS:" not in system_text(azure.calls[0])
    assert state["cards"] == []


def test_the_turn_offers_the_cards_both_this_agent_and_the_client_have():
    azure = FakeAzure([text_turn("3,679.")])
    run(azure, card_types=WEBSITE_CARDS)
    assert {"show_cards", "ask_with_card"} <= offered(azure.calls[0])
    rules = system_text(azure.calls[0])
    assert "- stats:" in rules and "- form:" in rules
    # Not this agent's: the data service has no mail, and nothing here changes data.
    assert "- emails:" not in rules and "- confirm:" not in rules

    azure = FakeAzure([text_turn("3,679.")])
    run(azure, card_types=["stats", "emails"])
    assert "ask_with_card" not in offered(azure.calls[0])


def test_shown_cards_go_out_with_the_reply_and_stay_in_the_trace():
    azure = FakeAzure([
        tool_turn(("c1", "show_cards", {"cards": [STATS], "ui_status": "Showing the totals"})),
        text_turn("Most are in Engineering."),
    ])
    state = run(azure, card_types=WEBSITE_CARDS)
    assert state["final_response"] == "Most are in Engineering."
    assert state["cards"] == [STATS]
    assert validate_message_sequence(state["messages"]) == []
    trace = build_turn_trace(state["messages"], state["tool_results"], "t1", state["planning_ms"])
    assert trace["spans"][0]["tool"] == "show_cards"
    assert trace["spans"][0]["ui_status"] == "Showing the totals"


def test_a_bad_card_is_sent_back_to_be_fixed_and_shows_nothing():
    bad = {"type": "events", "items": [{"title": "Audit", "start": "next week"}]}
    azure = FakeAzure([tool_turn(("c1", "show_cards", {"cards": [bad]})), text_turn("The audit is next week.")])
    state = run(azure, card_types=WEBSITE_CARDS)
    assert state["cards"] == []
    result = state["tool_results"]["c1"]
    assert result["outcome"] == "failure" and result["error_kind"] == "validation"
    tool_text = next(m["content"] for m in state["messages"] if m.get("role") == "tool")
    assert "cards[0] (events).items[0].start" in tool_text


def test_a_form_gets_an_id_the_answer_comes_back_with():
    form = {"type": "form", "title": "Report period", "fields": [
        {"name": "start", "label": "From", "kind": "date", "required": True},
        {"name": "department", "label": "Department", "kind": "select", "options": ["IT", "HR"]},
    ]}
    azure = FakeAzure([tool_turn(("c1", "ask_with_card", {"card": form})), text_turn("Pick the period below.")])
    state = run(azure, card_types=WEBSITE_CARDS)
    card = state["cards"][0]
    assert card["type"] == "form" and card["id"].startswith("form-")

    azure = FakeAzure([text_turn("Done.")])
    run(azure, card_types=WEBSITE_CARDS, form_response={
        "block_id": card["id"], "action": "submit", "values": {"start": "2026-10-01", "department": "IT"},
    })
    sent = azure.calls[0]["messages"]
    note = sent[-2]
    assert note["role"] == "system" and note["content"].startswith("CARD ANSWER")
    assert card["id"] in note["content"] and '"department": "IT"' in note["content"]
    assert sent[-1]["role"] == "user"


def test_a_malformed_answer_is_ignored():
    azure = FakeAzure([text_turn("Done.")])
    run(azure, card_types=WEBSITE_CARDS, form_response={"action": "maybe"})
    assert not any(m["content"].startswith("CARD ANSWER") for m in azure.calls[0]["messages"])
