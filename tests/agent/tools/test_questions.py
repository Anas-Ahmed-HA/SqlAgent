"""``ask_user``: offered only to an answering client, checked, capped per turn,
and its waiting time given back to the turn's deadline."""

from __future__ import annotations

import asyncio

import pytest

from agent.host.interactions import Answer, AskOutcome
from agent.tools.base import LocalCall
from agent.tools.questions import ASK_USER_TOOL_NAME, AskUserTool
from agent.observability.turn_trace import build_turn_trace
from shared.settings import MAX_ASKS_PER_TURN, MAX_QUESTIONS_PER_ASK, MAX_QUESTIONS_PER_TURN


class FakeInteractions:
    def __init__(self, outcome: AskOutcome) -> None:
        self.outcome = outcome
        self.asked: list = []

    async def ask(self, questions, *, wait_seconds, reconnect_seconds):
        self.asked.append(questions)
        return self.outcome


def _question(**over):
    return {"question": "Which period?", "options": [{"label": "This year"}, {"label": "All time"}], **over}


def _run(tool, arguments, **state):
    state = {"interactive": True, "questions_asked": 0, "deadline": 1000.0, **state}
    return asyncio.run(tool.run(LocalCall(ASK_USER_TOOL_NAME, arguments, "call-1", state)))


def test_offered_only_when_the_client_answers() -> None:
    tool = AskUserTool(FakeInteractions(AskOutcome("answered")))
    assert tool.handles("ask_user") and not tool.handles("run_select")
    assert tool.definitions({"interactive": False}) == []
    [definition] = tool.definitions({"interactive": True})
    schema = definition["function"]["parameters"]["properties"]["questions"]
    assert schema["minItems"] == 1 and schema["maxItems"] == MAX_QUESTIONS_PER_ASK


def test_answers_reach_the_model_and_the_wait_is_not_counted() -> None:
    fake = FakeInteractions(AskOutcome("answered", [Answer(["All time"]), Answer([], "Both")], waited_seconds=42.0))
    result = _run(AskUserTool(fake), {"questions": [
        _question(), _question(question="Where to?", allow_other=True, multi_select=True),
    ]})
    assert not result.is_error
    assert '- Which period? -> "All time"' in result.text
    assert '- Where to? -> wrote "Both"' in result.text
    assert result.update == {"questions_asked": 1, "questions_total": 2, "deadline": 1042.0}
    sent = fake.asked[0]
    assert sent[0] == {"question": "Which period?", "multi_select": False, "allow_other": False,
                       "options": [{"label": "This year", "description": "", "recommended": False},
                                   {"label": "All time", "description": "", "recommended": False}]}
    assert sent[1]["multi_select"] is True and sent[1]["allow_other"] is True
    interaction = result.data["interaction"]
    assert interaction["reason"] == "answered"
    assert interaction["answers"] == [{"choices": ["All time"], "other": None}, {"choices": [], "other": "Both"}]


@pytest.mark.parametrize("arguments, message", [
    ({}, "`questions` must list 1 to"),
    ({"questions": []}, "`questions` must list 1 to"),
    ({"questions": [_question()] * (MAX_QUESTIONS_PER_ASK + 1)}, "`questions` must list 1 to"),
    ({"questions": [_question(question=" ")]}, "Question 1 needs a non-empty `question`"),
    ({"questions": [_question(options=[{"label": "Only"}])]}, "Question 1: `options` must list 2 to 4"),
    ({"questions": [_question(), _question(options=[{"label": "A"}, {"label": "a"}])]}, "Question 2: two choices have the same label"),
    ({"questions": [_question(options=[{"label": "A"}, {"label": "x" * 61}])]}, "at most 60 characters"),
])
def test_bad_questions_are_sent_back_to_the_model(arguments, message) -> None:
    fake = FakeInteractions(AskOutcome("answered"))
    result = _run(AskUserTool(fake), arguments)
    assert result.is_error and result.error_kind == "validation" and message in result.text
    assert fake.asked == []


def test_a_turn_asks_at_most_the_limit() -> None:
    fake = FakeInteractions(AskOutcome("answered", [Answer(["This year"])]))
    result = _run(AskUserTool(fake), {"questions": [_question()]}, questions_asked=MAX_ASKS_PER_TURN)
    assert result.is_error and "the most allowed" in result.text and fake.asked == []


def test_a_turn_asks_at_most_the_total_number_of_questions() -> None:
    fake = FakeInteractions(AskOutcome("answered", [Answer(["This year"])]))
    tool = AskUserTool(fake)
    spent = _run(tool, {"questions": [_question()]}, questions_total=MAX_QUESTIONS_PER_TURN)
    assert spent.is_error and f"already asked {MAX_QUESTIONS_PER_TURN} questions" in spent.text
    too_many = _run(tool, {"questions": [_question(), _question(question="Where?")]},
                    questions_total=MAX_QUESTIONS_PER_TURN - 1)
    assert too_many.is_error and "Only 1 more question(s)" in too_many.text
    assert fake.asked == []
    fits = _run(tool, {"questions": [_question()]}, questions_total=MAX_QUESTIONS_PER_TURN - 1)
    assert not fits.is_error and fits.update["questions_total"] == MAX_QUESTIONS_PER_TURN


def test_the_recommended_choice_is_a_flag_not_label_text() -> None:
    fake = FakeInteractions(AskOutcome("answered", [Answer(["All time"]), Answer(["A", "B"])]))
    _run(AskUserTool(fake), {"questions": [
        _question(options=[{"label": "This year (Recommended)"}, {"label": "All time", "recommended": True}]),
        _question(question="Which?", multi_select=True,
                  options=[{"label": "A", "recommended": True}, {"label": "B [recommended]"}, {"label": "C"}]),
    ]})
    single, multi = fake.asked[0]
    # A single-choice question keeps one recommendation: the first marked.
    assert [(o["label"], o["recommended"]) for o in single["options"]] == [("This year", True), ("All time", False)]
    assert [(o["label"], o["recommended"]) for o in multi["options"]] == [("A", True), ("B", True), ("C", False)]


def test_no_answer_tells_the_model_to_go_on() -> None:
    for reason, wording in [("skipped", "skipped the questions"), ("timeout", "did not answer in time"),
                            ("disconnected", "did not answer in time"), ("unavailable", "Nobody can answer")]:
        result = _run(AskUserTool(FakeInteractions(AskOutcome(reason))), {"questions": [_question()]})
        assert wording in result.text and not result.is_error
        assert result.data["interaction"]["answers"] is None
    refused = _run(AskUserTool(FakeInteractions(AskOutcome("answered"))), {"questions": [_question()]}, interactive=False)
    assert refused.is_error and "nobody can answer" in refused.text
    limit = _run(AskUserTool(FakeInteractions(AskOutcome("answered"))), {"questions": [_question()]},
                 questions_asked=MAX_ASKS_PER_TURN)
    assert limit.is_error and "Do not ask again" in limit.text


def test_the_stored_trace_keeps_the_questions_and_answers() -> None:
    results = {"call-1": {
        "outcome": "success", "ui_status": "Asking you a question",
        "interaction": {"questions": [_question()], "reason": "skipped", "answers": None},
    }}
    messages = [
        {"role": "user", "content": "count laptops"},
        {"role": "assistant", "tool_calls": [{"id": "call-1", "function": {"name": "ask_user", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "The user skipped the questions."},
    ]
    span = build_turn_trace(messages, results, "turn-1")["spans"][0]
    assert span["interaction"]["reason"] == "skipped"
    assert span["ui_status"] == "Asking you a question"
