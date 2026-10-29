"""``ask_user``: the assistant asks the user one or more questions with choices.

Offered only in a turn whose client answers questions (``state["interactive"]``,
from ``ChatRequest.interactive``). One call holds 1 to ``MAX_QUESTIONS_PER_ASK``
questions, each with 2-4 choices; a choice may be marked recommended. Running it
publishes them on the turn's event socket and waits until the user answers them
all (:mod:`agent.host.interactions`); the model reads the answers as the tool
result and carries on in the same turn.

The time spent waiting is given back to the turn's deadline, so a user who
takes a minute to decide does not cost the assistant a minute of work. A turn
may ask at most ``MAX_ASKS_PER_TURN`` times and ``MAX_QUESTIONS_PER_TURN``
questions in all; after that the tool refuses and the model goes on with its
best judgement.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any

from agent.host.interactions import AskOutcome, Interactions, interactions as default_interactions
from agent.tools.base import LocalCall, LocalResult, LocalTool
from shared.settings import (
    MAX_ASKS_PER_TURN,
    MAX_QUESTIONS_PER_ASK,
    MAX_QUESTIONS_PER_TURN,
    QUESTION_RECONNECT_SECONDS,
    QUESTION_WAIT_SECONDS,
)

__all__ = ["ASK_USER_TOOL_NAME", "AskUserTool"]

logger = logging.getLogger("sql_agent.tools")

ASK_USER_TOOL_NAME = "ask_user"

MAX_QUESTION_CHARS = 300
MAX_LABEL_CHARS = 60
MAX_DESCRIPTION_CHARS = 160
MIN_OPTIONS = 2
MAX_OPTIONS = 4

# "(Recommended)" written into a label anyway: taken out and kept as the flag.
_RECOMMENDED_SUFFIX = re.compile(r"\s*[(\[]\s*(?:recommended|موصى به)\s*[)\]]\s*$", re.IGNORECASE)

_QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": "The question, one sentence, ending with a question mark.",
            "maxLength": MAX_QUESTION_CHARS,
        },
        "options": {
            "type": "array",
            "minItems": MIN_OPTIONS,
            "maxItems": MAX_OPTIONS,
            "items": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "The choice, a few words.",
                        "maxLength": MAX_LABEL_CHARS,
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional: what choosing it means, one short line.",
                        "maxLength": MAX_DESCRIPTION_CHARS,
                    },
                    "recommended": {
                        "type": "boolean",
                        "description": "True on the choice you recommend; the app marks it. "
                                       "Do not write \"(Recommended)\" in the label.",
                    },
                },
                "required": ["label"],
            },
        },
        "multi_select": {
            "type": "boolean",
            "description": "True when the user may pick several choices.",
        },
        "allow_other": {
            "type": "boolean",
            "description": "True to let the user type an answer of their own instead.",
        },
    },
    "required": ["question", "options"],
}

_DEFINITION = {
    "type": "function",
    "function": {
        "name": ASK_USER_TOOL_NAME,
        "description": (
            f"Ask the user 1 to {MAX_QUESTIONS_PER_ASK} questions, each with 2-4 choices, "
            "and wait until they answer them all; then continue this same reply with the "
            "answers. The questions replace the user's message box until they answer.\n"
            "Use it when the request leaves open a choice that changes the answer and you "
            "cannot find out yourself: which database, which of several matching views or "
            "columns, which period, which definition of a measure (for example, count "
            "assets or sum their value), which grouping. Ask with this tool, never by "
            "ending your reply with a question in plain text.\n"
            "But keep it short. Look first: never ask what the business rules, the object "
            "list or a quick query could tell you. Do not ask about details that have a "
            "sensible default (sort order, how many rows to show, formatting): choose them "
            "yourself and say so in your reply. Never ask for passwords, tokens or other "
            "credentials, and never ask just to confirm what the user already said.\n"
            "Put every question you need into ONE call; a second call is only for a new "
            "choice that the answers opened up. At most "
            f"{MAX_ASKS_PER_TURN} calls and {MAX_QUESTIONS_PER_TURN} questions per "
            "reply; most replies need none. Call it in a step of its own (no other tool "
            "calls with it). Write the questions and choices in the user's language. Put "
            "the choice you recommend first and set `recommended` on it.\n"
            "If the user skips or does not answer, go on with the most reasonable choices "
            "and say which ones you assumed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_QUESTIONS_PER_ASK,
                    "items": _QUESTION_SCHEMA,
                    "description": "The questions, in the order the user should answer them.",
                },
            },
            "required": ["questions"],
        },
    },
}


def _clean_question(raw: Any, number: int) -> dict[str, Any] | str:
    """One question as sent to the client, or what is wrong with it."""
    where = f"Question {number}"
    if not isinstance(raw, dict):
        return f"{where} must be an object with `question` and `options`."
    text = raw.get("question")
    if not isinstance(text, str) or not text.strip():
        return f"{where} needs a non-empty `question`."
    text = " ".join(text.split())
    if len(text) > MAX_QUESTION_CHARS:
        return f"{where}: `question` is at most {MAX_QUESTION_CHARS} characters; shorten it."
    options = raw.get("options")
    if not isinstance(options, list) or not MIN_OPTIONS <= len(options) <= MAX_OPTIONS:
        return f"{where}: `options` must list {MIN_OPTIONS} to {MAX_OPTIONS} choices."
    multi_select = raw.get("multi_select") is True
    cleaned: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in options:
        label = item.get("label") if isinstance(item, dict) else item
        if not isinstance(label, str) or not label.strip():
            return f"{where}: every choice needs a non-empty `label`."
        label = " ".join(label.split())
        marked = bool(_RECOMMENDED_SUFFIX.search(label))
        label = _RECOMMENDED_SUFFIX.sub("", label) or label
        if len(label) > MAX_LABEL_CHARS:
            return f"{where}: a choice label is at most {MAX_LABEL_CHARS} characters: `{label[:40]}...`."
        if label.casefold() in seen:
            return f"{where}: two choices have the same label, `{label}`."
        seen.add(label.casefold())
        description = item.get("description") if isinstance(item, dict) else None
        description = " ".join(description.split())[:MAX_DESCRIPTION_CHARS] if isinstance(description, str) else ""
        recommended = marked or (isinstance(item, dict) and item.get("recommended") is True)
        # One recommendation on a single-choice question: the first one marked.
        if recommended and not multi_select and any(o["recommended"] for o in cleaned):
            recommended = False
        cleaned.append({"label": label, "description": description, "recommended": recommended})
    return {
        "question": text,
        "options": cleaned,
        "multi_select": multi_select,
        "allow_other": raw.get("allow_other") is True,
    }


def _clean_questions(raw: Any) -> list[dict[str, Any]] | str:
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_QUESTIONS_PER_ASK:
        return f"`questions` must list 1 to {MAX_QUESTIONS_PER_ASK} questions."
    questions = []
    for number, item in enumerate(raw, start=1):
        question = _clean_question(item, number)
        if isinstance(question, str):
            return question
        questions.append(question)
    return questions


class AskUserTool(LocalTool):
    def __init__(self, registry: Interactions | None = None) -> None:
        self._interactions = registry or default_interactions

    def handles(self, name: str) -> bool:
        return name == ASK_USER_TOOL_NAME

    def definitions(self, state: Mapping[str, Any]) -> list[dict[str, Any]]:
        # Offered for the whole turn or not at all, so the tool list (part of
        # the cached prompt prefix) does not change mid-turn.
        return [_DEFINITION] if state.get("interactive") else []

    async def run(self, call: LocalCall) -> LocalResult:
        state = call.state
        if not state.get("interactive"):
            return LocalResult.validation_error(
                "`ask_user` is not available in this reply: nobody can answer. Go on with "
                "the most reasonable choices and say which ones you assumed."
            )
        asked = int(state.get("questions_asked") or 0)
        total = int(state.get("questions_total") or 0)
        left = MAX_QUESTIONS_PER_TURN - total
        if asked >= MAX_ASKS_PER_TURN or left <= 0:
            return LocalResult.validation_error(
                f"You already asked {total} questions in {asked} calls in this reply, the "
                "most allowed. Do not ask again: go on with the most reasonable choices and "
                "say which ones you assumed."
            )
        questions = _clean_questions(call.arguments.get("questions"))
        if isinstance(questions, str):
            return LocalResult.validation_error(questions + " Re-emit the call.")
        if len(questions) > left:
            return LocalResult.validation_error(
                f"Only {left} more question(s) may be asked in this reply. Keep the ones that "
                "matter most, choose the rest yourself, and re-emit the call."
            )

        outcome = await self._interactions.ask(
            questions, wait_seconds=QUESTION_WAIT_SECONDS,
            reconnect_seconds=QUESTION_RECONNECT_SECONDS,
        )
        update: dict[str, Any] = {"questions_asked": asked + 1, "questions_total": total + len(questions)}
        # Waiting for the user is not the assistant's working time.
        deadline = state.get("deadline")
        if isinstance(deadline, (int, float)) and outcome.waited_seconds > 0:
            update["deadline"] = deadline + outcome.waited_seconds
        interaction = {
            "questions": questions, "reason": outcome.reason,
            "answers": outcome.answers_payload(),
        }
        return LocalResult(
            text=_result_text(questions, outcome), data={"interaction": interaction}, update=update,
        )


def _result_text(questions: list[dict[str, Any]], outcome: AskOutcome) -> str:
    if outcome.answered:
        lines = ["The user answered:"]
        for question, answer in zip(questions, outcome.answers):
            parts = []
            if answer.choices:
                parts.append("; ".join(f'"{choice}"' for choice in answer.choices))
            if answer.other:
                parts.append(f'wrote "{answer.other}"')
            lines.append(f'- {question["question"]} -> {", ".join(parts)}')
        lines.append("Continue with these answers.")
        return "\n".join(lines)
    if outcome.reason == "skipped":
        return ("The user skipped the questions. Go on with the most reasonable choices and "
                "say which ones you assumed; do not ask any more questions in this reply.")
    if outcome.reason == "unavailable":
        return ("Nobody can answer questions in this reply. Go on with the most reasonable "
                "choices and say which ones you assumed.")
    return ("The user did not answer in time. Go on with the most reasonable choices and say "
            "which ones you assumed, or end your reply by asking the questions in plain "
            "text if no choice is safe.")
