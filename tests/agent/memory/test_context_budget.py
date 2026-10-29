"""Context budget: what the loop sends when the conversation is large."""

from __future__ import annotations

import asyncio

from agent.config import AgentConfig
from agent.errors import ContextTooLongException
from agent.memory.context_budget import (
    LOW_WATERMARK,
    MAX_OUTPUT_CHARS,
    SHORTENED_OUTPUT_CHARS,
    estimate_tokens,
    fit_context,
)
from agent.memory.history import validate_message_sequence
from agent.clients.azure import is_context_too_long
from tests.agent.fakes import FakeAzure, FakeMCP, text_turn, tool_turn

RULES = {"role": "system", "content": "Harness rules."}
DOC_CONTEXT = {"role": "system", "content": "An uploaded document is available: spec.pdf."}
REQUEST = {"role": "user", "content": "Tag every wall on Level 2."}
from tests.agent.fakes import make_loop


def build(azure, mcp, **config):
    return make_loop(azure, mcp, **config)


def run(loop, message, history=None):
    return asyncio.run(loop.run(user_message=message, history=history))



def step(call_id: str, output: str) -> list[dict]:
    call = {"id": call_id, "type": "function",
            "function": {"name": "get_current_view_elements", "arguments": "{}"}}
    return [{"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call_id, "content": output}]


def history_turn(i: int, size: int = 40) -> list[dict]:
    return [{"role": "user", "content": f"question {i} " + "q" * size},
            {"role": "assistant", "content": f"answer {i} " + "a" * size}]


def conversation(history_turns: int, steps: int, output_chars: int) -> list[dict]:
    messages = [RULES]
    for i in range(history_turns):
        messages += history_turn(i)
    messages += [DOC_CONTEXT, REQUEST]
    for i in range(steps):
        messages += step(f"call_{i}", f"result {i} " + "x" * output_chars)
    return messages


def test_under_budget_nothing_changes():
    messages = conversation(history_turns=3, steps=2, output_chars=100)
    fitted, plan = fit_context(messages, token_budget=100_000, max_messages=400)
    assert fitted == messages
    assert plan == {"history_drop": 0, "turn_drop": 0, "shortened": []}


def test_a_long_turn_never_loses_the_request_or_its_document_context():
    """The bug this module fixes: a count-capped window used to slide past the
    user's request and the per-request system messages mid-turn."""
    messages = conversation(history_turns=5, steps=30, output_chars=50)
    fitted, plan = fit_context(messages, token_budget=1_000_000, max_messages=20)

    assert len(fitted) <= 20
    assert fitted[0] == RULES
    assert DOC_CONTEXT in fitted and REQUEST in fitted
    assert fitted.index(DOC_CONTEXT) < fitted.index(REQUEST)
    # Earlier turns went first, then the oldest steps, with a note saying so.
    assert plan["history_drop"] == 10 and plan["turn_drop"] > 0
    assert any("omitted to fit the context budget" in (m.get("content") or "") for m in fitted)
    # The newest step is always there, and the sequence is still valid.
    assert fitted[-1] == {"role": "tool", "tool_call_id": "call_29", "content": messages[-1]["content"]}
    assert validate_message_sequence(fitted) == []


def test_old_tool_outputs_are_shortened_before_earlier_turns_are_dropped():
    messages = conversation(history_turns=4, steps=6, output_chars=20_000)
    # Over budget, but shortening alone gets it under the low watermark.
    budget = int(estimate_tokens(messages) * 0.9)
    fitted, plan = fit_context(messages, token_budget=budget, max_messages=400)

    assert estimate_tokens(fitted) <= budget * LOW_WATERMARK
    assert plan["history_drop"] == 0, "earlier turns are kept while shortening is enough"
    assert plan["shortened"] and "call_5" not in plan["shortened"]
    outputs = {m["tool_call_id"]: m["content"] for m in fitted if m.get("role") == "tool"}
    assert "characters omitted" in outputs["call_0"]
    assert len(outputs["call_0"]) < SHORTENED_OUTPUT_CHARS + 200
    assert outputs["call_5"] == messages[-1]["content"], "the latest step stays whole"


def test_earlier_turns_are_dropped_whole_and_oldest_first():
    messages = conversation(history_turns=10, steps=1, output_chars=10)
    budget = int(estimate_tokens(messages) * 0.9)
    fitted, plan = fit_context(messages, token_budget=budget, max_messages=400)

    history = fitted[1:fitted.index(DOC_CONTEXT)]
    assert history, "some recent history should still fit"
    assert history[0]["role"] == "user", "a kept turn always starts with its question"
    assert history[-1]["content"].startswith("answer 9"), "the newest turns are the ones kept"
    assert plan["history_drop"] % 2 == 0


def test_retry_guidance_after_the_latest_step_does_not_get_that_step_dropped():
    messages = conversation(history_turns=0, steps=5, output_chars=10)
    messages.append({"role": "system", "content": "The previous snippet FAILED. Send a corrected one."})
    fitted, _ = fit_context(messages, token_budget=1_000_000, max_messages=6)

    assert fitted[-1]["content"].startswith("The previous snippet FAILED")
    assert any(m.get("tool_call_id") == "call_4" for m in fitted)
    assert validate_message_sequence(fitted) == []


def test_the_cut_only_grows_so_the_prompt_prefix_stays_stable():
    """Prompt caching needs the start of the context to stay the same between
    the calls of one turn. Trimming to a low watermark means most calls reuse
    the previous plan instead of shifting the window every time."""
    base = conversation(history_turns=20, steps=0, output_chars=0)
    # Room for a few dozen steps, like the real 200K budget has for typical
    # tool output; the turn below still outgrows it several times.
    budget = estimate_tokens(base) + 12_000
    plan, previous, changes, trimmed, calls = None, None, 0, 0, 60
    messages = list(base)
    for i in range(calls):
        messages += step(f"call_{i}", "y" * 1_600)
        fitted, new_plan = fit_context(messages, token_budget=budget, max_messages=400, plan=plan)
        if plan is not None:
            # Monotonic: nothing that was cut comes back.
            assert new_plan["history_drop"] >= plan["history_drop"]
            assert new_plan["turn_drop"] >= plan["turn_drop"]
            assert set(plan["shortened"]) <= set(new_plan["shortened"])
            if new_plan == plan:
                # Unchanged plan: the previous context is an exact prefix.
                assert fitted[:len(previous)] == previous
        changes += new_plan != plan
        trimmed += new_plan != {"history_drop": 0, "turn_drop": 0, "shortened": []}
        plan, previous = new_plan, fitted
    assert trimmed > calls // 2, "the scenario must actually exercise trimming"
    assert changes <= calls // 6, f"plan changed on {changes} of {calls} calls"


def test_message_cap_still_applies_when_size_is_fine():
    messages = conversation(history_turns=50, steps=1, output_chars=10)
    fitted, _ = fit_context(messages, token_budget=10_000_000, max_messages=30)
    assert len(fitted) <= 30
    assert REQUEST in fitted


def test_context_too_long_errors_are_recognised():
    class BadRequest(Exception):
        def __init__(self, message, code=None, body=None):
            super().__init__(message)
            self.code, self.body = code, body

    assert is_context_too_long(BadRequest("400", code="context_length_exceeded"))
    assert is_context_too_long(BadRequest("400", body={"error": {"code": "context_length_exceeded"}}))
    assert is_context_too_long(BadRequest("This model's maximum context length is 1000000 tokens"))
    assert is_context_too_long(BadRequest("400", code="string_above_max_length"))
    assert is_context_too_long(BadRequest("400", body={"error": {"code": "string_above_max_length"}}))
    assert not is_context_too_long(BadRequest("rate limited", code="429"))


def test_planner_retries_once_with_half_the_context_when_the_model_refuses_it():
    class ShrinkingAzure(FakeAzure):
        async def complete(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise ContextTooLongException("maximum context length exceeded")
            return self.turns.pop(0)

    azure = ShrinkingAzure([text_turn("Done.")])
    history = []
    for i in range(30):
        history += history_turn(i, size=4_000)
    final = run(build(azure, FakeMCP()), "summarise", history=history)

    assert final["final_response"] == "Done."
    first, second = azure.calls
    sent, retried = estimate_tokens(first["messages"]), estimate_tokens(second["messages"])
    assert retried <= sent // 2
    assert retried > sent // 3, "the retry sends about half, not a quarter"
    assert second["messages"][-1] == {"role": "user", "content": "summarise"}


def test_an_oversized_latest_output_is_shortened_when_nothing_else_fits():
    """One huge result in the latest step must not make every call fail."""
    messages = conversation(history_turns=2, steps=2, output_chars=10)
    messages[-1] = {**messages[-1], "content": "z" * 400_000}
    fitted, plan = fit_context(messages, token_budget=20_000, max_messages=400)

    assert estimate_tokens(fitted) <= 20_000
    assert "call_1" in plan["shortened"]
    assert "characters omitted" in fitted[-1]["content"]
    assert validate_message_sequence(fitted) == []


def test_a_latest_output_that_fits_the_budget_stays_whole():
    messages = conversation(history_turns=2, steps=2, output_chars=30_000)
    budget = int(estimate_tokens(messages) * 0.9)
    fitted, plan = fit_context(messages, token_budget=budget, max_messages=400)
    assert "call_1" not in plan["shortened"]
    assert fitted[-1]["content"] == messages[-1]["content"]


def test_no_output_is_sent_longer_than_the_api_takes_whatever_the_budget():
    messages = conversation(history_turns=0, steps=1, output_chars=10)
    messages[-1] = {**messages[-1], "content": "z" * (MAX_OUTPUT_CHARS + 5_000)}
    fitted, _ = fit_context(messages, token_budget=10_000_000, max_messages=400)
    assert len(fitted[-1]["content"]) < MAX_OUTPUT_CHARS + 200


def test_end_to_end_a_thirty_round_turn_keeps_the_request_in_every_call():
    rounds = 30
    azure = FakeAzure([
        *(tool_turn((f"call_{i}", "get_current_view_info", {})) for i in range(rounds)),
        text_turn("All walls tagged."),
    ])
    workflow = build(azure, FakeMCP(), max_context_messages=12, max_steps=rounds + 1)
    final = run(workflow, "Tag every wall on Level 2.")

    assert final["final_response"] == "All walls tagged."
    for call in azure.calls:
        sent = call["messages"]
        assert len(sent) <= 12
        assert {"role": "user", "content": "Tag every wall on Level 2."} in sent
        assert validate_message_sequence(sent) == []


def test_defaults_size_the_budget_for_a_one_million_token_model():
    config = AgentConfig()
    assert config.context_token_budget == 200_000
    assert config.max_context_messages == 400
