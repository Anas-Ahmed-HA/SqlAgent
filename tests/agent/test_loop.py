"""The agent loop end to end, over fake Azure and MCP services.

Every path out of the loop is here: a plain answer, tool calls then an
answer, a tool error the model corrects, a repeated failing call, the step
and time limits, an unreachable server, a connection that breaks mid-turn,
steers, the task list and questions. Each run's messages are checked for the
sequence Azure accepts.
"""

from __future__ import annotations

import asyncio
import json

from agent.host.interactions import interactions
from agent.host.steers import steers
from agent.host.turn_events import TurnEvents, turn_events
from agent.memory.history import validate_message_sequence
from agent.prompts.templates import SYSTEM_PROMPT
from tests.agent.fakes import FakeAzure, FakeMCP, make_loop, ok, text_turn, tool_error, tool_turn


def run(loop, message: str = "How many laptops?", **kwargs):
    return asyncio.run(loop.run(user_message=message, session_id="HR-1|s", **kwargs))


def tool_messages(state) -> list[str]:
    return [m["content"] for m in state["messages"] if m.get("role") == "tool"]


def test_a_plain_answer_is_one_model_call() -> None:
    azure = FakeAzure([text_turn("Hello.")])
    state = run(make_loop(azure))

    assert state["final_response"] == "Hello."
    assert state["execution_status"] == "complete"
    assert len(azure.calls) == 1
    sent = azure.calls[0]["messages"]
    assert sent[0] == {"role": "system", "content": SYSTEM_PROMPT}
    # The server's own instructions follow the rules.
    assert "Call list_databases first." in sent[1]["content"]
    assert sent[-1] == {"role": "user", "content": "How many laptops?"}


def test_tool_calls_run_on_the_server_and_the_model_answers_from_them() -> None:
    azure = FakeAzure([
        tool_turn(("c1", "list_databases", {}), ("c2", "get_business_rules", {"database": "Apps"})),
        tool_turn(("c3", "run_select", {"database": "Apps", "sql": "SELECT COUNT(*) FROM v"})),
        text_turn("3,679 laptops."),
    ])
    mcp = FakeMCP({"run_select": ok('{"rows":[[3679]]}')})
    state = run(make_loop(azure, mcp))

    assert state["final_response"] == "3,679 laptops."
    assert [name for name, _ in mcp.calls] == ["list_databases", "get_business_rules", "run_select"]
    assert tool_messages(state)[-1] == '{"rows":[[3679]]}'
    assert {r["outcome"] for r in state["tool_results"].values()} == {"success"}
    assert validate_message_sequence(state["messages"]) == []
    # Every model call is offered the server's tools; the status argument is added.
    offered = {t["function"]["name"]: t for t in azure.calls[0]["tools"]}
    assert set(offered) == {"list_databases", "get_business_rules", "list_objects", "describe_object", "run_select"}
    assert "ui_status" in offered["run_select"]["function"]["parameters"]["properties"]


def test_a_tool_error_goes_back_to_the_model_with_its_fix() -> None:
    bad = {"database": "Apps", "sql": "SELECT nope FROM v"}
    good = {"database": "Apps", "sql": "SELECT n FROM v"}
    azure = FakeAzure([
        tool_turn(("c1", "run_select", bad)),
        tool_turn(("c2", "run_select", good)),
        text_turn("Done."),
    ])
    mcp = FakeMCP({"run_select": lambda args: (
        tool_error("query_failed", "Invalid column name 'nope'.", "Call describe_object.")
        if args["sql"] == bad["sql"] else ok("[[1]]")
    )})
    state = run(make_loop(azure, mcp))

    first = json.loads(tool_messages(state)[0])
    assert first["code"] == "query_failed" and first["fix"] == "Call describe_object."
    assert state["tool_results"]["c1"] == {**state["tool_results"]["c1"], "outcome": "failure", "error_kind": "query_failed"}
    assert state["final_response"] == "Done."


def test_an_identical_failed_call_is_refused_without_reaching_the_server() -> None:
    args = {"database": "Apps", "sql": "SELECT nope FROM v"}
    azure = FakeAzure([
        tool_turn(("c1", "run_select", {**args, "ui_status": "Trying"})),
        # Same call, reworded status: still the same call.
        tool_turn(("c2", "run_select", {**args, "ui_status": "Trying again"})),
        text_turn("I could not run it."),
    ])
    mcp = FakeMCP({"run_select": tool_error("query_failed", "Invalid column name 'nope'.")})
    state = run(make_loop(azure, mcp))

    assert len(mcp.calls) == 1
    assert tool_messages(state)[1].startswith("NOT RUN -- this exact call already failed")
    assert state["tool_results"]["c2"]["error_kind"] == "repeat"


def test_bad_argument_json_is_answered_without_a_call() -> None:
    azure = FakeAzure([tool_turn(("c1", "run_select", {})), text_turn("ok")])
    azure.turns[0].choices[0].message.tool_calls[0].function.arguments = "{not json"
    mcp = FakeMCP()
    state = run(make_loop(azure, mcp))

    assert mcp.calls == []
    assert tool_messages(state)[0].startswith("NOT RUN -- the arguments are not valid JSON")
    assert validate_message_sequence(state["messages"]) == []


def test_the_step_limit_ends_with_an_answer_written_without_tools() -> None:
    azure = FakeAzure([
        tool_turn(("c1", "list_databases", {})),
        tool_turn(("c2", "list_objects", {"database": "Apps"})),
        text_turn("Here is what I found so far."),
    ])
    state = run(make_loop(azure, max_steps=2))

    assert state["final_response"] == "Here is what I found so far."
    assert state["execution_status"] == "stopped"
    last = azure.calls[-1]
    assert "tools" not in last
    assert "Stop working now" in last["messages"][-1]["content"]


def test_the_wrap_up_falls_back_to_a_fixed_text_when_the_model_fails() -> None:
    class Failing(FakeAzure):
        async def complete(self, **kwargs):
            if "tools" not in kwargs:
                raise RuntimeError("down")
            return await super().complete(**kwargs)

    azure = Failing([tool_turn(("c1", "list_databases", {}))])
    state = run(make_loop(azure, max_steps=1))
    assert state["final_response"].startswith("(Stopped: the step limit")


def test_the_deadline_ends_the_turn() -> None:
    azure = FakeAzure([text_turn("Partial answer.")])
    state = run(make_loop(azure, max_request_seconds=0))
    assert state["final_response"] == "Partial answer."
    assert state["execution_status"] == "stopped"
    assert "tools" not in azure.calls[0]


def test_an_unreachable_server_ends_the_turn_without_a_model_call() -> None:
    azure = FakeAzure([text_turn("never")])
    state = run(make_loop(azure, FakeMCP(unreachable=True)))

    assert azure.calls == []
    assert state["execution_status"] == "failed"
    assert "http://mcp.test/mcp" in state["final_response"]
    assert "not reachable" in state["final_response"]


def test_a_connection_that_breaks_mid_turn_answers_every_call_and_stops() -> None:
    azure = FakeAzure([
        tool_turn(("c1", "list_databases", {}), ("c2", "list_objects", {"database": "Apps"}), ("c3", "describe_object", {})),
        text_turn("never"),
    ])
    mcp = FakeMCP()
    mcp.break_on_call = 2
    state = run(make_loop(azure, mcp))

    assert len(azure.calls) == 1
    assert state["execution_status"] == "failed"
    assert "connection to the data service broke" in state["final_response"]
    results = tool_messages(state)
    assert results[0] == "list_databases ok"
    assert results[1].startswith("NOT RUN") and results[2] == "NOT RUN -- the turn ended."
    assert validate_message_sequence(state["messages"]) == []


def test_the_conversation_is_saved_for_the_next_turn() -> None:
    azure = FakeAzure([
        tool_turn(("c1", "list_databases", {})),
        text_turn("One database."),
        text_turn("Still one."),
    ])
    loop = make_loop(azure)
    run(loop, "Which databases?")
    run(loop, "And now?")

    second = azure.calls[-1]["messages"]
    contents = [m.get("content") for m in second]
    assert "Which databases?" in contents and "One database." in contents
    # System messages are not stored: the rules appear once, re-added this turn.
    assert contents.count(SYSTEM_PROMPT) == 1
    assert validate_message_sequence([m for m in second if m.get("role") != "system"]) == []


def test_equipped_skills_reach_the_model() -> None:
    skill = {
        "id": "00000000-0000-0000-0000-000000000001", "name": "Counts only",
        "description": "Answer with counts.", "instructions": ["Never list names."], "version": 1,
    }
    azure = FakeAzure([text_turn("ok")])
    run(make_loop(azure), equipped_skills=[skill])
    joined = "\n".join(m["content"] for m in azure.calls[0]["messages"] if m["role"] == "system")
    assert "Counts only" in joined and "Never list names." in joined


# -- steers, the task list and questions ------------------------------------------


def bound_run(loop, *, interactive=True, message="Count laptops."):
    """Run a turn bound to a turn id, as the host does."""

    async def scenario():
        with turn_events.bind("HR-1", "t1"), interactions.bind(interactive), steers.bind(interactive):
            return await loop.run(user_message=message, session_id="HR-1|s", interactive=interactive)

    return asyncio.run(scenario())


def send_steer(text: str, steer_id: str = "s1"):
    def act(kwargs):
        assert steers.submit("HR-1", "t1", {"type": "steer", "id": steer_id, "text": text})
        return None

    return act


def test_a_steer_joins_the_turn_and_starts_a_task_list() -> None:
    def steer_then(turn):
        def act(kwargs):
            send_steer("also per company")(kwargs)
            return turn
        return act

    azure = FakeAzure([
        steer_then(tool_turn(("c1", "run_select", {"database": "Apps", "sql": "SELECT 1"}))),
        tool_turn(
            ("c2", "update_tasks", {"updates": [{"id": 1, "status": "done"}, {"id": 2, "status": "done"}]}),
            content="I will also split it per company.",
        ),
        text_turn("Total 10; per company: A 6, B 4."),
    ])
    state = bound_run(make_loop(azure))

    second = azure.calls[1]["messages"]
    steer_message = next(m for m in second if m.get("role") == "user" and "STEER" in (m.get("content") or ""))
    assert "also per company" in steer_message["content"]
    assert second[-1]["role"] == "system" and second[-1]["content"].startswith("TASK LIST")
    assert [t["status"] for t in state["tasks"]] == ["done", "done"]
    assert state["steers"][0]["reply"] == "I will also split it per company."
    assert state["final_response"] == "Total 10; per company: A 6, B 4."
    # update_tasks is offered once there is a list, and only then.
    assert "update_tasks" not in {t["function"]["name"] for t in azure.calls[0]["tools"]}
    assert "update_tasks" in {t["function"]["name"] for t in azure.calls[1]["tools"]}


def test_an_answer_with_open_task_items_is_sent_back_once() -> None:
    def steer_then(turn):
        def act(kwargs):
            send_steer("and monitors")(kwargs)
            return turn
        return act

    azure = FakeAzure([
        steer_then(tool_turn(("c1", "run_select", {"database": "Apps", "sql": "SELECT 1"}))),
        text_turn("Laptops: 10."),
        text_turn("Laptops: 10. Monitors: 5."),
    ])
    state = bound_run(make_loop(azure))

    assert len(azure.calls) == 3
    note = azure.calls[2]["messages"][-2]["content"]
    assert note.startswith("Before you finish")
    assert state["final_response"] == "Laptops: 10. Monitors: 5."


def test_a_steer_that_arrives_while_the_model_answers_replaces_the_answer() -> None:
    def steer_then(turn):
        def act(kwargs):
            send_steer("in Arabic please")(kwargs)
            return turn
        return act

    azure = FakeAzure([
        steer_then(text_turn("draft answer")),
        tool_turn(("c1", "update_tasks", {"updates": [{"id": 1, "status": "done"}, {"id": 2, "status": "done"}]})),
        text_turn("final answer"),
    ])
    state = bound_run(make_loop(azure))

    assert state["final_response"] == "final answer"
    assert "draft answer" not in [m.get("content") for m in state["messages"]]


def test_ask_user_is_offered_only_to_an_interactive_turn() -> None:
    azure = FakeAzure([text_turn("a"), text_turn("b")])
    loop = make_loop(azure)
    bound_run(loop, interactive=False)
    bound_run(loop, interactive=True)
    assert "ask_user" not in {t["function"]["name"] for t in azure.calls[0]["tools"]}
    assert "ask_user" in {t["function"]["name"] for t in azure.calls[1]["tools"]}


def test_a_question_waits_for_the_answer_and_the_model_reads_it() -> None:
    question = {"question": "Which period?", "options": [{"label": "This year"}, {"label": "All time"}]}
    events = TurnEvents()
    answered = []

    async def scenario():
        from agent.host.interactions import Interactions
        from agent.tools.base import LocalToolRegistry
        from agent.tools.questions import AskUserTool
        from agent.tools.tasks import UpdateTasksTool

        registry = Interactions(events)
        azure = FakeAzure([tool_turn(("c1", "ask_user", {"questions": [question]})), text_turn("All time: 10.")])
        loop = make_loop(azure)
        loop.local_tools = LocalToolRegistry([AskUserTool(registry), UpdateTasksTool()])
        queue, _ = events.subscribe("HR-1", "t1")

        async def answer():
            event = await queue.get()
            while event["type"] != "question":
                event = await queue.get()
            answered.append(registry.answer("HR-1", "t1", {
                "type": "answer", "id": event["id"], "answers": [{"choices": ["All time"], "other": None}],
            }))

        with events.bind("HR-1", "t1"), registry.bind(True), steers.bind(True):
            answering = asyncio.create_task(answer())
            state = await loop.run(user_message="Count laptops.", session_id="HR-1|s", interactive=True)
            await answering
        return state, azure

    state, azure = asyncio.run(scenario())
    assert answered == [True]
    assert '"All time"' in tool_messages(state)[0]
    assert state["tool_results"]["c1"]["interaction"]["answers"] == [{"choices": ["All time"], "other": None}]
    assert state["final_response"] == "All time: 10."
