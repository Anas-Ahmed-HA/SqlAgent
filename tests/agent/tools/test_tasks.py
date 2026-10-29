"""``update_tasks`` and the task list (agent/tools/tasks.py)."""

from __future__ import annotations

import asyncio

import pytest

from agent.tools.base import LocalCall
from agent.tools.tasks import UpdateTasksTool, add_steers, unfinished

TASKS = add_steers(None, "create walls on grid A", [
    {"id": "s1", "text": "make them 3m"}, {"id": "s2", "text": "then an excel sheet"},
])


def run(arguments: dict, tasks=TASKS):
    call = LocalCall(name="update_tasks", arguments=arguments, call_id="c1", state={"tasks": tasks})
    return asyncio.run(UpdateTasksTool().run(call))


def test_a_task_list_starts_with_the_request_and_adds_each_steer_word_for_word():
    assert [(t["id"], t["source"], t["status"], t["text"]) for t in TASKS] == [
        (1, "request", "in_progress", "create walls on grid A"),
        (2, "steer", "open", "make them 3m"),
        (3, "steer", "open", "then an excel sheet"),
    ]
    more = add_steers(TASKS, "ignored once the list exists", [{"id": "s3", "text": "tag them"}])
    assert [t["id"] for t in more] == [1, 2, 3, 4] and more[0]["text"] == "create walls on grid A"
    assert [t["id"] for t in unfinished(TASKS)] == [1, 2, 3]


def test_update_tasks_is_offered_only_once_the_turn_has_a_task_list():
    tool = UpdateTasksTool()
    assert tool.definitions({"tasks": []}) == []
    assert tool.definitions({"tasks": TASKS})[0]["function"]["name"] == "update_tasks"


def test_statuses_and_notes_are_updated_and_the_rest_is_reported():
    result = run({"updates": [
        {"id": 1, "status": "done"},
        {"id": "2", "status": "dropped", "note": "  the user  cancelled it "},
    ]})
    assert not result.is_error
    tasks = result.update["tasks"]
    assert [t["status"] for t in tasks] == ["done", "dropped", "open"]
    assert tasks[1]["note"] == "the user cancelled it"
    assert "Still to do: 3." in result.text
    assert TASKS[0]["status"] == "in_progress", "the turn's list is not changed in place"

    finished = run({"updates": [{"id": 3, "status": "done"}]}, tasks)
    assert "give your final answer" in finished.text


@pytest.mark.parametrize("arguments, problem", [
    ({"updates": []}, "at least one"),
    ({"updates": [{"id": 9, "status": "done"}]}, "No task-list item 9"),
    ({"updates": [{"id": True, "status": "done"}]}, "No task-list item"),
    ({"updates": [{"id": 1, "status": "finished"}]}, "`status` must be one of"),
    ({"updates": [{"id": 2, "status": "dropped"}]}, "needs a `note`"),
    ({"updates": [{"id": 1, "status": "done"}, "oops"]}, "must be an object"),
])
def test_a_bad_update_changes_nothing(arguments, problem):
    result = run(arguments)
    assert result.is_error and result.error_kind == "validation"
    assert problem in result.text
    assert result.update == {}


def test_a_title_names_an_item_for_the_user_and_keeps_its_words():
    result = run({"updates": [
        {"id": 1, "status": "in_progress", "title": "  Walls on   grid A "},
        {"id": 2, "status": "open", "title": "x" * 200},
        {"id": 3, "status": "open", "title": 7},
    ]})
    tasks = result.update["tasks"]
    assert tasks[0]["title"] == "Walls on grid A"
    assert tasks[0]["text"] == "create walls on grid A"
    assert len(tasks[1]["title"]) == 60
    assert "title" not in tasks[2]


def test_without_a_task_list_there_is_nothing_to_update():
    result = run({"updates": [{"id": 1, "status": "done"}]}, tasks=[])
    assert result.is_error and "no task list" in result.text
