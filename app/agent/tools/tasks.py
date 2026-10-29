"""The turn's task list, and ``update_tasks``: how the model keeps it current.

A turn gets a task list when the user first steers it (agent/host/steers.py):
item 1 is the request the turn started with, and every steer adds an item,
word for word. Code writes the items, so a steer is never lost or reworded
by the model; the model only sets their status, with ``update_tasks``, and
may give an item a short ``title``: the label a client shows for it, while
the item keeps the user's words.

The list is what the turn is checked against. It is shown to the model at the
end of every planner call (:func:`build_task_list_message`), the planner sends
the model back once when it answers with items still unfinished, and a
wrap-up names the unfinished items so the reply reports them. A client
gets it as ``EVENT_TASKS`` whenever it changes, and the turn trace keeps the
final list.

A turn the user never steers has no list, and ``update_tasks`` is not offered.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent.host.turn_events import publish_tasks
from agent.tools.base import LocalCall, LocalResult, LocalTool

__all__ = [
    "TASK_STATUSES",
    "UPDATE_TASKS_TOOL_NAME",
    "UpdateTasksTool",
    "add_steers",
    "unfinished",
]

UPDATE_TASKS_TOOL_NAME = "update_tasks"

TASK_STATUSES = ("open", "in_progress", "done", "dropped")
#: Statuses that still need work before the turn may end.
_UNFINISHED = frozenset({"open", "in_progress"})
MAX_NOTE_CHARS = 300
MAX_TITLE_CHARS = 60

_DEFINITION = {
    "type": "function",
    "function": {
        "name": UPDATE_TASKS_TOOL_NAME,
        "description": (
            "Update the status of items in this request's task list (the TASK LIST "
            "shown to you: the user's request and the steers they sent while you "
            "worked). Mark an item in_progress when you start it, done when it is "
            "finished, dropped when a later steer cancelled it or it cannot be done "
            "(a note saying why is required). The first time you update an item, give "
            "it a `title`: a few words in the user's language naming the work (\"Count "
            "laptops per company\"); the user sees it. Update several items in one call, "
            "in the same step as your other tool calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "updates": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer", "description": "The item's number."},
                            "status": {"type": "string", "enum": list(TASK_STATUSES)},
                            "title": {
                                "type": "string",
                                "description": "A few words naming the item, for "
                                               "the user. Set it once.",
                                "maxLength": MAX_TITLE_CHARS,
                            },
                            "note": {
                                "type": "string",
                                "description": "Optional; required for dropped. One "
                                               "short line for the user.",
                                "maxLength": MAX_NOTE_CHARS,
                            },
                        },
                        "required": ["id", "status"],
                    },
                },
            },
            "required": ["updates"],
        },
    },
}


def unfinished(tasks: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """The items that still need work: open or in progress."""
    return [task for task in tasks or () if task.get("status") in _UNFINISHED]


def add_steers(
    tasks: list[dict[str, Any]] | None, request: str, steers: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """The task list with one item per steer, starting it if there is none.

    A new list starts with the request the turn began with, in progress: the
    model has been working on it.
    """
    updated = [dict(task) for task in tasks or ()]
    if not updated:
        updated.append({
            "id": 1, "text": request, "source": "request", "status": "in_progress", "note": None,
        })
    for steer in steers:
        updated.append({
            "id": len(updated) + 1, "text": steer["text"], "source": "steer",
            "status": "open", "note": None,
        })
    return updated


class UpdateTasksTool(LocalTool):
    def handles(self, name: str) -> bool:
        return name == UPDATE_TASKS_TOOL_NAME

    def definitions(self, state: Mapping[str, Any]) -> list[dict[str, Any]]:
        # Offered from the first steer on. That changes the tool list once in
        # the turn, which costs one cache miss in a turn the user steered.
        return [_DEFINITION] if state.get("tasks") else []

    async def run(self, call: LocalCall) -> LocalResult:
        tasks = [dict(task) for task in call.state.get("tasks") or ()]
        if not tasks:
            return LocalResult.validation_error(
                "This request has no task list: there is nothing to update. Carry on "
                "without `update_tasks`."
            )
        updates = call.arguments.get("updates")
        if not isinstance(updates, list) or not updates:
            return LocalResult.validation_error(
                "`updates` must list at least one {id, status}. Re-emit the call."
            )
        by_id = {task["id"]: task for task in tasks}
        # "2" for 2: the model's slip, not a reason to refuse the call.
        updates = [
            {**u, "id": int(u["id"])}
            if isinstance(u, dict) and isinstance(u.get("id"), str) and u["id"].strip().isdigit()
            else u
            for u in updates
        ]
        for update in updates:
            problem = _problem(update, by_id)
            if problem:
                return LocalResult.validation_error(problem + " Re-emit the call.")
        for update in updates:
            task = by_id[update["id"]]
            task["status"] = update["status"]
            note = update.get("note")
            if isinstance(note, str) and note.strip():
                task["note"] = " ".join(note.split())[:MAX_NOTE_CHARS]
            title = update.get("title")
            if isinstance(title, str) and title.strip():
                task["title"] = " ".join(title.split())[:MAX_TITLE_CHARS]
        publish_tasks(tasks)
        left = unfinished(tasks)
        text = "Task list updated." + (
            f" Still to do: {', '.join(str(task['id']) for task in left)}."
            if left else " Every item is done or dropped: give your final answer."
        )
        return LocalResult(text=text, data={"tasks": tasks}, update={"tasks": tasks})


def _problem(update: Any, by_id: dict[int, dict[str, Any]]) -> str | None:
    """What is wrong with one entry of ``updates``, or None."""
    if not isinstance(update, dict):
        return "Each update must be an object with `id` and `status`."
    task_id = update.get("id")
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id not in by_id:
        return f"No task-list item {task_id!r}; the items are {', '.join(map(str, by_id))}."
    status = update.get("status")
    if status not in TASK_STATUSES:
        return f"`status` must be one of {', '.join(TASK_STATUSES)}."
    note = update.get("note")
    if status == "dropped" and not (isinstance(note, str) and note.strip()):
        return f"Dropping item {task_id} needs a `note` saying why."
    return None
