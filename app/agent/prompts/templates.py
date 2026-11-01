"""
All prompts, in one place.

Callers assemble prompts through these constants and builders instead of
concatenating strings at call sites, so a prompt change is a change here.
"""

from __future__ import annotations

__all__ = [
    "CHARTS_NOTE",
    "EARLIER_TOOL_RESULTS_NOTE",
    "NO_CHARTS_NOTE",
    "SKILL_APP_CONTEXT",
    "SYSTEM_PROMPT",
    "build_open_tasks_note",
    "build_server_instructions_message",
    "build_steer_message",
    "build_task_list_message",
    "build_unfinished_tasks_note",
    "build_wrap_up_note",
]


SYSTEM_PROMPT = (
    "You are a data assistant. You answer questions about the company's data by "
    "reading SQL Server databases through the tools of an MCP server. The tools "
    "are read-only: nothing you do changes any data.\n"
    "How to work:\n"
    "- Find out before you query: list the databases, read the business rules of "
    "the one you use, and describe an object before you write SQL against it. The "
    "business rules say what the columns mean and how to count; follow them.\n"
    "- Prefer aggregates (COUNT, SUM, GROUP BY, TOP n) over fetching raw rows, and "
    "say how many rows a figure is based on when the rules ask for it.\n"
    "- When a query fails, read the error's `fix` and correct the query; do not "
    "send the same query again.\n"
    "- Never guess numbers. Every figure in your answer comes from a query result "
    "in this conversation.\n"
    "- Answer in the user's language. Lead with the answer, then the detail; use a "
    "short table when it helps. Mention assumptions you made.\n"
    "- Never show more personal data (names, HR codes) than the question needs."
)

#: Added when the conversation contains tool results from earlier turns.
EARLIER_TOOL_RESULTS_NOTE = (
    "Tool results from earlier turns in this conversation show the data as it was "
    "at that time. Use them to know what was already found (object names, columns, "
    "errors), but query again when the user asks for current figures."
)

#: Added when the turn offers the charts server's tools (the client draws charts).
CHARTS_NOTE = (
    "You can draw charts with the create_* tools. Draw one when the user asks for a "
    "chart, or when a breakdown or a trend reads better as one than as a table.\n"
    "- Query first, aggregated to what the chart shows, and pass run_select's result "
    "as `data` unchanged.\n"
    "- Keep format html (the default). The chart is drawn under your reply, "
    "interactive; you never see it, only its summary. Do not ask for png.\n"
    "- In your reply, say what the chart shows (the pattern, the figure that "
    "matters). Do not link to it or call it a file.\n"
    "- For two or more charts, use one create_dashboard call, not one call per chart. "
    "Give it columns 2. With indicator figures, columns 4 fits four in a row; then "
    "give each other chart columnSpan 2. The page keeps "
    "that grid where the chat is wide enough and stacks the charts where it is not, "
    "so never lower columns to fit a small screen. Give a chart with many or long "
    "labels columnSpan 2, and put indicator figures together in a row of their own."
)

#: Added when the client cannot draw charts, so the model says why there is
#: none instead of guessing at what it lacks.
NO_CHARTS_NOTE = (
    "This chat cannot show charts. When the user asks for one, say that charts "
    "show in the chat apps that draw them, and give the figures as a short table."
)

#: What a skill is for in this app, for the AI that rewrites a user's wording
#: into a skill (skills_hub.rewrite.skill_rewrite_prompt).
SKILL_APP_CONTEXT = (
    "The skill is for a data assistant that answers questions about company data "
    "by running read-only SQL queries against SQL Server databases, through an MCP "
    "server that exposes the allowed views and their business rules. Its users ask "
    "for counts, lists, breakdowns and checks of the data. Read the user's words "
    "as data terms wherever they can be: \"views\", \"tables\", \"columns\", "
    "\"rows\", \"filters\" and \"reports\" are the database things of those names."
)


def build_server_instructions_message(instructions: str) -> str:
    """The MCP server's own instructions: how to use its tools."""
    return f"Instructions from the MCP server about its tools:\n{instructions.strip()}"


def build_wrap_up_note(reason: str) -> str:
    """The last model call of a turn that hit a limit: answer with what is known."""
    return (
        f"Stop working now: {reason} No more tools can be called. Write your final "
        "answer from the results you already have: what you found, what is still "
        "unanswered, and what the user could ask next to get the rest."
    )


#: How each task status reads in the task list the model sees.
_TASK_STATUS_LABELS = {
    "open": "open",
    "in_progress": "in progress",
    "done": "done",
    "dropped": "dropped",
}
#: The original request as a task-list line; the full text is in the request.
_TASK_REQUEST_CHARS = 300


def _task_lines(tasks: list[dict]) -> str:
    lines = []
    for task in tasks:
        text = " ".join(str(task.get("text") or "").split())
        if task.get("source") == "request":
            if len(text) > _TASK_REQUEST_CHARS:
                text = text[: _TASK_REQUEST_CHARS - 3].rstrip() + "..."
            label = "Request"
        else:
            label = "Steer"
        status = _TASK_STATUS_LABELS.get(str(task.get("status")), str(task.get("status")))
        title = f" (title: {task['title']})" if task.get("title") else ""
        lines.append(f"{task.get('id')}. [{status}] {label}: {text}{title}")
        if task.get("note"):
            lines.append(f"   note: {task['note']}")
    return "\n".join(lines)


def build_steer_message(texts: list[str]) -> str:
    """The user message that carries steers to the model (agent/host/steers.py).

    A steer amends the task the model is doing: it is framed so the model
    carries on with the corrected task instead of answering it as a new one.
    """
    if len(texts) == 1:
        quoted = texts[0]
    else:
        quoted = "\n\n".join(f"Message {n}: {text}" for n, text in enumerate(texts, start=1))
    plural = "these messages" if len(texts) > 1 else "this message"
    return (
        f"STEER: the user sent {plural} while you were working on their request "
        "above. It amends the task you are doing; it is not a new request. It may "
        "correct the work, redirect it, or add to it.\n"
        "- A correction or redirect applies from now on: redo the queries it "
        "affects. Results that do not conflict with it stay.\n"
        "- An addition joins the task where the user put it (\"after you are "
        "done\" means after the current work, not now).\n"
        "- Where it conflicts with the original request, the latest steer wins.\n"
        "- It is now in the task list; keep the list current with `update_tasks`.\n"
        "- In the text of this response, tell the user in one sentence, in their "
        "language, how the plan changes, and make the next tool calls in the SAME "
        "response. Do not stop to wait for them.\n\n"
        f"The user wrote:\n{quoted}"
    )


def build_task_list_message(tasks: list[dict]) -> str:
    """The turn's task list, sent at the end of every model call once the user steers."""
    return (
        "TASK LIST for this request: the user's request and the steers they sent "
        "while you worked. Keep it current with `update_tasks` (in_progress when you "
        "start an item, done when it is finished, dropped with a note when a steer "
        "cancelled it or it cannot be done; a short `title` the first time). Do not "
        "give your final answer while an item is open or in progress; your final "
        "answer covers every item.\n"
        f"{_task_lines(tasks)}"
    )


def build_open_tasks_note(tasks: list[dict]) -> str:
    """Sent once when the model answers while task-list items are unfinished."""
    return (
        "Before you finish: these task-list items are not marked done:\n"
        f"{_task_lines(tasks)}\n"
        "Do what is left of them now. If they are already done, or a steer "
        "cancelled them, or they cannot be done, call `update_tasks` to mark them "
        "done or dropped (with a note why). Then give your final answer, covering "
        "every item of the task list."
    )


def build_unfinished_tasks_note(tasks: list[dict]) -> str:
    """Added to a wrap-up: the task-list items the stopping turn leaves undone."""
    return (
        "These task-list items are not done:\n"
        f"{_task_lines(tasks)}\n"
        "Your reply must say, for each, what of it was done and what was not, so "
        "the user can ask for the rest again."
    )
