"""
The agent: one plain tool loop over an MCP server reached by URL.

    build the messages (rules, the server's instructions, history, skills, the user message)
    repeat, at most `max_steps` times and until the deadline:
        take the steers the user sent meanwhile
        call the model with the MCP tools and the local ones (ask_user, update_tasks)
        no tool calls -> that is the answer (unless a steer or open task items keep it going)
        otherwise run each call and add its result as a `tool` message
    out of steps or time -> one last call without tools writes the answer
    save the conversation

The MCP server checks every call itself (unknown tools, bad arguments, bad
SQL) and answers with an error the model can act on, so nothing is validated
twice here. The loop adds only what the server cannot know: the step and
time limits, refusing a call identical to one that already failed this turn,
and keeping the context under its token budget.

An unreachable MCP server ends the turn at once with a message saying so:
more model calls cannot help.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from agent.clients.mcp_session import MCPSession
from agent.config import AgentConfig
from agent.errors import ContextTooLongException, MCPConnectionException, ToolTimeoutException
from agent.host.steers import Steers, steers as default_steers
from agent.host.turn_events import publish_message, publish_status, publish_tasks, publish_thinking
from agent.memory.context_budget import describe_plan, estimate_tokens, fit_context, trim_earlier_turns
from agent.memory.history import (
    STEER_KEY,
    extract_last_assistant_text,
    sanitize_client_history,
    strip_for_persistence,
    truncate_messages,
)
from agent.memory.store import MemoryStore
from agent.observability.turn_trace import (
    expand_history_traces,
    fallback_ui_status,
    split_ui_status,
    turn_step_count,
    with_ui_status,
)
from agent.prompts.templates import (
    EARLIER_TOOL_RESULTS_NOTE,
    SYSTEM_PROMPT,
    build_open_tasks_note,
    build_server_instructions_message,
    build_steer_message,
    build_task_list_message,
    build_unfinished_tasks_note,
    build_wrap_up_note,
)
from agent.tools.base import LocalCall, LocalToolRegistry
from agent.tools.questions import AskUserTool
from agent.tools.tasks import UpdateTasksTool, add_steers, unfinished
from agent.utils.fingerprint import failure_signature
from skills_hub.prompt import build_equipped_skills_message

__all__ = ["AgentLoop", "MAX_REPLY_CHARS"]

logger = logging.getLogger("sql_agent.loop")

#: The longest reply to a steer shown mid-turn; it is meant to be a sentence.
MAX_REPLY_CHARS = 1000

_STEP_LIMIT_FALLBACK = "(Stopped: the step limit for one request was reached before an answer.)"
_DEADLINE_FALLBACK = "(Stopped: this request ran past its time limit. Send a narrower request.)"
_REPEAT_REFUSAL = (
    "NOT RUN -- this exact call already failed in this request. Change the "
    "arguments (read the earlier error and its fix) or answer with what you have."
)


class _Unreachable(Exception):
    """The MCP server cannot be reached: the turn ends with ``message``."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class AgentLoop:
    """Runs one chat turn. Holds no per-turn state: every turn builds its own."""

    def __init__(
        self,
        *,
        mcp: MCPSession,
        azure: Any,
        memory: MemoryStore,
        config: AgentConfig,
        steers: Steers | None = None,
        local_tools: LocalToolRegistry | None = None,
    ) -> None:
        self.mcp = mcp
        self.azure = azure
        self.memory = memory
        self.config = config
        self.steers = steers or default_steers
        self.local_tools = local_tools or LocalToolRegistry([AskUserTool(), UpdateTasksTool()])

    async def run(
        self,
        user_message: str,
        session_id: str = "default",
        history: list[dict[str, Any]] | None = None,
        hr_code: str | None = None,
        equipped_skills: list[dict[str, Any]] | None = None,
        interactive: bool = False,
    ) -> dict[str, Any]:
        """Run one turn and return its state: ``final_response``, ``messages``,
        ``tool_results``, ``planning_ms``, ``context_usage``, ``steers``, ``tasks``.

        ``history`` from the client wins over the server's memory of the
        session. ``equipped_skills`` are already checked (agent.host.turns).
        ``interactive`` says the client answers questions, so ``ask_user`` is
        offered.
        """
        started = time.monotonic()
        state: dict[str, Any] = {
            "request_id": uuid.uuid4().hex[:12],
            "session_id": session_id,
            "hr_code": hr_code or session_id.split("|", 1)[0],
            "user_message": user_message,
            "interactive": bool(interactive),
            "deadline": started + self.config.max_request_seconds,
            "questions_asked": 0,
            "questions_total": 0,
            "tool_results": {},
            "planning_ms": [],
            "failed_signatures": [],
            "context_plan": None,
            "context_usage": None,
            "tasks": [],
            "steers": [],
            "tasks_checked": False,
            "execution_status": "running",
        }
        logger.info("turn.start", extra={"session": session_id, "chars": len(user_message)})
        try:
            state["messages"] = await self._initial_messages(state, history, equipped_skills)
            answer = await self._loop(state)
        except _Unreachable as exc:
            logger.error("turn.mcp_unreachable", extra={"detail": exc.message})
            state.setdefault("messages", [{"role": "user", "content": user_message}])
            state["execution_status"] = "failed"
            answer = exc.message
        state["messages"] = [*state["messages"], {"role": "assistant", "content": answer}]
        state["raw_reply"] = state["final_response"] = extract_last_assistant_text(state["messages"])
        await self._save(state)
        logger.info(
            "turn.end",
            extra={
                "status": state["execution_status"],
                "steps": len(state["planning_ms"]),
                "elapsed_ms": round((time.monotonic() - started) * 1000),
            },
        )
        return state

    # -- setup -------------------------------------------------------------

    async def _initial_messages(
        self,
        state: dict[str, Any],
        history: list[dict[str, Any]] | None,
        equipped_skills: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Rules first, then earlier turns, then this turn's context and the user message."""
        if history:
            earlier = list(history)
            # A client's history may already end with this turn's message.
            last = earlier[-1] if earlier else None
            if isinstance(last, dict) and last.get("role") == "user" and last.get("content") == state["user_message"]:
                earlier.pop()
            earlier = sanitize_client_history(expand_history_traces(earlier))
        else:
            earlier = list((await self.memory.load(state["session_id"])).messages)
        earlier = truncate_messages(earlier, self.config.max_context_messages)
        # Cut once here, so every model call of this turn sends the same prefix.
        earlier = trim_earlier_turns(earlier, self.config.full_turns)

        await self._tools(check=True)  # connects, and reads the server's instructions
        rules = [{"role": "system", "content": SYSTEM_PROMPT}]
        if self.mcp.instructions:
            rules.append({"role": "system", "content": build_server_instructions_message(self.mcp.instructions)})

        context = [{"role": "system", "content": build_equipped_skills_message(equipped_skills or [])}]
        if any(m.get("role") == "tool" for m in earlier):
            context.append({"role": "system", "content": EARLIER_TOOL_RESULTS_NOTE})
        return [*rules, *earlier, *context, {"role": "user", "content": state["user_message"]}]

    async def _tools(self, *, check: bool = False) -> list[dict[str, Any]]:
        """The server's tools; with ``check``, first make sure it answers now."""
        try:
            if check:
                await self.mcp.ping()
            return await self.mcp.get_tools_for_openai()
        except MCPConnectionException as exc:
            raise _Unreachable(
                f"The data service (MCP server at {self.mcp.url}) is not reachable, so the "
                f"question could not be answered.\n\n{exc}\n\nCheck that the server is running "
                "and send the request again."
            ) from exc

    # -- the loop ----------------------------------------------------------

    async def _loop(self, state: dict[str, Any]) -> str:
        for step in range(1, self.config.max_steps + 1):
            if time.monotonic() >= state["deadline"]:
                logger.warning("loop.deadline")
                return await self._wrap_up(
                    state,
                    f"this request reached its {self.config.max_request_seconds:.0f}s time limit.",
                    _DEADLINE_FALLBACK,
                )
            steered = self._take_steers(state)
            tools = with_ui_status([*await self._tools(), *self.local_tools.definitions(state)])
            message = await self._complete(state, tools=tools)

            if not message.tool_calls:
                if self._keep_going(state, steered, message.content):
                    continue
                state["execution_status"] = "complete"
                return message.content or ""

            calls = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
                for call in message.tool_calls
            ]
            logger.info("loop.tool_calls", extra={"step": step, "tools": ",".join(c["function"]["name"] for c in calls)})
            # The model's text is shown mid-turn only as its reply to a steer
            # this step read; otherwise the user never sees it, so it is not kept.
            reply = self._steer_reply(state, steered, message.content)
            state["messages"] = [*state["messages"], {"role": "assistant", "content": reply, "tool_calls": calls}]
            for call in calls:
                state["messages"] = [*state["messages"], await self._run_call(state, call)]

        logger.warning("loop.step_limit", extra={"steps": self.config.max_steps})
        return await self._wrap_up(
            state, "the limit of steps for one request is reached.", _STEP_LIMIT_FALLBACK
        )

    async def _complete(self, state: dict[str, Any], **kwargs: Any) -> Any:
        """One model call over the context fitted to the budget, with the task list last."""
        budget = self.config.context_token_budget
        messages, plan = self._fit(state, budget)
        publish_thinking()
        started = time.monotonic()
        try:
            response = await self.azure.complete(messages=self._with_task_list(state, messages), **kwargs)
        except ContextTooLongException:
            # The estimate was too generous for this model: retry once with half.
            smaller = max(1, estimate_tokens(messages) // 2)
            logger.warning("loop.context_too_long", extra={"retry_budget": smaller})
            messages, plan = self._fit(state, smaller, target=smaller)
            response = await self.azure.complete(messages=self._with_task_list(state, messages), **kwargs)
        state["planning_ms"] = [*state["planning_ms"], round((time.monotonic() - started) * 1000)]
        state["context_plan"] = plan
        state["context_usage"] = {
            "used_tokens": estimate_tokens(messages),
            "budget_tokens": budget,
            **describe_plan(plan),
        }
        return response.choices[0].message

    def _fit(self, state: dict[str, Any], budget: int, target: int | None = None):
        return fit_context(
            list(state["messages"]),
            token_budget=budget,
            max_messages=self.config.max_context_messages,
            plan=state["context_plan"],
            target=target,
        )

    async def _wrap_up(self, state: dict[str, Any], reason: str, fallback: str) -> str:
        """The last call of a turn that hit a limit: no tools, answer with what is known."""
        state["execution_status"] = "stopped"
        notes = [build_wrap_up_note(reason)]
        if unfinished(state["tasks"]):
            notes.append(build_unfinished_tasks_note(unfinished(state["tasks"])))
        state["messages"] = [*state["messages"], *({"role": "system", "content": n} for n in notes)]
        try:
            message = await self._complete(state)
        except Exception:  # noqa: BLE001 - a stopped turn must still get a reply
            logger.exception("loop.wrap_up_failed")
            return fallback
        return (message.content or "").strip() or fallback

    # -- one tool call -----------------------------------------------------

    async def _run_call(self, state: dict[str, Any], call: dict[str, Any]) -> dict[str, Any]:
        """Run one call and return its ``tool`` message. Every call gets one."""
        name = call["function"]["name"]
        started = time.monotonic()

        def answer(text: str, outcome: str, error_kind: str | None = None, **extra: Any) -> dict[str, Any]:
            state["tool_results"] = {
                **state["tool_results"],
                call["id"]: {
                    "outcome": outcome,
                    "error_kind": error_kind,
                    "duration_ms": round((time.monotonic() - started) * 1000),
                    "ui_status": ui_status,
                    **extra,
                },
            }
            return {"role": "tool", "tool_call_id": call["id"], "content": text}

        ui_status: str | None = None
        try:
            arguments = json.loads(call["function"].get("arguments") or "{}")
        except json.JSONDecodeError as exc:
            return answer(f"NOT RUN -- the arguments are not valid JSON ({exc}). Re-emit the call.", "blocked", "validation")
        if not isinstance(arguments, dict):
            return answer("NOT RUN -- the arguments must be a JSON object. Re-emit the call.", "blocked", "validation")
        arguments, ui_status = split_ui_status(arguments)
        signature = failure_signature(name, arguments)
        if signature in state["failed_signatures"]:
            return answer(_REPEAT_REFUSAL, "blocked", "repeat")
        publish_status(ui_status or fallback_ui_status(name), tool=name, call_id=call["id"])

        local = self.local_tools.find(name)
        if local is not None:
            result = await local.run(LocalCall(name=name, arguments=arguments, call_id=call["id"], state=state))
            state.update(result.update)
            if result.is_error:
                state["failed_signatures"] = [*state["failed_signatures"], signature]
            extra = {"interaction": result.data["interaction"]} if isinstance(result.data, dict) and "interaction" in result.data else {}
            return answer(result.text, "failure" if result.is_error else "success", result.error_kind, **extra)

        try:
            result = await self.mcp.call_tool_structured(name, arguments)
        except ToolTimeoutException as exc:
            state["failed_signatures"] = [*state["failed_signatures"], signature]
            return answer(f"{exc} Try a narrower query.", "failure", "timeout")
        except MCPConnectionException as exc:
            # Answer this call before ending the turn, so the history stays valid.
            state["messages"] = [*state["messages"], answer(f"NOT RUN -- {exc}", "failure", "connection")]
            self._answer_rest(state)
            raise _Unreachable(
                f"The connection to the data service broke while answering, so the "
                f"answer is incomplete.\n\n{exc}\n\nSend the request again."
            ) from exc
        if result.is_error:
            state["failed_signatures"] = [*state["failed_signatures"], signature]
            return answer(result.text, "failure", _error_code(result.structured, result.text))
        return answer(result.text, "success")

    @staticmethod
    def _answer_rest(state: dict[str, Any]) -> None:
        """Answer the last step's calls that have no result yet: Azure rejects
        an assistant message whose calls are not all answered."""
        answered = {m.get("tool_call_id") for m in state["messages"] if m.get("role") == "tool"}
        last = next(m for m in reversed(state["messages"]) if m.get("tool_calls"))
        state["messages"] = [
            *state["messages"],
            *(
                {"role": "tool", "tool_call_id": c["id"], "content": "NOT RUN -- the turn ended."}
                for c in last["tool_calls"]
                if c["id"] not in answered
            ),
        ]

    # -- steers and the task list -----------------------------------------

    def _take_steers(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """Add the steers waiting for this turn as a user message; the ones taken.

        Called only at the start of a step, when every call of the last step is
        answered: Azure accepts nothing between tool calls and their results.
        """
        taken = self.steers.take()
        if not taken:
            return []
        step = turn_step_count(state["messages"])
        state["tasks"] = add_steers(state["tasks"], state["user_message"], taken)
        state["steers"] = [
            *state["steers"],
            *({"id": s["id"], "text": s["text"], "after_step": step, "reply": None} for s in taken),
        ]
        state["tasks_checked"] = False
        publish_tasks(state["tasks"])
        state["messages"] = [
            *state["messages"],
            {"role": "user", "content": build_steer_message([s["text"] for s in taken]), STEER_KEY: True},
        ]
        logger.info("steers.applied", extra={"steers": len(taken), "after_step": step})
        return taken

    @staticmethod
    def _steer_reply(state: dict[str, Any], steered: list[dict[str, Any]], content: str | None) -> str | None:
        """The model's text of a step that read a steer: its reply, shown to the user."""
        text = (content or "").strip()[:MAX_REPLY_CHARS]
        if not steered or not text:
            return None
        records = [dict(record) for record in state["steers"]]
        records[-1]["reply"] = text
        state["steers"] = records
        publish_message(text, reply_to=records[-1]["id"])
        return text

    def _keep_going(self, state: dict[str, Any], steered: list[dict[str, Any]], content: str | None) -> bool:
        """Whether an answer is not the end of the turn.

        A steer that arrived while the model wrote it makes it a draft: dropped,
        and the next step reads the steer. Unfinished task-list items send the
        model back once (again after each new steer).
        """
        if self.steers.pending():
            logger.info("loop.answer_superseded_by_steer")
            return True
        left = unfinished(state["tasks"])
        if not left or state["tasks_checked"]:
            return False
        state["tasks_checked"] = True
        reply = self._steer_reply(state, steered, content)
        if reply:
            state["messages"] = [*state["messages"], {"role": "assistant", "content": reply}]
        state["messages"] = [*state["messages"], {"role": "system", "content": build_open_tasks_note(left)}]
        logger.info("loop.tasks_unfinished", extra={"tasks": len(left)})
        return True

    @staticmethod
    def _with_task_list(state: dict[str, Any], messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The messages to send, with the task list last when the turn has one."""
        if not state["tasks"]:
            return messages
        return [*messages, {"role": "system", "content": build_task_list_message(state["tasks"])}]

    # -- memory ------------------------------------------------------------

    async def _save(self, state: dict[str, Any]) -> None:
        """Store the conversation, normalised so the next request is valid."""
        messages = truncate_messages(
            strip_for_persistence(list(state["messages"])), self.config.max_history_messages
        )
        await self.memory.save(
            session_id=state["session_id"], messages=messages, lessons=[], successful_tools=[]
        )


def _error_code(structured: dict[str, Any] | None, text: str) -> str:
    """The server's error ``code`` for the trace, from its JSON error when it sent one."""
    payload = structured
    if payload is None:
        try:
            payload = json.loads(text)
        except ValueError:
            payload = None
    code = payload.get("code") if isinstance(payload, dict) else None
    return code if isinstance(code, str) and code else "tool_error"
