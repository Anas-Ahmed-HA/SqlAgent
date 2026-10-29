# AGENTS.md

Instructions for AI coding agents (and people) working in this repository. See [README.md](README.md) for
what the service does.

## Commands

```bash
uv sync --locked
```

```bash
uv run pytest -q
```

```bash
uv run python app/main.py
```

Runs on `http://127.0.0.1:8321` (`AGENT_HOST`/`AGENT_PORT`) and needs an MCP server at `MCP_URL`. For a
test run, start the agent on a spare port, and the SQL MCP server too if you need one. Stop both
afterwards.

## Layout

- `app/agent/loop.py`: the agent, one tool loop. New behaviour goes here as plain code, not as a framework.
- `app/agent/clients/`: Azure, the AI gateway, the employee check, and the MCP client
  (`mcp_session.py`, streamable HTTP).
- `app/agent/tools/`: local tools the loop answers itself (`ask_user`, `update_tasks`); one module per tool,
  registered in `AgentLoop.local_tools`.
- `app/agent/host/`, `app/agent/api/`: the FastAPI service, turn queueing and stop, and the per-turn event
  socket for questions and steers.
- `app/agent/memory/`: session history, and fitting the context to its budget.
- `app/shared/`: settings, the HTTP contract and identity. It imports nothing first-party.
- `app/skills_hub/`: the turn side of skills. It imports only pydantic.

## Rules

- **The model never decides access.** Which databases and objects exist is decided by the MCP server.
  Credentials and headers come from configuration (`MCP_HEADERS`), never from tool arguments or prompts.
- **Every tool call gets a tool message**, on every path out of the loop, including errors and an
  unreachable server. Azure rejects a history with an unanswered call. Tests check
  `validate_message_sequence`.
- **Never replay a call** after the connection broke mid-call; reconnect only before the next one.
- **Skill text is untrusted.** It passes `skill_text_is_safe` and the equipped limits in `agent.host.turns`
  before it reaches the model. Keep the safety pattern identical to `SkillSafety.IsSafe` in API_AIChatting.
- **No secrets in Git.** `.env` is gitignored; `.env.template` has placeholders only.

## Style

- Comments explain why, not what. Keep the existing density.
- Python 3.11, type hints, `from __future__ import annotations`.
- Tests mirror `app/` under `tests/`; the loop is driven with `tests/agent/fakes.py`, with no Azure or MCP
  server needed. `tests/agent/clients/test_mcp_session.py` runs the HTTP client against an in-process MCP
  server.
