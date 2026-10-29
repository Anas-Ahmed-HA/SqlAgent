# SQL Agent

A chat agent that answers questions about company data through an MCP server reached by URL, by default
the read-only SQL MCP server (`AI_SqlMCP`). It is a plain tool loop, with no graph framework. It keeps the
interactive features of the Revit agent it was taken from (`AI_RevitMCP`):

- **Steering:** the user sends notes to a running turn. The model reads them at its next step and they
  start a task list it keeps current.
- **Questions:** the model asks the user multiple-choice questions mid-turn (`ask_user`) and waits for the
  answers.
- **Skills:** the client sends the user's equipped skills with each turn. They are checked for injection,
  capped, and put in front of the model. `/skills/propose` rewrites a user's wording into a skill.

There is no UI. Clients talk to it over HTTP and a per-turn WebSocket (see [Contract](#contract)).

## Quickstart

```powershell
uv sync --locked
copy .env.template .env      # then set MCP_URL (and MCP_HEADERS if the server needs a key)
uv run python app/main.py    # listens on http://127.0.0.1:8321
uv run pytest -q
```

To try it, open `http://127.0.0.1:8321/`: a plain chat page where you paste your login token. It shows live
status, the agent's questions, notes you send while it works, the task list, and the steps behind each answer.
**Skills** opens a panel to create, edit, order and switch on skills (or have the model write one from your
own words); the ones that are on go with every message, and each answer says which were used or left out.
The token stays in that browser tab (sessionStorage); skills are kept in the browser (localStorage).

The SQL MCP server must be running at `MCP_URL`. Without it the service still starts: `/health` reports
`degraded` with the reason, and each turn tries to connect again.

## How a turn runs

`app/agent/loop.py` is the whole agent:

```
messages = rules + the MCP server's instructions + earlier turns + skills + the user message
repeat up to AGENT_MAX_STEPS times, until AGENT_MAX_REQUEST_SECONDS:
    take the steers the user sent meanwhile
    call the model with the MCP tools and the local ones (ask_user, update_tasks)
    no tool calls -> the answer (unless a new steer or open task items keep it going)
    otherwise run each call; its result (or the server's error JSON) becomes a tool message
out of steps or time -> one last call without tools writes the answer
save the conversation
```

The MCP server checks every call (unknown tools, bad arguments, bad SQL) and returns an error with a `fix`,
so the loop only adds what the server can't know:

- the step and time limits;
- refusing a call identical to one that already failed in this turn;
- keeping the context under its token budget;
- keeping the message history valid for Azure (every tool call answered, history cut only at turn
  boundaries).

If the server is unreachable, the turn ends at once with a message saying so.

## Configuration

Everything is in `.env` (template: `.env.template`):

| Setting | What it is |
|---|---|
| `MCP_URL` | The MCP server's streamable-HTTP endpoint, e.g. `http://localhost:5100/mcp` |
| `MCP_HEADERS` | JSON object of headers sent on every MCP request, e.g. `{"X-Api-Key": "..."}`. Configuration only; the model never sees or sets them |
| `AGENT_HOST`, `AGENT_PORT` | Where this service listens (default `127.0.0.1:8321`) |
| `AIAPI_BASE_URL`, `AIAPI_CHAT_MODEL` | The AI gateway the Azure model config is fetched from, with the caller's own login token |
| `AGENT_MAX_STEPS`, `AGENT_MAX_REQUEST_SECONDS`, `AGENT_CONTEXT_TOKEN_BUDGET` | Turn limits |
| `AGENT_ALLOWED_ORIGINS_REGEX` | Other web pages (besides the one served at `/`) allowed to open the events WebSocket, matched against the whole `Origin`. Unset, none |

## Contract

`app/shared/api_contract.py` lists the routes and events; `app/shared/api_schemas.py` the bodies.

| Route | Purpose |
|---|---|
| `POST /chat` | Run one turn. Body: `message`, optional `session_id`, `turn_id`, `history`, `skills`, `interactive` |
| `WS /chat/events/{turn_id}` | Live events for the turn (`thinking`, `status`, `question`, `steer`, `message`, `tasks`, `done`); the client sends `answer`, `steer` and `steer_cancel` here |
| `POST /chat/ticket` | A single-use ticket that opens the events socket of one turn. Body `turn_id`; answer `ticket`, `expires_in` |
| `POST /chat/stop` | Stop the running turn of a session |
| `POST /skills/propose` | Rewrite a user's wording into a skill (nothing is stored) |
| `GET /health`, `GET /tools` | Whether the MCP server answers now, and its tools |
| `GET /` | The test chat page |

Every route except `/health`, `/tools` and the page needs the gateway login token as
`Authorization: Bearer ...`. Browsers can't set headers on a WebSocket, so they open the events socket with
`?ticket=...` from `POST /chat/ticket` instead: a ticket opens one socket for one turn of its caller, within
30 seconds, and is spent by its first use, so the long-lived token never travels on the socket. Each reconnect
needs a new one. CORS doesn't cover WebSockets, so the socket also refuses a web page whose `Origin` is neither
this service's own nor matched by `AGENT_ALLOWED_ORIGINS_REGEX`; clients that send no `Origin` (scripts) only
need the ticket or the header. Tickets live in process memory, like the turns, so run one instance.

## What was left out of `AI_RevitMCP`

Revit and Excel (MCP servers, add-ins, C# repair and checks, edit confirmations), the LangGraph graph, the
desktop UI, uploaded documents and images, datasets, long-term memory (LangMem/Postgres), Langfuse tracing,
and packaging. `skills_hub` keeps only the turn side (safety, limits, prompt, rewrite); the cloud skills API
client stays with the client app.
