"""MCP server settings: the required SQL server and any number of extra ones."""

from __future__ import annotations

import json

import pytest

from agent.config import MCPServerSettings

_VARS = ("MCP_EXTRA_SERVERS", "CHARTS_MCP_URL", "CHARTS_MCP_HEADERS",
         "CHARTS_MCP_CONNECT_TIMEOUT_SECONDS", "CHARTS_MCP_CALL_TIMEOUT_SECONDS")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # A developer's .env must not leak into these.
    for name in _VARS:
        monkeypatch.delenv(name, raising=False)


def extras(monkeypatch, servers=None, **env):
    if servers is not None:
        monkeypatch.setenv("MCP_EXTRA_SERVERS", servers if isinstance(servers, str) else json.dumps(servers))
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return MCPServerSettings.extras_from_env()


def test_no_extra_servers_by_default(monkeypatch):
    assert extras(monkeypatch) == []
    assert MCPServerSettings.from_env().name == "sql"


def test_every_listed_server_is_built_in_order_with_its_own_settings(monkeypatch):
    found = extras(monkeypatch, {
        "charts": {"url": "http://localhost:5200/mcp"},
        "docs": {"url": " https://docs.example/mcp ", "headers": {"X-Api-Key": "k"},
                 "connect_timeout_seconds": 3, "call_timeout_seconds": 30},
    })
    assert [(s.name, s.url) for s in found] == [("charts", "http://localhost:5200/mcp"), ("docs", "https://docs.example/mcp")]
    assert found[0].headers == {} and found[0].connect_timeout_seconds == 5.0 and found[0].call_timeout_seconds == 60.0
    assert found[1].headers == {"X-Api-Key": "k"} and found[1].connect_timeout_seconds == 3.0


def test_charts_mcp_url_is_a_shortcut_for_the_charts_entry(monkeypatch):
    found = extras(monkeypatch, CHARTS_MCP_URL="http://localhost:5200/mcp", CHARTS_MCP_HEADERS='{"X-Api-Key": "c"}')
    assert [(s.name, s.url, s.headers) for s in found] == [("charts", "http://localhost:5200/mcp", {"X-Api-Key": "c"})]


def test_a_charts_entry_in_the_list_wins_over_the_shortcut(monkeypatch):
    found = extras(monkeypatch, {"charts": {"url": "http://charts:9000/mcp"}}, CHARTS_MCP_URL="http://localhost:5200/mcp")
    assert [s.url for s in found] == ["http://charts:9000/mcp"]


def test_the_shortcut_adds_charts_after_the_listed_servers(monkeypatch):
    found = extras(monkeypatch, {"docs": {"url": "http://docs/mcp"}}, CHARTS_MCP_URL="http://localhost:5200/mcp")
    assert [s.name for s in found] == ["docs", "charts"]


@pytest.mark.parametrize(("servers", "problem"), [
    ("{not json", "not valid JSON"),
    ('["http://x/mcp"]', "JSON object of servers by name"),
    ({"sql": {"url": "http://x/mcp"}}, "required server"),
    ({"9lives": {"url": "http://x/mcp"}}, "a name is a letter"),
    ({"docs": "http://x/mcp"}, "must be an object"),
    ({"docs": {}}, "url must be"),
    ({"docs": {"url": "ftp://x"}}, "url must be"),
    ({"docs": {"url": "http://x/mcp", "headers": {"k": 1}}}, "headers must be"),
    ({"docs": {"url": "http://x/mcp", "call_timeout_seconds": 0}}, "seconds above 0"),
    ({"docs": {"url": "http://x/mcp", "timeout": 5}}, "unknown settings"),
])
def test_a_mistake_stops_startup_naming_what_to_fix(monkeypatch, servers, problem):
    with pytest.raises(ValueError, match=problem):
        extras(monkeypatch, servers)
