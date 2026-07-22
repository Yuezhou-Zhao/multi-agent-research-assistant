"""Tests for the web-research MCP server and its client.

Keyless: every test that exercises a tool body patches the underlying
rag.tools call, so CI needs no API keys. Most connect client and server
in-process (real protocol traffic, no subprocess); one spawns the real
subprocess to prove the module is launchable.
"""
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

import backend.nodes.web_agent as web_agent
import mcp_servers.web_research as server_module
from backend.state import new_job_state
from mcp_servers.web_research import mcp as web_research_mcp_server
from rag.mcp_client import WebResearchMCPClient


async def _fake_tavily(query, max_results=5):
    return [
        {"content": f"result for {query}", "source": "https://example.com/a", "title": "A"}
    ][:max_results]


class TestToolDiscovery:
    """The protocol surface: what a client sees before it calls anything."""

    @pytest.mark.asyncio
    async def test_server_advertises_the_web_agent_tool_set(self):
        async with create_connected_server_and_client_session(
            web_research_mcp_server._mcp_server
        ) as session:
            listed = await session.list_tools()

        assert {t.name for t in listed.tools} == {"tavily_search", "url_scraper"}

    @pytest.mark.asyncio
    async def test_tools_carry_descriptions_and_input_schemas(self):
        """Docstrings and type hints are the protocol-level contract."""
        async with create_connected_server_and_client_session(
            web_research_mcp_server._mcp_server
        ) as session:
            listed = await session.list_tools()

        tools = {t.name: t for t in listed.tools}

        assert "Search the web" in tools["tavily_search"].description
        search_schema = tools["tavily_search"].inputSchema
        assert search_schema["properties"]["query"]["type"] == "string"
        assert search_schema["properties"]["max_results"]["default"] == 5
        assert search_schema["required"] == ["query"]

        assert "visible text" in tools["url_scraper"].description
        assert tools["url_scraper"].inputSchema["required"] == ["url"]


class TestToolInvocation:
    @pytest.mark.asyncio
    async def test_tavily_search_round_trips_structured_results(self, monkeypatch):
        monkeypatch.setattr(server_module, "tavily_search_tool", _fake_tavily)

        async with create_connected_server_and_client_session(
            web_research_mcp_server._mcp_server
        ) as session:
            result = await session.call_tool(
                "tavily_search", {"query": "chain of thought", "max_results": 5}
            )

        assert result.isError is False
        payload = result.structuredContent["result"]
        assert payload[0]["source"] == "https://example.com/a"
        assert "chain of thought" in payload[0]["content"]

    @pytest.mark.asyncio
    async def test_arguments_reach_the_tool_body(self, monkeypatch):
        """max_results is not silently dropped in transit."""
        seen = {}

        async def _capture(query, max_results=5):
            seen["query"] = query
            seen["max_results"] = max_results
            return []

        monkeypatch.setattr(server_module, "tavily_search_tool", _capture)

        async with create_connected_server_and_client_session(
            web_research_mcp_server._mcp_server
        ) as session:
            await session.call_tool("tavily_search", {"query": "hyde", "max_results": 3})

        assert seen == {"query": "hyde", "max_results": 3}

    @pytest.mark.asyncio
    async def test_url_scraper_returns_plain_text(self, monkeypatch):
        async def _fake_scrape(url, timeout=15.0):
            return "the visible text"

        monkeypatch.setattr(server_module, "url_scraper_tool", _fake_scrape)

        async with create_connected_server_and_client_session(
            web_research_mcp_server._mcp_server
        ) as session:
            result = await session.call_tool("url_scraper", {"url": "https://example.com"})

        assert result.isError is False
        assert "the visible text" in result.content[0].text

    @pytest.mark.asyncio
    async def test_tool_exception_is_reported_as_protocol_error(self, monkeypatch):
        """A tool failure must not look like a successful empty result."""

        async def _boom(query, max_results=5):
            raise RuntimeError("TAVILY_API_KEY is not set")

        monkeypatch.setattr(server_module, "tavily_search_tool", _boom)

        async with create_connected_server_and_client_session(
            web_research_mcp_server._mcp_server
        ) as session:
            result = await session.call_tool("tavily_search", {"query": "q"})

        assert result.isError is True
        assert "TAVILY_API_KEY" in result.content[0].text


class TestWebAgentDegradation:
    """Pins the layering: MCP -> in-process fallback -> empty web_chunks."""

    @staticmethod
    def _state():
        return new_job_state(job_id="test-job", query="how does HyDE work?")

    @pytest.mark.asyncio
    async def test_uses_mcp_result_when_the_server_answers(self, monkeypatch):
        called = {}

        class _StubMCP:
            async def call(self, name, arguments):
                called["name"] = name
                called["arguments"] = arguments
                return [{"content": "via mcp", "source": "https://m.com", "title": "M"}]

        monkeypatch.setattr(web_agent, "web_research_mcp", _StubMCP())

        result = await web_agent.web_agent_node(self._state())

        assert called["name"] == "tavily_search"
        assert called["arguments"] == {"query": "how does HyDE work?", "max_results": 5}
        assert result["web_chunks"] == [
            {"content": "via mcp", "source": "https://m.com", "title": "M"}
        ]

    @pytest.mark.asyncio
    async def test_falls_back_in_process_when_mcp_is_unavailable(self, monkeypatch):
        class _DeadMCP:
            async def call(self, name, arguments):
                raise RuntimeError("server did not start")

        async def _direct(query, max_results=5):
            return [{"content": "via direct", "source": "https://d.com", "title": "D"}]

        monkeypatch.setattr(web_agent, "web_research_mcp", _DeadMCP())
        monkeypatch.setattr(web_agent, "tavily_search_tool", _direct)

        result = await web_agent.web_agent_node(self._state())

        assert result["web_chunks"] == [
            {"content": "via direct", "source": "https://d.com", "title": "D"}
        ]

    @pytest.mark.asyncio
    async def test_degrades_to_empty_chunks_when_both_paths_fail(self, monkeypatch):
        class _DeadMCP:
            async def call(self, name, arguments):
                raise RuntimeError("server did not start")

        async def _also_dead(query, max_results=5):
            raise RuntimeError("TAVILY_API_KEY is not set")

        monkeypatch.setattr(web_agent, "web_research_mcp", _DeadMCP())
        monkeypatch.setattr(web_agent, "tavily_search_tool", _also_dead)

        result = await web_agent.web_agent_node(self._state())

        assert result["web_chunks"] == []


class TestClientOverRealSubprocess:
    """Proves mcp_servers.web_research is launchable as a real server."""

    @pytest.mark.asyncio
    async def test_client_discovers_tools_over_stdio(self):
        client = WebResearchMCPClient()
        try:
            assert await client.tool_names() == {"tavily_search", "url_scraper"}
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_unknown_tool_is_rejected_before_dispatch(self):
        client = WebResearchMCPClient()
        try:
            with pytest.raises(ValueError, match="tavily_search"):
                await client.call("arxiv_search", {"query": "q"})
        finally:
            await client.aclose()
