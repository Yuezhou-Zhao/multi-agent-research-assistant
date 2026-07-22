"""Client for the web-research MCP server (see mcp_servers/web_research.py).

Holds ONE long-lived stdio session for the process, created lazily on
first use. That matters here: web_agent_node is re-entered on every
Critic rollback (route_after_critic sends the draft back to the
Supervisor, which fans out again, up to max_critic_loops times), so a
session per call would spawn and tear down a subprocess several times per
query. This mirrors the lazy-singleton pattern the retriever and
guardrail already use in backend/nodes/arxiv_agent.py, for the same
reason: expensive to construct, safe to share, pointless to rebuild.

Tool names are discovered, not hardcoded. `call()` validates against what
the server actually advertised in its list_tools response, so a rename on
the server side surfaces as a clear ValueError naming the tools that do
exist, rather than as a protocol error from the far end.

Caveat worth stating: the session is entered inside whichever asyncio
task first calls it and is not auto-closed. anyio cancel scopes are
task-affine, so `aclose()` must run in that same task — tests do this
explicitly; the app relies on process exit to reap the subprocess. In the
graph this is a non-issue, since every node runs in one event loop.
"""
import asyncio
import logging
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent


class WebResearchMCPClient:
    """Lazily-connected MCP client over a stdio subprocess."""

    def __init__(self, server_module: str = "mcp_servers.web_research") -> None:
        # server_module is overridable so scripts/bench_mcp_overhead.py can
        # point the *real* client at a stubbed server and measure protocol
        # cost rather than Tavily's network latency. Nothing else overrides it.
        self._server_module = server_module
        self._session: ClientSession | None = None
        self._stack: AsyncExitStack | None = None
        self._tool_names: set[str] = set()
        self._lock = asyncio.Lock()

    async def _ensure_session(self) -> ClientSession:
        if self._session is not None:
            return self._session

        async with self._lock:
            if self._session is not None:  # won the race while waiting
                return self._session

            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", self._server_module],
                cwd=str(_REPO_ROOT),
                # The SDK's default environment is a minimal allow-list that
                # drops TAVILY_API_KEY, which the server needs at call time.
                # Pass the real environment through explicitly.
                env=dict(os.environ),
            )

            stack = AsyncExitStack()
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            listed = await session.list_tools()
            self._tool_names = {t.name for t in listed.tools}
            log.info(
                "web-research MCP server connected; discovered tools: %s",
                ", ".join(sorted(self._tool_names)) or "(none)",
            )

            self._stack = stack
            self._session = session
            return session

    async def tool_names(self) -> set[str]:
        """Tool names as advertised by the server (connects if needed)."""
        await self._ensure_session()
        return set(self._tool_names)

    async def call(self, name: str, arguments: dict) -> Any:
        """Invoke a discovered tool and unwrap its return value.

        Raises RuntimeError if the server reports the call as failed, so a
        tool-side error reaches the caller as an exception rather than as a
        result object that looks superficially like success.
        """
        session = await self._ensure_session()
        if name not in self._tool_names:
            raise ValueError(
                f"{name!r} is not exposed by the web-research MCP server. "
                f"Discovered tools: {', '.join(sorted(self._tool_names)) or '(none)'}"
            )

        result = await session.call_tool(name, arguments)
        if result.isError:
            detail = " ".join(
                block.text for block in result.content if getattr(block, "text", None)
            )
            raise RuntimeError(f"MCP tool {name!r} failed: {detail or 'no detail given'}")

        # FastMCP wraps a tool's return value under a "result" key when the
        # return type is structured (list[dict] here). Plain strings come
        # back as text content blocks with no structuredContent.
        if result.structuredContent is not None and "result" in result.structuredContent:
            return result.structuredContent["result"]
        return "".join(
            block.text for block in result.content if getattr(block, "text", None)
        )

    async def aclose(self) -> None:
        """Close the session and reap the subprocess. Call from the task that opened it."""
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None
        self._tool_names = set()


# Process-wide singleton. Import this, not the class.
web_research_mcp = WebResearchMCPClient()
