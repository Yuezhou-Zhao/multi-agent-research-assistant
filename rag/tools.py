"""arXiv API + Tavily wrappers (Section 4.2, Week 2 scope).

These are the retrieval-tool functions each sub-agent's independent tool
set is built from:
  ArXivResearchAgent: arxiv_search_tool
    (+ faiss_retriever_tool, pdf_parser_tool, calculator_tool — added when
     the sub-agent itself is wired up in Week 5)
  WebResearchAgent:   tavily_search_tool, url_scraper_tool

Kept as plain async functions rather than LangChain @tool-wrapped callables:
that wrapping happens once when the sub-agents are constructed (Week 5), so
these stay directly unit-testable without an agent runtime in the loop.

arXiv's client library is synchronous under the hood (urllib); calls are
pushed through asyncio.to_thread so nothing here blocks the event loop.
Tavily's AsyncTavilyClient is natively async.
"""
import asyncio
import os
from html.parser import HTMLParser

import aiohttp
import arxiv

_ARXIV_CLIENT = arxiv.Client()


async def arxiv_search_tool(query: str, max_results: int = 5) -> list[dict]:
    """Search arXiv for papers matching `query`.

    Returns metadata + abstract only — full text is fetched on demand via
    pdf_parser_tool, not here, to keep this call cheap and cacheable.
    """
    def _search() -> list[dict]:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        return [
            {
                "id": paper.get_short_id(),
                "title": paper.title,
                "summary": paper.summary.replace("\n", " ").strip(),
                "authors": [a.name for a in paper.authors],
                "published": paper.published.isoformat() if paper.published else None,
                "url": paper.entry_id,
                "pdf_url": paper.pdf_url,
            }
            for paper in _ARXIV_CLIENT.results(search)
        ]

    return await asyncio.to_thread(_search)


def _get_tavily_client():
    from tavily import AsyncTavilyClient

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Set it in your environment or .env file "
            "to use tavily_search_tool."
        )
    return AsyncTavilyClient(api_key=api_key)


async def tavily_search_tool(query: str, max_results: int = 5) -> list[dict]:
    """Search the web via Tavily. Returns content in the same chunk shape
    the Web sub-agent will merge alongside arxiv_chunks (Section 4.2)."""
    client = _get_tavily_client()
    response = await client.search(query, max_results=max_results)
    return [
        {"content": r["content"], "source": r["url"], "title": r.get("title", "")}
        for r in response.get("results", [])
    ]


class _TextExtractor(HTMLParser):
    """Minimal HTML-to-text stripper — stdlib only, no bs4 dependency."""

    _SKIP_TAGS = {"script", "style", "noscript"}

    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self.chunks: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self.chunks.append(stripped)

    def text(self) -> str:
        return " ".join(self.chunks)


async def url_scraper_tool(url: str, timeout: float = 15.0) -> str:
    """Fetch `url` and return its visible text content."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            resp.raise_for_status()
            html = await resp.text()

    extractor = _TextExtractor()
    extractor.feed(html)
    return extractor.text()
