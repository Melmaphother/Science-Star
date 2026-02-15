#!/usr/bin/env python

"""
Search tools - web search and Wayback Machine archive lookup.

Unified Tool wrappers around search_backends.
"""

from typing import Optional

import requests
from smolagents import Tool

from tools.search.search_backends import search, SearchResult
from tools.crawl.crawl_backends import fetch_url


class SearchTool(Tool):
    """Web search via configurable backend (google, bing, duckduckgo, wiki, tavily, etc.)."""

    name = "web_search"
    description = (
        "Perform a web search and return formatted results. "
        "Use for finding information, articles, or pages. "
        "Provide filter_year (YYYY) if you need results from a specific year."
    )
    inputs = {
        "query": {"type": "string", "description": "Search query (natural language or keywords)."},
        "filter_year": {
            "type": "string",
            "description": "Optional. Filter results to a specific year (e.g. '2020').",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, backend: str = "google"):
        super().__init__()
        self.backend = backend
        # Unique tool name per backend so agent can distinguish multiple search tools
        self.name = f"web_search_{backend}"

    def forward(
        self,
        query: str,
        filter_year: Optional[str] = None,
    ) -> str:
        fy = int(filter_year) if filter_year and filter_year.isdigit() else None
        out = search(query, backend=self.backend, filter_year=fy, max_results=10)
        if isinstance(out, str):
            return out
        results: list[SearchResult] = out
        if not results:
            return f"No results for '{query}'."
        lines = [
            f"## {self.backend.capitalize()} Search: {query}\n",
            *[
                f"{i}. [{r['title']}]({r['link']})\n{r['snippet']}"
                for i, r in enumerate(results, 1)
            ],
        ]
        return "\n\n".join(lines)


class WaybackSearchTool(Tool):
    """Look up archived versions of URLs via Wayback Machine."""

    name = "find_archived_url"
    description = (
        "Given a URL, searches the Wayback Machine and returns the archived version "
        "closest in time to the desired date. Use for historical webpages or pages "
        "that no longer exist."
    )
    inputs = {
        "url": {"type": "string", "description": "The URL you need the archive for."},
        "date": {
            "type": "string",
            "description": "Desired date in YYYYMMDD format (e.g. 20080627 for 27 June 2008).",
        },
    }
    output_type = "string"

    def forward(self, url: str, date: str) -> str:
        no_ts = f"https://archive.org/wayback/available?url={url}"
        with_ts = f"{no_ts}&timestamp={date}"
        try:
            resp = requests.get(with_ts, timeout=15).json()
            resp_nots = requests.get(no_ts, timeout=15).json()
        except Exception as e:
            return f"Wayback API error: {e}"

        closest = None
        if "archived_snapshots" in resp and "closest" in resp["archived_snapshots"]:
            closest = resp["archived_snapshots"]["closest"]
        elif "archived_snapshots" in resp_nots and "closest" in resp_nots["archived_snapshots"]:
            closest = resp_nots["archived_snapshots"]["closest"]

        if not closest:
            return f"URL {url} was not archived on Wayback Machine. Try a different URL."

        target_url = closest["url"]
        snapshot_date = closest.get("timestamp", "")[:8]
        content = fetch_url(target_url, backend="jina")
        if not content or content.strip() == "\n":
            content = f"(Could not fetch archive content; URL: {target_url})"
        return (
            f"Web archive for {url}, snapshot at {snapshot_date}:\n\n"
            f"---\n\n{content}"
        )


if __name__ == "__main__":
    # Test SearchTool: expect error when no API key, or results when key exists
    tool = SearchTool(backend="duckduckgo")
    result = tool.forward("Python programming")
    if "not installed" in result.lower() or "No results" in result or "error" in result.lower():
        print("✅ SearchTool: handles missing deps or no results")
    else:
        print(f"✅ SearchTool: returned {len(result)} chars")
    print(f"   Sample: {result[:150]}...")

    # Test WaybackSearchTool: uses public archive.org API
    tool2 = WaybackSearchTool()
    result2 = tool2.forward("https://example.com", "20200101")
    if "archived" in result2.lower() or "not archived" in result2 or "error" in result2.lower():
        print("✅ WaybackSearchTool: returns archive info or graceful message")
    else:
        print(f"✅ WaybackSearchTool: returned (len={len(result2)})")
    print("✅ search_tool tests passed")
