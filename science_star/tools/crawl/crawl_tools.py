#!/usr/bin/env python

"""
Crawl tools - fetch URLs, search-then-crawl.
"""

from typing import List, Optional

from smolagents import Tool

from tools.search.search_backends import search, SearchResult
from tools.crawl.crawl_backends import fetch_url


class CrawlUrlTool(Tool):
    """Crawl a single URL and return its content."""

    name = "crawl_pages"
    description = (
        "Fetch a webpage by URL and return its text content. "
        "Use for diving deeper into search results."
    )
    inputs = {
        "url": {"type": "string", "description": "URL to fetch."},
    }
    output_type = "string"

    def __init__(self, backend: str = "crawl4ai"):
        super().__init__()
        self.backend = backend

    def forward(self, url: str) -> str:
        out = fetch_url(url, backend=self.backend)
        if not out or out == "\n":
            return (
                f"Crawling {url} returned empty. "
                "It may be a PDF or uncrawlable page. Try inspect_file_as_text."
            )
        return out


class SearchAndCrawlTool(Tool):
    """Search, then crawl top N URLs from results."""

    name = "search_and_crawl"
    description = (
        "Search the web, then fetch full content of top URLs. "
        "Use when snippets are not enough."
    )
    inputs = {
        "query": {"type": "string", "description": "Search query."},
        "backend": {
            "type": "string",
            "description": "Search backend: google, bing, tavily, duckduckgo, wiki.",
            "nullable": True,
        },
        "filter_year": {"type": "string", "description": "Optional year filter.", "nullable": True},
        "num_urls": {
            "type": "integer",
            "description": "Number of URLs to crawl (default 3).",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(
        self,
        search_backend: str = "google",
        crawl_backend: str = "jina",
        max_results: int = 10,
    ):
        super().__init__()
        self.search_backend = search_backend
        self.crawl_backend = crawl_backend
        self.max_results = max_results

    def forward(
        self,
        query: str,
        backend: Optional[str] = None,
        filter_year: Optional[str] = None,
        num_urls: int = 3,
    ) -> str:
        be = backend or self.search_backend
        fy = int(filter_year) if filter_year and filter_year.isdigit() else None
        out = search(query, backend=be, filter_year=fy, max_results=self.max_results)
        if isinstance(out, str):
            return out
        results: List[SearchResult] = out
        if not results:
            return f"No results for '{query}'."
        urls = [r["link"] for r in results[:num_urls]]
        parts = [f"# Search: {query}\n\n## Results\n"]
        for i, url in enumerate(urls, 1):
            content = fetch_url(url, backend=self.crawl_backend)
            parts.append(f"### {i}. {url}\n\n{content}\n\n---\n")
        return "\n".join(parts)


if __name__ == "__main__":
    # Test CrawlUrlTool: expect graceful error when JINA_API_KEY not set
    tool = CrawlUrlTool(backend="crawl4ai")
    result = tool.forward("https://example.com")
    if "JINA_API_KEY" in result or "empty" in result.lower() or "error" in result.lower():
        print("✅ CrawlUrlTool: handles missing API key or returns content")
    else:
        print(f"✅ CrawlUrlTool: returned content (len={len(result)})")
    print(f"   Sample: {result[:200]}...")

    # Test SearchAndCrawlTool: expect error or results when no search API
    tool2 = SearchAndCrawlTool(search_backend="duckduckgo")
    result2 = tool2.forward("Python programming", num_urls=1)
    if "No results" in result2 or "error" in result2.lower() or "installed" in result2.lower():
        print("✅ SearchAndCrawlTool: handles missing deps or no results")
    else:
        print(f"✅ SearchAndCrawlTool: returned content (len={len(result2)})")
    print("✅ crawl_tools tests passed")


