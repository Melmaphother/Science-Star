#!/usr/bin/env python

"""
Search backends - unified input/output format.

Each function: (query, filter_year?, max_results?) -> List[SearchResult] | str (error)
"""

import os
from typing import List, Optional, TypedDict


class SearchResult(TypedDict):
    """Unified search result format."""

    title: str
    link: str
    snippet: str
    date: str
    source: str


def _norm_result(
    title: str,
    link: str,
    snippet: str,
    date: str = "",
    source: str = "",
) -> SearchResult:
    return {"title": title, "link": link, "snippet": snippet, "date": date, "source": source}


def serpapi_search(
    query: str,
    filter_year: Optional[int] = None,
    max_results: int = 10,
    engine: str = "google",
) -> List[SearchResult] | str:
    """Search via SerpAPI (Google, Bing, Yahoo, Baidu) using serpapi.Client."""
    api_key = os.getenv("SERP_API_KEY")
    if not api_key:
        return "SERP_API_KEY not set. SerpAPI search unavailable."

    try:
        import serpapi

        client = serpapi.Client(api_key=api_key)

        if engine == "google":
            params = {"engine": "google", "q": query, "num": max_results}
        elif engine == "bing":
            params = {"engine": "bing", "q": query, "count": max_results}
        elif engine == "baidu":
            params = {"engine": "baidu", "q": query, "rn": max_results}
        elif engine == "yahoo":
            params = {"engine": "yahoo", "p": query}
        else:
            return f"Unsupported engine: {engine}"

        if filter_year is not None:
            params["tbs"] = (
                f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"
            )

        results = client.search(params)
    except Exception as e:
        return f"SerpAPI error: {e}"

    if "organic_results" not in results or not results["organic_results"]:
        suffix = f" (filter year={filter_year})" if filter_year else ""
        return f"No results for '{query}'{suffix}. Try a broader query."

    out: List[SearchResult] = []
    for p in results["organic_results"][:max_results]:
        out.append(
            _norm_result(
                title=p.get("title", ""),
                link=p.get("link", ""),
                snippet=p.get("snippet", ""),
                date=p.get("date", ""),
                source=p.get("source", ""),
            )
        )
    return out


def tavily_search(
    query: str,
    filter_year: Optional[int] = None,
    max_results: int = 10,
) -> List[SearchResult] | str:
    """Search via Tavily API."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "TAVILY_API_KEY not set. Tavily search unavailable."

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        params = {"query": query, "max_results": max_results}
        if filter_year is not None:
            from datetime import datetime

            now = datetime.now().year
            params["days"] = max(1, (now - filter_year) * 365)
        response = client.search(**params)
    except ImportError:
        return "tavily package not installed. pip install tavily-python"
    except Exception as e:
        return f"Tavily error: {e}"

    if "results" not in response or not response["results"]:
        return f"No results for '{query}'. Try a broader query."

    out: List[SearchResult] = []
    for r in response["results"][:max_results]:
        out.append(
            _norm_result(
                title=r.get("title", ""),
                link=r.get("url", ""),
                snippet=r.get("content", ""),
                date=str(r.get("published_date", "")),
                source="",
            )
        )
    return out


def duckduckgo_search(
    query: str,
    filter_year: Optional[int] = None,
    max_results: int = 10,
) -> List[SearchResult] | str:
    """Search via DuckDuckGo (no API key)."""
    try:
        from ddgs import DDGS

        ddgs = DDGS()
        results = list(ddgs.text(keywords=query, max_results=max_results))
    except ImportError:
        return "duckduckgo_search not installed. pip install duckduckgo-search"
    except Exception as e:
        return f"DuckDuckGo error: {e}"

    if not results:
        return f"No results for '{query}'. Try a broader query."

    out: List[SearchResult] = []
    for r in results:
        out.append(
            _norm_result(
                title=r.get("title", ""),
                link=r.get("href", ""),
                snippet=r.get("body", ""),
            )
        )
    return out


def wiki_search(query: str, max_results: int = 5) -> List[SearchResult] | str:
    """Search Wikipedia."""
    import requests

    base = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|info",
        "exintro": True,
        "explaintext": True,
        "titles": query,
        "redirects": 1,
        "inprop": "url",
    }
    try:
        r = requests.get(base, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return f"Wikipedia error: {e}"

    pages = data.get("query", {}).get("pages", {})
    out: List[SearchResult] = []
    for pid, info in list(pages.items())[:max_results]:
        if int(pid) < 0:
            continue
        out.append(
            _norm_result(
                title=info.get("title", ""),
                link=info.get("fullurl", ""),
                snippet=info.get("extract", ""),
            )
        )
    if not out:
        return f"No Wikipedia results for '{query}'."
    return out


# Registry for aggregated search
SEARCH_BACKENDS = {
    "google": lambda q, fy, m: serpapi_search(q, fy, m, "google"),
    "bing": lambda q, fy, m: serpapi_search(q, fy, m, "bing"),
    "baidu": lambda q, fy, m: serpapi_search(q, fy, m, "baidu"),
    "yahoo": lambda q, fy, m: serpapi_search(q, fy, m, "yahoo"),
    "tavily": tavily_search,
    "duckduckgo": duckduckgo_search,
    "wiki": lambda q, fy, m: wiki_search(q, max_results=m),
}


def search(
    query: str,
    backend: str = "google",
    filter_year: Optional[int] = None,
    max_results: int = 10,
) -> List[SearchResult] | str:
    """Aggregated search - dispatches to backend by name."""
    if backend not in SEARCH_BACKENDS:
        return f"Unknown backend: {backend}. Choose from: {list(SEARCH_BACKENDS)}"
    fn = SEARCH_BACKENDS[backend]
    if backend == "wiki":
        return fn(query, max_results=max_results)
    return fn(query, filter_year, max_results)
