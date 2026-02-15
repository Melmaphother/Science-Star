#!/usr/bin/env python

"""
Crawl backends - fetch URL content.

Unified interface: (url) -> str (content or error message)
"""

import asyncio
import os
from typing import Literal


def jina_read(url: str) -> str:
    """Fetch URL content via Jina Reader API (r.jina.ai)."""
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        return "JINA_API_KEY not set. Jina read unavailable."

    import requests

    jina_url = f"https://r.jina.ai/{url}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-Engine": "browser",
        "X-Return-Format": "text",
        "X-Timeout": "10",
        "X-Token-Budget": "80000",
    }
    try:
        r = requests.get(jina_url, headers=headers)
        r.raise_for_status()
        return r.text
    except Exception as e:
        return f"Jina read error for {url}: {e}"


def crawl4ai_fetch(url: str) -> str:
    """Fetch URL content via crawl4ai (browser-based)."""
    try:
        from crawl4ai import AsyncWebCrawler
    except ImportError:
        return "crawl4ai not installed. pip install crawl4ai"

    async def _run():
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(url=url)
            return result.markdown or result.raw_markdown or ""

    try:
        return asyncio.run(_run())
    except Exception as e:
        return f"Crawl4ai error for {url}: {e}"


def fetch_url(
    url: str,
    backend: Literal["jina", "crawl4ai"] = "jina",
) -> str:
    """Fetch URL content. Dispatch by backend."""
    if backend == "jina":
        return jina_read(url)
    if backend == "crawl4ai":
        return crawl4ai_fetch(url)
    return f"Unknown backend: {backend}. Use jina or crawl4ai."
