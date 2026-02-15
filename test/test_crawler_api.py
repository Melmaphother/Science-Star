#!/usr/bin/env python3
"""
Test Crawler API: JINA_API_KEY and/or crawl4ai.

At least one must work:
- JINA_API_KEY: Jina Reader API (https://r.jina.ai/)
- crawl4ai: Free, no API key (AsyncWebCrawler)

Refs:
- Jina: https://jina.ai/reader
- crawl4ai: https://github.com/unclecode/crawl4ai
"""

import asyncio
import os
import sys
from pathlib import Path

# Load .env from project root
_root = Path(__file__).resolve().parent.parent
_dotenv = _root / ".env"
if _dotenv.exists():
    from dotenv import load_dotenv
    load_dotenv(_dotenv)


def test_jina_api() -> bool:
    """Test Jina Reader API via official REST endpoint."""
    key = os.getenv("JINA_API_KEY")
    if not key or key.strip() == "":
        print("   JINA_API_KEY not set, skipping")
        return False

    try:
        import httpx

        url = "https://r.jina.ai/https://example.com"
        resp = httpx.get(
            url,
            headers={
                "Authorization": f"Bearer {key}",
                "X-Return-Format": "text",
            },
            timeout=15,
        )
        resp.raise_for_status()
        if resp.text and len(resp.text) > 0:
            print("✅ JINA_API_KEY OK")
            return True
        return False
    except Exception as e:
        print(f"   JINA_API_KEY failed: {e}")
        return False


async def _test_crawl4ai() -> bool:
    """Test crawl4ai AsyncWebCrawler (official API)."""
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url="https://example.com")
            if result and result.markdown and len(result.markdown) > 0:
                print("✅ crawl4ai OK")
                return True
        return False
    except ImportError:
        print("   crawl4ai: not installed (pip install crawl4ai)")
        return False
    except Exception as e:
        err_msg = str(e)
        if "Executable doesn't exist" in err_msg or "playwright" in err_msg.lower():
            print(f"   crawl4ai failed: Playwright browser not installed. Run: playwright install chromium")
        else:
            print(f"   crawl4ai failed: {e}")
        return False


def test_crawl4ai() -> bool:
    """Run crawl4ai async test."""
    return asyncio.run(_test_crawl4ai())


def test_crawler_api() -> bool:
    """
    Test at least one crawler works.

    Returns:
        bool: True if JINA_API_KEY or crawl4ai works.
    """
    jina_ok = test_jina_api()
    crawl4ai_ok = test_crawl4ai()

    if jina_ok or crawl4ai_ok:
        return True
    print("❌ Both JINA_API_KEY and crawl4ai failed")
    return False


if __name__ == "__main__":
    success = test_crawler_api()
    sys.exit(0 if success else 1)
