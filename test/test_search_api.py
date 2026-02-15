#!/usr/bin/env python3
"""
Test Search API: SERP_API_KEY and/or TAVILY_API_KEY.

At least one must work. Uses official SDKs:
- SerpAPI: https://serpapi.com/integrations/python (serpapi.Client)
- Tavily: https://docs.tavily.com/sdk/python/quick-start (TavilyClient)
"""

import os
import sys
from pathlib import Path

# Load .env from project root
_root = Path(__file__).resolve().parent.parent
_dotenv = _root / ".env"
if _dotenv.exists():
    from dotenv import load_dotenv
    load_dotenv(_dotenv)


def test_serp_api() -> bool:
    """Test SerpAPI via serpapi.Client."""
    key = os.getenv("SERP_API_KEY")
    if not key or key.strip() == "":
        return False

    try:
        import serpapi

        client = serpapi.Client(api_key=key)
        results = client.search({"engine": "google", "q": "test", "num": 1})
        if results and "organic_results" in results:
            print("✅ SERP_API_KEY OK")
            return True
        return False
    except Exception as e:
        print(f"   SERP_API_KEY failed: {e}")
        return False


def test_tavily_api() -> bool:
    """Test Tavily Search via TavilyClient. Requires: pip install tavily-python"""
    key = os.getenv("TAVILY_API_KEY")
    if not key or key.strip() == "":
        return False

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=key)
        response = client.search("test", max_results=1)
        if response and "results" in response:
            print("✅ TAVILY_API_KEY OK")
            return True
        return False
    except ImportError:
        print("   TAVILY_API_KEY: tavily-python not installed (pip install tavily-python)")
        return False
    except Exception as e:
        print(f"   TAVILY_API_KEY failed: {e}")
        return False


def test_search_api() -> bool:
    """
    Test at least one search API works.

    Returns:
        bool: True if SERP_API_KEY or TAVILY_API_KEY works.
    """
    if not os.getenv("SERP_API_KEY") and not os.getenv("TAVILY_API_KEY"):
        print("❌ Neither SERP_API_KEY nor TAVILY_API_KEY is set")
        return False

    serp_ok = test_serp_api()
    tavily_ok = test_tavily_api()

    if serp_ok or tavily_ok:
        return True
    print("❌ Both SERP_API_KEY and TAVILY_API_KEY failed")
    return False


if __name__ == "__main__":
    success = test_search_api()
    sys.exit(0 if success else 1)
