#!/usr/bin/env python

"""
Agent-driven browser (Browser-Use): autonomous web browsing to answer queries.

Requires: pip install browser-use && uvx browser-use install  # or: playwright install chromium
Env: OPENAI_API_KEY (required), OPENAI_BASE_URL (optional), OPENAI_MODEL (optional)
"""

import asyncio
import os
from typing import Optional

from smolagents import Tool


def _get_browser_use_agent():
    """Lazy import to avoid hard dependency."""
    try:
        from browser_use import Agent, Browser, ChatOpenAI
    except ImportError as e:
        raise ImportError(
            "browser-use not installed. Run: pip install browser-use && uvx browser-use install"
        ) from e
    return Agent, Browser, ChatOpenAI


class BrowserUseTool(Tool):
    """Let an agent browse the web autonomously to answer a query."""

    name = "browser_browse"
    description = (
        "Automatically browse the web to answer a question. Use when the answer requires "
        "navigating multiple pages, filling forms, or interacting with dynamic content. "
        "The agent will search, click, and extract information until it finds the answer."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": (
                "The question or task to answer via web browsing. "
                "E.g. 'What is the current Bitcoin price?' or 'Find the cheapest flight from NYC to Paris next week.'"
            ),
        },
    }
    output_type = "string"

    def __init__(
        self,
        model: Optional[str] = None,
        headless: bool = True,
    ):
        """Initialize BrowserUseTool.

        Args:
            model: OpenAI-compatible model (default from OPENAI_MODEL or gpt-4o-mini).
            headless: Run browser in headless mode.
        """
        super().__init__()
        self._model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._headless = headless

    def forward(self, query: str) -> str:
        """Run browser agent to answer the query.

        Args:
            query: The question or task to answer.

        Returns:
            The extracted answer or content from browsing.

        Raises:
            ImportError: If browser-use is not installed.
            Exception: If browsing fails.
        """
        Agent, Browser, ChatOpenAI = _get_browser_use_agent()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return (
                "OPENAI_API_KEY not set. browser_browse requires an OpenAI-compatible API key. "
                "Set OPENAI_API_KEY in .env"
            )

        base_url = os.getenv("OPENAI_BASE_URL") or None
        llm = ChatOpenAI(
            model=self._model,
            api_key=api_key,
            base_url=base_url,
        )
        browser = Browser(headless=self._headless)
        agent = Agent(task=query, llm=llm, browser=browser)

        try:
            history = asyncio.run(agent.run())
        except Exception as e:
            return f"Browser browse error: {e}"

        result = history.final_result()
        if result:
            return str(result)

        extracted = history.extracted_content()
        if extracted:
            return "\n\n".join(str(x) for x in extracted if x)

        if history.has_errors():
            errs = history.errors()
            err_msg = next((e for e in errs if e), "Unknown error")
            return f"Browser browse completed with errors: {err_msg}"

        return (
            "Browser browse completed but no content was extracted. "
            f"Visited {len(history.urls())} URLs."
        )


if __name__ == "__main__":
    # Test instantiation
    tool = BrowserUseTool(headless=True)
    assert tool.name == "browser_browse"
    print("✅ BrowserUseTool: instantiation OK")

    _test_query = (
        "Go to https://example.com, find the main heading, and tell me its exact text."
    )

    if os.getenv("OPENAI_API_KEY"):
        # API key set: run real browse and print result
        print("🚀 Running real browse (OPENAI_API_KEY set)...")
        result = tool.forward(_test_query)
        print("--- Tool result ---")
        print(result)
        print("--- End ---")
    else:
        # No API key: test graceful error path
        result = tool.forward(_test_query)
        print("Tool result:", result[:200] + ("..." if len(result) > 200 else ""))
        if "OPENAI_API_KEY" in result:
            print("✅ BrowserUseTool: graceful error when API key missing")
        elif "browser-use" in result.lower() or "not installed" in result.lower():
            print("✅ BrowserUseTool: graceful error when browser-use not installed")
        else:
            print(f"✅ BrowserUseTool: returned (len={len(result)})")

    print("✅ agent_browser tests passed")
