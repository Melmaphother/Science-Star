#!/usr/bin/env python

"""Multi-agent runner for GAIA benchmark.

Uses CodeAgent as manager with ToolCallingAgent (search_agent) as managed agent.
"""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import login
from smolagents import CodeAgent, OpenAIServerModel, ToolCallingAgent

from experiment_runner import BaseAgentRunner
from templates import (
    CODE_AGENT_WEB_SEARCH_INSTRUCTIONS,
    SEARCH_AGENT_CLARIFICATION_REQUEST,
    SEARCH_AGENT_DESCRIPTION,
    SEARCH_AGENT_PROMPT_EXTRAS,
)
from tools.browser.agent_browser import BrowserUseTool
from tools.code import AUTHORIZED_IMPORTS
from tools.crawl.crawl_tools import CrawlUrlTool
from tools.inspector.audio_inspector_tool import AudioInspectorTool
from tools.inspector.document_inspector_tool import DocumentInspectorTool
from tools.inspector.visual_inspector_tool import VisualInspectorTool
from tools.retriever.retriever_tool import RetrieverTool
from tools.search.search_tool import SearchTool, WaybackSearchTool

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=_REPO_ROOT / ".env", override=True)
login(os.getenv("HF_TOKEN"))


class MultiAgentRunner(BaseAgentRunner):
    """Runner using CodeAgent manager + ToolCallingAgent (search) as managed agent."""

    def create_agent(self, model: OpenAIServerModel) -> CodeAgent:
        """Create CodeAgent with search_agent as managed agent."""
        web_tools = [
            CrawlUrlTool(backend="jina"),
            WaybackSearchTool(),
            RetrieverTool(),
            DocumentInspectorTool(model),
            BrowserUseTool(model=self.config.models.name),
            SearchTool(backend="tavily"),
        ]

        agents_config = getattr(self.config, "agents", self.config)
        max_steps = getattr(agents_config, "max_steps", 12)
        planning_interval = getattr(agents_config, "planning_interval", 1)
        verbosity_level = 2 if getattr(agents_config, "verbose", False) else 1

        search_agent = ToolCallingAgent(
            model=model,
            tools=web_tools,
            max_steps=max_steps,
            verbosity_level=verbosity_level,
            planning_interval=planning_interval,
            name="search_agent",
            description=SEARCH_AGENT_DESCRIPTION,
            provide_run_summary=True,
        )
        search_agent.prompt_templates["managed_agent"]["task"] += (
            SEARCH_AGENT_PROMPT_EXTRAS + SEARCH_AGENT_CLARIFICATION_REQUEST
        )

        manager = CodeAgent(
            model=model,
            tools=[
                VisualInspectorTool(model),
                AudioInspectorTool(model),
                DocumentInspectorTool(model),
            ],
            max_steps=max_steps,
            verbosity_level=verbosity_level,
            additional_authorized_imports=AUTHORIZED_IMPORTS,
            planning_interval=planning_interval,
            managed_agents=[search_agent],
            instructions=CODE_AGENT_WEB_SEARCH_INSTRUCTIONS,
        )
        return manager

    def _extra_result_fields(self, agent: Any, example: dict) -> dict[str, Any]:
        """Add search_agent_actions for multi-agent runs."""
        if not getattr(agent, "managed_agents", None):
            return {}
        search = agent.managed_agents.get("search_agent")
        if search is None:
            return {}
        return {
            "search_agent_actions": getattr(search, "task_records", []),
        }


def main() -> None:
    runner = MultiAgentRunner(_REPO_ROOT)
    runner.run()


if __name__ == "__main__":
    main()
