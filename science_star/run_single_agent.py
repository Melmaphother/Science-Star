#!/usr/bin/env python

"""Single ToolCallingAgent runner for GAIA benchmark.

Uses ToolCallingAgent directly with all tools in one flat agent.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import login
from smolagents import OpenAIServerModel, ToolCallingAgent

from experiment_runner import BaseAgentRunner
from tools.browser.agent_browser import BrowserUseTool
from tools.crawl.crawl_tools import CrawlUrlTool
from tools.inspector.audio_inspector_tool import AudioInspectorTool
from tools.inspector.document_inspector_tool import DocumentInspectorTool
from tools.inspector.visual_inspector_tool import VisualInspectorTool
from tools.retriever.retriever_tool import RetrieverTool
from tools.search.search_tool import SearchTool, WaybackSearchTool

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=_REPO_ROOT / ".env", override=True)
login(os.getenv("HF_TOKEN"))


class SingleAgentRunner(BaseAgentRunner):
    """Runner using a single ToolCallingAgent with all tools."""

    def create_agent(self, model: OpenAIServerModel) -> ToolCallingAgent:
        """Create ToolCallingAgent with web + inspector tools."""
        tools = [
            CrawlUrlTool(backend="jina"),
            WaybackSearchTool(),
            RetrieverTool(),
            DocumentInspectorTool(model),
            BrowserUseTool(model=self.config.models.name),
            SearchTool(backend="tavily"),
            VisualInspectorTool(model),
            AudioInspectorTool(model),
        ]

        agents_config = getattr(self.config, "agents", self.config)
        return ToolCallingAgent(
            model=model,
            tools=tools,
            max_steps=getattr(agents_config, "max_steps", 12),
            verbosity_level=2 if getattr(agents_config, "verbose", False) else 1,
            planning_interval=getattr(agents_config, "planning_interval", 1),
        )


def main() -> None:
    runner = SingleAgentRunner(_REPO_ROOT)
    runner.run()


if __name__ == "__main__":
    main()
