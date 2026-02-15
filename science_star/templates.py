#!/usr/bin/env python

"""Prompt templates for multi-agent GAIA benchmark evaluation."""

# ---------------------------------------------------------------------------
# Search Agent Prompts
# ---------------------------------------------------------------------------

SEARCH_AGENT_DESCRIPTION = """A team member that will search the internet to answer your question.
Ask him for all your questions that require browsing the web.
Provide him as much context as possible, in particular if you need to search on a specific timeframe!
And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
"""

SEARCH_AGENT_PROMPT_EXTRAS = """You can navigate to .txt online files.
If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
For historical webpages or pages that no longer exist, use `find_archived_url` with the original URL and desired date (YYYYMMDD) to fetch from Wayback Machine.
When you have long text from crawl/search and need to find the most relevant parts for a specific question, use `retrieve_content` with the query and content.
For complex tasks requiring multi-page navigation, form filling, or dynamic content, use `browser_browse` with the question as query."""

SEARCH_AGENT_CLARIFICATION_REQUEST = """ Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

# ---------------------------------------------------------------------------
# Code Agent (Manager) Instructions
# ---------------------------------------------------------------------------

CODE_AGENT_WEB_SEARCH_INSTRUCTIONS = """
CRITICAL: You do NOT have web_search, visit_webpage, wikipedia_search, or similar functions.
For ANY web search, browsing, or information lookup, you MUST use: search_agent(task="your detailed task", additional_args={})
Example: result = search_agent(task="Find the authors of the paper 'Pie Menus or Linear Menus, Which Is Better?' from 2015", additional_args={})
Never write web_search(...) or visit_webpage(...) - they do not exist and will fail.
"""

# ---------------------------------------------------------------------------
# Main Task Prompts
# ---------------------------------------------------------------------------

AUGMENTED_QUESTION_PREFIX = """You have one question to answer. It is paramount that you provide a correct answer.
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist). 
Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
Run verification steps if that's needed, you must make sure you find the correct answer!
Here is the task:
"""

FILE_ATTACHMENT_SINGLE_PREFIX = (
    "\n\nTo solve the task above, you will have to use this attached file:"
)

FILE_ATTACHMENT_MULTIPLE_PREFIX = (
    "\n\nTo solve the task above, you will have to use these attached files:\n"
)
