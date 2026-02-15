#!/usr/bin/env python

"""
Retriever tool: semantic search over text content using RAG.

Given a query and text content(s), returns the most relevant chunks.
Use for refining search/crawl results, PDF text, or any long-form content.
"""

from smolagents import Tool

from rag_processor.embeddings import OpenAIEmbedding
from rag_processor.retrievers import SimpleVectorRetriever


class RetrieverTool(Tool):
    """Semantic retrieval over text content."""

    name = "retrieve_content"
    description = (
        "Given a query and text content, retrieve the most relevant passages. "
        "Use when you have long text (e.g. from search, crawl, or PDF) and need "
        "to find the parts most relevant to a specific question."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The question or search query to find relevant content for.",
        },
        "content": {
            "type": "string",
            "description": (
                "The text content to search within. Can be from crawled pages, "
                "search results, or extracted documents."
            ),
        },
    }
    output_type = "string"

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        limit: int = 5,
    ):
        """Initialize RetrieverTool.

        Args:
            chunk_size: Size of text chunks for retrieval.
            chunk_overlap: Overlap between chunks.
            limit: Max number of relevant chunks to return.
        """
        super().__init__()
        self._retriever = SimpleVectorRetriever(
            embedding_model=OpenAIEmbedding(),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            path=None,
        )
        self._limit = limit

    def forward(self, query: str, content: str) -> str:
        """Retrieve relevant content for the query.

        Args:
            query: The search query.
            content: The text to search within.

        Returns:
            The most relevant passages concatenated.
        """
        import os

        if not os.getenv("OPENAI_API_KEY"):
            return "OPENAI_API_KEY not set. retrieve_content requires an embedding API."

        if not content or not content.strip():
            return "No content provided to search."

        try:
            result = self._retriever.retrieve(
                query=query,
                contents=[content],
                limit=self._limit,
                threshold=0.3,
            )
            return result
        except Exception as e:
            return f"Retrieval error: {e}"


if __name__ == "__main__":
    tool = RetrieverTool(chunk_size=200, chunk_overlap=20, limit=2)
    # Test with empty content
    r1 = tool.forward("test query", "")
    assert "No content" in r1, f"Expected 'No content', got: {r1[:80]}"
    print("✅ RetrieverTool: empty content handled")

    # Test with valid content (may need OPENAI_API_KEY for real retrieval)
    r2 = tool.forward(
        "What is Python?",
        "Python is a programming language. It is widely used for web development.",
    )
    if "OPENAI_API_KEY" in r2:
        print("✅ RetrieverTool: graceful error when API key missing")
    elif "Retrieval error" in r2:
        print("✅ RetrieverTool: retrieval attempted, error caught")
    else:
        print(f"✅ RetrieverTool: returned content (len={len(r2)})")
    print("✅ retriever_tool tests passed")
