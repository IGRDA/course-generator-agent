"""Web search using Tavily."""

import os
from langchain_tavily import TavilySearch


def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using Tavily.
    
    Args:
        query: Search query string
        max_results: Maximum number of results (default: 5)
        
    Returns:
        String containing search results
    
    Note:
        Requires TAVILY_API_KEY environment variable.
    """
    try:
        if not os.environ.get("TAVILY_API_KEY"):
            return "Error: TAVILY_API_KEY environment variable is not set."
        
        search_tool = TavilySearch(max_results=max_results, topic="general")
        result = search_tool.invoke({"query": query})
        return result
    except Exception as e:
        return f"Search failed: {str(e)}"

