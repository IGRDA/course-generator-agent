"""Web search tool using LangChain's built-in DuckDuckGo integration."""

from langchain_community.tools import DuckDuckGoSearchRun
from langsmith import traceable


@traceable(name="web_search")
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo via LangChain.
    
    Args:
        query: Search query string
        max_results: Maximum number of results (default: 5)
        
    Returns:
        String containing search results
        
    Example:
        results = web_search("Python async programming", max_results=3)
        print(results)
    """
    try:
        search_tool = DuckDuckGoSearchRun(max_results=max_results)
        return search_tool.run(query)
    except Exception as e:
        return f"Search failed: {str(e)}"

# Default search tool instance
search_tool = DuckDuckGoSearchRun(max_results=5)
