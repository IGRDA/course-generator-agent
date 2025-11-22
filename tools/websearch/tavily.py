"""Web search tool using LangChain's Tavily integration."""

import os
from langchain_tavily import TavilySearch

def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using Tavily via LangChain.
    
    Args:
        query: Search query string
        max_results: Maximum number of results (default: 5)
        
    Returns:
        String containing search results
        
    Example:
        results = web_search("Python async programming", max_results=3)
        print(results)
    
    Note:
        Requires TAVILY_API_KEY environment variable to be set.
        Get your API key at: https://app.tavily.com/sign-in
    """
    try:
        # Check if API key is set
        if not os.environ.get("TAVILY_API_KEY"):
            return "Error: TAVILY_API_KEY environment variable is not set. Get your API key at https://app.tavily.com/sign-in"
        
        # Initialize Tavily Search Tool
        search_tool = TavilySearch(
            max_results=max_results,
            topic="general",
        )
        
        # Invoke the search
        result = search_tool.invoke({"query": query})
        return result
    except Exception as e:
        return f"Search failed: {str(e)}"

if __name__ == "__main__":
    result = web_search("Python async programming")
    print(result)


