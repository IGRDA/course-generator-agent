"""Web search tool using LangChain's built-in DuckDuckGo integration."""

from langchain_community.tools import DuckDuckGoSearchRun

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

if __name__ == "__main__":
    result = web_search("Python async programming")
    print(result)

