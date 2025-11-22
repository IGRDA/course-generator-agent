"""Wikipedia search using LangChain."""

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


def web_search(query: str, max_results: int = 5) -> str:
    """
    Search Wikipedia using LangChain.
    
    Args:
        query: Search query string
        max_results: Maximum number of results (default: 5)
        
    Returns:
        String containing Wikipedia search results
    """
    try:
        # Initialize Wikipedia API wrapper with max_results
        wikipedia = WikipediaAPIWrapper(
            top_k_results=max_results,
            doc_content_chars_max=4000
        )
        search_tool = WikipediaQueryRun(api_wrapper=wikipedia)
        result = search_tool.run(query)
        return result
    except Exception as e:
        return f"Wikipedia search failed: {str(e)}"

