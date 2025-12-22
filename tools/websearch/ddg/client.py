"""Web search using DuckDuckGo."""

import logging
from langchain_community.tools import DuckDuckGoSearchRun

# Configure logger for this module
logger = logging.getLogger(__name__)


def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo.
    
    Args:
        query: Search query string
        max_results: Maximum number of results (default: 5)
        
    Returns:
        String containing search results
    """
    try:
        search_tool = DuckDuckGoSearchRun(max_results=max_results)
        return search_tool.run(query)
    except Exception as e:
        logger.error(
            "DuckDuckGo search failed for query '%s': %s",
            query,
            str(e)
        )
        return f"Search failed: {str(e)}"
