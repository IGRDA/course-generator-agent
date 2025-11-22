from typing import Callable, Dict

from .ddg.client import web_search as ddg_web_search
from .tavily.client import web_search as tavily_web_search
from .wikipedia.client import web_search as wikipedia_web_search

WebSearchFunc = Callable[[str, int], str]

SEARCH_PROVIDERS: Dict[str, WebSearchFunc] = {
    "ddg": ddg_web_search,
    "tavily": tavily_web_search,
    "wikipedia": wikipedia_web_search,
}


def available_search_providers() -> list[str]:
    """Return the list of registered web search providers."""
    return sorted(SEARCH_PROVIDERS.keys())


def create_web_search(provider: str) -> WebSearchFunc:
    """
    Get web search function for the specified provider.
    
    Args:
        provider: Web search provider name (ddg | tavily | wikipedia).
        
    Returns:
        A web search function that accepts (query: str, max_results: int).
    """
    if not provider:
        raise ValueError("Provider is required. Must be one of: ddg, tavily, wikipedia")
    
    key = provider.lower()
    try:
        return SEARCH_PROVIDERS[key]
    except KeyError as exc:
        available = ", ".join(available_search_providers())
        raise ValueError(
            f"Unsupported web search provider '{provider}'. "
            f"Available providers: {available}"
        ) from exc

